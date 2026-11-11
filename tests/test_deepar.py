"""Unit and integration tests for DeepAR forecasting model."""
from functools import partial
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.backtesting import (
    PREDICTION_COLUMNS,
    attach_actuals,
    construct_inputs,
    run_backtest,
    validate_predictions,
)
from src.evaluation.splits import development_splits
from src.models.deepar_config import DeepARConfig
from src.models.deepar_dataset import (
    StaticCategoryEncoder,
    extract_dynamic_covariates,
    prepare_deepar_data,
    validate_deepar_inputs,
)
from src.models.deepar_forecaster import DeepARForecaster, make_deepar_factory
from src.models.deepar_model import DeepARNetwork, NegativeBinomialHead, get_device
from src.models.deepar_trainer import DeepARTrainer


def test_deepar_config_validation():
    """Verify config validation and dictionary serialization."""
    cfg = DeepARConfig(context_length=56, prediction_length=28)
    cfg.validate()
    d = cfg.to_dict()
    assert d["context_length"] == 56
    reconstructed = DeepARConfig.from_dict(d)
    assert reconstructed.context_length == 56

    with pytest.raises(ValueError, match="context_length"):
        DeepARConfig(context_length=-5).validate()
    with pytest.raises(ValueError, match="distribution"):
        DeepARConfig(distribution="unsupported").validate()


def test_static_category_encoder():
    """Verify categorical encoder reserves index 0 for unknown categories."""
    df = pd.DataFrame({
        "item_id": ["item_1", "item_2", "item_3"],
        "store_id": ["CA_1", "CA_2", "CA_1"],
        "dept_id": ["D1", "D2", "D1"],
        "cat_id": ["C1", "C2", "C1"],
        "state_id": ["CA", "CA", "CA"],
    })
    encoder = StaticCategoryEncoder().fit(df)
    assert encoder.vocab_sizes_["item_id"] == 4  # 3 items + 1 <unk>
    assert encoder.vocab_sizes_["store_id"] == 3 # 2 stores + 1 <unk>

    test_df = pd.DataFrame({
        "item_id": ["item_1", "unseen_item"],
        "store_id": ["CA_1", "unseen_store"],
        "dept_id": ["D1", "D1"],
        "cat_id": ["C1", "C1"],
        "state_id": ["CA", "CA"],
    })
    encoded = encoder.transform_series(test_df)
    assert encoded["item_id"][0] > 0
    assert encoded["item_id"][1] == 0  # <unk> mapped to 0
    assert encoded["store_id"][1] == 0 # <unk> mapped to 0


def test_dynamic_covariates_extraction():
    """Verify extraction of cyclical and indicator calendar features."""
    df = pd.DataFrame({
        "wday": [1, 2, 3, 7],
        "month": [1, 6, 12, 1],
        "weekday": ["Saturday", "Sunday", "Monday", "Friday"],
        "event_name_1": ["SuperBowl", None, "nan", "Easter"],
        "event_name_2": [None, None, None, None],
        "day_num": [1, 100, 500, 1941],
    })
    cov = extract_dynamic_covariates(df)
    assert cov.shape == (4, 8)
    assert np.isfinite(cov).all()
    # wday=1 and wday=2 are weekend
    assert cov[0, 4] == 1.0  # is_weekend
    assert cov[1, 4] == 1.0  # is_weekend
    assert cov[2, 4] == 0.0  # Monday


def test_negative_binomial_head():
    """Verify Negative Binomial parameter head, distribution, and log_prob."""
    head = NegativeBinomialHead(hidden_dim=16)
    h = torch.randn(4, 5, 16)
    scale = torch.ones(4) * 2.0
    mu, alpha = head(h, scale)
    assert (mu > 0).all()
    assert (alpha > 0).all()

    dist = head.distribution(mu, alpha)
    y = torch.tensor([[[0.0], [1.0], [3.0], [0.0], [5.0]]]).expand(4, 5, 1).squeeze(-1)
    lp = dist.log_prob(y)
    assert torch.isfinite(lp).all()

    samples = head.sample(mu, alpha)
    assert samples.shape == mu.shape
    assert (samples >= 0).all()


def test_deepar_network_forward_and_forecast():
    """Verify neural network forward pass and autoregressive forecast."""
    config = DeepARConfig(context_length=14, prediction_length=7, hidden_dim=32, num_layers=1)
    vocab_sizes = {"item_id": 5, "store_id": 3, "dept_id": 2, "cat_id": 2, "state_id": 2}
    model = DeepARNetwork(config, vocab_sizes, covariate_dim=8)

    B, T = 4, 21
    target = torch.poisson(torch.ones(B, T) * 2.0)
    covariates = torch.randn(B, T, 8)
    static_cat = torch.ones(B, 5, dtype=torch.long)
    scale = torch.ones(B) * 3.0

    loss, (mu, alpha) = model(target, covariates, static_cat, scale)
    assert torch.isfinite(loss)
    assert mu.shape == (B, T)

    # Forecast inference
    ctx_target = target[:, :14]
    ctx_cov = covariates[:, :14]
    fut_cov = covariates[:, 14:]
    samples = model.forecast(ctx_target, ctx_cov, fut_cov, static_cat, scale, num_samples=25)
    assert samples.shape == (B, 25, 7)
    assert (samples >= 0).all()


def test_deepar_forecaster_tiny_integration(tiny_panel):
    """Test full BaseForecaster lifecycle and run_backtest integration on tiny panel."""
    panel, cfg = tiny_panel
    deepar_cfg = DeepARConfig(
        context_length=4,
        prediction_length=3,
        hidden_dim=16,
        num_layers=1,
        epochs=3,
        batch_size=4,
        num_samples=20,
        val_window_length=2,
        windows_per_series_train=3,
        device="cpu",
        seed=42,
    )
    factory = make_deepar_factory(deepar_cfg)
    predictions, metrics, metadata = run_backtest(panel, cfg, factory)

    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == 18  # 3 series * 2 folds * 3 horizons
    assert predictions["model"].eq("deepar").all()
    assert (predictions["y_pred"] >= 0).all()
    assert np.isfinite(predictions["y_pred"]).all()

    # Check quantiles ordering
    assert (predictions["q10"] <= predictions["q50"]).all()
    assert (predictions["q50"] <= predictions["q90"]).all()

    # Verify metrics were computed
    assert len(metrics) == 3  # 2 folds + 1 pooled
    assert metrics.scope.iloc[-1] == "all_folds_pooled"
    assert "mae" in metrics.columns
    assert "pinball_q50" in metrics.columns
    assert np.isfinite(metrics["pinball_q50"].iloc[-1])


def test_deepar_checkpointing(tiny_panel, tmp_path):
    """Verify checkpoint save and reload produce identical model architectures."""
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])

    deepar_cfg = DeepARConfig(
        context_length=4,
        prediction_length=3,
        hidden_dim=16,
        num_layers=1,
        epochs=2,
        batch_size=4,
        val_window_length=2,
        device="cpu",
    )

    forecaster = DeepARForecaster(config=deepar_cfg, checkpoint_path=tmp_path / "ckpt")
    forecaster.fit(inputs)
    assert (tmp_path / "ckpt" / "deepar_checkpoint.pt").exists()

    reloaded_model, reloaded_cfg, meta = DeepARTrainer.load_checkpoint(tmp_path / "ckpt", device=torch.device("cpu"))
    assert reloaded_cfg.hidden_dim == deepar_cfg.hidden_dim
    assert meta["best_epoch"] >= 1


def test_deepar_reproducibility(tiny_panel):
    """Verify that same random seed produces reproducible forecasts."""
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])

    deepar_cfg = DeepARConfig(
        context_length=4,
        prediction_length=3,
        hidden_dim=16,
        num_layers=1,
        epochs=3,
        batch_size=4,
        num_samples=25,
        val_window_length=2,
        device="cpu",
        seed=123,
    )

    pred1 = DeepARForecaster(config=deepar_cfg).fit(inputs).predict(inputs)
    pred2 = DeepARForecaster(config=deepar_cfg).fit(inputs).predict(inputs)

    np.testing.assert_allclose(
        pred1["y_pred"].to_numpy(),
        pred2["y_pred"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pred1["q10"].to_numpy(),
        pred2["q10"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pred1["q90"].to_numpy(),
        pred2["q90"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_deepar_future_leakage_safety(tiny_panel):
    """Mutating future sales in panel must never affect forecasts produced at origin."""
    panel, cfg = tiny_panel
    fold = development_splits(cfg)[0]
    inputs_original = construct_inputs(panel, fold)

    deepar_cfg = DeepARConfig(
        context_length=4,
        prediction_length=3,
        hidden_dim=16,
        num_layers=1,
        epochs=2,
        batch_size=4,
        num_samples=20,
        val_window_length=2,
        device="cpu",
        seed=42,
    )

    pred_original = DeepARForecaster(config=deepar_cfg).fit(inputs_original).predict(inputs_original)

    # Mutate future targets in panel
    mutated_panel = panel.copy(deep=True)
    future_mask = mutated_panel["day_num"] > fold.origin
    mutated_panel.loc[future_mask, "units_sold"] += 1000.0

    inputs_mutated = construct_inputs(mutated_panel, fold)
    pred_mutated = DeepARForecaster(config=deepar_cfg).fit(inputs_mutated).predict(inputs_mutated)

    pd.testing.assert_frame_equal(pred_original, pred_mutated)


def test_deepar_covariate_toggles(tiny_panel):
    """Verify that static-only and no-calendar configurations execute properly."""
    panel, cfg = tiny_panel
    inputs = construct_inputs(panel, development_splits(cfg)[0])

    # No calendar features
    cfg_no_cal = DeepARConfig(
        context_length=4,
        prediction_length=3,
        hidden_dim=16,
        num_layers=1,
        epochs=1,
        batch_size=4,
        num_samples=10,
        val_window_length=2,
        use_calendar_features=False,
        use_static_features=True,
        device="cpu",
    )
    pred_no_cal = DeepARForecaster(config=cfg_no_cal).fit(inputs).predict(inputs)
    assert len(pred_no_cal) == 9
    assert (pred_no_cal["y_pred"] >= 0).all()

    # Sales only (no static, no calendar)
    cfg_sales_only = DeepARConfig(
        context_length=4,
        prediction_length=3,
        hidden_dim=16,
        num_layers=1,
        epochs=1,
        batch_size=4,
        num_samples=10,
        val_window_length=2,
        use_calendar_features=False,
        use_static_features=False,
        device="cpu",
    )
    pred_sales_only = DeepARForecaster(config=cfg_sales_only).fit(inputs).predict(inputs)
    assert len(pred_sales_only) == 9
    assert (pred_sales_only["y_pred"] >= 0).all()

