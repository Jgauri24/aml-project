"""Configuration for DeepAR Global Probabilistic Forecasting Model.

Defines all hyperparameters for model architecture, training, distribution choice,
and autoregressive inference sampling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json
import yaml


@dataclass
class DeepARConfig:
    """Hyperparameters and execution configuration for DeepAR."""

    # Time series sequence lengths
    context_length: int = 56
    prediction_length: int = 28

    # Network architecture
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1

    # Feature representations
    use_static_features: bool = True
    use_calendar_features: bool = True
    embedding_dim_map: Dict[str, int] = field(default_factory=lambda: {
        "item_id": 16,
        "store_id": 4,
        "dept_id": 4,
        "cat_id": 2,
        "state_id": 2,
    })

    # Distribution & scaling
    distribution: str = "negative_binomial"  # "negative_binomial" or "gaussian"
    scaling: bool = True                     # Mean scaling nu = 1 + (1/T) sum y_t
    missing_target_policy: str = "zero"      # "zero", "forward_fill", or "mean"

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 35
    early_stopping_patience: int = 8
    gradient_clip_norm: float = 10.0
    val_window_length: int = 28              # Inner chronological validation window (days)
    windows_per_series_train: int = 8        # Number of rolling windows sampled per series

    # Inference & sampling
    num_samples: int = 200
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Runtime & reproducibility
    device: str = "auto"                     # "auto", "cpu", "mps", "cuda"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeepARConfig:
        """Create configuration from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        Path(path).write_text(yaml.dump(self.to_dict(), default_flow_style=False))

    @classmethod
    def from_yaml(cls, path: Path | str) -> DeepARConfig:
        """Load configuration from YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate parameter consistency."""
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.prediction_length <= 0:
            raise ValueError(f"prediction_length must be positive, got {self.prediction_length}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if self.distribution not in ("negative_binomial", "gaussian"):
            raise ValueError(f"Unsupported distribution '{self.distribution}'")
        if self.missing_target_policy not in ("zero", "forward_fill", "mean"):
            raise ValueError(f"Unsupported missing_target_policy '{self.missing_target_policy}'")
        if self.quantiles != sorted(self.quantiles):
            raise ValueError("Quantiles must be sorted in ascending order")

