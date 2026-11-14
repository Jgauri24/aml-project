# Forecast dashboard

Run `python -m streamlit run app.py` after installing `requirements_ui.txt`.
The app reads genuine saved evaluation artifacts from `artifacts/release/`.
A compact historical slice supports every selected SKU without downloading raw
M5 data. No model training or synthetic prediction fallback occurs in the app.

## Forecast explorer

Select an evaluation scope, store, product, origin and model subset. Show 28–112
historical days, toggle the Kalman/DeepAR P10–P90 intervals, and optionally reveal
actual evaluation sales. Download exactly the selected forecast rows as CSV.

## Model benchmarks

Compare pooled cohort metrics, WAPE by forecast week and measured model costs.
Metrics here concern all 240 series, not only the currently selected product.
WAPE is a ratio; MAE, RMSE and width use sales units. NA means unavailable.

## Demand segments

Inspect historical volume and zero-sales frequency, plus MAE by frequency group.
Labels use history through d_1801. Dense means fewer than 10% zero days,
intermittent means 10–less than 50%, and sparse means at least 50%.
Neither a zero-sales day nor a missing listing proves a stockout.

## Research protocol

The final evaluation scope is the single recorded April 25–May 22, 2016 holdout.
The viewer reads this saved evidence and does not perform new parameter
selection. The protocol tab exposes the completion receipt. For full data and
model replication, follow [REPRODUCING.md](REPRODUCING.md).
