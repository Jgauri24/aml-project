# Forecast performance plots

Figures and plotted datasets were generated from the checked-in experimental CSV summaries. No model values are embedded in the plotting script. See `plotted_data/analysis_availability.csv` for supported and unavailable analyses. The source folder contains pooled retail summaries, but no row-level scored predictions or demand panel; SKU/store residual plots and date-level actual-versus-quantile trajectories cannot be reconstructed. Runtime sources use different experiment scopes and should not be read as a standardized cross-family speed ranking.

Run from the project root: `python -m scripts.plot_forecast_performance`.
