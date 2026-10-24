"""Reproducible descriptive analysis; every analytical observation is <= EDA cutoff."""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.config import load_config, data_path
from src.data.understanding import markdown_table
from src.data.preprocess import load_panel

GROUPS = ["low", "medium", "high"]


def longest_zero_run(values):
    mask = np.asarray(values) == 0  # Missing observations break a run.
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if len(starts) else 0


def autocorrelation(values, lag):
    x = np.asarray(values, dtype=float)
    a, b = x[:-lag], x[lag:]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def analyze(panel, selected, cfg):
    cutoff = cfg["eda"]["cutoff_day"]
    # Filter first, before means, representatives, rolling statistics, or correlations.
    safe = panel.loc[panel.day_num.le(cutoff)].copy()
    safe = safe.merge(selected[["series_id", "volume_group"]], on="series_id", how="inner", validate="many_to_one")
    if safe.empty or safe.day_num.max() > cfg["selection"]["cutoff_day"]:
        raise ValueError("EDA exceeds safe observation range")
    safe = safe.sort_values(["series_id", "day_num"]).reset_index(drop=True)
    rows = []
    lags = cfg["eda"]["acf_lags"]
    for series_id, frame in safe.groupby("series_id", observed=True):
        y = frame.units_sold.astype(float)
        positive = y[y > 0]
        mean, std = y.mean(), y.std(ddof=1)
        adi = float(y.notna().sum() / len(positive)) if len(positive) else float("inf")
        cv2 = float((positive.std(ddof=1) / positive.mean()) ** 2) if len(positive) > 1 else float("nan")
        if not len(positive):
            pattern = "all_zero"
        elif not np.isfinite(cv2):
            pattern = "insufficient_positive_sales"
        else:
            intermittent = adi >= cfg["eda"]["intermittency_adi_threshold"]
            variable = cv2 >= cfg["eda"]["variability_cv2_threshold"]
            pattern = ("lumpy" if variable else "intermittent") if intermittent else ("erratic" if variable else "smooth")
        price_valid = frame[["units_sold", "sell_price"]].dropna()
        enough = len(price_valid) >= cfg["eda"]["minimum_price_pairs"] and price_valid.sell_price.nunique() > 1 and price_valid.units_sold.nunique() > 1
        price_corr = float(price_valid.units_sold.astype(float).corr(price_valid.sell_price.astype(float))) if enough else float("nan")
        price_values = frame.sell_price.dropna()
        rows.append({"series_id": str(series_id), "volume_group": frame.volume_group.iloc[0],
                     "mean": mean, "median": y.median(), "variance": y.var(ddof=1), "std": std,
                     "maximum": y.max(), "cv": std / mean if mean else float("nan"),
                     "zero_fraction": float(y.eq(0).sum() / y.notna().sum()),
                     "max_zero_run": longest_zero_run(y), "adi": adi, "positive_cv2": cv2,
                     "pattern": pattern, "price_missing_fraction": frame.sell_price.isna().mean(),
                     "price_unique_values": frame.sell_price.nunique(),
                     "price_min": price_values.min(), "price_max": price_values.max(),
                     "price_sales_corr": price_corr,
                     **{f"acf_{lag}": autocorrelation(y, lag) for lag in lags}})
    stats = pd.DataFrame(rows)
    means = stats.set_index("series_id")["mean"]
    safe["relative_sales"] = safe.units_sold / safe.series_id.map(means).astype(float).replace(0, np.nan)
    # Equal series weights for seasonal comparisons, then average within volume group.
    weekday = safe.groupby(["series_id", "volume_group", "wday", "weekday"], observed=True).relative_sales.mean().reset_index()
    weekday = weekday.groupby(["volume_group", "wday", "weekday"], observed=True).relative_sales.mean().reset_index().sort_values("wday")
    monthly = safe.groupby(["series_id", "volume_group", "month"], observed=True).relative_sales.mean().reset_index()
    monthly = monthly.groupby(["volume_group", "month"], observed=True).relative_sales.mean().reset_index()
    safe["listed_event"] = safe.event_name_1.notna() | safe.event_name_2.notna()
    events = safe.groupby(["series_id", "volume_group", "listed_event"], observed=True).relative_sales.mean().reset_index()
    events = events.groupby(["volume_group", "listed_event"], observed=True).relative_sales.mean().reset_index()
    snap = safe.groupby(["series_id", "state_id", "snap"], observed=True).relative_sales.mean().reset_index()
    snap = snap.groupby(["state_id", "snap"], observed=True).relative_sales.mean().reset_index()
    named_events = safe.loc[safe.event_name_1.notna()].groupby("event_name_1", observed=True).agg(
        mean_relative_sales=("relative_sales", "mean"), series_days=("series_id", "size"), distinct_dates=("date", "nunique")).reset_index()
    return safe, stats, {"weekday": weekday, "monthly": monthly, "events": events, "snap": snap, "named_events": named_events}


def make_figures(safe, stats, summaries, cfg):
    directory = data_path(cfg, "reports") / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    filenames = []
    cutoff_date = safe.date.max().date()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=cfg["eda"]["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
        filenames.append(name)

    fig, ax = plt.subplots(figsize=(9, 4))
    for group in GROUPS:
        values = np.log1p(safe.loc[safe.volume_group.eq(group), "units_sold"].dropna())
        ax.hist(values, bins=np.linspace(0, np.log1p(safe.units_sold.max()), 45), density=True, histtype="step", linewidth=1.6, label=group)
    ax.set(title=f"Observed sales distributions through {cutoff_date}", xlabel="log(1 + daily unit sales)", ylabel="Density")
    ax.legend(title="Selection group")
    save(fig, "01_sales_distribution.png")

    representative = {}
    for group in GROUPS:
        pool = stats.loc[stats.volume_group.eq(group)].copy()
        representative[group] = pool.loc[(pool["mean"] - pool["mean"].median()).abs().idxmin(), "series_id"]
    for rolling in [False, True]:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for ax, group in zip(axes, GROUPS):
            series_id = representative[group]
            frame = safe.loc[safe.series_id.astype("string").eq(series_id)].copy()
            if rolling:
                for window in cfg["eda"]["rolling_windows"]:
                    frame[f"rolling_{window}"] = frame.units_sold.rolling(window, min_periods=window).mean()
                frame = frame.tail(cfg["eda"]["representative_days"])
            ax.plot(frame.date, frame.units_sold, alpha=0.45, linewidth=0.7, label="daily observed sales")
            if rolling:
                for window in cfg["eda"]["rolling_windows"]:
                    ax.plot(frame.date, frame[f"rolling_{window}"], label=f"trailing {window}-day mean", linewidth=1.4)
            ax.set(title=f"{group}: {series_id}", ylabel="Units/day")
            ax.legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel("Date (past data only)")
        save(fig, "03_rolling_averages.png" if rolling else "02_representative_histories.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=summaries["weekday"], x="wday", y="relative_sales", hue="volume_group", hue_order=GROUPS, marker="o", ax=axes[0])
    order = safe[["wday", "weekday"]].drop_duplicates().sort_values("wday")
    axes[0].set_xticks(order.wday, order.weekday.astype(str).str[:3])
    axes[0].set(title="Weekly pattern (equal weight per series)", xlabel="Weekday", ylabel="Sales / series historical mean")
    sns.lineplot(data=summaries["monthly"], x="month", y="relative_sales", hue="volume_group", hue_order=GROUPS, marker="o", ax=axes[1])
    axes[1].set(title="Month-of-year pattern (unadjusted)", xlabel="Month", ylabel="Sales / series historical mean", xticks=range(1, 13))
    save(fig, "04_seasonality.png")

    acf = stats.melt(id_vars=["series_id", "volume_group"], value_vars=[f"acf_{lag}" for lag in cfg["eda"]["acf_lags"]], var_name="lag", value_name="autocorrelation")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=acf, x="lag", y="autocorrelation", hue="volume_group", hue_order=GROUPS, ax=ax, showfliers=False)
    ax.set(title="Per-series autocorrelation distributions", xlabel="Lag in days", ylabel="Pearson correlation of lagged sales")
    save(fig, "05_autocorrelation.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=stats, x="volume_group", y="zero_fraction", order=GROUPS, ax=axes[0])
    sns.boxplot(data=stats, x="volume_group", y="max_zero_run", order=GROUPS, ax=axes[1])
    axes[0].set(title="Zero-sales share", xlabel="Group", ylabel="Fraction of observed days")
    axes[1].set(title="Longest consecutive zero-sales run", xlabel="Group", ylabel="Days")
    save(fig, "06_sparsity.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=summaries["events"], x="volume_group", y="relative_sales", hue="listed_event", order=GROUPS, ax=axes[0])
    sns.barplot(data=summaries["snap"], x="state_id", y="relative_sales", hue="snap", ax=axes[1])
    axes[0].legend(title="Listed event", loc="lower left")
    axes[1].legend(title="State SNAP flag", loc="lower left")
    axes[0].set(title="Listed event vs other days (unadjusted)", xlabel="Group", ylabel="Sales / series mean")
    axes[1].set(title="State SNAP flag association (unadjusted)", xlabel="State", ylabel="Sales / series mean")
    save(fig, "07_events_snap.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for group in GROUPS:
        frame = safe.loc[safe.series_id.astype("string").eq(representative[group])]
        normalized = frame.sell_price / frame.sell_price.dropna().median()
        axes[0].plot(frame.date, normalized, label=group, linewidth=0.8)
    axes[0].set(title="Representative historical price patterns", xlabel="Date", ylabel="Listed price / series median listed price")
    axes[0].legend(title="Group")
    sns.boxplot(data=stats, x="volume_group", y="price_sales_corr", order=GROUPS, ax=axes[1])
    axes[1].set(title="Within-series price/sales association", xlabel="Group", ylabel="Pearson correlation on paired days")
    save(fig, "08_prices.png")
    return filenames, representative


def write_findings(safe, stats, summaries, figures, representative, cfg):
    destination = data_path(cfg, "reports")
    group_summary = stats.groupby("volume_group", observed=True).agg(
        series=("series_id", "size"), median_mean=("mean", "median"), median_cv=("cv", "median"),
        median_zero_fraction=("zero_fraction", "median"), median_max_zero_run=("max_zero_run", "median"),
        median_acf_7=("acf_7", "median"), median_price_corr=("price_sales_corr", "median")).reindex(GROUPS)
    weekday = summaries["weekday"].groupby("wday").relative_sales.mean()
    acf_medians = {str(lag): float(stats[f"acf_{lag}"].median()) for lag in cfg["eda"]["acf_lags"]}
    price_corr = stats.price_sales_corr.dropna()
    patterns = stats.pattern.value_counts().to_dict()
    # Include simple state contrasts; exposure groups are descriptive, not a causal estimate.
    snap_wide = summaries["snap"].pivot(index="state_id", columns="snap", values="relative_sales")
    snap_ratio = (snap_wide[1] / snap_wide[0]).round(3).to_dict()
    facts = {"cutoff_day": int(safe.day_num.max()), "date_end": str(safe.date.max().date()),
             "series": len(stats), "rows": len(safe), "group_summary": group_summary.reset_index().to_dict("records"),
             "acf_medians": acf_medians, "patterns": patterns,
             "zero_fraction_pooled": float(safe.units_sold.eq(0).mean()),
             "longest_zero_run": int(stats.max_zero_run.max()),
             "weekday_peak_to_trough": float(weekday.max() / weekday.min()),
             "price_missing_fraction": float(safe.sell_price.isna().mean()),
             "valid_price_correlations": len(price_corr),
             "median_price_correlation": float(price_corr.median()),
             "snap_day_ratio_by_state": snap_ratio, "representative_series": representative,
             "figures": figures}
    (destination / "eda_summary.json").write_text(json.dumps(facts, indent=2))
    means = ", ".join(f"{g}: {group_summary.loc[g, 'median_mean']:.3f}" for g in GROUPS)
    zeros = ", ".join(f"{g}: {group_summary.loc[g, 'median_zero_fraction']:.1%}" for g in GROUPS)
    text = f'''# EDA findings

All observations are from **d_1–d_{facts['cutoff_day']}**, {safe.date.min().date()}–{facts['date_end']}: {facts['rows']:,} rows, {facts['series']} selected series. Nothing after the first development origin is used for statistics, representative-series choice, rolling averages or figures. Selection uses only the trailing 365 days at that origin; EDA describes all preceding available history. No holdout target is inspected here.

## 1. Volume and variability differ across the selected groups

**Observed:** Median per-series historical mean unit sales/day: {means}. The group summary below reports standard-deviation/mean variability as well as sparsity. Per-series mean, median, sample variance, sample standard deviation, maximum and CV are in `eda_series_statistics.csv`.

**Evidence:** Figure 01 and the table below; group membership came from a different, trailing-365-day mean, so these full-history summaries need not have perfectly disjoint ranges.

**Forecasting relevance:** Scale differences justify reporting a common pooled metric alongside per-group diagnostics in later milestones. A global model may share information while per-series models face different signal levels.

**Cannot conclude:** Stratified selection is not representative of all M5 series; larger volume does not prove easier prediction, and these summaries do not establish any model's performance.

## 2. Calendar structure and persistence are present but heterogeneous

**Observed:** Equal-series normalized weekday peak/trough ratio is {facts['weekday_peak_to_trough']:.3f}. Median per-series sales autocorrelations are {acf_medians}. Weekly and monthly profiles and representative trailing 7/28-day averages are plotted.

**Evidence:** Figures 02–05; `eda_weekday.csv` and `eda_monthly.csv`. Monthly means are month-of-year descriptions, not a detrended seasonality decomposition.

**Forecasting relevance:** Calendar covariates and a weekly seasonal naive reference will be reasonable hypotheses to test. Rolling plots separate short-term noise from longer changes descriptively.

**Cannot conclude:** Autocorrelation may reflect trends, prelaunch stretches or changing availability; pooled profiles do not guarantee stable seasonality or out-of-sample gains. No forecasting model was fitted.

## 3. Zeros and intermittent behavior require care

**Observed:** Median per-series zero-sales shares: {zeros}. Across selected series, {facts['zero_fraction_pooled']:.1%} of observed days have zero sales; the longest zero run is {facts['longest_zero_run']} days. Descriptive patterns: {patterns}.

**Evidence:** Figure 06 and per-series zero fractions, longest zero runs, average inter-demand intervals (ADI), and squared CV of positive sales. ADI is observed days / positive-sale days; classification thresholds are configurable ({cfg['eda']['intermittency_adi_threshold']}, {cfg['eda']['variability_cv2_threshold']}). All-zero and insufficient-positive-sale cases are explicit.

**Forecasting relevance:** Intermittency and long zero runs can make percentage metrics unstable, motivate count-aware uncertainty later, and challenge smooth per-series state models.

**Cannot conclude:** Zero sales are not confirmed stockouts or zero latent demand. Long zero runs can include an unobserved launch period. The descriptive classifications are not established product labels and do not filter out sparse series.

## 4. Event and SNAP associations are descriptive

**Observed:** State SNAP-flagged-day / other-day normalized mean ratios are {snap_ratio}. Event-day and non-event-day comparisons are reported by volume group, with named-event counts in `eda_named_events.csv`.

**Evidence:** Figure 07, `eda_events.csv`, `eda_snap.csv`, and `eda_named_events.csv`. Normalization gives series equal scale before exposure summaries; no weekday, price, season, trend or assortment adjustment is made.

**Forecasting relevance:** Known holidays/events are legitimate candidate future covariates. The initial availability policy conservatively uses SNAP only historically and excludes future SNAP until a realistic schedule assumption is documented.

**Cannot conclude:** These comparisons do not identify causal holiday or SNAP effects. Groups have unequal exposures, repeated observations and confounding; no significance test or independent-sample assumption is claimed.

## 5. Prices vary, and missing listings create availability uncertainty

**Observed:** Historical price-missing fraction is {facts['price_missing_fraction']:.1%}. There are {len(price_corr)} usable within-series price/sales correlations (minimum {cfg['eda']['minimum_price_pairs']} paired days, nonconstant price and sales), with median {facts['median_price_correlation']:.3f}. Distinct price counts and ranges are saved per series.

**Evidence:** Figure 08 and `eda_series_statistics.csv`. Correlations are computed within series on days with both observed sales and listed price; omitted undefined correlations are not treated as zero.

**Forecasting relevance:** Historical price can be useful context, but future actual prices are excluded from all forecast-input objects. Missing listings and launch/availability uncertainty should remain explicit quality flags.

**Cannot conclude:** The correlation is not a causal price elasticity: promotions, trends, seasonality and stock conditions can confound it. Missing price records do not prove unavailability. Weekly repeated price observations are not independent daily price experiments.

## Group evidence

{markdown_table(group_summary.reset_index().round(4))}

## Figures and reproducibility

''' + "\n".join(f"![{name}](figures/{name})\n" for name in figures) + '''
Run `python -m src.data.eda` after preprocessing, or execute `notebooks/02_eda.ipynb`. All statistics tables and figures are regenerated from the cutoff-safe panel. Sample variances use ddof=1; lag correlations are pairwise finite; constant-series correlations and zero-mean CV are undefined (NA), not silently set to zero. No imputation, causal estimation, fitted preprocessing or forecasting model is performed.
'''
    (destination / "eda_findings.md").write_text(text)
    return facts


def run_eda(cfg):
    processed, interim, reports = (data_path(cfg, key) for key in ["processed", "interim", "reports"])
    panel = load_panel(cfg, cutoff=cfg["eda"]["cutoff_day"])
    selected = pd.read_csv(interim / "selected_skus.csv")
    safe, stats, summaries = analyze(panel, selected, cfg)
    figures, representative = make_figures(safe, stats, summaries, cfg)
    stats.to_csv(reports / "eda_series_statistics.csv", index=False)
    for name, table in summaries.items():
        table.to_csv(reports / f"eda_{name}.csv", index=False)
    facts = write_findings(safe, stats, summaries, figures, representative, cfg)
    return safe, stats, summaries, facts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    _, _, _, facts = run_eda(load_config(args.config))
    print(json.dumps(facts, indent=2))
