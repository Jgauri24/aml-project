"""Compile the IEEE paper from recorded release evidence, not copied headline scores."""
import json
from pathlib import Path
import subprocess
import pandas as pd
from src.config import ROOT

OUT=ROOT/'artifacts/paper'
D=json.loads((ROOT/'artifacts/presentation/evidence.json').read_text())
N={'seasonal_naive':'Weekly naive','kalman':'Seasonal Kalman','lightgbm':'LightGBM','deepar':'DeepAR'}

def f(v): return '--' if v is None or pd.isna(v) else f'{v:.4f}'
def esc(s): return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def metric(scope,model): return next(r for r in D['benchmark'] if r['scope']==scope and r['model']==model)
def figure(name,caption,label,wide=True,width=None):
    kind='figure*' if wide else 'figure'
    width=width or (r'\textwidth' if wide else r'\columnwidth')
    return '\n'+rf'\begin{{{kind}}}[t]\centering\includegraphics[width={width}]{{figures/{name}.pdf}}\caption{{{caption}}}\label{{fig:{label}}}\end{{{kind}}}'+'\n'
def table(caption,cols,headers,rows,label,wide=False):
    kind='table*' if wide else 'table'
    return '\n'+rf'\begin{{{kind}}}[t]\caption{{{caption}}}\label{{tab:{label}}}\centering\footnotesize\begin{{tabular}}{{{cols}}}\toprule'+'\n'+ ' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+rf'\bottomrule\end{{tabular}}\end{{{kind}}}'+'\n'

parts=[r'''\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{mathptmx,amsmath,amssymb,graphicx,booktabs,array,microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{url,balance}
\graphicspath{{figures/}}
\hypersetup{pdftitle={Retail Demand Forecasting with Local State Space Models and Global Probabilistic Networks},pdfauthor={Tanish Garg, Ayush Mittal, Gauri Jindal, Sanchit Garg, Saumya Soni}}
\newcommand{\E}{\mathbb{E}}
\begin{document}
\title{Retail Demand Forecasting with Local State Space Models and Global Probabilistic Networks}
\author{\IEEEauthorblockN{Tanish Garg, Ayush Mittal, Gauri Jindal, Sanchit Garg, and Saumya Soni}
\IEEEauthorblockA{Department of Computer Science and Artificial Intelligence\\School of Technology, Rishihood University\\Sonipat, Haryana, India}}
\maketitle
\begin{abstract}
Retail sales combine sparse counts, heterogeneous product scales and calendar effects. We extend a local Kalman and global LightGBM comparison with a custom PyTorch DeepAR network, conditional uncertainty diagnostics and a reproducible forecasting dashboard. A fixed cohort of 240 Walmart M5 product-store series supports four rolling development origins and a separately frozen 28-day final holdout. Model selection uses earlier development evidence; a source and configuration manifest precedes final target scoring. All four model families share target keys, calendar availability and missing-actual masks. DeepAR generates 200 trajectories per series to obtain P10, P50 and P90 forecasts. We retain earlier development results with their original scope, including a single-origin neural ablation, rather than interpreting them as matched final-test evidence.
''']
final=sorted([r for r in D['benchmark'] if r['scope']=='final'],key=lambda r:r['mae'])
best=final[0]
parts += [f"On the final holdout, {N[best['model']]} obtains the lowest MAE, {best['mae']:.4f} units, with WAPE {best['wape']:.4f}. "
          f"Seasonal Kalman and DeepAR achieve empirical central-80\\% coverage of {100*metric('final','kalman')['coverage_80']:.2f}\\% and {100*metric('final','deepar')['coverage_80']:.2f}\\%, respectively. "
          "Conditional coverage, width and tail diagnostics expose differences hidden by aggregate accuracy. These are observed-sales forecasts, with no claim of recovered latent demand or measured inventory-cost savings.\n"]
parts += [r'''\end{abstract}
\begin{IEEEkeywords}retail demand, DeepAR, Kalman filtering, LightGBM, probabilistic forecasting, rolling-origin evaluation\end{IEEEkeywords}

\section{Introduction}
Retail planning requires forecasts for products whose daily sales can remain zero for long stretches and then spike. A point prediction helps estimate likely sales, while a distribution describes the range of plausible outcomes. Their quality must be assessed separately: low absolute error need not imply useful uncertainty for inventory decisions. Observed transactions also differ from unconstrained customer demand when availability suppresses purchases. Without inventory records, a zero cannot identify a stockout.

The initial project compared last-value and weekly seasonal baselines, several independent state-space models, and a global boosted-tree forecaster. This extension addresses milestones 10--16: a global probabilistic recurrent model, standardized datasets, matched rolling evaluation, demand-segment analysis, uncertainty diagnostics and final research packaging. The principal question is whether pooling information in a probabilistic neural model improves point accuracy or uncertainty relative to local Gaussian state models and global L1 boosting.

Our contributions are (i) a common, validated forecast schema and chronological protocol; (ii) a reloadable global neural model with static embeddings and known-future calendar inputs; (iii) an explicit separation of earlier development, current development and final holdout evidence; (iv) conditional coverage, tail and per-product error diagnostics; and (v) a read-only dashboard backed by saved predictions. We retain the initial cohort and model-selection evidence rather than redesigning the sample after observing final results.

\section{Related Work}
Kalman filtering estimates hidden states recursively in linear systems \cite{kalman}. Structural state-space models represent level, trend and seasonality with explicit transition and observation equations. Their analytic prediction variance offers interpretable uncertainty, although Gaussian assumptions require scrutiny for zero-heavy retail counts.

LightGBM implements efficient gradient boosting \cite{lightgbm}. In this project it learns a direct mapping from historical summaries, product identity, calendar variables and horizon to sales. DeepAR instead fits an autoregressive recurrent distribution jointly across related time series \cite{deepar}. Our implementation follows this global probabilistic approach with a PyTorch LSTM and negative-binomial likelihood. It is a custom implementation, not a claim of exact equivalence to the GluonTS estimator.

The M5 competition establishes a retail forecasting setting with heterogeneous item-store histories \cite{m5}. This study uses a small balanced cohort and pooled MAE, RMSE and WAPE. It does not reproduce the competition's full hierarchical WRMSSE ranking. Proper probabilistic scoring complements point losses: pinball loss assesses particular quantiles, while coverage alone can improve simply because intervals become wider \cite{gneiting}.

\section{Data and Information Boundaries}
\subsection{Fixed Cohort and Canonical Panel}
The M5 sales, calendar and selling-price files are joined into a daily canonical panel. The overlapping validation and evaluation sales files are never concatenated. Seed 42 selects 240 product-store series, 80 from each trailing-365-day volume tercile, using only information through $d_{1801}$. A store/category diversity rule preserves heterogeneity; no positive-sales minimum removes sparse products. The sample spans ten stores and three states. Its equal volume allocation is useful for diagnostics but is not population proportional.

The panel contains 465,840 rows, or 240 series over 1,941 indexed days. Integrity checks cover unique keys, date/day alignment, continuity, stable metadata and nonnegative targets. Missing sales remain unavailable for scoring rather than becoming genuine zeros. Historical missing prices do not remove observations or establish unavailability. Earlier exploration found 69.33\% zero-sales days through $d_{1801}$; that statistic describes historical observed sales, not lost demand.

\subsection{Chronology and Available Covariates}
The development origins are $1801,1829,1857,1885$, each followed by 28 target days. Thus each model contributes $240\times4\times28=26,880$ development forecasts. The final origin is $1913$, followed by days $1914$--$1941$ (April 25--May 22, 2016), giving 6,720 final forecasts per model. The four development target windows end at day 1913 and do not overlap the final targets.

Inputs are explicit projections: sales history through the origin, static identities and allowed future calendar fields. Models never receive future actual sales, selling prices, SNAP indicators or availability flags. The evaluator joins actuals only after predictions exist. Identical target histories are available at the interface, but effective use differs: Kalman fits a trailing 364-day window, LightGBM builds historical direct-horizon training blocks, and DeepAR samples training contexts across the expanding history. Those differences limit attribution of gains solely to local versus global estimation.
''']
parts += [table('Experiment design and evidence scopes','lrr',['Scope','Origins','Rows/model'],[
['Earlier baseline benchmark','4','26,880'],['Earlier DeepAR ablation','1 ($d_{1801}$)','6,720'],
['Frozen development reproduction','4','26,880'],['Final holdout','1 ($d_{1913}$)','6,720']], 'scopes')]
parts += [r'''\section{Forecasting Methods}
\subsection{Weekly Baseline and Local State Models}
The weekly seasonal baseline repeats the last seven calendar days. The selected local state model combines a level with weekly seasonal states and fits independent parameters per product-store series. In standard notation,
\begin{align}
y_t &= Z\alpha_t+\epsilon_t, & \epsilon_t&\sim N(0,R),\\
\alpha_{t+1}&=T\alpha_t+\eta_t, & \eta_t&\sim N(0,Q).
\end{align}
The selected seasonal level model has seven states: a level and six weekly seasonal coordinates. Kalman recursions update the filtered state and covariance. Forecast quantiles follow the predictive Gaussian variance, with nonnegative clipping. Clipping prevents negative demand quantiles but does not create a correctly specified count distribution. Earlier local-trend variants remain in the historical benchmark, while the final comparison uses the previously selected seasonal level variant.

\subsection{Global Direct-Horizon LightGBM}
One pooled LightGBM model per origin predicts all products and all 28 horizons. Its direct training rows use historical origin/horizon pairs with targets already observed before the current origin. Lag availability follows the requested horizon, and origin-level rolling statistics remain frozen during inference. Features include lagged demand, rolling means and dispersion, horizon, categorical identities, calendar fields and permitted historical price/SNAP information. Future realized price and SNAP values remain excluded.

The L1 objective targets conditional medians, aligning naturally with MAE. Previously selected settings use 31 leaves, 500 trees and learning rate 0.05. The initial tuning used the first three development origins, reserving the fourth from that tuning procedure. We carry the chosen settings into the release without a new search. This implementation produces point forecasts; no residual band or artificial quantile is attached to it.

\subsection{DeepAR-Compatible Datasets}
Each product-store history becomes a separate target sequence. Static categorical mappings cover item, store, department, category and state. Unknown categories map to index zero, but an unknown embedding alone does not validate generalization to unseen items. Dynamic inputs encode weekday and month cyclically, weekend and event indicators, and normalized day progression. Only calendar features available at the forecast origin enter future positions.

A 56-day context provides eight weekly cycles and the prediction length is fixed at 28. Inner chronological validation reserves the last 28 historical days before each outer origin. Per-series scaling uses $s_i=1+\operatorname{mean}(y_{i,t})$ over the inner training portion. The configurable missing-target policy is frozen to zero imputation. Importantly, the evaluated canonical targets have no missing sales; missing-demand behavior is a software capability rather than an empirical result on this cohort. Any application with missing targets should evaluate masking-aware likelihoods before treating imputations as observed outcomes.

Training windows and validation windows derive from history only. Static category encoding is fitted to available series identities. The dataset includes target tensors, calendar covariates, category indices and scales; inference tensors combine the most recent context with the 28 known future calendar positions.

\subsection{Global Autoregressive Distribution}
A two-layer LSTM with 32 hidden units shares parameters across all 240 series. Static embeddings condition the recurrent state alongside scaled lagged demand and calendar inputs. The network parameterizes a negative-binomial distribution with mean $\mu$ and positive dispersion $a$, using the count variance $\mu+\mu^2/a$. This supports overdispersion but does not by itself implement a separate zero-inflation component.
\begin{equation}
p(\mathbf y_{T+1:T+28}\mid\mathcal F_T)=\prod_{h=1}^{28}p(y_{T+h}\mid\mathbf y_{T+1:T+h-1},\mathcal F_T).
\end{equation}
Training minimizes average negative log likelihood with Adam, gradient clipping and an inner-validation learning-rate scheduler. Early stopping has patience eight, with a maximum of 20 epochs. The best inner-validation checkpoint supplies inference weights. Teacher forcing uses observed historical lagged targets during training; inference recursively samples generated lagged targets.

For every series, 200 sampled trajectories yield empirical P10, P50 and P90 values at each day. The shared point forecast is P50; sampled means are distinct and are not substituted silently. Quantile order and finiteness are validated against the exact future grid. A finite sample of 200 trajectories introduces Monte Carlo variability, particularly in tails; seed 42 and recorded CPU worker settings record the tested reproducibility conditions.
''']
parts += [table('Frozen release hyperparameters','ll',['Component','Setting'],[
['Kalman','Seasonal level, 364 days'],['LightGBM','31 leaves, 500 trees, LR 0.05'],
['DeepAR context / horizon','56 / 28 days'],['DeepAR hidden / layers','32 / 2'],['DeepAR optimizer','Adam, LR 0.001, batch 32'],
['DeepAR stopping','20 epochs maximum, patience 8'],['DeepAR distribution','Negative binomial, 200 paths'],['Reproducibility','Seed 42; Kalman 4 workers']], 'hyper')]
parts += [figure('training','Final DeepAR training and inner-validation negative log likelihood. Inner validation ends at the forecast origin and excludes the final holdout. The best checkpoint is selected within the frozen stopping rule.','training',False)]
parts += [r'''\section{Evaluation and Finalization Protocol}
\subsection{Shared Scoring and Metrics}
Forecast rows identify model, series, origin, horizon, target day and date, point prediction and optional quantiles. The validator requires a complete unique grid, finite nonnegative point predictions, and either all valid ordered quantiles or no quantiles. An independent expected grid catches dropped difficult series and mismatched dates. All models use the same canonical actuals and missing-observation mask.

For $M$ observed forecast targets, we compute
\begin{align}
\operatorname{MAE}&=\frac{1}{M}\sum_j|y_j-\hat y_j|,\\
\operatorname{RMSE}&=\sqrt{\frac{1}{M}\sum_j(y_j-\hat y_j)^2},\\
\operatorname{WAPE}&=\frac{\sum_j|y_j-\hat y_j|}{\sum_j|y_j|}.
\end{align}
WAPE is a ratio and is undefined when its denominator is zero. On a common target grid, MAE and WAPE necessarily rank models identically because their denominators differ only by a model-independent constant. They are not independent confirmations of a ranking.

Quantile loss is
\begin{equation}
\operatorname{PL}_{\tau}=\frac1M\sum_j\max\{\tau(y_j-\hat q_{\tau,j}),(\tau-1)(y_j-\hat q_{\tau,j})\}.
\end{equation}
We report inclusive P10--P90 coverage and mean interval width. Lower misses satisfy $y<q_{.1}$ and upper misses $y>q_{.9}$. Coverage plus these two miss fractions equals one. For discrete demand, ties imply that a valid quantile can satisfy $P(Y<q_\tau)\leq\tau\leq P(Y\leq q_\tau)$. Accordingly, exact 80\% empirical coverage is not guaranteed even for a valid count forecast. CDF bounds, width and pinball losses support interpretation.

\subsection{Freeze Before Final Scoring}
The release records hashes of model/evaluation sources, experiment settings, selected hyperparameters and the canonical dataset before final scoring. LightGBM remains the prespecified primary model based on earlier development WAPE. The Kalman variant and DeepAR architecture also come from earlier evidence; the latter selection relies on a single-origin ablation and therefore has limited selection robustness.

A separate final runner leaves the development backtester unchanged. It fits every model on projected history, validates forecasts and saves artifacts before attaching holdout actuals. Exclusive creation of a scoring-start record prevents accidental repeated scoring in this release, and the completion receipt records target limits, prediction hashes and the frozen specification. Existing completed outputs are reused rather than automatically rerun. This software guard cannot prove that no human ever inspected historical files; it documents the explicit separation enforced by this execution.

No settings change after final metrics become available. Four frozen models each contribute 6,720 holdout predictions. Training and inference timers exclude shared plotting and scoring. All cost comparisons use the same machine; PyTorch and LightGBM use one thread while Kalman uses four independent fit workers. Thus model-specific preparation and algorithms differ. They describe this implementation, not universal runtime rankings.

\section{Results}
\subsection{Earlier Development Evidence}
The earlier eight-variant benchmark favored global boosting for absolute error and seasonal Kalman for squared error. Table~\ref{tab:earlier} preserves those scores for comparison. These are selection-stage development results, not final holdout outcomes. The new release deliberately limits the matched comparison to four model families with their frozen settings.
''']
oldnames={'kalman_single_sku':'Kalman reference','lightgbm':'LightGBM','kalman_local_level_seasonal':'Kalman LL+S','kalman_local_level':'Kalman LL','kalman_local_trend':'Kalman LT','kalman_local_trend_seasonal':'Kalman LT+S','naive_last_value':'Last value','naive_weekly_seasonal':'Weekly naive'}
parts += [table('Earlier development benchmark (four origins)','lrrr',['Model','MAE','RMSE','WAPE'],[[oldnames[r['model']],f(r['mae']),f(r['rmse']),f(r['wape'])] for r in sorted(D['historical'],key=lambda r:r['mae'])],'earlier')]
parts += [r'''The previous DeepAR experiments used only $d_{1801}$, as specified by their runner. They therefore cannot be placed in the earlier four-origin ranking as equally pooled results. Table~\ref{tab:ablation} preserves their actual scope. The hidden-32 network had the lowest WAPE in that experiment. The sales-only configuration also outperformed the full-feature hidden-64 configuration, so the table does not establish a universal benefit from added covariates. Small differences between context lengths should not be overstated from one origin.
''']
abl=[]
for r in D['ablation']:
    names={'Exp_A_Basic_DeepAR':'Full, hidden 64','Exp_B_Context_84':'Context 84','Exp_C_Capacity_Hidden32':'Hidden 32','Exp_D_Sales_Only':'Sales only','Exp_D_Sales_Plus_Static':'Sales + static'}
    abl.append([names[r['experiment']],str(r['context_length']),str(r['hidden_dim']),f(r['mae']),f(r['wape'])])
parts += [table('Earlier DeepAR ablation: origin 1801 only','lrrrr',['Configuration','Context','Hidden','MAE','WAPE'],abl,'ablation')]
parts += [r'''\subsection{Frozen Development and Final Holdout}
Table~\ref{tab:current} contains the current matched development and final benchmarks. Development scores remain influenced by earlier model selection. The final holdout offers a separate historical check, but its demand distribution differs from the development windows. Changes between those scopes must not be described as training improvements. We report the prespecified primary model even if another comparator performs better on a particular final metric.
''']
rows=[]
for scope in ['development','final']:
    for model in N:
        r=metric(scope,model)
        rows.append(['Dev.' if scope=='development' else 'Final',N[model],f(r['mae']),f(r['rmse']),f(r['wape']),f(r['pinball_q10']),f(r['pinball_q50']),f(r['pinball_q90']),f(r['coverage_80']),f(r['mean_interval_width_80'])])
parts += [table('Current matched benchmark: 26,880 development or 6,720 final observations per model. Coverage is a fraction; width and errors are in units sold. -- denotes unavailable uncertainty.','llrrrrrrrr',['Scope','Model','MAE','RMSE','WAPE','PL10','PL50','PL90','Cov80','Width'],rows,'current',True)]
parts += [figure('runtime','Final model fit and inference costs, using a logarithmic time axis to preserve visibility of the simple baseline. PyTorch and LightGBM use one thread; Kalman uses four fit workers.','runtime',False)]
parts += [figure('benchmark','Scope-separated WAPE for the four frozen models. Development and final values use different target windows; the panels are not a before/after causal estimate.','benchmark')]
for model in N:
    r=metric('final',model)
    parts.append(f"{N[model]} achieves final MAE {r['mae']:.4f}, RMSE {r['rmse']:.4f} and WAPE {r['wape']:.4f}. ")
parts += ['\n\n'+r'''RMSE weights large misses more strongly than MAE. The distinction matters when a model predicts medians to optimize absolute error while another uses a Gaussian state mean. Distribution and loss-function differences, effective history and covariates all contribute to these comparisons; they are not a controlled isolation of neural architecture.

Figure~\ref{fig:horizon} separates forecast weeks 1--7, 8--14, 15--21 and 22--28. Day-level and week-level tables remain machine readable. Pooled horizon curves can hide product-specific reversals, so the package also retains per-series comparisons and selected forecast examples.
''']
parts += [figure('horizon','WAPE by forecast week for development and final targets. Every curve within a panel shares the same series and target-date grid.','horizon')]
parts += [r'''\subsection{Demand Frequency and Cold-Start Limits}
We compute volume labels and zero fractions using history through the first development origin. Descriptive frequency groups use zero-sales shares below 10\% (dense), 10--less than 50\% (intermittent), and at least 50\% (sparse). These heuristic labels should not be confused with a formal ADI/CV-squared taxonomy. No target-period behavior determines group membership.

Figure~\ref{fig:segments} relates historical scale to zero frequency and compares final error within those groups. Because MAE is expressed in units, volume differences can drive segment contrasts. WAPE and observation counts accompany the released tables, and undefined zero-denominator scores remain unavailable. Group averages do not establish why a model fails for a particular item.

The cohort does not include a deliberately held-out unseen-item experiment. Nor do these new results establish performance under retrained 28-, 56- or 84-day histories. The earlier short-window/censoring exploration uses proxy or illustrative assumptions in places; it is not promoted here as a new measured cold-start result. A valid unseen-item test must define vocabulary fitting and the information available for a new item before evaluating it.
''']
parts += [figure('segments','Historical demand structure and final error by demand-frequency segment. Segment definitions use only pre-evaluation history. Neither zero-sales frequency nor missing prices identifies true stockouts.','segments')]
parts += [r'''\subsection{Uncertainty and Failure Diagnostics}
Table~\ref{tab:tails} reports the two genuine probabilistic models. Point-only forecasts retain unavailable interval metrics. Conditional results in Fig.~\ref{fig:calibration} compare coverage and width across training-volume groups. An aggregate coverage close to 80\% can conceal asymmetric tails or substantial differences across products. Discrete ties, especially at zero, also affect the meaning of overcoverage.
''']
tails=[]
for model in ['kalman','deepar']:
    p=pd.read_parquet(ROOT/'artifacts/release/final/scored_predictions.parquet').query('model == @model')
    v=(p.actual.notna())
    tails.append([N[model],f(metric('final',model)['coverage_80']),f(float((p.loc[v,'actual']<p.loc[v,'q10']).mean())),f(float((p.loc[v,'actual']>p.loc[v,'q90']).mean())),f(metric('final',model)['mean_interval_width_80'])])
parts += [table('Final coverage, lower/upper misses and interval width','lrrrr',['Model','Coverage','Lower','Upper','Width'],tails,'tails')]
parts += [figure('calibration','Final inclusive P10--P90 coverage and interval width by historical volume. The dashed coverage reference is 0.8. Width must be considered alongside coverage and count-distribution ties.','calibration')]
pairs=pd.DataFrame(D['pairs']).query("scope == 'final'").winner.value_counts()
parts += [f"On per-series final MAE, Kalman outperforms DeepAR on {int(pairs.get('kalman',0))} series, DeepAR outperforms Kalman on {int(pairs.get('deepar',0))}, and {int(pairs.get('tie',0))} are numerical ties. "
          "These counts are paired on the same targets and do not imply statistical independence across products. A numerical tie uses absolute tolerance $10^{-12}$.\n\n"]
parts += [r'''Representative products in Fig.~\ref{fig:examples} are selected from the middle of each training-volume rank, with identifiers resolving ties. This rule does not cherry-pick final successes. The plots show recent history, all four point forecasts, genuine Kalman/DeepAR ranges and observed final outcomes. Long zero stretches can make medians zero while upper quantiles remain positive; occasional demand spikes can still exceed P90. A wide band is not proof of calibrated tail risk, and an interval miss does not identify a stockout.
''']
parts += [figure('forecast_examples','Final 28-day forecasts for one deterministic representative from each training-volume group. Shading shows the Kalman and DeepAR P10--P90 ranges; the dotted line is observed holdout sales.','examples')]
parts += [r'''\subsection{Computational Cost}
Figure~\ref{fig:runtime} and Table~\ref{tab:runtime} report measured costs. The timer surrounds each model's fit and predict calls separately. Shared input construction, score calculation and document rendering are excluded. Neural training includes internal dataset preparation and checkpoint handling inside fitting; model sizes and hardware-specific kernels can change rankings. These costs should therefore be interpreted as reproducibility evidence for the tested machine.
''']
parts += [table('Final fit and inference time (seconds)','lrr',['Model','Fit','Predict'],[[N[r['model']],f(r['fit_seconds']),f(r['predict_seconds'])] for r in D['runtime']],'runtime',False)]
parts += [r'''\section{Dashboard and Reproducible Package}
The Streamlit viewer selects a store, SKU, origin and model subset, displays observed history and the 28-day forecast, and toggles predictive intervals and observed evaluation outcomes. Benchmark, runtime and demand-segment views use the same saved artifacts. Downloads expose selected prediction rows rather than screenshots alone. The application performs no online training, recomputes no model selection, and never fabricates a missing model's forecasts.

The repository separates reusable model, data, evaluation and analysis modules from CLI runners, notebooks, final artifacts and documentation. The release includes model checkpoints, LightGBM feature state, category mappings, scale factors, forecasts, metrics, runtimes and SHA-256 manifests. A compact historical slice supports the demonstration after cloning without redistributing the raw M5 files. Full reproduction requires the original dataset and the pinned model environment.

A clean clone can inspect the packaged demonstration immediately, prepare the canonical dataset with documented commands, and rerun development experiments. The recorded final evaluation is deliberately not automatically overwritten: an independent replication uses a new output root with its own freeze record. Published receipts preserve the original single evaluation rather than silently treating a rerun as a fresh untouched holdout. Repository location: \url{https://github.com/Jgauri24/aml-project}.

\section{Limitations and Future Work}
The final window supplies one additional historical origin, not repeated independent out-of-sample trials. Four development origins and a 240-series balanced sample are insufficient to claim universal retail superiority. Some earlier hyperparameter decisions used development outcomes, and the DeepAR architecture selection used one origin. Cross-series dependence and selection uncertainty constrain formal significance claims; no independent-day confidence interval is presented.

The model families differ in history length, covariates, likelihood and objective. A matched feature/window ablation would help identify the effect of pooling. The custom negative-binomial DeepAR is not a tuned reproduction of every GluonTS default, and a 20-epoch budget with 200 sampled paths leaves optimization and tail-sampling uncertainty. Gaussian Kalman intervals remain approximations for discrete nonnegative counts. Quantile LightGBM, dedicated intermittent-demand baselines, hierarchical reconciliation and count-aware calibration are useful extensions.

Observed sales do not identify lost sales or verified availability. Sparse-history experiments require actual retraining under a declared truncation scheme; unseen-item experiments require a deliberate unknown-item strategy. We do not report improvements in monetary cost, service level or true stockout rate. Forecast-to-inventory translation would additionally require lead-time distributions, holding/shortage costs and a validated replenishment simulator.

\section{Conclusion}
This release extends the original deterministic retail forecasting experiment with a global probabilistic recurrent comparator, conditional uncertainty diagnostics, a guarded final evaluation and an artifact-backed demonstration. The final benchmark separates the prespecified model decision from post-evaluation rankings, and the historical comparison retains each experiment's actual scope. Useful uncertainty assessment requires coverage, width, tail behavior and quantile loss together, especially for zero-heavy counts. The package supports inspection and independent replication while preserving the recorded final-evaluation provenance.

\balance
\begin{thebibliography}{00}
\bibitem{kalman} R. E. Kalman, ``A new approach to linear filtering and prediction problems,'' \emph{J. Basic Eng.}, vol. 82, no. 1, pp. 35--45, 1960, doi: 10.1115/1.3662552.
\bibitem{lightgbm} G. Ke et al., ``LightGBM: A highly efficient gradient boosting decision tree,'' in \emph{Advances in Neural Information Processing Systems}, vol. 30, 2017, pp. 3146--3154.
\bibitem{deepar} D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski, ``DeepAR: Probabilistic forecasting with autoregressive recurrent networks,'' \emph{Int. J. Forecast.}, vol. 36, no. 3, pp. 1181--1191, 2020, doi: 10.1016/j.ijforecast.2019.07.001.
\bibitem{m5} S. Makridakis, E. Spiliotis, and V. Assimakopoulos, ``M5 accuracy competition: Results, findings, and conclusions,'' \emph{Int. J. Forecast.}, vol. 38, no. 4, pp. 1346--1364, 2022, doi: 10.1016/j.ijforecast.2021.11.013.
\bibitem{gneiting} T. Gneiting and A. E. Raftery, ``Strictly proper scoring rules, prediction, and estimation,'' \emph{J. Amer. Stat. Assoc.}, vol. 102, no. 477, pp. 359--378, 2007, doi: 10.1198/016214506000001437.
\end{thebibliography}
\end{document}
''']
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'main.tex').write_text('\n'.join(line.rstrip() for line in ''.join(parts).splitlines())+'\n')
import shutil
(ROOT/'.build/final').mkdir(parents=True,exist_ok=True)
latex=shutil.which('pdflatex') or '/Library/TeX/texbin/pdflatex'
for _ in range(2):
    subprocess.run([latex,'-interaction=nonstopmode','-halt-on-error','-output-directory',str(ROOT/'.build/final'),str(OUT/'main.tex')],cwd=OUT,check=True,stdout=subprocess.DEVNULL)
import shutil
shutil.copy2(ROOT/'.build/final/main.pdf',OUT/'AML_IEEE_Final_Report.pdf')
print('Built',OUT/'AML_IEEE_Final_Report.pdf')
