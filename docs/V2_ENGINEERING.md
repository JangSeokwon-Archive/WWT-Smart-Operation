# WWT Smart Operation V2

V1 remains on `main`. V2 is developed on `codex/wwt-forecast-v2`; do not merge
or promote an experimental model without reviewing its held-out results.

## Reproduction

Use Python 3.11 in a fresh virtual environment, from the repository root:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-v2.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run wwt_toc_dashboard/streamlit_app.py
```

Direct dependencies are pinned to the versions used for validation. PyArrow 17
is pinned because the latest PyArrow encountered during verification requires
NumPy 2, whereas the recovered V1 model environment uses NumPy 1.26. Platform
differences and unpinned transitive dependencies can still affect reproduction.

Private experiment, with no publication or automatic model promotion:

```sh
.venv/bin/python wwt_toc_dashboard/scripts/evaluate_v2.py \
  --raw /absolute/private/path/raw.csv \
  --output .local/v2 --history-days 180 --folds 3 --estimators 300 --save-models
```

The evaluation produces private `report.json`, per-origin predictions and optional
model bundles under `.local/`, which is ignored by Git. The report records input
SHA256, dependency versions, all fold timestamps, configuration, parameters and
seed. Existing `model_bundle` files are not overwritten.

Explicit local opt-in after reviewing the report:

```sh
WWT_V2_MODEL_DIR="$PWD/.local/v2" .venv/bin/python -m streamlit run wwt_toc_dashboard/streamlit_app.py
```

Unset `WWT_V2_MODEL_DIR` to use the preserved V1 bundles. V1 model files are
unchanged, but V2 inference intentionally corrects timestamp handling, reporting
delay and noncausal preprocessing. This is not a claim of identical V1 output.
Only load joblib bundles from trusted sources; deserialization executes Python.
The summarizer records an acceptance gate: validation and final test MAE must
beat persistence, and nominal 90% intervals must cover at least 85% of test
observations. Inference rejects experimental bundles without a passing gate;
failed interval calibration is not displayed as a validated interval. Passing
this minimal gate still does not establish prospective or causal validity.

## Findings And Recovery Limits

- The main disk cache checked a version without checking the input CSV. Both
  pages now use content hashes of the CSV, model bundles, configs and Python
  pipeline code. In-memory raw-data caches also include the CSV content digest.
- Embedded prediction columns in a CSV had no model version or origin and could
  silently supply an old nonmissing value. They no longer serve as forecasts.
- V1 inference shifted the target into the future, then called a feature builder
  that drops all missing rows. That discarded the newest horizon, resulting in a
  historical origin being displayed as current. Live features now do not require
  future labels, and inference refuses a feature row older than the input origin.
- The recovered missing-value function forward-filled without a limit and filled
  initial gaps with the full-series median. The latter observes future values.
- The recovered TOC cleaner used centered medians/means and whole-run lengths.
  These use observations that were unavailable at the prediction origin.
- V1 label alignment uses `lead_hours + analysis_delay_hours`. The new UI records
  the actual effective horizon and target timestamp. V2 labels use exactly the
  requested +6/+12/+24h; only past-target availability is delayed.
- Recovered residual training predicts its training rows with an already fitted
  baseline. This is not out-of-fold residual fitting. V2 uses strictly purged,
  expanding-window inner folds before fitting residual corrections.
- V1 24h jump limiting can hide an alarm when short-term predictions remain low.
  V2 retains raw predictions and the old guarded value for comparison, does not
  suppress the displayed raw risk, and evaluates the guard separately.
- The AI drawer manufactured intervals from forecast differences. These have
  been removed. Only intervals with recorded calibration and test coverage are
  shown. A probability model has not been validated; exceedance probabilities
  and arbitrary LLM confidence scores are not emitted.
- Simulator clipping thresholds were computed before the chronological split.
  They are now fitted on training controls only, without filtering the holdout
  or deleting high target values. The support bounds restrict optimizer search.

Recovered local sources were found in `wwt-toc-automl/src`. Only the exact feature
builder and LightGBM wrapper are vendored, with no dependency on that external
directory. Their SHA256 values before copying:

| Source | SHA256 |
|---|---|
| `src/features/build.py` | `74c3e80348133fb06c26e55e96f02139f9d0ae986fd80bcbd7f80185d9e95822` |
| `src/models/lgbm.py` | `87af0b04b8691b0a82369cea22d03fc09ace32beac4f261e61edbbaee48bb654` |
| `src/main.py` (inspected, not vendored) | `211064dc524fef21440c4d38ec80553a9d48f3a9543d0898d6e2f3766f0ee648` |

The six packaged models contain column lists and horizon names but lack a source
commit, input hash, training cutoff and preprocessing digest. The recovered source
cannot establish which revision trained each model. Historical bundled reports
cannot establish unbiased prospective accuracy. Do not call them a verified V1
future-data benchmark. Artifact modification time is displayed as file metadata,
not represented as training completion time.

## Data Collection And Time Contract

The current adapter is a batch CSV reader, not a new sensor ingestion service.
Keep immutable source records outside the repository, ideally with instrument ID,
unit, sample timestamp, received/report timestamp and a quality code. Deduplicate
by instrument and sample time; keep revision provenance. Publish hourly snapshots
atomically after recording the source hash. A production collector must provide
the actual per-sensor availability delay and source timezone.

The pipeline requires exact hourly sample timestamps. It rejects off-hour data
instead of guessing an aggregation policy. Duplicate timestamps use the last row
in stable input order. Missing hourly slots become missing observations.
The source timezone is not documented. A Korean timezone assumption is used only
for the dashboard's stale-data age warning; forecasts retain original timestamps.

Target reporting delay defaults to one hour, following the existing config.
Other sensor delays default to zero additional hours, an explicit unverified
assumption. `PipelineConfig.sensor_delays` supports measured per-column delays.
Validate these against laboratory and instrument records before operational use.

## Missingness, Outliers And Features

Nonfinite values and negative non-temperature measurements become missing; pH
outside [0, 14] is invalid. Zero is preserved because the source lacks sensor
maintenance codes that would establish whether it is an invalid reading.
No arbitrary upper TOC cutoff removes potential shock events.

Features use at most three hours of forward fill after applying availability
delays. Initial and long gaps remain NaN. Trees can consume these values; missing
indicators and elapsed hours since the last available observation are features.
Labels remain observed and never forward-filled or smoothed.

All lag/rolling/difference operations are trailing. Prefix-invariance tests ensure
that appending or changing future records cannot change previous feature rows.

Added process features, only when actual source columns exist:

- `FLOW * EQ_TOC` organic load proxy; no kg/h conversion without confirmed units.
- EQ-to-DAF and DAF-to-aeration TOC differences and apparent removal ratios.
  Same-time ratios are observational proxies, not residence-time-matched removal.
- Parallel A/B differences for TOC, DO, MLSS, return, wasting, SRT, SVI and F/M.
- MLSS relative 24h change and 24/72h trend differences.
- Consecutive available low-DO hours, using a configurable 1.0 threshold inherited
  as a heuristic, not an independently established equipment or regulatory limit.
- Shock scores from the previous 24h median and interquartile scale. Shock values
  are retained; statistical unusualness does not cause deletion.

Existing SRT, SVI and F/M columns receive trailing features. They are not recreated
from guessed process volumes. Hydraulic residence features remain unverified
because effective volumes and consistent flow units are not available.

## Evaluation And Parameters

All models are evaluated on identical eligible origins and observed labels.
The last 14 days form the test period; a disjoint 14-day calibration period
precedes it. Expanding-window validation runs only on earlier development data.
Every boundary requires training target time plus reporting delay to be strictly
earlier than the first held-out origin. Purges use timestamps, not row counts.

Comparisons: persistence; seasonal naive at the previous day's target clock time;
original V1 feature/parameter LightGBM refit; temporal-only LightGBM; process-feature
LightGBM; LightGBM with purged OOF residual correction; XGBoost when installed.
The V1 refit uses recovered feature schemas and recorded hyperparameters, with
`n_estimators=300` for this comparison. It is not the frozen V1 model or an exact
reproduction. Those historical feature/parameter choices may themselves reflect
later data and are not a pristine untouched benchmark.

Process LightGBM: 300 trees, learning rate 0.035, 31 leaves, minimum 30 observations
per leaf, L2=2, column fraction 0.85, seed 42, one thread, deterministic mode.
Residual correction: two purged inner folds, at most 100 trees, 7 leaves, minimum
50 observations per leaf. XGBoost: 300 trees, learning rate 0.035, max depth 4,
minimum child weight 10, L2=2, column fraction 0.85, histogram trees, one thread.
No tuning on final test results or deep-learning experiment is performed.

V2 candidate selection uses validation MAE; the report separately checks whether
the selected candidate beats persistence. It does not silently deploy the winner.
Model bundles record exact pipeline version, config digest, feature order,
training end and calibration radius. Schema/version mismatches fail explicitly.

The private report includes MAE/RMSE/Bias, high-load/normal/shock MAE, rise recall,
exceedance recall, false-positive rate and median event warning time. High load is
the training-only 90th percentile of available load proxy; shock is a positive
target change above max(1, training-only 90th percentile). Alarm defaults to the
existing management threshold of 20, not an assertion of a legal limit. Event
onsets lacking a complete pre-event forecast window are excluded as censored.
No-event and no-high-load metrics are null (unverified), never perfect scores.

Intervals use a finite-sample-adjusted absolute residual quantile from calibration
only; empirical coverage is checked on the untouched test. Time dependence and
distribution shift prevent guaranteed future coverage. No calibrated probability
classifier has been fitted; probability calibration remains unverified.

## Simulator And Advisor

The response surface uses contemporaneous observational controls and therefore
does not identify intervention effects. MLSS/temperature/load confounding,
operator reactions, changing delays and combined interventions remain unresolved.
Historical delay coefficients and hardcoded penalties are assumptions; they are
not validated equipment limits. Zero control change is an exact identity.
Missing current state prevents optimizer recommendations. Candidate optimization
is restricted to observed training support as well as the existing heuristic
range, and is labeled a simulation estimate. It is not an operator setpoint.

Ollama remains supported, but may select only evidence codes whose conditions
are established by deterministic inputs. Numerical forecasts, statuses and text
are rendered by code; arbitrary model text, control deltas and confidence scores
are discarded. This deliberately narrows freeform diagnosis. The drawer uses
deterministic status and actual calibrated intervals, and withholds actionable
recommendations while causal effects and equipment limits are unverified.
No control interface, PLC connection or automatic setpoint transmission exists.

## Publication Review

See `DATA_PUBLICATION_REVIEW.md`. Existing raw files and history are preserved.
New private evaluations, models, per-origin predictions and runtime caches must
not be added to the public PR. Tests use generated data, not plant records.
