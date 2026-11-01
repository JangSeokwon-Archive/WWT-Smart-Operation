# V2.1 Model Improvement

This branch investigates why V2 fitted models did not beat persistence. It does
not change the V1 or V2 branches, original model bundles, live dashboard model
choice, PLC settings or process data. Detailed plant-derived metrics, model
predictions and diagnostic values remain in ignored `.local/` files.

## Reproduction

From the repository root, using the Python 3.11 environment described in
`docs/V2_ENGINEERING.md`:

```sh
python -m pip install -r requirements-v2.txt
python wwt_toc_dashboard/scripts/evaluate_v21.py --raw /private/path/raw.csv --prior-report /private/path/v2-experiment/report.json --output .local/v21-final
python wwt_toc_dashboard/scripts/summarize_v21.py .local/v21-final --raw /private/path/raw.csv
python -m pytest -q
```

The output directory must be new and empty. The input SHA256 must match the
prior V2 report, ensuring the previous 180-day window can be established from
the same dataset. Input SHA256, package versions,
date boundaries, folds, model metrics, calibration and diagnostics are recorded.
The input is rehashed before completion. Do not publish private output files.

## Prior Evidence And Hypotheses

The prior V2 test favored persistence for every horizon; fitted models exhibited
negative bias and much worse than nominal prediction-interval coverage. No
threshold-exceedance events were present in that test. These observations do
not establish why the models failed or whether persistence will always win.

The private V2.1 report distinguishes measured evidence and uncertain mechanisms:

1. It compares train and test TOC level and absolute changes, rather than
   assuming the previous test was unusually stable.
2. It compares train/test level distributions and residual bias as indicators
   of potential distribution shift. These are associations, not process causes.
3. Full and training-fold-selected features are evaluated under identical
   time windows. Selection/fold overlap and gain, permutation and native
   LightGBM TreeSHAP diagnostics assess redundancy and stability; causality
   or overfitting cannot be inferred from importance alone.
4. Training-only correlations compare past EQ, DAF, flow, DO, MLSS and SRT
   values at explicit lags. Target delay is applied separately. Reporting
   times other than the target remain assumed and require site records.
5. Rising/shock metrics compare delta, direct and persistence forecasts. A
   low event count limits the ability to infer performance on rare shocks.
6. OOF residual versus direct-model errors are compared on the same rows;
   neither an observational association nor one holdout proves a lasting effect.
7. The report records event counts and avoids treating missing-event recall as
   zero or perfect. Synthetic data is used only in regression tests.

## Experiment And Separation

V2's already-inspected final period is not used for optimization or retesting.
The new historical final test is chosen from timestamps alone, before looking
at its outcomes: it ends 30 days before the beginning of V2's 180-day window
and spans the preceding 30 days. Its targets must also lie inside that window.
This provides an additional historical holdout outside the previous experiment;
it is not a newly collected future period, and V1 training provenance is unknown.

Past development uses three expanding-window folds with horizon plus reporting
delay embargo. The following 30-day selection period chooses the model, followed
by a disjoint 14-day calibration period and final test. Labels from each earlier
partition must have been reportable strictly before the next partition starts.
The last holdout is evaluated once after the selection winner is fixed.
During this run, the seasonal benchmark's daily-clock alignment was corrected
after evaluation had started. Its deterministic predictions and selection score
were recomputed on the same stored rows without retraining or tuning any model;
the original private output remains available. Future runs use the corrected
benchmark directly. This correction is part of the audit trail, not a new model
selection using final-test scores.
Event onset after a missing hour, or without a complete contiguous preceding
warning window, is now censored instead of counted as a confirmed new event.

The following use the same origins, untouched labels and train/calibration/test
intervals: persistence, seasonal naive, V2 LightGBM, full and selected-feature
delta LightGBM, OOF residual LightGBM, XGBoost, and fixed/regime/learned-gate
persistence hybrids. The change target is `future_toc - available_toc`; inference
adds the prediction to the same availability-delayed baseline.
Seasonal naive uses the most recent available TOC at the target's daily clock
time. For a 24-hour horizon with a one-hour reporting delay, this is the value
from 24 hours before the origin, because the origin's own sample is not yet
available. If the seasonal observation is missing, persistence is used.

Fixed/regime weights and the ridge gate are fitted exclusively from purged OOF
development predictions. The load regime threshold is the development 90th
percentile. Hybrid predictions fall back to persistence when the current target
is missing, the load proxy is absent, or more than 20% of inputs are missing;
this 20% is a fixed explicit rule, not a validated equipment limit. The separate
selection period assesses all models and selects the lowest MAE, including the
naive baselines. Time-dependent feature selection is refit on each training
fold and final training period only. Gain-ranked variables are deduplicated using
training-only correlations; the threshold is 0.98. Inner-development held-out
permutation and TreeSHAP are diagnostics, never used to choose on the final test.

## Metrics And Decision

The private report includes the requested horizon/model validation and test MAE,
RMSE, bias and persistence-relative improvement table, plus normal/high-load,
rising/shock MAE, event detection and false alarms, lead time, interval coverage
and width, and paired 24-hour contiguous-block bootstrap confidence intervals.
Thresholds and shock definitions come from development data or existing settings.
Empty event samples are unavailable. Data shifts and rare events may invalidate
exchangeability; nominal 90% intervals have only empirical coverage.

The experiment never auto-promotes a model, including one that happens to exceed
the 5% MAE target. Promotion requires new prospective data, confirmed sensor and
laboratory reporting times, no material high-load/shock deterioration, meaningful
event coverage and calibrated intervals. A broad or weak interval must not be
presented as operationally reliable merely because empirical coverage is high.
Missing records and unverified causal effects cannot justify automatic control.

| Horizon | Models | Validation MAE | Test MAE | RMSE | Persistence comparison |
|---|---|---|---|---|---|
| 6h | All candidates | Private report | Private report | Private report | Private report |
| 12h | All candidates | Private report | Private report | Private report | Private report |
| 24h | All candidates | Private report | Private report | Private report | Private report |
