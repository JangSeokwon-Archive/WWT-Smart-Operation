# V2 Validation Record

Validated on Python 3.11 with the direct dependency versions in
`requirements-v2.txt` and `wwt_toc_dashboard/requirements.txt`.

## Automated Checks

33 tests pass, including:

- Prefix invariance under arbitrary changes to future records.
- Bounded causal fills, no backward fill/full-series median, no imputed labels.
- Exact lead labels, separate target/sensor availability delays.
- Missing hourly slots, duplicate timestamps, rejection of off-hour data.
- Preservation of genuine high TOC shocks and handling of physical invalids.
- Time-based purge on irregular evaluation rows and purged OOF residual fitting.
- Correct missing-event metrics and issue-time-based event warning measurement.
- CSV content changes even with preserved mtime/size, model/config changes,
  legacy cache rejection and changes occurring during inference.
- Live inference retaining the newest row without a future label.
- Model feature-order/config/version checks and failed-deployment rejection.
- LLM numerical/action/confidence injection rejection; current or later alarm
  cannot be downgraded by a lower short-horizon forecast.
- Simulator holdout distribution independence and no-action identity.
- Main and Simulator rendering, diagnosis-button and slider interactions using
  generated data and deterministic mocks.

`git diff --check` passes. The existing model bundle, CSV data and images have
no changes relative to the V1 base. Source parsing/imports and standalone bundled
inference are exercised. Main and Simulator were also run with the existing local
dataset via Streamlit AppTest without uncaught exceptions. Browser pixel-level
layout, a live Ollama server and live prospective model performance have not been
validated. Existing third-party Matplotlib/LightGBM warnings remain non-fatal.

## Private Experiment

Three expanding-window folds, disjoint calibration and final test periods were
run for +6/+12/+24h. Seven model/naive variants were compared. No candidate
passed deployment acceptance. Persistence outperformed all fitted candidates in
the final held-out period, and interval empirical coverage was insufficient.
Do not describe the resulting V2 candidates as a demonstrated production accuracy
improvement or show their failed intervals as calibrated production uncertainty.

No threshold-exceedance events occurred in the final period. Exceedance-event
recall and early warning time are therefore unverified. The synthetic shock test
shows that the old jump guard can suppress a rising alarm, but the historical test
does not establish its real-event miss rate. A new forward test, verified sensor
timestamps and independently approved data are required for promotion.

Private details, per-model hyperparameters, 6/12/24h MAE/RMSE/Bias, regime metrics,
calibration coverage and guard comparisons remain in `.local/v2-experiment/`.
They are not public PR attachments. The selected candidate is determined from
validation MAE; final test scores are reported without selecting a new test winner.

## Changed Areas

| Files | Purpose |
|---|---|
| `core/forecast_pipeline.py` | Shared causal cleaning, availability, temporal/process features, quality |
| `core/forecast_evaluation.py`, `core/forecast_training.py` | Timestamp purge, metrics, calibration, reproducible models, OOF residual |
| `core/automl_infer.py`, `core/legacy_features.py`, `src/models/lgbm.py` and package initializers | Independent inference and recovered serialization/source compatibility |
| `core/prediction_cache.py`, `core/forecast_status.py` | Content invalidation, timestamps, actual horizons, stale/quality/validation status |
| `core/data.py`, `core/llm_advisor.py`, `core/simulator.py` | Parsing, evidence-bound interpretation, observational simulation constraints |
| `streamlit_app.py`, `pages/1_Simulator.py` | Connect cache/status, remove invented intervals and stale fallback forecasts, preserve layout |
| `scripts/evaluate_v2.py`, `scripts/summarize_v2.py` | Private model comparisons, artifacts, reports and deployment acceptance |
| `scripts/make_portfolio_sample.py` | Correct privacy claim and use a private default output |
| `tests/`, `pytest.ini`, `.github/workflows/tests.yml` | Synthetic regression/integration checks and CI |
| `requirements-v2.txt`, `wwt_toc_dashboard/requirements.txt` | Pinned direct runtime/experiment/test dependencies |
| `.gitignore`, runtime `.cache` tracking | Exclude private outputs and untrack prediction/response caches without deleting local files |
| `docs/`, `wwt_toc_dashboard/README.md` | Reproduction, audit findings, privacy and validation limitations |
