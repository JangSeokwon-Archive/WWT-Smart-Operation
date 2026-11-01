# V2.2 Shadow Validation

This branch adds a local, read-only-to-operations validation workflow. It does
not replace an operating forecast model, communicate with a PLC, change a
setpoint or establish prospective performance from historical records.
Process-derived records, timestamps, model hashes, diagnostics and reports stay
under ignored `.local/`; do not attach them to public PRs or CI artifacts.
CI uses a sparse partial checkout of source, configuration and synthetic tests;
raw CSVs, bundled models and private outputs are excluded and their absence is
checked before tests. It does not retrieve external operational data or upload
data artifacts.

## Architecture

```text
sensor/lab arrival -> append-only observation ledger -> available-as-of snapshot
    -> immutable horizon/model forecast -> later arrived truth -> scoring/report
                              |                     |
                      model/config hashes     calibration + drift monitoring
```

`core/shadow_store.py` stores observations, forecasts and explicit invalidations
as separate SQLite events. Each event records a local ingestion/issue timestamp,
a unique key and a hash chained to the previous event. SQLite triggers reject
updates and deletes. Duplicate runs for the same origin return the existing
forecast; a different model/calibration configuration is rejected instead of
overwriting it. Corrections are appended by invalidating the old event and
inserting a new observation with a distinct source id. Scoring at an earlier
time still sees the record valid then.

The SQLite file and its hash chain are not an externally anchored, tamper-proof
audit system. A privileged operator can replace files or remove triggers. Use
restricted filesystem access, backups and an independent append-only audit anchor
before operational reliance. Integrity checks scan ledger history, so long-term
throughput and database growth have not been validated; plan secure rotation and
archival without deleting the original records. Only trusted model bundles may be loaded: joblib
deserialization executes code.

Replay and prospective records use different database modes; mixing modes is
rejected. Replay requires a simulated timestamp. Prospective mode assigns the
local clock on ingestion and issue, rejects injected timestamps, and refuses an
origin if any forecast target is no longer in the future. A real collector and
the host clock must still be trusted. A stated source arrival timestamp is not
proof of actual instrument publication time; separately verify source clocks,
sensor latencies, lab analysis delay and timezone with the data owner.

For each forecast, the ledger stores issue/origin and target times, available
TOC, persistence and any supplied model predictions, model versions and SHA256,
the input snapshot hash and references to immutable observation events,
sample/declared availability/actual ingestion time for each input, quality fields,
calibration method/version and radius. `verify_snapshot` reconstructs the input
snapshot and checks its hash and availability against the original issue. Snapshots
include observations recorded and available by issue and sampled at or before
origin, with at most 168 hours of history. Target samples must be at least the
configured reporting delay old. Its inherited one-hour default is an assumption
requiring site verification. The delay is fixed per ledger; use a new ledger if
the verified policy changes. Model preprocessing and drift reference delay must
match the ledger.
If no sufficiently recent TOC exists, no forecast is issued.

Stored artifact metadata must include `version`, `horizon`, `pipeline_version`,
`config_digest`, `columns`, `training_label_end`, `trained_available_at`,
`artifact_created_at` and `model`. Training label availability and artifact
creation claims must precede the issue time; the shared feature code receives
only the available snapshot, verifies horizon, config and feature ordering,
and stores the artifact hash. These artifact claims need independent provenance
verification; metadata alone cannot prove a model was trained without leakage.
Supply one trusted artifact per model and horizon when available. Existing V1/V2
bundles lack verified training provenance, and the V2.1 candidate was not saved;
their predictions are recorded as unavailable rather than invented or rebuilt.

The scoring view joins only an observation ingested after the immutable forecast,
with the exact target sample timestamp and at least the assumed lab delay. A
future or absent observation is `pending` or `missing`, not a score. Multiple
simultaneous source values are rejected until explicitly resolved. Scored
performance is grouped by horizon, model and model hash over 7 days, 30 days
and all data; interval coverage is shown only where an interval existed at issue.

## Collecting New Data

Install the environment described in `V2_ENGINEERING.md`. Configure the data
owner-approved collector to append UTF-8 JSON Lines. Each line has timezone-aware
`sample_at`, `available_at`, `column`, numeric `value` and `source_id`. Source
availability must reflect when the value was actually made available, not a
timestamp reconstructed from a later export. This synthetic example is only a
format illustration:

```json
{"sample_at":"2030-01-01T00:00:00+09:00","available_at":"2030-01-01T01:00:00+09:00","column":"FINAL_TOC","value":10.0,"source_id":"lab-synthetic"}
```

Run once or keep a collector process running:

```sh
python wwt_toc_dashboard/scripts/shadow.py --db .local/shadow/prospective.sqlite sync --input /private/arrivals.jsonl --calibration rolling
python wwt_toc_dashboard/scripts/shadow.py --db .local/shadow/prospective.sqlite sync --input /private/arrivals.jsonl --calibration rolling --watch-seconds 60
python wwt_toc_dashboard/scripts/shadow.py --db .local/shadow/prospective.sqlite report
```

`sync` ingests only records whose declared availability has passed, issues an
hourly forecast and evaluates already-arrived labels. Repeated input lines are
idempotent. Supplying `--artifact model_name=/private/model.joblib` for each
trusted horizon adds that model to the shadow record; artifacts are not copied
to Git. The command is an opt-in local process, not a system service or an
automatically configured scheduler. Monitor its process health and collector
clock; stopping it leaves gaps that cannot be counted as genuine prior forecasts.

For a correction, use `shadow.py ... invalidate SEQ REASON` and add a new input
line with a distinct source id. Do not overwrite the original. The Model
Validation Streamlit page reads local ledgers without creating or changing them.
With no prospective records it explicitly says prospective data is uncollected;
historical replay is labeled separately.

## Calibration And Drift

Fixed, rolling and adaptive empirical residual intervals require at least 30
already-scored observations for the same horizon, model and model hash. A
fixed radius freezes when it first has enough outcomes; rolling and adaptive use
the preceding 30 days. Adaptive alpha updates use only previously issued
intervals whose truth had arrived. Per-regime coverage is descriptive; no
separate subgroup calibration is fitted on a small sample. These are empirical
intervals, not guaranteed 90% coverage or validated exceedance probabilities.

To evaluate input drift, create a private training reference with a timezone-aware
cutoff and optional **development-derived** per-column PSI thresholds:

```sh
python wwt_toc_dashboard/scripts/shadow.py --db .local/shadow/prospective.sqlite reference --input /private/training.csv --cutoff '2025-01-01T00:00:00+09:00' --timezone Asia/Seoul --thresholds /private/development-thresholds.json
```

The input adapter filters reference rows before cutoff, removes target samples
not yet reportable under the configured delay assumption, and records the reference
file hash. The reference is immutable by this command; use a new directory for
a new version. It computes stable PSI bins, including constant-valued sensors.
Drift is a distribution signal, not proof of process abnormality or performance
degradation. Actual error/bias, coverage and persistence-relative performance
are monitored separately. Without development-validated thresholds or enough
observations, the result is observational or insufficient, never a fabricated
confidence level or automatic control action.
The optional threshold file is a JSON object containing `psi_thresholds` and a
timezone-aware `calibrated_through` strictly before the reference cutoff; its
hash is retained. Performance thresholds likewise need a development hash and
calibration end before the forecasts. These provenance fields are operator
claims that still need independent review.

## Event Rules And Deployment

`model_bundle/configs/shadow_validation.json` records an existing internal TOC
management threshold. It is **not a legal discharge limit**. TOC-rise, inflow
rise, load-change, high-load, MLSS-change and missingness thresholds are unset
until defined from development data and reviewed for the site. The report
calculates pointwise recall/false alarms and per-regime MAE where configured.
Input-change events currently support the configured 24-hour window only; other
window sizes are rejected explicitly. TOC rise is measured over each forecast's
own horizon relative to the TOC available at issuance.
Event-level warning uses the actual forecast issuance time and a fully observed
pre-event window; gaps and unfinished events are censored. Event-level pre/post
MAE requires at least five matched forecast targets and is descriptive only. No event means recall
and warning time are unavailable. Synthetic events only test software behavior.

Deployment requires independently verified sensor and lab timing, a new
prospective comparison against persistence, consistent block-bootstrap results,
adequate empirical coverage **and useful interval width**, sufficient high-load
and shock-event evidence, and explicit operational review. Missing candidate
artifacts, unverified timestamps, insufficient events or uncertain improvement
keep deployment on hold. Coverage sample planning in the private report is a
normal approximation under independence, not a guarantee: overlapping horizons
and serial dependence reduce effective sample size. Required event observation
time cannot be estimated without a trustworthy independent event rate.

## Reproduction And Limitations

```sh
python wwt_toc_dashboard/scripts/validate_v22.py --raw /private/raw.csv --v21-folder /private/v21-reviewed --output .local/v22-validation --timezone Asia/Seoul
python -m pytest -q
```

The raw input hash must match the private V2.1 report. Its stored 24-hour
predictions are used only for 24/48/72/168-hour block-length sensitivity, not
for selecting the most favorable block. Too few non-overlapping blocks yields
an unavailable interval. Calibration uses a separate earlier historical period,
with fixed prespecified methods, and tests only persistence because no V2.1
candidate artifact exists. The timezone is an explicit, unverified replay
assumption. The generated SQLite is **Historical Replay**, not prospective.
The initial report is preserved separately if a reporting bug must be corrected;
corrections do not retune a method or candidate on the evaluation window.
If an older generated report predates the V2.1 comparison table, add the table
without rerunning the experiment using `python wwt_toc_dashboard/scripts/summarize_v22.py .local/v22-validation /private/v21-reviewed`.
