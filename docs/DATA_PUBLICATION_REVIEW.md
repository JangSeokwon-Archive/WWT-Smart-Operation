# Data Publication Review

Status: **owner approval not established**. File existence in an already-public
repository does not establish authorization or successful anonymization.

| Existing material | Finding | V2 handling |
|---|---|---|
| `wwt_toc_dashboard/data/raw.csv` | Numeric, timestamped process measurements; no proof of publication authorization in repository | Preserved without modification; do not attach records to PR |
| `wwt_toc_dashboard/data/raw_sample.csv` | Sample pipeline retains measurements and only shifts timestamps | Preserved; do not describe as anonymized |
| `wwt_toc_dashboard/data_notes/media/*` | Visually inspected: one code/debugging screenshot and one dashboard screenshot with process/threshold values | Preserved; publication permission remains unverified; no new PR attachments |
| `wwt_toc_dashboard/model_bundle/models/*` | Preexisting serialized models may retain operational information; provenance incomplete | Preserved byte-for-byte; no newly fitted model uploaded |
| `wwt_toc_dashboard/model_bundle/reports/*` | Existing process analysis aggregates are not evidence of permission | Preserved; no new local aggregates attached to PR |
| Runtime prediction/response `.cache` JSON | Data-derived values and potentially stale metadata | Removed from Git tracking on V2 only; runtime files remain local |
| `.local/v2*` | New fitted models, metrics, data hashes and per-origin predictions | Git ignored; private local output only |
| Local `wwt-toc-automl/` outside V2 | Uncommitted source, environments and process data | Left untouched; only two reviewed source files were copied |

Changing dates, removing a site name or limiting rows does not anonymize process
values, operational patterns or instrument identifiers. The sample script now
states this and writes to an ignored `.local/` folder by default. Public examples
should use separately generated synthetic data or a data-owner-approved release.

This change does not delete raw data, rewrite history or claim to certify the
existing public dataset. A full permission determination requires the data owner;
review of publication permissions and provenance records remains outstanding.
The public change contains source, synthetic tests and engineering documentation,
not new plant measurements, new fitted models or private experiment results.
