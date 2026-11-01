"""Private retrospective diagnostics and persistence-only calibration replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data import load_raw_csv
from core.forecast_pipeline import hourly_observations
from core.shadow_analysis import bootstrap_sensitivity, coverage_sample_plan
from core.shadow_calibration import interval_for_issue
from core.shadow_store import ShadowStore
from core.prediction_cache import file_digest


def calibration_study(raw, prior, timezone):
    # This fixed window precedes the entire V2.1 development history. No V2.1
    # model is reconstructed: only an available-TOC persistence baseline exists.
    end = pd.Timestamp(prior["design"]["test_end_exclusive"]) - pd.Timedelta(days=prior["design"]["history_days"] + 30)
    start = end - pd.Timedelta(days=30)
    observed = hourly_observations(raw.loc[(raw.index >= start - pd.Timedelta(days=45)) & (raw.index < end)])
    values = observed.FINAL_TOC.where(observed.FINAL_TOC >= 0)
    available = values.shift(1).ffill(limit=3)
    target = values.shift(-24)
    index = observed.index
    issue = pd.DatetimeIndex(index).tz_localize(timezone).tz_convert("UTC") + pd.Timedelta(minutes=1)
    rows = pd.DataFrame({"issue_at": issue, "actual_available_at": issue + pd.Timedelta(hours=25) - pd.Timedelta(minutes=1),
                         "actual": target.to_numpy(), "prediction": available.to_numpy(), "status": "scored", "horizon": 24, "model": "persistence",
                         "interval_low": np.nan, "interval_high": np.nan}, index=index)
    rows = rows[rows.actual.notna() & rows.prediction.notna()].copy()
    eval_index = rows.index[(rows.index >= start) & (rows.index < end - pd.Timedelta(hours=24))]
    boundary = pd.Timestamp(start).tz_localize(timezone).tz_convert("UTC")
    previous = rows[rows.actual_available_at < boundary].copy()
    fixed = interval_for_issue(previous, boundary, 24, "fixed")
    stats = {}
    for method in ("fixed", "rolling", "adaptive"):
        history = rows.copy()
        history["interval_low"] = np.nan
        history["interval_high"] = np.nan
        outputs = []
        for timestamp in eval_index:
            record = history.loc[timestamp]
            interval = fixed if method == "fixed" else interval_for_issue(history, record.issue_at, 24, method)
            if not interval:
                continue
            radius = interval["radius"]
            history.loc[timestamp, "interval_low"] = record.prediction - radius
            history.loc[timestamp, "interval_high"] = record.prediction + radius
            outputs.append({"origin": timestamp, "covered": abs(record.actual - record.prediction) <= radius, "width": 2 * radius})
        o = pd.DataFrame(outputs)
        if o.empty:
            stats[method] = {"n": 0, "status": "insufficient"}
            continue
        o = o.set_index("origin")
        rolling = o.covered.rolling("7D", min_periods=120).mean()
        rolling = rolling.loc[rolling.index >= o.index.min() + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
        # Regime thresholds are calculated once from pre-evaluation samples.
        obs_at = observed.reindex(o.index)
        training = observed.loc[observed.index < start]
        load = obs_at.FLOW * obs_at.EQ_TOC if {"FLOW", "EQ_TOC"}.issubset(obs_at) else pd.Series(np.nan, index=o.index)
        ref_load = training.FLOW * training.EQ_TOC if {"FLOW", "EQ_TOC"}.issubset(training) else pd.Series(dtype=float)
        eq_delta = observed.EQ_TOC.diff(24).reindex(o.index) if "EQ_TOC" in observed else pd.Series(np.nan, index=o.index)
        eq_ref = observed.EQ_TOC.diff(24).loc[training.index].abs().quantile(.9) if "EQ_TOC" in observed else np.nan
        mlss_delta = observed.AERB_MLSS.diff(24).reindex(o.index) if "AERB_MLSS" in observed else pd.Series(np.nan, index=o.index)
        mlss_ref = observed.AERB_MLSS.diff(24).loc[training.index].abs().quantile(.9) if "AERB_MLSS" in observed else np.nan
        missing = obs_at.isna().mean(axis=1)
        level = values.shift(1).reindex(o.index)
        outside = ~level.between(values.loc[training.index].quantile(.01), values.loc[training.index].quantile(.99))
        subsets = {"high_load": load >= ref_load.quantile(.9), "normal_load": load < ref_load.quantile(.9),
                   "inflow_shock": eq_delta.abs() >= eq_ref, "mlss_change": mlss_delta.abs() >= mlss_ref,
                   "missing_inputs": missing > .2, "outside_training_range": outside}
        regimes = {}
        for name, mask in subsets.items():
            part = o.loc[mask.fillna(False)]
            regimes[name] = {"n": len(part), "coverage": float(part.covered.mean()) if len(part) >= 30 else None,
                             "mean_width": float(part.width.mean()) if len(part) >= 30 else None,
                             "status": "descriptive" if len(part) >= 30 else "insufficient"}
        stats[method] = {"n": len(o), "coverage": float(o.covered.mean()), "mean_width": float(o.width.mean()),
                         "worst_7d_coverage": float(rolling.min()) if rolling.notna().any() else None,
                         "regimes": regimes, "status": "historical_replay_not_operationally_validated"}
    return {"model": "persistence", "test_start": str(start), "test_end_exclusive": str(end), "timezone_assumption": timezone,
            "candidate_model": "unavailable; V2.1 artifact was not saved", "methods": stats,
            "note": "Prespecified comparison only. Adaptive/rolling use outcomes when they would have arrived; this is not prospective collection."}


def ledger_replay(raw, study, path, timezone):
    start = pd.Timestamp(study["test_start"])
    # A short sequential replay verifies event ordering and scoring, not model accuracy.
    data = hourly_observations(raw.loc[(raw.index >= start - pd.Timedelta(hours=8)) & (raw.index < start + pd.Timedelta(hours=96))])
    store = ShadowStore(path, "historical_replay")
    try:
        for sample, value in data.FINAL_TOC.items():
            if not np.isfinite(value):
                continue
            sample_utc = sample.tz_localize(timezone).tz_convert("UTC")
            arrival = sample_utc + pd.Timedelta(hours=1)
            store.ingest("FINAL_TOC", sample_utc, arrival, value, source_id="historical-replay", replay_at=arrival)
            if start <= sample < start + pd.Timedelta(hours=48):
                origin = sample_utc + pd.Timedelta(hours=1)
                store.issue(origin, replay_at=arrival + pd.Timedelta(seconds=1))
        asof = data.index[-1].tz_localize(timezone).tz_convert("UTC") + pd.Timedelta(hours=2)
        rows = store.scored(as_of=asof)
        return {"mode": "historical_replay", "forecasts": len(store.events("forecast")),
                "scored": int((rows.status == "scored").sum()) if not rows.empty else 0,
                "pending": int((rows.status == "pending").sum()) if not rows.empty else 0,
                "missing": int((rows.status == "missing").sum()) if not rows.empty else 0,
                "integrity": store.verify(), "candidate": "unavailable"}
    finally:
        store.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--v21-folder", required=True)
    p.add_argument("--output", default=".local/v22-validation")
    p.add_argument("--timezone", required=True, help="Explicit assumption for historical replay; confirm with data owner")
    args = p.parse_args()
    out, prior = Path(args.output), Path(args.v21_folder)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be new and empty")
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    report = json.loads((prior / "report.json").read_text(encoding="utf-8"))
    if file_digest(args.raw) != report["input_sha256"]:
        raise ValueError("Input differs from V2.1 evaluation")
    table = pd.read_csv(prior / "private_predictions_24h.csv", index_col=0, parse_dates=True)
    bootstrap = bootstrap_sensitivity(table.actual, table.persistence, table.fixed_hybrid, table.index)
    raw = load_raw_csv(args.raw)
    study = calibration_study(raw, report, args.timezone)
    replay = ledger_replay(raw, study, out / "historical_replay.sqlite", args.timezone)
    result = {"mode": "historical_backtest_and_replay", "prospective": "not collected", "v21_24h_bootstrap_sensitivity": bootstrap,
              "calibration_study": study, "ledger_replay": replay, "coverage_sample_plan": coverage_sample_plan(),
              "deployment": "withheld: no genuine prospective results, candidate artifact or verified sensor availability"}
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    text = "# V2.2 Private Validation\n\nThis report contains process-derived results; do not publish.\n\nProspective validation: **not collected**. Candidate artifact: **unavailable**. Deployment: **withheld**.\n\n"
    text += "## 24h Bootstrap Sensitivity\n\n```json\n" + json.dumps(bootstrap, indent=2) + "\n```\n\n"
    text += "## Historical Calibration Comparison\n\nPersistence-only replay; timezone is an explicit unverified assumption. No candidate model was evaluated.\n\n```json\n" + json.dumps(study, indent=2) + "\n```\n\n"
    text += "## Ledger Replay And Sample Planning\n\n```json\n" + json.dumps({"replay": replay, "coverage_sample_plan": result["coverage_sample_plan"]}, indent=2) + "\n```\n"
    (out / "RESULTS.md").write_text(text, encoding="utf-8")
    from summarize_v22 import run as summarize
    summarize(out, prior)
    print(json.dumps({"status": "completed", "prospective": "not collected", "deployment": "withheld"}))


if __name__ == "__main__":
    main()
