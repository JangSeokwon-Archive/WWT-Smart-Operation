"""Local ingestion, forecast and scoring commands; no plant control interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.shadow_store import ShadowStore, now_utc, utc
from core.shadow_monitor import summarize_scores, performance_status, event_windows, fit_reference, input_drift

APP = Path(__file__).resolve().parents[1]


def config():
    value = json.loads((APP / "model_bundle/configs/shadow_validation.json").read_text(encoding="utf-8"))
    if value.get("event_window_hours") != 24:
        raise ValueError("Only 24-hour input-change event windows are implemented")
    if type(value.get("target_delay_hours_assumption")) is not int or value["target_delay_hours_assumption"] < 0:
        raise ValueError("Target delay must be a nonnegative integer")
    return value


def sync(store, file, *, replay_at=None, origin=None, calibration=None, artifacts=None):
    now = store._clock(replay_at)
    with Path(file).open(encoding="utf-8") as stream:
        observations = [json.loads(line) for line in stream if line.strip()]
    # Future observations can be present in the file but are never ingested
    # into a prediction snapshot until their declared arrival time.
    for observation in sorted(observations, key=lambda x: x["available_at"]):
        if utc(observation["available_at"]) > now:
            continue
        store.ingest(observation["column"], observation["sample_at"], observation["available_at"], observation["value"],
                     source_id=observation["source_id"], replay_at=replay_at)
    if origin is None:
        origin = pd.Timestamp(now).floor("h")
    reference = Path(store.path).parent / "reference.json"
    seq = store.issue(origin, replay_at=replay_at, calibration=calibration, artifacts=artifacts, reference=reference if reference.exists() else None)
    return seq


def report(store, *, as_of=None):
    when = utc(as_of) if as_of is not None else now_utc()
    cfg = config()
    rows = store.scored(as_of=when)
    summary = summarize_scores(rows, when, alarm=cfg.get("internal_toc_alarm"), rise_delta=cfg.get("toc_rise_delta"), min_samples=cfg["min_scored_per_horizon"], regime_config=cfg)
    summary["mode"] = store.mode
    summary["performance_status"] = performance_status(summary["groups"], cfg.get("performance_thresholds"))
    invalid = {e["target_seq"] for e in store.events("invalidation", when)}
    forecasts = [e for e in store.events("forecast", when) if e["seq"] not in invalid]
    observations = [e for e in store.events("observation", when) if e["seq"] not in invalid]
    summary["latest_issue_at"] = forecasts[-1]["issue_at"] if forecasts else None
    summary["latest_data_at"] = max((e["sample_at"] for e in observations), default=None)
    summary["latest_received_at"] = max((e["recorded_at"] for e in observations), default=None)
    summary["latest_quality"] = forecasts[-1]["quality"] if forecasts else None
    summary["event_summary"] = {}
    ref_path = Path(store.path).parent / "reference.json"
    if ref_path.exists() and observations:
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
        if utc(reference["cutoff"]) > when:
            raise ValueError("Drift reference is newer than the report time")
        recent = pd.DataFrame(observations)
        recent["sample_at"] = pd.to_datetime(recent.sample_at, utc=True)
        recent = recent[recent.sample_at >= pd.Timestamp(when) - pd.Timedelta(days=7)]
        if recent.duplicated(["sample_at", "column"]).any():
            raise ValueError("Ambiguous observations in drift window")
        recent = recent.pivot(index="sample_at", columns="column", values="value")
        summary["input_drift"] = input_drift(reference, recent)
    else:
        summary["input_drift"] = {"status": "reference_unavailable"}
    if cfg.get("internal_toc_alarm") is not None:
        models = sorted({name for f in forecasts for h in f["predictions"].values() for name in h})
        for h in (6, 12, 24):
            for model in models:
                summary["event_summary"][f"{h}:{model}"] = event_windows(observations, forecasts, alarm=cfg["internal_toc_alarm"], horizon=h, model=model)
    summary["forecast_scope"] = "genuine prospective shadow records" if store.mode == "prospective" else "historical replay; not prospective performance"
    summary["deployment"] = "withheld; independent prospective verification and operational review required"
    summary["publication"] = "private operational records; do not publish"
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=".local/shadow/prospective.sqlite")
    p.add_argument("--mode", choices=("historical_replay", "prospective"), default="prospective")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("sync")
    run.add_argument("--input", required=True, help="JSONL with sample_at, available_at, column, value, source_id")
    run.add_argument("--origin", help="Timezone-aware origin; defaults to current UTC hour")
    run.add_argument("--calibration", choices=("fixed", "rolling", "adaptive"))
    run.add_argument("--artifact", action="append", default=[], help="Trusted name=local-joblib-path; metadata required")
    run.add_argument("--replay-at", help="Simulated clock; replay mode only")
    run.add_argument("--watch-seconds", type=int, default=0, help="Poll continuously for new arrivals (0 runs once)")
    view = sub.add_parser("report")
    view.add_argument("--as-of", help="Replay evaluation clock")
    bad = sub.add_parser("invalidate")
    bad.add_argument("seq", type=int)
    bad.add_argument("reason")
    bad.add_argument("--replay-at")
    reference = sub.add_parser("reference")
    reference.add_argument("--input", required=True, help="Private training CSV")
    reference.add_argument("--cutoff", required=True, help="Timezone-aware training cutoff")
    reference.add_argument("--timezone", required=True, help="Source timezone for naive timestamps")
    reference.add_argument("--thresholds", help="JSON file with development-derived per-column PSI thresholds")
    args = p.parse_args()
    if args.command == "reference":
        from core.data import load_raw_csv
        from core.prediction_cache import file_digest
        frame = load_raw_csv(args.input)
        frame.index = frame.index.tz_localize(args.timezone) if frame.index.tz is None else frame.index.tz_convert(args.timezone)
        threshold_doc = json.loads(Path(args.thresholds).read_text()) if args.thresholds else {}
        thresholds = threshold_doc.get("psi_thresholds", {})
        if args.thresholds and (not isinstance(thresholds, dict) or not threshold_doc.get("calibrated_through")):
            raise ValueError("Threshold file requires psi_thresholds and calibrated_through")
        ref = fit_reference(frame, cutoff=args.cutoff, thresholds=thresholds, thresholds_fit_before=threshold_doc.get("calibrated_through"), target_delay_hours=int(config()["target_delay_hours_assumption"]))
        ref["thresholds_sha256"] = file_digest(args.thresholds) if args.thresholds else None
        ref["source_sha256"] = file_digest(args.input)
        ref["provenance"] = "historical CSV sample timestamps; actual arrival times and sensor delays require verification"
        target = Path(args.db).parent / "reference.json"
        if target.exists():
            raise ValueError("Reference exists; preserve it and choose a new shadow directory")
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(ref, stream, indent=2, allow_nan=False)
        print("Private drift reference created")
        return
    if args.command == "report" and not Path(args.db).exists():
        print(json.dumps({"mode": args.mode, "status": "insufficient", "prospective": "not collected", "deployment": "withheld"}))
        return
    store = ShadowStore(args.db, args.mode, create=args.command != "report", target_delay_hours=int(config()["target_delay_hours_assumption"]))
    try:
        if args.command == "report":
            print(json.dumps(report(store, as_of=args.as_of), ensure_ascii=False, indent=2, allow_nan=False))
        elif args.command == "invalidate":
            print(store.invalidate(args.seq, args.reason, replay_at=args.replay_at))
        else:
            if args.mode == "historical_replay" and args.watch_seconds:
                p.error("Continuous watch is supported only for prospective mode")
            if args.watch_seconds < 0:
                p.error("Watch interval must be nonnegative")
            artifacts = [tuple(x.split("=", 1)) for x in args.artifact]
            while True:
                try:
                    seq = sync(store, args.input, replay_at=args.replay_at, origin=args.origin, calibration=args.calibration, artifacts=artifacts)
                    print(json.dumps({"forecast_seq": seq, "summary": report(store, as_of=args.replay_at)}, ensure_ascii=False, allow_nan=False), flush=True)
                except (ValueError, OSError) as error:
                    if not args.watch_seconds:
                        raise
                    print(json.dumps({"status": "waiting", "reason": str(error)}, ensure_ascii=False), flush=True)
                if not args.watch_seconds:
                    break
                time.sleep(max(1, args.watch_seconds))
    finally:
        store.close()


if __name__ == "__main__":
    main()
