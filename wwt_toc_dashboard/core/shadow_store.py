"""Append-only, provenance-aware shadow evaluation ledger (local only)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from core.forecast_pipeline import PipelineConfig, build_features

MODES = {"historical_replay", "prospective"}
HORIZONS = (6, 12, 24)


def utc(value) -> datetime:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("Timestamp must include an explicit timezone")
    return value.tz_convert("UTC").to_pydatetime()


def iso(value) -> str:
    return utc(value).isoformat(timespec="microseconds")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class ShadowStore:
    """SQLite triggers preserve rows; hash chaining detects accidental tampering."""

    def __init__(self, path, mode: str, *, create: bool = True, target_delay_hours: int = 1):
        if mode not in MODES:
            raise ValueError("Invalid validation mode")
        self.path = Path(path)
        self.mode = mode
        if not isinstance(target_delay_hours, int) or target_delay_hours < 0:
            raise ValueError("Target reporting delay must be a nonnegative integer")
        self.target_delay_hours = target_delay_hours
        if not create and not self.path.exists():
            raise FileNotFoundError(self.path)
        if create:
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if not create:
            self.db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=30, isolation_level=None)
            self.db.row_factory = sqlite3.Row
            row = self.db.execute("SELECT mode FROM metadata").fetchone()
            if not row or row["mode"] != mode:
                raise ValueError("Cannot mix historical replay and prospective records")
            has_settings = self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='settings'").fetchone()
            saved = self.db.execute("SELECT value FROM settings WHERE key='target_delay_hours'").fetchone() if has_settings else None
            if int(saved["value"] if saved else 1) != target_delay_hours:
                raise ValueError("Ledger target delay differs from requested configuration")
            return
        new_database = not self.path.exists()
        self.db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        if new_database:
            os.chmod(self.path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (mode TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, unique_key TEXT NOT NULL UNIQUE,
          recorded_at TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL,
          event_hash TEXT NOT NULL UNIQUE
        );
        CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        CREATE TRIGGER IF NOT EXISTS metadata_no_update BEFORE UPDATE ON metadata BEGIN SELECT RAISE(ABORT, 'immutable mode'); END;
        CREATE TRIGGER IF NOT EXISTS metadata_no_delete BEFORE DELETE ON metadata BEGIN SELECT RAISE(ABORT, 'immutable mode'); END;
        CREATE TRIGGER IF NOT EXISTS settings_no_update BEFORE UPDATE ON settings BEGIN SELECT RAISE(ABORT, 'immutable settings'); END;
        CREATE TRIGGER IF NOT EXISTS settings_no_delete BEFORE DELETE ON settings BEGIN SELECT RAISE(ABORT, 'immutable settings'); END;
        """)
        row = self.db.execute("SELECT mode FROM metadata").fetchone()
        if row is None:
            self.db.execute("INSERT INTO metadata(mode) VALUES (?)", (mode,))
        elif row["mode"] != mode:
            raise ValueError("Cannot mix historical replay and prospective records")
        saved = self.db.execute("SELECT value FROM settings WHERE key='target_delay_hours'").fetchone()
        if saved is None:
            # Existing ledgers were created using a one-hour delay.
            if self.db.execute("SELECT 1 FROM events LIMIT 1").fetchone() and target_delay_hours != 1:
                raise ValueError("Existing ledger uses the one-hour target delay")
            self.db.execute("INSERT INTO settings(key,value) VALUES ('target_delay_hours',?)", (str(target_delay_hours),))
        elif int(saved["value"]) != target_delay_hours:
            raise ValueError("Ledger target delay differs from requested configuration")

    def close(self):
        self.db.close()

    def _clock(self, replay_at=None):
        if self.mode == "prospective":
            if replay_at is not None:
                raise ValueError("Prospective timestamps are assigned by the local clock")
            return now_utc()
        if replay_at is None:
            raise ValueError("Replay requires a simulated clock")
        return utc(replay_at)

    def _append(self, kind, key, payload, when):
        data, stamp = canonical(payload), iso(when)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT * FROM events WHERE unique_key=?", (key,)).fetchone()
            if old:
                if old["kind"] != kind or old["payload"] != data:
                    raise ValueError("Conflicting duplicate; append an explicit invalidation instead")
                self.db.execute("COMMIT")
                return int(old["seq"])
            previous = self.db.execute("SELECT * FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            if previous and stamp < previous["recorded_at"]:
                raise ValueError("Ledger events must be appended in clock order")
            prev_hash = previous["event_hash"] if previous else "0" * 64
            event_hash = digest({"mode": self.mode, "kind": kind, "key": key, "recorded_at": stamp, "payload": payload, "previous_hash": prev_hash})
            cur = self.db.execute("INSERT INTO events(kind,unique_key,recorded_at,payload,previous_hash,event_hash) VALUES (?,?,?,?,?,?)", (kind, key, stamp, data, prev_hash, event_hash))
            self.db.execute("COMMIT")
            return int(cur.lastrowid)
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def verify(self):
        previous = "0" * 64
        for row in self.db.execute("SELECT * FROM events ORDER BY seq"):
            expected = digest({"mode": self.mode, "kind": row["kind"], "key": row["unique_key"], "recorded_at": row["recorded_at"], "payload": json.loads(row["payload"]), "previous_hash": previous})
            if row["previous_hash"] != previous or row["event_hash"] != expected:
                raise ValueError("Ledger hash chain mismatch")
            previous = row["event_hash"]
        return True

    def events(self, kind=None, as_of=None):
        self.verify()
        rows = self.db.execute("SELECT * FROM events ORDER BY seq").fetchall()
        return [{"seq": r["seq"], "kind": r["kind"], "recorded_at": r["recorded_at"], "event_hash": r["event_hash"], **json.loads(r["payload"])} for r in rows if (not kind or r["kind"] == kind) and (as_of is None or r["recorded_at"] <= iso(as_of))]

    def ingest(self, column, sample_at, available_at, value, *, source_id, replay_at=None):
        when = self._clock(replay_at)
        sample, available = utc(sample_at), utc(available_at)
        if sample > available or available > when:
            raise ValueError("Observation is not yet available at ingestion time")
        if sample.minute or sample.second or sample.microsecond:
            raise ValueError("Observation sample must be hourly")
        if not isinstance(source_id, str) or not source_id.strip() or not isinstance(column, str) or not column.strip() or not math.isfinite(float(value)):
            raise ValueError("A source id, column and finite value are required")
        if column == "FINAL_TOC" and available < sample + timedelta(hours=self.target_delay_hours):
            raise ValueError("TOC observation precedes the configured reporting delay")
        payload = {"column": str(column), "sample_at": iso(sample), "available_at": iso(available), "value": float(value), "source_id": str(source_id)}
        return self._append("observation", f"obs:{source_id}:{column}:{iso(sample)}", payload, when)

    def invalidate(self, event_seq: int, reason: str, *, replay_at=None):
        when = self._clock(replay_at)
        if not reason.strip():
            raise ValueError("Invalidation requires a reason")
        if not self.db.execute("SELECT 1 FROM events WHERE seq=? AND kind IN ('observation','forecast')", (event_seq,)).fetchone():
            raise ValueError("Invalid event reference")
        return self._append("invalidation", f"invalid:{event_seq}", {"target_seq": int(event_seq), "reason": reason.strip()}, when)

    def snapshot(self, origin, issue_time):
        origin, issue = utc(origin), utc(issue_time)
        invalid = {e["target_seq"] for e in self.events("invalidation", issue)}
        items = [e for e in self.events("observation", issue) if e["seq"] not in invalid and origin - timedelta(hours=168) <= utc(e["sample_at"]) <= origin and utc(e["available_at"]) <= issue]
        # Multiple sources for a sample are ambiguous until explicitly invalidated.
        seen = set()
        for e in items:
            key = (e["column"], e["sample_at"])
            if key in seen:
                raise ValueError("Ambiguous observations for the same sensor and sample time")
            seen.add(key)
        return sorted(items, key=lambda e: (e["sample_at"], e["column"], e["seq"]))

    def verify_snapshot(self, forecast_seq: int):
        forecasts = [e for e in self.events("forecast") if e["seq"] == forecast_seq]
        if len(forecasts) != 1:
            raise ValueError("Forecast not found")
        forecast = forecasts[0]
        if "inputs" in forecast:
            inputs = forecast["inputs"]
        else:
            observations = {e["seq"]: e for e in self.events("observation")}
            try:
                inputs = [{k: observations[seq][k] for k in ("seq", "column", "sample_at", "available_at", "recorded_at", "value", "source_id", "event_hash")} for seq in forecast["input_event_sequences"]]
            except KeyError as error:
                raise ValueError("Forecast input observation is missing") from error
        if digest(inputs) != forecast["input_sha256"]:
            raise ValueError("Forecast input snapshot hash mismatch")
        issue, origin = utc(forecast["issue_at"]), utc(forecast["origin_at"])
        if any(utc(e["available_at"]) > issue or utc(e["recorded_at"]) > issue or utc(e["sample_at"]) > origin for e in inputs):
            raise ValueError("Forecast snapshot contains unavailable data")
        return True

    def issue(self, origin, *, replay_at=None, artifacts=None, calibration=None, reference=None):
        issue, origin = self._clock(replay_at), utc(origin)
        if origin > issue or origin.minute or origin.second or origin.microsecond:
            raise ValueError("Invalid forecast origin")
        self.verify()
        existing = self.db.execute("SELECT seq,payload FROM events WHERE unique_key=?", (f"forecast:{iso(origin)}",)).fetchone()
        if existing:
            saved = json.loads(existing["payload"])
            artifact_items = artifacts.items() if isinstance(artifacts, dict) else (artifacts or [])
            requested = sorted((name, hashlib.sha256(Path(path).read_bytes()).hexdigest()) for name, path in artifact_items)
            recorded = sorted((name, meta["sha256"]) for name, entry in saved["models"].items() if name != "persistence" for meta in entry.get("horizons", {}).values())
            reference_sha = hashlib.sha256(Path(reference).read_bytes()).hexdigest() if reference else None
            if requested != recorded or saved.get("calibration_requested") != calibration or saved.get("reference_sha256") != reference_sha:
                raise ValueError("Forecast origin is already immutable; changed configuration cannot overwrite it")
            return int(existing["seq"])
        if origin + timedelta(hours=min(HORIZONS)) <= issue:
            raise ValueError("Forecast target must be in the future when issued")
        snapshot = self.snapshot(origin, issue)
        toc = [e for e in snapshot if e["column"] == "FINAL_TOC" and utc(e["sample_at"]) <= origin - timedelta(hours=self.target_delay_hours)]
        if not toc:
            raise ValueError("No target measurement available at origin")
        latest = max(toc, key=lambda e: e["sample_at"])
        if origin - utc(latest["sample_at"]) > timedelta(hours=self.target_delay_hours + 3):
            raise ValueError("Available TOC exceeds maximum age")
        value = float(latest["value"])
        inputs = [{k: e[k] for k in ("seq", "column", "sample_at", "available_at", "recorded_at", "value", "source_id", "event_hash")} for e in snapshot]
        input_hash = digest(inputs)
        expected = ("FINAL_TOC", "EQ_TOC", "DAF_TOC", "FLOW", "TEMP", "AERB_DO", "AERB_MLSS", "AERB_RET", "AERB_WIT")
        recent = {c: max((e for e in snapshot if e["column"] == c and utc(e["sample_at"]) >= origin - timedelta(hours=4)), key=lambda e: e["sample_at"], default=None) for c in expected}
        context = {c: (recent[c]["value"] if recent[c] else None) for c in expected}
        for c in ("EQ_TOC", "FLOW", "AERB_MLSS"):
            old = [e for e in snapshot if recent[c] and e["column"] == c and utc(e["sample_at"]) == utc(recent[c]["sample_at"]) - timedelta(hours=24)]
            context[c + "_change24"] = context[c] - old[0]["value"] if old and context[c] is not None else None
        context["organic_load_proxy"] = context["EQ_TOC"] * context["FLOW"] if context["EQ_TOC"] is not None and context["FLOW"] is not None else None
        if all(context[c] is not None for c in ("EQ_TOC", "FLOW", "EQ_TOC_change24", "FLOW_change24")):
            context["organic_load_change24"] = context["organic_load_proxy"] - (context["EQ_TOC"] - context["EQ_TOC_change24"]) * (context["FLOW"] - context["FLOW_change24"])
        else:
            context["organic_load_change24"] = None
        reference_sha = None
        if reference:
            ref_path = Path(reference)
            content = ref_path.read_bytes()
            ref = json.loads(content)
            if utc(ref["cutoff"]) >= issue:
                raise ValueError("Drift reference cutoff is not before the forecast")
            if ref.get("target_delay_hours", 1) != self.target_delay_hours:
                raise ValueError("Drift reference target delay differs from the ledger")
            reference_sha = hashlib.sha256(content).hexdigest()
            comparisons = [not limits["min"] <= context[c] <= limits["max"] for c, limits in ref["columns"].items() if c in context and context[c] is not None and "min" in limits and "max" in limits]
            context["outside_reference_fraction"] = sum(comparisons) / len(comparisons) if comparisons else None
        else:
            context["outside_reference_fraction"] = None
        missing = [c for c in expected if recent[c] is None]
        predictions = {str(h): {"persistence": value} for h in HORIZONS}
        model_meta = {"persistence": {"version": "persistence-v1", "sha256": digest({"version": "persistence-v1"})}}
        unavailable = {"v1_v2": "training provenance unavailable", "v21_candidate": "trained candidate artifact not available"}
        if artifacts:
            # Only trusted versioned bundles, trained and published before the
            # forecast, can receive the causally constructed ledger snapshot.
            import joblib
            records = pd.DataFrame(inputs)
            frame = records.pivot(index="sample_at", columns="column", values="value")
            frame.index = pd.to_datetime(frame.index, utc=True)
            frame = frame.reindex(pd.date_range(frame.index.min(), pd.Timestamp(origin), freq="h"))
            pipeline_cfg = PipelineConfig(target_delay_hours=self.target_delay_hours)
            features = build_features(frame, pipeline_cfg)
            artifact_items = artifacts.items() if isinstance(artifacts, dict) else artifacts
            for name, path in artifact_items:
                p = Path(path)
                artifact_bytes = p.read_bytes()
                artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
                bundle = joblib.load(io.BytesIO(artifact_bytes))
                if bundle.get("pipeline_version") != "wwt-causal-v2.1" or utc(bundle["trained_available_at"]) >= issue or utc(bundle["artifact_created_at"]) > issue or utc(bundle["training_label_end"]) + timedelta(hours=self.target_delay_hours) >= issue:
                    raise ValueError("Model provenance, pipeline or training availability is invalid")
                if bundle.get("config_digest") != pipeline_cfg.digest():
                    raise ValueError("Model preprocessing configuration mismatch")
                horizon = int(bundle["horizon"])
                if horizon not in HORIZONS or list(bundle["columns"]) != list(dict.fromkeys(bundle["columns"])):
                    raise ValueError("Invalid artifact horizon or feature schema")
                if name in predictions[str(horizon)]:
                    raise ValueError("Duplicate model/horizon artifact")
                model = bundle["model"]
                if hasattr(model, "columns") and list(model.columns) != list(bundle["columns"]):
                    raise ValueError("Model and bundle feature order mismatch")
                estimator = getattr(model, "model", model)
                schema = getattr(estimator, "feature_names_in_", getattr(estimator, "feature_name_", None))
                if schema is not None and list(schema) != list(bundle["columns"]):
                    raise ValueError("Trained model feature order mismatch")
                missing_features = set(bundle["columns"]) - set(features.columns)
                if missing_features or features.index[-1] != pd.Timestamp(origin):
                    raise ValueError("Latest artifact features unavailable")
                prediction = float(bundle["model"].predict(features.loc[:, bundle["columns"]].tail(1))[0])
                if not math.isfinite(prediction):
                    raise ValueError("Nonfinite artifact forecast")
                predictions[str(horizon)][name] = prediction
                meta = model_meta.setdefault(name, {"version": bundle["version"], "sha256": artifact_hash, "horizons": {}, "provenance": "artifact claims require external verification"})
                meta["horizons"][str(horizon)] = {"version": bundle["version"], "sha256": artifact_hash,
                    "training_label_end": bundle["training_label_end"], "trained_available_at": bundle["trained_available_at"],
                    "artifact_created_at": bundle["artifact_created_at"], "pipeline_version": bundle["pipeline_version"],
                    "config_digest": bundle["config_digest"], "feature_schema_sha256": digest(bundle["columns"])}
                unavailable.pop(name, None)
        cal = {}
        if calibration:
            from core.shadow_calibration import interval_for_issue
            if calibration not in {"fixed", "rolling", "adaptive"}:
                raise ValueError("Calibration must be computed from prior scored ledger events")
            scored = self.scored(as_of=issue)
            invalid = {e["target_seq"] for e in self.events("invalidation", issue)}
            for h in HORIZONS:
                for model in predictions[str(h)]:
                    meta = model_meta[model]
                    model_sha = meta.get("horizons", {}).get(str(h), {}).get("sha256", meta["sha256"])
                    prior = [e for e in self.events("forecast", issue) if e["seq"] not in invalid] if calibration == "fixed" else []
                    frozen = next((x for e in prior for x in [e["calibration"].get(str(h), {}).get(model)] if x and x.get("method") == "fixed" and x.get("model_sha256") == model_sha), None)
                    interval = frozen or interval_for_issue(scored, issue, h, calibration, model=model, model_sha=model_sha)
                    if interval:
                        interval = dict(interval, model_sha256=model_sha)
                        cal.setdefault(str(h), {})[model] = interval
        payload = {"origin_at": iso(origin), "issue_at": iso(issue), "targets": {str(h): iso(origin + timedelta(hours=h)) for h in HORIZONS},
                   "latest_toc": value, "latest_toc_sample_at": latest["sample_at"], "latest_toc_available_at": latest["available_at"],
                   "predictions": predictions, "models": model_meta, "unavailable_models": unavailable,
                   "input_sha256": input_hash, "input_event_sequences": [e["seq"] for e in snapshot],
                   "latest_input_availability": {c: {k: recent[c][k] for k in ("sample_at", "available_at", "recorded_at")} for c in expected if recent[c]}, "context": context,
                   "quality": {"n_observations": len(inputs), "missing_columns": missing, "missing_fraction": len(missing) / len(expected), "sensor_delays": "unverified", "target_delay_hours": self.target_delay_hours},
                   "calibration_requested": calibration, "calibration": cal, "reference_sha256": reference_sha}
        return self._append("forecast", f"forecast:{iso(origin)}", payload, issue)

    def scored(self, *, as_of=None):
        when = utc(as_of) if as_of is not None else now_utc()
        if self.mode == "prospective" and when > now_utc():
            raise ValueError("Cannot evaluate prospective data in the future")
        observations = self.events("observation", when)
        invalid = {e["target_seq"] for e in self.events("invalidation", when)}
        forecasts = [e for e in self.events("forecast", when) if e["seq"] not in invalid]
        truth = {}
        for e in observations:
            if e["seq"] in invalid or e["column"] != "FINAL_TOC" or utc(e["available_at"]) > when:
                continue
            if e["sample_at"] in truth:
                raise ValueError("Ambiguous actual observation")
            truth[e["sample_at"]] = e
        rows = []
        for f in forecasts:
            for h in HORIZONS:
                target = f["targets"][str(h)]
                obs = truth.get(target)
                # The actual must be a later ledger event, not preexisting truth.
                if obs and (obs["seq"] <= f["seq"] or utc(obs["available_at"]) < utc(target) + timedelta(hours=self.target_delay_hours)):
                    obs = None
                for model, prediction in f["predictions"][str(h)].items():
                    c = f["calibration"].get(str(h), {}).get(model, {})
                    model_meta = f["models"][model]
                    model_sha = model_meta.get("horizons", {}).get(str(h), {}).get("sha256", model_meta["sha256"])
                    rows.append({"forecast_seq": f["seq"], "mode": self.mode, "origin_at": f["origin_at"], "issue_at": f["issue_at"], "target_at": target,
                                 "horizon": h, "model": model, "prediction": prediction, "persistence": f["predictions"][str(h)]["persistence"],
                                 "actual": obs["value"] if obs else None, "actual_available_at": obs["available_at"] if obs else None,
                                 "interval_low": prediction - float(c["radius"]) if c else None, "interval_high": prediction + float(c["radius"]) if c else None,
                                 "status": "scored" if obs else ("pending" if utc(target) + timedelta(hours=self.target_delay_hours) > when else "missing"),
                                 "input_sha256": f["input_sha256"], "model_sha256": model_sha,
                                 "missing_fraction": f["quality"]["missing_fraction"], **f["context"]})
        return pd.DataFrame(rows)
