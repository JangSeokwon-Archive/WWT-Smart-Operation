import json
import sqlite3
import stat
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

from core.forecast_pipeline import PipelineConfig
from core.shadow_store import ShadowStore
from core.shadow_calibration import interval_for_issue
from core.shadow_monitor import fit_reference, input_drift, summarize_scores, event_windows, performance_status
from core.shadow_analysis import bootstrap_sensitivity, coverage_sample_plan


@pytest.fixture
def ledger(tmp_path):
    db = ShadowStore(tmp_path / "replay.sqlite", "historical_replay")
    yield db
    db.close()


def seed_forecast(db):
    db.ingest("FINAL_TOC", "2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", 10, source_id="lab", replay_at="2025-01-01T01:00:00Z")
    return db.issue("2025-01-01T01:00:00Z", replay_at="2025-01-01T01:01:00Z")


def test_immutable_forecast_and_no_score_before_arrival(ledger):
    seq = seed_forecast(ledger)
    snapshot = ledger.events("forecast")[0]
    assert snapshot["targets"]["6"].startswith("2025-01-01T07:00:00")
    assert all(snapshot["models"][m]["sha256"] for m in snapshot["models"])
    assert len(snapshot["input_sha256"]) == 64
    assert ledger.verify_snapshot(seq)
    assert snapshot["input_event_sequences"]
    assert ledger.scored(as_of="2025-01-01T07:30:00Z").status.eq("pending").all()
    with pytest.raises(ValueError, match="reporting delay"):
        ledger.ingest("FINAL_TOC", "2025-01-01T07:00:00Z", "2025-01-01T07:30:00Z", 11, source_id="lab", replay_at="2025-01-01T07:30:00Z")
    ledger.ingest("FINAL_TOC", "2025-01-01T07:00:00Z", "2025-01-01T08:00:00Z", 11, source_id="lab", replay_at="2025-01-01T08:00:00Z")
    rows = ledger.scored(as_of="2025-01-01T08:00:00Z")
    assert rows.loc[rows.horizon == 6, "actual"].iloc[0] == 11
    assert rows.loc[rows.horizon == 12, "status"].iloc[0] == "pending"
    assert ledger.events("forecast")[0] == snapshot
    assert ledger.issue("2025-01-01T01:00:00Z", replay_at="2025-01-01T08:00:00Z") == seq
    with pytest.raises(ValueError, match="immutable"):
        ledger.issue("2025-01-01T01:00:00Z", replay_at="2025-01-01T08:00:00Z", calibration="rolling")
    assert len(ledger.events("forecast")) == 1


def test_missing_and_correction_are_separate_events(ledger):
    seed_forecast(ledger)
    assert ledger.scored(as_of="2025-01-02T12:00:00Z").status.eq("missing").all()
    actual = ledger.ingest("FINAL_TOC", "2025-01-01T07:00:00Z", "2025-01-01T08:00:00Z", 11, source_id="lab", replay_at="2025-01-02T12:00:00Z")
    ledger.invalidate(actual, "laboratory correction", replay_at="2025-01-02T12:01:00Z")
    ledger.ingest("FINAL_TOC", "2025-01-01T07:00:00Z", "2025-01-02T12:02:00Z", 12, source_id="corrected", replay_at="2025-01-02T12:02:00Z")
    assert ledger.scored(as_of="2025-01-02T12:00:00Z").loc[lambda x: x.horizon == 6, "actual"].iloc[0] == 11
    assert ledger.scored(as_of="2025-01-02T12:02:00Z").loc[lambda x: x.horizon == 6, "actual"].iloc[0] == 12
    assert len(ledger.events("observation")) == 3
    assert ledger.verify()


def test_database_rejects_mutations_and_mode_mixing(ledger):
    seed_forecast(ledger)
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger.db.execute("UPDATE events SET payload='{}'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger.db.execute("DELETE FROM events")
    with pytest.raises(ValueError, match="mix"):
        ShadowStore(ledger.path, "prospective")


def test_replay_clock_and_availability_guard(ledger, tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        ledger.ingest("FLOW", "2025-01-01 00:00", "2025-01-01 01:00", 1, source_id="sensor", replay_at="2025-01-01T01:00:00Z")
    with pytest.raises(ValueError, match="not yet available"):
        ledger.ingest("FLOW", "2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z", 1, source_id="sensor", replay_at="2025-01-01T01:00:00Z")
    live = ShadowStore(tmp_path / "live.sqlite", "prospective")
    try:
        with pytest.raises(ValueError, match="local clock"):
            live.issue("2025-01-01T01:00:00Z", replay_at="2025-01-01T01:00:00Z")
        assert live.scored().empty
    finally:
        live.close()


def test_real_clock_forecast_is_future_and_unscored(tmp_path):
    now = pd.Timestamp.now(tz="UTC")
    origin = now.floor("h")
    live = ShadowStore(tmp_path / "prospective" / "live.sqlite", "prospective")
    try:
        live.ingest("FINAL_TOC", origin - pd.Timedelta(hours=1), origin, 10, source_id="lab")
        live.issue(origin)
        forecast = live.events("forecast")[0]
        assert all(pd.Timestamp(target) > pd.Timestamp(forecast["issue_at"]) for target in forecast["targets"].values())
        assert live.scored().status.eq("pending").all()
    finally:
        live.close()


def test_configured_delay_is_immutable_and_used_for_scoring(tmp_path):
    path = tmp_path / "two-hour.sqlite"
    db = ShadowStore(path, "historical_replay", target_delay_hours=2)
    db.ingest("FINAL_TOC", "2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z", 10, source_id="lab", replay_at="2025-01-01T02:00:00Z")
    db.issue("2025-01-01T02:00:00Z", replay_at="2025-01-01T02:01:00Z")
    assert db.scored(as_of="2025-01-01T09:00:00Z").loc[lambda x: x.horizon == 6, "status"].iloc[0] == "pending"
    db.ingest("FINAL_TOC", "2025-01-01T08:00:00Z", "2025-01-01T10:00:00Z", 11, source_id="lab", replay_at="2025-01-01T10:00:00Z")
    assert db.scored(as_of="2025-01-01T10:00:00Z").loc[lambda x: x.horizon == 6, "status"].iloc[0] == "scored"
    db.close()
    with pytest.raises(ValueError, match="delay differs"):
        ShadowStore(path, "historical_replay", target_delay_hours=1, create=False)


class DummyPredictor:
    def predict(self, X):
        return X.TOC_AVAILABLE.to_numpy() + 1


def test_trusted_artifact_hash_and_future_training_guard(ledger, tmp_path):
    seed_forecast(ledger)
    path = tmp_path / "candidate.joblib"
    bundle = {"pipeline_version": "wwt-causal-v2.1", "config_digest": PipelineConfig().digest(), "version": "test",
              "trained_available_at": "2025-01-01T01:30:00Z", "artifact_created_at": "2025-01-01T01:40:00Z",
              "training_label_end": "2024-12-31T22:00:00Z", "horizon": 6, "columns": ["TOC_AVAILABLE"], "model": DummyPredictor()}
    joblib.dump(bundle, path)
    ledger.ingest("FINAL_TOC", "2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z", 11, source_id="lab", replay_at="2025-01-01T02:00:00Z")
    ledger.issue("2025-01-01T02:00:00Z", replay_at="2025-01-01T02:01:00Z", artifacts={"candidate": path})
    forecast = ledger.events("forecast")[-1]
    assert forecast["predictions"]["6"]["candidate"] == 12
    assert len(forecast["models"]["candidate"]["sha256"]) == 64
    bundle["trained_available_at"] = "2025-01-02T00:00:00Z"
    joblib.dump(bundle, path)
    ledger.ingest("FINAL_TOC", "2025-01-01T02:00:00Z", "2025-01-01T03:00:00Z", 12, source_id="lab", replay_at="2025-01-01T03:00:00Z")
    with pytest.raises(ValueError, match="provenance"):
        ledger.issue("2025-01-01T03:00:00Z", replay_at="2025-01-01T03:01:00Z", artifacts={"candidate": path})


def test_calibration_never_uses_future_labels():
    issue = pd.Timestamp("2025-02-01T00:00:00Z")
    dates = pd.date_range("2025-01-01", periods=60, freq="12h", tz="UTC")
    rows = pd.DataFrame({"horizon": 24, "model": "persistence", "status": "scored", "actual_available_at": dates,
                         "issue_at": dates - pd.Timedelta(days=2), "actual": 11., "prediction": 10.,
                         "interval_low": np.nan, "interval_high": np.nan})
    future = pd.DataFrame({"horizon": [24], "model": ["persistence"], "status": ["scored"], "actual_available_at": [issue + pd.Timedelta(hours=1)],
                           "issue_at": [issue - pd.Timedelta(days=1)], "actual": [100000.], "prediction": [10.], "interval_low": [np.nan], "interval_high": [np.nan]})
    for method in ("fixed", "rolling", "adaptive"):
        first = interval_for_issue(rows, issue, 24, method)
        second = interval_for_issue(pd.concat([rows, future]), issue, 24, method)
        assert first == second
        assert first["radius"] == 1
    assert interval_for_issue(rows.iloc[:3], issue, 24, "rolling") is None


def test_calibration_does_not_mix_model_versions():
    issue = pd.Timestamp("2025-02-01T00:00:00Z")
    dates = pd.date_range("2025-01-10", periods=40, freq="h", tz="UTC")
    old = pd.DataFrame({"horizon": 24, "model": "candidate", "model_sha256": "old", "status": "scored", "actual_available_at": dates,
                        "issue_at": dates - pd.Timedelta(days=2), "actual": 100., "prediction": 0., "interval_low": np.nan, "interval_high": np.nan})
    new = old.copy()
    new["model_sha256"] = "new"
    new["actual"] = 1.
    radius = interval_for_issue(pd.concat([old, new]), issue, 24, "rolling", model="candidate", model_sha="new")
    assert radius["radius"] == 1
    assert radius["n"] == 40


def test_fixed_calibration_freezes_after_arrived_labels(ledger):
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    for hour in range(1, 46):
        origin = start + pd.Timedelta(hours=hour)
        sample = origin - pd.Timedelta(hours=1)
        ledger.ingest("FINAL_TOC", sample, origin, 10 + hour / 10, source_id="lab", replay_at=origin)
        ledger.issue(origin, replay_at=origin + pd.Timedelta(minutes=1), calibration="fixed")
    forecasts = ledger.events("forecast")
    calibrated = [f["calibration"]["6"]["persistence"] for f in forecasts if "6" in f["calibration"]]
    assert len(calibrated) > 1
    assert calibrated[0] == calibrated[-1]
    assert calibrated[0]["n"] >= 30
    assert pd.Timestamp(calibrated[0]["calibrated_through"]) < pd.Timestamp(forecasts[-1]["issue_at"])


def test_drift_and_performance_are_separate():
    idx = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    frame = pd.DataFrame({"FLOW": np.linspace(0, 1, 200), "EQ_TOC": 10., "FINAL_TOC": 10.}, index=idx)
    reference = fit_reference(frame, cutoff="2025-01-07T00:00:00Z", thresholds={"FLOW": 0.2, "EQ_TOC": 0.2}, thresholds_fit_before="2025-01-06T00:00:00Z")
    drift = input_drift(reference, pd.DataFrame({"FLOW": [100.] * 40, "EQ_TOC": [100.] * 40}))
    assert drift["FLOW"]["status"] == "observe"
    assert drift["EQ_TOC"]["status"] == "observe"
    assert performance_status([], {}) == "검증 데이터 부족"
    assert performance_status([{"n": 40, "window": "7d", "model": "candidate", "bias": 0, "improvement_pct": 1}], {}) == "관찰 필요"
    with pytest.raises(ValueError, match="development"):
        fit_reference(frame, cutoff="2025-01-07T00:00:00Z", thresholds={"FLOW": .2}, thresholds_fit_before="2025-01-08T00:00:00Z")


def test_bootstrap_rejects_insufficient_independent_blocks():
    idx = pd.date_range("2025-01-01", periods=696, freq="h")
    result = bootstrap_sensitivity(np.ones(696), np.zeros(696), np.full(696, 0.5), idx, draws=20)
    assert result["blocks"]["24"]["ci95"] == pytest.approx([0.5, 0.5])
    assert result["blocks"]["168"]["status"] == "insufficient"
    assert result["blocks"]["168"]["ci95"] is None
    assert coverage_sample_plan()["independent_samples_minimum"] > 100


def test_missing_event_window_is_not_scored():
    obs = [{"column": "FINAL_TOC", "sample_at": f"2025-01-01T{h:02d}:00:00+00:00", "value": 21 if h == 10 else 10} for h in range(20) if h != 8]
    result = event_windows(obs, [], alarm=20, horizon=6, model="persistence")
    assert result["event_n"] == 0
    assert result["censored_n"] == 1
    assert result["recall"] is None


def test_event_warning_uses_actual_issue_time():
    obs = [{"column": "FINAL_TOC", "sample_at": f"2025-01-01T{h:02d}:00:00+00:00", "value": 21 if h == 10 else 10} for h in range(20)]
    forecasts = [{"origin_at": f"2025-01-01T{h:02d}:00:00+00:00", "issue_at": f"2025-01-01T{h:02d}:30:00+00:00",
                  "targets": {"6": f"2025-01-01T{h+6:02d}:00:00+00:00"}, "predictions": {"6": {"persistence": 25 if h == 4 else 10}}} for h in range(4, 10)]
    result = event_windows(obs, forecasts, alarm=20, horizon=6, model="persistence")
    assert result["event_n"] == 1
    assert result["recall"] == 1
    assert result["events"][0]["warning_hours"] == 5.5


def test_new_validation_page_runs_without_private_data():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    page = Path(__file__).resolve().parents[1] / "wwt_toc_dashboard/pages/4_Model_Validation.py"
    app = AppTest.from_file(str(page), default_timeout=60).run()
    assert not app.exception
