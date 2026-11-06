import hashlib

import pytest

from core.model_provenance import validate_manifest


def test_manifest_checks_hash_and_temporal_order_without_trusting_claims(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"synthetic artifact; never deserialize")
    manifest = {
        "model_id": "synthetic-new-candidate", "model_type": "synthetic", "horizon": 24,
        "training_data_sha256": "a" * 64, "model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "training_start": "2025-01-01T00:00:00Z", "training_end": "2025-01-02T00:00:00Z",
        "target_available_through": "2025-01-02T01:00:00Z", "training_run_started_at": "2025-01-02T02:00:00Z",
        "training_run_completed_at": "2025-01-02T03:00:00Z", "artifact_created_at": "2025-01-03T00:00:00Z",
        "feature_pipeline_version": "test", "code_commit": "a" * 40, "hyperparameters": {},
        "environment": {"python": "test"}, "calibration_source": "none",
        "verification_status": "externally_reviewed", "deployment_status": "approved",
    }
    result = validate_manifest(manifest, artifact, "2025-01-04T00:00:00Z")
    assert not result["claims_verified"] and not result["deployment_authorized"]
    with pytest.raises(ValueError, match="times"):
        validate_manifest(manifest, artifact, "2025-01-03T00:00:00Z")
    with pytest.raises(ValueError, match="times"):
        validate_manifest({**manifest, "target_available_through": "2025-01-04T00:00:00Z"}, artifact, "2025-01-05T00:00:00Z")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_manifest({**manifest, "model_sha256": "b" * 64}, artifact, "2025-01-04T00:00:00Z")


@pytest.mark.parametrize("field,value", [("column", " "), ("source_id", "\t"), ("sample_at", None), ("available_at", "NaT")])
def test_invalid_observation_contract_is_rejected(ledger_v23, field, value):
    args = {"column": "FLOW", "sample_at": "2025-01-01T00:00:00Z", "available_at": "2025-01-01T01:00:00Z",
            "value": 0, "source_id": "sensor", "replay_at": "2025-01-01T01:00:00Z"}
    args[field] = value
    with pytest.raises(ValueError):
        ledger_v23.ingest(**args)
    assert ledger_v23.events() == []


@pytest.fixture
def ledger_v23(tmp_path):
    from core.shadow_store import ShadowStore
    store = ShadowStore(tmp_path / "replay.sqlite", "historical_replay")
    yield store
    store.close()
