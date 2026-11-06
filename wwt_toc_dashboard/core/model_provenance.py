"""Validate a private model manifest without loading its executable artifact.

This checks consistency, not the truth of operator-supplied provenance claims.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from core.shadow_store import utc


def validate_manifest(manifest: dict, artifact_path, issue_at) -> dict:
    required_text = ("model_id", "model_type", "feature_pipeline_version", "code_commit", "calibration_source")
    if any(not isinstance(manifest.get(k), str) or not manifest[k].strip() for k in required_text):
        raise ValueError("Model identification, pipeline, commit and calibration source are required")
    for key in ("training_data_sha256", "model_sha256"):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[key]):
            raise ValueError(f"{key} must be a lowercase SHA256")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["code_commit"]):
        raise ValueError("code_commit must be a full Git commit hash")
    if type(manifest.get("horizon")) is not int or manifest["horizon"] not in (6, 12, 24):
        raise ValueError("Invalid horizon")
    if not isinstance(manifest.get("hyperparameters"), dict) or not isinstance(manifest.get("environment"), dict) or not manifest["environment"]:
        raise ValueError("Hyperparameters and training environment are required")
    if manifest.get("verification_status") not in ("unverified", "externally_reviewed"):
        raise ValueError("Invalid verification status")
    if manifest.get("deployment_status") not in ("withheld", "approved"):
        raise ValueError("Invalid deployment status")
    time_keys = ("training_start", "training_end", "target_available_through", "training_run_started_at", "training_run_completed_at", "artifact_created_at")
    if any(k not in manifest for k in time_keys):
        raise ValueError("Training and artifact timestamps are required")
    start, end, labels, run_start, run_end, created = (utc(manifest[k]) for k in time_keys)
    issue = utc(issue_at)
    if not start <= end <= labels <= run_start <= run_end <= created < issue:
        raise ValueError("Training, target availability, artifact creation and issue times are inconsistent")
    actual = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
    if actual != manifest["model_sha256"]:
        raise ValueError("Model SHA256 mismatch")
    return {"model_sha256": actual, "claims_verified": False, "deployment_authorized": False}
