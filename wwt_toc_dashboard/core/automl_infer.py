"""Standalone inference. V1 bundles remain available; V2 promotion is explicit."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from core.data import load_raw_csv
from core.forecast_pipeline import (
    PIPELINE_VERSION, PipelineConfig, available_inputs, build_features,
    fill_inputs, hourly_observations, quality_status,
)
from core.legacy_features import build_features_equipment_aware


def _guard_24h_jump(out: dict, current_toc: float) -> dict:
    """V1 postprocessor retained for comparison, never risk suppression."""
    result = dict(out)
    p6, p12, p24 = (result.get(f"pred_t{h}", np.nan) for h in (6, 12, 24))
    if all(np.isfinite(v) for v in (p6, p12, p24, current_toc)) and abs(p12 - current_toc) <= 1.2:
        jump = max(1.2, abs(p12 - p6) * 3.0 + 0.4)
        result["pred_t24"] = float(np.clip(p24, p12 - jump, p12 + jump))
    return result


def resolve_bundle(automl_root_path: str, model_dir_path: str | None = None) -> tuple[Path, Path]:
    if model_dir_path:
        models = Path(model_dir_path)
        return models, models.parent / "configs"
    local = Path(automl_root_path)
    if (local / "outputs" / "models").is_dir() and (local / "configs").is_dir():
        return local / "outputs" / "models", local / "configs"
    packaged = Path(__file__).resolve().parents[1] / "model_bundle"
    return packaged / "models", packaged / "configs"


def _checked_predict(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    columns = bundle.get("columns")
    if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
        raise ValueError("Model must record an ordered, unique feature schema")
    missing = [c for c in columns if c not in frame]
    if missing:
        raise ValueError(f"Feature schema mismatch: {len(missing)} missing columns, e.g. {missing[:4]}")
    model = bundle["model"]
    estimator = getattr(model, "model", model)
    if getattr(estimator, "n_features_in_", len(columns)) != len(columns):
        raise ValueError("Bundle schema length does not match model")
    schema_model = getattr(estimator, "base", estimator)
    names = getattr(schema_model, "feature_names_in_", getattr(schema_model, "feature_name_", None))
    if names is not None and list(names) != columns:
        raise ValueError("Bundle feature order does not match the trained model schema")
    if frame.empty:
        raise ValueError("Empty feature matrix")
    pred = np.asarray(model.predict(frame[columns]), dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Model returned nonfinite predictions")
    return pred


def _legacy_frame(observed: pd.DataFrame, cfg: PipelineConfig, feature_cfg: dict) -> pd.DataFrame:
    frame = fill_inputs(available_inputs(observed, cfg), cfg).copy()
    # A dummy label prevents legacy dropna() deleting the live forecast horizon.
    frame["FINAL_TOC_raw"] = frame["FINAL_TOC"]
    frame["FINAL_TOC_is_bad"] = frame["FINAL_TOC"].isna().astype(int)
    frame["FINAL_TOC_is_maint"] = 0
    frame["FINAL_TOC"] = 0.0
    frame = frame.rename_axis("날짜").reset_index()
    return build_features_equipment_aware(frame, time_col="날짜", target_col="FINAL_TOC", **feature_cfg)


def _predict_legacy(observed: pd.DataFrame, lead: int, models: Path, configs: Path, cfg: PipelineConfig) -> tuple[float, str]:
    feature_cfg = yaml.safe_load((configs / "features.yaml").read_text(encoding="utf-8"))
    frame = _legacy_frame(observed.tail(512), cfg, feature_cfg)
    if frame.empty or pd.Timestamp(frame["날짜"].iloc[-1]) != observed.index[-1]:
        raise ValueError("Latest origin has insufficient valid V1 features; refusing an older forecast")
    base_bundle = joblib.load(models / f"toc_lgbm_lead{lead}h_baseline.joblib")
    if int(base_bundle.get("lead_hours", -1)) != lead:
        raise ValueError("Model horizon mismatch")
    baseline = _checked_predict(base_bundle, frame)
    value = float(baseline[-1])
    res_path = models / f"toc_lgbm_lead{lead}h_residual.joblib"
    if not res_path.exists():
        return value, "baseline_only"
    res_cfg = yaml.safe_load((configs / "features_residual.yaml").read_text(encoding="utf-8"))
    frame["pred_baseline"] = baseline
    frame["residual"] = 0.0
    try:
        residual = build_features_equipment_aware(frame, time_col="날짜", target_col="FINAL_TOC", **res_cfg)
        if residual.empty or pd.Timestamp(residual["날짜"].iloc[-1]) != observed.index[-1]:
            raise ValueError("Residual origin differs from baseline origin")
        res_bundle = joblib.load(res_path)
        if int(res_bundle.get("lead_hours", -1)) != lead:
            raise ValueError("Residual horizon mismatch")
        value += float(_checked_predict(res_bundle, residual.tail(1))[-1])
        return value, "legacy_residual_oof_unverified"
    except (ValueError, KeyError) as error:
        return value, f"baseline_only; residual unavailable: {error}"


def _predict_v2(raw: pd.DataFrame, lead: int, models: Path) -> tuple[float, dict]:
    bundle = joblib.load(models / f"forecast_{lead}h.joblib")
    if bundle.get("pipeline_version") != PIPELINE_VERSION or bundle.get("horizon") != lead:
        raise ValueError("V2 pipeline version/horizon mismatch")
    if not bundle.get("deployment_gate_passed", False):
        raise ValueError("V2 deployment gate failed or not evaluated; retain the existing model")
    config_data = dict(bundle["pipeline_config"])
    config_data["sensor_delays"] = tuple(tuple(x) for x in config_data.get("sensor_delays", ()))
    cfg = PipelineConfig(**config_data)
    if bundle.get("config_digest") != cfg.digest():
        raise ValueError("V2 configuration fingerprint mismatch")
    X = build_features(raw, cfg)
    pred = float(_checked_predict(bundle, X.tail(1))[-1])
    metadata = {"mode": "v2_explicit_opt_in", "trained_through": bundle.get("trained_through"), "pipeline_config": config_data}
    radius = bundle.get("interval_radius")
    if bundle.get("interval_calibration_passed") and isinstance(radius, (float, int)) and np.isfinite(radius) and radius >= 0:
        metadata["interval"] = [max(0.0, pred - radius), pred + radius]
        metadata["interval_nominal_coverage"] = bundle.get("interval_coverage")
        metadata["interval_test_coverage"] = bundle.get("interval_test_coverage")
        metadata["interval_note"] = "Calibration residual interval; time-series coverage is empirical, not guaranteed"
    return pred, metadata


def predict_multi_leads_from_automl(raw_path: str, automl_root_path: str = "", model_dir_path: str | None = None,
                                   leads: tuple[int, ...] = (6, 12, 24)) -> dict[str, Any]:
    raw = load_raw_csv(raw_path)
    observed = hourly_observations(raw)
    models, configs = resolve_bundle(automl_root_path, model_dir_path)
    base_cfg = yaml.safe_load((configs / "base.yaml").read_text(encoding="utf-8"))
    delay_minutes = int(base_cfg.get("toc_delay_minutes", 60))
    if delay_minutes < 0 or delay_minutes % 60:
        raise ValueError("An integral, nonnegative hourly target delay is required")
    cfg = PipelineConfig(target_delay_hours=delay_minutes // 60)
    v2_dir = os.getenv("WWT_V2_MODEL_DIR", "").strip()
    out: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for lead in leads:
        try:
            if v2_dir:
                value, detail = _predict_v2(raw, lead, Path(v2_dir))
                effective_horizon = lead
            else:
                value, mode = _predict_legacy(observed, lead, models, configs, cfg)
                detail = {"mode": mode, "validation": "unverified; recovered sources do not establish training provenance"}
                effective_horizon = lead + cfg.target_delay_hours
            out[f"pred_t{lead}"] = value
            detail["effective_horizon_hours"] = effective_horizon
            detail["target_at"] = (observed.index[-1] + pd.Timedelta(hours=effective_horizon)).isoformat()
            details[str(lead)] = detail
        except Exception as error:
            out[f"pred_t{lead}"] = float("nan")
            details[str(lead)] = {"error": f"{type(error).__name__}: {error}"}
    if v2_dir and details:
        data = next((d["pipeline_config"] for d in details.values() if "pipeline_config" in d), None)
        if data:
            data = dict(data)
            data["sensor_delays"] = tuple(tuple(x) for x in data.get("sensor_delays", ()))
            cfg = PipelineConfig(**data)
    latest_toc = available_inputs(observed, cfg)["FINAL_TOC"].iloc[-1]
    current = float(latest_toc) if pd.notna(latest_toc) else float("nan")
    guarded = _guard_24h_jump(out, current)
    out.update({f"pred_t{lead}_raw": out[f"pred_t{lead}"] for lead in leads})
    out["pred_t24_legacy_guarded"] = guarded.get("pred_t24", float("nan"))
    out["current_toc_ref"] = current
    artifacts = list((Path(v2_dir) if v2_dir else models).glob("*.joblib"))
    out["_meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "origin_at": observed.index[-1].isoformat(),
        "current_toc_sample_at": (observed.index[-1] - pd.Timedelta(hours=cfg.target_delay_hours)).isoformat(),
        "timestamp_timezone": "source timezone not recorded; no timezone conversion applied",
        "pipeline_version": PIPELINE_VERSION,
        "model_updated_at": datetime.fromtimestamp(max(p.stat().st_mtime for p in artifacts), timezone.utc).isoformat() if artifacts else None,
        "model_updated_at_note": "artifact file modification time; training time unavailable for V1",
        "quality": quality_status(raw, cfg), "horizons": details,
        "probability_status": "unverified; no calibrated probability is reported",
    }
    return out
