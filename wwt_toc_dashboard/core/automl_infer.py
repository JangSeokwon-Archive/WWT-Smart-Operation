from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


FALLBACK_BASE_CFG: dict[str, Any] = {
    "time_col": "날짜",
    "target_col": "FINAL_TOC",
    "sampling_minutes": 60,
    "toc_delay_minutes": 60,
    "k_drop": 8.0,
    "maint_min_run": 7,
    "keep_long_drop": True,
}

FALLBACK_FEAT_CFG: dict[str, Any] = {
    "equipment_prefixes": ["AERA", "AERB", "CLAA", "CLAB", "EQ", "DAF"],
    "global_cols": ["FLOW", "TEMP"],
    "global_feature_params": {"lags": [12, 24, 48], "rolls": [24, 48, 72], "diffs": [1, 3, 6]},
    "per_equipment": {
        "AERA": {"lags": [1, 2, 3, 6, 12, 24, 48, 72, 96, 120], "rolls": [3, 12, 24, 48, 72], "diffs": [1, 3, 6, 12, 24, 48]},
        "AERB": {"lags": [2, 3, 6, 12, 24, 48, 72, 96, 120], "rolls": [24, 48, 72], "diffs": [1, 3, 6, 12, 24, 48]},
        "CLAA": {"lags": [1, 2, 3, 12, 24, 48, 72, 96, 120], "rolls": [3, 6, 12, 24, 48, 72], "diffs": [1, 3, 6, 12, 24]},
        "CLAB": {"lags": [1, 2, 3, 6, 12, 24], "rolls": [3, 6, 12, 24, 48, 72], "diffs": [1, 3, 6, 12, 24]},
        "EQ": {"lags": [1, 6, 12, 24, 48, 96, 120], "rolls": [3, 6, 12, 24, 48, 72], "diffs": [1, 3, 6, 12]},
        "DAF": {"lags": [6, 12, 24, 72, 96, 120], "rolls": [24, 48, 72], "diffs": [1, 3, 6, 12]},
    },
    "cross_features": {"aer_ab_mean_diff": True, "cla_ab_mean_diff": True},
    "leakage_guard": {"exclude_prefixes": ["FINAL"]},
}

FALLBACK_FEAT_RES_CFG: dict[str, Any] = {
    "equipment_prefixes": ["AERA", "AERB", "CLAA", "CLAB"],
    "global_cols": [],
    "global_feature_params": {"lags": [], "rolls": [], "diffs": []},
    "per_equipment": {
        "AERA": {"lags": [12, 24, 48, 72], "rolls": [24, 48], "diffs": [24]},
        "AERB": {"lags": [12, 24, 48, 72], "rolls": [24, 48], "diffs": [24]},
        "CLAA": {"lags": [12, 24, 48], "rolls": [24], "diffs": []},
        "CLAB": {"lags": [12, 24, 48], "rolls": [24], "diffs": []},
    },
    "cross_features": {"aer_ab_mean_diff": True, "cla_ab_mean_diff": False},
    "leakage_guard": {"exclude_prefixes": ["FINAL", "EQ", "DAF"]},
}


def _load_cfg(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if yaml is None:
        return dict(fallback)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else dict(fallback)
    except Exception:
        return dict(fallback)


def _load_csv_auto(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e
    raise last_err if last_err else ValueError(f"CSV read failed: {path}")


def _import_automl_modules(automl_root: Path):
    sys.path.insert(0, str(automl_root))
    from src.features.build import build_features_equipment_aware  # type: ignore
    from src.io.validate import validate_basic  # type: ignore
    from src.preprocessing.align import align_target_delay  # type: ignore
    from src.preprocessing.missing import handle_missing_hourly  # type: ignore
    from src.preprocessing.toc_cleaning import preprocess_final_toc  # type: ignore

    return (
        build_features_equipment_aware,
        validate_basic,
        align_target_delay,
        handle_missing_hourly,
        preprocess_final_toc,
    )


def _prepare_aligned(
    raw_df: pd.DataFrame,
    lead: int,
    base_cfg: dict[str, Any],
    validate_basic,
    handle_missing_hourly,
    preprocess_final_toc,
    align_target_delay,
) -> pd.DataFrame:
    time_col = base_cfg["time_col"]
    target_col = base_cfg["target_col"]

    df = validate_basic(raw_df.copy(), time_col, target_col)
    df.columns = df.columns.str.strip()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = handle_missing_hourly(df, time_col=time_col)
    df = preprocess_final_toc(
        df=df,
        time_col=time_col,
        toc_col=target_col,
        k_drop=float(base_cfg.get("k_drop", 8.0)),
        maint_min_run=int(base_cfg.get("maint_min_run", 7)),
        keep_long_drop=bool(base_cfg.get("keep_long_drop", True)),
    )
    df = align_target_delay(
        df=df,
        time_col=time_col,
        target_col=target_col,
        toc_delay_minutes=int(base_cfg.get("toc_delay_minutes", 0)),
        sampling_minutes=int(base_cfg.get("sampling_minutes", 60)),
        lead_hours=int(lead),
    )
    return df


def _predict_for_lead(
    raw_df: pd.DataFrame,
    lead: int,
    automl_root: Path,
    model_dir_override: Path | None,
    base_cfg: dict[str, Any],
    feat_cfg: dict[str, Any],
    feat_res_cfg: dict[str, Any],
    build_features_equipment_aware,
    validate_basic,
    align_target_delay,
    handle_missing_hourly,
    preprocess_final_toc,
) -> float:
    time_col = base_cfg["time_col"]
    target_col = base_cfg["target_col"]
    model_dir = model_dir_override if isinstance(model_dir_override, Path) else (automl_root / "outputs" / "models")

    base_model_path = model_dir / f"toc_lgbm_lead{lead}h_baseline.joblib"
    if not base_model_path.exists():
        raise FileNotFoundError(f"baseline model missing: {base_model_path}")

    aligned = _prepare_aligned(
        raw_df=raw_df,
        lead=lead,
        base_cfg=base_cfg,
        validate_basic=validate_basic,
        handle_missing_hourly=handle_missing_hourly,
        preprocess_final_toc=preprocess_final_toc,
        align_target_delay=align_target_delay,
    )

    df_feat = build_features_equipment_aware(
        df=aligned.copy(),
        time_col=time_col,
        target_col=target_col,
        equipment_prefixes=feat_cfg["equipment_prefixes"],
        global_cols=feat_cfg.get("global_cols", []),
        global_feature_params=feat_cfg.get("global_feature_params", {}),
        per_equipment=feat_cfg["per_equipment"],
        cross_features=feat_cfg.get("cross_features", {}),
        leakage_guard=feat_cfg.get("leakage_guard", {}),
    )
    if len(df_feat) == 0:
        return float("nan")

    X_base = df_feat.drop(columns=[target_col, time_col], errors="ignore")
    base_bundle = joblib.load(base_model_path)
    base_model = base_bundle["model"]
    base_cols = base_bundle.get("columns", list(X_base.columns))
    if any(c not in X_base.columns for c in base_cols):
        return float("nan")

    pred_base_all = base_model.predict(X_base[base_cols])
    pred_base_last = float(pred_base_all[-1])

    # residual model is optional by lead. If missing, fallback to baseline prediction.
    res_model_path = model_dir / f"toc_lgbm_lead{lead}h_residual.joblib"
    if not res_model_path.exists():
        return pred_base_last

    df_feat["pred_baseline"] = pred_base_all
    df_feat["residual"] = 0.0
    df_res = build_features_equipment_aware(
        df=df_feat.copy(),
        time_col=time_col,
        target_col=target_col,
        equipment_prefixes=feat_res_cfg["equipment_prefixes"],
        global_cols=feat_res_cfg.get("global_cols", []),
        global_feature_params=feat_res_cfg.get("global_feature_params", {}),
        per_equipment=feat_res_cfg["per_equipment"],
        cross_features=feat_res_cfg.get("cross_features", {}),
        leakage_guard=feat_res_cfg.get("leakage_guard", {}),
    )
    if len(df_res) == 0:
        return pred_base_last

    res_bundle = joblib.load(res_model_path)
    res_model = res_bundle["model"]
    res_cols = res_bundle.get("columns", [])
    if any(c not in df_res.columns for c in res_cols):
        return pred_base_last

    pred_res_all = res_model.predict(df_res[res_cols])
    pred_res_last = float(pred_res_all[-1])
    # Combine residual correction on the latest baseline horizon prediction.
    return pred_base_last + pred_res_last

def _latest_valid_toc(raw_df: pd.DataFrame) -> float:
    if "FINAL_TOC" not in raw_df.columns:
        return float("nan")
    s = pd.to_numeric(raw_df["FINAL_TOC"], errors="coerce")
    s = s[(s > 0) & (s < 50)]
    if len(s) == 0:
        return float("nan")
    return float(s.iloc[-1])

def _recent_volatility(raw_df: pd.DataFrame, n: int = 72) -> float:
    if "FINAL_TOC" not in raw_df.columns:
        return float("nan")
    s = pd.to_numeric(raw_df["FINAL_TOC"], errors="coerce")
    s = s[(s > 0) & (s < 50)].tail(int(n))
    if len(s) < 8:
        return float("nan")
    v = s.diff().dropna().std()
    return float(v) if pd.notna(v) else float("nan")

def _stabilize_multi_leads(out: dict[str, float], current_toc: float, vol: float) -> dict[str, float]:
    out2 = dict(out)
    p6 = out2.get("pred_t6", float("nan"))
    p12 = out2.get("pred_t12", float("nan"))
    p24 = out2.get("pred_t24", float("nan"))

    if not pd.notna(current_toc):
        return out2

    # Volatility-adaptive jump limits.
    # Low vol => tighter cap, high vol => wider cap.
    base = 2.6
    if pd.notna(vol):
        base = float(max(2.6, min(6.5, 1.8 + 4.0 * vol)))
    lim6 = base * 0.55
    lim12 = base * 0.75
    lim24 = base

    if pd.notna(p6):
        p6 = float(current_toc + max(-lim6, min(lim6, float(p6) - current_toc)))
        out2["pred_t6"] = p6

    if pd.notna(p12):
        anchor12 = p6 if pd.notna(p6) else current_toc
        p12 = float(anchor12 + max(-lim12, min(lim12, float(p12) - anchor12)))
        out2["pred_t12"] = p12

    if pd.notna(p24):
        anchor24 = p12 if pd.notna(p12) else (p6 if pd.notna(p6) else current_toc)
        p24_clip = float(anchor24 + max(-lim24, min(lim24, float(p24) - anchor24)))
        # Blend to reduce isolated 24h spikes while preserving trend direction.
        p24 = float(0.75 * p24_clip + 0.25 * anchor24)
        out2["pred_t24"] = p24

    return out2


def _guard_24h_jump(out: dict[str, float], current_toc: float) -> dict[str, float]:
    """Apply a light temporal-consistency guard only on 24h prediction."""
    out2 = dict(out)
    p6 = out2.get("pred_t6", float("nan"))
    p12 = out2.get("pred_t12", float("nan"))
    p24 = out2.get("pred_t24", float("nan"))
    if not (pd.notna(p6) and pd.notna(p12) and pd.notna(p24) and pd.notna(current_toc)):
        return out2
    p6 = float(p6)
    p12 = float(p12)
    p24 = float(p24)
    cur = float(current_toc)
    # If short-term (6/12h) is relatively stable but 24h jumps abruptly, clamp jump.
    short_span = abs(p12 - cur)
    continuity = abs(p12 - p6)
    if short_span <= 1.2:
        max_jump = max(1.2, (continuity * 3.0) + 0.4)
        upper = p12 + max_jump
        lower = p12 - max_jump
        if p24 > upper:
            out2["pred_t24"] = upper
        elif p24 < lower:
            out2["pred_t24"] = lower
    return out2


def predict_multi_leads_from_automl(
    raw_path: str,
    automl_root_path: str,
    model_dir_path: str | None = None,
    leads: tuple[int, ...] = (12, 24, 36),
) -> dict[str, float]:
    """
    Returns {"pred_t12": v, "pred_t24": v, "pred_t36": v}
    using automl baseline+residual bundles when available.
    """
    automl_root = Path(automl_root_path)
    model_dir_override = Path(model_dir_path) if isinstance(model_dir_path, str) and model_dir_path.strip() else None
    raw_df = _load_csv_auto(Path(raw_path))
    base_cfg = _load_cfg(automl_root / "configs" / "base.yaml", FALLBACK_BASE_CFG)
    feat_cfg = _load_cfg(automl_root / "configs" / "features.yaml", FALLBACK_FEAT_CFG)
    feat_res_cfg = _load_cfg(automl_root / "configs" / "features_residual.yaml", FALLBACK_FEAT_RES_CFG)

    (
        build_features_equipment_aware,
        validate_basic,
        align_target_delay,
        handle_missing_hourly,
        preprocess_final_toc,
    ) = _import_automl_modules(automl_root)

    out: dict[str, float] = {}
    for lead in leads:
        key = f"pred_t{lead}"
        try:
            out[key] = float(
                _predict_for_lead(
                    raw_df=raw_df,
                    lead=int(lead),
                    automl_root=automl_root,
                    model_dir_override=model_dir_override,
                    base_cfg=base_cfg,
                    feat_cfg=feat_cfg,
                    feat_res_cfg=feat_res_cfg,
                    build_features_equipment_aware=build_features_equipment_aware,
                    validate_basic=validate_basic,
                    align_target_delay=align_target_delay,
                    handle_missing_hourly=handle_missing_hourly,
                    preprocess_final_toc=preprocess_final_toc,
                )
            )
        except Exception:
            out[key] = float("nan")

    # Return raw base+residual predictions directly.
    out_raw = {f"{k}_raw": float(v) for k, v in out.items()}
    cur_toc = _latest_valid_toc(raw_df)
    vol = _recent_volatility(raw_df, n=72)
    out = _guard_24h_jump(out, current_toc=cur_toc)
    out.update(out_raw)
    out["current_toc_ref"] = float(cur_toc) if pd.notna(cur_toc) else float("nan")
    out["recent_vol_ref"] = float(vol) if pd.notna(vol) else float("nan")
    return out
