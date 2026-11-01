"""Separate forecast performance from input distribution monitoring."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from core.shadow_store import utc

DRIFT_COLUMNS = ("FINAL_TOC", "EQ_TOC", "DAF_TOC", "FLOW", "TEMP", "AERB_DO", "AERB_MLSS", "AERB_RET", "AERB_WIT", "organic_load_proxy")


def fit_reference(data: pd.DataFrame, *, cutoff, thresholds=None, min_samples=100, target_delay_hours=1, thresholds_fit_before=None):
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("Reference requires timezone-aware sample times")
    frame = data.loc[data.index < pd.Timestamp(utc(cutoff))].copy()
    if "FINAL_TOC" in frame:
        frame.loc[frame.index + pd.Timedelta(hours=target_delay_hours) >= pd.Timestamp(utc(cutoff)), "FINAL_TOC"] = np.nan
    if {"FLOW", "EQ_TOC"}.issubset(frame.columns):
        frame["organic_load_proxy"] = frame.FLOW * frame.EQ_TOC
    if thresholds and (thresholds_fit_before is None or utc(thresholds_fit_before) >= utc(cutoff)):
        raise ValueError("Threshold development must end before the reference cutoff")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in (thresholds or {}).values()):
        raise ValueError("Drift thresholds must be finite and nonnegative")
    thresholds = {str(k): float(v) for k, v in (thresholds or {}).items()}
    result = {"cutoff": pd.Timestamp(utc(cutoff)).isoformat(), "target_delay_hours": target_delay_hours,
              "thresholds": thresholds or {}, "thresholds_fit_before": str(thresholds_fit_before) if thresholds_fit_before else None, "columns": {}}
    for col in DRIFT_COLUMNS:
        if col not in frame:
            continue
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if len(s) < min_samples:
            continue
        quantiles = np.unique(np.quantile(s, np.linspace(0, 1, 11)))
        if len(quantiles) == 1:
            epsilon = max(abs(float(quantiles[0])) * 1e-6, 1e-6)
            edges = np.array([quantiles[0] - epsilon, quantiles[0] + epsilon])
        else:
            edges = quantiles[1:-1]
        counts = np.histogram(s, bins=np.r_[-np.inf, edges, np.inf])[0]
        result["columns"][col] = {"edges": edges.tolist(), "counts": counts.tolist(), "n": len(s), "mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}
    return result


def input_drift(reference, recent: pd.DataFrame, *, min_samples=30):
    frame = recent.copy()
    if {"FLOW", "EQ_TOC"}.issubset(frame.columns):
        frame["organic_load_proxy"] = frame.FLOW * frame.EQ_TOC
    results = {}
    for col, ref in reference.get("columns", {}).items():
        s = pd.to_numeric(frame[col], errors="coerce").dropna() if col in frame else pd.Series(dtype=float)
        if len(s) < min_samples:
            results[col] = {"status": "insufficient", "n": len(s)}
            continue
        counts = np.histogram(s, bins=np.r_[-np.inf, ref["edges"], np.inf])[0]
        a = (np.asarray(ref["counts"]) + 0.5) / (sum(ref["counts"]) + 0.5 * len(counts))
        b = (counts + 0.5) / (len(s) + 0.5 * len(counts))
        psi = float(np.sum((b - a) * np.log(b / a)))
        threshold = reference.get("thresholds", {}).get(col)
        try:
            if threshold is not None and utc(reference["thresholds_fit_before"]) >= utc(reference["cutoff"]):
                threshold = None
        except (KeyError, TypeError, ValueError):
            threshold = None
        results[col] = {"status": "observe" if threshold is not None and psi >= threshold else ("normal" if threshold is not None else "threshold_unvalidated"),
                        "n": len(s), "psi": psi, "mean_difference": float(s.mean() - ref["mean"]), "threshold": threshold}
    return results


def summarize_scores(rows: pd.DataFrame, as_of, *, alarm=None, rise_delta=None, min_samples=30, regime_config=None):
    if rows.empty:
        return {"groups": [], "status": "insufficient", "pending": 0, "missing": 0}
    when = pd.Timestamp(utc(as_of))
    groups = []
    for (h, model, model_sha), group in rows.groupby(["horizon", "model", "model_sha256"], sort=True):
        for label, days in (("7d", 7), ("30d", 30), ("all", None)):
            part = group if days is None else group[pd.to_datetime(group.issue_at, utc=True) >= when - pd.Timedelta(days=days)]
            scored = part[part.status == "scored"]
            n = len(scored)
            item = {"horizon": int(h), "model": model, "model_sha256": model_sha, "window": label, "n": n,
                    "first_issue_at": str(part.issue_at.min()) if len(part) else None,
                    "pending": int((part.status == "pending").sum()), "missing": int((part.status == "missing").sum()),
                    "status": "insufficient" if n < min_samples else "observed_not_validated"}
            if n:
                error = scored.prediction.to_numpy(float) - scored.actual.to_numpy(float)
                base = scored.persistence.to_numpy(float) - scored.actual.to_numpy(float)
                mae = float(np.abs(error).mean())
                base_mae = float(np.abs(base).mean())
                item.update(mae=mae, rmse=float(np.sqrt(np.mean(error ** 2))), bias=float(error.mean()),
                            improvement_pct=(100 * (base_mae - mae) / base_mae if base_mae else None))
                interval = scored.dropna(subset=["interval_low", "interval_high"])
                item["interval_n"] = len(interval)
                item["coverage"] = float(((interval.actual >= interval.interval_low) & (interval.actual <= interval.interval_high)).mean()) if len(interval) >= min_samples else None
                item["interval_width"] = float((interval.interval_high - interval.interval_low).mean()) if len(interval) >= min_samples else None
                item["coverage_status"] = "observed_not_validated" if len(interval) >= min_samples else "insufficient"
                if alarm is not None:
                    positive = scored.actual >= alarm
                    alarmed = scored.prediction >= alarm
                    item["exceedance_n"] = int(positive.sum())
                    item["exceedance_recall"] = float((positive & alarmed).sum() / positive.sum()) if positive.sum() else None
                    item["false_alarm_rate"] = float((~positive & alarmed).sum() / (~positive).sum()) if (~positive).sum() else None
                    item["false_alarm_n"] = int((~positive & alarmed).sum())
                if rise_delta is not None:
                    actual_rise = scored.actual - scored.persistence >= rise_delta
                    predicted_rise = scored.prediction - scored.persistence >= rise_delta
                    item["rise_n"] = int(actual_rise.sum())
                    item["rise_recall"] = float((actual_rise & predicted_rise).sum() / actual_rise.sum()) if actual_rise.sum() else None
                cfg = regime_config or {}
                definitions = {"high_load": ("organic_load_proxy", cfg.get("high_load_threshold")),
                               "inflow_rise": ("EQ_TOC_change24", cfg.get("inflow_toc_rise_delta")),
                               "load_change": ("organic_load_change24", cfg.get("load_change_delta")),
                               "mlss_change": ("AERB_MLSS_change24", cfg.get("mlss_change_delta")),
                               "missing_inputs": ("missing_fraction", cfg.get("missing_fraction_threshold")),
                               "outside_reference": ("outside_reference_fraction", cfg.get("outside_reference_fraction_threshold"))}
                for regime, (column, threshold) in definitions.items():
                    if threshold is None or column not in scored:
                        item[regime + "_n"] = None
                        item[regime + "_mae"] = None
                        item[regime + "_coverage"] = None
                        continue
                    values = pd.to_numeric(scored[column], errors="coerce")
                    if regime in ("load_change", "mlss_change"):
                        values = values.abs()
                    mask = values >= threshold
                    item[regime + "_n"] = int(mask.sum())
                    item[regime + "_mae"] = float(np.abs(scored.loc[mask, "prediction"] - scored.loc[mask, "actual"]).mean()) if mask.sum() >= min_samples else None
                    subset = scored.loc[mask].dropna(subset=["interval_low", "interval_high"])
                    item[regime + "_coverage"] = float(((subset.actual >= subset.interval_low) & (subset.actual <= subset.interval_high)).mean()) if len(subset) >= min_samples else None
                threshold = cfg.get("high_load_threshold")
                if threshold is not None and "organic_load_proxy" in scored:
                    value = pd.to_numeric(scored.organic_load_proxy, errors="coerce")
                    normal = value.notna() & (value < threshold)
                    item["normal_load_n"] = int(normal.sum())
                    item["normal_load_mae"] = float(np.abs(scored.loc[normal, "prediction"] - scored.loc[normal, "actual"]).mean()) if normal.sum() >= min_samples else None
                    subset = scored.loc[normal].dropna(subset=["interval_low", "interval_high"])
                    item["normal_load_coverage"] = float(((subset.actual >= subset.interval_low) & (subset.actual <= subset.interval_high)).mean()) if len(subset) >= min_samples else None
                else:
                    item["normal_load_n"] = None
                    item["normal_load_mae"] = None
                    item["normal_load_coverage"] = None
            groups.append(item)
    return {"groups": groups, "status": "insufficient" if all(g["status"] == "insufficient" for g in groups) else "observed_not_validated",
            "pending": int((rows.status == "pending").sum()), "missing": int((rows.status == "missing").sum())}


def performance_status(groups, thresholds=None):
    thresholds = thresholds or {}
    if not groups or all(g.get("n", 0) < thresholds.get("min_samples", 30) for g in groups):
        return "검증 데이터 부족"
    recent = [g for g in groups if g.get("window") == "7d" and g.get("n", 0) >= thresholds.get("min_samples", 30)]
    if not recent:
        return "검증 데이터 부족"
    if not any(g["model"] != "persistence" for g in recent):
        return "검증 데이터 부족"
    if "min_improvement_pct" not in thresholds or "max_abs_bias" not in thresholds or "min_coverage" not in thresholds or "max_interval_width" not in thresholds:
        return "관찰 필요"
    try:
        development_hash = thresholds.get("development_sha256", "")
        if len(development_hash) != 64:
            return "관찰 필요"
        int(development_hash, 16)
        if any(utc(thresholds["calibrated_through"]) >= utc(g["first_issue_at"]) for g in recent if g["model"] != "persistence"):
            return "관찰 필요"
    except (ValueError, TypeError, KeyError):
        return "관찰 필요"
    for g in recent:
        if g["model"] == "persistence":
            continue
        if g.get("improvement_pct") is None:
            return "관찰 필요"
        if g.get("improvement_pct") is not None and g["improvement_pct"] < thresholds["min_improvement_pct"]:
            return "모델 성능 저하 의심"
        if abs(g.get("bias", 0)) > thresholds["max_abs_bias"]:
            return "모델 성능 저하 의심"
        if g.get("interval_n", 0) < thresholds.get("min_samples", 30):
            return "관찰 필요"
        if g.get("coverage") is not None and g["coverage"] < thresholds["min_coverage"]:
            return "모델 성능 저하 의심"
        if g.get("interval_width") is not None and g["interval_width"] > thresholds["max_interval_width"]:
            return "모델 성능 저하 의심"
    return "정상"


def event_windows(observations, forecasts, *, alarm, horizon, model):
    """Score uncensored event onsets using actual issuance times."""
    observed = sorted((e for e in observations if e["column"] == "FINAL_TOC"), key=lambda e: e["sample_at"])
    events = []
    censored = int(bool(observed and observed[0]["value"] >= alarm))
    for i in range(1, len(observed)):
        prev, cur = observed[i - 1], observed[i]
        onset = utc(cur["sample_at"])
        if onset - utc(prev["sample_at"]) != pd.Timedelta(hours=1):
            censored += int(cur["value"] >= alarm)
            continue
        if onset - utc(prev["sample_at"]) != pd.Timedelta(hours=1) or prev["value"] >= alarm or cur["value"] < alarm:
            continue
        end = onset
        finished = False
        for later in observed[i + 1:]:
            stamp = utc(later["sample_at"])
            if stamp - end != pd.Timedelta(hours=1):
                break
            if later["value"] < alarm:
                finished = True
                break
            end = stamp
        if not finished:
            censored += 1
            continue
        # Require a complete observation and forecast issuance window.
        expected = {onset - pd.Timedelta(hours=j) for j in range(1, horizon + 1)}
        available_samples = {utc(e["sample_at"]) for e in observed}
        candidates = [f for f in forecasts if model in f["predictions"].get(str(horizon), {}) and onset - pd.Timedelta(hours=horizon) <= utc(f["issue_at"]) < onset]
        issue_hours = {utc(f["origin_at"]) for f in candidates}
        if not expected.issubset(available_samples) or not expected.issubset(issue_hours):
            censored += 1
            continue
        alarms = [utc(f["issue_at"]) for f in candidates if f["predictions"][str(horizon)][model] >= alarm and onset <= utc(f["targets"][str(horizon)]) <= end]
        actuals = {utc(e["sample_at"]): float(e["value"]) for e in observed}
        before, after = [], []
        for f in forecasts:
            if model not in f["predictions"].get(str(horizon), {}):
                continue
            target = utc(f["targets"][str(horizon)])
            if target not in actuals or utc(f["issue_at"]) >= target:
                continue
            error = abs(f["predictions"][str(horizon)][model] - actuals[target])
            if onset - pd.Timedelta(hours=horizon) <= target < onset:
                before.append(error)
            elif onset <= target <= end + pd.Timedelta(hours=horizon):
                after.append(error)
        events.append({"onset": cur["sample_at"], "detected": bool(alarms), "warning_hours": (onset - min(alarms)).total_seconds() / 3600 if alarms else None,
                       "pre_event_n": len(before), "pre_event_mae": float(np.mean(before)) if len(before) >= 5 else None,
                       "post_event_n": len(after), "post_event_mae": float(np.mean(after)) if len(after) >= 5 else None,
                       "event_error_status": "descriptive_not_validated"})
    return {"event_n": len(events), "censored_n": censored, "detected": sum(e["detected"] for e in events), "missed": sum(not e["detected"] for e in events),
            "recall": sum(e["detected"] for e in events) / len(events) if events else None, "events": events}
