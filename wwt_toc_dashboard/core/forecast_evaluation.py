"""Time-based purge, metrics and empirical calibration utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def purged_folds(index: pd.DatetimeIndex, horizon: int, delay: int, n_splits: int = 3):
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Folds require increasing, unique timestamps")
    if n_splits < 2 or horizon <= 0 or delay < 0:
        raise ValueError("Invalid split parameters")
    blocks = np.array_split(np.arange(len(index)), n_splits + 1)
    for valid in blocks[1:]:
        if len(valid) == 0:
            raise ValueError("Insufficient data for folds")
        # Strict inequality: a training label must be reported before prediction.
        train = np.flatnonzero(index + pd.Timedelta(hours=horizon + delay) < index[valid[0]])
        if len(train) < 48:
            raise ValueError("Insufficient training observations after time purge")
        yield train, valid


def conformal_radius(y, prediction, coverage: float = 0.9) -> float:
    errors = np.abs(np.asarray(y, float) - np.asarray(prediction, float))
    errors = errors[np.isfinite(errors)]
    if not 0 < coverage < 1 or len(errors) < 30:
        raise ValueError("Calibration requires >=30 finite observations and 0<coverage<1")
    quantile = min(1.0, np.ceil((len(errors) + 1) * coverage) / len(errors))
    return float(np.quantile(errors, quantile, method="higher"))


def _rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def regression_metrics(y, prediction, current, high_load, alarm: float, shock: float) -> dict:
    actual, pred, current = (np.asarray(x, dtype=float) for x in (y, prediction, current))
    high_load = np.asarray(high_load, dtype=bool)
    valid = np.isfinite(actual) & np.isfinite(pred)
    if not valid.any():
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    error = pred[valid] - actual[valid]
    high = valid & high_load
    rising = valid & np.isfinite(current) & ((actual - current) >= shock)
    predicted_rise = (pred - current) >= shock
    positives = valid & (actual >= alarm)
    negatives = valid & (actual < alarm)
    alarms = valid & (pred >= alarm)
    return {
        "n": int(valid.sum()), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(error.mean()), "high_load_n": int(high.sum()),
        "high_load_mae": float(np.abs(pred[high] - actual[high]).mean()) if high.any() else None,
        "normal_mae": float(np.abs(pred[valid & ~high_load] - actual[valid & ~high_load]).mean()) if (valid & ~high_load).any() else None,
        "shock_n": int(rising.sum()), "shock_mae": float(np.abs(pred[rising] - actual[rising]).mean()) if rising.any() else None,
        "rise_detection_rate": _rate(int((predicted_rise & rising).sum()), int(rising.sum())),
        "exceedance_recall": _rate(int((positives & alarms).sum()), int(positives.sum())),
        "false_alarm_rate": _rate(int((negatives & alarms).sum()), int(negatives.sum())),
        "exceedance_n": int(positives.sum()),
    }


def event_metrics(origins: pd.DatetimeIndex, y, predictions, horizon: int, alarm: float) -> dict:
    actual = pd.Series(np.asarray(y, float), index=origins + pd.Timedelta(hours=horizon))
    pred = pd.Series(np.asarray(predictions, float), index=actual.index)
    over = actual.ge(alarm)
    # A gap cannot establish a new event onset; that transition is censored.
    continuous = actual.index.to_series().diff().eq(pd.Timedelta(hours=1))
    starts = over & ~over.shift(1, fill_value=False) & continuous
    leads = []
    count = 0
    for start in actual.index[starts]:
        if start - actual.index[0] < pd.Timedelta(hours=horizon):
            continue  # Left-censored onset: insufficient pre-event forecast window.
        previous = pd.date_range(start - pd.Timedelta(hours=horizon), start - pd.Timedelta(hours=1), freq="h")
        if not previous.isin(actual.index).all():
            continue  # A gap in the pre-event window makes warning time unknown.
        count += 1
        end = start
        while end + pd.Timedelta(hours=1) in actual.index and bool(over.loc[end + pd.Timedelta(hours=1)]):
            end += pd.Timedelta(hours=1)
        targets = pred.index[(pred.index >= start) & (pred.index <= end)]
        issue_times = targets - pd.Timedelta(hours=horizon)
        detected = issue_times[(issue_times < start) & pred.reindex(targets).ge(alarm).to_numpy()]
        if len(detected):
            leads.append(float((start - detected.min()).total_seconds() / 3600))
    return {"exceedance_events": count, "event_recall": _rate(len(leads), count),
            "median_warning_hours": float(np.median(leads)) if leads else None}
