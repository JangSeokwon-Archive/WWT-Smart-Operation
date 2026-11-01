"""Statistical diagnostics; all outputs require independent interpretation."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def bootstrap_sensitivity(y, reference, candidate, index, *, blocks=(24, 48, 72, 168), draws=1000, seed=42, min_independent=5):
    delta = np.abs(np.asarray(y, float) - np.asarray(reference, float)) - np.abs(np.asarray(y, float) - np.asarray(candidate, float))
    idx = pd.DatetimeIndex(index)
    finite = np.isfinite(delta)
    delta, idx = delta[finite], idx[finite]
    if idx.has_duplicates or not idx.is_monotonic_increasing:
        raise ValueError("Bootstrap requires unique, time-ordered forecasts")
    breaks = np.r_[0, np.flatnonzero(np.diff(idx.asi8) != pd.Timedelta(hours=1).value) + 1, len(idx)]
    segments = [np.arange(a, b) for a, b in zip(breaks[:-1], breaks[1:])]
    result = {}
    for hours in blocks:
        independent = sum(len(s) // hours for s in segments)
        valid = [segment[i:i+hours] for segment in segments for i in range(max(0, len(segment)-hours+1))]
        item = {"n": len(delta), "mean_mae_reduction": float(delta.mean()) if len(delta) else None,
                "valid_overlapping_blocks": len(valid), "nonoverlap_blocks": independent,
                "positive_block_fraction": float(np.mean([delta[b].mean() > 0 for b in valid])) if valid else None,
                "status": "insufficient" if independent < min_independent else "diagnostic"}
        if independent >= min_independent and valid:
            rng = np.random.default_rng(seed)
            samples = []
            for _ in range(draws):
                choices = rng.integers(0, len(valid), size=math.ceil(len(delta) / hours))
                samples.append(float(np.concatenate([delta[valid[j]] for j in choices])[:len(delta)].mean()))
            item["ci95"] = np.quantile(samples, [0.025, 0.975]).tolist()
            item["bootstrap_positive_fraction"] = float(np.mean(np.array(samples) > 0))
        else:
            item["ci95"] = None
            item["bootstrap_positive_fraction"] = None
        result[str(hours)] = item
    series = pd.Series(delta, index=idx).asfreq("h") if len(idx) else pd.Series(dtype=float)
    autocorrelation = {}
    for lag in (1, 24, 48, 72, 168):
        value = series.autocorr(lag=lag) if len(series) > lag + 1 else np.nan
        autocorrelation[str(lag)] = float(value) if np.isfinite(value) else None
    return {"blocks": result, "error_difference_autocorrelation": autocorrelation,
            "note": "Sensitivity diagnostic only; overlapping blocks are not independent observations."}


def coverage_sample_plan(*, coverage=0.9, margin=0.05, z=1.96):
    if not 0 < coverage < 1 or not 0 < margin < 1:
        raise ValueError("Coverage and margin must be in (0,1)")
    n = math.ceil(z * z * coverage * (1 - coverage) / (margin * margin))
    return {"nominal_coverage": coverage, "margin": margin, "independent_samples_minimum": n,
            "hours_at_one_independent_label_per_hour": n,
            "event_recall_independent_events_for_plus_minus_10pct_worst_case": math.ceil(z * z * .25 / .1**2),
            "event_observation_time": "unknown without an independently estimated event rate",
            "limitation": "Normal approximation; overlapping horizons, autocorrelation and event scarcity increase the real requirement."}
