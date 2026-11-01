"""Empirical interval methods using only outcomes available before an issue time."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def interval_for_issue(scored: pd.DataFrame, issue_at, horizon: int, method: str, *, model="persistence", model_sha=None, window_days=30, min_samples=30):
    if method not in {"fixed", "rolling", "adaptive"}:
        raise ValueError("Unknown calibration method")
    issue = pd.Timestamp(issue_at)
    if issue.tzinfo is None:
        raise ValueError("Explicit timezone required")
    if scored.empty:
        return None
    rows = scored[(scored.horizon == horizon) & (scored.model == model) & (scored.status == "scored")].copy()
    if model_sha is not None:
        rows = rows[rows.model_sha256 == model_sha]
    available = pd.to_datetime(rows.actual_available_at, utc=True)
    issued = pd.to_datetime(rows.issue_at, utc=True)
    rows = rows.loc[(available < issue) & (issued < issue)].copy()
    rows["available"] = pd.to_datetime(rows.actual_available_at, utc=True)
    if method in {"rolling", "adaptive"}:
        rows = rows[rows.available >= issue - pd.Timedelta(days=window_days)]
    rows = rows.sort_values(["available", "issue_at"])
    if len(rows) < min_samples:
        return None
    alpha = 0.1
    if method == "adaptive":
        # Only already-issued intervals with subsequently received truth can
        # update alpha; simultaneous labels are processed in issue order.
        for row in rows.itertuples():
            if pd.notna(row.interval_low) and pd.notna(row.interval_high):
                miss = not (row.interval_low <= row.actual <= row.interval_high)
                alpha = float(np.clip(alpha + 0.01 * (0.1 - float(miss)), 0.01, 0.3))
    errors = np.abs(rows.actual.to_numpy(dtype=float) - rows.prediction.to_numpy(dtype=float))
    rank = math.ceil((len(errors) + 1) * (1 - alpha))
    if rank > len(errors):
        return None
    radius = float(np.sort(errors)[rank - 1])
    return {"version": f"{method}-conformal-v1", "method": method, "radius": radius,
            "calibrated_through": rows.available.max().isoformat(), "n": len(rows), "alpha": alpha,
            "window_days": window_days, "coverage_claim": "empirical only; not validated for operation"}
