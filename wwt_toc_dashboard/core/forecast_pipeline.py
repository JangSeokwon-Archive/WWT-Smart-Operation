"""Shared, causal preprocessing for training and live forecasts.

The index denotes sample time. Target history is available only after the
configured reporting delay; labels always refer to sample time + horizon.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

PIPELINE_VERSION = "wwt-causal-v2.1"
PREFIXES = ("EQ_", "DAF_", "AERA_", "AERB_", "CLAA_", "CLAB_")
TARGET = "FINAL_TOC"


@dataclass(frozen=True)
class PipelineConfig:
    sampling_minutes: int = 60
    target_delay_hours: int = 1
    max_fill_hours: int = 3
    do_low_threshold: float = 1.0
    # Additional reporting lags must be supplied from actual instrument records.
    sensor_delays: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        if self.sampling_minutes != 60:
            raise ValueError("Only verified hourly sampling is supported")
        if self.target_delay_hours < 0 or self.max_fill_hours < 0:
            raise ValueError("Delays and fill limits must be nonnegative")
        if any(delay < 0 for _, delay in self.sensor_delays):
            raise ValueError("Sensor delays must be nonnegative")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def hourly_observations(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("A DatetimeIndex with sample timestamps is required")
    if df.index.hasnans or df.empty:
        raise ValueError("No valid timestamped data")
    if (df.index != df.index.floor("h")).any():
        raise ValueError("Off-hour timestamps need an explicit aggregation policy")
    df = df.sort_index(kind="stable")
    df = df.loc[~df.index.duplicated(keep="last")]
    cols = [c for c in df if c == TARGET or c in ("FLOW", "TEMP") or c.startswith(PREFIXES)]
    df = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if TARGET not in df:
        raise ValueError("Missing FINAL_TOC")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    for col in df:
        s = df[col]
        if col.endswith("_pH"):
            df[col] = s.where(s.between(0, 14))
        elif col != "TEMP":
            df[col] = s.where(s >= 0)
    return df


def available_inputs(observed: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = observed.copy()
    delays = dict(cfg.sensor_delays)
    for col in out:
        delay = cfg.target_delay_hours if col == TARGET else delays.get(col, 0)
        out[col] = out[col].shift(delay)
    return out


def fill_inputs(available: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if cfg.max_fill_hours == 0:
        return available.copy()
    return available.ffill(limit=cfg.max_fill_hours)


def build_features(raw: pd.DataFrame, cfg: PipelineConfig = PipelineConfig(), *, process: bool = True) -> pd.DataFrame:
    observed = hourly_observations(raw)
    available = available_inputs(observed, cfg)
    inputs = fill_inputs(available, cfg)
    features: dict[str, pd.Series] = {}
    for col, s in inputs.items():
        name = "TOC_AVAILABLE" if col == TARGET else col
        features[name] = s
        features[f"{name}__missing"] = available[col].isna().astype(float)
        last_seen = available.index.to_series().where(available[col].notna()).ffill()
        features[f"{name}__age_h"] = (available.index.to_series() - last_seen).dt.total_seconds() / 3600
        for lag in (1, 3, 6, 12, 24, 48, 72):
            features[f"{name}__lag{lag}"] = s.shift(lag)
        for window in (6, 24, 72):
            features[f"{name}__mean{window}"] = s.rolling(window, min_periods=window).mean()
        for lag in (3, 12, 24):
            features[f"{name}__diff{lag}"] = s.diff(lag)
        if process:
            past = s.shift(1)
            median = past.rolling(24, min_periods=12).median()
            window = past.rolling(24, min_periods=12)
            scale = (window.quantile(0.75) - window.quantile(0.25)) / 1.349
            features[f"{name}__shock"] = (s - median) / (scale + 1e-6)
    if process:
        if {"FLOW", "EQ_TOC"}.issubset(inputs):
            # Units are not verified. This is a load proxy, not a kg/h estimate.
            features["organic_load_proxy"] = inputs.FLOW * inputs.EQ_TOC
        for before, after, name in (("EQ_TOC", "DAF_TOC", "DAF"), ("DAF_TOC", "AERA_TOC", "AERA"), ("DAF_TOC", "AERB_TOC", "AERB")):
            if before in inputs and after in inputs:
                delta = inputs[before] - inputs[after]
                features[f"{name}__toc_drop"] = delta
                features[f"{name}__apparent_removal"] = delta / inputs[before].where(inputs[before] > 0)
        for suffix in ("TOC", "DO", "MLSS", "RET", "WIT", "SRT", "SVI", "FM"):
            a, b = f"AERA_{suffix}", f"AERB_{suffix}"
            if a in inputs and b in inputs:
                features[f"AER_{suffix}__difference"] = inputs[a] - inputs[b]
        for prefix in ("AERA", "AERB"):
            do, mlss = f"{prefix}_DO", f"{prefix}_MLSS"
            if do in inputs:
                s = inputs[do]
                low = s.lt(cfg.do_low_threshold) & s.notna()
                groups = (~low).cumsum()
                run = low.astype(float).groupby(groups).cumsum()
                features[f"{prefix}__low_do_run_h"] = run.where(s.notna())
            if mlss in inputs:
                s = inputs[mlss]
                features[f"{prefix}__mlss_change24"] = s.diff(24) / s.shift(24).where(s.shift(24) > 0)
                features[f"{prefix}__mlss_trend72"] = s.rolling(72, min_periods=72).mean() - s.rolling(24, min_periods=24).mean()
    return pd.DataFrame(features, index=observed.index).replace([np.inf, -np.inf], np.nan)


def make_labels(raw: pd.DataFrame, horizon: int) -> pd.Series:
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    # Never impute the truth, even if input sensors have bounded forward fills.
    return hourly_observations(raw)[TARGET].shift(-horizon).rename("target")


def quality_status(raw: pd.DataFrame, cfg: PipelineConfig = PipelineConfig()) -> dict:
    observed = hourly_observations(raw)
    available = available_inputs(observed, cfg)
    missing = float(available.tail(24).isna().mean().mean())
    latest = available.iloc[-1]
    return {
        "recent_missing_fraction": missing,
        "latest_missing_columns": latest.index[latest.isna()].tolist(),
        "status": "degraded" if latest.isna().any() or missing > 0.1 else "ok",
        "target_reporting_delay_hours": cfg.target_delay_hours,
        "sensor_availability": "zero additional lag assumed unless configured; verify instrument timestamps",
    }
