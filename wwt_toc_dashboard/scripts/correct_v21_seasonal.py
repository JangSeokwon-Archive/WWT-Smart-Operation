"""Correct deterministic seasonal baseline on existing frozen predictions.

This never retrains a model and chooses the winner using selection data only.
Original private outputs are retained unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data import load_raw_csv
from core.forecast_pipeline import PipelineConfig, build_features, make_labels, hourly_observations
from core.forecast_evaluation import conformal_radius, regression_metrics, event_metrics
from core.modeling_v21 import seasonal_baseline, block_bootstrap_difference
from core.prediction_cache import file_digest


def run(source: Path, output: Path, raw_path: str, prior_path: str, alarm: float):
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    prior = json.loads(Path(prior_path).read_text(encoding="utf-8"))
    if set(report["horizons"]) != {"6", "12", "24"}:
        raise ValueError("Incomplete experiment")
    if file_digest(raw_path) != report["input_sha256"] or prior.get("input_sha256") != report["input_sha256"]:
        raise ValueError("Input does not match the original and prior V2 reports")
    if "alarm" in report["design"] and report["design"]["alarm"] != alarm:
        raise ValueError("Alarm differs from the experiment")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Correction output must be a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    raw = load_raw_csv(raw_path)
    end = pd.Timestamp(report["design"]["test_end_exclusive"])
    raw = raw.loc[(raw.index >= end - pd.Timedelta(days=report["design"]["history_days"])) & (raw.index < end)]
    X, observed = build_features(raw, PipelineConfig()), hourly_observations(raw).FINAL_TOC
    for hs, detail in report["horizons"].items():
        h = int(hs)
        y, base = make_labels(raw, h), X.TOC_AVAILABLE
        eligible = y.notna() & base.notna() & base.shift(24-h).notna()
        idx = X.index[eligible]
        selection = idx[(idx >= pd.Timestamp(detail["selection_start"])) & (idx <= pd.Timestamp(detail["selection_end"]))]
        calibration = idx[(idx >= pd.Timestamp(detail["calibration_start"])) & (idx <= pd.Timestamp(detail["calibration_end"]))]
        table = pd.read_csv(source / f"private_predictions_{h}h.csv", index_col=0, parse_dates=True)
        test = table.index
        seasonal = seasonal_baseline(observed, base, h, 1)
        selection_mae = float(np.abs(y.loc[selection] - seasonal.loc[selection]).mean())
        detail["selection_mae"]["seasonal_naive"] = selection_mae
        detail["selected_by_selection_mae"] = min(detail["selection_mae"], key=detail["selection_mae"].get)
        pred = seasonal.loc[test].to_numpy()
        actual, baseline = table.actual.to_numpy(), table.available.to_numpy()
        load = X.get("organic_load_proxy", X.EQ_TOC)
        high = (load.loc[test] >= detail["thresholds_training_only"]["high_load"]).to_numpy()
        result = regression_metrics(actual, pred, baseline, high, alarm, detail["thresholds_training_only"]["shock"])
        result.update(event_metrics(test, actual, pred, h, alarm))
        rising = actual > baseline
        result["rising_n"] = int(rising.sum())
        result["rising_mae"] = float(np.abs(actual[rising] - pred[rising]).mean()) if rising.any() else None
        result["selection_mae"] = selection_mae
        radius = conformal_radius(y.loc[calibration], seasonal.loc[calibration])
        result["interval_radius"] = radius
        result["interval_coverage"] = float(np.mean(np.abs(actual - pred) <= radius))
        result["interval_width"] = radius * 2
        result["bootstrap_vs_persistence"] = block_bootstrap_difference(actual, baseline, pred, index=test, block=24, draws=500)
        base_mae = detail["models"]["persistence"]["mae"]
        result["improvement_vs_persistence_pct"] = float(100 * (base_mae - result["mae"]) / base_mae) if base_mae else None
        detail["models"]["seasonal_naive"] = result
        table["seasonal_naive"] = pred
        for name, metrics in detail["models"].items():
            metrics.update(event_metrics(test, actual, table[name].to_numpy(), h, alarm))
        table.to_csv(output / f"private_predictions_{h}h.csv")
    report["design"]["seasonal_correction"] = ("After the original evaluation had started, the deterministic seasonal benchmark and gap-censored event counting were corrected on the same rows. No model was retrained or tuned; selection scores were recomputed using the selection period only. Original outputs remain unchanged.")
    report["design"]["alarm"] = alarm
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--prior-report", required=True)
    parser.add_argument("--alarm", type=float, required=True, help="Exact alarm used by the original experiment")
    args = parser.parse_args()
    run(args.source, args.output, args.raw, args.prior_report, args.alarm)
