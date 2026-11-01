"""Create a private diagnosis from a completed sealed historical experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data import load_raw_csv
from core.forecast_pipeline import PipelineConfig, build_features, make_labels
from core.prediction_cache import file_digest


def fmt(value, precision=4):
    return "unavailable" if value is None or not np.isfinite(value) else f"{value:.{precision}f}"


def run(args):
    folder = Path(args.folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if set(report["horizons"]) != {"6", "12", "24"}:
        raise ValueError("The sealed experiment has not completed all horizons")
    if file_digest(args.raw) != report["input_sha256"]:
        raise ValueError("Input hash differs from evaluation")
    raw = load_raw_csv(args.raw)
    end = pd.Timestamp(report["design"]["test_end_exclusive"])
    raw = raw.loc[(raw.index >= end - pd.Timedelta(days=report["design"]["history_days"])) & (raw.index < end)]
    X = build_features(raw, PipelineConfig())
    lines = ["# V2.1 Private Model Evaluation", "", "Contains process-derived information. Do not publish.", "",
             "The final test interval was selected by timestamps before inspecting its outcomes; it precedes the V2 180-day experiment. It is a historical retrospective interval, not live prospective validation. V1 training provenance is unknown.", "",
             f"Test window: {report['design']['test_start']} to {report['design']['test_end_exclusive']} (exclusive).", "",
             report["design"].get("seasonal_correction", "Seasonal benchmark was evaluated using daily-clock alignment."), "",
             "## Model Comparison", "", "| Horizon | Model | Selection MAE | Test MAE | RMSE | Bias | vs Persistence |", "|---|---|---:|---:|---:|---:|---:|"]
    for h, detail in report["horizons"].items():
        for name, m in detail["models"].items():
            lines.append(f"| {h}h | {name} | {fmt(m['selection_mae'])} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['bias'])} | {fmt(m['improvement_vs_persistence_pct'],2)}% |")
    lines += ["", "## Chosen Model And Regimes", "", "| Horizon | Selection winner | Normal MAE | High-load MAE (n) | Rising MAE (n) | Shock MAE (n) | Shock recall | Interval coverage | Interval width |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for h, d in report["horizons"].items():
        m = d["models"][d["selected_by_selection_mae"]]
        lines.append(f"| {h}h | {d['selected_by_selection_mae']} | {fmt(m['normal_mae'])} | {fmt(m['high_load_mae'])} ({m['high_load_n']}) | {fmt(m['rising_mae'])} ({m['rising_n']}) | {fmt(m['shock_mae'])} ({m['shock_n']}) | {fmt(m['rise_detection_rate'])} | {fmt(m['interval_coverage'])} | {fmt(m['interval_width'])} |")
    lines += ["", "## Diagnosis By Horizon", ""]
    for hs, d in report["horizons"].items():
        h = int(hs)
        y, base = make_labels(raw, h), X.TOC_AVAILABLE
        dev_end = pd.Timestamp(d["folds"][-1]["valid_end"])
        dev_mask = (X.index <= dev_end) & y.notna() & base.notna()
        test_frame = pd.read_csv(folder / f"private_predictions_{h}h.csv", index_col=0, parse_dates=True)
        train_base = base.loc[dev_mask]
        train_delta = (y - base).loc[dev_mask]
        test_delta = test_frame.actual - test_frame.available
        selected = d["selected_by_selection_mae"]
        m = d["models"][selected]
        lines += [f"### {h}h", "", f"Selected by selection-period MAE: **{selected}**. Holdout improvement vs persistence: {fmt(m['improvement_vs_persistence_pct'],2)}%. Deployment: **withheld**.",
                  f"Train available TOC mean/std: {fmt(train_base.mean())}/{fmt(train_base.std())}; test mean/std: {fmt(test_frame.available.mean())}/{fmt(test_frame.available.std())}.",
                  f"Train absolute change median/90th: {fmt(train_delta.abs().median())}/{fmt(train_delta.abs().quantile(.9))}; test: {fmt(test_delta.abs().median())}/{fmt(test_delta.abs().quantile(.9))}.",
                  f"Direct-model bias: {fmt(d['models']['v2_lightgbm']['bias'])}; direct/OOF-residual MAE: {fmt(d['models']['v2_lightgbm']['mae'])}/{fmt(d['models']['lgbm_oof_residual']['mae'])}.",
                  f"Feature counts: selected {d['selected_feature_count']} of {d['total_feature_count']}; fold feature Jaccard: {d['fold_feature_jaccard']}.",
                  f"Hybrid weights: fixed={d['fixed_weight']}, low-load={d['stable_weight']}, high-load={d['high_load_weight']}.",
                  f"Inner-development TreeSHAP: {d['inner_validation_shap']}; permutation MAE increases: {d['inner_validation_permutation']}.",
                  f"Exceedance n={m['exceedance_n']}, recall={m['exceedance_recall']}, event recall={m['event_recall']}, false alarm rate={m['false_alarm_rate']}, warning hours={m['median_warning_hours']}.",
                  f"Paired 24-hour block bootstrap vs persistence: {m['bootstrap_vs_persistence']}.",
                  "Lag association (training data only; not an inferred physical retention time):"]
        for name, entries in d["training_association_by_lag"].items():
            if entries:
                peak = max(entries, key=lambda row: abs(row["correlation"]))
                lines.append(f"- {name}: {peak['lag_hours']}h, correlation {fmt(peak['correlation'])}")
        lines.append("")
    lines += ["## Interpretation Limits", "", "Correlations are observational and cannot prove causality or establish HRT. Reporting lags except TOC are unverified. Intervals have empirical rather than guaranteed time-series coverage. Zero event samples are unassessable. Baseline and fitted models are scored on identical available-label rows. Bootstrap intervals use contiguous hourly blocks.", "", "No trained model is promoted by this experiment. A new prospective test, verified reporting timestamps, subgroup safeguards and calibration review remain necessary.", ""]
    (folder / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--raw", required=True)
    run(parser.parse_args())
