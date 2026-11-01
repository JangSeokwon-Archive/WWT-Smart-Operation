"""Run private, reproducible model comparisons. Never promotes a model by itself."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.automl_infer import _guard_24h_jump, _legacy_frame
from core.data import load_raw_csv
from core.forecast_evaluation import conformal_radius, event_metrics, purged_folds, regression_metrics
from core.forecast_pipeline import PIPELINE_VERSION, PipelineConfig, build_features, hourly_observations, make_labels
from core.forecast_training import fit_model, lgbm_params
from core.prediction_cache import file_digest


def run(args) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(target_delay_hours=args.target_delay)
    raw = load_raw_csv(args.raw)
    raw = raw.loc[raw.index >= raw.index.max() - pd.Timedelta(days=args.history_days)]
    observed = hourly_observations(raw)
    temporal = build_features(raw, cfg, process=False)
    improved = build_features(raw, cfg, process=True)
    packaged = Path(__file__).resolve().parents[1] / "model_bundle"
    legacy_cfg = yaml.safe_load((packaged / "configs" / "features.yaml").read_text(encoding="utf-8"))
    old_frame = _legacy_frame(observed, cfg, legacy_cfg).set_index("날짜")
    legacy = old_frame.drop(columns=["FINAL_TOC", "FINAL_TOC_raw", "FINAL_TOC_is_bad", "FINAL_TOC_is_maint"], errors="ignore")
    # Frozen original bundles lack training cutoffs. They cannot be claimed to
    # be out-of-sample on historical rows. Refit original settings instead.
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "PyYAML")}
    models = ["persistence", "seasonal_naive", "legacy_feature_lgbm_refit", "temporal_lgbm", "process_lgbm", "lgbm_oof_residual"]
    if importlib.util.find_spec("xgboost"):
        models.append("xgboost")
        versions["xgboost"] = importlib.metadata.version("xgboost")
    report = {
        "pipeline_version": PIPELINE_VERSION, "pipeline_config": asdict(cfg), "packages": versions,
        "created_at": datetime.now(timezone.utc).isoformat(), "input_sha256": file_digest(args.raw),
        "experiment": vars(args), "seed": 42,
        "notes": ["Original V1 bundle out-of-sample accuracy is unverified: training cutoff/hash absent.",
                  "V1 feature/parameter refit uses causal preprocessing and capped estimator count, not an exact historical reproduction.",
                  "All compared models use identical origins and untouched observed labels.",
                  "Sensor reporting lag is assumed zero except target delay; verify sample/report timestamps before operational use.",
                  "FLOW * EQ_TOC is a proxy; units and hydraulic residence volumes are not verified.",
                  "Calibration residual intervals have empirical time-series coverage only.",
                  "No artifact is automatically promoted; no plant commands are implemented."],
        "unverified": ["calibrated_exceedance_probability", "hydraulic_residence_features", "causal_control_effects"],
        "horizons": {},
    }
    if "xgboost" not in models:
        report["unverified"].append("catboost_or_xgboost: package unavailable")
    test_start = observed.index.max() - pd.Timedelta(days=args.test_days)
    for horizon in (6, 12, 24):
        gap = pd.Timedelta(hours=horizon + cfg.target_delay_hours)
        calibration_start = test_start - gap - pd.Timedelta(days=args.calibration_days)
        y_all = make_labels(raw, horizon)
        # Require observed naive forecasts on the same rows for fair comparison.
        index = improved.index.intersection(legacy.index)
        current = temporal["TOC_AVAILABLE"]
        seasonal = current.shift(24 - horizon)
        good = y_all.reindex(index).notna() & current.reindex(index).notna() & seasonal.reindex(index).notna()
        index = index[good]
        dev = index[index + gap < calibration_start]
        calibration = index[(index >= calibration_start) & (index + gap < test_start)]
        test = index[index >= test_start]
        if len(dev) < 700 or len(calibration) < 100 or len(test) < 100:
            raise ValueError(f"Insufficient disjoint data: horizon={horizon}, dev={len(dev)}, cal={len(calibration)}, test={len(test)}")
        legacy_path = packaged / "models" / f"toc_lgbm_lead{horizon}h_baseline.joblib"
        legacy_bundle = joblib.load(legacy_path)
        legacy_params = legacy_bundle.get("params", {})
        old_columns = legacy_bundle["columns"]
        if set(old_columns) - set(legacy.columns):
            raise ValueError("Cannot reconstruct legacy feature schema")
        matrices = {name: improved for name in models}
        matrices["temporal_lgbm"] = temporal
        matrices["legacy_feature_lgbm_refit"] = legacy[old_columns]
        folds = list(purged_folds(dev, horizon, cfg.target_delay_hours, args.folds))
        validation = {name: [] for name in models}
        fold_metadata = []
        for fold_id, (train, valid) in enumerate(folds):
            tr, va = dev[train], dev[valid]
            fold_metadata.append({"train_start": str(tr[0]), "train_end": str(tr[-1]), "latest_train_label_available": str(tr[-1] + gap),
                                  "valid_start": str(va[0]), "valid_end": str(va[-1]), "n_train": len(tr), "n_valid": len(va)})
            for name in models:
                print(f"h={horizon} fold={fold_id + 1}/{args.folds} model={name}", flush=True)
                if name in ("persistence", "seasonal_naive"):
                    prediction = (current if name == "persistence" else seasonal).loc[va].to_numpy()
                else:
                    X = matrices[name]
                    model = fit_model(name, X.loc[tr], y_all.loc[tr], horizon, cfg.target_delay_hours, args.estimators, legacy_params)
                    prediction = model.predict(X.loc[va])
                validation[name].append({"mae": float(np.abs(prediction - y_all.loc[va].to_numpy()).mean()), "n": len(va)})
        scores = {name: float(sum(row["mae"] * row["n"] for row in rows) / sum(row["n"] for row in rows)) for name, rows in validation.items()}
        # V2 deployment candidates share build_features(process=True). Naive and
        # original-feature results remain benchmarks, not hidden fallback models.
        eligible = [name for name in ("process_lgbm", "lgbm_oof_residual", "xgboost") if name in scores]
        selected = min(eligible, key=scores.get)
        load = improved.get("organic_load_proxy", current * np.nan)
        high_threshold = float(load.loc[dev].quantile(0.9))
        shock = max(1.0, float((y_all.loc[dev] - current.loc[dev]).quantile(0.9)))
        results = {}
        predictions = {}
        selected_bundle = None
        for name in models:
            print(f"h={horizon} final model={name}", flush=True)
            if name in ("persistence", "seasonal_naive"):
                series = current if name == "persistence" else seasonal
                cal_pred, test_pred = series.loc[calibration].to_numpy(), series.loc[test].to_numpy()
                model = None
            else:
                X = matrices[name]
                model = fit_model(name, X.loc[dev], y_all.loc[dev], horizon, cfg.target_delay_hours, args.estimators, legacy_params)
                cal_pred, test_pred = model.predict(X.loc[calibration]), model.predict(X.loc[test])
            metrics = regression_metrics(y_all.loc[test], test_pred, current.loc[test], load.loc[test] >= high_threshold, args.alarm, shock)
            metrics.update(event_metrics(test, y_all.loc[test], test_pred, horizon, args.alarm))
            radius = conformal_radius(y_all.loc[calibration], cal_pred)
            metrics["interval_radius"] = radius
            metrics["interval_test_coverage"] = float((np.abs(y_all.loc[test].to_numpy() - test_pred) <= radius).mean())
            metrics["validation_mae"] = scores[name]
            results[name] = metrics
            predictions[name] = test_pred
            if name == selected:
                selected_bundle = {"model": model, "columns": list(matrices[name].columns), "horizon": horizon,
                                   "pipeline_version": PIPELINE_VERSION, "pipeline_config": asdict(cfg), "config_digest": cfg.digest(),
                                   "trained_through": str(dev[-1]), "interval_radius": radius, "interval_coverage": 0.9,
                                   "interval_test_coverage": metrics["interval_test_coverage"], "model_name": name,
                                   "input_sha256": report["input_sha256"], "training_params": lgbm_params(args.estimators)}
        if args.save_models:
            joblib.dump(selected_bundle, output / f"forecast_{horizon}h.joblib")
        reference = results["legacy_feature_lgbm_refit"]["mae"]
        for metrics in results.values():
            metrics["mae_improvement_vs_legacy_refit_pct"] = float((reference - metrics["mae"]) / reference * 100) if reference else None
        report["horizons"][str(horizon)] = {
            "n_features": {name: len(matrices[name].columns) for name in models if name not in ("persistence", "seasonal_naive")},
            "n_dev": len(dev), "n_calibration": len(calibration), "n_test": len(test), "folds": fold_metadata,
            "calibration_start": str(calibration[0]), "calibration_end": str(calibration[-1]), "test_start": str(test[0]), "test_end": str(test[-1]),
            "selected_by_validation": selected, "selection_beats_persistence_on_validation": scores[selected] < scores["persistence"],
            "high_load_threshold_train_only": high_threshold, "shock_threshold_train_only": shock,
            "models": results, "validation_folds": validation,
        }
        table = pd.DataFrame(predictions, index=test)
        table["actual"] = y_all.loc[test]
        table["current_available"] = current.loc[test]
        table.to_csv(output / f"private_predictions_{horizon}h.csv")
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    # Evaluate the V1 guard on matched, freshly refitted V2 predictions, not on
    # pre-trained V1 models with unknown overlap with the evaluation period.
    frames = {h: pd.read_csv(output / f"private_predictions_{h}h.csv", index_col=0, parse_dates=True) for h in (6, 12, 24)}
    shared = frames[6].index.intersection(frames[12].index).intersection(frames[24].index)
    for name in ("process_lgbm", "lgbm_oof_residual"):
        frame = frames[24].loc[shared]
        guarded = [_guard_24h_jump({f"pred_t{h}": float(frames[h].loc[t, name]) for h in (6, 12, 24)}, float(frame.loc[t, "current_available"]))["pred_t24"] for t in shared]
        report.setdefault("guard_comparison", {})[name] = {
            "n": len(shared), "raw_mae": float(np.abs(frame[name] - frame.actual).mean()),
            "guarded_mae": float(np.abs(np.asarray(guarded) - frame.actual).mean()),
            "raw_exceedance_misses": int(((frame.actual >= args.alarm) & (frame[name] < args.alarm)).sum()),
            "guarded_exceedance_misses": int(((frame.actual >= args.alarm) & (np.asarray(guarded) < args.alarm)).sum()),
        }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", default=".local/v2")
    parser.add_argument("--history-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=14)
    parser.add_argument("--calibration-days", type=int, default=14)
    parser.add_argument("--target-delay", type=int, default=1)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--alarm", type=float, default=20.0)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if min(args.history_days, args.test_days, args.calibration_days, args.estimators) <= 0:
        parser.error("Time windows and estimator count must be positive")
    report = run(args)
    from summarize_v2 import summarize
    summarize(Path(args.output))
    print(json.dumps({h: {"selected": r["selected_by_validation"], "models": {n: {k: v for k, v in m.items() if k in ("mae", "rmse", "bias", "mae_improvement_vs_legacy_refit_pct")} for n, m in r["models"].items()}} for h, r in report["horizons"].items()}, indent=2))


if __name__ == "__main__":
    main()
