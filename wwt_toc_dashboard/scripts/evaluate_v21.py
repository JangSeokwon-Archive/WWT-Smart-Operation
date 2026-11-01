"""Sealed historical evaluation of delta and persistence hybrid forecasts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data import load_raw_csv
from core.forecast_pipeline import PipelineConfig, PIPELINE_VERSION, build_features, make_labels, hourly_observations
from core.forecast_training import fit_model
from core.forecast_evaluation import purged_folds, conformal_radius, regression_metrics, event_metrics
from core.modeling_v21 import blend, block_bootstrap_difference, fit_delta, fit_fixed_weight, fit_learned_gate, fit_regime_weights, regime_matrix, seasonal_baseline
from core.prediction_cache import file_digest


def metric(y, pred, base, high, alarm, shock, index, h):
    result = regression_metrics(y, pred, base, high, alarm, shock)
    result.update(event_metrics(index, y, pred, h, alarm))
    rise = np.asarray(y) > np.asarray(base)
    result["rising_n"] = int(rise.sum())
    result["rising_mae"] = float(np.abs(np.asarray(y)[rise] - np.asarray(pred)[rise]).mean()) if rise.any() else None
    return result


def lag_analysis(X, y, current):
    # Historical association only; labels are available in the training interval.
    output = {}
    change = y - current
    for c in ("FLOW", "EQ_TOC", "DAF_TOC", "AERA_DO", "AERB_DO", "AERA_MLSS", "AERB_MLSS", "AERA_SRT", "AERB_SRT"):
        if c not in X:
            continue
        rows = []
        for lag in (0, 1, 3, 6, 12, 24, 48, 72):
            feature = c if not lag else f"{c}__lag{lag}"
            if feature in X:
                corr = X[feature].corr(change)
                if np.isfinite(corr):
                    rows.append({"lag_hours": lag, "correlation": float(corr)})
        output[c] = rows
    return output


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new empty directory; preserve prior experiments")
    output.mkdir(parents=True, exist_ok=True)
    input_hash = file_digest(args.raw)
    previous = json.loads(Path(args.prior_report).read_text(encoding="utf-8"))
    if previous.get("input_sha256") != input_hash:
        raise ValueError("Prior V2 input hash differs; cannot establish an untouched historical window")
    raw = load_raw_csv(args.raw)
    # V2 used the final 180 days. This test precedes that window by 30 days and
    # is fixed using timestamps alone, before inspecting this test's outcomes.
    old_start = raw.index.max() - pd.Timedelta(days=int(previous["experiment"]["history_days"]))
    test_end = old_start - pd.Timedelta(days=30)
    test_start = test_end - pd.Timedelta(days=30)
    raw = raw.loc[(raw.index >= test_end - pd.Timedelta(days=args.history_days)) & (raw.index < test_end)]
    cfg = PipelineConfig(target_delay_hours=1)
    X = build_features(raw, cfg)
    report = {"input_sha256": input_hash, "pipeline_version": PIPELINE_VERSION, "config": asdict(cfg),
              "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "lightgbm", "scikit-learn", "xgboost")},
              "design": {"old_v2_window_start": str(old_start), "test_start": str(test_start), "test_end_exclusive": str(test_end),
                         "history_days": args.history_days, "estimators": args.estimators, "target_delay_hours": 1, "alarm": args.alarm,
                         "warning": "Retrospective test predates V2's 180-day experiment; earlier V1 provenance is unknown. No live prospective claim."},
              "horizons": {}}
    for h in (6, 12, 24):
        y = make_labels(raw, h)
        base = X["TOC_AVAILABLE"]
        seasonal = seasonal_baseline(hourly_observations(raw).FINAL_TOC, base, h, cfg.target_delay_hours)
        # Preserve a common, predeclared eligible sample across all candidates.
        valid = y.notna() & base.notna() & base.shift(24 - h).notna()
        idx = X.index[valid]
        gap = pd.Timedelta(hours=h + cfg.target_delay_hours)
        cal_start = test_start - gap - pd.Timedelta(days=14)
        select_start = cal_start - gap - pd.Timedelta(days=30)
        dev = idx[idx + gap < select_start]
        selection = idx[(idx >= select_start) & (idx + gap < cal_start)]
        calibration = idx[(idx >= cal_start) & (idx + gap < test_start)]
        test = idx[(idx >= test_start) & (idx + pd.Timedelta(hours=h) < test_end)]
        if min(len(dev), len(selection), len(calibration), len(test)) < 120:
            raise ValueError(f"Insufficient data for {h}h: {list(map(len, (dev, selection, calibration, test)))}")
        print(f"{h}h: {len(dev)} development, {len(selection)} selection, {len(calibration)} calibration, {len(test)} test", flush=True)
        oof = []
        folds = []
        selections = []
        for tr, va in purged_folds(dev, h, 1, 3):
            train, validation = dev[tr], dev[va]
            print(f"{h}h fold {len(folds)+1}", flush=True)
            dm = fit_delta(X.loc[train], y.loc[train], selected=True, estimators=args.estimators)
            p = dm.predict(X.loc[validation])
            oof.append(pd.DataFrame({"y": y.loc[validation], "base": base.loc[validation], "ml": p}, index=validation))
            folds.append({"train_end": str(train[-1]), "valid_start": str(validation[0]), "valid_end": str(validation[-1]),
                          "delta_mae": float(np.abs(y.loc[validation] - p).mean()),
                          "persistence_mae": float(np.abs(y.loc[validation] - base.loc[validation]).mean()),
                          "features": dm.columns})
            selections.append(set(dm.columns))
        oof = pd.concat(oof)
        load = X.get("organic_load_proxy", X["EQ_TOC"])
        load_threshold = float(load.loc[dev].quantile(0.9))
        shock = max(1.0, float((y.loc[dev] - base.loc[dev]).quantile(0.9)))
        fixed = fit_fixed_weight(oof.y, oof.base, oof.ml)
        stable_w, high_w = fit_regime_weights(oof.y, oof.base, oof.ml, load.loc[oof.index], load_threshold)
        gate = fit_learned_gate(X.loc[oof.index], oof.y, oof.base, oof.ml)

        def train_predictions(training, target):
            xx, yy = X.loc[training], y.loc[training]
            xx_eval = X.loc[target]
            direct = fit_model("process_lgbm", xx, yy, h, 1, args.estimators)
            delta = fit_delta(xx, yy, selected=False, estimators=args.estimators)
            compact = fit_delta(xx, yy, selected=True, estimators=args.estimators)
            residual = fit_model("lgbm_oof_residual", xx, yy, h, 1, args.estimators)
            xgb = fit_model("xgboost", xx, yy, h, 1, args.estimators)
            p = compact.predict(xx_eval)
            b = base.loc[target].to_numpy()
            high = load.loc[target].to_numpy() >= load_threshold
            missing_cols = [c for c in xx_eval if c.endswith("__missing")]
            safe = load.loc[target].notna().to_numpy() & (xx_eval[missing_cols].mean(axis=1).to_numpy() <= 0.2) & (xx_eval["TOC_AVAILABLE__missing"].to_numpy() == 0)
            hybrid_p = np.where(safe, p, np.nan)
            gated = np.clip(gate.predict(regime_matrix(xx_eval)), 0, 1) if gate is not None else np.zeros(len(target))
            preds = {"persistence": b, "seasonal_naive": seasonal.loc[target].to_numpy(),
                     "v2_lightgbm": direct.predict(xx_eval), "delta_lightgbm": delta.predict(xx_eval),
                     "delta_selected": p, "lgbm_oof_residual": residual.predict(xx_eval),
                     "xgboost": xgb.predict(xx_eval), "fixed_hybrid": blend(b, hybrid_p, fixed),
                     "regime_hybrid": blend(b, hybrid_p, np.where(high, high_w, stable_w)),
                     "learned_hybrid": blend(b, hybrid_p, gated)}
            return preds, compact

        select_preds, _ = train_predictions(dev, selection)
        scores = {name: float(np.abs(y.loc[selection].to_numpy() - pred).mean()) for name, pred in select_preds.items()}
        chosen = min(scores, key=scores.get)
        # Freeze selection before calibration and test; refit on past development + selection only.
        final_train = idx[idx + gap < cal_start]
        targets = calibration.append(test)
        preds, final_delta = train_predictions(final_train, targets)
        stats = {}
        base_test = base.loc[test].to_numpy()
        actual = y.loc[test].to_numpy()
        high = (load.loc[test] >= load_threshold).to_numpy()
        pred_table = pd.DataFrame(index=test)
        for name, prediction in preds.items():
            cal, pred = prediction[:len(calibration)], prediction[len(calibration):]
            met = metric(actual, pred, base_test, high, args.alarm, shock, test, h)
            met["selection_mae"] = scores[name]
            met["interval_radius"] = conformal_radius(y.loc[calibration], cal)
            met["interval_coverage"] = float(np.mean(np.abs(actual - pred) <= met["interval_radius"]))
            met["interval_width"] = 2 * met["interval_radius"]
            met["bootstrap_vs_persistence"] = block_bootstrap_difference(actual, base_test, pred, index=test, block=24, draws=500)
            stats[name] = met
            pred_table[name] = pred
        base_mae = stats["persistence"]["mae"]
        for m in stats.values():
            m["improvement_vs_persistence_pct"] = 100 * (base_mae - m["mae"]) / base_mae if base_mae else None
        # TreeSHAP and permutation diagnostics on an inner chronological held-out
        # part of development only, never on selection/calibration/test.
        inner_train = dev[dev + gap < dev[-1] - pd.Timedelta(days=14)]
        inner_valid = dev[dev >= dev[-1] - pd.Timedelta(days=14)]
        diagnostic = fit_delta(X.loc[inner_train], y.loc[inner_train], selected=True, estimators=args.estimators)
        sample = X.loc[inner_valid].tail(256)
        native_shap = diagnostic.model.predict(sample[diagnostic.columns], pred_contrib=True)
        shap_rank = sorted(zip(diagnostic.columns, np.abs(native_shap[:, :-1]).mean(axis=0).tolist()), key=lambda v: v[1], reverse=True)[:15]
        baseline_error = float(np.abs(y.loc[sample.index] - diagnostic.predict(sample)).mean())
        rng = np.random.default_rng(42)
        permutation = []
        for col, _ in shap_rank[:8]:
            if col == "TOC_AVAILABLE":
                continue  # Baseline and feature must remain physically consistent.
            shuffled = sample.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            permutation.append((col, float(np.abs(y.loc[sample.index] - diagnostic.predict(shuffled)).mean() - baseline_error)))
        pairwise = [len(a & b) / len(a | b) for i, a in enumerate(selections) for b in selections[i+1:] if a | b]
        selected_stats = stats[chosen]
        report["horizons"][str(h)] = {"folds": folds, "selection_start": str(selection[0]), "selection_end": str(selection[-1]),
            "calibration_start": str(calibration[0]), "calibration_end": str(calibration[-1]), "test_start": str(test[0]), "test_end": str(test[-1]),
            "selected_by_selection_mae": chosen, "test_n": len(test), "fixed_weight": fixed, "stable_weight": stable_w, "high_load_weight": high_w,
            "thresholds_training_only": {"high_load": load_threshold, "shock": shock}, "selection_mae": scores,
            "selected_feature_count": len(final_delta.columns), "total_feature_count": len(X.columns),
            "fold_feature_jaccard": pairwise, "training_association_by_lag": lag_analysis(X.loc[dev], y.loc[dev], base.loc[dev]),
            "inner_validation_shap": shap_rank, "inner_validation_permutation": permutation, "models": stats,
            "deployable": False, "deployment_reason": "Retrospective evaluation only; sensor reporting delays and independent prospective evaluation remain unverified."}
        pred_table["actual"] = actual
        pred_table["available"] = base_test
        pred_table.to_csv(output / f"private_predictions_{h}h.csv")
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if input_hash != file_digest(args.raw):
        raise RuntimeError("Input changed during experiment; discard results")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prior-report", required=True, help="Private V2 report.json for input hash and historical-window verification")
    ap.add_argument("--history-days", type=int, default=450)
    ap.add_argument("--estimators", type=int, default=120)
    ap.add_argument("--alarm", type=float, default=20.)
    args = ap.parse_args()
    report = run(args)
    print(json.dumps({h: {"selected": x["selected_by_selection_mae"], "models": {n: {k: m[k] for k in ("selection_mae", "mae", "rmse", "bias", "improvement_vs_persistence_pct")} for n, m in x["models"].items()}} for h, x in report["horizons"].items()}, indent=2))


if __name__ == "__main__":
    main()
