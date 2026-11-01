"""Produce a PRIVATE report and finalize candidate-artifact validation metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(folder: Path) -> None:
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    lines = ["# V2 Private Evaluation", "", "Do not publish this file or the underlying process-derived outputs.", "",
             "V1 frozen model future performance is unverified. The reference below is a causal refit of recovered V1 features/settings with 300 trees.", "",
             "Candidates were selected using development walk-forward MAE before reading final test metrics. No model is automatically promoted.", "",
             "| Horizon | Reference MAE | Selected MAE | Reference RMSE | Selected RMSE | MAE improvement | Selected model | Persistence MAE |", "|---|---|---|---|---|---|---|---|"]
    for h, detail in report["horizons"].items():
        models = detail["models"]
        chosen = detail["selected_by_validation"]
        reference, selected = models["legacy_feature_lgbm_refit"], models[chosen]
        lines.append(f"| {h}h | {reference['mae']:.4f} | {selected['mae']:.4f} | {reference['rmse']:.4f} | {selected['rmse']:.4f} | {selected['mae_improvement_vs_legacy_refit_pct']:+.2f}% | {chosen} | {models['persistence']['mae']:.4f} |")
        passed = bool(detail["selection_beats_persistence_on_validation"] and selected["mae"] < models["persistence"]["mae"] and selected["interval_test_coverage"] >= 0.85)
        detail["deployment_gate_passed"] = passed
        detail["deployment_gate"] = "requires validation and test MAE better than persistence and >=85% empirical coverage for nominal 90% interval; does not establish prospective/causal validity"
        path = folder / f"forecast_{h}h.joblib"
        if path.exists():
            bundle = joblib.load(path)
            model = bundle["model"]
            bundle["training_params"] = {"base": model.base.get_params(), "correction": model.correction.get_params()} if hasattr(model, "correction") else model.get_params()
            bundle["deployment_gate_passed"] = passed
            bundle["interval_calibration_passed"] = selected["interval_test_coverage"] >= 0.85
            joblib.dump(bundle, path)
    lines += ["", "## Regimes And Uncertainty", "", "| Horizon | Selected normal MAE | High-load MAE (n) | Shock MAE (n) | Rise recall | 90% interval test coverage | Deployment gate |", "|---|---|---|---|---|---|---|"]
    for h, detail in report["horizons"].items():
        m = detail["models"][detail["selected_by_validation"]]
        lines.append(f"| {h}h | {m['normal_mae']} | {m['high_load_mae']} ({m['high_load_n']}) | {m['shock_mae']} ({m['shock_n']}) | {m['rise_detection_rate']} | {m['interval_test_coverage']:.1%} | {'pass' if detail['deployment_gate_passed'] else 'FAIL'} |")
    if all(detail["models"][detail["selected_by_validation"]]["exceedance_n"] == 0 for detail in report["horizons"].values()):
        lines += ["", "No threshold-exceedance events occurred in this final test: exceedance-event recall and early warning time are unverified, not zero or perfect."]
    lines += ["", "## All Models", "", "| Horizon | Model | Validation MAE | Test MAE | RMSE | Bias |", "|---|---|---|---|---|---|"]
    for h, detail in report["horizons"].items():
        for name, m in detail["models"].items():
            lines.append(f"| {h}h | {name} | {m['validation_mae']:.4f} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['bias']:.4f} |")
    lines += ["", "## Guard Comparison", "", "```json", json.dumps(report.get("guard_comparison"), indent=2), "```", "",
              "## Reproduction", "", "```json", json.dumps({"experiment": report["experiment"], "packages": report["packages"], "pipeline_config": report["pipeline_config"], "input_sha256": report["input_sha256"]}, indent=2), "```", "",
              "The final window is a retrospective holdout under stated timestamp assumptions, not a live prospective deployment. Do not tune on it after reviewing these results. Collect a new forward test before promotion.", ""]
    (folder / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
