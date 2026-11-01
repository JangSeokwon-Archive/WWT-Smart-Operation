"""Add a local-only V2.1 score and error-distribution audit to V2.2 results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def run(folder: Path, prior: Path):
    result = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    previous = json.loads((prior / "report.json").read_text(encoding="utf-8"))
    rows, diagnostics = [], {}
    for h, detail in previous["horizons"].items():
        data = pd.read_csv(prior / f"private_predictions_{h}h.csv", index_col=0, parse_dates=True)
        chosen = detail["selected_by_selection_mae"]
        for name, m in detail["models"].items():
            ci = m.get("bootstrap_vs_persistence") or {}
            rows.append({"horizon": h, "model": name, "mae": m["mae"], "rmse": m["rmse"], "bias": m["bias"],
                         "improvement_pct": m["improvement_vs_persistence_pct"], "bootstrap_ci": ci.get("ci95"), "coverage": m["interval_coverage"]})
        error = data[chosen] - data.actual
        diagnostics[h] = {"selected": chosen, "error_quantiles_05_50_95": np.quantile(error, [.05, .5, .95]).tolist(),
                          "normal_mae": detail["models"][chosen]["normal_mae"], "high_load_mae": detail["models"][chosen]["high_load_mae"],
                          "high_load_n": detail["models"][chosen]["high_load_n"], "shock_mae": detail["models"][chosen]["shock_mae"],
                          "shock_n": detail["models"][chosen]["shock_n"], "exceedance_n": detail["models"][chosen]["exceedance_n"]}
    result["v21_historical_performance"] = rows
    result["v21_error_diagnostics"] = diagnostics
    (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report_path = folder / "RESULTS.md"
    text = report_path.read_text(encoding="utf-8")
    marker = "\n## V2.1 Historical Model Comparison\n"
    text = text.split(marker)[0]
    text += marker + "\nExisting, already-reviewed historical predictions; not prospective validation.\n\n"
    text += "| Horizon | Model | MAE | RMSE | Bias | Persistence improvement | Bootstrap CI | Coverage |\n|---|---|---:|---:|---:|---:|---|---:|\n"
    for r in rows:
        text += f"| {r['horizon']}h | {r['model']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['bias']:.4f} | {r['improvement_pct']:.2f}% | {r['bootstrap_ci']} | {r['coverage']:.1%} |\n"
    text += "\n### Selected-Model Error And Regime Diagnostics\n\n```json\n" + json.dumps(diagnostics, indent=2) + "\n```\n"
    report_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("prior", type=Path)
    args = parser.parse_args()
    run(args.folder, args.prior)
