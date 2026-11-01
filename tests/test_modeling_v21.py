import numpy as np
import pandas as pd
import pytest
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from core.forecast_pipeline import build_features, make_labels
from core.forecast_evaluation import event_metrics
from core.modeling_v21 import blend, block_bootstrap_difference, fit_delta, fit_fixed_weight, fit_regime_weights, seasonal_baseline
from core.prediction_cache import file_digest


def test_delta_prediction_uses_available_not_future_toc(history):
    X, y = build_features(history), make_labels(history, 6)
    train = y.notna() & X.TOC_AVAILABLE.notna()
    model = fit_delta(X.loc[train], y.loc[train], estimators=5)
    later = X.loc[train].tail(5)
    direct_delta = model.model.predict(later[model.columns])
    np.testing.assert_allclose(model.predict(later), later.TOC_AVAILABLE + direct_delta)
    assert later.TOC_AVAILABLE.iloc[-1] == history.FINAL_TOC.loc[later.index[-1] - pd.Timedelta(hours=1)]


def test_hybrid_preserves_naive_with_missing_ml_and_bounds_weights():
    result = blend([10, 10, 10, 10], [np.nan, 14, 14, 14], [0.9, -2, 2, np.nan])
    np.testing.assert_allclose(result, [10, 10, 14, 10])


@pytest.mark.parametrize("h,lag", [(6, 18), (12, 12), (24, 24)])
def test_seasonal_uses_latest_available_same_clock_time(history, h, lag):
    observed = history.FINAL_TOC
    available = observed.shift(1)
    predicted = seasonal_baseline(observed, available, h, 1)
    assert predicted.iloc[300] == observed.iloc[300-lag]
    assert lag >= 1


def test_fixed_and_regime_weights_are_learned_from_supplied_history():
    y, base, ml = np.array([12.] * 100), np.array([10.] * 100), np.array([12.] * 100)
    assert fit_fixed_weight(y, base, ml) == pytest.approx(1)
    stable, high = fit_regime_weights(y, base, ml, np.arange(100), 50)
    assert stable == pytest.approx(1)
    assert high == pytest.approx(1)


def test_block_bootstrap_requires_contiguous_hours():
    y, base, candidate = np.ones(100), np.zeros(100), np.full(100, 0.5)
    index = pd.date_range("2025-01-01", periods=100, freq="2h")
    assert block_bootstrap_difference(y, base, candidate, index=index) is None
    index = pd.date_range("2025-01-01", periods=100, freq="h")
    result = block_bootstrap_difference(y, base, candidate, index=index, draws=20)
    assert result["ci95"] == pytest.approx([0.5, 0.5])


def test_missing_pre_event_hour_censors_onset():
    index = pd.date_range("2025-01-01", periods=100, freq="h")
    y = pd.Series(0.0, index=index)
    y.iloc[40:44] = 25
    missing_before = y.drop(index[39])
    result = event_metrics(missing_before.index, missing_before, missing_before, 12, 20)
    assert result["exceedance_events"] == 0
    assert result["event_recall"] is None
    missing_in_warning_window = y.drop(index[32])
    result = event_metrics(missing_in_warning_window.index, missing_in_warning_window, missing_in_warning_window, 12, 20)
    assert result["exceedance_events"] == 0


def test_feature_selection_does_not_use_future_labels(history):
    X, y = build_features(history), make_labels(history, 6)
    rows = y.notna() & X.TOC_AVAILABLE.notna()
    train_X, train_y = X.loc[rows].iloc[:300], y.loc[rows].iloc[:300]
    a = fit_delta(train_X, train_y, selected=True, estimators=5)
    changed = y.copy()
    changed.iloc[400:] = 1e9
    b = fit_delta(train_X, changed.loc[train_X.index], selected=True, estimators=5)
    assert a.columns == b.columns
    np.testing.assert_allclose(a.predict(train_X.tail(10)), b.predict(train_X.tail(10)))


def test_evaluation_rejects_input_different_from_prior_v2(tmp_path):
    path = Path(__file__).resolve().parents[1] / "wwt_toc_dashboard/scripts/evaluate_v21.py"
    spec = importlib.util.spec_from_file_location("evaluate_v21_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw, prior = tmp_path / "raw.csv", tmp_path / "prior.json"
    raw.write_text("synthetic")
    prior.write_text(json.dumps({"input_sha256": "not-the-same"}))
    args = SimpleNamespace(output=str(tmp_path / "output"), raw=str(raw), prior_report=str(prior))
    with pytest.raises(ValueError, match="input hash differs"):
        module.run(args)


def test_seasonal_correction_preserves_original_and_uses_selection_only(history, tmp_path):
    script = Path(__file__).resolve().parents[1] / "wwt_toc_dashboard/scripts/correct_v21_seasonal.py"
    spec = importlib.util.spec_from_file_location("correct_v21_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / "raw.csv"
    history.rename_axis("날짜").reset_index().to_csv(raw, index=False)
    digest = file_digest(raw)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"input_sha256": digest}))
    original, corrected = tmp_path / "original", tmp_path / "corrected"
    original.mkdir()
    X = build_features(history)
    report = {"input_sha256": digest, "design": {"test_end_exclusive": str(history.index[-1] + pd.Timedelta(hours=1)),
              "history_days": 30, "alarm": 20.0}, "horizons": {}}
    for h in (6, 12, 24):
        y, base = make_labels(history, h), X.TOC_AVAILABLE
        test = history.index[350:400]
        table = pd.DataFrame({"actual": y.loc[test], "available": base.loc[test], "persistence": base.loc[test],
                              "seasonal_naive": base.loc[test]}, index=test)
        table.to_csv(original / f"private_predictions_{h}h.csv")
        report["horizons"][str(h)] = {"selection_start": str(history.index[100]), "selection_end": str(history.index[199]),
            "calibration_start": str(history.index[230]), "calibration_end": str(history.index[300]),
            "selection_mae": {"persistence": 0.01, "seasonal_naive": 99.0}, "selected_by_selection_mae": "persistence",
            "thresholds_training_only": {"high_load": 99999.0, "shock": 1.0},
            "models": {"persistence": {"mae": 1.0}, "seasonal_naive": {}}}
    original_report = json.dumps(report)
    (original / "report.json").write_text(original_report)
    module.run(original, corrected, str(raw), str(prior), 20.)
    assert (original / "report.json").read_text() == original_report
    updated = json.loads((corrected / "report.json").read_text())
    assert updated["horizons"]["24"]["selected_by_selection_mae"] == "persistence"
    corrected_table = pd.read_csv(corrected / "private_predictions_24h.csv", index_col=0, parse_dates=True)
    assert corrected_table.seasonal_naive.iloc[0] == pytest.approx(history.FINAL_TOC.iloc[350 - 24])
