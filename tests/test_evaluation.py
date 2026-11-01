import numpy as np
import pandas as pd
import pytest

from core.forecast_evaluation import conformal_radius, event_metrics, purged_folds, regression_metrics
from core.forecast_training import fit_model


def test_purge_uses_elapsed_time_even_when_rows_are_missing():
    index = pd.date_range("2025-01-01", periods=1000, freq="h").delete(np.arange(200, 450, 2))
    for train, valid in purged_folds(index, horizon=24, delay=2):
        assert index[train[-1]] + pd.Timedelta(hours=26) < index[valid[0]]
        assert index[train[-1]] < index[valid[0]]


def test_oof_residuals_never_use_in_sample_predictions(history, monkeypatch):
    import core.forecast_training as training
    fits, queries = [], []

    class Recorder:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            self.index = X.index
            self.n_features_in_ = X.shape[1]
            fits.append(self.index)
            return self

        def predict(self, X):
            queries.append((self.index, X.index))
            return np.zeros(len(X))

    monkeypatch.setattr(training, "LGBMRegressor", Recorder)
    model = fit_model("lgbm_oof_residual", history[["EQ_TOC"]], history.FINAL_TOC, 12, 1, 10)
    assert len(fits) == 4
    for trained, predicted in queries:
        assert trained.max() + pd.Timedelta(hours=13) < predicted.min()
    assert model.n_features_in_ == 1


def test_missing_events_are_unverified_not_perfect_scores():
    metrics = regression_metrics([10, 11], [10, 10], [10, 10], [False, False], 20, 5)
    assert metrics["exceedance_recall"] is None
    assert metrics["rise_detection_rate"] is None
    assert metrics["high_load_mae"] is None
    assert metrics["mae"] == 0.5


def test_false_alarm_denominator_is_actual_negatives():
    metrics = regression_metrics([10, 11, 21], [22, 10, 22], [10, 10, 10], [False] * 3, 20, 2)
    assert metrics["false_alarm_rate"] == 0.5
    assert metrics["exceedance_recall"] == 1


def test_calibration_quantile_and_minimum_size():
    assert conformal_radius(np.arange(100), np.zeros(100), coverage=0.9) >= 90
    with pytest.raises(ValueError):
        conformal_radius([1, 2], [0, 0])


def test_event_warning_time_uses_issue_time():
    origins = pd.date_range("2025-01-01", periods=100, freq="h")
    y = np.zeros(100)
    y[40:44] = 25
    result = event_metrics(origins, y, y, 12, 20)
    assert result["exceedance_events"] == 1
    assert result["event_recall"] == 1
    assert result["median_warning_hours"] == 12
