import numpy as np
import pandas as pd
import pytest

from core.automl_infer import _checked_predict, _guard_24h_jump, _legacy_frame
from core.forecast_pipeline import PipelineConfig, hourly_observations
from core.llm_advisor import _validate_and_guardrail, build_quick_report, rule_status
from core.simulator import apply_simulation, fit_quadratic_response_model, recommend_optimal_controls_24h


def test_inference_keeps_latest_origin_without_future_target(history):
    cfg = {"equipment_prefixes": ["EQ"], "global_cols": [], "global_feature_params": {},
           "per_equipment": {"EQ": {"lags": [1], "rolls": [3], "diffs": []}},
           "cross_features": {"trend_features": {"enabled": False}}, "leakage_guard": {"exclude_prefixes": ["FINAL"]}}
    frame = _legacy_frame(hourly_observations(history), PipelineConfig(), cfg)
    assert frame["날짜"].iloc[-1] == history.index[-1]


def test_schema_mismatch_is_explicit():
    with pytest.raises(ValueError, match="schema"):
        _checked_predict({"columns": ["missing"], "model": object()}, pd.DataFrame({"actual": [1]}))


def test_legacy_guard_can_hide_a_rising_alarm():
    raw = {"pred_t6": 14.0, "pred_t12": 14.2, "pred_t24": 25.0}
    assert _guard_24h_jump(raw, 14)["pred_t24"] < 20
    assert raw["pred_t24"] == 25


def test_llm_cannot_invent_numbers_controls_or_confidence():
    context = {"current_toc": 10, "pred_t6": 11, "pred_t12": 12, "pred_t24": 25}
    malicious = {"status": "양호", "report": "TOC 999, set DO 9", "confidence": 0.99,
                 "actions": [{"target": "AERB_DO", "delta": 9}], "evidence_codes": ["invented", "forecast_rising"]}
    result = _validate_and_guardrail(malicious, context)
    assert result["status"] == "주의"
    assert result["confidence"] is None
    assert not result["actions"]
    assert "999" not in result["report"]
    assert "25.00" in result["report"]


def test_declining_forecast_does_not_downgrade_current_alarm():
    assert rule_status({"current_toc": 25, "pred_t6": 12, "pred_t12": 11, "pred_t24": 10}) == "주의"
    assert rule_status({"current_toc": 10, "pred_t6": np.nan, "pred_t12": 11, "pred_t24": 10}) == "주의"
    assert not build_quick_report({}).payload["actions"]


def test_simulator_validation_does_not_fit_holdout_distribution(history):
    first = fit_quadratic_response_model(history)
    changed = history.copy()
    changed.iloc[400:, :] *= 5
    second = fit_quadratic_response_model(changed)
    assert first["ok"] and second["ok"]
    assert first["beta"] == second["beta"]
    assert first["support_bounds"] == second["support_bounds"]
    assert not first["causal_validated"]


def test_no_action_is_exact_identity(history):
    series = history.FINAL_TOC.tail(24)
    pd.testing.assert_series_equal(apply_simulation(series, 0, 0, 0, current_do=1.8, current_ret=120, current_wit=5), series)


def test_optimizer_refuses_missing_state():
    result = recommend_optimal_controls_24h(10, 11, 12, 13, np.nan, 120, 5)
    assert not result["ok"]
