import numpy as np
import pandas as pd
import pytest

from core.forecast_pipeline import PipelineConfig, build_features, hourly_observations, make_labels


def test_future_edits_do_not_change_past_features(history):
    cutoff = history.index[300]
    original = build_features(history)
    changed = history.copy()
    changed.loc[changed.index > cutoff, :] = 999999
    pd.testing.assert_frame_equal(original.loc[:cutoff], build_features(changed).loc[:cutoff])
    pd.testing.assert_series_equal(original.loc[cutoff], build_features(history.loc[:cutoff]).iloc[-1])


def test_bounded_fill_never_uses_future_median_or_bfill(history):
    history.iloc[:5, history.columns.get_loc("EQ_TOC")] = np.nan
    history.iloc[100:110, history.columns.get_loc("EQ_TOC")] = np.nan
    X = build_features(history, PipelineConfig(max_fill_hours=3))
    assert X.EQ_TOC.iloc[:5].isna().all()
    assert X.EQ_TOC.iloc[100:103].notna().all()
    assert X.EQ_TOC.iloc[103:110].isna().all()
    assert X.EQ_TOC__age_h.iloc[109] == 10


def test_missing_labels_are_never_filled(history):
    history.iloc[100, history.columns.get_loc("FINAL_TOC")] = np.nan
    labels = make_labels(history, 12)
    assert np.isnan(labels.iloc[88])
    assert labels.iloc[-12:].isna().all()
    assert len(build_features(history)) == len(history)


def test_delay_affects_availability_not_horizon(history):
    cfg = PipelineConfig(target_delay_hours=2, sensor_delays=(("EQ_TOC", 3),))
    X = build_features(history, cfg)
    assert X.TOC_AVAILABLE.iloc[100] == history.FINAL_TOC.iloc[98]
    assert X.EQ_TOC.iloc[100] == history.EQ_TOC.iloc[97]
    assert make_labels(history, 6).iloc[100] == history.FINAL_TOC.iloc[106]


def test_preserve_shock_but_reject_physical_invalids(history):
    history.loc[history.index[300], "EQ_TOC"] = 500
    history.loc[history.index[301], "FLOW"] = -1
    history["EQ_pH"] = 7.0
    history.loc[history.index[301], "EQ_pH"] = 15
    observed = hourly_observations(history)
    assert observed.EQ_TOC.iloc[300] == 500
    assert np.isnan(observed.FLOW.iloc[301])
    assert np.isnan(observed.EQ_pH.iloc[301])
    assert build_features(history).EQ_TOC__shock.iloc[300] > 5


def test_no_precomputed_predictions_in_features(history):
    history["pred_final"] = history.FINAL_TOC.shift(-24)
    history["pred_t24"] = history.FINAL_TOC.shift(-24)
    X = build_features(history)
    assert not any(c.startswith("pred_") for c in X)


def test_missing_hours_duplicates_and_off_hour(history):
    duplicate = pd.concat([history, history.iloc[[100]].assign(FLOW=777)])
    result = hourly_observations(duplicate.drop(history.index[200]))
    assert result.FLOW.iloc[100] == 777
    assert result.iloc[200].isna().all()
    history.index = history.index + pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="Off-hour"):
        hourly_observations(history)


def test_missing_load_inputs_do_not_create_values(history):
    X = build_features(history.drop(columns=["FLOW"]))
    assert "organic_load_proxy" not in X


@pytest.mark.parametrize("config", [{"sampling_minutes": 30}, {"target_delay_hours": -1}, {"max_fill_hours": -1}, {"sensor_delays": (("EQ_TOC", -1),)}])
def test_reject_invalid_contract(config):
    with pytest.raises(ValueError):
        PipelineConfig(**config)
