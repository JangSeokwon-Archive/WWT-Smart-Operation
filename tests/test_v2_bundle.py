from dataclasses import asdict

import joblib
import numpy as np
import pytest
from lightgbm import LGBMRegressor

from core.automl_infer import _predict_v2
from core.forecast_pipeline import PIPELINE_VERSION, PipelineConfig, build_features, make_labels


def test_training_and_inference_share_identical_features(history, tmp_path):
    cfg = PipelineConfig()
    X, y = build_features(history, cfg), make_labels(history, 6)
    train = y.notna()
    model = LGBMRegressor(n_estimators=5, n_jobs=1, verbosity=-1).fit(X.loc[train], y.loc[train])
    bundle = {"model": model, "columns": list(X.columns), "horizon": 6,
              "pipeline_version": PIPELINE_VERSION, "pipeline_config": asdict(cfg), "config_digest": cfg.digest(),
              "deployment_gate_passed": True, "interval_calibration_passed": False, "interval_radius": 0.01}
    joblib.dump(bundle, tmp_path / "forecast_6h.joblib")
    prediction, meta = _predict_v2(history, 6, tmp_path)
    assert prediction == pytest.approx(float(model.predict(X.tail(1))[0]))
    assert "interval" not in meta
    bundle["columns"] = list(reversed(X.columns))
    joblib.dump(bundle, tmp_path / "forecast_6h.joblib")
    with pytest.raises(ValueError, match="order"):
        _predict_v2(history, 6, tmp_path)
    bundle["columns"] = list(X.columns)
    bundle["deployment_gate_passed"] = False
    joblib.dump(bundle, tmp_path / "forecast_6h.joblib")
    with pytest.raises(ValueError, match="gate"):
        _predict_v2(history, 6, tmp_path)
    bundle["deployment_gate_passed"] = True
    bundle["config_digest"] = "wrong config"
    joblib.dump(bundle, tmp_path / "forecast_6h.joblib")
    with pytest.raises(ValueError, match="fingerprint"):
        _predict_v2(history, 6, tmp_path)
