import json
import os

import pytest

from core import prediction_cache as cache


def setup_inputs(tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    raw.write_text("old")
    models, configs = tmp_path / "models", tmp_path / "configs"
    models.mkdir()
    configs.mkdir()
    (models / "test.joblib").write_bytes(b"version1")
    (configs / "features.yaml").write_text("lags: [1]")
    monkeypatch.setattr(cache, "resolve_bundle", lambda *a: (models, configs))
    monkeypatch.delenv("WWT_V2_MODEL_DIR", raising=False)
    return raw, models, configs


def test_input_change_invalidates_even_with_same_size_and_mtime(tmp_path, monkeypatch):
    raw, _, _ = setup_inputs(tmp_path, monkeypatch)
    before = cache.prediction_fingerprint(str(raw))
    stat = raw.stat()
    raw.write_text("new")
    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cache.prediction_fingerprint(str(raw)) != before


@pytest.mark.parametrize("component", ["model", "features"])
def test_model_and_feature_changes_invalidate(tmp_path, monkeypatch, component):
    raw, models, configs = setup_inputs(tmp_path, monkeypatch)
    before = cache.prediction_fingerprint(str(raw))
    path = models / "test.joblib" if component == "model" else configs / "features.yaml"
    path.write_text("changed")
    assert cache.prediction_fingerprint(str(raw)) != before


def test_legacy_cache_and_stale_csv_predictions_are_ignored(tmp_path, monkeypatch):
    raw, _, _ = setup_inputs(tmp_path, monkeypatch)
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"version": 5, "preds": {"pred_t24": 999}}))
    calls = []

    def infer(*a, **k):
        calls.append(True)
        return {"pred_t6": 10.0, "pred_t12": 11.0, "pred_t24": 12.0, "_meta": {"origin_at": "2025-01-01"}}

    monkeypatch.setattr(cache, "predict_multi_leads_from_automl", infer)
    assert cache.cached_predictions(str(raw), "", path)["pred_t24"] == 12
    assert cache.cached_predictions(str(raw), "", path)["pred_t24"] == 12
    assert len(calls) == 1
    assert json.loads(path.read_text())["preds"]["_meta"]["fingerprint"]


def test_racing_inputs_are_not_cached(tmp_path, monkeypatch):
    raw, _, _ = setup_inputs(tmp_path, monkeypatch)

    def racing(*a, **k):
        raw.write_text("changed during prediction")
        return {"_meta": {}}

    monkeypatch.setattr(cache, "predict_multi_leads_from_automl", racing)
    path = tmp_path / "cache.json"
    with pytest.raises(RuntimeError, match="changed"):
        cache.cached_predictions(str(raw), "", path)
    assert not path.exists()
