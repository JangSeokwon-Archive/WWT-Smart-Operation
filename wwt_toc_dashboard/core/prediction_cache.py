"""Content-addressed cache shared by both Streamlit pages."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from core.automl_infer import predict_multi_leads_from_automl, resolve_bundle

CACHE_VERSION = 2


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_fingerprint(raw_path: str, automl_root_path: str = "") -> str:
    models, configs = resolve_bundle(automl_root_path)
    v2_dir = os.getenv("WWT_V2_MODEL_DIR", "").strip()
    code_root = Path(__file__).resolve().parent
    paths = [Path(raw_path), *sorted(models.glob("*.joblib")), *sorted(configs.glob("*.yaml")), *sorted(code_root.glob("*.py"))]
    paths += sorted((code_root.parent / "src").rglob("*.py"))
    if v2_dir:
        paths += sorted(Path(v2_dir).glob("*.joblib"))
    payload = [(str(p.resolve()), file_digest(p)) for p in paths]
    return hashlib.sha256(json.dumps([CACHE_VERSION, v2_dir, payload]).encode()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cached_predictions(raw_path: str, automl_root_path: str, cache_path: str | Path, fingerprint: str | None = None) -> dict:
    key = fingerprint or prediction_fingerprint(raw_path, automl_root_path)
    path = Path(cache_path)
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("version") == CACHE_VERSION and cached.get("fingerprint") == key:
            preds = cached["preds"]
            if not isinstance(preds, dict) or not isinstance(preds.get("_meta"), dict):
                raise ValueError("Invalid prediction cache schema")
            for h in (6, 12, 24):
                for name in (f"pred_t{h}", f"pred_t{h}_raw"):
                    if preds.get(name) is None:
                        preds[name] = float("nan")
            return preds
    except (OSError, ValueError, KeyError, TypeError):
        pass
    preds = predict_multi_leads_from_automl(raw_path, automl_root_path, leads=(6, 12, 24))
    if prediction_fingerprint(raw_path, automl_root_path) != key:
        raise RuntimeError("Inputs changed during inference; rerun to obtain a consistent forecast")
    preds["_meta"]["fingerprint"] = key
    successful = all(isinstance(preds.get(f"pred_t{h}"), (int, float)) and math.isfinite(preds[f"pred_t{h}"]) for h in (6, 12, 24))
    if successful:
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = stream.name
                json.dump({"version": CACHE_VERSION, "fingerprint": key, "preds": _json_safe(preds)}, stream, allow_nan=False)
            os.replace(temporary, path)
        except OSError:
            pass
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()
    return preds
