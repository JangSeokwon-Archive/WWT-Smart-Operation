from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    "main_target_toc": 25.0,
    "main_warn_ratio": 0.60,
    "main_alarm_ratio": 0.80,
    "sim_default_target_toc": 20.0,
    "flow_capacity_tph": 500.0,
    "flow_high_threshold_tph": 400.0,
    "eq_toc_warn": 12.0,
    "ollama_model": "qwen2.5:7b-instruct",
    "diag_drawer_width_px": 420,
    "diag_main_scale": 0.94,
}


def _settings_path(app_dir: Path) -> Path:
    return app_dir / ".cache" / "app_settings.json"


def load_app_settings(app_dir: Path) -> dict[str, Any]:
    p = _settings_path(app_dir)
    out = dict(DEFAULT_SETTINGS)
    try:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                out.update(obj)
    except Exception:
        pass
    return out


def save_app_settings(app_dir: Path, settings: dict[str, Any]) -> None:
    p = _settings_path(app_dir)
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings or {})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_app_settings(app_dir: Path) -> None:
    save_app_settings(app_dir, dict(DEFAULT_SETTINGS))
