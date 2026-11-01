"""UI integration uses generated inputs and deterministic mocks only."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_main_diagnosis_and_simulator_render(history, tmp_path, monkeypatch):
    from core import llm_advisor, prediction_cache

    path = tmp_path / "synthetic.csv"
    history.rename_axis("날짜").reset_index().to_csv(path, index=False)
    monkeypatch.setenv("WWT_RAW_PATH", str(path))
    predictions = {"pred_t6": 11.0, "pred_t12": 12.0, "pred_t24": 13.0,
                   "current_toc_ref": float(history.FINAL_TOC.iloc[-2]),
                   "_meta": {"origin_at": history.index[-1].isoformat(), "quality": {"status": "ok"},
                             "horizons": {str(h): {"mode": "synthetic_test", "effective_horizon_hours": h} for h in (6, 12, 24)}}}
    monkeypatch.setattr(prediction_cache, "cached_predictions", lambda *a, **k: predictions)
    monkeypatch.setattr(llm_advisor, "run_llm_advisor", lambda context, *a: llm_advisor.build_quick_report(context))
    root = Path(__file__).resolve().parents[1] / "wwt_toc_dashboard"
    main = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=60).run()
    assert not main.exception
    main.button(key="diagnose_btn").click().run()
    assert not main.exception
    simulator = AppTest.from_file(str(root / "pages" / "1_Simulator.py"), default_timeout=60).run()
    assert not simulator.exception
    assert len(simulator.slider) == 3
    simulator.slider[0].set_value(0.2).run()
    assert not simulator.exception
