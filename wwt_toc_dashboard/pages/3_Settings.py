from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.settings_store import (
    DEFAULT_SETTINGS,
    load_app_settings,
    reset_app_settings,
    save_app_settings,
)

st.set_page_config(page_title="Settings", layout="wide", initial_sidebar_state="collapsed")

APP_DIR = Path(__file__).resolve().parents[1]
css_path = APP_DIR / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

settings = load_app_settings(APP_DIR)

st.markdown("### Settings")
st.caption("운전 대시보드 기준값과 진단 동작을 여기서 관리합니다.")

c1, c2 = st.columns([1.8, 1.2], gap="large")

with c1:
    st.markdown("#### Main / Signal")
    main_target_toc = st.number_input("Main 목표 TOC", min_value=1.0, max_value=100.0, value=float(settings.get("main_target_toc", 25.0)), step=0.5)
    main_warn_ratio = st.slider("Main 경고 비율", min_value=0.10, max_value=0.95, value=float(settings.get("main_warn_ratio", 0.60)), step=0.01)
    main_alarm_ratio = st.slider("Main 주의 비율", min_value=0.10, max_value=0.99, value=float(settings.get("main_alarm_ratio", 0.80)), step=0.01)

    st.markdown("#### Simulator")
    sim_default_target_toc = st.number_input("Simulator 기본 목표 TOC", min_value=1.0, max_value=100.0, value=float(settings.get("sim_default_target_toc", 20.0)), step=0.5)

    st.markdown("#### Process / Capacity")
    flow_capacity_tph = st.number_input("설비 캐파 (t/h)", min_value=100.0, max_value=5000.0, value=float(settings.get("flow_capacity_tph", 500.0)), step=10.0)
    flow_high_threshold_tph = st.number_input("고유입 기준 (t/h)", min_value=50.0, max_value=5000.0, value=float(settings.get("flow_high_threshold_tph", 400.0)), step=10.0)
    eq_toc_warn = st.number_input("균등조 TOC 주의 기준", min_value=0.1, max_value=100.0, value=float(settings.get("eq_toc_warn", 12.0)), step=0.1)

    st.markdown("#### Diagnosis UI")
    ollama_model = st.text_input("Ollama 모델", value=str(settings.get("ollama_model", "qwen2.5:7b-instruct")))
    diag_drawer_width_px = st.slider("진단 패널 폭(px)", min_value=320, max_value=700, value=int(settings.get("diag_drawer_width_px", 420)), step=10)
    diag_main_scale = st.slider("진단 열림 시 본문 축소 비율", min_value=0.85, max_value=1.00, value=float(settings.get("diag_main_scale", 0.94)), step=0.01)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("저장", use_container_width=True, type="primary"):
            if main_warn_ratio >= main_alarm_ratio:
                st.error("경고 비율은 주의 비율보다 작아야 합니다.")
            elif flow_high_threshold_tph > flow_capacity_tph:
                st.error("고유입 기준은 설비 캐파 이하로 설정하세요.")
            else:
                new_settings = {
                    "main_target_toc": float(main_target_toc),
                    "main_warn_ratio": float(main_warn_ratio),
                    "main_alarm_ratio": float(main_alarm_ratio),
                    "sim_default_target_toc": float(sim_default_target_toc),
                    "flow_capacity_tph": float(flow_capacity_tph),
                    "flow_high_threshold_tph": float(flow_high_threshold_tph),
                    "eq_toc_warn": float(eq_toc_warn),
                    "ollama_model": str(ollama_model).strip() or "qwen2.5:7b-instruct",
                    "diag_drawer_width_px": int(diag_drawer_width_px),
                    "diag_main_scale": float(diag_main_scale),
                }
                save_app_settings(APP_DIR, new_settings)
                st.success("저장 완료. Main/Simulator 페이지 새로고침 시 반영됩니다.")
    with b2:
        if st.button("기본값 복원", use_container_width=True):
            reset_app_settings(APP_DIR)
            st.success("기본값으로 복원했습니다. 페이지를 새로고침하세요.")

with c2:
    st.markdown("#### 안내")
    st.caption("설정값은 저장 즉시 `.cache/app_settings.json`에 반영됩니다.")
