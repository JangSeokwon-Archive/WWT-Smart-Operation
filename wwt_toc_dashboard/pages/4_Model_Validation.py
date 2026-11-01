"""Read-only local shadow report; never starts recording or control operations."""
from pathlib import Path
import json
import streamlit as st

from core.shadow_store import ShadowStore, now_utc
from core.shadow_monitor import summarize_scores, performance_status, input_drift, event_windows
import pandas as pd

st.set_page_config(page_title="Model Validation", layout="wide")
st.title("Model Validation")
st.caption("Shadow 기록은 의사결정 검증용이며 설비 제어에 연결되지 않습니다.")
st.warning("배포 판단: 보류. 센서 도착 시각, 미래 성능, 이벤트 표본과 현장 검토가 필요합니다.")
root = Path(__file__).resolve().parents[2] / ".local" / "shadow"
cfg = json.loads((Path(__file__).resolve().parents[1] / "model_bundle/configs/shadow_validation.json").read_text(encoding="utf-8"))
if cfg.get("event_window_hours") != 24:
    st.error("현재 유입·부하·MLSS 변화 이벤트는 24시간 창만 지원합니다.")
    st.stop()
if type(cfg.get("target_delay_hours_assumption")) is not int or cfg["target_delay_hours_assumption"] < 0:
    st.error("TOC 보고 지연은 0 이상의 정수 시간으로 설정해야 합니다.")
    st.stop()

for label, mode, name in (("Prospective", "prospective", "prospective.sqlite"), ("Historical Replay", "historical_replay", "replay.sqlite")):
    with st.container(border=True):
        st.subheader(label)
        path = root / name
        if mode == "historical_replay" and not path.exists():
            path = root.parent / "v22-validation" / "historical_replay.sqlite"
        if not path.exists():
            st.info("Prospective 데이터 미수집" if mode == "prospective" else "Historical Replay 기록 없음")
            continue
        try:
            db = ShadowStore(path, mode, create=False, target_delay_hours=int(cfg["target_delay_hours_assumption"]))
            rows = db.scored()
            summary = summarize_scores(rows, now_utc(), alarm=cfg.get("internal_toc_alarm"), rise_delta=cfg.get("toc_rise_delta"), regime_config=cfg)
            invalid = {e["target_seq"] for e in db.events("invalidation")}
            forecasts = [e for e in db.events("forecast") if e["seq"] not in invalid]
            observations = [e for e in db.events("observation") if e["seq"] not in invalid]
            st.warning("과거 재생 데이터입니다. 실제 미래 예측 성능이 아닙니다." if mode == "historical_replay" else "Shadow 관측 중: 모델 성능과 센서 도착 지연은 별도 검증이 필요합니다.")
            st.write(f"성능 상태: {performance_status(summary['groups'], cfg.get('performance_thresholds'))}")
            st.caption(f"미평가 {summary['pending']}건 · 실측 누락 {summary['missing']}건")
            if forecasts:
                st.caption(f"최종 예측 생성: {forecasts[-1]['issue_at']} · 최종 관측 표본: {max((e['sample_at'] for e in observations), default='없음')} · 최종 수집: {max((e['recorded_at'] for e in observations), default='없음')}")
                st.caption(f"입력 결측: {forecasts[-1]['quality']['missing_fraction']:.1%}")
                st.caption(f"모델 미제공: {forecasts[-1]['unavailable_models']}")
                with st.expander("최신 예측 시각과 모델 이력"):
                    st.json({"origin": forecasts[-1]["origin_at"], "issue": forecasts[-1]["issue_at"], "targets": forecasts[-1]["targets"],
                             "latest_toc_sample": forecasts[-1]["latest_toc_sample_at"], "latest_toc_available": forecasts[-1]["latest_toc_available_at"],
                             "sensor_availability": forecasts[-1].get("latest_input_availability", {}),
                             "input_sha256": forecasts[-1]["input_sha256"], "models": forecasts[-1]["models"], "calibration": forecasts[-1]["calibration"]})
            reference_path = root / "reference.json"
            if reference_path.exists() and observations:
                reference = json.loads(reference_path.read_text(encoding="utf-8"))
                if pd.Timestamp(reference["cutoff"]) >= pd.Timestamp(now_utc()):
                    raise ValueError("드리프트 기준 시각이 현재보다 늦습니다")
                items = pd.DataFrame(observations)
                items["sample_at"] = pd.to_datetime(items.sample_at, utc=True)
                items = items[items.sample_at >= pd.Timestamp(now_utc()) - pd.Timedelta(days=7)]
                if not items.empty and not items.duplicated(["sample_at", "column"]).any():
                    drift = input_drift(reference, items.pivot(index="sample_at", columns="column", values="value"))
                    st.write("입력 분포 변화 (성능 저하와 별도):", drift)
                else:
                    st.info("입력 드리프트 평가 데이터 부족")
            else:
                st.info("입력 드리프트 기준 데이터 또는 임계값 미설정")
            if summary["groups"]:
                st.dataframe(summary["groups"], hide_index=True, width="stretch")
            else:
                st.info("아직 평가된 예측이 없습니다.")
            if forecasts and cfg.get("internal_toc_alarm") is not None:
                events = event_windows(observations, forecasts, alarm=cfg["internal_toc_alarm"], horizon=24, model="persistence")
                warning = [e["warning_hours"] for e in events["events"] if e["warning_hours"] is not None]
                median = float(pd.Series(warning).median()) if warning else None
                st.write(f"24h 기준 초과 이벤트: {events['event_n']}건 · 미탐 {events['missed']}건 · 불완전 {events['censored_n']}건 · 탐지율: {events['recall'] if events['recall'] is not None else '미검증'} · 사전 경고시간 중앙값: {median if median is not None else '미검증'}")
                if events["events"]:
                    st.dataframe(events["events"], hide_index=True, width="stretch")
            st.caption("관리 기준 초과와 급상승 이벤트는 설정 기준이며 법적 배출 기준이 아닙니다. 표본이 없으면 탐지율은 미검증입니다.")
            db.close()
        except (ValueError, OSError) as error:
            st.error(f"Shadow 기록을 검증할 수 없습니다: {error}")
