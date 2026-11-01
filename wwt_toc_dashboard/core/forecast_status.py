"""Small status panel without changing the dashboard layout."""
import math
from datetime import datetime, timezone


def show_forecast_status(predictions: dict) -> None:
    import pandas as pd
    import streamlit as st

    meta = predictions.get("_meta", {})
    with st.expander("예측 기준 · 데이터 품질 · 검증 상태", expanded=False):
        if meta.get("error"):
            st.warning(f"예측 생성 실패: {meta['error']}")
            return
        origin = meta.get("origin_at")
        st.caption(f"기준 시각: {origin} | 생성 시각(UTC): {meta.get('generated_at')} | 모델 파일 갱신(UTC): {meta.get('model_updated_at')}")
        st.caption(f"사용 가능한 최신 TOC의 시료 시각: {meta.get('current_toc_sample_at', '확인 필요')}")
        st.caption("기준 시각은 원본 계측 시각이며 시간대는 확인이 필요합니다. 파일 갱신 시각은 학습 완료 시각과 다를 수 있습니다.")
        if origin:
            timestamp = pd.Timestamp(origin)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("Asia/Seoul")
            age = (pd.Timestamp(datetime.now(timezone.utc)) - timestamp).total_seconds() / 3600
            if age > 3:
                st.warning(f"과거 데이터 기반 예측입니다. 마지막 데이터로부터 약 {age:.0f}시간 경과했습니다. 신선도 표시에만 한국 시간대를 가정했습니다.")
        quality = meta.get("quality", {})
        st.write(f"데이터 품질: {quality.get('status', 'unknown')}")
        rows = []
        for lead in (6, 12, 24):
            detail = meta.get("horizons", {}).get(str(lead), {})
            rows.append({"모델 명칭": f"{lead}h", "실제 예측 시각": detail.get("target_at"),
                         "TOC": predictions.get(f"pred_t{lead}"), "불확실성 구간": str(detail.get("interval", "미검증")),
                         "검증 상태": detail.get("validation", detail.get("mode", detail.get("error", "unknown")))})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("V1 학습식은 horizon에 분석 지연을 추가했습니다. V2는 정확히 +6/+12/+24h를 정답으로 사용합니다. 그래프의 시간축에서도 실제 예측 시각을 확인하세요.")
        raw = predictions.get("pred_t24_raw", float("nan"))
        guarded = predictions.get("pred_t24_legacy_guarded", float("nan"))
        if isinstance(raw, (float, int)) and isinstance(guarded, (float, int)) and math.isfinite(raw) and math.isfinite(guarded) and raw != guarded:
            st.write(f"24h 원본: {raw:.2f} | V1 점프 제한 적용 시: {guarded:.2f}. 현재는 원본을 표시합니다.")
        st.caption("기준 초과확률: 미검증. 실측 보정된 확률 모델 없이 확률 또는 신뢰도 수치를 표시하지 않습니다.")
