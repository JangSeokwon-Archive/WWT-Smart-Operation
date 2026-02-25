# 대산 WWT 방류수 예측 대시보드

폐수처리(WWT) 운전 데이터를 기반으로 `FINAL_TOC`를 예측하고, 운전 의사결정(폭기조 DO/반송/인발)까지 연결하는 Streamlit 대시보드입니다.

## 1) 프로젝트 핵심
- 문제: 방류수 TOC 변동을 선제적으로 예측하고, 운전 조치 우선순위를 제시
- 접근: 시계열 피처 + LightGBM(Base) + Residual 보정
- 결과: Main 모니터링 + Simulator + AI 진단 Drawer

## 2) 분석/고도화 단계(포트폴리오 설명용)
1. 데이터 정합화
- 결측/시간축 보정, TOC 이상치 전처리, 타깃 지연 정렬(lead별)

2. Base 모델 구축
- EQ/DAF 포함 전 공정 feature 기반으로 `pred_baseline` 학습
- lead 6/12/24h 각각 학습/튜닝

3. Residual 보정
- `residual = y_true - pred_baseline`을 별도 학습
- 폭기/침전 계열(AERA/AERB/CLAA/CLAB) lag/rolling/diff로 미세 오차 보정
- 최종 예측: `pred_final = pred_baseline + pred_residual`

4. 지연/비선형 분석
- DO/반송/인발 영향 지연(6~24h) 리포트 생성
- 운전 조작 대비 응답 구간 분석(control delay/sensitivity)

5. 운전 시뮬레이터/진단 연동
- 12h 최소 TOC 기준 Best 조합 제안
- Main은 12h 중심, 24h는 리스크 전망으로 분리

## 3) 모델 구조
- 모델 타입: LightGBM 회귀
- 파이프라인: Base + Residual
- 예측 리드: 6h / 12h / 24h
- 운전 판단 기준: 12h 중심

## 4) 포트폴리오 공개용 구성(모델 포함)
다른 사람이 “같은 모델 결과”를 보려면 아래를 같이 배포해야 합니다.

필수:
- `streamlit_app.py`
- `assets/style.css`
- `core/*.py`
- `data/raw_sample.csv` (샘플 데이터)
- `requirements.txt`
- `README.md`
- `../wwt-toc-automl/configs/base.yaml`
- `../wwt-toc-automl/configs/features.yaml`
- `../wwt-toc-automl/configs/features_residual.yaml`
- `../wwt-toc-automl/outputs/models/toc_lgbm_lead6h_baseline.joblib`
- `../wwt-toc-automl/outputs/models/toc_lgbm_lead6h_residual.joblib`
- `../wwt-toc-automl/outputs/models/toc_lgbm_lead12h_baseline.joblib`
- `../wwt-toc-automl/outputs/models/toc_lgbm_lead12h_residual.joblib`
- `../wwt-toc-automl/outputs/models/toc_lgbm_lead24h_baseline.joblib`
- `../wwt-toc-automl/outputs/models/toc_lgbm_lead24h_residual.joblib`

권장(근거 리포트):
- `../wwt-toc-automl/outputs/reports/holdout_residual_*.json`
- `../wwt-toc-automl/outputs/reports/recommendation_backtest_*.json`
- `../wwt-toc-automl/outputs/reports/control_delay_multivar_*.json`

## 5) 샘플 데이터 처리
원본 민감 데이터를 그대로 공개하지 말고 샘플로 변환해서 사용하세요.

```bash
cd /Users/jangseog-won/wwt_predict/wwt_toc_dashboard
python scripts/make_portfolio_sample.py --input data/raw.csv --output data/raw_sample.csv --rows 1080
```

처리 내용:
- 최근 N행만 추출(기본 45일)
- 날짜를 2031년 기준으로 시프트(실운전 시점 비식별)
- 대시보드 핵심 컬럼 우선 정렬

## 6) 실행
```bash
cd /Users/jangseog-won/wwt_predict/wwt_toc_dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 7) 배포
- 내부망: `http://내IP:8501`
- 외부 공유: 포트포워딩 또는 ngrok/cloudflared

## 8) 주의
- 공개본은 샘플/익명 데이터만 사용
- `__pycache__`, `.DS_Store`, `.sqlite`, 대용량 zip은 커밋 제외 권장
