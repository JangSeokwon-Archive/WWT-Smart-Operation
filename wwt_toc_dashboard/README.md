# 대산 WWT 방류수 예측 대시보드

폐수처리(WWT) 운전 데이터를 기반으로 `FINAL_TOC`를 예측하고, 운전자가 즉시 의사결정할 수 있도록 시각화/시뮬레이션/진단 기능을 제공하는 Streamlit 대시보드입니다.

## 1) 포트폴리오 핵심 요약
- 문제: 방류수 TOC 변동을 사전에 예측하고 운전 조치(폭기조 DO, 반송, 인발) 의사결정 지원
- 접근: 시계열 파생 피처(`lag`, `rolling`, `diff`) + LightGBM 회귀 + Residual 보정
- 산출물: Main 모니터링, Simulator(조치 시나리오), Notes, Settings, AI 운전 진단
- 강점: 현업 용어 중심 UI(반송량/인발량/DO양), 12h 중심 운전 판단, 24h 리스크 분리

## 2) 모델 구조
본 대시보드는 `wwt-toc-automl` 학습 산출물을 추론에 사용합니다.

- 모델 유형: 회귀(트리 기반 부스팅, LightGBM)
- 구조: Base + Residual
- 예측 리드: 6h / 12h / 24h
- 현재 운영 원칙:
  - 메인/진단 기준: 12h 중심
  - 24h: 리스크 전망(보조)

관련 리포트 예시(automl):
- `delay_ccf_granger_*.json`
- `nonlinear_zones_lead24h.json`
- `regime_dynamic_lead24h.json`
- `walkforward_stability.json`

## 3) 주요 화면
- Main (`streamlit_app.py`)
  - 현재 TOC, 예측, 최근 신호, 트렌드
  - 우측 진단 drawer
- Simulator (`pages/1_Simulator.py`)
  - 조작 변수 기반 12시간 시나리오
- Notes (`pages/2_Notes.py`)
  - 운전 기록/메모
- Settings (`pages/3_Settings.py`)
  - 목표치, 신호 임계비율, 진단 패널 설정

## 4) 로컬 실행
### 요구사항
- Python 3.11+ 권장
- pip

### 설치
```bash
cd /Users/jangseog-won/wwt_predict/wwt_toc_dashboard
pip install -r requirements.txt
```

### 실행
```bash
streamlit run streamlit_app.py
```

접속: `http://localhost:8501`

## 5) 배포(포트폴리오 공개)
GitHub 업로드만으로는 외부 접속이 되지 않습니다. 반드시 배포 URL이 필요합니다.

권장: Streamlit Community Cloud
1. GitHub에 레포 푸시
2. Streamlit Cloud에서 New app 생성
3. Main file: `streamlit_app.py`
4. Deploy

배포 후 README 상단에 `Live Demo` 링크를 추가하세요.

## 6) 포트폴리오에 같이 넣으면 좋은 내용
- 데이터 개요: 기간, 샘플 수, 주요 센서/공정 변수
- 검증 지표: RMSE, MAE, jump rate, overshoot rate
- 의사결정 로직: 12h 기준 + 24h 리스크 보조
- 한계: 장기 리드(24h+) 불안정 구간, 레짐 전이 구간 민감도
- 개선 계획: 레짐 분기 강화, 워크포워드 재튜닝, 진단 근거 문구 자동화

## 7) 디렉토리
```text
wwt_toc_dashboard/
  streamlit_app.py
  requirements.txt
  assets/style.css
  core/
    automl_infer.py
    llm_advisor.py
    simulator.py
    settings_store.py
  pages/
    1_Simulator.py
    2_Notes.py
    3_Settings.py
  data/
    raw.csv
```

## 8) 라이선스/주의
- 내부 공정 데이터(원본 csv) 공개 시 민감정보 마스킹 필요
- 포트폴리오 공개본에는 샘플 데이터 또는 익명화 데이터 사용 권장
