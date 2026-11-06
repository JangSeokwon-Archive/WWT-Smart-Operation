# 실운전 데이터 계약

## 필수 계약과 현재 상태

| 항목 | 필요한 계약 | 현재 확인 상태 |
|---|---|---|
| sample_at | 장비/실험실 채취 시각, timezone 포함, 시각 정의와 집계 경계 | 원장은 aware 정시만 허용; 원본 현장 정의 미확인 |
| available_at | 원천 시스템에서 실제 사용할 수 있었던 시각과 근거 | 원장에 선언값 기록; 독립적 진위 미확인 |
| received_at | collector가 실제 수신한 시각, 신뢰 가능한 동기화 시계 | 원장의 `recorded_at`은 로컬 DB 기록시각으로 collector 수신시각과 같다고 보장하지 않음 |
| 수정/재전송 | 센서 ID·원본 ID·revision, 이전 값 및 정정 이유 | 중복은 멱등/충돌 거부; 무효화와 새 source_id로 이력 보존, 원천 revision 자동 수집 없음 |
| 단위/품질 | 센서 ID, 단위, 품질코드, 교정·점검·정지 상태 | 원장 observation에는 단위·품질 필드 없음; 품질 상태로 자동 제외하지 못함 |
| 시계·timezone | 현장과 collector 시간대, UTC offset, 동기화·오차 기록 | aware timestamp 형식만 확인; 원천 timezone/시계 정확도 미검증 |
| TOC 분석 | 채취, 분석, 보고, 정정 시각과 지연 분포 | 기본 1시간은 가정, 실제 기록 필요 |
| FLOW | 순간/평균/합계, 집계 구간, 측정 단위 | 미확인; 정시가 집계 의미를 증명하지 않음 |
| 결측과 0 | 품질코드와 전송 실패를 구분; 실제 0 보존 | 유한 숫자만 ingest, 결측은 누락으로 표현; 0은 유지, 원인은 불명 |
| 계통/위치 | P&ID, 센서 위치, A/B 유량 배분, 설비 상태 | 변수 이름만 존재; 실제 경로 미확인 |
| 정상/고부하 | 사전 합의된 정의, 기간과 event 기준 | 개발용 통계 기준은 현장 승인 기준 아님 |

예측 발행 시각 `issue_at`에서 입력은 `sample_at <= origin_at`, `available_at <= issue_at`, `recorded_at <= issue_at`을 모두 만족해야 한다. TOC는 추가로 원장 고정 지연을 충족한다. 정답 평가는 예측 발행 이후 수신된 해당 target 시각 관측에만 허용된다. 원장은 이 순서와 스냅샷 해시를 확인하지만 선언된 `available_at`의 진실성, 과거 모델 학습 데이터의 이용시각, collector 수신 누락은 보증하지 않는다. 장비 상태 미확인 값은 안전한 관측이라 가정할 수 없다. 지연이 바뀌면 동일 원장 설정을 변경하지 말고 새 원장과 비교 정책을 사용한다.

원천에서 실제 도착 기록이 없는 과거 export는 historical replay만 가능하다. `recorded_at`을 원천 `received_at`으로 재명명하거나 임의 시각을 만들어 prospective로 저장하지 않는다. 시각 동기화 로그, 불변 원본, 외부 감사 앵커와 접근 제어는 별도 구축해야 한다. 구현: `wwt_toc_dashboard/core/shadow_store.py`, `wwt_toc_dashboard/scripts/shadow.py`; 한계와 사용법: `docs/V22_SHADOW_VALIDATION.md`.
