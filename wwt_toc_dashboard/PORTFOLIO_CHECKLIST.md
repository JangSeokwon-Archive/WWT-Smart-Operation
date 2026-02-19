# 포트폴리오 제출 체크리스트

## A. 기능 동작
- [ ] Main 페이지 정상 로드
- [ ] 트렌드 그래프 표시
- [ ] `진단하기` 클릭 시 로딩 표시
- [ ] 진단 drawer에 HTML 문자열(`<div ...>`) 노출 없음
- [ ] Simulator에서 슬라이더 조정 시 12h 조정 예측 반영
- [ ] Notes 작성/조회/삭제 정상
- [ ] Settings 저장 후 재반영 정상

## B. 모델/지표 정리
- [ ] 모델 구조(Base + Residual, LightGBM) 설명 작성
- [ ] 6h/12h/24h 리드 사용 방식 설명
- [ ] 핵심 지표(RMSE/MAE/jump/overshoot) 표 작성
- [ ] 12h 운영 기준, 24h 리스크 보조 원칙 명시

## C. 배포
- [ ] GitHub 최신 코드 push
- [ ] Streamlit Cloud 배포 URL 생성
- [ ] README에 Live Demo 링크 추가
- [ ] requirements.txt 설치 검증

## D. 발표/면접 대비
- [ ] 왜 랜덤포레스트/LSTM 대신 LightGBM인지 설명 가능
- [ ] 레짐(정상/고부하/쇼크) 개념 설명 가능
- [ ] 지연효과(CCF/Granger) 해석 가능
- [ ] 실제 운전 의사결정에 어떻게 연결되는지 1분 설명 준비

## E. 공개 전 점검
- [ ] 민감 데이터 마스킹
- [ ] 경로 하드코딩 제거 여부 확인
- [ ] 에러/경고 로그 없는지 확인
