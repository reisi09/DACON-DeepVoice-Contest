# 대회 규칙 재검토와 MLAAD 추가 편입 — 2026-09-24

MLAAD는 이번 대회용 비상업 탐지 연구·평가에 사용할 수 있다고 판단해 기존 보류를 해제했다. CPU 검사 후 146개를 추가했으며, 신규 학습 편입은 LibriSpeech 449개·Music Arena 882개·MLAAD 146개로 총 1,477개다. 최신 통합 역할표는 `roles-v6`, 전체 manifest는 44,424행이다. GPU 학습과 공식 제출은 실행하지 않았다.

## 판단 근거

[대회 개요](https://dacon.io/competitions/official/236749/overview/description)는 딥보이스 탐지 모델 개발과 모델·학습데이터 보고서 평가를 명시한다. [규칙](https://dacon.io/competitions/official/236749/overview/rules)은 누구나 접근할 수 있고 최소 비영리 사용이 허용되는 외부 자원을 인정한다.

[9월 21일 운영진 답변](https://dacon.io/competitions/official/236749/talkboard/417857)은 연구·비상업용 사전학습 모델을 코드 제출, 평가 서버 추론, 2차 자료 제출, 상금 수령에 사용할 수 있다고 안내한다. 이 답변은 사전학습 모델에 관한 것이므로 MLAAD 데이터에 적용한 부분은 본 프로젝트의 해석이다.

[MLAAD 공식 카드](https://huggingface.co/datasets/mueller91/MLAAD/blob/main/README.md)는 탐지 모델 학습·평가를 용도로 제시하며, CC BY-NC 4.0과 비상업적 학술연구 조건을 명시한다. 확인한 카드에는 기관 소속, 논문 발표, 연구마다 별도 서면허가를 요구하는 조항이 없다. 사용자가 밝힌 대회 모델 제작 목적과 현재 수행하는 개발·실험·평가·보고서 작성 범위를 함께 검토해 이번 연구에 한정해 적격으로 판단했다. 이전의 학술연구 범위 미확정을 이유로 한 일괄 보류는 지나치게 보수적이었다.

[검증 자료 제출 답변](https://dacon.io/competitions/official/236749/talkboard/417280)은 운영진 검증을 일반 공개와 구분하며 실제 학습 파일 제출을 요구한다. [외부 자원 권리 답변](https://dacon.io/competitions/official/236749/talkboard/417310)은 검증 제출 후에도 외부 자원의 기존 라이선스가 유지된다고 설명한다. MLAAD의 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)은 조건을 지키는 비상업 복제·공유를 허용한다.

MLAAD 배포자가 이 대회를 개별 승인했다는 뜻은 아니다. 출처·라이선스·학술연구 제한·수정 내역을 보존하고 상용 서비스나 판매로 확대하지 않는다. 이전 실행의 이용 조건과 결과 파일은 변경하지 않고 이번 결정을 별도 기록했다. Jamendo는 [약관](https://licensing.jamendo.com/en/legal/termsofuse)의 연구 TDM/ML/AI 사전 서면허가 조항 때문에 계속 보류한다.

## 실제 CPU 처리

MLAAD 400개 중 평가 원천과 연결된 216개와 출처 미해결 5개를 먼저 제외했다. 나머지 179개의 15초 클립을 기존 AST로 검사해 두 구간의 Speech 점수가 모두 0.5 이상인 146개를 채택했다. 33개는 기준 미달로 제외했다. CPU 8스레드에서 85.62초 걸렸으며, 사람의 청취 판정은 수행하지 않았다. AST는 음성 존재 선별에만 사용했고 가짜 음성 정답은 배포 출처에서 유지했다.

기존 44,278행은 전체 필드가 그대로 유지됐으며 새 146개는 모두 train이다. 원본 400개 해시가 유지됐고 역할 간 원천 그룹 중복은 0개다. 실제 학습 loader는 train/selection/calibration 33,674행과 development 9,842행을 읽었다. 신규 자료는 voice fake만 감독하고 미확인 music 진위는 마스킹한다.

최종 manifest의 역할별 행 수는 train 29,796, selection 2,435, calibration 2,073, development 10,120이다. Bank는 감독할 성분 정답이 있는 행만 내보내므로 manifest 총행 수와 다르다.

## 전달

기존 VESSL volume의 `lb-improvement-20260924/datasets/mlaad-admitted-v1/`에 146개 파생 음원과 roles-v6·출처·판정 기록을 추가했다. Archive와 영수증을 전체 재다운로드해 SHA-256을 대조했다. Archive는 71,206,318 bytes, SHA-256 `4f090b462ef76b422afe5d27608b41fb38e0b56f5ada83bae8a14aeae924ca1d`다. 이전 `admitted-v2`의 1,331개와 기존 역사 음원은 계속 필요하다.

[이용 조건](evidence/lb_mlaad_admission_20260924/rights_decisions.json) · [실제 입력 검증](evidence/lb_mlaad_admission_20260924/integration_verification.json) · [전송 검증](evidence/lb_mlaad_admission_20260924/transfers.json) · [출처 표시](evidence/lb_mlaad_admission_20260924/THIRD_PARTY_NOTICES.md)
