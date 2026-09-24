# 완료된 캐시로 E08 학습 — 2026-09-24

사용자 요청에 따라 특징 추출을 5,952개에서 중단했다. 완료된 캐시를 기존 object volume에 저장하고, 음성 구간 집계를 max에서 상위 30% logit 평균으로 바꾼 E08을 학습했다. 같은 데이터·seed 41로 E00 기준선도 학습했다. E03 공동 특징 헤드나 백본 fine-tuning을 실행한 것은 아니다. MERT·DF-Arena·PANNs·HTDemucs는 고정하고 music WindowMIL과 voice linear head만 학습했다.

기존 22:22:52 KST 종료 제한은 사용자의 추가 지시에 따라 해제하고, 22:44:10 KST까지 별도 30분 한도를 설정했다. 실제 학습은 E08 약 48.2초, 기준선 약 39.0초에 끝났다. 결과를 검증한 후 일찍 정지했으며 22:18:55 KST에 workspace `wsp-nvxqw1jf9zdv`가 `stopped / standby`인 것을 확인했다.

원래 행의 ID·라벨·역할을 바꾸지 않았다. 사용한 파일은 train 5,196개, selection 421개, calibration 335개다. calibration을 학습·epoch 선택에 쓰지 않았다. 성분별 추가 학습 대상까지 포함하면 매 epoch 음악 373개, 음성 5,001개가 optimizer에 들어갔다. 각 epoch의 처리 행 수와 고유 행 수가 일치했다.

| selection 지표 | E00 max | E08 상위 30% 평균 |
|---|---:|---:|
| 음성 EER, 401개 | 0.0627014702 | 0.0501611762 |
| 음악 EER, 35개 | 0.1715686275 | 0.1715686275 |
| 파일 EER, 421개 | 0.0726953842 | 0.0627458694 |

이는 추출 순서상 앞부분만 포함한 부분 데이터 결과다. 원래 전체 selection 2,386개와 development 9,842개를 평가한 결과가 아니다. 음악 검증 표본이 작고 학습 데이터도 음성 fake에 치우쳐 있다. 음성 EER 관련 정답 5개, 음성 존재 정답 3개가 미확정이므로 평가기의 종합 score는 null이다. 독립 테스트·공식 점수·전체 데이터에서의 개선으로 해석하지 않는다. F0 보정과 E00/E04 전체 비교도 완료하지 않았다.

E08은 음악 27 epoch, 음성 26 epoch를 수행했고 각각 epoch 22, 21 checkpoint를 선택했다. 기준선은 음악 27 epoch, 음성 12 epoch를 수행하고 각각 epoch 22, 7을 선택했다. 최대 50 epoch, patience 5이며 시간 한도 때문에 잘린 학습이 아니다.

완료 캐시는 `objvol-a5gxdg7er70i`의 `lb-improvement-20260924/a100-partial-v1/completed_cache.zip`에 별도 보존했다. 크기 162,285,344 bytes, SHA-256 `cbd0230645b3f27886ec2d5201736f5d64943c781bf8c5af853b2344c11dd1ed`이며 클라우드 마운트에서 복사 후 다시 읽어 검증했다. 캐시와 네 개의 선택 checkpoint, 학습 재개 상태, 예측·지표를 포함한 최종 묶음은 `lb-improvement-20260924/a100-run-v1/results.zip`이다. 크기 167,713,133 bytes, SHA-256 `96e5e1139b73418e2274713310d554b546b6a015793524217bae67d76af0349e`다. 최종 묶음은 별도 CLI 다운로드 후 ZIP 전체와 내부 6,009개 파일의 해시를 검증했다. 각 모델의 checkpoint 해시와 결과 기록이 일치하며 저장된 예측에서 지표를 재계산해 동일함을 확인했다.

실행 중인 기존 추출기와 init script를 확인해 중단하고, 새 학습 프로세스를 별도 세션으로 시작했다. 기존 정지 제어 프로세스는 새 한도로 교체했다. SSH 호스트 키 검증이 일치하지 않아 SSH 실행은 중단했으며, VESSL이 해당 workspace에 제공한 HTTPS Jupyter API로 대상 경로와 프로세스를 확인·제어했다. 인증정보는 기록이나 Git에 포함하지 않았다. `evidence`의 실행 스크립트는 당시 PID·경로·절대 시각에 묶인 감사 기록이며 그대로 재실행할 수 있는 범용 실행기가 아니다.

[실제 비교·분할·checkpoint 검증](evidence/lb_a100_partial_train_20260924/integration_audit.json) · [전체 결과 다운로드 검증](evidence/lb_a100_partial_train_20260924/final_export_verified.json) · [workspace 정지 확인](evidence/lb_a100_partial_train_20260924/workspace_final.json) · [전환 실행 코드](evidence/lb_a100_partial_train_20260924/transition.py)
