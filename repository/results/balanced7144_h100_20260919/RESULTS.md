# Balanced7144 H100 결과

학습과 검증 결과 회수를 완료했다. 승인한 7,144개 전부에서 특징을 추출했으며, train 5,716개·val 1,428개를 사용했다. train은 real/fake 각 2,858개, val은 각 714개다. 음성·음악 각각의 진위 비율은 1:1이 아니다.

HTDemucs base, frozen MERT330M layer12 + WindowMIL, 기존 DF-Arena fc5, PANNs 구조를 유지했다. seeds 41/42/43의 예측 확률을 평균하고, 파일 점수는 두 경로의 존재 확률과 가짜 확률을 곱한 값 중 최댓값으로 계산한다.

| 측정 항목 | 검증 결과 | 평가 파일 수 |
|---|---:|---:|
| 음성 EER | 1.04935% | 678 |
| 음악 EER | 8.65731% | 750 |
| 파일 EER | 9.38375% | 1,428 |
| 음악 존재 AUC | 0.998783 | 1,428 |

음악 체크포인트는 seed 순서대로 11/13/11 epoch에서 선택했다. 실제 학습은 16/18/16 epoch에서 조기 종료됐다. 음성은 세 seed 모두 50 epoch를 수행했고 마지막 epoch가 선택됐다. 각 epoch에서 음악 학습 파일 3,000개와 음성 학습 파일 2,716개가 정확히 한 번씩 사용됐음을 기록과 체크포인트의 train_ids로 확인했다. 두 경로 모두 실측 microbatch와 effective batch는 256이었다.

음악 모델은 선택 epoch 이후 검증 BCE가 상승했다. 최저 검증 EER, 동률이면 BCE로 고른 이전 체크포인트를 제출용으로 사용한다. 음성은 마지막 epoch까지 검증 BCE가 감소했다. 이는 이번 검증셋에서 관측한 추세이며, 과적합이 없다는 증거는 아니다.

검증 파일 561개는 과거 실험 노출 이력이 있어 독립 테스트로 볼 수 없다. 음악 파일 750개의 음성 관련 정답이 없고, 정답이 있는 음성 존재 항목에는 양성만 있어 음성 존재 AUC와 공식 종합 점수는 계산하지 않았다. 공식 리더보드 개선이나 L4 실행을 검증한 결과가 아니다.

완료 증거: completion_audit.json, downloaded/verification.json, downloaded/unpacked/output/train/results.json, downloaded/unpacked/output/train/validation_metrics.json. 학습 후 다운로드를 해시·CRC로 검증하고 GPU 정지를 확인했다(resource_stop.json).

제출 파일: submissions/Balanced7144_MIL/submit.zip (저장소 루트 기준). 6,347,503,163 bytes. SHA-256: 665cb1b72b0a9823532d76f1eda8ffd9b00c4cbd358a6e96f2c170261ad059d7.

제출 패키지 추론 검증을 완료했다. 로컬 ZIP과 SHA-256이 같은 ZIP을 H100에서 재구성·추출한 후 실제 script.py를 Python 네트워크 연결 차단 상태에서 실행했다. 원본 5개(디코딩 오류 이력이 있는 MP3 포함)와 역순 부분집합 2개의 캐시 예측 대비 최대 절대 오차는 모두 0이었다. 기존 gain을 유지하는 2개는 패키지 Engine의 preserve_gain=True 경로로 별도 검사했고, 오차 역시 0이었다. 사전에 정한 절대 허용 오차는 0.002, 상대 허용 오차는 0이다. 이 검사는 패키지의 수치적 일치를 확인하며, 미지 데이터 성능이나 L4 적합성을 증명하지 않는다.

검증 증거 ZIP을 SHA-256·CRC로 확인한 뒤 같은 H100의 stopped/standby 상태를 재확인했다. QA 임시 package/extracted/submit.zip 복사본을 제거했으며, 학습 체크포인트·원본 데이터·캐시는 보존했다. 증거는 submission_qa2/smoke_result.json, download_receipt.json, resource_stop.json에 있다. 제출 ZIP은 준비됐지만 DACON에는 제출하지 않았다.
