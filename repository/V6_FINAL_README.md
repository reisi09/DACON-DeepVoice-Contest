# V6 최종 파이프라인

H0~H4 비교 계획을 대신하는 단일 구성이다.

`원본 → PANNs 존재 확률 / DF-Arena 원본 보조 점수`

`원본 → HTDemucs → 음성: XLS-R300M+AASIST LA / 음악: MERT-v1-95M + 학습 분류층`

`같은 시간 구간의 존재 확률 × 위조 확률 → 파일 단위 로지스틱 결합`

- 음성은 공개된 LA 학습 모델의 ONNX를 고정해 사용한다. DF 버전이나 새로 전체 미세조정한 음성 모델이 아니다.
- MERT는 24 kHz 입력을 정규화한 뒤 프레임 특징을 평균한다. 인코더는 고정하고 768차원 선형 음악 분류층을 300 epoch 학습한다.
- 파일 라벨을 모든 구간에 복제하지 않는다. 구간 로짓의 최댓값으로 파일 단위 BCE를 계산한다.
- PANNs, HTDemucs, 원본 DF-Arena도 고정한다. 큰 모델 추론 결과는 저장하고 분류층 학습에 재사용한다.
- 파일 결합기는 시간 정렬 증거 2개, 원본 위조 확률, 원본과 두 성분의 불일치 2개를 사용한다.
- 원본 그룹·동일 PCM·ID가 학습과 평가 사이에 겹치면 실행을 중단한다. 결합기 학습 자료도 음악 분류층 학습 자료와 분리한다.
- 기존 개발 평가 자료를 다시 사용한다. 새로운 비공개 검증이나 리더보드 성능 향상을 입증하는 자료는 아니다.

## H100 실행

1. `V6_H100_RunBundle.zip`과 `V6_H100_Experiment.ipynb`를 같은 Jupyter 폴더에 업로드한다.
2. 노트북의 VESSL 잔액, 실제 시간당 요금, GPU 재개 시각을 채운다. 첫 실행 연산 예산은 최대 $15이며, 기존 $20 시험 예산에서 잔액과 저장 비용을 고려한다.
3. 노트북을 위에서부터 실행한다. 입력 확인 → 의존성 설치 → 데이터 분할 → 특징 추출 → 분류층·결합기 학습 → 평가 → 제출 ZIP 생성 순서다.
4. 오류 또는 시간 한도에 도달하면 특징 캐시는 남는다. 같은 자산·코드·환경에서 다시 실행하면 완료된 파일은 건너뛴다.
5. 완료 후 결과를 내려받고 VESSL 워크스페이스를 Pause한다. **Python 작업 종료만으로 GPU 요금이 멈추지는 않는다.**

학습 산출물은 `v6_run/heads.npz`, `training.json`, `dev_metrics.json`, `dev_predictions.json`, `losses.json`이다. `progress.json`으로 진행 상태를 확인할 수 있다. 완전한 학습 메타데이터와 가중치 해시가 일치해야 제출 ZIP을 만들 수 있다.

## CLI

```bash
python run_v6_final.py prepare --work v6_run
python run_v6_final.py train --assets model/assets --work v6_run --device cuda --epochs 300 --max-hours 2
python package_v6_final.py --assets model/assets --work v6_run --output V6_final_submission.zip
```

제출 ZIP의 `script.py`는 공식 `--test-dir`, `--sample-submission`, `--output`, `--device` 인자를 받는다. 추론은 ZIP 안의 자산만 사용한다. 학습 데이터는 제출 ZIP에 포함하지 않는다.

## 검증 범위

`python -m unittest test_v6_final`은 시간 정렬 계산, 파일 라벨의 gradient 연결, 분류층 최적화, 원본/PCM 누수 차단, 변경된 캐시 거부, 저장·재로딩 일치를 확인한다. 실제 음원을 통한 전체 경로 점검 결과는 `.omo/evidence/v6-final/`에 별도로 기록한다.

H100에서 학습을 완료해도 심사 환경 L4의 60분 제한을 통과했다는 뜻은 아니다. 새 모델과 음악 분류층의 효과는 실행 후 개발 지표와 실제 제출 점수로 판단해야 한다.

## 출처

- 음성: [공개 LA 체크포인트와 모델 설명](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST/tree/196128e5a5101d5cb6ac7701597891bc7de7e7b5), [원본 코드](https://github.com/TakHemlata/SSL_Anti-spoofing/tree/4acaa61dcef5f7610f43aa4d0b29c4559b970cd2)
- 음악: [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5), CC-BY-NC-4.0
- PANNs·HTDemucs·DF-Arena: 주최 측 제공 베이스라인. 원본 자산과 라이선스 파일을 포함한다.

자산별 SHA-256과 출처는 `model/assets/assets.json`에 기록한다. 공개 모델을 포함했다는 사실만으로 대회 규정 적합성을 새로 인증하지는 않는다.
