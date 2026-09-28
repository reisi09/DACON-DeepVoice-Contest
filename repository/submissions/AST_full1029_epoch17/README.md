# AST 전체 데이터 최종 학습 · 1,029개

제출 파일은 이 폴더의 **`DACON_236749_AST_full1029_epoch17.zip` 하나**입니다. 학습과 로컬 오프라인 실행 검증을 마쳤습니다. 이후 사용자가 `DACON_Sumbit_1.zip`으로 제출한 파일의 해시가 이 ZIP과 같음을 확인했습니다. 제출 88673의 공식 점수는 **0.6174819841**, 소요 시간은 **1분 22초**입니다. 베이스라인 0.6909110688보다 낮으므로 현재 최고 후보는 베이스라인입니다.

[결과 Drive 폴더](https://drive.google.com/drive/folders/1ESqsFmC-P80kKzoHzGWgzoQAv-RjT0zH) · [Colab 재현 노트북](https://colab.research.google.com/drive/1nEl0_QkW4vPmC4and5aj376smkIRPjbM)

## 실제 학습 범위

기존 학습 633개, 개발 122개, 내부 평가 274개를 합쳐 **구축해 둔 공개 자료 1,029개 전부**를 학습에 사용했습니다. 대회의 비공개 평가 데이터 1,200개는 포함하지 않았습니다.

| 단계 | 학습 범위 | epoch | 파일별 참여 횟수 |
| --- | --- | ---: | ---: |
| Head warmup | AST를 고정하고 5개 분류 head 학습 | 8 | 모두 8회 |
| Fine-tuning | 마지막 encoder block·최종 LayerNorm·5개 head 학습 | 17 | 모두 17회 |

사전학습 AudioSet AST부터 새로 시작했습니다. 기존 633개로 학습한 epoch17 가중치를 이어 학습한 결과가 아닙니다. 각 epoch마다 고유 ID 1,029개를 정확히 한 번씩 처리했는지 검사했습니다. head 단계의 총 참여 횟수는 8,232회, 미세조정 단계는 17,493회입니다.

기존 모델의 입력·손실 규칙을 유지했습니다. 파일마다 고정된 최대 10초 crop과 gain augmentation을 사용하며, 모든 파일의 전체 시간 구간을 학습한 것은 아닙니다. 불명확하거나 해당 성분이 없는 라벨은 항목별로 손실에서 제외합니다. 손실 계산 대상 수는 파일 진위 1,029개, 음성 진위 624개, 음악 진위 501개, 음성 존재 756개, 음악 존재 1,029개입니다. 파일 자체를 학습 목록에서 제외하지는 않았습니다.

## 결과와 확인 범위

- 최종 head warmup의 평균 batch loss: `0.183903136`
- 미세조정 첫 epoch → 마지막 epoch의 평균 batch loss: `0.176974248` → `0.013916772`
- 완료한 재실행의 학습·패키징·검증 총시간: `36.96`분. 초기 오류 진단과 Drive 업로드 시간은 별도입니다.
- 최종 가중치의 모든 tensor가 유한하며, 동결된 backbone tensor 181개는 사전학습 가중치와 정확히 같습니다. 학습 대상 backbone의 변경과 5개 head의 미세조정도 확인했습니다.
- ZIP을 새 디렉터리에 풀고 네트워크를 차단한 별도 프로세스에서 `python script.py`를 실행했습니다. 모노·스테레오·60초 입력의 ID 순서와 확률이 원본 checkpoint 추론과 허용 오차 1e-5 안에서 일치했습니다. 최대 차이는 `3.5762786865234375e-07`입니다.

이번 학습에는 이전 개발·평가 자료까지 들어갔으므로 **독립 자체 평가 점수는 없습니다**. 이전 proxy나 이번 학습 loss를 새 모델의 공식 성능으로 해석할 수 없습니다. DACON 제출 화면에서 점수 0.6174819841과 소요 시간 1분 22초를 확인했습니다. 서버의 파일별 실행 로그는 확보하지 않았습니다.

Colab 브라우저 제어 제한으로 실제 재학습은 로컬 Mac GPU(MPS), Python 3.11·PyTorch 2.7.1에서 실행했습니다. MPS의 부분 LayerNorm backward 오류는 CPU 대조와 재현 테스트로 확인한 뒤 수정했습니다. 앞쪽 블록의 가중치는 계속 동결됩니다. Colab 노트북은 동일 작업의 재현용이며 이번에 Colab에서 실행한 기록은 아닙니다.

## 제출 파일

ZIP에는 `script.py`, `requirements.txt`, `model/ast/`의 최종 가중치·설정·라이선스, `model/runtime/`의 추론 코드가 들어 있습니다. 입력 오디오·학습 코드·학습 로그·기존 예측 CSV는 제출 ZIP에 넣지 않았습니다.

서버의 `data/sample_submission.csv` 순서대로 `data/test/`를 읽어 `output/submission.csv`를 생성합니다. 각 파일 안에서 최대 10초 구간별 확률을 평균하며, 다른 파일의 정보로 예측을 보정하거나 추론 중 모델을 학습하지 않습니다. 이는 [공식 규칙](https://dacon.io/competitions/official/236749/overview/rules)의 파일별 독립 예측 및 비공개 평가 자료 추가 학습 금지 조건에 맞춘 동작입니다.

ZIP 크기: `344836628` bytes

```text
ZIP SHA-256
af071e6090def81ac3074dea65235502ea35c2c9d59dc04e88aafdbc5c61d424

model.safetensors SHA-256
a6e97595b15e12322a5703561e8923b4723b9b515da3646bf37210b790539408
```

[DACON 제출 페이지](https://dacon.io/competitions/official/236749/mysubmission)에서 위 ZIP만 선택합니다. 메모는 `AST full public corpus refit; 1029 files; head 8 epochs + last-block 17 epochs; file-independent offline inference`를 사용하면 됩니다.

학습 증거는 `results/full_refit_20260913/`의 `full_training_inventory.json`, `warmup_report.json`, `finetune_report.json`, `epoch_history.csv`, `final_weight_audit.json`, `verification.json`, `training.log`에 저장했습니다. 이전 633개 학습 후보는 별도 폴더에 보존했습니다.

## Drive 분할 백업 복원

Drive 도구의 파일당 100MiB 업로드 제한 때문에 원본 ZIP은 `.part00`부터 `.part03`까지 4개로 나눠 보관했습니다. 로컬의 이 폴더에는 완성 ZIP이 있으므로 곧바로 제출에 사용할 수 있습니다. **분할 파일 자체를 DACON에 제출하면 안 됩니다.**

Drive에서 복원하려면 4개 part 파일과 `split_manifest.json`, `restore_submission.py`를 같은 폴더에 내려받고 Python 3.11 이상에서 다음 명령을 실행합니다. 추가 라이브러리나 GPU는 필요 없습니다. 각 part와 복원한 ZIP의 SHA-256을 확인한 뒤 완성 ZIP을 저장합니다.

```sh
python restore_submission.py
```

실제 분할 파일을 다시 합쳐 원본 ZIP과 같은 SHA-256이 나오는 것을 확인했습니다. 손상된 조각은 거부하고 기존 복원 중간 파일은 보존하는 동작도 검증했습니다.

실제 실행 결과를 검증한 노트북: `DACON_236749_full_refit_run_record.ipynb`. Colab 재현용 노트북과 구분해서 보관했습니다.
