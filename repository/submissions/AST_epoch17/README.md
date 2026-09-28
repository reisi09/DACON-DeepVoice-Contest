# 제출 준비: AST 17 epoch

DACON에 업로드할 파일은 **`DACON_236749_AST_epoch17.zip` 하나**다. 개발 평가 122개에서 최고 proxy `0.9038005610039294`를 기록한 17 epoch 가중치를 사용한다. 22 epoch는 조기 종료 시점이며 제출 가중치가 아니다.

[제출 전용 Drive 폴더](https://drive.google.com/drive/folders/1M5-JhWlZxFzTBq6bziU7Ff86CsWBSQSI) · [제출 ZIP](https://drive.google.com/file/d/1ROtdWWWFhjs-iCpLABWiPHa3AtDwjbog/view)

## 정리한 파일

| 파일 | 용도 | DACON 업로드 |
| --- | --- | --- |
| `DACON_236749_AST_epoch17.zip` | 실제 추론 코드와 17 epoch 가중치, Drive에 보관 | 이 파일만 |
| `README.md` | 제출 방법과 모델 선택 근거 | 제외 |
| `submission_manifest.json` | 모델·원본·복사본 식별자와 검증 기록 | 제외 |

로컬의 `DACON_236749_AST_epoch17.webloc`은 ZIP으로 연결되는 링크다. 업로드할 ZIP 자체가 아니다. Drive 도구의 다운로드 한도 256MiB를 넘어 ZIP은 Drive 안에서 복사했으며, 원본을 보존했다. 새 폴더에는 제출 후보 하나만 두었다.

## ZIP 구성과 실행

검증에 사용한 패키지 생성 코드와 실행 기록에 따른 구성이다. 파일 이름만 명확하게 바꿔 Drive에서 복사했으며 ZIP을 다시 만들거나 모델을 바꾸지 않았다.

```text
DACON_236749_AST_epoch17.zip
├── script.py
├── requirements.txt
└── model/
    ├── ast/
    │   ├── model.safetensors
    │   ├── config.json
    │   ├── preprocessor_config.json
    │   ├── multitask_config.json
    │   ├── MODEL_INFO.json
    │   └── LICENSE.txt
    └── runtime/
        ├── ast_multitask.py
        └── v2_inference.py
```

서버에서 `python script.py`를 실행하면 제공된 `data/sample_submission.csv`의 ID 순서대로 `data/test/`의 오디오를 읽고 `output/submission.csv`를 만든다. 출력은 ID와 파일 진위·음성 진위·음악 진위·음성 존재·음악 존재 확률 5개다. 입력 데이터와 완성된 예측 CSV를 제출 ZIP에 추가하지 않는다.

모델은 AST 단독이다. G0 또는 G0+AST 결합 모델이 아니다. 오디오를 모노 16kHz로 변환하고 파일 내부의 최대 10초 구간들을 추론한 뒤 확률을 평균한다. 파일 간 통계·예측 공유나 추론 중 학습은 없다. 가중치와 설정을 로컬 경로로 읽으며, 패키지 설치 후 네트워크가 없어도 동작하도록 구성했다. `requirements.txt`는 비어 있다. 실행에 필요한 라이브러리는 확인해 둔 대회 기본 환경에 포함돼 있다.

## 제출 순서

1. 위 Drive ZIP을 로컬에 다운로드한다. 크기는 `344,836,171` bytes다.
2. 아래 SHA-256과 다운로드 파일의 해시가 같은지 확인한다.
3. [DACON 코드 제출 페이지](https://dacon.io/competitions/official/236749/mysubmission)에서 이 ZIP 한 개를 선택한다. 현재 남은 제출 횟수와 선택된 파일명을 확인한다.
4. 메모는 `AST last-block fine-tuning; best dev checkpoint epoch 17; train 633 / dev 122; offline inference`로 입력한다.
5. 제출한 뒤 접수 번호·파일명·상태를 기록한다. 채점 완료 후 공식 점수와 기존 베이스라인 `0.6909110688`을 비교한다.

```text
ZIP SHA-256
5aa3c61dc017a85cdf795de5b730da85d461d51ec3c4b9f731dfe54ef43bc712

model.safetensors SHA-256
b51e8645730626c2b5c26647e4fb5d02b5d117a542ac2d71c77ee9cde282a107
```

## 확인된 범위와 현재 상태

원본 ZIP은 Colab A100에서 새 디렉터리에 풀어 네트워크를 차단한 별도 프로세스로 실행했다. 형식 확인용 3개와 60초 stress 파일의 예측이 기준 결과와 일치했고 최대 차이는 `0.0`이었다. ZIP 크기와 가중치 해시는 해당 실행 기록에서 가져왔다. 이번 Drive 복사본은 메타데이터로 이름·부모 폴더·크기를 확인하며, 내려받은 바이트의 해시는 제출 직전에 다시 확인한다.

실제 DACON의 L4·Python 3.11·PyTorch 2.7.1 환경에서 전체 1,200개 추론은 아직 검증하지 않았다. 기존 평가 274개의 AST proxy는 `0.775809`, 결합 모델은 `0.792205`였으므로, 이번 파일을 모든 평가에서 최고인 모델이라고 부르지는 않는다. 사용자의 최고 epoch 제출 요청에 따라 개발 최고점의 AST 후보를 준비했다. 원래 G0 선택 기록은 그대로 보존한다.

**현재 상태는 파일 준비 완료, 미제출이다.** Computer Use의 이전 접근 제한은 해제된 것으로 간주하지 않으며, 이 정리 작업에서는 브라우저 제어·업로드·제출을 시도하지 않았다.

원본 실험과 검증 자료: [실험 보고서](../../RESULTS_EPOCH_EXTENSION.md), [패키지 검증](../../results/epoch_extension_20260913_013624/packaged_verification.json), [checkpoint 기록](../../results/epoch_extension_20260913_013624/candidate_lock.json).
