# 추가 수집 후 CPU 작업 결과 — 2026-09-24

추가 원본 2,800개와 이전 LibriSpeech 파일럿 200개를 합쳐 **3,000개 후보**를 검토 목록으로 정리했다. 기존 42,947행의 출처 관계를 확장하고 역할 충돌을 격리한 `roles-v4`와 학습 입력 CSV를 만들었다. GPU 실행·새 유료 자원·공식 제출은 없었다. 기존 사용자 수정 9개는 SHA-256으로 보존을 확인했다.

다음 기준 실험은 이 문서의 `roles-v4`를 사용한다. `roles-v3`와 이전 결과는 과거 기록으로 보존한다. 새 후보 3,000개는 아직 학습·평가 입력에 편입하지 않는다.

## 고친 문제와 실제 검증

### 미승인 후보가 학습 입력으로 나가는 문제

기존 exporter는 `training_admitted=false`라도 분할값과 라벨이 있으면 CSV로 내보냈다. 실패하는 회귀 검증으로 재현한 뒤 명시적인 미승인 상태를 차단했다. 승인된 자료와 해당 필드가 없던 기존 자료는 그대로 처리한다. 실제 후보 3,000개에 모두 train 분할을 주는 오입력에서도 전부 제외되는 것을 확인했다.

### 원문만으로는 잡히지 않던 참조 화자 연결

고정 revision의 영어 MLAAD 메타데이터로 기존 2,482행과 새 400행의 참조 기록을 연결했다. M-AILABS의 참조 클립·화자 경로와 같은 시스템의 명명된 voice를 그룹에 추가했다. 이는 보수적인 출처 묶음이며, 출력 음성이 실제로 같은 인물처럼 들린다는 측정은 아니다. LibriSpeech의 기존 화자 키 별칭과 PCM 동일성도 반영했다.

- 새 MLAAD **247개**가 기존 자료와 연결됐다. 이 중 **216개**는 기존 평가 역할을 포함한 그룹과 연결되고, **31개**는 기존 train에만 연결된다.
- 확장된 관계에서 기존 **37개 그룹·4,054행**의 역할이 충돌했다. 해당 그룹을 development로 모았고, 이미 development였던 자료를 제외하면 **1,603행**의 역할이 바뀌었다.
- 기존 42,947행의 ID·오디오 경로·SHA-256·다섯 라벨은 전부 유지했다. 역할별 그룹 중복은 **0개**다.
- 실제 `full_bank_features.load_rows`로 train/selection/calibration 입력 **32,197행**, development 입력 **9,842행**을 읽었다. 두 입력의 그룹 교집합도 0개다. 학습 가능한 head 라벨이 없는 행은 기존 exporter 규칙대로 별도 기록한다.

| 역할 | roles-v3 | roles-v4 |
|---|---:|---:|
| train | 29,643 | 28,319 |
| selection | 2,585 | 2,435 |
| calibration | 2,202 | 2,073 |
| development | 8,517 | 10,120 |
| 합계 | 42,947 | 42,947 |

영어 메타데이터가 없는 기존 시스템·언어와 시스템 간 voice 별칭은 미해결이다. 새 자료에도 해석되지 않은 참조 표기 5개, 정체가 확인되지 않은 preview 생성기 자료 48개가 남았다. `sealed`는 만들지 않았다.

### 파일 형식으로 정답을 구별할 수 있는 문제

실제 발화 600개는 FLAC, 합성 발화 400개는 WAV다. 기존 `split.metadata_probe`로 길이·파일 형식 등 비내용 정보만 사용한 grouped CV를 실행했고, voice 5개 fold의 AUC가 모두 **1.0**이었다. 소리 탐지 모델의 성능이 아니라 수집 원천과 라벨이 맞물린 지름길의 증거다. 음악은 음성 존재 라벨이 불명확해 이 probe에서는 평가하지 않았다.

학습용 발화는 같은 디코딩·저장 형식을 적용하고, 실·합성 양쪽의 길이 분포도 맞춰야 한다. 같은 WAV로 저장하는 것만으로 기존 codec 흔적이나 원천 차이까지 없어진다고 보지 않는다. 변환·혼합 오디오는 이번 작업에서 만들지 않았다.

## 캐시 확인과 보존

로컬에서 과거 `features.zip`을 찾았다. 저장 영수증과 파일 전체 SHA-256이 일치했고, **1,322개 NPZ·3,966개 window**를 CPU로 검사했다. 수치 유한성, window 범위·padding·mask, 배열과 producer identity, MERT 정밀도 및 자산·생성 코드 기록을 확인했다. pickle과 모델 코드는 실행하지 않았다.

이 백업에는 MERT L6/L12/L24와 stem spectrum이 있다. 현재 42,947행의 오디오 SHA-256과 정확히 일치하는 항목은 **0개**이며, 현재 full-bank에 필요한 `voice_embeddings`와 `presence`도 없다. 따라서 복구한 파일을 보존했지만 현재 학습 캐시로 승격하지 않았다. 파일명이나 metadata hash만 바꿔 연결하지 않는다.

지정한 VESSL object prefix와 로컬 `results/`, `_workspace/`를 조사했다. 이 범위에서 DataV4 전체 특징 캐시는 찾지 못했다. **중지된 workspace의 cluster filesystem은 검사하지 않았으므로 그곳의 캐시 유무는 미확인**이다. VESSL의 `wsp-9683432r72m7`는 조회 시 `stopped`였다.

## 검토 자료와 이용 조건

`candidate_manifest.jsonl`에는 3,000개의 출처 그룹과 검토 사유를 기록했다. `review_queue.csv`는 전체 검토 목록이며, `listening_sample.csv`에는 원천별 20개씩 서로 다른 그룹에서 고른 **80개 청취 표본**을 넣었다. 판정 칸은 비어 있다. 실제 청취를 수행했다고 주장하지 않는다. 이전 파일럿 200개도 다시 전체 디코딩해 새 400개 실제 발화 및 기존 manifest와 정확한 중복이 없음을 확인했다.

기존 기술 검토 표시 229개는 유지한다. 음악 2,000개의 음성 존재·음성 진위 라벨은 미확인 상태다. 수집과 출처 검토만으로 라벨·음질·학습 적합성을 승인하지 않는다.

[대회 규칙](https://dacon.io/competitions/official/236749/overview/rules)은 공개 접근 가능하고 최소 비영리 사용을 허용한 자원을 자체 이용 조건에 따라 사용할 수 있도록 하며, 2차 평가에는 출처와 학습 데이터 제공을 요구한다. 원천별로 다음 근거와 남은 조건을 기록했다.

- [LibriSpeech](https://www.openslr.org/12/)의 CC BY 4.0 표기와 [라이선스의 공유·변환 및 출처 표시 조건](https://creativecommons.org/licenses/by/4.0/)을 확인했다.
- Jamendo는 고정 메타데이터의 곡별 CC BY/CC BY-SA 기록을 보존했다. 전체 곡의 현재 권리 상태를 새로 확인한 것은 아니다. [BY-SA 조건](https://creativecommons.org/licenses/by-sa/3.0/)은 변환 음원을 배포할 때도 검토해야 한다.
- [Music Arena](https://huggingface.co/datasets/music-arena/music-arena-dataset)의 CC BY 표기는 확인했지만, 원서비스와 익명 생성기의 출처 조건 검토는 남았다.
- [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD)는 CC BY-NC 외에 비상업 학술 연구 용도를 명시한다. 대회의 비영리 자원 허용 조항만으로 이 추가 조건과 검증자 제공 조건을 충족했다고 판단하지 않았다. 새 접근 동의는 하지 않았다.

## 전달과 재현

목적지는 기존 `qwdkw / objvol-a5gxdg7er70i`의 `lb-improvement-20260924/cpu-followup-v1/`다.

- `cpu-review-v1.tar.gz`: 새 역할표·bank, 검토 목록, 80개 표본 목록, 검증·권리 기록, 재현용 메타데이터 및 입력 출처 목록. 명시한 173개 archive member의 해시를 대조했다.
- `legacy-mert-features.zip`: 위에서 검증한 58,617,233-byte 과거 음악 특징 백업.
- 클라우드 재다운로드 검증: **두 파일 모두 완료, 전체 바이트 크기와 SHA-256 일치**.

코드 commit은 출처·미승인 차단 `b6f0b46`, 캐시 감사 `14f032b`다. 기존 `.venv`와 설치된 라이브러리만 사용했다. 새 검증은 미승인 자료의 bank 진입, 참조 화자/PCM에 의한 평가 누출, 역할 격리와 원본 보존을 보호한다. 캐시 검사는 실제 백업 전체를 읽어 수행했다. 이는 새 모델의 품질이나 LB 개선 검증이 아니다.

재현 명령과 경로 설명은 전달 묶음의 `review/README.txt`에 있다. 원본 `roles-v3`는 기존 VESSL 경로에 보존돼 있다. 산출물 로컬 위치는 `_workspace/lb_cpu_followup_20260924/`다.

근거: [출처 검토 집계](evidence/lb_cpu_followup_20260924/review_summary.json), [실제 입력 검증](evidence/lb_cpu_followup_20260924/integration_verification.json), [메타데이터 지름길 진단](evidence/lb_cpu_followup_20260924/metadata_shortcut_probe.json), [캐시 감사](evidence/lb_cpu_followup_20260924/feature_backup_audit.json), [이용 조건 조사](evidence/lb_cpu_followup_20260924/rights_review.json), [전송 검증](evidence/lb_cpu_followup_20260924/transfers.json).
