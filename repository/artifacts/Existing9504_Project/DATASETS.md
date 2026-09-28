# 데이터 목록과 분할 설명

## 데이터 규모

이 패키지에는 실제 오디오 대신 CSV·JSON 목록을 넣었습니다. 최종 후보 목록은 59,770행이며, 상태는 편입 가능(`admitted`) 43,603행, 진단용(`diagnostic`) 16,042행, 제외(`excluded`) 125행입니다. 편입 가능한 행 중 9,504개를 사용했고 34,099개는 예비 목록에 남았습니다. **목록에 기록된 모든 행을 학습한 것은 아닙니다.**

선택 원본의 크기 합은 14,939,622,208바이트였습니다. 이는 선택된 오디오의 합계이며, 이번 ZIP의 용량이나 전체 수집 용량과 다릅니다.

| 구분 | train | val | test | 합계 |
|---|---:|---:|---:|---:|
| 전체 | 6,652 | 1,426 | 1,426 | 9,504 |
| real | 3,326 | 713 | 713 | 4,752 |
| fake | 3,326 | 713 | 713 | 4,752 |
| 일반 발화 | 2,034 | 746 | 744 | 3,524 |
| 가창 | 2,034 | 0 | 0 | 2,034 |
| 음악 | 2,584 | 680 | 682 | 3,946 |
| 과거 개발 노출 | 4,176 | 997 | 1,001 | 6,174 |

## 출처별 선택 수

아래는 최종 `selected.csv`의 `source` 값을 그대로 집계한 결과입니다. `FMA`와 `FMA/FMA`처럼 다른 이름으로 기록된 항목은 임의로 합치지 않았습니다. real/fake는 파일 라벨 기준입니다. 생성기별 상세 목록은 [source_generator_counts.csv](data/source_generator_counts.csv)에 있습니다.

| 출처 | train | val | test | real | fake | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| asvspoof2019_la | 1,457 | 746 | 744 | 1,185 | 1,762 | 2,947 |
| CtrSVDD | 2,034 | 0 | 0 | 1,017 | 1,017 | 2,034 |
| HAIM | 619 | 176 | 171 | 465 | 501 | 966 |
| JamendoMaxCaps | 565 | 144 | 145 | 854 | 0 | 854 |
| FakeMusicCaps | 400 | 105 | 105 | 0 | 610 | 610 |
| FMA | 393 | 103 | 103 | 599 | 0 | 599 |
| librispeech_test-clean | 577 | 0 | 0 | 577 | 0 | 577 |
| SONICS | 213 | 148 | 149 | 0 | 510 | 510 |
| FMA/FMA | 42 | 4 | 9 | 55 | 0 | 55 |
| Echoes/acestep/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/audioldm/ATA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/audioldm/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/brev/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/diffrhythm/ATA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/diffrhythm/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/elevenlabs/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/musicgen/ATA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/musicgen/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/producer/ATA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/producer/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/songgen/ATA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/songgen/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/suno/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/udio/TTA | 20 | 0 | 0 | 0 | 20 | 20 |
| suno | 20 | 0 | 0 | 0 | 20 | 20 |
| udio | 20 | 0 | 0 | 0 | 20 | 20 |
| Echoes/mubert/TTA | 6 | 0 | 0 | 0 | 6 | 6 |
| Echoes/stableaudio/TTA | 6 | 0 | 0 | 0 | 6 | 6 |

## 분할 정책

전체 real/fake 1:1을 유지하고 알려진 부모·화자·원본 관계로 연결된 `component_id`를 split 사이에 겹치지 않게 배치했습니다. 이번 패키지 제작에서도 최종 CSV의 ID 집합, 각 split의 class 수, 알려진 component 중복, 원본 CSV의 SHA-256을 다시 확인했습니다.

가창은 원본·가수 관계의 남은 불확실성 때문에 train 전용입니다. 가창 일반화 성능을 val/test로 검증한 것은 아닙니다. 발화 평가에는 ASV 계열의 real/fake를 함께 사용하며 LibriSpeech는 train에만 들어갑니다. 각 출처·생성기가 모든 split에 존재하는 정책은 아닙니다.

선별은 데이터·그룹 조건을 사용했고 모델 예측이나 이번 test 점수로 조정하지 않았습니다. 다만 과거 개발 노출은 남아 있어 독립 평가로 해석할 수 없습니다. 음악 출처·길이 편향도 남아 있습니다.

PartialSpoof, HAIM C1/C2, CtrSVDD SVC 등 보류 항목은 원본 확보 여부와 학습 편입 여부를 구분해야 합니다. 부모 관계·구간 라벨·대응 real 대조군 등의 미해결 항목은 편입하지 않았습니다. 행별 상태와 사유는 후보 목록의 `admission`, `reason`, `exclusion_reason`, `rejection_reason`에 있습니다. 빈칸은 해당 정보가 없다는 뜻이며 자동으로 통과를 의미하지 않습니다.

## 파일 읽는 법

- [selected_list.csv](data/selected_list.csv): 이번 실험의 9,504개 선택 행을 핵심 열로 정리했습니다.
- [catalog_list.csv](data/catalog_list.csv): 59,770개 후보의 출처·상태·선택 여부·사유를 정리했습니다.
- [source_counts.csv](data/source_counts.csv): 출처별 train/val/test와 real/fake 수입니다.
- [source_generator_counts.csv](data/source_generator_counts.csv): 출처·생성기·split·라벨 조합별 수입니다.
- [summary.json](data/summary.json): 전체 수, 노출 수, 성분별 마스크 집계입니다.
- [runtime_manifest.csv](data/runtime_manifest.csv): 완료 H100 Handoff에 들어 있는 실제 실행 목록입니다. 원격 작업 경로가 포함돼 있습니다.
- `repository/results/existing_only_20260919_final/`: 원본 `selected.csv`, `train.csv`, `val.csv`, `test.csv`, `combined_catalog.csv`, 검증·출처 기록을 수정 없이 보존했습니다.

**최종 분할은 원본 `split` 열과 train/val/test CSV의 실제 소속이 기준입니다.** `assigned_group_split`, `previous_split`, `prior_split`은 이전 단계의 배치 정보가 남아 있을 수 있으므로 현재 split으로 사용하면 안 됩니다. 읽기용 목록에는 최종 `split`에서 만든 `current_split` 열을 넣었고, 미선택 후보는 이 열을 비웠습니다.

## 주요 열

| 열 | 의미 |
|---|---|
| `ID` | 행 식별자. 파생본·다른 출처를 파일명만으로 합치지 않습니다. |
| `current_split` | 최종 train/val/test. 미선택 후보는 빈칸입니다. |
| `source`, `source_corpus`, `generator` | 기록된 출처·하위 코퍼스·생성기. 빈 생성기 값을 임의의 모델명으로 채우지 않았습니다. |
| `route` | 분류기 학습 경로인 voice 또는 music. 실제 성분 존재 라벨과는 다른 개념입니다. |
| `kind` | 발화·가창·음악 등의 추가 설명. 미기록 행도 있습니다. |
| `FILE_FAKE`, `VOICE_FAKE`, `MUSIC_FAKE` | 알려진 진위 라벨. 0은 real, 1은 fake입니다. |
| `VOICE_PRESENT`, `MUSIC_PRESENT` | 알려진 성분 존재 라벨. 0은 부재, 1은 존재입니다. |
| `*_mask` | 1인 라벨만 유효합니다. 0인 라벨을 real/부재로 해석하면 안 됩니다. |
| `component_id` | 알려진 부모·화자 등의 관계를 연결한 분할 단위입니다. |
| `group_id`, `parent_id`, `speaker_id` | 출처에서 확보한 그룹·원본·화자 식별 정보입니다. |
| `historical_development_exposure` | 과거 개발 노출 기록. False도 사전학습 미노출을 보증하지 않습니다. |
| `license`, `license_url`, `source_url` | 확보 당시 기록한 라이선스·출처. 이번 패키지 제작에서 외부 약관을 새로 검증한 것은 아닙니다. |
| `path`, `archive_path`, `archive_member` | 원본 또는 아카이브 내 위치를 기록한 문자열. 실제 오디오는 포함하지 않았습니다. |
| `audio_sha256`, `source_sha256` | 확보된 원본 해시. 비어 있는 열이 있으면 원본 manifest와 provenance를 함께 확인합니다. |
| `admission`, `selected`, `reason` | 편입 상태, 최종 선택 여부, 보류·예비 사유입니다. |

원본 CSV에는 출처별로 더 많은 열이 있습니다. 위 목록은 분석하기 쉽게 줄인 보기이며, 원본을 대체하거나 라벨을 수정한 파일이 아닙니다. 파일별 검증 해시는 `PACKAGE_MANIFEST.json`과 원본 `provenance.json`에 있습니다.
