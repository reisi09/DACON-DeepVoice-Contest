# 7,144개 균형 분할 결과

2026-09-19 KST. 사용자 선택에 따라 **전체 real:fake 1:1과 출처·생성기별 train:val 8:2를 우선**했다. 음성·음악 각각의 real:fake를 1:1로 맞춘 분할은 아니다. 기존 16,042개 은행에서 선별했으며 새 데이터 수집·오디오 가공·학습·업로드는 하지 않았다.

| 구분 | real | fake | 합계 |
|---|---:|---:|---:|
| train | 2,858 | 2,858 | **5,716** |
| val | 714 | 714 | **1,428** |
| 전체 | **3,572** | **3,572** | **7,144** |

정수 단위로 나누어 실제 비율은 train 80.0112%, val 19.9888%다. 모든 출처·생성기·진위 조합 52개에 train과 val이 모두 있다. 각 조합의 val 수는 전체 수의 20%에서 최대 0.4개만 차이 난다. 표본이 6개인 희소 생성기는 5:1로 배정되므로 작은 셀의 검증 비율은 16.67%다. 그 1개로 생성기 성능을 안정적으로 추정할 수 있다는 뜻은 아니다.

## 출처별 구성

| 출처 | train | val | 전체 |
|---|---:|---:|---:|
| ASVspoof2019 LA | 1,890 | 472 | 2,362 |
| LibriSpeech test-clean | 826 | 206 | 1,032 |
| JamendoMaxCaps | 683 | 171 | 854 |
| Echoes | 515 | 129 | 644 |
| FMA | 479 | 120 | 599 |
| HAIM | 415 | 103 | 518 |
| FakeMusicCaps | 414 | 104 | 518 |
| SONICS | 412 | 102 | 514 |
| 기타 기존 크롭(FMA/FMA, suno, udio) | 82 | 21 | 103 |
| 합계 | **5,716** | **1,428** | **7,144** |

ASVspoof의 가짜 음성은 **A01~A19 각각 train 56개·val 14개**, 총 1,330개다. 특정 공격 유형이 검증에서 사라지지 않도록 했다. SONICS는 기존의 2,929:3 분할에서 이번 412:102로 바뀌었다. 같은 표본 집합에 대한 비율 변화는 아니며, 이번에는 선별된 514개만 사용한다.

## 분야별 라벨 구성

| 분야·라벨 | train | val | 전체 |
|---|---:|---:|---:|
| 음성 real | 1,652 | 412 | 2,064 |
| 음성 fake | 1,064 | 266 | 1,330 |
| 음악 real | 1,206 | 302 | 1,508 |
| 음악 fake | 1,794 | 448 | 2,242 |

음성은 3,394개, 음악은 3,750개다. 여기서 음성·음악은 기존 학습 경로를 뜻하며, 파일에 다른 성분이 없다고 확정한 분류가 아니다. 기존의 미확인 보컬 라벨과 마스크는 그대로 보존했다.

## 선별·분할 기준

- 기존 합성물 `constructed_music` 128개는 이번 선별에서 보류했다. 이 합성물들은 여러 부모를 하나의 큰 연결 그룹으로 묶어 출처별 비율을 맞추기 어렵게 한다. 파일을 삭제하지 않았으며, 향후 필요하면 같은 분할의 부모끼리 합성할 수 있다.
- 남은 15,914개에서 원본·파생물·화자·아티스트·프롬프트·원음 참조·해시를 기준으로 연결 그룹을 다시 계산했다. 기존 합성물만으로 연결되던 서로 다른 부모는 합성물을 사용하지 않으므로 같은 그룹으로 강제하지 않았다. 원본을 공유하는 나머지 연결은 유지했다.
- 부족한 실제 음악과 희소 음악 생성기를 우선 보존했다. 실제 음성은 ASVspoof와 LibriSpeech를 각각 1,032개로 맞췄다. 큰 가짜 음악 원천 HAIM·FakeMusicCaps·SONICS는 비슷한 수로 줄였다.
- 그룹의 train/val 배정과 그룹 안의 선별 수를 함께 최적화했다. 같은 그룹의 일부만 사용할 수 있지만, 사용한 파일은 모두 같은 분할에만 들어간다. 모델 예측값이나 평가 점수는 선별에 사용하지 않았다.
- 안정적인 해시 순서를 사용했다. 입력 행 순서를 뒤집어 다시 계산해도 선택 ID와 분할이 같았다.

기존 은행 16,042개는 **선택 7,144개 + 보류 128개 + 나머지 미선택 8,770개**로 전부 기록했다. 미선택 파일도 삭제하지 않았다.

## 검증 결과와 한계

선택한 **7,144개 오디오의 SHA-256을 실제 파일에서 전수 확인**했다. LibriSpeech는 기존 tar.gz 안의 선택된 멤버를 읽어 검증했다. 오디오 원본 바이트 합은 **10,345,454,546바이트(10.35 GB, 약 9.64 GiB)**이며 길이는 약 **122.69시간**이다. 이 값은 선택된 원본의 합계로, 기존 은행 전체의 저장공간이나 새로 복사한 데이터 용량을 뜻하지 않는다. 이번 작업은 목록 파일만 추가했다.

선택 그룹은 2,657개(train 2,276개·val 381개)다. 정의한 연결 식별자와 정확 오디오 해시의 train/val 교차는 0이었다. 분할 함수와 별도로 실제 CSV의 음성 화자, 음악 아티스트, 원본·PCM 해시도 대조했다. 알 수 없는 동일인·유사 음원까지 전부 배제했다는 보장은 아니다.

**독립 테스트셋은 아니다.** 선택 자료 2,912개, 그중 val 561개에 `prior_seen=True` 이력이 있다. 이 표시는 과거 학습·평가 또는 원천 연결 이력을 넓게 잡은 값이며 모두 과거 평가 행이라는 뜻은 아니다. 기존 전문가 모델의 사전학습 노출도 완전히 확인되지 않았다.

원본 manifest 해시는 변경 전후 같았다. 기존 full-bank 실험 설정과 배포는 교체하지 않았다. VESSL `wsp-9683432r72m7`는 확인 시 `stopped / standby`, 자동 재개는 `PAUSED`였다. 사용자가 요청한 분할·보고 단계에서 멈춘 상태다.

## 파일과 재현

- [train.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/balanced7144_20260919/train.csv): 5,716개 학습 목록
- [val.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/balanced7144_20260919/val.csv): 1,428개 검증 목록
- [selected.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/balanced7144_20260919/selected.csv): 전체 7,144개와 새 분할
- [source_summary.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/balanced7144_20260919/source_summary.csv): 출처·생성기·진위별 상세 수
- [selection_ledger.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/balanced7144_20260919/selection_ledger.csv): 기존 16,042개 각각의 선택 여부·사유
- [split_summary.json](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/balanced7144_20260919/split_summary.json): 정책·집계·CSV 해시·오디오 검증

CSV의 `path`는 `/Users/choiwuseck/Documents/ChatGPT/데이콘`을 기준으로 한다. `path`가 빈 LibriSpeech 행은 `archive_path`와 `archive_member`로 원본을 찾는다. `previous_split`과 `previous_component_id`는 직전 full-bank 분할을 보존하고, `prior_split`은 그보다 앞선 이력이다.

재현 스크립트: [balanced_bank_split.py](/Users/choiwuseck/Documents/ChatGPT/데이콘/balanced_bank_split.py). 기존 프로젝트 환경에서 아래 명령으로 실행한다. GPU·네트워크 호출은 없다.

```bash
.venv/bin/python balanced_bank_split.py
.venv/bin/python -m pytest -q test_balanced_bank_split.py
```

검증은 사용자 수량·라벨 보존·출처별 비율·부모 그룹 분리·입력 순서에 무관한 재현성을 확인한다. 모델 성능 개선을 입증하는 실험은 아직 하지 않았다.
