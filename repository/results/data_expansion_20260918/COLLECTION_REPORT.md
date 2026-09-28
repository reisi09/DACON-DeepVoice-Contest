# 외부 데이터 수집 기록

2026-09-18. **수집과 최종 교차 검증을 마치고 멈췄다.** 신규 원본 11,703개를 확보했고, 기존 원본 3,466개를 재사용해 이번 목록은 총 15,169개, 248.37시간이다. 사용자의 마지막 지시에 따라 GPU 실험이나 학습으로 이어가지 않았다.

## 확보 현황

여기서 원본은 서로 다른 파일을 뜻한다. 같은 prompt의 생성기별 출력, 동일 화자의 발화, 같은 아티스트의 곡을 각각 독립 표본으로 세지 않는다. 기존 파일의 재사용과 이번에 내려받은 파일도 나누어 집계한다.

새 원본은 `/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/data_expansion_20260918/` 아래 자료별 폴더에 있다. 아래 원장의 `path` 또는 `archive_path + archive_member`로 각 음원과 출처를 찾을 수 있다.

| 자료 | 이번에 추가한 원본 | 기존 원본 재사용 | 검증·제외 결과 |
|---|---:|---:|---|
| [HAIM](haim_sources.csv) | 1,346 | 256 | 총 1,602개. ACE-Step 800개와 MusicGen 800개 무결성 검사 통과, 기존 자료와 중복인 2개 격리. 통과본 중 클리핑 경고 2개는 별도 검토 필요 |
| [SONICS](sonics_retained_sources.jsonl) | 2,932 | 0 | Suno 1,466개·Udio 1,466개. 전체 CRC·완전 디코드·원본 및 PCM 해시 검사 통과 |
| [Jamendo](jamendo_sources.csv) | 534 | 320 | 총 854곡·669아티스트. 클리핑 검토 35곡, 평가용 27아티스트·66곡 보존 및 학습 후보 제외. 기술 조건을 만족하는 후보 755곡 |
| [FMA](fma_bank_final_manifest.csv) | 331 | 270 | 신규 331곡 중 후보 326곡. 길이 확인 3곡·극단적으로 짧은 원본 1곡·PCM 중복 1곡 제외. 기존 원본 중 후보 67곡 |
| [FakeMusicCaps](fakemusiccaps_sources.csv) | 2,560 | 0 | MusicCaps prompt 512개 × 생성기 5종. 전체 CRC·원본 SHA·native 및 canonical PCM 검사 통과. 자동 품질 경고와 정확한 내부 중복 없음 |
| [LibriSpeech test-clean](speech_bank_test-clean.csv) | 0 | 2,620 | 기존 압축파일을 재사용해 전체 디코드 검증. 평가용 8화자의 494발화 제외, 나머지 2,126발화는 미분할 후보. 무음·극단적으로 짧은 파일은 없으며 near-full-scale 비율 검토 3개는 별도 기록 |
| [ASVspoof 2019 LA](speech_bank_asv_selected.csv) | 4,000 | 0 | 진짜 2,000개·가짜 2,000개. 전체 CRC·원본 SHA·native 및 canonical PCM 검사 통과, 내부 PCM 중복 없음. 실제 화자·변환 입력 화자와 공격 유형을 함께 보존 |

표의 후보 수는 자료별 기술 검사 결과다. 향후 학습에서는 자료 간 연결 관계와 미확인 라벨까지 반영해 다시 선정해야 한다. 이번 수집에서는 새 학습·평가 분할을 만들지 않았다.

## 용량 기준

상한은 보수적으로 **20,000,000,000 bytes 미만**으로 해석했다. 새 수집 폴더와 기존 pilot 수집 폴더를 함께 측정하고, 하드링크는 실제 같은 파일을 한 번만 셌다. 메타데이터를 포함했으며, 수집 폴더에 미완성 음원·전송 파일은 남아 있지 않다. 파일 크기 합계와 실제 디스크 할당량은 다음과 같다.

| 측정 범위 | 파일 크기 기준 | 실제 디스크 할당량 |
|---|---:|---:|
| 이번 수집 폴더 + 기존 pilot 수집 폴더 | 14.86 GB | **15.64 GB** |
| 위 자료 + 기존 실험용 사본·업로드 ZIP·결과 기록 | 18.58 GB | **19.38 GB** |

GB는 10억 bytes 기준이다. [용량 측정 기록](storage_snapshot.json)에 측정 시각, 폴더별 bytes, 재현 명령을 남겼다. 두 범위 모두 20GB 미만이다.

기존 실험용 데이터 사본과 업로드 ZIP도 별도 합산해 확인한다. 예전 실험·제출물 전체를 20GB 안으로 줄이라는 뜻으로 해석해 삭제하지는 않았다. FMA 기존 원본 270개는 기존 위치를 참조한다.

이번 수집에서 생긴 미완성 전송 파일과 불필요한 전체 압축파일만 정리했다. SONICS는 이미 검증한 압축파일에서 선택 원본을 복원하고 원본 해시를 확인한 뒤 압축파일을 지웠다. 삭제량은 품질 불량 음원의 양을 뜻하지 않는다. 세부 내역은 자료별 cleanup·retention 영수증에 남겼다.

## 출처와 사용 조건

[DACON 대회 규칙](https://dacon.io/competitions/official/236749/overview/rules)을 확인하고 공개 자료의 데이터 라이선스와 원본 출처를 보존했다. 코드 저장소의 라이선스를 음원 라이선스로 대신 적용하지 않았다.

| 자료 | 확인한 출처·조건 |
|---|---|
| HAIM | 공개 HF 원본, revision `371be01815b65274762d6d0a33e2cb70e24b9baa`, CC BY-NC 4.0. ACE-Step·MusicGen만 이번 범위에 포함 |
| SONICS | [공식 HF 자료](https://huggingface.co/datasets/awsaf49/sonics), revision `3788dca9f9f11ad92e9097ef4b58eee247661e7f`, CC BY-NC 4.0 |
| FakeMusicCaps | [공식 Zenodo 원본](https://zenodo.org/records/15063698), CC BY-NC 4.0. 개별 ZIP member의 크기·CRC·해시를 검증하며 전체 12.9GB archive MD5를 확인했다고 주장하지 않음 |
| Jamendo | 곡별 원본 URL·CC 라이선스·아티스트·발매일 보존. ND 및 다운로드 불가 항목 제외. 메타데이터 revision `13090d785d071dcd617636deaa4a9775936ac995` |
| FMA | 원본 metadata의 곡별 URL과 BY/BY-SA/BY-NC/BY-NC-SA 조건 보존 |
| LibriSpeech | [OpenSLR 원본](https://www.openslr.org/12/), CC BY 4.0. 기존 test-clean archive의 공식 MD5 재확인 |
| ASVspoof | [공식 Zenodo 원본](https://zenodo.org/records/6906306), ODC-By 및 공식 README·CM protocol·실제 화자 연결표 보존. 선택한 ZIP member의 CRC를 확인했으며 전체 archive MD5는 검증하지 않음 |

출처 표시와 NC·SA 등 각 자료의 조건은 후속 사용에도 적용된다. 합성 자료가 참조한 YouTube 원음을 별도로 내려받지는 않았다.

## 검증으로 확인한 것과 남은 한계

원본 해시, 전체 디코드, 알려진 평가 그룹, 정확한 중복을 확인했다. 자료별 검증 결과와 교차 검토의 입력 해시는 JSON·CSV에 남겼다. 음악 자료 8,549개와 ASVspoof 4,000개, 총 12,549개는 교차 검토자가 실제 파일을 다시 읽어 SHA256과 크기를 대조했고 불일치가 없었다. LibriSpeech 2,620개도 압축파일의 원본과 검증 원장을 대조했다. 자료 간 원본·canonical PCM의 정확한 중복은 없었으며, HAIM의 기존 자료 중복 2개와 FMA의 내부 PCM 중복 1개는 후보에서 제외했다.

FMA와 Jamendo에는 아티스트 이름을 정규화했을 때 `mikrobenkrieg`로 일치하는 연결 후보가 하나 있다. 동일 인물이라고 확정한 것은 아니지만, 향후 분할에서는 보수적으로 함께 다뤄야 한다. 현재 확인된 Jamendo 평가 아티스트와 FMA 이름의 교집합은 없다.

ASVspoof의 선정 4,000개는 target 화자와 VC 입력 화자를 모두 연결하면 107개 화자 ID가 4개 연결집합으로 묶인다. 집합별 화자는 37·30·24·16명이다. 이 수치는 실제 누수의 발견이 아니라 출처 관계표를 적용한 결과다. target 화자만 나눈 뒤 VC 입력까지 독립이라고 주장해서는 안 된다. 세부 관계는 `speech_parent_groups.json`에 기록했다.

변환 음성 110개는 이번에 고른 진짜 녹음 100개를 입력 원본으로 참조한다. 이들 110개는 모두 target 화자 기준 `group_id`가 원본과 다르다. 특히 40개는 공식 `train` 변환 음성이 공식 `dev` 원본을 참조한다. 기존 `official_partition`을 새 학습·평가 분할로 그대로 복사하지 않고, 원본 녹음과 변환 입력 화자 관계를 함께 적용해야 한다. 이번 ASVspoof 자료의 새 분할은 모두 미지정 상태이므로 이미 학습 누수가 발생했다는 뜻은 아니다.

음악 자료의 보컬 유무와 보컬 합성 여부는 확인되지 않은 값을 그대로 비워 두었다. SONICS의 `half fake`를 음원 절반이 합성됐다는 시간 라벨로 바꾸지 않았다. LibriSpeech·ASVspoof의 음악 합성 라벨도 미확인 상태를 유지한다.

이 검증은 사람의 청취 평가를 대신하지 않는다. 음질·자연스러움·장르·보컬·프레임별 라벨을 전수 청취로 확인하지 않았고, 재인코딩된 유사 음원이나 알려지지 않은 아티스트 별칭도 완전히 배제하지 못했다. 자료별 자동 검사 기준도 달라 경고 개수만으로 품질 순위를 매길 수 없다. HAIM의 모든 원 prompt 연결과 Echoes↔MusicCaps 연결은 미해결이다. ASVspoof는 기존 DF-Arena 사전학습과 겹칠 수 있으므로 독립 평가 자료라고 부르지 않는다.

FakeMusicCaps의 512개 prompt는 중단된 파일에서 복원 가능한 AudioLDM2 후보 중 기술 조건을 통과한 자료에서 골랐다. 선택 과정에서 모델 예측을 사용하지 않았지만, 전체 자료를 무조건 무작위 추출한 표본은 아니다.

## 실험 상태

GPU와 학습은 시작하지 않았다. 기존 2,470행 pilot manifest와 업로드 ZIP은 보존했으며 이번 확장 자료로 덮어쓰지 않았다. 새 자료는 로컬에 있으며 A100 실험용 업로드·학습 manifest에는 아직 반영하지 않았다. 이 수집 결과만으로 모델 성능이나 DACON 점수가 개선됐다고 주장할 수 없다.

자료별 최종 manifest·영수증은 이 폴더에 있으며, [교차 검토](INTEGRATION_REVIEW.md), [평가 그룹 잠금](LOCKED_GROUPS.md), [음성 자료 설명](VOICE_CANDIDATES.md)을 함께 보존한다. [최종 수집 기록](collection_summary.json)에 용량과 수집 프로세스 종료를 기록했고, [GPU 상태](gpu_not_started.json)에서 기존 A100·H100 워크스페이스가 모두 stopped/standby임을 확인했다. 자동 재개 작업도 만들지 않았다.
