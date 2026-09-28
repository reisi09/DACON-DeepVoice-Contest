# 수집 bank 최종 통합 검토 — 2026-09-18

최종 snapshot: **2026-09-18T06:14:41.610344+00:00**. 7개 bank의 최종 source manifest, 수집 receipt, 고정 평가 제외 목록, 음성 품질 sidecar를 결합해 확인했다. 결과 데이터는 [results/data_expansion_20260918/cross_bank_audit.json](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/cross_bank_audit.json)이다. 이번 검토에서 다운로드·원본 수정·새 split·학습·GPU 실행·commit은 하지 않았다.

## 최종 수량과 라벨

보존 source record는 **15,169개**, 전체 원본 재생시간은 **894,132.2524375초 / 248.370070시간**이다. 신규 원본 수집 record는 **11,703개**, 재사용한 자료의 record는 **3,466개**다. 이 수량에는 격리·검토 대상과 평가용 자료가 포함되므로 학습 적격 수량과 구분해야 한다.

| Bank | 보존 record | 신규 수집 | 재사용 | 알려진 group 키 | 전체 원본 시간 | 로컬 기술 후보 |
|---|---:|---:|---:|---:|---:|---:|
| HAIM | 1,602 | 1,346 | 256 | 1602 | 26.579784시간 | 1,598 |
| SONICS | 2,932 | 2,932 | 0 | 2932 | 120.029659시간 | 2,932 |
| Jamendo | 854 | 534 | 320 | 669 | 57.058652시간 | 755 |
| FMA | 601 | 331 | 270 | 600 | 28.428675시간 | 393 |
| FakeMusicCaps | 2,560 | 2,560 | 0 | 512 | 7.171129시간 | 2,560 |
| LibriSpeech | 2,620 | 0 | 2,620 | 40 | 5.403467시간 | 2,126 |
| ASVspoof | 4,000 | 4,000 | 0 | 107 speakers → 4 연결 그룹 | 3.698705시간 | 4,000 |

로컬 기술 후보 합계는 **14,364개 / 228.907675시간**이다. HAIM은 `integrity_checked` 1,600개 중 clipping 검토 2개를 별도로 남겨 1,598개로 계산했고, 나머지는 각 최종 source manifest의 후보 조건을 적용했다. LibriSpeech 후보 중 신호 검토 대상 3개까지 보수적으로 분리하면 14,361개지만, 이번 수집 단계에서 이를 자동 탈락시키거나 실제 학습 집합으로 승인하지 않았다.

재사용 3,466개는 HAIM pilot 256 + Jamendo pilot 320 + FMA local 270 + 이미 받은 LibriSpeech archive의 2,620개 member다. LibriSpeech에서 이전에 등록된 원본 hash와 일치하는 것은 16개이며, 나머지 2,604개는 추가로 목록화했다. **추가 목록화와 새 오디오 다운로드는 다른 수량**이다. FMA local `4384`는 이전 artist ID는 알려져 있지만 현재 전체 원본 hash가 prior registry에 없어, hash 미일치 건수를 신규 수집량으로 해석하면 안 된다.

| Label | 1 | 0 | unknown |
|---|---:|---:|---:|
| FILE_FAKE | 9,094 | 6,075 | 0 |
| MUSIC_FAKE | 7,094 | 1,455 | 6,620 |
| MUSIC_PRESENT | 8,549 | 6,620 | 0 |
| VOICE_FAKE | 2,000 | 4,620 | 8,549 |
| VOICE_PRESENT | 6,620 | 0 | 8,549 |

라벨 분포는 보존 record 전체 기준이며 격리·평가용 row도 포함한다. music bank 8,549개의 voice labels는 모두 unknown으로 유지했다. speech MUSIC_FAKE 6,620개도 unknown이다. 15,169행의 label 값과 speech의 명시적 mask 33,100개를 확인했으며 known/unknown과 mask의 불일치는 0개였다. 이 source annotation을 청취 확인이나 frame label로 바꾸지 않았다.

## 원본·중복·평가 잠금 검증

- 직접 파일로 보존한 원본 **12,549개**를 단계별로 다시 읽어 SHA256·크기를 대조했고 불일치는 **0개**였다. ASV 4,000개는 CRC도 독립 재검사했다. 기존 4개 music bank의 실제 검증과 최종 CSV는 ID/path/SHA256/bytes/PCM hash projection이 모두 같음을 확인해 연결했다. LibriSpeech 2,620개 archive member는 전수 native PCM sidecar의 원본·PCM hash 일치 및 최종 manifest 결합을 확인했다.
- bank 간 원본 SHA256 교집합 **0개**, full-source canonical PCM 교집합 **0개**다. LibriSpeech↔ASV는 같은 16kHz mono native `<f4` hash 규격으로 따로 비교해 교집합 0개를 확인했다. 이 검사는 재인코딩·발췌·near duplicate 부재를 입증하지 않는다.
- FakeMusicCaps는 **512 prompt × 5 generator = 2,560개**, 각 generator 512개이며 incomplete group·quality flag·known prior/eval prompt 충돌이 모두 0개다. 실제 원본 **1,652,410,368 bytes**가 ledger와 일치했다. 전체 archive MD5는 검증하지 않았으며 선택 member CRC와 고정 central directory가 출처 근거다.
- 기존 평가 자료를 bank에 보존하면서 후보에서 제외한 항목이 있다. Jamendo 평가 artist 27명에 속한 66개, LibriSpeech 평가 narrator 8명에 속한 494개, FMA의 current/historical 평가 ID·hash 관련 row다. 모든 bank의 로컬 기술 후보 중 기록된 current/historical lock과 겹친 항목은 0개였다.
- FMA 내부의 `120632`와 `122155`는 원본 byte hash는 다르고 PCM hash는 같다. 둘 다 신규 수집이며 artist ID도 다르다. `122155`는 후보에서 제외했다. HAIM의 historical exact 중복 2개는 격리 상태로 남았다.
- Jamendo heldout artist 이름 **27/27개**를 복원해 `NFKD → casefold → isalnum` exact-name 비교를 했고, FMA와의 교집합은 0개였다. 전체 bank 간 이름 연결 후보는 `mikrobenkrieg` 1개다: FMA artist `16670`/track `97270` ↔ Jamendo artist `443873`/track `1165463`. 동일 인물로 확정하지 않았으며 보수적 분할 제약 후보를 [results/data_expansion_20260918/cross_catalogue_artist_candidates.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/cross_catalogue_artist_candidates.csv)에 기록했다.

원본 재검사에서 읽은 파일의 논리적 byte 합은 **14,244,577,013 bytes**다. 기존 local 자료도 포함하며 metadata·archive 저장 방식이 다르므로 전역 20GB 예산 합계로 사용할 수 없다. 최종 저장량은 root의 고유 inode 집계를 따른다.

## ASV 부모·화자 연결

최종 ASV는 bona fide 2,000개 + spoof 2,000개, 원본 **242,884,605 bytes / 3.698705시간**이다. 실제 VCTK target 및 VC input 화자를 연결하면 **107명 → 4개 연결 그룹(37·30·24·16명)**으로 줄어든다. 독립 재계산 결과가 [results/data_expansion_20260918/speech_parent_groups.json](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/speech_parent_groups.json)의 구성원까지 일치했다. 그룹별 row 수는 1,444 / 1,175 / 819 / 562개다. 이를 4개 독립 표본이라고 부르거나 107개의 서로 독립적인 화자로 분할해서도 안 된다.

VC 526개는 input reference 491개·input speaker 29명·화자 연결 322개를 가진다. 이 중 **110개 VC가 함께 선택된 bona fide 부모 100개를 참조**하며, 모두 부모와 target-only `group_id`가 다르다. **official train의 VC 40개가 official dev의 정확한 부모 35개를 참조**한다. input speaker 수준에서는 train VC 142개가 dev에 나타나는 화자와 연결된다. 24명·16명 연결 그룹은 official train/dev 양쪽에 걸쳐 있다.

예: [results/data_expansion_20260918/speech_bank_asv_selected.csv:3207](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/speech_bank_asv_selected.csv:3207)의 `LA_T_1151825`는 `p257_339`를 입력으로 쓰며, [results/data_expansion_20260918/speech_bank_asv_selected.csv:373](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/speech_bank_asv_selected.csv:373)의 실제 부모 `LA_D_6549233`는 official dev다. **현재 project split은 전부 unassigned다. 이는 완료된 학습의 누수 판정이 아니라 향후 분할 제약이다.** `official_partition`은 출처 정보로 남기고 새 project split으로 그대로 복사하지 않아야 한다.

현재 registry와 ASV의 기록된 hash·speaker·parent ID exact 교집합은 0개다. 다만 registry에는 ASV/VCTK 화자 namespace와 다른 corpus의 실명·별명 연결이 확립돼 있지 않다. 따라서 이 결과로 corpus 간 인물 독립성을 입증할 수 없다. DF-Arena가 ASVspoof2019를 학습 corpus로 명시한 점과 실제 녹음 단위 사전학습 중복 미확인도 보존한다.

## 품질 flags와 상태 해석

| Bank | 확인한 제외·검토 항목 |
|---|---|
| HAIM | historical exact 중복 2개 격리; 별도 `more_than_one_percent_clipped_samples` 2개 검토 |
| SONICS | 최종 retained 2,932개, rejected 0개; source ID 잠금과 기술 검증 통과 |
| Jamendo | `high_clip_fraction` 35개 검토; 평가 artist track 66개 제외. `duplicate_pcm_for_training=true` 66개는 평가 제외 row도 포함하므로 내부 중복 음원 66개라는 뜻이 아님 |
| FMA | 신규 duration selection review 3개; 0.0271875초 원본의 기술 탈락 1개; 동일 PCM로 제외한 신규 1개. duration review 3개를 오디오 파손이라고 부르지 않음 |
| FakeMusicCaps | 최종 quality flag 0개 |
| LibriSpeech | 전수 2,620개 중 near-full-scale review 3개, 나머지 3종 flag는 0개. 3개 모두 로컬 후보에 포함돼 있어 검토 대상으로 별도 표시 |
| ASV | 전수 4,000개에서 all-zero, RMS<1e-6, duration<0.25s, near-full-scale fraction≥0.001의 4종 flag 모두 0개 |

Speech sidecar의 row ID 집합·원본/PCM hash·duration·flag 집계를 최종 manifest와 다시 맞췄다. Libri 검토 ID는 `2094-142345-0000`, `2094-142345-0005`, `2094-142345-0025`다. near-full-scale 신호는 원본 특성일 수 있으므로 자동 탈락 규칙으로 바꾸지 않았다. 모든 bank에서 **독립 청취·자연스러움·발음 명료도·보컬 유무의 실제 품질은 미검증**이다.

SONICS의 reject 후 재시작 receipt 불일치는 수정됐다. [new_panel_data_sonics_bank.py:214](/Users/choiwuseck/Documents/ChatGPT/데이콘/new_panel_data_sonics_bank.py:214)에서 rejected 상태·discarded_path를 원자적으로 저장한 뒤 삭제하며, retained hash 검사는 유지한다. 실제 MP3 사본만 사용한 격리 회귀를 독립 실행해 재시작 및 변조 거부를 확인했다. [results/data_expansion_20260918/sonics_resume_fix_verification.json](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/sonics_resume_fix_verification.json)과 [results/data_expansion_20260918/sonics_operation_status.json](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/sonics_operation_status.json)이 근거다. 옛 running/35GB receipt는 최신 retention 결과로 대체됐음을 명시한다.

## 최종 결합 계약

| 계약 | 요구되는 동작 |
|---|---|
| 원본과 출처 | 직접 `path` 또는 `archive_path + archive_member`를 유지한다. source URL·revision·정확한 per-track license/attribution·bytes·hash·입력 ledger SHA256를 보존한다. member hash와 archive hash를 혼동하지 않는다. |
| 라벨과 flags | CSV bool을 명시적으로 파싱하며 `bool("False")`를 사용하지 않는다. flags의 JSON 배열/semicolon 문자열을 구분한다. unknown label은 mask=0으로 유지한다. |
| 최종 상태 | `retained`, `integrity_checked`, `verified`, `train_eligible_by_group`는 서로 다른 검사 범위다. rights·무결성·identity lock·중복·신호 검토를 분리해 결합하며, source annotation을 frame label이나 청취 확인으로 승격하지 않는다. |
| journal | Jamendo journal 873행은 현재 854개 ID와 이전 `budget_deferred` 19건을 포함한다. ID별 마지막 상태를 읽거나 최종 854행 CSV를 사용한다. 완료 receipt가 중간 progress보다 우선한다. |
| 그룹과 중복 | source parent·prompt·artist·화자와 VC input을 전이적으로 연결한다. 동일 음악의 다른 encoding, 다른 artist ID, prompt siblings를 단일 row hash만으로 분리하지 않는다. Exact-name 후보는 동일 인물 확정과 구분한다. |
| 보존과 예산 | 신규·재사용·평가용·격리 row를 구분한다. root가 이번 bank와 이전 pilot의 `(st_dev, st_ino)`별 고유 bytes, metadata/partial과 잔여 저장공간을 합산한다. 요구 상한은 20,000,000,000 bytes 미만, 여유 공간은 70 GiB 이상이다. |
| 작업 경계 | 모든 source split은 unassigned로 보존한다. 수집 검증 완료가 새 split·훈련·GPU 실험의 실행을 의미하지 않는다. |

## 최종 입력 식별

각 bank의 최종 파일을 snapshot으로 읽어 SHA256를 저장했다. JSON에는 row별 lock 근거, 중복 그룹, ASV 부모 edge, 신호 sidecar binding 및 source별 전체/후보/신규 duration·label 분포가 포함된다.

| Bank | 최종 입력 | SHA256 |
|---|---|---|
| HAIM | [haim_sources.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/haim_sources.csv) | `fae1e22ee5087205f0c201c77022433aea21492b7d40198d66a4e55557db4d32` |
| SONICS | [sonics_retained_sources.jsonl](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/sonics_retained_sources.jsonl) | `32369bb3b5e209710068bcb9d2192b43bc82692843abc058c4994f97a0677acf` |
| Jamendo | [jamendo_sources.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/jamendo_sources.csv) | `10e391337c06a3f35a30ae1c44f459b8f7ad6c02edd93375b8466f80efb5c2e4` |
| FMA | [fma_bank_final_manifest.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/fma_bank_final_manifest.csv) | `a5f953ecea74c26fce008ffe6a7506d7c6c21b775599e81c28c3d75e111307ff` |
| FakeMusicCaps | [fakemusiccaps_sources.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/fakemusiccaps_sources.csv) | `9b723e85309a902f9b6608d416e91dfa226794295382a61123513538cf7d27c0` |
| LibriSpeech | [speech_bank_test-clean.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/speech_bank_test-clean.csv) | `dc2fa4c3b302f2881f7b1c8177451bae9b9e1d571254b00031b779fe8f02fa4f` |
| ASVspoof | [speech_bank_asv_selected.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/speech_bank_asv_selected.csv) | `f2774050524a1b332e520ce59c173f4f47b561d790e4bbbca21fbf7837b16c36` |

고정 평가 registry SHA256: `2ca4cf947b2dddd72b4767337ac94601d10b380a9b1b76414fa835e46f640846`. Echoes↔MusicCaps crosswalk, HAIM의 모든 원 prompt, 아티스트·화자 별명, 사전학습 녹음 단위 중복은 여전히 미해결이다. 알려진 group 키 수와 exact hash 통과를 유효 독립 표본 수로 환산하지 않았다.
