**데이터 확보 내역 — 2026-09-18, 1차 고정본**

추가 음악 원본 640개, 고정 음악 평가용 원본 64개, 음성·악기 존재 여부 평가용 원본 28개를 확보했다. 합계 **732개는 이번에 정한 1차 subset이며, 확보 가능한 최대 수량이 아니다.** 아래 수치는 이 1차 고정본의 기록이다. 후속 수집분은 별도 데이터 뱅크로 관리하며, 기존 평가 분할은 보존한다.

기존 자료와 파생 오디오를 합친 manifest는 **2,470행**, 그중 학습용은 **1,469행**이다. 한 원본에서 여러 길이의 오디오와 편집 대조군을 만들었으므로, manifest 행수를 독립 원본곡 수로 해석하면 안 된다. [최종 manifest](/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/localization_a100_20260918/dataset/manifest.csv), [통합 감사 결과](/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/localization_a100_20260918/dataset/dataset_audit.json)

| 출처 | 실제 확보 원본 | 원본 그룹·분할 | 확인한 이용 조건 |
|---|---:|---|---|
| [JamendoMaxCaps](https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps/tree/13090d785d071dcd617636deaa4a9775936ac995)의 Jamendo 음원 | 320 | 135명 artist ID, 아티스트당 최대 3곡. 학습 254곡 / 외부 평가 66곡 | 각 곡의 CC BY 또는 CC BY-SA URL과 `audiodownload_allowed=true` 확인. 버전·국가별 조건은 source CSV에 보존 |
| [HAIM](https://huggingface.co/datasets/mippia/HAIM/tree/371be01815b65274762d6d0a33e2cb70e24b9baa) | 256 | ACE-Step 1.5 128곡 + MusicGen 128곡, 전부 학습 | 데이터셋 저자 표시 CC BY-NC 4.0 |
| [SunoCaps](https://www.kaggle.com/datasets/miguelcivit/sunocaps), version 7 | 64 | 서로 다른 MusicCaps prompt 그룹 64개에서 각 1곡, 전부 외부 평가 | 데이터셋 저자 표시 CC BY-SA 4.0 |
| [FMA](https://github.com/mdeff/fma) | 32 | 기존 사용 목록에서 제외한 32개 track·artist ID, 전부 고정 평가 | 곡별 CC BY 또는 CC BY-SA |
| [SONICS](https://huggingface.co/datasets/awsaf49/sonics/tree/3788dca9f9f11ad92e9097ef4b58eee247661e7f) | 32 | 기존 사용 목록에서 제외한 Suno 16개·Udio 16개 ID, 전부 고정 평가 | 데이터셋 저자 표시 CC BY-NC 4.0 |
| [GuitarSet](https://zenodo.org/records/3371780) | 12 | 연주자 6명 × 2개 기타 연주, 전부 presence 평가 | CC BY 4.0 |
| [LibriSpeech test-clean](https://www.openslr.org/12/) | 16 | 낭독자 8명 × 2개 발화, 전부 presence 평가 | CC BY 4.0 |

추가 음악 640개의 출처 비중은 Jamendo 50%, HAIM 40%, SunoCaps 10%이며, 진짜·합성 음악은 각각 320개다. Jamendo는 HF에 고정한 메타데이터와 공식 Jamendo 다운로드 주소를 연결했고, 원래의 track ID·artist ID·라이선스 URL을 남겼다. HAIM의 MusicGen 구성은 small 55개, medium 52개, large 21개다. 출처별 조건과 저작자 표시는 유지해야 하며, 데이터셋의 라이선스 표시를 모든 원자료의 권리 독립성을 확인한 결과로 확대 해석하지 않는다. [추가 원본 목록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/local_sampler_20260918/data_preparation_additional_training_sources.csv), [확보·병합 영수증](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/local_sampler_20260918/data_preparation_additional_training.json)

| 최종 split | 오디오 행수 | 구성 |
|---|---:|---|
| `train` | 1,469 | 기존 base 831 + 기존 학습 원본으로 만든 partial/sham 128 + 추가 원본의 view 510 |
| `dev` | 164 | 기존 분할 유지 |
| `calibration` | 200 | 기존 분할 유지 |
| `test` | 127 | 기존 분할 유지 |
| `confirmation` | 352 | 고정 원본 64개 × 4/8/16/30/60초 view 320개 + 편집 대조군 32개 |
| `confirmation_external` | 130 | Jamendo 66개 + SunoCaps 64개 |
| `confirmation_presence` | 28 | GuitarSet 12개 + LibriSpeech 16개 |
| **합계** | **2,470** | 원본과 파생 view를 구분해 집계 |

추가 학습 510행은 전체 학습의 34.7%이며, 진짜 254개·합성 256개다. 기존 자료까지 포함한 전체 학습 1,469행은 진짜 457개·합성 1,012개이므로, 전체 학습셋이 균형이라는 뜻은 아니다. 외부 평가는 진짜 66개·합성 64개다.

새 분할은 모델 점수를 보기 전에 seed `20260918`로 고정했다. Jamendo 아티스트 135명 중 27명과 그들의 66곡을 통째로 외부 평가에 배정했다. SunoCaps는 64개 prompt 그룹 전부를 외부 평가에 두었다. HAIM은 MusicGen의 조건부 생성 reference 연결이 없어 전부 학습에 두었으며, 임의의 reference 독립 평가 분할을 만들지 않았다. [고정한 분할 규칙·ID 목록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/local_sampler_20260918/data_preparation_additional_partition.json)

추가 원본 640개는 자연 길이의 중앙 8초 view를 만들었다. 단, SunoCaps 원본 한 개는 실제 길이가 5초여서 그대로 사용했다. 고정 음악 64개는 모두 60초 이상인 원본을 확보했다. 편집 대조군 32개는 동일한 위치·길이·게인 처리의 real-real sham과 real-fake 쌍 16개이며, 기존 학습 구성과 같은 10ms crossfade를 적용했다. Presence 28개는 발화·연주 전체 길이를 유지했다. 어느 패널에도 길이를 늘리기 위한 반복이나 padding을 넣지 않았다. [고정 음악 구성 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/local_sampler_20260918/data_preparation_confirmation.json)

추가 640개와 고정 음악 352개 view는 원본과 기록한 구간·편집 규칙으로 재생성한 결과를 대조했다. 통합 감사에서는 모든 오디오 해시와 원본 연결을 확인했고, split 간 기록된 source atom·원본 해시·오디오 해시 중복은 0건이었다. 추가 학습 원본과 기존 원본 사이의 ID·해시 중복도 검출되지 않았다. 이 검사는 재인코딩된 근접 중복이나 알려지지 않은 원본 참조까지 배제하지는 못한다.

음악 704개 원본의 voice 관련 미확인 라벨은 비워 두었다. GuitarSet은 출처가 설명한 기타 녹음 특성, LibriSpeech는 출처가 설명한 실제 낭독에 근거해 presence 라벨을 부여했다. 독립 청취 검수는 하지 않았다. 편집 구간 라벨은 제작 기록에 따른 경계이며, 실제 유통되는 모든 편집 음악을 대표하지 않는다. [GuitarSet 근거](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/local_sampler_20260918/data_preparation_guitarset.json), [LibriSpeech 근거](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/local_sampler_20260918/data_preparation_speech.json)

남은 한계는 다음과 같다.

- FMA와 Jamendo의 ID 체계가 달라 정규화한 아티스트명 일치도 제외했지만, 별명·협업 명의까지 동일 인물 여부를 해결하지는 못했다.
- SunoCaps는 MusicCaps prompt ID를 제공하지만 파일별 Suno 플랫폼 UUID는 제공하지 않는다. HAIM의 조건부 원본 연결도 일부 미확인이다.
- 이번 확보는 새 데이터셋·곡·prompt와 기록된 버전의 확장이다. Suno·Udio·ACE-Step·MusicGen을 처음 보는 생성기 계열이라고 주장하지 않는다.
- 진짜·합성의 출처 차이는 여전히 남아 있다. 외부 평가 결과만으로 생성 방식과 데이터셋 특성의 영향을 완전히 분리할 수 없다.

원본과 다운로드 메타데이터는 [_workspace/local_sampler_20260918/new_data/](/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/local_sampler_20260918/new_data/)에, 모델 입력과 통합 provenance는 [_workspace/localization_a100_20260918/dataset/](/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/localization_a100_20260918/dataset/)에 보존했다. 1차 최종 manifest SHA-256은 `daf9317856e63fba6cdba04f8fa28c48afa447fffe9046b246571ddecdbebff4`이며, [입력 패키지 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/localization_a100_20260918/input_build.json)의 해시와 일치한다. 확보·무결성 확인은 모델 성능이나 공식 점수를 입증하지 않는다.
