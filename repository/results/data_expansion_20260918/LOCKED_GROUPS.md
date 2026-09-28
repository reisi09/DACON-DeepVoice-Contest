# 데이터 확대 전 고정 평가 그룹 목록

2026-09-18 snapshot. [locked_groups.json](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/locked_groups.json)에 현재 고정 평가 제외집합, 이전 사용 이력, source 간 identity 연결 규칙을 저장했다. 모델 score는 읽거나 사용하지 않았다. 원본 manifest를 바꾸지 않았고 오디오 다운로드·GPU 학습도 실행하지 않았다. 작업 시작 시 로컬 여유 공간은 208 GiB였으며 이 작업은 metadata JSON·Markdown만 추가했다.

## 현재 고정 평가 제외집합

현재 split의 기준은 `/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/localization_a100_20260918/dataset/manifest.csv`의 2,470행이다. train 1,469행을 제외한 calibration·dev·test·confirmation·confirmation_external·confirmation_presence는 모두 신규 학습 유입에서 보호한다. 기존 test가 반복해서 열렸다는 문제와 별개로, 이번 비교에서 정한 split은 유지한다.

| 집합 | 현재 고정 평가에 묶인 ID 수 | JSON 필드 |
|---|---:|---|
| FMA artist | 111 | `current_locked_eval.fma_artist_ids` |
| FMA track | 111 | `current_locked_eval.fma_track_ids` |
| SONICS song | 62 | `current_locked_eval.sonics_ids` |
| Jamendo artist | 27 | `current_locked_eval.jamendo_artist_ids` |
| Jamendo track | 66 | `current_locked_eval.jamendo_track_ids` |
| MusicCaps prompt / YouTube ID | 64 | `current_locked_eval.musiccaps_ids` |
| Echoes reference title | 35 | `current_locked_eval.echoes_reference_titles` |
| Echoes 생성 archive member | 382 | `current_locked_eval.echoes_archive_members` |

위 artist·track·song·prompt·Echoes reference와 현재 train의 교집합은 모두 0이었다. 파일 수와 독립 그룹 수는 다르다. JSON에는 native group atom·parent ID·기록된 byte/PCM hash도 따로 저장했다. `speaker_ids`는 원본 표기와 namespace가 섞인 identifier 목록이므로 그 길이를 독립 화자 수로 해석하지 않는다.

FMA 고정 평가 artist는 calibration 31, dev 25, test 23, confirmation 32개다. SONICS는 각각 11, 10, 9, 32개다. `confirmation_external`에는 Jamendo 27 artist의 66곡과 SunoCaps 64개 prompt가 있다.

**Jamendo의 최종 split이 우선이다.** 개별 수집 단계의 `data_preparation_jamendo_sources.csv`와 `data_preparation_jamendo_manifest.csv`는 320개를 전부 train이라고 적고 있다. 이후 `data_preparation_additional_training_manifest.csv`와 현재 2,470행 manifest가 27 artist의 66곡을 confirmation_external로 옮겼다. 새 Jamendo 수집에서 옛 파일의 train 표기를 그대로 따르면 이 27 artist가 학습에 다시 들어올 수 있다.

## 이전에 사용한 그룹과 신규 수집의 중복 제외

`all_prior_seen`은 현재 train도 포함한 수집 중복 확인용 집합이다. FMA 270 artist·270 track, SONICS 110 song, Jamendo 135 artist·320 track, MusicCaps 64 prompt, Echoes reference title 102개를 확인했다. 이미 승인한 train 자료를 삭제하라는 목록이 아니다. 새로 확보했다고 세는 자료가 기존 원본이나 같은 artist·prompt의 재수집인지 판별하는 데 쓴다.

`historical_evaluation_seen`은 현재 split 이전의 dev·test·sealed·challenge 등에 기록된 identity를 따로 보존한다. 재분할이나 full refit 뒤에도 과거 평가·선택에서의 노출 이력이 사라지지 않기 때문이다. 현재 locked eval과 이 목록을 합쳐 단일한 독립 검증 집단으로 부르지 않는다.

V1 validation, V2 development·locked test, recovery, v4 train·dev·sealed, six, voice/music, eight와 현재 localization의 manifest 및 source registry를 직접 읽었다. 과거 179개 metadata 입력을 기록한 `data_preparation_history.json`과도 교차 확인했다. 그 파일의 실제 숫자형 FMA artist 238개와 SONICS 제외 ID는 새 `all_prior_seen`에 모두 포함됐다. 옛 목록의 추가 artist `shared-artist`는 `.omo` fixture에서 온 비숫자 값이라 실제 artist 수에 넣지 않았다. fixture의 zero hash도 실제 음원 중복 근거로 승격하지 않았다.

## source 간 공통 연결 규칙

FMA는 `artist_id`를 `artist:fma:<id>`, `track_id`를 `reference:fma:<id>`로 연결한다. 옛 `fma_artist_<id>`는 같은 artist namespace의 별칭이다. Jamendo는 `artist:jamendo:<id>`·`reference:jamendo:<id>`를 사용한다. 두 코퍼스에서 숫자가 같다고 같은 artist로 묶지 않는다. artist 이름이 같다는 사실만으로 alias를 확정하지도 않는다.

SONICS는 원본 metadata의 `id`를 `reference:sonics:<id>`로 연결한다. `sonics_id_<id>`는 같은 key의 옛 표기다. 같은 ID의 generator version·파일 suffix·crop이 달라져도 부모 lock을 그대로 물려준다.

Echoes는 `path_in_dataset`으로 생성 member를 찾고 `original_audio`를 `echoes_fma_unique.json.echoes_original_audio`에 정확히 연결해 FMA track·artist로 돌아간다. 프로젝트 전체에서 확인한 102개 reference 연결은 `echoes_fma_reference_links`에 저장했다. 최신 기존 음악 중 Echoes 1,026개는 FMA track 90개에 연결되며 누락은 0개다.

파생 파일이 둘 이상의 부모를 가지면 각 부모의 모든 artist·track·prompt atom을 유지해야 한다. 하나라도 고정 평가 집합에 닿으면 그 파생 파일을 신규 train으로 넣지 않는다. caption·generator 이름·source 이름은 확인되지 않은 부모 관계를 대신할 수 없다.

## SunoCaps와 FakeMusicCaps의 prompt 연결

SunoCaps 고정 평가 64개의 `reference_id`는 `musiccaps:<11-character YouTube ID>`다. 원본 `SunoCaps_dataset.csv`는 같은 64개 prompt에 대해 version 1–4를 가진 256행이다. `ytid_V.rsplit("_", 1)[0]`으로 ID를 추출해야 한다. YouTube ID 자체에 `_`가 포함되므로 `split("_")`로 나누면 잘못 연결된다. `-YIT4HBM__g`, `0ZXWSO_Y2C0`, `1_YHHL_t2GI`가 실제 예다.

64개 prompt 안에서는 caption이 version 간 동일했고 서로 다른 caption 문자열도 64개였다. `musiccaps_prompt_locks`에 전체 variant ID, caption 원문·SHA-256, aspect list SHA-256과 metadata 위치를 기록했다. 선택한 SunoCaps 64개 파일만 제외하는 것으로는 부족하다. **같은 MusicCaps ID로 만든 모든 generator의 출력을 평가 prompt의 파생물로 취급한다.**

수집 중인 FakeMusicCaps ZIP의 앞부분에서 `audioldm2/dHGAXJ9RPJs.wav`, `audioldm2/kjcXM60HrRA.wav`, `audioldm2/MpWGx5odhh8.wav` 같은 member를 확인했다. README도 MusicCaps caption을 바탕으로 생성한 데이터라고 설명한다. 따라서 `<generator>/<11-character YouTube ID>.wav`의 basename을 공통 key로 연결할 수 있다는 근거가 있다. 다만 확인 당시 archive는 `.part` 상태였고 전체 27,605개는 배포자가 제시한 수치다. 완성된 member 수, 실제 64개 lock과의 교집합 수, caption 연결은 아직 검증하지 않았다.

완성본에서는 `__MACOSX/**`와 `._*`를 제외하고 basename 전체가 `[A-Za-z0-9_-]{11}`인지 검사한 뒤 MusicCaps metadata와 대조해야 한다. 이 ID가 `current_locked_eval.musiccaps_ids`에 있으면 generator와 무관하게 train 유입을 제외한다. ID를 파싱하지 못한 파일은 별도 미해결 집합에 남기고, 독립성이 확인됐다고 간주하지 않는다. 이미 받은 전체 archive 안에 제외 대상이 있어도 acquisition 보관과 train admission은 따로 판단할 수 있다.

## Echoes와 MusicCaps 사이에 확인되지 않은 연결

Echoes 전체 archive metadata는 4,468개 생성 파일과 `original_audio` title 296개를 가진다. fields는 `path_in_dataset`, `original_audio`, `generator`, `type`, `genre`, `description`, `duration`이며 MusicCaps ID나 YouTube ID가 없다. 알려진 reference는 FMA 곡이다. 현재 가지고 있는 자료만으로 Echoes title을 MusicCaps ID로 바꾸는 exact crosswalk는 만들 수 없다.

SunoCaps의 64개 caption과 Echoes의 description을 casefold·공백 정리 후 정확 비교한 교집합은 0개였다. 이것은 좁은 문자열 비교 결과다. 같은 음원이 다른 caption을 가질 수 있고 다른 upload나 부분 crop도 가능하므로, corpus 간 오디오 overlap이 없다는 증거로 사용할 수 없다. HAIM 일부의 human reference binding도 미해결 상태다.

현재 가능한 고신뢰 차단은 SunoCaps64 ↔ FakeMusicCaps의 MusicCaps ID 연결과 Echoes ↔ FMA의 track·artist 연결이다. Echoes ↔ MusicCaps 및 서로 다른 source의 artist alias·시간 이동·재인코딩 중복은 unknown으로 남긴다. 이후 metadata crosswalk나 실제 오디오 대조로 확인한 관계만 공통 provenance atom에 추가한다.

## 파일 사용법

신규 학습 admission은 `current_locked_eval`, 신규 수집 중복·과거 노출 확인은 `all_prior_seen`, 과거 평가 이력은 `historical_evaluation_seen`을 읽는다. `identity_join_contracts`에 key 변환 규칙이 있고, `input_files`에는 이번에 읽은 metadata의 SHA-256이 있다. 현재 manifest가 바뀌면 `current_manifest_sha256`과 대조해 재고정 여부를 판단한다. generator를 늘린 수와 독립 reference를 늘린 수는 따로 집계해야 한다.

<!-- HUMANIZE-SUMMARY: Korean final editing pass applied conservatively. Counts, identifiers, namespace rules, artifact paths, confirmed observations, and unknown cross-corpus mappings preserved. -->
