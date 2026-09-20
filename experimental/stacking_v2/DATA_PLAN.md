# 데이터 계획 및 미확보 계약

## 목표와 실제 확보를 분리

| 상호배타적인 최상위 범주 | dev 목표 | locked OOD 목표 | 합계 목표 | 이번 신규 확보 |
|---|---:|---:|---:|---:|
| Voice-only | 640 | 160 | 800 | 0 |
| Music-only | 640 | 160 | 800 | 0 |
| Mixed | 960 | 240 | 1,200 | 0 |
| Stress | 320 | 80 | 400 | 0 |
| 합계 | 2,560 | 640 | 3,200 | 0 |

이는 수집 시작점이지 채워야 할 quota가 아니다. 독립 parent·화자·곡·source·generator family가 부족하면 파일 복제/증강으로 수를 맞추지 않는다. 이번 입력은 실제 오디오가 아니라 10개 캐시·6개 소형 head·기존 metadata/예측이다. 신규 독립 original, 화자, 곡, 생성기 확보 수는 모두 0이다. 합성 wiring의 72행은 이 표에 포함하지 않는다.

## 입력 metadata 실측

현재 runtime_manifest: 9,504행, train 6,652 / val 1,426 / test 1,426. 선택 목록의 과거 개발 노출 True는 6,174행이다. catalog 전체 59,770행에는 과거 개발 노출 True 13,807행이 있어, 선택된 9,504개만 검사하면 놓칠 수 있다.

runtime_manifest에서 기록된 고유 speaker_id는 109개(3,524행만 값 있음), track_id는 1,832개(1,834행만 값 있음), parent_id는 3,361개(6,143행 누락), component_id는 2,613개다. source는 28종, 원문 generator 문자열은 40종이며 **canonical generator family 40개라는 뜻이 아니다**. audio hash는 runtime_manifest에 9,504개 있지만 selected_list에는 407개만 있다. 반드시 현재 runtime_manifest와 원본 provenance를 함께 본다. 고유 hash/ID 수가 독립 창작물/인물 수를 증명하지 않는다. 상세 필드별 결측/고유값 수는 `artifacts/audit/input_audit.json`에 있다.

## 원본 먼저 split

먼저 original ledger를 만들고 canonical parent_id, 파일 SHA-256, 가능하면 디코딩 PCM SHA-256, speaker, track/artist, source/revision/license URL, 원본 생성기 family와 버전을 적는다. codec이 달라 byte hash가 달라도 parent/PCM/화자/곡 관계로 연결한다. 문자열 alias만 달라 다른 family로 간주하지 않도록 alias 검토 기록을 남긴다.

source와 fake generator family의 OOD 예약 목록은 점수를 보기 전에 정한다. `split-parents`는 원본 연결 그룹을 분리하지 않고 해당 예약 목록을 적용한다. 서로 모순되는 source/family 예약은 실패한다. 비율은 결과로 보고하며 80:20을 강제하지 않는다. 그 뒤 혼합·partial fake·gain·codec·noise 변형을 만든다. `derivative_split`에 부모 ID 전부를 넘겨 같은 split인 경우에만 생성한다.

Mixed는 음성 parent와 음악 parent를 각각 저장한다. Voice-only/Music-only도 crop/partial/codec 원본과 생성된 fake의 계보를 보존한다. 동일 화자·동일 곡·부모가 연결되면 간접 연결까지 한 그룹으로 묶는다. Stress는 독립적인 최상위 category 값으로 한 번만 센다. 예를 들어 낮은 SNR의 mixed stress는 category=Stress, composition=FR로 기록하며 Mixed 개수에도 더하지 않는다. 같은 원본의 원형/변형은 서로 다른 파일이지만 독립 parent 수는 늘어나지 않는다.

## 라벨 계약

RR/FR/RF/FF 순서는 **음성/음악**이며 R=실제, F=합성이다. Mixed에서는 둘 다 존재한다. RR→File 0, 나머지→File 1이다. 두 부모의 사실 라벨을 확인하지 못하면 composition=unknown이며 미확인 성분 라벨/mask를 unknown/0으로 남긴다. 한 성분이 확실히 fake면 File=1을 알 수 있지만 다른 unknown을 real=0으로 바꾸지 않는다.

Voice-only는 VOICE_PRESENT=1, MUSIC_PRESENT=0이며 없는 성분 Fake는 보통 unknown/mask=0으로 기록한다. Music-only는 반대다. 부분 fake의 구간/비율은 생성 ledger에 저장하되 모델 피처가 아니다. 무음/극소음량/악기 solo/노래/작은 stem leakage처럼 의미 라벨이 애매한 stress는 전문가 라벨링 또는 unknown 처리가 필요하다. source나 route만으로 File 라벨을 추정하지 않는다. 다섯 라벨과 다섯 mask를 항상 명시한다.

## 노출 감사

현재 base-head 최적화, 현재 head 선택, 과거 개발 노출, upstream encoder 사전학습 노출을 다른 필드로 보존한다. production meta-dev/OOD는 앞 세 항목이 증거를 갖춘 no일 때만 입장한다. parent 전체를 검토했다는 `reviewed_parent_ids`와 근거 문서 식별자가 필요하다. upstream은 자료가 없으면 unknown으로 남긴다. yes/unknown upstream은 downstream 독립성과 구분하여 표에 보고하고, encoder까지 완전히 미노출이라고 주장하지 않는다.

실제 코드의 registry는 selected/runtime 데이터와 catalog의 selected/historically-exposed 항목을 합친다. ID, parent, file/PCM/source hash, speaker, track, component를 교차 확인한다. 충돌은 거부한다. 충돌이 없다는 사실만으로 신규성을 입증하지 않으므로, 공급자 원문/provenance 및 원본 scope를 사람이 확인해야 한다. 잘못된 source/generator alias, 오디오를 실제 들여오기 전의 hash, 공개되지 않은 사전학습 목록은 자동으로 해결되지 않는다.

현재 val/test 예측은 어느 이름으로 옮겨도 신규 meta-dev/OOD가 아니다. reference_predictions, 이미 선택된 checkpoint의 점수는 재생 테스트 전용이다. 남은 catalog reserve도 자동 신규가 아니다. 기존 base 학습 데이터 재사용 시 필요한 nested/cross-fitting은 DESIGN.md §6이며 기본 학습 명령에서 이러한 재사용을 허용하지 않는다.

## 후보와 라이선스

공식 원문 URL, 2026-09-21 웹 확인 상태, 재배포 미해결 항목을 `DATA_CANDIDATES.csv`에 기록했다. 확인한 페이지의 라이선스 표기와 **실제 오디오 파일의 권리/대회 자료 제출 가능성**은 다르다. 이번에 audio tar/ZIP을 다운로드하거나 수집 완료로 표시한 데이터는 없다.

ASVspoof 5는 공식 Zenodo 및 LICENSE.txt를 확인했다. 데이터베이스 ODC Attribution 라이선스는 개별 음원 등 모든 권리를 자동 허가하지 않는다고 명시한다. source/speaker/generator와 upstream 노출을 추가 확인해야 한다. LibriSpeech는 공식 OpenSLR CC BY 4.0을 확인했지만 현재 학습에 같은 corpus가 있어 신규 source OOD로 취급하지 않는다. HAIM/SONICS는 현재 모델에서 이미 사용한 source이므로 다른 파일이라는 이유만으로 source-disjoint OOD가 되지 않는다. HAIM은 일부 상용 플랫폼 자료를 URL manifest로만 제공하며 플랫폼 약관 및 원작 권리까지 확인해야 한다. 데이터 card의 변화는 로컬 고정 revision을 소급 변경하지 않는다.

권한을 확인한 새 real 음성·음악 원본, 동의받은 화자/연주자/곡 및 공개적 접근 조건을 충족하는 합성 자원을 확보하는 것이 추가 후보 경로다. 특정 생성기의 이용조건은 아직 확인하지 않았으므로 그 합성 자료의 자동 수집 코드는 포함하지 않았다. 모든 파일의 URL/revision/라이선스 원문 사본·해시/변형 방법/재배포 허용 여부를 ledger에 보관한다. 대회에서 요구하는 학습 데이터 제공과 권리조건이 충돌하면 해당 자료를 제외한다.

## 추가로 필요한 자산

새 원본 오디오와 원본별 라벨/parent/speaker/track/source/generator/provenance; 재배포 조건 증거; 전체 대형 PANN/HTDemucs/DF-Arena/MERT 가중치와 local config/processor/code; 번들 ffmpeg; manifest가 가리키는 실제 6개 head 파일; 새 meta-dev/OOD의 current GPU F1/F2 telemetry 또는 schema 일치 extended cache가 필요하다. 기존 10개 캐시에는 PANN 527-class clip 배열, raw branch 특징, stem/raw 에너지 원신호가 없다. 이 항목들은 0으로 채우지 않는다.

실제 independent counts가 확정되기 전에는 학습/공식 점수/일반화 개선/소요 GPU 시간을 보고하지 않는다.
