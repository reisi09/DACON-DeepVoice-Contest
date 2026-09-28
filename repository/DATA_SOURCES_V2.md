# V2 학습·검증 데이터 출처

이 데이터 묶음은 DACON 236749 규정에 따라 공개 접근 가능하고 비상업적 이용이 허용된 자료만 사용한다. `source_inventory.json`에는 실제로 사용한 원본 파일의 URL 또는 ZIP 멤버, 고정 리비전, SHA-256, 디코딩 후 PCM SHA-256, 라이선스, 출처와 원자 단위 그룹을 기록한다. URL만 보관하는 대신 실제 원본도 `_workspace/v2_sources/originals/`에 유지한다.

| 자료 | 고정 버전 | 사용 범위 | 라벨 근거 | 라이선스 |
|---|---|---|---|---|
| MLAAD-tiny / M-AILABS | `9143e5ea709575ebab6bec52840a1043aada7bb1` | 10개 생성기에서 실제·합성 음성 20쌍씩 | 음성 존재=1, 음악 존재=0, 실제·합성은 저장소 디렉터리 | 합성 CC BY-NC 4.0, 실제 M-AILABS 고지 |
| Echoes | `14b0c76c6a691c42fadfab9fb6a4eb1ee8c628a2` | TTA만 사용, 60개 FMA 기준곡마다 생성기 3개 | TTA 합성 음악=1; 음성 존재 여부는 알 수 없어 공란 | CC BY-SA 4.0 |
| Free Music Archive | 범위 추출한 메타데이터 스냅샷 | Echoes 기준 제목·아티스트와 유일하게 대응한 60곡 | 실제 음악=0; 음성 존재 여부는 알 수 없어 공란 | 곡별 비 ND Creative Commons, inventory에 원문 URL 기록 |
| NSynth test | `5ba57c8114d9d4d7cf076c4cb3c8f508f105fca8` | acoustic·non-vocal 25악기, 악기당 4음 | 실제=0, 음악 존재=1, 음성 존재=0 | CC BY 4.0 |

Echoes 전체 8.6GB ZIP은 받지 않는다. 중앙 디렉터리에서 선택한 멤버의 정확한 바이트 범위만 받고 로컬 헤더, 압축 방식, 비압축 크기와 CRC32를 검증한다. FMA 메타데이터 멤버는 ZIP method 12(BZIP2)로 검증했다. AIME를 새로 추가하지 않았으며, 기존 개발 자료에 포함된 25개는 유지했다.

기존 225개는 모두 개발 자료로 재분류한다. 새 자료는 원자 그룹과 혼합물의 모든 구성 그룹을 연결 성분으로 다루며 `dev_train`, `dev_select`, `test_generator`, `test_source` 중 하나에만 둔다. `test_generator`와 `test_source`는 후보 선택이 끝난 후 한 번만 평가한다. 공란은 근거가 없는 라벨이며 0으로 바꾸거나 손실·지표에 포함하지 않는다.

최종 묶음은 새 원본 740개, 기존 파일 225개, 새 혼합 파일 64개로 총 1,029개다. 역할별 파일 수는 633/122/176/98, 혼합 부모를 연결한 집단 수는 197/40/58/30이다. 새 혼합 파일은 각 역할에 16개씩 들어간다. 실제 전경 음성에 합성 음악을 섞으면 배경에 합성 보컬이 있을 수 있어 `VOICE_FAKE`를 알 수 없음으로 둔다. 기존 혼합 파일의 V2 사본까지 포함해 32개의 음성 진위 정답을 보수적으로 마스킹했다.

실제 저장된 1,029개 WAV를 다시 디코딩해 `v2_validation/pcm_sha256.csv`와 모두 일치함을 확인했다. 고정 타임스탬프로 ZIP을 두 번 재생성한 SHA-256도 같았다: `f08eef8f309fab3a176a89fe34fffeaf687fc99ffe4f16c60157addaf9796b9e`.

Drive의 `v2_source_originals.zip.part00`–`part15`에는 원본·메타데이터·라이선스가 들어 있다. `v2_source_originals_parts.json`의 순서로 합치고 각 조각과 전체 ZIP의 SHA-256을 검증해 복원한다. 최종 출처 목록은 같은 폴더의 `v2_source_inventory.json`이다. 학습에 사용하는 가공 WAV와 manifest는 `v2_validation.zip.part00`–`part02`에 있으며 새 Colab 노트북에서 자동으로 검증·복원한다.

이 분리는 이번 실험 안에서 원문·아티스트·악기 중복을 막는다. MLAAD의 서로 다른 원문에 같은 화자가 있을 가능성이나 사전학습 모델의 데이터 노출까지 배제했다는 뜻은 아니다. NSynth의 짧은 악기 음은 음성 부재 대조군으로 사용하며 실제 노래와 같은 분포라고 가정하지 않는다.
