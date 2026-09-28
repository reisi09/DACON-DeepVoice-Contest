# 기존 음악 1,322개 출처·라이선스 복원

2026-09-18 현재 선택된 `original_music_ids.json`의 1,322개를 모두 과거 원본 기록에 연결했다. 현재 parent ID와 source SHA-256이 upstream registry에 정확히 일치했으며, 라이선스·라이선스 URL·출처 URL·attribution이 미확인인 행은 0개다. 데이터셋이나 오디오를 수정하지 않았고 이번 조회에서 오디오 전체 byte를 다시 해시하지는 않았다.

| 출처 | 행 | 연결 근거 | 라이선스 처리 |
|---|---:|---|---|
| FMA | 218 | canonical 90개는 v4 ID 및 FMA content ID, fresh 128개는 원본 ID·SHA-256 | 개별 곡의 license title·URL·artist·track title 보존 |
| Echoes | 1,026 | canonical parent ID·prepared audio SHA-256, v4 ID, archive member | 보관한 README의 CC-BY-SA-4.0과 개별 FMA reference 정보 별도 보존 |
| SONICS | 78 | fresh manifest 원본 ID·SHA-256, pinned archive·member | 보관한 README의 CC-BY-NC-4.0 및 citation 저자 보존 |

canonical 1,116개(FMA 90개·Echoes 1,026개)는 모두 `_workspace/v2_discovery/echoes_fma_unique.json`의 개별 FMA reference metadata와 정확히 연결됐다. FMA에는 dataset 전체의 CC-BY-SA를 대신 부여하지 않았다. Echoes·SONICS 출처는 다운로드 가능한 pinned archive URL과 정확한 archive member를 함께 적었다.

현재 `source_sha256`이 historical prepared audio의 hash인 경우에는 v4의 원본 hash를 `raw_original_sha256`에 따로 기록했다. 두 hash는 서로 다른 처리 단계의 파일을 가리키므로 일치해야 하는 값으로 취급하지 않았다.

## 보존해야 할 조건

SONICS 78개에는 비상업적 조건이 명시돼 있다. 기존 FMA 중 아래 한 곡도 개별 metadata에서 `Attribution-Noncommercial-Share Alike 3.0 United States`로 확인됐다. 이전에 학습에 사용했다는 사실을 별도 허가로 취급하지 않았다.

- ID: `core_b1f8c01603f779c9b200_base`
- parent ID: `b2d4078be26482295d50e369`
- artist: `Los Llamarada`
- license URL: `http://creativecommons.org/licenses/by-nc-sa/3.0/us/`
- source URL: `https://files.freemusicarchive.org/storage-freemusicarchive-org/music/WFMU/Los_Llamarada/Close_Your_Eyes/Los_Llamarada_-_07_-_Im_Gonna_Try_To_Reach.mp3`

이 FMA 곡을 reference로 삼은 Echoes 생성물도 선택 목록에 4개 있다. 모두 TTA이며 `source`는 `Echoes/diffrhythm/TTA`, `Echoes/acestep/TTA`, `Echoes/stableaudio/TTA`, `Echoes/audioldm/TTA`다. reference의 조건과 Echoes dataset의 선언을 별도 열에 남겼다. reference의 조건이 TTA 생성물에 자동으로 이전되거나 dataset 선언이 reference의 조건을 대체한다고 추정하지 않았다.

이 산출물은 보관된 출처 조건을 복원한 것이다. 대회의 외부 데이터 규정 충족 여부, 특정 사용의 비상업성, 재배포 조건을 새로 판정한 법률 검토가 아니다. root가 확인한 대회 규정과 사용 범위에 이 기록을 연결하면 된다. 출처 metadata와 README의 현재 웹 상태도 새로 조회하지 않았다.

## 산출물과 근거

행별 기록은 [historical_license_lineage.csv](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/localization_a100_20260918/historical_license_lineage.csv), 연결 결과·조건별 목록·입력 파일 SHA-256은 [historical_license_lineage.json](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/localization_a100_20260918/historical_license_lineage.json)에 저장했다. CSV의 `provenance_file`, `license_evidence_file` 열에서 각 근거로 돌아갈 수 있다.

주요 근거는 `/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/redesign_20260917/input-preparation/sources.csv`, `/Users/choiwuseck/Documents/ChatGPT/데이콘/v4_validation/provenance.json`, `/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/v2_discovery/echoes_fma_unique.json`, 기존 두 pilot의 fresh FMA·SONICS manifest다. Echoes README의 2행에 CC-BY-SA-4.0이 있고, SONICS README의 2행 및 131–132행에 CC-BY-NC-4.0과 비상업적 연구용 조건이 있다.

<!-- HUMANIZE-SUMMARY: Conservative final Korean editing pass; identities, row counts, exact license names and URLs, attribution evidence, and uncertainty boundaries preserved. -->
