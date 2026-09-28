# Jamendo 수집 완료 기록

Jamendo 원본 **854곡, 669아티스트, 4,803,175,390바이트**를 확보했다. 기존 파일럿 320곡은 hardlink로 재사용했다. 이번에 새로 내려받은 원본은 **534곡, 534아티스트, 2,956,657,556바이트**이며, 이 534아티스트는 기존 파일럿에 없던 아티스트다. 기존 파일럿과 다른 데이터 은행은 수정하지 않았다.

전체 원본을 끝까지 디코딩하고 SHA-256을 원장과 다시 대조했다. 819곡은 기술 품질 검사를 통과했다. 나머지 35곡은 디코딩 파형에서 `abs(sample) >= 0.999`인 샘플 비율이 5%를 넘어 검토 대상으로 남겼다. 원본의 음량이나 파형은 바꾸지 않았다. 신규 원본만 보면 509곡 통과, 25곡 검토 대상이다. 이는 직접 청취한 음질 평가가 아니다.

원본 SHA-256과 16kHz mono float32 전체 디코딩 PCM SHA-256은 각각 854개가 모두 달랐다. 고정 평가용 27아티스트의 66곡은 그대로 예약했다. 평가 그룹·알려진 평가 해시·기술 품질 조건을 적용한 학습 후보는 755곡이며, 최종 실험 분할이나 학습을 수행한 것은 아니다. 음악의 `VOICE_FAKE`, `VOICE_PRESENT` 라벨은 모두 미확인 상태로 보존했다.

출처는 [JamendoMaxCaps의 고정 revision](https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps/tree/13090d785d071dcd617636deaa4a9775936ac995)에 수록된 2008~2015년 공개 메타데이터다. `audiodownload_allowed=true`이고 개별 곡의 CC BY, BY-SA, BY-NC, BY-NC-SA 조건을 확인한 곡만 골랐다. 제목·아티스트·원본 URL·정확한 라이선스 URL·공개일·메타데이터 원시 해시를 [곡별 원장](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/jamendo_sources.csv)에 남겼다. 저장소의 포괄적 라이선스로 곡별 라이선스를 덮어쓰지 않았다.

메타데이터 112파일은 473,299,278바이트다. 원본 5GB, 메타데이터 600MB 배정을 각각 지켰다. 미완성 응답 13개, 44,191,744바이트는 전체 예산 변경 때 제거했고, 최종 수집 후 남은 미완성 응답은 없었다. 원본과 메타데이터는 `_workspace/data_expansion_20260918/jamendo/`에 보관한다.

메타데이터의 `instrumental` 표시는 854곡 모두에 있지만, 이를 검증된 음성 부재 라벨로 사용하지 않았다. 2016년 이전 공개일도 출처 측 메타데이터이며 사람의 창작 여부를 독립적으로 검증한 증거는 아니다. 별칭·협업·재인코딩·부분 복제와 출처별 데이터 분포 차이는 여전히 확인이 필요하다.

수집 프로세스는 종료했고 다운로드 해제 marker도 닫았다. GPU, 학습, 자동화를 실행하지 않았다. 상세 수치와 검증 해시는 [최종 감사 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/jamendo_final_audit.json), 삭제 근거는 [미완성 응답 정리 기록](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_expansion_20260918/jamendo_partial_cleanup.json)에 남겼다.
