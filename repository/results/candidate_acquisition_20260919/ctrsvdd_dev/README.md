# CtrSVDD dev 원본 검사

`audit.json`과 `catalog.csv`는 41,728개 원본에 대한 최초 전수 검사 결과입니다. 원본 ZIP 및 프로토콜 해시와 각 파일의 CRC/SHA-256을 기록했습니다. 배포 프로토콜의 43,625행 중 실제 오디오가 없는 1,897행은 `missing_audio.csv`에 따로 보관했습니다.

편입 후보는 26,849개(real 4,650, fake 22,199)입니다. SVC 14,835개는 입력 부모 관계가 불명확해 보류했습니다. 품질 기준 미달 44개는 전부 길이 0.25초 미만이며 `quality_exclusions.json`에 실제 길이를 남겼습니다. 최초 보고서 오류 문자열의 BytesIO 메모리 주소는 재현 가능한 ID가 아닙니다. 다음 실행부터 이 주소는 제거합니다. 완료된 최초 CSV/감사 해시는 변경하지 않았습니다.

편입은 학습 준비 완료를 뜻하지 않습니다. 실제 선택 결과와 남은 한계는 `results/reconstructed_bank_20260919_ctrsvdd_dev/README.md`를 확인하세요. `claude_review_ax.txt`는 Opus 5의 보조 의견이며 원천 증거가 아닙니다.
