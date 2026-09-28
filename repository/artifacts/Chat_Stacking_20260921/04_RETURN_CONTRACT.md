# 반환 산출물과 검사 기준

최종 파일명: `DACON_Stacking_Return.zip`. ZIP의 최상위에서 `RETURN_REPORT.md`를 바로 찾을 수 있게 한다. 아래 문서 및 실제 구현을 포함하고, 불필요한 입력 원본 전체·큰 가중치를 다시 복제하지 않는다.

## 필수 산출물

- `RETURN_REPORT.md`: 완료/미완료, 실제 실행 환경, 검증 명령 및 결과, 가설과 관측 결과 구분, 남은 입력.
- `DESIGN.md`: 코드 근거, baseline 흐름, F0/F1/F2, 비교 모델, split·OOF·선택·최종 holdout 절차.
- `FEATURE_SPEC.csv`: 이름, 원본 필드/함수, 계산식, 시간 단위, availability, missing 처리, 학습 전처리, 추론 비용, shortcut 위험, 사용 실험.
- `DATA_PLAN.md`: 목표와 실제 확보 구분, parent/화자/곡/생성기 정책, split-before-mixing, 라벨/노출/license audit, 신규 확보 목록. 공개 데이터 후보에 접근하지 못했다면 미확인 표시.
- `src/`: 실제 캐시 adapter, 피처 생성, grouped 평가·학습, 직렬화·추론, 필요한 추가 추출 코드. 단일 CLI 또는 소수 명확한 명령으로 구성한다.
- `configs/`: 작은 후보 grid, seed, feature allowlist, split/identity 계약. 비밀번호·원격 토큰 없음.
- `tests/`: 실제 결함을 방지하는 최소 검증. 설명만 있는 테스트 계획은 구현 완료가 아니다.
- `RUNBOOK.md`: CPU fixture 확인, 신규 데이터 검사, 추가 GPU 추출, dev 모델 선택, lock, locked OOD, 추론 순서. 명령별 필요한 파일·기대 출력·실패 조건.
- `CHANGES.json`: 추가/수정 파일, 원래 저장소 적용 경로, 이유, 수정 대상의 입력 SHA-256. 기존 파일을 수정하면 완전한 수정본과 unified diff를 모두 제공한다.
- `VERIFICATION.json` 및 `logs/`: 실행/미실행/실패 상태와 명령·오류·허용 오차. 테스트 개수만 쓰지 않는다.
- `OUTPUT_SHA256.json`: 반환 파일별 해시. 자기 자신의 해시는 제외한다.
- notebook은 선택 사항이다. 만들면 위 모듈을 호출하는 얇은 실행·설명 노트북으로 제한한다.

## Codex의 수락 기준

| 검사 | 보호하는 실패 모드 |
|---|---|
| 현재 snapshot 및 입력 해시 일치 | 다른 버전의 모델에 잘못 패치 |
| 실제 head/cache → 기존 수치 재구성 | logit/probability 혼동, seed/시간 집계 순서 변경 |
| 기본 네 출력 및 baseline 경로 보존 | File 실험에 성분 모델 변경이 섞임 |
| 금지 피처·마스크·window_truth 차단 | 정답·출처·생성 recipe 누수 |
| base 학습·선택 노출 및 parent 교차 거부 | in-sample을 OOF로 오인, 혼합 부모 누수 |
| fold 내부 전처리와 early stopping | 검증 통계/라벨의 학습 유입 |
| 캐시 identity 및 시간축 검사 | 오래된 캐시, 어긋난 window 곱셈 |
| serialise/reload 동일 결과 | 배포 때 피처 순서/모델 버전 변경 |
| 순서·부분집합·단독 파일 동일 결과 | 다른 평가 파일 통계 사용 |
| 공식 scorer 및 unknown 처리 | EER 정의 변경, unknown을 real로 취급 |
| 개발 lock과 OOD 분리 | 최종 평가로 후보·threshold 선택 |

`fixtures` 10개는 일반화 성능 비교에 충분하지 않다. 실제 GPU·L4·새 OOD·공식 제출은 수행하지 않았으면 `NOT_RUN`으로 남긴다. 계약을 만족하는 코드를 만들어도 성능 개선은 별도 실험 결과가 있어야 주장할 수 있다.
