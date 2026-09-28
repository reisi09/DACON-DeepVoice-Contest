너는 DACON 236749 오디오 딥페이크 판별 프로젝트의 실험 설계와 구현을 맡는다. 나는 사용량을 아끼기 위해 일반 ChatGPT 채팅에서 설계·코드·검증 자료를 먼저 만들고, 로컬 Codex에서는 반환된 산출물을 검사하려고 한다. Work 작업을 만들거나 GitHub·VESSL에 접속하지 마라. 계획만 제시하고 끝내지 말고, 이 채팅의 도구로 가능한 구현과 CPU 검증까지 진행하라.

첨부한 `DACON_Chat_Stacking_Input.zip`이 입력이다. 먼저 `00_START_HERE.md`, `02_CONTEXT_AND_CONSTRAINTS.md`, `03_INPUT_MAP.md`, `04_RETURN_CONTRACT.md`를 읽고, 거기에 지정된 현재 추론 소스와 학습 소스를 확인하라. 과거 루트 파일이나 대화의 일반론보다 현재 제출 snapshot의 실제 코드가 우선이다. ZIP을 열 수 없다면 읽었다고 하지 말고 필요한 파일 이름을 구체적으로 알려라. 실행 도구가 없다면 실행·테스트 통과를 주장하지 말고 완전한 파일 내용을 경로별 코드 블록으로 제공하라.

목표는 PANN Presence와 음성·음악 Fake 점수를 고정 수식 `max(Pv*Fv, Pm*Fm)`으로 결합하는 부분을, 예측 과정의 더 다양한 피처를 쓰는 학습형 스태킹과 비교하는 것이다. 첫 구현은 `FILE_FAKE_PROB`만 변경한다. 나머지 네 출력은 기존 동작을 유지한다. 기존 GPU encoder를 새로 학습하거나 아키텍처를 교체하지 않는다. 스태킹은 모델 출력들을 메타 모델로 결합하는 구조이고, 메타 모델 후보는 L2 Logistic Regression, 배깅 계열 ExtraTrees, 얕고 정규화한 XGBoost다. 개선은 가설이며 확정된 결과가 아니다.

다음 작업을 순서대로 수행하라. 사용자에게 단계마다 재확인을 요구하지 말고, 첨부 자료로 결정 가능한 것은 근거와 함께 결정하라. 필요한 외부 자료가 없으면 입력 계약과 미확보 목록을 남기고 나머지 작업을 계속하라.

1. 현재 구조와 피처 가용성 감사
   - 정확한 로딩·음량 처리·분리·window·head·seed 평균·Presence 집계·무음 처리 순서를 코드로 확인하라.
   - `reference/legacy/`는 과거 참고 코드다. 현재 head나 schema에 그대로 연결하지 마라.
   - `diagnostic/`의 실제 예측 CSV에서 네 기본 확률, seed별 점수, 원래 DF-Arena 점수 등을 확인하라.
   - `fixtures/cache/`의 실제 NPZ와 `fixtures/heads/`를 읽고 CPU에서 재구성 가능한 값과 새 GPU 추출이 필요한 값을 나눠라.
   - 음악과 음성/PANN의 시간축이 같다고 가정하지 마라. 단위·window 길이·hop·마지막 window·padding 유효 길이를 명시하고, 정렬 피처가 필요하면 실제 시간 겹침 또는 동일 window 재추출의 규칙을 구현하라.

2. 데이터 및 split 계획
   - 초기 규모는 약 3,200개(Voice-only 800, Music-only 800, Mixed 1,200, Stress 400)를 목표로 하되, 독립 원본·출처·생성기가 부족하면 숫자를 억지로 채우지 마라. 파일 수와 독립 parent/화자/곡/생성기 수를 따로 보고하라.
   - 약 2,560개 dev와 640개 locked OOD를 출발점으로 검토하라. 원본을 먼저 split하고 split 안에서만 혼합·partial·codec 변형을 만든다. Mixed는 두 부모를 모두 추적한다. Stress와 다른 범주가 중복 집계되지 않게 한다. RR/FR/RF/FF 및 알려진 성분 라벨을 명시한다.
   - 기존 9,504개 학습·선택·과거 개발 노출과 upstream 사전학습 노출을 구분하라. CSV에 없다는 이유로 새 데이터라고 판정하지 마라. hash, parent, speaker, track, generator family 및 provenance로 가능한 범위의 중복을 확인한다. 불명확한 upstream 노출은 unknown으로 기록한다.
   - 주 경로는 현재 전문가를 고정하고, 그 학습·선택에 쓰이지 않은 새 meta-dev 및 locked OOD를 확보하는 방식이다. 이 경우 meta-dev에서 grouped CV를 수행하고 모델 선택과 early stopping은 fold 내부에서만 한다.
   - 기존 base-head 학습 데이터를 재사용하는 대안은 별도로 설명하라. 학습에 쓴 head의 in-sample 점수를 OOF라고 부르면 안 된다. base-head 학습·정규화·checkpoint 선택·meta-head 학습의 각 단계에서 평가 그룹이 배제되도록 nested/cross-fitting 설계를 명시하라. 전역 OOF 행을 단순 재분할하는 것만으로 전체 스택의 평가가 독립적이라고 주장하지 마라. frozen encoder 특징만 재사용할 수 있는 조건과 base-head 재학습 비용을 분리하라.
   - 현재 예측 CSV는 진단·코드 재생에만 사용한다. 현재 val은 기존 head 선택에 쓰였고 test도 과거 개발 노출이 있다. 이름을 바꿔 독립 평가로 만들지 마라.
   - 데이터 후보·라이선스·재배포 조건은 공식 원문 URL과 확인 상태를 기록한다. 웹 확인이 불가능하면 미확인으로 남긴다. 실제 데이터가 없는 상태에서 수집 완료를 주장하지 마라.

3. 피처와 비교 실험
   - F0: 네 기본 점수(Pv, Pm, Fv, Fm). 기존 max와 noisy-OR를 비학습 비교 기준으로 둔다.
   - F1: seed별 점수·분산·범위, 음성/음악 window 점수의 mean/max/std/quantile/top-k 요약, 기존 원래 DF-Arena 점수 등 실제로 얻을 수 있는 피처.
   - F2: 새로 추출할 PANN 클래스/시간 요약, 정렬된 Presence×Fake, 지속 길이·coverage, raw/stem 불일치, 에너지 비율 등. 추가 encoder 호출·캐시·배포 비용을 함께 기록한다. raw/stem 신뢰도는 가설이지 정답 라벨이 아니다.
   - 우선 수십 개의 피처로 시작한다. 수천 차원 임베딩 전체, 거대한 탐색, 새로운 encoder는 기본안에서 제외한다. 임베딩 축소가 필요하면 fold 안에서만 학습하고 별도 후속 실험으로 둔다.
   - 정답·mask·window_truth·route·source·generator·ID·경로·split·노출 여부·정답에서 계산한 SNR/fake 비율은 모델 입력 금지다. 그룹 및 평가 메타데이터로만 사용한다. 길이·음량·codec 관련 피처는 shortcut 위험군으로 분리해 추가/제거 실험을 한다.
   - 동일한 split과 입력 행에서 F0/F1의 Logistic Regression, ExtraTrees, XGBoost를 비교해 모델 효과와 피처 효과를 구분하라. F2는 추가 추출 뒤 별도 비교한다. 모델별 탐색은 사전 선언한 소수 조합으로 제한한다. 작은 데이터에서 깊은 트리·과도한 탐색을 피한다.
   - 데이터 증강이 많은 parent가 과대 가중되지 않도록 그룹별 표본 수를 고려하되, 학습 가중치가 확률 보정에 주는 영향을 설명하라. bagging의 OOB 점수를 source-disjoint 검증의 대체물로 쓰지 마라.

4. 구현
   - 기존 프로젝트를 대규모 리팩터링하지 마라. 실제 feature adapter, 학습/evaluation entrypoint, 모델 저장·로드, 파일별 추론 함수를 구현하라. 최소한의 파일로 기존 수치 연산·scorer를 재사용한다.
   - XGBoost를 사용할 수 없으면 해당 후보를 명시적으로 NOT_RUN으로 남기고 LR/ExtraTrees 및 adapter 작업은 계속하라. 다른 모델로 몰래 대체하지 마라. 필요한 설치 버전·의존성은 별도 파일에 고정하되, 이 채팅에서 확인하지 못한 호환성을 보장하지 마라.
   - 캐시를 재계산 없이 읽는 CPU 경로와 새 오디오의 GPU 추출 경로를 분리한다. 확장 캐시에는 schema, feature order, audio/code/checkpoint hash, timebase, preprocessing, base model exposure provenance를 저장한다.
   - 없는 피처를 0으로 채워 ‘측정값’처럼 쓰지 마라. 불필요한 피처는 해당 실험에서 제외하고, 필수 피처 누락은 명확히 실패시킨다. 허용한 missing value 처리가 있다면 학습과 추론에서 같은 계약을 적용한다.
   - 모델 입력은 feature allowlist만 허용하고 추론 시 컬럼 순서를 검증한다. unknown 라벨은 제외하며 0으로 바꾸지 않는다. 비어 있거나 단일 클래스인 fold는 진단 이유와 함께 실패/미평가로 처리한다.
   - scaling/imputation/feature selection/calibration/early stopping을 평가 fold나 locked OOD에 맞추지 마라. locked OOD는 별도 명령과 후보 lock 후에만 평가할 수 있게 한다.
   - 출력 CSV는 기존 6개 열(ID+5개 확률)을 유지한다. 첫 단계에는 File만 변경한다. 추론 시 다른 파일의 통계나 라벨을 사용하지 않는다.
   - 저장 형식은 라이브러리의 표준 기능을 사용한다. XGBoost 모델은 JSON/UBJ 등 지원 포맷으로 저장하고 입력 schema·학습 provenance를 동반한다. 신뢰되지 않은 pickle을 실행하는 배포 흐름은 만들지 마라.
   - CPU 실험 notebook을 만들 경우 핵심 로직은 모듈을 호출하게 하고, notebook에 중복 구현하지 마라. GPU/VESSL 실행 안내는 작성하되 실제 인스턴스 실행·업로드·제출은 하지 마라.

5. 평가와 유의미한 검증
   - 기존 scorer의 이산 ROC/EER 정의와 unknown/presence mask를 유지하라. 기본 학습 loss는 logloss/BCE로 두고, EER을 미분 가능한 loss처럼 취급하지 마라.
   - File EER을 주 비교로 사용하되 5개 출력 전체의 지표·분모·누락 라벨을 보고한다. 완전한 라벨이 없으면 공식 총점은 null이다. 네 출력이 고정됐을 때 총점 변화의 해석도 분리한다.
   - parent 연결 그룹 단위 paired bootstrap과 조건별 결과를 설계하라. bootstrap에서 클래스가 사라지는 경우를 처리한다. 확률의 정확성은 Brier/logloss 등으로 따로 보고, EER 개선을 calibration 개선이라고 쓰지 마라.
   - 각 테스트가 방지하는 결함을 먼저 적어라. 실제 head+cache 재계산, 누수 provenance 거부, train-fold만으로 전처리, feature order 불일치 거부, 저장 후 재로드 일치, 파일 순서/부분집합 독립성, unknown 라벨 처리, 원본 baseline 재현을 우선한다. 단순 파일 존재·shape 확인·테스트 개수 늘리기는 하지 마라.
   - synthetic fixture는 wiring 확인용으로 표시한다. 10개 실제 캐시와 과거 val/test 결과로 새 모델의 일반화 개선을 주장하지 마라. GPU parity·L4·공식 점수는 실행 전에는 모두 미검증이다.

6. 결과 반환
   - `04_RETURN_CONTRACT.md`에 맞춰 완전한 구현 파일, 설정, 피처 명세, 데이터 계획, 실행 명령, 검증 로그와 미검증 목록을 `DACON_Stacking_Return.zip`으로 제공하라.
   - 기준 입력의 SHA-256, 변경 파일과 이유, 실행한 명령과 결과, 추가로 필요한 오디오/가중치/캐시를 기록하라. 생략 부호·TODO로 핵심 함수를 비워 두지 마라. 외부 자산이 없어 실행하지 못해도 구현 자체는 가능한 범위에서 끝내라.
   - 끝에 ‘완료한 작업 / 실제 실행한 검증 / 아직 실행하지 않은 작업 / Codex가 검사할 위치’를 구분해 짧게 보고하라. 다른 세션의 도구·로그를 지금 실행한 것처럼 서술하지 마라.
