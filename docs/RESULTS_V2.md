# DACON 236749 V2 실행 기록

후속 [epoch 연장 실험](RESULTS_EPOCH_EXTENSION.md)에서 6 epoch 모델을 최대 30 epoch까지 이어 학습했다. 22 epoch에 조기 종료하고 17 epoch를 선택했으며, 개발 개선에 비해 기존 평가의 개선은 작았다. 아래는 원래 V2 실험 기록이다.

## 현재 상태

기존 그래프의 성능 저하를 진단하고, 출처를 분리한 1,029개 오디오로 AST 고정 표현·마지막 블록 미세조정·G0 결합을 실제 A100 Colab에서 비교했다. 원래 세 후보의 개발 선택은 `G0`다. 별도로 추가한 탐색 후보 `G0_AST_MUSIC`의 외부 평가 proxy는 `0.759036 → 0.792205`였지만, 새 생성기 구간에서는 `0.776938 → 0.756339`로 낮아졌다. 모든 조건에서 개선됐다고 결론 내릴 수는 없다.

학습·평가·오류 진단·수정 후 패키지 검증을 마쳤다. 가중치, 원본 자료, 실행 노트북, 예측 CSV, 로그, 검증한 후보 ZIP은 Drive에 저장했다. 처음 발생한 정밀도 오류와 원래 선택 기록도 보존했다. 새 DACON 제출은 하지 않았다.

기존 공식 베이스라인만 제출했다. 제출 번호 `88418`은 `baseline_submit.zip`으로 2026-09-13 00:35:37에 제출됐고, DACON 점수는 `0.6909110688`(ADS `0.6577539683`, CPS `0.9893249735`)이다. 이 점수는 DACON 평가 결과이며 아래의 부분 라벨 proxy나 외부 검증 점수와 섞지 않는다.

## 기존 그래프가 약화된 양상

기존 225개 자료의 65개 holdout은 이미 설계·비교에 쓰였으므로 독립 성능이 아니라 후향 진단에만 썼다. 이 holdout은 원자 source group 21개지만 mixture 부모를 연결하면 17개 component다. 음성 부재 라벨도 없어 음성 존재 AUC를 계산하지 못했다.

가장 중요한 관찰은 EER과 AUC가 항상 같은 방향으로 움직이지 않는다는 점이다. G1 음성은 EER이 `0.041958 → 0.083916`으로 나빠졌지만 AUC는 `0.980769 → 0.984266`으로 높아졌고, 선택된 ROC 점의 FP/FN만 `(1,1) → (2,2)`가 됐다. G2 파일은 EER이 G0와 같지만 AUC가 `0.945122 → 0.924797`로 낮아졌다. G3 음악은 EER이 `0.307895 → 0.384211`, FP/FN이 `(3,6) → (4,7)`로 바뀌었지만 AUC는 `0.721053`으로 같다. G4는 dev 파일 EER이 `0.187500 → 0.143229`으로 좋아졌어도 기존 holdout에서는 `0.123476 → 0.156504`으로 나빠졌다. 이는 순위 전체 악화나 특정 생성기 탓을 단정할 근거가 아니며, 작은 연결 집단에서 그래프별 ROC 작동점과 전이 양상이 다르다는 증거다. 전체 수치와 pair-order 변화는 [forensic_interpretation.md](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/v2_evidence/forensic_interpretation.md)에 남겼다.

## 데이터와 분리

실제 실행 자료는 1,029개 WAV다. 새 원본 740개, 기존 225개, 새 mixture 64개로 구성했고, 역할별 파일 수는 `dev_train/dev_select/test_generator/test_source = 633/122/176/98`, mixture 부모를 연결한 집단 수는 `197/40/58/30`이다. 개발과 두 테스트 역할은 원문·아티스트·악기·mixture 구성원을 넘나들지 않게 분리했다.

음성은 MLAAD-tiny/M-AILABS의 실제·합성 짝, 음악은 Echoes TTA와 FMA 실제곡의 제목·아티스트 대응쌍, 음성 부재 대조는 NSynth acoustic non-vocal 악기를 사용했다. Echoes는 TTA만 사용했고, FMA·Echoes의 음성 존재 여부는 근거가 없으면 빈 라벨로 남겼다. NSynth는 실제 노래 일반을 대표하는 자료가 아니라 음성 부재 악기 대조군이다. 사용 원본·라이선스·리비전·SHA-256·PCM SHA-256은 `source_inventory.json`과 Drive의 분할 원본 ZIP에 보존했다. 자세한 출처와 복원 방법은 [DATA_SOURCES_V2.md](/Users/choiwuseck/Documents/ChatGPT/데이콘/DATA_SOURCES_V2.md)에 있다.

## 모델과 실제 학습

비교 기준은 기존 G0이고, 새 backbone은 고정 revision의 AudioSet AST다. AST 출력 위에 `FILE_FAKE`, `VOICE_FAKE`, `MUSIC_FAKE`, `VOICE_PRESENT`, `MUSIC_PRESENT` 다섯 head를 두었다. `AST_FROZEN`은 backbone을 고정하고 head만 학습했으며, `AST_LAST_BLOCK`은 마지막 encoder block·최종 LayerNorm·head만 미세조정했다. 미상 라벨과 존재하지 않는 성분의 진위 라벨은 loss와 지표에서 마스킹했다.

Colab A100에서 `AST_FROZEN`은 8 epoch, `45.069`초를 학습했다. 3,845개 파라미터가 학습 대상이었고, head gradient는 기록됐지만 backbone gradient는 매 epoch `0`이었다. `15.964070906396955`는 epoch별 절대 가중치 변화량 합의 최댓값이며 최종 checkpoint와 시작 checkpoint의 net 변화량은 아니다. `AST_LAST_BLOCK`은 6 epoch, `46.648`초를 학습했다. 학습 대상은 7,093,253개였고, 첫 epoch의 head/backbone gradient norm은 `2.212255/2.300445`, 마지막 epoch는 `1.461752/2.511912`였다. `424.7645787281217`도 같은 정의의 epoch별 절대 변화량 합 최댓값이다. 즉 마지막 block 경로에 gradient가 실제로 흘렀고, 학습 중 가중치 갱신도 기록됐다.

`training_report.json`의 `5.2554168701171875 GiB` allocated, `5.802734375 GiB` reserved는 두 후보를 같은 Colab 프로세스에서 순차 실행한 과정의 process-level 최대치다. 이를 각 후보의 독립 peak으로 해석하지 않는다. A100 수치 역시 DACON L4의 60분 조건 충족 증거는 아니다.

## 개발 선택과 lock

원래 세 후보의 `dev_select` 부분 라벨 proxy는 G0 `0.757797`, AST_FROZEN `0.822717`, AST_LAST_BLOCK `0.865720`이었다. AST_FROZEN은 proxy gain `0.064920`, bootstrap win rate `0.9195`, p10 `0.003764`를 보였고, AST_LAST_BLOCK은 `0.107922`, `0.9955`, `0.055208`을 보였다. 그러나 두 AST 후보 모두 음성 EER 및 대응 AUC 악화 gate를 넘지 못해 원래 규칙에서는 G0가 선택됐다. 이 proxy는 완전한 DACON 점수가 아니라 개발 선택용 부분 라벨 값이다.

원래 프로토콜의 최종 선택은 G0로 유지한다. 다음 결합 모델은 개발 결과를 보고 추가한 탐색 후보이며, 원래 세 후보와 같은 사전 등록 검증으로 인정하지 않는다. 별도 `hybrid_selection.json`과 replay `finished.json`의 `selected`는 이 추가 실험 안에서의 개발 선택을 뜻한다. 이를 원래 `candidate_lock.json`의 선택 변경으로 해석하지 않는다.

이후 `FILE_FAKE`·`VOICE_FAKE`는 G0, `MUSIC_FAKE`는 AST_LAST_BLOCK, `VOICE_PRESENT`·`MUSIC_PRESENT`도 AST_LAST_BLOCK을 쓰는 고정 routing `G0_AST_MUSIC`을 별도 개발 프로토콜로 평가했다. routing protocol은 17:54:29 UTC에 등록됐고 root가 fresh 예측·지표를 읽기 전이었다. 다만 primary script는 이 protocol 등록 전에 이미 test를 열었으므로, protocol이 program의 test 개봉보다 앞섰다고 쓰지 않는다. 별도의 dev selection/checkpoint lock은 17:56:42.639526 UTC에 작성됐고, root fresh read는 17:57 이후였다. 이 후보의 dev proxy는 `0.884715`, G0 대비 `+0.126918`, component bootstrap win rate `1.0`, p10 `0.087857`로 모든 gate를 통과했다. lock에는 선택 결과, 개발 manifest, 원래 baseline ZIP, AST checkpoint와 selection hash가 함께 기록돼 있다. root fresh read 전에 fixed routing과 dev-only selection을 등록했으며, 이후 fresh 결과를 보고 후보·임계값·학습 설정을 바꾼 것이 아니다.

## 봉인 후 외부 테스트 결과

원래 세 후보용 primary script가 274개 외부 테스트를 연 뒤, 별도 hybrid routing을 잠갔다. `test_generator` 176개는 학습에 없던 생성기 조건이고, `test_source` 98개는 다른 source 조건이다. 이 결과 역시 DACON 리더보드가 아니다. 빈 라벨은 제외했으므로 공식 종합 점수는 모두 `null`이다.

| 구간 | 후보 | 파일 EER | 음성 EER | 음악 EER | 음악 존재 AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| combined 274 | G0 | 0.204530 | 0.086557 | 0.492867 | 0.994584 |
| combined 274 | G0_AST_MUSIC | 0.204530 | 0.086557 | 0.371024 | 1.000000 |
| test_generator 176 | G0 | 0.220982 | 0.141098 | 0.322479 | 1.000000 |
| test_generator 176 | G0_AST_MUSIC | 0.220982 | 0.141098 | 0.394958 | 1.000000 |
| test_source 98 | G0 | 0.163333 | 0.034648 | 0.653673 | 0.982860 |
| test_source 98 | G0_AST_MUSIC | 0.163333 | 0.034648 | 0.173163 | 1.000000 |

hybrid는 combined 음악 EER을 낮췄지만, `test_generator` 음악 EER은 G0보다 높고 `test_source`에서는 낮다. 파일·음성 출력은 hybrid에서 G0를 그대로 사용하므로 표에서도 동일하다. 이 slice 차이는 전체 개선이나 새 생성기 일반화를 주장할 근거가 아니다.

부분 라벨 proxy와 component bootstrap도 구간별로 갈린다. 개발 proxy는 선택에 썼고, 아래 fresh proxy와 bootstrap은 설명에만 썼다. DACON 점수는 아니다. generator 구간에서는 음성 존재 라벨이 한 클래스뿐이라 해당 AUC를 계산할 수 없다. 구간마다 계산 가능한 지표의 가중치를 다시 정규화했으므로 서로 다른 구간의 proxy 크기를 직접 비교하지 않는다.

| 구간 | G0 proxy | hybrid proxy | component 수 | bootstrap win rate | p10 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined | 0.759036 | 0.792205 | 88 | 0.9350 | 0.005478 |
| test_generator | 0.776938 | 0.756339 | 58 | 0.3245 | -0.040855 |
| test_source | 0.742915 | 0.873509 | 30 | 0.9990 | 0.098781 |

post-lock 설명 분석에서 FMA 실제/ Echoes TTA의 raw 음악 파일은 generator 구간 `20/60`, source 구간 `5/15`개였다. generator 구간에서 threshold 0.5 정확도는 G0 `54/80`, AST_LAST_BLOCK `43/80`이었고, source 구간은 `14/20`, `16/20`이었다. 특히 source FMA 실제곡은 5개뿐이므로 그 차이는 안정적인 추정이 아니다. NSynth 실제 악기 16개에서 G0의 `MUSIC_FAKE` 평균은 `0.912315`, AST_LAST_BLOCK은 `0.003449`였지만, 이 대조군 차이가 raw FMA/Echoes나 mixture 전체의 원인을 설명하지는 않는다. mixture는 별도 분포로 기록했다. 상세 분포와 금지한 해석은 [fresh_error_analysis.md](/Users/choiwuseck/Documents/ChatGPT/데이콘/results/v2_evidence/fresh_error_analysis.md)에 있다.

## Colab·Drive·패키지 상태

초기 A100 Colab 실행은 첫 셀부터 실제 `13.54`분 동안 진행됐고, 학습·선택·봉인 후 비교 결과까지 Drive에 저장했다. 그러나 fresh offline package를 새 디렉터리에 풀어 실행하는 단계에서 AST_LAST_BLOCK 최대 차이 `2.2172927856445312e-05`가 기준 `1e-5`를 넘었다. 오류와 `release_requested.json`을 저장한 뒤 런타임을 자동 해제했고 active GPU session은 0으로 확인했다.

후속 GPU 진단은 네 checkpoint 자산 해시가 기준과 같음을 먼저 확인했다. 기본값 `cudnn_tf32=true`, `matmul_tf32=false`에서 같은 `2.2172927856445312e-05` 차이가 재현됐고, 두 TF32 toggle을 모두 `false`로 하면 차이가 `0.0`이었다. `cudnn_tf32=false`, `matmul_tf32=true`의 최대 차이는 `0.0007384419441223145`로 기준을 통과하지 못했다. 따라서 원인은 `confirmed_cudnn_tf32`로 기록했고, weight를 바꾸지 않은 채 package 추론에서 두 TF32를 끄도록 수정했다. 허용치 `1e-5`는 완화하지 않았다.

수정된 AST와 hybrid ZIP을 각각 새 디렉터리에 풀고, socket guard로 네트워크를 차단한 새 프로세스에서 실제 `script.py`를 실행했다. 공식 형식 확인용 dummy 3개와 60초 stress audio를 사용했으며, ID 순서와 모든 확률을 기준 예측과 비교했다. 이 네 파일의 검증은 배포 경로와 수치 재현성 확인용이다. 모델 품질은 앞의 274개 외부 평가로 따로 측정했다.

| 패키지 | 최대 확률 차이 | 모델 로드 포함 실행 시간 | ZIP 크기 | 해제 크기 |
| --- | ---: | ---: | ---: | ---: |
| AST_LAST_BLOCK | 0.0 | 13.453초 | 344,836,171 B | 344,834,903 B |
| G0_AST_MUSIC | 0.0 | 30.353초 | 5,348,298,498 B | 5,348,291,116 B |

AST의 60초 파일 추론은 모델 로드 후 `0.175010`초, 최대 allocated GPU memory `0.631093 GiB`였다. 이 값은 AST 경로의 측정이며 전체 hybrid의 독립 peak은 측정하지 않았다. ZIP 10GB·해제 32GB 조건은 충족하지만, DACON L4에서 1,200개 파일을 60분 안에 처리하는지는 아직 검증하지 않았다. [패키지 검증 JSON](https://drive.google.com/file/d/1Z3gOIfE8uForkoufoXpK-69hvC12M2xG/view)

첫 셀부터 해제 요청까지의 두 실행 시간 합은 `23.60`분이다. GPU 연결·인증 시간을 포함한 할당은 약 `24.23`분이었고, 종료 직후 터미널 확인으로 잠깐 시작된 추가 연결도 바로 해제했다. 전체 작업 전후 화면의 잔여 컴퓨팅 단위는 `55.96 → 54.49`로, 약 `1.47`개 감소했다. 최종 화면에서 활성 세션 `0개`, 사용률 `0/시간`을 확인했다. 기존 2시간 한도 안에서 종료했다.

- [학습·외부 평가 Colab](https://colab.research.google.com/drive/1cesnr-fH1lQm5-fFraiw4D2n-NfujeKa): 18개 코드 셀의 실행 기록과 최초 정밀도 오류를 보존했다.
- [정밀도 수정·패키지 검증 Colab](https://colab.research.google.com/drive/19_Uvy8xgwlED-akOyBQRIk0bWAJs8CKz): 4개 코드 셀 실행 완료, 저장된 notebook output에 오류가 없다.
- [전체 결과 Drive 폴더](https://drive.google.com/drive/folders/1IqE0FBeDhoSrxtrwZQOZeJoK93faWosO)
- [수정·검증 결과 Drive 폴더](https://drive.google.com/drive/folders/1XmEldtxnct9p1Xtv142lAUUeyFAYP--L)
- [추가 탐색 후보 G0_AST_MUSIC ZIP](https://drive.google.com/file/d/12q2MlR3OmiTtug5uwEbIAqWbjIGYjhGX/view): SHA-256 `a784484ca7fcc42a8ee8092d9f0ec382621d4ec7392e6e6f5a700971792ed735`.
- [파이프라인 그래프와 선택 규칙](/Users/choiwuseck/Documents/ChatGPT/데이콘/PIPELINE_GRAPH_V2.md)

로컬 검증에서는 출처·mixture 부모 분리, 미상 라벨 마스킹, 실제 gradient와 가중치 갱신, 학습·배포 resampling 일치, checkpoint 저장·복원, 파일 간 독립성, 고정 routing을 확인했다. 최종 회귀 검증 기록은 `52 passed, 8 subtests passed in 12.28s`다. 이는 실험 품질 점수가 아니며, 실제 측정 결과와 적용 범위는 위에 구분해 기록했다.
