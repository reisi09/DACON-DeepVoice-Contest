(a) **INFERRED — Track 1은 음성 회귀 검사로만 채택. 제출을 기다리게 할 이유는 없음.**

- **UNKNOWN:** DACON에 더 정확한 대리 지표라는 근거 없음. E2의 AT-ADD 비중은 **10,000/12,426 = 80.48%**. Track 1 추가도 같은 데이터 계열 의존도를 높임.
- **INFERRED:** `eval`에서 **real 1,000 + fake 1,000**, source·language·generator별 분산 추출. 기존 학습·val·E1/E2와 중복 제거 후 표본 고정. 추가 학습·threshold 조정 금지.
- **VERIFIED:** Track 1은 대부분의 fake `source_id`·`speaker_id`를 `-`로 가림. **완전한 lineage 독립성은 UNKNOWN**. [공식 데이터 카드](https://huggingface.co/datasets/xieyuankun/AT-ADD-Track1)
- **INFERRED:** S7 대비 **VOICE EER +0.02 이상** 또는 old-val TPR95 threshold에서 **real FPR +0.03 이상**이면 해당 후보 보류. 그 외 **s7_new → W_new** 유지. MUSIC EER·전체 ADS 판단에는 사용하지 않음. **23:00 미완료면 생략**.

(b) 경로는 모두 `code_r14/`.

1. **BLOCKING · VERIFIED — parity 실패가 제출 준비를 중단시키지 않음.**  
   `w_runtime_parity.py:39`는 `FAIL`을 출력하고 정상 종료. `:19`는 receipt SHA/device를 확인하지 않으며 FILE도 비교하지 않음. `pkg_chain.sh:16`의 `pipefail`로 해결되지 않음.  
   **수정:** 검증된 S8 parity 방식 재사용. 패키지 자체 heads, SHA·device·정확한 ID 집합·finite 확인, **VOICE ≤0.02 / MUSIC ≤0.01 / FILE ≤0.01**, 실패 시 nonzero. **두 패키지 모두 수정된 검사와 CUDA 확인 전 제출 보류.** 실제 예측 오류가 발견됐다는 뜻은 아님.

2. **MAJOR · VERIFIED — AT-ADD/OpenBMAT lineage 분리를 보장하지 않음.**  
   `assign_splits.py:25`는 eval을 강제로 newtest에 두면서 train/dev의 동일 group을 제거하지 않음. `open_rows.py:128`은 첫 music track만 사용하고 `art`로 그룹을 분리. `openbmat_extra.py:29`는 basename으로 행을 대응시킴. **실제 누출 건수는 UNKNOWN.**  
   **수정:** 전체 원천 track·speaker·reference 연결을 반영하고 기존 bank까지 중복 검사. 당장은 학습과 연결된 E2 행을 제외해 재평가. **수정 전 E2 CI를 독립 검증 근거로 사용하지 말 것.**

3. **MAJOR · VERIFIED — DB의 FILE 규칙과 P1 기본 동작이 다름.**  
   `new_eval.py:61`은 **0/9/1 고정**. `train_backbone_heads.py:548`은 재탐색하며 `:549`의 `--file-weights`가 있어야 고정됨. **실제 두 패키지의 설정은 UNKNOWN.**  
   **수정:** 아카이브의 `file_ensemble.json` 확인. 평가 recipe를 유지하려면 `--file-weights 0,9,1`. 다른 가중치면 해당 규칙으로 DB 재평가 후 판단.

4. **MAJOR · VERIFIED — E2는 최종 P1 모델의 held-out이 아님.**  
   `train_backbone_heads.py:371`은 **newtest까지 학습에 추가**. NEW_VAL=0이면 newval도 포함. 이는 선언한 P1 정책이며 자체로 제출 결함은 아님.  
   **수정:** E1/E2 결과를 **DB 모델 평가**로만 표기. 최종 P1의 독립 검사는 새 audit 표본으로 수행. `:376`, `:398`, `:467`, `:527`은 구 경로를 사용하므로 NEW_VAL=1이어도 calibration은 사실상 old val만 사용함. 이 정책을 명시하거나 전체 경로·presence 처리를 함께 수정.

5. **MAJOR · VERIFIED — 조건부 오라벨 경로가 있음. 실제 오라벨 수는 UNKNOWN.**  
   - `open_rows.py:39–46`: 한 component만 real이고 나머지는 unknown이어도 FILE=0 확정. **원본 전체의 real 근거가 없으면 FILE mask=0.**
   - `:79`: HAIM B6의 **20초 crop에 vocal이 있는지 확인 없이** VP/VF=1. **crop vocal 확인 전 두 mask=0.**
   - `:101`: vocal 활성 비율 **30% 미만을 무조건 부재** 처리. **경계 표본은 unknown.**
   - `:185–192`: donor RMS=0이면 vocal을 더하지 않고도 VP=1 반환. **생성 실패로 제외.**
   
   오라벨 확인 시 영향받은 행 수정·제외 후 해당 heads 재학습. 코드만으로 두 후보 전체 폐기를 결정할 근거는 없음.

6. **MAJOR · VERIFIED — 평가 누락·오류가 정상 결과처럼 남을 수 있음.**  
   `new_eval.py:42`는 후보별 ID 교집합, `:48`은 presence 누락 행 제외, `:99`는 모든 예외를 `None`으로 처리. 현재 기록의 presence 누락은 **0**.  
   **수정:** 고정 manifest와 네 member·세 seed의 ID/길이/finite 일치 assert. 단일 클래스만 명시적으로 N/A; 다른 오류는 중단. **UNKNOWN:** `F.Boot` 구현이 제공되지 않아 group bootstrap 자체는 확인 불가. `e2_subsets.txt`의 **NaN과 p=0.00은 기각 근거가 아님.**

7. **MAJOR · VERIFIED — W의 목표 stratum 비중은 최종 보장되지 않음.**  
   `weighting.py:108`에서 비중을 맞춘 뒤 `:134`가 class 전체에 cap을 적용해 stratum 비중을 바꿀 수 있음.  
   **수정:** 최종 class×stratum 질량과 duration separability를 보고. **25/45/5/25는 cap 이전 목표**로 명시. 이것만으로 W_new 제출 차단은 불필요.

8. **MINOR · VERIFIED — OOM 재시도 설명이 부정확함.**  
   `dump_features.py:154–162`는 lock 해제 후 다음 행으로 이동하며 자체 재시도하지 않음. `:141–142`도 lock을 남김.  
   **수정:** 제한된 재시도와 `finally` lock 해제, 최종 manifest 대조. 기록상 제외 **8/40,222 = 0.0199%**. 누락 ID 고정 후 후보 간 동일하게 적용.

**VERIFIED:** MERT→EAT→SSLAM 결합 순서와 peak cap·stem rounding에서 새로운 불일치는 찾지 못함 (`dump_features.py:39–59`, `full_bank_submission.py:182–203`). rebuilt tree는 매번 member hash를 확인함 (`verify_s8_vessl.py:23–28`). **UNKNOWN:** CUDA 결과와 L4 전체 시간·메모리 적합성은 아직 증명되지 않음.

