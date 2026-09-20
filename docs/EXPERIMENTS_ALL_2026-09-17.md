# DACON 236749 전체 실험 기록과 방법론

작성 기준: 2026-09-17에 로컬 저장소에서 확인한 실험 보고서, JSON·CSV 결과, 실행 기록, 제출 화면 저장본. 대상은 2026-09-12부터 2026-09-17까지 이 저장소에 남은 오디오 딥페이크 탐지 실험이다. 이번 정리에서는 새 학습·추론·공식 제출·클라우드 상태 조회를 수행하지 않았다. GPU 정지 상태와 공식 점수는 각 원본 기록의 확인 시점에 해당한다.

## 1. 전체 흐름과 현재까지의 결과

공식 제출 기록에서 가장 높은 점수는 **M330_MLP_full1206_submit.zip의 0.7396465503**이다. 베이스라인 0.6909110688보다 0.0487354815 높다. C08의 0.7314822646도 베이스라인을 넘었다. 반면 AST, V5 B1, V6, C3, S2 EAT는 각 제출에서 베이스라인보다 낮았다. 이 비교는 저장된 제출 화면에 근거하며 현재 리더보드를 새로 조회한 결과는 아니다. [E01]

실험은 다음 순서로 이어졌다.

1. **결합 규칙 변경:** 공식 G0의 특징을 재사용해 G1–G4를 비교했으나 채택할 개선을 확인하지 못했다.
2. **AST로 전체 출력 교체:** 개발 지표는 올랐지만 공식 점수는 하락했다. epoch 연장과 전체 공개 자료 재학습도 이 문제를 해결하지 못했다.
3. **음악 경로와 학습 방식 분리:** SONICS 고정 추론, SONICS·AST 전체 미세조정, 전문가별 미세조정과 채널 증강을 비교했다.
4. **고정 표현 위의 작은 분류기:** MERT의 특징을 재사용하면서 데이터 가중치, 구간 처리, 배치, 증강, Conv1D 크기를 바꿨다. C08에서 공식 베이스라인을 넘었다.
5. **표현·분류기 비교:** MERT330+MLP의 로컬 개선과 공식 점수 상승을 확인했다. cosine, label smoothing, scheduler, attention, prototype, AF-Whisper는 조건별 효과가 달랐다.
6. **전체 미세조정·음원 분리 적응:** S2 EAT는 로컬 특정 분할에서 좋았지만 해당 모델로 식별된 공식 제출 점수는 낮았다.
7. **raw/stem과 File 결합 재설계:** raw+stem, 직접 File head, noisy-OR, OOF 관계 모델을 비교했다. 최신 B 전체 학습 패키지는 검증을 마쳤지만 공식 점수와 독립 평가 성능은 미확인이다.

실험에서는 **개발 Music EER의 개선이 공식 총점 개선을 보장하지 않는 문제**가 반복됐다. 실제 음악의 독립 원본이 적고, NSynth와 자연 음악의 분포가 다르며, 음악 속 음성 라벨이 비어 있다. 재사용한 검증셋과 전체 데이터 재학습의 결과를 구분해야 한다.

## 2. 지표를 읽는 방법

이 문서의 EER·AUC는 별도 표시가 없으면 0–1 값이다. EER은 낮을수록 좋고, AUC와 공식 총점은 높을수록 좋다. 예를 들어 EER 0.30→0.25는 5%p 감소다.

저장된 공식 평가 계약과 프로젝트 구현의 총점은 다음과 같다. [E02]

```text
Score = 0.45 × (1 − File EER)
      + 0.18 × (1 − Voice EER)
      + 0.27 × (1 − Music EER)
      + 0.05 × Voice Presence AUC
      + 0.05 × Music Presence AUC
      = 0.9 × ADS + 0.1 × CPS
```

FAKE가 양성이다. EER은 ROC 전체 지점에서 |FPR−FNR|이 가장 작은 지점을 고르고 두 값의 평균으로 계산한다. 성분별 EER 대상은 모델이 예측한 존재 여부가 아니라 **정답에서 해당 성분이 존재하는 파일**이다.

| 표기 | 의미 | 해석 제한 |
|---|---|---|
| 공식 점수 | 저장된 DACON 제출 화면의 채점 결과 | 로컬 EER·proxy와 직접 치환하지 않는다 |
| 부분 라벨 proxy | 계산 가능한 항목만 가중·정규화한 개발 선택 지표 | 구간마다 유효 항목이 다르면 구간 간 크기도 직접 비교하지 않는다 |
| known-subset EER/AUC | 정답을 아는 파일만 계산한 값 | 미상 라벨을 실제·부재로 채우지 않는다 |
| 시드 평균±표준편차 | 각 모델을 따로 평가한 지표의 평균과 모집단 표준편차(ddof=0) | 확률 앙상블의 지표와 다르다 |
| 확률 앙상블 | 파일별 확률을 먼저 평균한 뒤 EER 계산 | 개별 EER의 평균으로 대체하지 않는다 |
| OOF | 해당 원본 그룹을 학습하지 않은 fold 모델의 예측 | 결합기 학습이나 선택에 썼다면 독립 최종 평가가 아니다 |
| historical test | 과거부터 여러 번 확인한 공개 평가 자료 | 새 모델의 학습에서 제외해도 연구 과정 전체의 새 blind test는 아니다 |
| null / 미측정 | 필요한 라벨·실행 결과가 없거나 계산 불가 | 0점·실패율 0·성능 없음으로 해석하지 않는다 |

음악 확률에 엄격히 단조인 변환을 적용하면 그 출력의 순위 기반 EER은 유지된다. 다만 음성 확률과 다시 결합한 File 점수의 순위는 바뀔 수 있다. 단일 임계값을 바꾸는 작업과 탐지 순위를 개선하는 작업도 구분한다.

## 3. 공식 제출 결과

아래 표는 2026-09-17에 저장된 인증된 제출 페이지 DOM을 기준으로 정리했다. 같은 베이스라인의 재제출도 따로 표시했다. `submit.zip`의 S2 식별은 사용자 확인에 따른 것이며 제출 화면의 파일명만으로 모델을 식별한 것은 아니다. [E01]

| 제출 일시(화면 표시) | 파일 / 모델 | 공식 점수 | 베이스라인 대비 | 상태 |
|---|---|---:|---:|---|
| 09-13 00:35:37 | baseline_submit.zip / G0, 제출 88418 | 0.6909110688 | 기준 | 채점 완료 |
| 09-13 11:25:23 | DACON_236749_AST_epoch17.zip, 제출 88630 | 0.6453405503 | −0.0455705185 | 채점 완료 |
| 09-13 12:48:06 | DACON_Sumbit_1.zip / AST full1029, 제출 88673 | 0.6174819841 | −0.0734290847 | 채점 완료 |
| 09-13 17:25:28 | baseline_submit.zip 재제출 | 0.6909110688 | 0 | 채점 완료 |
| 09-14 11:17:33 | V5_B1.zip | 미측정 | — | 제출 오류 |
| 09-14 15:48:35 | V5_B1_fixed.zip | 0.6776485873 | −0.0132624815 | 채점 완료 |
| 09-15 07:22:59 | V6_final_submission.zip, 제출 89989 | 미측정 | — | 제출 오류 |
| 09-15 08:42:35 | V6_final_submission_fixed.zip | 0.6727822646 | −0.0181288042 | 채점 완료 |
| 09-15 14:06:12 | C3_full_clean_submission.zip, 제출 90217 | 0.6843536931 | −0.0065573757 | 채점 완료 |
| 09-16 02:34:32 | C08_submit.zip | 0.7314822646 | +0.0405711958 | 채점 완료 |
| 09-16 07:05:24 | M330_MLP_full1206_submit.zip | **0.7396465503** | **+0.0487354815** | 저장된 기록 내 최고 |
| 09-16 23:06:28 | submit.zip / 사용자 식별 S2 EAT | 0.6378679788 | −0.0530430900 | 채점 완료 |
| 미확인 | B_full1696/submit.zip | 미확인 | — | 전체 학습·패키지 검증 완료 |

점수 하락과 실행 오류는 다르다. AST의 두 제출은 각각 81초·82초에 **정상 채점된 결과**다. G0의 1,319초보다 빨랐다는 사실은 학습이 누락됐다는 증거가 아니다. G0의 여러 모델·분리 경로를 AST 하나로 바꿨으므로 추론 작업량도 달라졌다. [E06]

| 제출 | ADS | CPS | 서버 실행 시간 |
|---|---:|---:|---:|
| G0 88418 | 0.6577539683 | 0.9893249735 | 21분 59초 |
| AST epoch17 88630 | 0.6212619048 | 0.8620483598 | 1분 21초 |
| AST full1029 88673 | 0.5881111111 | 0.8818198413 | 1분 22초 |
| C3 90217 | 0.6504603175 | 0.9893940741 | 21분 24초 |

이 네 행의 상세 근거는 제출 감사와 C3의 사용자 제공 화면 기록에 있다. 최신 전체 표의 점수는 이후 저장된 제출 DOM에서도 확인된다. 내부 File·Voice·Music EER은 ADS만으로 역산할 수 없다. [E06] [E14]

## 4. 데이터의 변화와 평가 범위

| 단계 | 자료와 분할 | 핵심 제한 |
|---|---|---|
| 최초 G0–G4 | 전체 225개, dev 160 / holdout 65 | holdout 원자 그룹 21개, mixture 부모 연결 시 17개 component |
| V2 AST | 1,029개: train 633 / dev 122 / generator test 176 / source test 98 | 평가 274개는 후속 실험에서 재사용 |
| V4 기본 풀 | train 1,516 / dev 210 / sealed 210 = 1,936행 | 고유 PCM 1,896개; dev·sealed 각각 splice 중복 20개 |
| V5 비교 | train 1,496 / dev 230 | NSynth 20개를 개발로 이동; dev 고유 음원 210개 |
| V6 | 음악 학습 823 / File 결합 학습 459 / dev 210 | 중복 제거 dev 190개에서 지표 재계산 |
| temporal·C08·MERT 비교 | train 1,008 / tune 248 / historical test 190 | test Music EER 대상 150개; 여러 연구에서 반복 확인 |
| C3·M330 최종 학습 | 음악 1,206 / C3 결합기 1,886 | 기존 train/dev/sealed 사용; 독립 최종 holdout 없음 |
| S0 fullFT | 음악 train 918, 활동성 필터 후 916 / tune 248 / historical test 190 | test 음악 지표 150개 |
| S1·S2 fullFT | 필터 전 train 474 / tune 156 / test 190 | S1/S2 test 동일; real 12 / fake 178, S0 test와 다름 |
| fast H100 | 원본 1,327/187/172 + 혼합 320/40/40 | 최종 train/val/test 1,647/227/212; 과거 원본 풀 재분할 |
| redesign A/B | train 1,469 / val 129 / sealed 98 | val 원본 연결 그룹 12개; sealed 미개봉 |
| B full1696 | 원본 1,696개, 음악 라벨 유효 1,216개 | 이전 val·sealed까지 학습에 포함; 음성 480개 음악 라벨 마스킹 |

V2는 MLAAD-tiny/M-AILABS 음성 짝, Echoes TTA·FMA 음악, NSynth 악기 대조군, 혼합본을 사용했다. 후속 음악 풀도 FMA 실제 음악, Echoes 생성 음악, NSynth 실제 악기 비중이 크다. NSynth의 짧은 악기 음을 자연 노래와 동일한 실제 음악 분포로 취급하면 평가가 왜곡될 수 있다. [E04] [E12]

### 4.1 데이터 감사로 확인한 수치

- V4 1,936개 파일 감사에서 decode/hash, 분할 간 정확히 같은 PCM, 기록된 계보, 라벨 논리의 위반은 발견되지 않았다. 이는 모든 라벨의 청감 검수나 알려지지 않은 같은 가수·재인코딩 근접 중복의 부재를 입증하지 않는다.
- dev 210→190개 중복 제거만으로 V6 File EER은 **0.385714→0.335897**, Music EER은 **0.482857→0.466667**로 달라졌다. 모델 개선이 아니라 표본 가중 변화다.
- V4 train 음악은 FMA real 70, NSynth real 100, Echoes fake 946개다. V6 음악 학습 823개는 52/84/687개이며 실제 클래스 중 NSynth가 61.8%다.
- 원본 길이 중앙값은 FMA 230.87초, NSynth 4초, Echoes 77.39초다. 저수준 특징 7개만 쓴 group-OOF AUC는 0.878307이었고, NSynth를 뺀 뒤 **길이만 사용해도 OOF AUC 0.893899**였다. 순수 dev에서는 각각 0.545와 0.5로 떨어졌다. 데이터의 지름길 학습 위험을 보여 주지만 MERT가 그 특징을 실제로 사용했다는 증거는 아니다.
- V4 train 음성 진위 unknown은 1,116/1,516, 음성 존재 unknown은 1,016/1,516이다. dev는 각각 110/210, 90/210이다.
- 80% 초과 저활동 구간으로 표시된 학습 파일은 12개(NSynth 10, Echoes 2)였다. 무음 제거만으로 일반화 문제가 해결된다는 근거는 없다.

근거: 데이터 감사 원본·요약과 후속 로컬 감사. [E12] [E30]

## 5. 초기 G0–G4 결합 규칙 비교

공통 특징을 한 번 추출한 뒤 그래프별 결합만 바꿨다. G0는 PANNs 존재 확률, HTDemucs 분리, DF-Arena 1B 음성·음악 탐지와 `max(Pv×Sv, Pm×Sm)`을 사용한다. G1은 상위 두 구간 평균, G2는 시간 정렬과 성분 합집합, G3는 원본·분리 신호 결합, G4는 학습형 결합이다. G4의 파일·음성·음악 head는 각각 dev 160/128/68개로 학습했다. [E03]

| 후보 | File EER, 65개 | Voice EER, known 48개 | Music EER, 29개 |
|---|---:|---:|---:|
| G0 | 0.123476 | 0.041958 | 0.307895 |
| G1 | 0.123476 | 0.083916 | 0.307895 |
| G2 | 0.123476 | 0.041958 | 0.307895 |
| G3 | 0.156504 | 0.041958 | 0.384211 |
| G4 | 0.156504 | 0.041958 | 0.307895 |

모든 후보의 Music Presence AUC는 1.0이며 Voice Presence AUC는 계산 불가였다. G0를 유지했다. A100 노트북 전체 614.35초, 비교 코드 251.06초. 공통 G0와 공식 함수의 실제 혼합 음원 예측 차이는 최대 4.0435765846780214e-11이었다.

후속 진단에서는 G1 Voice EER이 나빠져도 AUC가 0.980769→0.984266으로 오를 수 있음을 확인했다. G3 Music EER이 악화됐을 때도 AUC는 0.721053으로 같았다. 작은 평가셋에서는 EER 작동점과 AUC를 함께 봐야 한다. [E04]

## 6. AST 고정·부분 미세조정·epoch 연장·전체 재학습

### 6.1 V2 비교와 hybrid

AudioSet AST에 다섯 출력을 연결했다. AST_FROZEN은 head 3,845개 파라미터만 8 epoch 학습했고, AST_LAST_BLOCK은 마지막 encoder block·LayerNorm·head 7,093,253개를 6 epoch 학습했다. 미상 라벨과 부재 성분의 진위 loss는 마스킹했다. 실제 head/backbone gradient와 가중치 변화를 기록했다. [E04]

| 후보 | dev 부분 라벨 proxy | 원래 선택 |
|---|---:|---|
| G0 | 0.757797 | 유지 |
| AST_FROZEN | 0.822717 | Voice EER·대응 AUC gate 탈락 |
| AST_LAST_BLOCK | 0.865720 | Voice EER·대응 AUC gate 탈락 |
| 추가 탐색 G0_AST_MUSIC | 0.884715 | 별도 hybrid 절차의 선택; 원래 G0 lock을 대체하지 않음 |

hybrid는 파일·음성 진위를 G0에서, 음악 진위와 두 존재 출력을 AST에서 가져왔다. primary script가 test를 연 뒤 추가된 탐색이므로 처음부터 사전 등록된 세 후보와 동일한 검증 수준으로 볼 수 없다.

| 평가 구간 | G0 Music EER | hybrid Music EER | G0 proxy | hybrid proxy |
|---|---:|---:|---:|---:|
| combined 274 | 0.492867 | 0.371024 | 0.759036 | 0.792205 |
| generator 176 | 0.322479 | 0.394958 | 0.776938 | 0.756339 |
| source 98 | 0.653673 | 0.173163 | 0.742915 | 0.873509 |

combined File/Voice EER은 두 방식 모두 0.204530/0.086557이다. source 구간 개선과 새 생성기 구간 악화가 함께 나타났다. 전체 평균만으로 생성기 일반화를 주장할 수 없다.

### 6.2 epoch 6→17 연장

모델·AdamW·난수 상태를 복원하고 epoch/patience만 바꿨다. head LR 1e-4, backbone LR 1e-5, batch 8×accumulation 2, weight decay 0.01, clip 1.0. 최대 30 epoch 중 22에서 종료하고 17을 선택했다. [E05]

| 항목 | epoch 6 | 선택 epoch 17 |
|---|---:|---:|
| 학습 loss | 0.092527 | 0.017512 |
| dev proxy | 0.865720 | 0.903801 |
| historical 274 AST proxy | 0.774420 | 0.775809 |
| historical File EER | 0.233355 | 0.230269 |
| historical Voice EER | 0.113300 | 0.113300 |
| historical Music EER | 0.371024 | 0.371024 |
| historical hybrid proxy | 0.792205 | 0.792205 |

epoch 22 loss는 0.007588까지 내려갔지만 dev proxy는 더 오르지 않았다. 274개 각각의 확률과 가중치는 실제로 바뀌었다. EER이 같다는 사실은 학습 미실행을 뜻하지 않는다. 추가 16 epoch 학습·개발 평가 합 123.476초, 첫 셀부터 해제 요청까지 345.715초였다.

### 6.3 AST full1029

사전학습 AST부터 다시 시작해 공개 자료 1,029개 전체를 head warmup 8 epoch, 마지막 block 미세조정 17 epoch 학습했다. 기존 epoch17 checkpoint의 이어 학습이 아니다. 파일별 참여는 warmup 8회, fine-tuning 17회이며 누적 8,232/17,493회다. 실제 실행은 Mac MPS였고 Colab 노트북은 재현용이다. [E07]

최종 warmup loss 0.183903136, fine-tuning loss 0.176974248→0.013916772. 학습·패키징·검증 36.96분. 동결 tensor 181개는 사전학습 값과 같고 학습 대상의 갱신을 확인했다. 전체 재학습 후 독립 평가셋은 없으며, 공식 점수는 0.6174819841이었다.

## 7. Recovery V3와 V4 전체 미세조정

### 7.1 SONICS 고정 추론 Recovery V3

음성·존재 출력은 G0로 고정하고 음악만 SONICS raw/stem 또는 G0와 50:50 blend로 바꿨다. 16kHz, 5초 구간, SONICS 추가 학습 없음. dev 122개에서 선택하고 재사용 test는 탈락 확인용으로 사용했다. [E06]

| 후보 | dev proxy | File EER | Music EER |
|---|---:|---:|---:|
| G0 | 0.757797 | 0.155711 | 0.603030 |
| SONICS_RAW | 0.694016 | 0.262123 | 0.661905 |
| SONICS_STEM | 0.717860 | 0.262123 | 0.573593 |
| BLEND_RAW | 0.728218 | 0.221444 | 0.603030 |
| BLEND_STEM | 0.743560 | 0.205011 | 0.573593 |

후보 네 개 모두 탈락했다. 실제 음악 fake 확률을 낮추는 동시에 Echoes 합성 음악도 놓쳤다. threshold 0.5의 진단에서 Echoes 검출은 G0 17/21, SONICS raw 3/21, stem 2/21이었다. 이 threshold는 재선택에 쓰지 않았다. 기존 자료 396개와 새 자료 88개를 처리했지만 탈락한 후보를 새 자료에서 다시 고르지 않았다.

### 7.2 V4 P1–P4

P1은 SONICS 전체 encoder+classifier, P2는 AST 전체 backbone+음악 head, P3는 SONICS 전체 미세조정+codec/noise 증강이다. P4는 선택 parent를 이어 학습하며 temporal bag loss와 partial-edit를 함께 적용한 실험이다. P4의 변화는 pooling 단독 효과가 아니다. music-eligible train 1,116행을 사용했다. [E08]

공통 설계는 AdamW, weight decay 0.01, clip 1.0, seed 20260913, seen/OOD 개발 평가다. P1/P3 encoder LR 후보 1e-5/3e-5/1e-4, P2는 1e-6/3e-6/1e-5였으며 probe와 본 학습을 구분했다. 미상 라벨은 loss에서 제외했다. codec/noise 실험은 identity, MP3/AAC, resampling, additive noise를 사용했다. 원래 SONICS의 MixUp·spectrogram masking은 통제했다.

| 후보 | 선택 epoch | File EER | Music EER | seen D | OOD D | gate |
|---|---:|---:|---:|---:|---:|---|
| P1 | 5 | 0.342857 | 0.341429 | 0.109500 | −0.045000 | 탈락 |
| P2 | 4 | 0.289286 | 0.370714 | 0.177000 | −0.045000 | 탈락 |
| P3 | 5 | 0.353571 | 0.358571 | 0.045000 | −0.117000 | 탈락 |
| P4 | 2 | 0.367857 | 0.446429 | 0.109500 | −0.045000 | 탈락 |

epoch는 `min(seen D, OOD D)`를 최대화하도록 선택했다. 모든 후보가 OOD gain 기준 0.005에 미달했다. G0 유지, train+dev 1,286행 refit 미실행, sealed 미개봉이다.

첫 P1/P3의 초기 RNG 일치 증거가 부족해 별도 matched 재검증을 했다. 두 arm 모두 같은 초기 weights·RNG hash, seed 20260913, 10 epoch, 350 update, 11,160 window exposure를 완료했다.

| matched arm | 선택 epoch | File EER | Music EER | seen D | OOD D | 실행 초 |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 6 | 0.371429 | 0.387857 | 0.067500 | −0.045000 | 127.962 |
| P3 | 8 | 0.335714 | 0.341429 | 0.049500 | −0.045000 | 383.178 |

선택 epoch끼리는 노출량이 다르다. 같은 epoch 10에서 P3−P1은 File EER 0, Music EER +0.012143, seen D +0.037500, OOD D 0이었다. 한 seed에서 전반적 증강 우위는 확인되지 않았다.

## 8. V5 전문가 미세조정과 B1/B2 최종 패키지

B0는 공식 분리 기반 G0다. B1은 **원음**에서 DF-Arena 음성, SONICS 음악, PANNs 존재 모델을 전체 미세조정한다. B2는 같은 구조에 gain, low-pass/전화 음질, noise, MP3 replay 계열 채널 증강을 추가한다. 따라서 B0→B1은 미세조정 여부 외에도 분리·음악 모델이 함께 달라진다. [E09]

| 후보 | File EER | Voice EER | Music EER | Voice Presence AUC | Music Presence AUC |
|---|---:|---:|---:|---:|---:|
| B0 | 0.378175 | 0.150000 | 0.542222 | 0.997917 | 0.985132 |
| B1 | 0.321825 | 0.170833 | 0.368333 | 0.998333 | 0.985855 |
| B2 | 0.378175 | 0.150000 | 0.457778 | 0.969583 | 0.593553 |

표는 dev 230행 기준이다. 고유 음원 210개만 계산한 File EER은 B0/B1/B2 0.337981/0.310096/0.375962다. B1의 개선 순서는 같지만 크기는 달라진다. Voice EER은 known 100개, Voice Presence AUC는 known 140개만 사용했다. 라벨 누락으로 공식 총점은 null이다.

| 학습 작업 | epoch당 파일 | 완료 epoch | update | 선택 epoch |
|---|---:|---:|---:|---:|
| B1 voice | 400 | 5 | 250 | 4 |
| B1 music | 1,096 | 8 | 552 | 5 |
| B1 presence | 1,496 | 9 | 846 | 5 |
| B2 voice | 400 | 6 | 300 | 4 |
| B2 music | 1,096 | 5 | 345 | 1 |
| B2 presence | 1,496 | 4 | 376 | 1 |

목표 60 epoch와 달리 시간 예산으로 종료했다. B2의 Music Presence AUC 하락을 증강 단독 원인으로 확정할 수 없다. 전문가별 실제 학습량과 선택 시점이 다르다. 230개 추론 합계는 B1 71.71초, B2 72.15초이며 로드·설치·학습은 제외했다.

B3는 B2 raw/stem 특징·logit·차이를 temporal Transformer로 결합하는 독립 sigmoid 출력, B4는 voice×music의 absent/real/fake 9-state joint softmax 후 주변확률 출력으로 설계했다. 최초 비교에서는 전처리 시간 초과로 B3/B4 학습을 시작하지 못했다. 후속 최종 풀은 1,936행/1,896 PCM을 사용했고 B1/B2 전문가 6개를 완료했으나 B3/B4 완료 결과는 확보되지 않았다. 최종 B1/B2 ZIP은 별도로 만들어 보존했으며 이 최종 풀의 성능을 앞의 dev 비교값으로 대신하지 않는다. `B4_joint`의 end-to-end 설계도 완료된 cached-feature 결과와 구분한다. [E10]

## 9. V6·V7 전처리와 C3

### 9.1 V6

PANNs 존재, HTDemucs, XLS-R300M+AASIST LA 음성, frozen MERT95+768차원 Linear 음악, 파일 logistic 결합 구조다. 원본 DF-Arena 보조 점수와 시간 정렬 증거·불일치를 결합기에 사용한다. 음악은 구간 최대 logit의 파일 BCE, seed 41, 300 epoch, LR 0.003, weight decay 0.01이다. 음성 LA 모델은 고정 ONNX이며 새 전체 미세조정 모델이 아니다. 실제 결과 위치는 L40S 실행 폴더다. [E11]

dev 210개 File/Music EER은 0.385714/0.482857, 중복 제거 190개는 0.335897/0.466667이다. 알려진 음성 표본의 Voice EER은 0.429167이었다. 공식 fixed 제출 점수는 0.6727822646.

### 9.2 V7 데이터·구간 처리 비교

같은 unique dev 190개, Music EER 150개를 사용했다. 인코더는 고정하고 분류층과 결합기만 학습했다. [E13]

| 조건 | File EER | Music EER |
|---|---:|---:|
| V6_saved / C0_class_balance | 0.335897 | 0.466667 |
| C1_nsynth_5pct | 0.316026 | 0.466667 |
| C2_no_nsynth | 0.351923 | 0.413889 |
| C3_equal_windows | 0.316026 | 0.313889 |
| 같은 패널 공식 baseline | 0.432051 | 0.552778 |
| C4_four_seconds seed41 | 0.396154 | 0.347222 |
| C4_four_seconds seed42 | 0.383974 | 0.386111 |
| C4_four_seconds seed43 | 0.367949 | 0.352778 |
| C5_four_seconds_no_nsynth seed41 | 0.316026 | 0.352778 |
| C5_four_seconds_no_nsynth seed42 | 0.316026 | 0.352778 |
| C5_four_seconds_no_nsynth seed43 | 0.316026 | 0.319444 |

C3는 FMA 45%·NSynth 5%·fake 50%, fake 생성기 균등 가중, 파일당 같은 정규화 비중, 매 epoch 파일당 가청 구간 하나의 무작위 학습을 사용했다. 추론은 max를 유지했다. V6 대비 Music EER 15.28%p, File EER 1.99%p 감소. 다만 C0→C1은 NSynth와 생성기 가중치를, C1→C3는 정규화와 학습 구간 집계를 함께 바꿨으므로 한 요소의 인과 효과로 해석하지 않는다.

C4/C5는 정확한 4초로 다시 추출했다. 기존 4.0375초와 길이 외에 Demucs 실행·crop 위치·pool도 달라졌다. 길이 0.0375초만의 효과가 아니다.

### 9.3 C3 full-data와 early stopping 후속 구현

음악 1,206개, 결합기 1,886개, 25,879 windows를 사용했다. 음악 head는 300 epoch, File 결합은 원본 부모 그룹 5-fold OOF로 학습했다. 실제 패키지와 reference의 실음원 4개 예측 차이는 0이었다. 전체 공개 split을 사용해 독립 최종 성능은 없으며 공식 제출 90217은 0.6843536931이었다. [E14]

이후 구현한 early stopping은 balanced validation BCE, patience 20, min_delta 1e-4, 최대 300, 최저 loss 가중치 복원이다. 외부 OOF 대상은 내부 epoch 선택에서 제외하고, 전체 재학습 epoch는 내부 선택값의 중앙값을 사용하도록 구성했다. **이미 제출한 C3는 이 후속 변경 이전의 300 epoch 모델**이다. 구현·연결 검증과 새 GPU 학습 결과를 혼동하지 않는다. [E15]

## 10. temporal Conv1D 실험 계열

공통 풀은 train 1,008, tune 248, historical test 190개다. train은 Echoes 857/FMA 61/NSynth 90이며 원본 연결 그룹 86개다. test Music EER은 150개. FMA와 Echoes는 같은 61개 원본 음악 그룹에 연결된다. 1,008개를 독립 원곡 수로 해석하지 않는다. [E16]

최초 head는 frozen MERT95의 768차원 구간 평균 특징에 128채널 투영, dilation 1/2/4 residual block(Conv1D 6층), gated attention+max, Linear를 붙인 **414,017 파라미터** 구조다. 인코더는 학습하지 않았다. AdamW와 최대 300/patience 20/min_delta 1e-4를 사용했다. 각 fit의 checkpoint는 tune balanced BCE로, 후보는 tune Music EER 기준으로 골랐다.

### 10.1 배치와 증강

| 단계 | 선택·조건 | tune Music EER | historical Music EER | historical File EER |
|---|---|---:|---:|---:|
| Linear control, seed41 | 769 파라미터 | 0.3627 | 0.366667 | 0.351923 |
| batch16 | T7, seed41 | 0.349891 | 0.547222 | 0.380128 |
| batch16 T7, 3 seed | LR 1e-4/dropout .4/WD .05 | — | 0.566667±0.023679 | 부록 |
| batch128 T0, seed41 | LR 3e-4/dropout .2/WD .01 | 0.303012 | 0.466667 | 0.367949 |
| A0_none, batch128 3 seed | 증강 없음 | 0.380765 | 0.517593±0.082037 | 부록 |
| A1_crop, 3 seed | crop 확률 .7 | 0.308221 | 0.433333±0.0386 | 약 0.3280 |
| A2_crop_mask, 3 seed | crop .7 + mask .3 | 0.311014 | 0.419444±0.0272 | 부록 |

batch16과 batch128에서는 LR 3e-4/1e-4 × dropout .2/.4 × WD .01/.05의 8개 조합을 비교했다. batch128은 모든 설정에서 tune EER이 낮아졌지만 평가 시드 편차가 커졌다. T0의 선택 epoch는 2→4로 늘어도 update는 126→32로 줄었다. epoch만으로 학습량을 비교하면 안 된다. [E17] [E18]

crop은 가능한 3·4·6·8·12개의 연속 구간을 고르며 유효 음악을 남겼다. mask는 768차원 중 38개를 가렸다. 파형 혼합이나 MERT 재추출은 하지 않았다. tune으로 선택한 조건은 A1이다. test에서 A2 평균이 낮다는 이유로 선택을 뒤집지 않았다. [E19]

### 10.2 과적합 진단

T7 학습 BCE는 epoch 1→21에서 0.7723→0.1592로 감소했지만 tune BCE는 0.7548→3.2021로 증가했다. T0 마지막 학습 BCE 0.0000277, tune BCE 4.2607. class/source weighting이 다른 train/tune loss의 절대 차이를 일반화 오차로 계산하지 않았다. [E16]

batch16에서 FMA가 하나도 없는 완전한 배치 확률은 36.548%, batch64 1.609%, batch128 0.019%였다. epoch당 update는 63/16/8. 가중치 총량 변동계수는 .412/.202/.135였다. 이는 표본 구성·가중치 변동이며 실제 gradient 분산 측정은 아니다.

학습에는 mixture/splice가 없지만 tune은 100/248개, 평가 음악은 100/150개가 혼합·편집 음원이다. T7 입력 순서를 뒤집은 tune EER은 0.3599, 파일 평균 특징으로 대체하면 0.3471, 원래는 0.3499였다. 이 checkpoint에서 시간 순서의 이득이 뚜렷하지 않았지만 재학습한 구조 비교는 아니다.

### 10.3 구조 단순화와 compact C08

| 구조 / 3 seed | 파라미터 | tune Music EER | historical Music EER | historical File EER |
|---|---:|---:|---:|---:|
| 기존 6층 + attention/max, A1 | 414,017 | 0.308221 | 0.433333 | 약 0.3280 |
| 128채널 2층 + max | 199,169 | 0.300596 | 0.373148 | 0.322650 |
| 64채널 3층 C00 | 88,257 | 0.313429 | 0.288889 | 0.310684 |
| 64채널 3층 tune 선택 C08 | 88,257 | 0.275484 | 0.295370 | 0.321368 |

첫 단순화는 dilation 2/4 block과 attention을 제거했다. 다음 compact는 64채널 3층으로 바꿨다. 층 수·폭·pooling을 단계별로 함께 바꾼 부분이 있어 효과를 완전히 분리하지는 못한다. [E20] [E21]

compact에서는 C00과 고정 난수 23조합, 상위 C08/C05/C21/C19의 추가 시드, C00·S128 대조군을 합쳐 37회 fit했다. 전체 grid 전수 탐색은 아니다. C08 설정은 LR 1e-4, batch64, WD0, dropout0, crop .7, mask .3, NSynth mass0이다. 학습 ID에 NSynth가 있어도 직접 loss 기여는 0이다. 선택 epoch는 12/15/9. C00보다 tune은 좋지만 historical Music/File EER은 높았다.

C08 제출은 **seed41 epoch12 단일 모델**이며 3-seed 평균이나 전체 데이터 재학습 모델이 아니다. 해당 seed의 historical Music/File EER은 0.313889/0.316026. 공식 점수는 0.7314822646이었다. [E22]

## 11. 시간 해상도와 sigmoid/softmax

동일 MERT 출력에서 4.0375초 window당 평균 벡터 1개(R1) 또는 시간 bin 8개(R8)를 추출했다. 8개 벡터는 맥락을 공유하는 MERT 특징이며 독립적인 0.5초 오디오 추론 8회가 아니다. sigmoid+BCE와 2-logit softmax+CE를 함께 비교했다. C08 설정과 seed41/42/43을 유지했다. [E23]

| 구성 | tune Music EER 평균 | historical Music EER 평균 | historical File EER 평균 |
|---|---:|---:|---:|
| R1 sigmoid | 0.276415 | 0.311111 | 0.326709 |
| R1 softmax | 0.303012 | 0.337963 | 0.322650 |
| R8 sigmoid | 0.276415 | 0.309259 | 0.326709 |
| R8 softmax | 0.286832 | 0.300000 | 0.310684 |
| 이전 캐시 C08 | 0.275484 | 0.295370 | 0.321368 |

원 실행은 tune 평균의 5.55e-17 차이 때문에 R8 sigmoid를 선택했다. 12자리 반올림 후 명시된 표준편차 동점 규칙을 적용하면 **R1 sigmoid**가 맞다(std .011637 대 .021471). 선택 오류를 감사하고 수정했으며 원본 기록은 보존했다. historical 점수를 이용한 후보 변경이 아니다.

특징 추출 2,296.85초, head 12회 학습 합 257.54초. 해상도를 높이면 같은 Conv1D의 초 단위 수용영역도 바뀌므로 해상도만의 순수 효과로 해석하지 않는다.

## 12. MERT95→MERT330과 MLP

고정 Demucs·MERT 뒤에 공통 64채널 3-convolution residual trunk, max pooling, MLP 64→32→2, GELU, dropout .1, softmax를 사용했다. AdamW LR1e-4/WD .01, batch64, crop .7, mask .3, NSynth mass0, seed41/42/43. 두 encoder 모두 정규화된 같은 head 설정으로 새로 학습했다. [E24]

| 구성 | tune Music EER 평균 | historical Music EER 평균 | historical File EER 평균 |
|---|---:|---:|---:|
| MERT95 + MLP | 0.271206 | 0.297222 | 0.321368 |
| MERT330 + MLP | **0.255580** | **0.220370** | **0.266667** |

M330의 Music EER은 seed41/42/43 각각 0.213889/0.233333/0.213889, 모집단 SD 0.009166. 선택 epoch는 14/19/17, M95는 12/15/14였다. tune으로 M330을 선택했다. 입력은 768→1024차원이며 인코더는 동결했다. M95 특징은 기존 H100 캐시, M330 추출과 여섯 head 학습은 A100이다. 새 특징 1,446개 추출 4,973.93초, 여섯 fit 83.16초.

### 12.1 M330 전체 자료 5-fold 제출

승인된 음악 1,206개를 source-group 5-fold로 나눴다. 각 파일은 네 모델의 학습과 한 모델의 검증에 참여하며 NSynth 90개에도 양수 가중치(mass .05)를 줬다. fold seed41–45는 같은 split의 시드 반복과 다르다. 선택 epoch 36/43/40/42/28, 실제 종료 56/63/60/62/48이다. fold별 train/val은 991/215, 930/276, 973/233, 969/237, 961/245. [E25]

다섯 음악 확률을 평균하고 기존 File max 결합을 적용했다. 인코더는 파일당 한 번 실행한다. 전체 공개 자료가 fold 학습에 들어가 독립 최종 holdout 성능은 없다. 공식 점수 0.7396465503은 이 최종 패키지의 제출 기록이고, 앞의 0.220370은 이전 고정 split에서 개별 세 seed의 평균이다.

## 13. cosine·label smoothing·scheduler·분류기 16설정

### 13.1 첫 cosine 묶음 실험

MLP를 L2 정규화 특징·클래스 대표 벡터의 cosine×10으로 바꾸고 LS .05와 ReduceLROnPlateau를 동시에 추가했다. unsmoothed validation CE, factor .5, patience4(연속 5회 정체 뒤 감소), min LR1e-6, early stopping20을 사용했다. [E26]

| 구성, 3-seed 평균 | tune Music EER | historical Music EER | historical File EER |
|---|---:|---:|---:|
| M95 MLP | 0.271206 | 0.297222 | 0.321368 |
| M95 cosine+LS+Plateau | 0.299665 | 0.360185 | 0.316026 |
| M330 MLP | 0.255580 | 0.220370 | 0.266667 |
| M330 cosine+LS+Plateau | 0.285901 | 0.235185 | 0.277350 |

M330의 Music/File EER이 모두 높아졌다. 세 요소를 한꺼번에 바꿨으므로 원인을 하나로 특정하지 않았다.

### 13.2 16설정 × 3 seed = 48 fit

MERT330 특징과 공통 trunk를 고정했다. MLP·cosine은 classifier×LS(0/.05)×Plateau(없음/있음) 요인을 비교했고, prototype·attention·Conv mean/LME는 기본과 LS+Plateau 묶음을 비교했다. 모든 fit은 24–55 epoch에서 종료했다. [E27]

| 설정 | tune Music EER 평균±SD | historical Music EER 평균±SD | historical File EER 평균±SD |
|---|---:|---:|---:|
| Cosine + LS .05 | .23140±.05525 | .22407±.02179 | .27735±.00937 |
| Attention + MLP | .23233±.01164 | .21296±.00917 | .26795±.01308 |
| Attention + MLP + LS + Plateau | .23233±.01164 | .21296±.00917 | .26261±.00755 |
| MLP + LS .05 | .24182±.02401 | .21759±.00262 | .25726±.00755 |
| MLP | .25558±.01342 | .22037±.00917 | .26667±.01155 |
| MLP + Plateau | .25558±.01342 | .22037±.00917 | .26667±.01155 |
| MLP + LS + Plateau | .25558±.01702 | .22222±.00818 | .25726±.00755 |
| Conv1d + mean | .27121±.01342 | .21759±.02734 | .28397±.00000 |
| Conv1d + mean + LS + Plateau | .27883±.00811 | .21944±.02722 | .28932±.00755 |
| Cosine | .27976±.02984 | .23981±.01982 | .28141±.01155 |
| Cosine + Plateau | .27976±.02984 | .24444±.01571 | .27735±.01330 |
| Conv1d + LME + LS + Plateau | .28069±.01342 | .22685±.01982 | .28397±.00000 |
| Conv1d + LME | .28404±.01851 | .22870±.02179 | .28397±.00000 |
| Cosine + LS + Plateau | .28590±.01412 | .23519±.01367 | .27735±.00683 |
| 클래스당 3 prototype | .29967±.02683 | .26019±.07551 | .30534±.03298 |
| prototype + LS + Plateau | .31436±.01412 | .28889±.08016 | .31325±.02853 |

tune 선택은 cosine+LS, Plateau 없음이다. historical Music 최저는 attention .21296, File 최저는 MLP+LS 계열 .25726이다. 서로 다른 기준의 최고를 하나의 우승 모델로 합쳐 쓰지 않는다. MLP에 Plateau만 추가한 경우 실제 LR 감소가 있어도 선택 가중치와 성능은 같았다.

### 13.3 학습 없는 확률 변환

| MLP 저장 확률 변환 | Music EER 평균 | File EER 평균 | 음악 balanced BCE 평균 |
|---|---:|---:|---:|
| 없음 | .22037 | .26667 | .52023 |
| temperature .5 | .22037 | .27201 | .63615 |
| temperature 2 | .22037 | .26261 | .54659 |
| p′=.95p+.025 | .22037 | .26261 | .51846 |

test에 맞춰 temperature를 학습하거나 선택하지 않았다. 학습 target의 label smoothing과 추론 확률 smoothing은 별개의 방법이다.

## 14. AF-Whisper637 frozen encoder 비교

Audio Flamingo 3의 sound tower만 사용했고 LLM은 로드하지 않았다. frozen encoder 실제 추론 파라미터는 MERT330 315,428,992→AF 636,968,960, 입력 1024→1280, head는 107,234→124,130개다. 공통 compact trunk+MLP, LR1e-4/WD .01/dropout .1/batch64, scheduler 없음. LS0/.05×3 seed의 신규 여섯 fit과 기존 MERT 대조군 여섯 개를 비교했다. [E28]

| 구성 | tune Music EER 평균±SD | historical Music EER 평균±SD | historical File EER 평균±SD |
|---|---:|---:|---:|
| M330 MLP | .2556±.0134 | .2204±.0092 | .2667±.0116 |
| M330 MLP + LS .05 | .2418±.0240 | .2176±.0026 | .2573±.0076 |
| AF637 MLP | .2323±.0208 | .2574±.0065 | .2774±.0094 |
| AF637 MLP + LS .05 | .2323±.0208 | .2620±.0065 | .2786±.0076 |

tune 동률에서 BCE로 AF637+LS를 골랐지만 historical Music/File은 M330보다 악화됐다. AF 선택 epoch는 두 LS 조건 모두 11/12/14, 종료 31/32/34. 특징 추출 5,234.27초, 신규 여섯 fit·tune 예측 약 86.80초. 크기 외에 사전학습·frontend·구조도 달라졌으므로 파라미터 수만의 효과는 아니다.

## 15. H100 전체 fine-tuning 8개와 HTDemucs 적응

seed41, 5초 clips, 학습 4/eval 8 windows, 최대20/patience3, encoder LR1e-5/head LR1e-4, WD .01, warmup5%. weighted sampling+CE, tune Music EER→balanced BCE로 checkpoint를 선택했다. S1은 MusicGen, S2는 AudioLDM을 train/tune에서 제외했다. S1/S2 test는 같은 190개이며 S0와 다르다. [E29]

| 실험 | 선택 epoch | tune Music EER | test Music EER | 제외 생성기+real EER |
|---|---:|---:|---:|---:|
| S0 MERT frozen | 5 | .246093 | .266667 | — |
| S0 MERT fullFT | 3 | .230467 | .319444 | — |
| S0 EAT fullFT | 1 | .217634 | .300000 | — |
| S1 MERT fullFT | 8 | .100694 | .248596 | .270833 |
| S1 EAT fullFT | 6 | .006944 | .248596 | .250000 |
| S2 MERT fullFT | 2 | .097222 | .332397 | .333333 |
| S2 EAT fullFT | 4 | .013889 | .167603 | .166667 |
| S0 adapted HTDemucs + EAT fullFT | 6 | .170755 | .252778 | — |

S0에서는 fullFT가 frozen MERT보다 좋지 않았다. S2 EAT는 같은 S1/S2 test에서 낮은 EER을 보였지만 real 음악은 12개뿐이다. tune .013889와 test .167603의 차이도 남았다. S2 checkpoint를 그대로 패키징했으며 8:1:1 재학습 모델이 아니다. 사용자 식별 공식 제출 점수 .6378679788은 로컬 Music EER 개선이 공식 총점으로 이어지지 않은 사례다.

05/06/07은 batch4×accum2, 나머지는 batch1×accum8이다. update당 8곡은 같지만 난수·연산 순서까지 동일하지는 않다.

HTDemucs 적응은 고정 교사의 train stem 네 개를 서로 다른 그룹에서 재혼합해 학생 전체를 400 step 학습했다. 41,984,456개 파라미터의 비영 gradient와 update L2 약 .993을 확인했다. 선택 step250의 pseudo-validation L1은 .011324429186061025. 이는 실제 정답 stem 정확도가 아니라 교사와의 일치다. 같은 S0 EAT에서 test Music EER .300000→.252778, 4.72%p 개선이지만 한 seed·재사용 test에 한정한다.

## 16. fast H100 raw/stem·직접 File head 비교

기존 1,686개 원본 풀을 group 8:1:1로 다시 나누고 split 안에서만 새 혼합 320/40/40개를 만들었다. frozen M330 raw/stem 음악 head 여섯 개와 File MLP 세 개, 총 아홉 fit을 완료했다. [E31]

학습 범위에 차이가 있다. **음악 head는 새 mixture를 학습하지 않았다.** 음악 학습은 자연 음악 943개(FMA70/NSynth73/Echoes800), checkpoint validation은 자연 음악 139개다. 직접 File head만 전체 train1,647개와 mixture320개를 사용했다. File 입력은 raw1024+stem1024+고정 음성/존재3 특징이다. [E30]

| 3-seed 확률 앙상블 | validation File | test File | validation Music | test Music |
|---|---:|---:|---:|---:|
| stem + max | .078462 | .110063 | .133165 | .206564 |
| raw + max | .135879 | .110063 | .211646 | .210564 |
| stem music + 직접 File head | .156923 | .150943 | .133165 | .206564 |

validation으로 raw blend weight0을 선택했다. .25와 EER이 같았지만 사전 규칙으로 더 작은 비중을 골랐다. 선택된 dual은 stem과 같다. test Music EER의 **시드별 평균**은 stem .221778±.021515, raw .190017±.034169로, 위 **앙상블** 순위와 다르다.

| test 부분집합 | stem Music EER | raw Music EER |
|---|---:|---:|
| 원본만 | .114536 | .159900 |
| NSynth 제외 원본 음악 | .148214 | .248810 |
| 새 혼합 | .250000 | .400000 |

직접 File head는 혼합 test File EER .200000→.116667, 원본 test .093023→.162791로 바뀌었다. 혼합 validation은 .100000→.300000으로 나빠졌다. 유리한 test 부분집합만 보고 채택할 근거는 없다.

고정 DF-Arena의 전처리 경로 비교 24개에서 음성 확률 최대 차이 .061789를 확인했다. 모두 16kHz mono여서 stereo 정보 손실을 검증한 실험은 아니며, 정확도나 S2 공식 점수 하락의 원인도 확정하지 못했다. 특징 추출 약59분9초, 음악 학습·예측 약72초, File head·평가 약8초다.

## 17. redesign A/B·noisy-OR·OOF 관계 모델

train1,469/val129/sealed98, frozen MERT330·HTDemucs·DF-Arena·PANNs 조건이다. A는 stem 단독, B는 shared temporal encoder와 gated raw+stem residual이다. AdamW LR1e-4/WD .01, batch64, max300/patience20. class·연결 그룹 가중 validation loss `0.27×bag CE + 0.10×window BCE`로 checkpoint를 고른 **같은 validation**에서 아래 성능도 측정했다. [E32]

| 확률 앙상블 | Music EER | File EER |
|---|---:|---:|
| A stem + max | .081155 | .109477 |
| B raw+stem + max | .081155 | .109477 |
| B + noisy-OR | .081155 | .081155 |
| B + OOF relation | .081155 | .104575 |

File noisy-OR는 `1−(1−Pv×Sv)(1−Pm×Sm)`이다. 이는 두 성분 증거의 결합이며 V4에서 검토한 길이 의존 window noisy-OR와 구분한다. File 관계 모델은 train의 5-fold grouped OOF로 학습했다. 본 학습 선택 epoch는 A23/15/14, B17/17/8. 본 학습 여섯 개와 OOF30개 checkpoint를 보존했다.

| arm | seed41 Music EER | seed42 | seed43 | 평균±SD |
|---|---:|---:|---:|---:|
| A | .076253 | .104575 | .147603 | .109477±.029334 |
| B | .109477 | .104575 | .109477 | .107843±.002311 |

| 비교 | 계산 가능한 가중 기여 차이 | 2,000회 group bootstrap 95% 구간 |
|---|---:|---:|
| A→B, max | .000000 | [−.050560, .029771] |
| B max→noisy-OR | +.012745 | [−.003798, .028066] |
| B max→OOF relation | +.002206 | [−.024021, .054495] |

모두 구간이 0을 포함하고 필수 subgroup도 없어 `diagnostic_only_not_accepted`다. 사후 저장 예측 비교에서 A+noisy-OR File EER은 .109477로 같고 B만 .081155였다. 이 상호작용의 bootstrap 구간도 [−.043559, .037099]로 0을 포함한다. raw 분기의 일반적 이득이나 noisy-OR 채택을 확정한 결과가 아니다.

### 17.1 NSynth 제외 민감도와 시간 정렬 사후 진단

129개 val은 독립 연결 그룹 12개, FMA 실제곡7/NSynth20/Echoes102다. NSynth를 빼면 109파일/7그룹만 남는다. [E33]

| 저장된 A/B validation 예측 | 전체 File EER | NSynth 제외 File EER |
|---|---:|---:|
| A + max | .1095 | .1450 |
| B + max | .1095 | .1695 |
| B + noisy-OR | .0812 | .2507 |
| B + OOF relation | .1046 | .1352 |
| B + 시간 정렬 음성 증거 + noisy-OR | .0812 | .1499 |

시간 정렬 진단은 `max(Pv_t)×max(Sv_t)`를 `max(Pv_t×Sv_t)`로 바꾸고 나머지 출력은 유지했다. `Empower Thee`의 음성 증거 .316329→.091538에도 음악 증거 .683626이 남았고, `Wir Werden Gott`는 음성 .749264→.698860, 음악 .095231이었다. 서로 다른 오류 경로가 있다. File 점수를 낮추기만 하는 변경은 실제곡 오탐을 줄이면서 가짜를 놓칠 수 있어 가짜 음성·혼합 대조가 필요하다.

이 표는 **이전 A/B validation 예측의 사후 분석**이다. B full1696 독립 평가로 부르지 않는다.

## 18. B full1696 최종 학습과 패키지

이전 A/B 진단에서 정한 B의 17/17/8 epoch를 seed41/42/43에 고정해 전체 원본1,696개로 재학습했다. 음악 라벨 유효1,216개가 음악 loss에 참여했고 음성 전용480개의 음악 라벨은 unknown으로 마스킹했다. 네 대형 전문가와 encoder는 동결했으며 작은 B head만 학습했다. File은 noisy-OR, 세 음악 확률의 앙상블이다. [E34]

이전 train1,469+val129+sealed98을 모두 포함했으므로 독립 validation은 없다. 공식 점수는 null이다. 제출 파일 `submissions/B_full1696/submit.zip`은 6,267,071,302바이트이며 SHA-256은 다음과 같다.

```text
1fda295b44925d5b255cc1235f9bee57141ace7469370ae3245af113572bfb00
```

실제 원본5개+4초 무음+60초 stereo music의 H100 오프라인 패키지 추론을 reference와 비교하고 파일 순서 변경·부분집합 독립성을 확인했다. 두 최대 절대오차는 4.869993297518249e-11, 허용 오차1e-5. 총120 windows, 추론35.419초/새 subprocess 포함38.884초, GPU allocated peak20,652,236,800바이트. 이는 패키지 재현성 측정이며 새 품질 점수나 L4 전체 실행 검증은 아니다.

첫 시도는 학습 프로세스의 GPU allocator cache가 남아 OOM이 났고 해제 후 성공했다. ZIP 전체와50개 member hash, 결과 export22개를 검증한 뒤 `wsp-9683432r72m7`의 standby를 확인한 기록이 있다. 5분 monitor는 PAUSED다. 이번 문서 작성에서 원격 현재 상태를 다시 조회하지는 않았다.

## 19. 실행 실패·배포 검증에서 얻은 결과

| 사건 | 확인한 원인·현상 | 조치와 확인 범위 |
|---|---|---|
| AST 패키지 수치 불일치 | 기본 cuDNN TF32에서 최대 차이 2.2172927856445312e-05, 기준1e-5 초과 | 두 TF32를 끈 뒤 차이0.0; weight·허용치 유지 |
| AST full1029 MPS | 부분 LayerNorm backward 오류 | CPU 대조 후 수정; 동결 backbone 유지; 패키지 차이3.5762786865234375e-07 |
| V5 B1 설치 오류 | torch/torchaudio/torchvision ABI 불일치 경로 | torch2.7.1+cu128/torchaudio2.7.1+cu128/torchvision0.22.1+cu128 함께 고정; 54개 비-requirements member 불변 |
| V5 최종 temporal 진행 | softmax 불변 pool_score.bias의 zero gradient를 지나치게 엄격하게 거부 | 전문가 완료와 temporal 미완료를 분리 기록; 완료되지 않은 B3/B4 성능 없음 |
| V6 제출89989 | NumPy dtype ABI 오류, import 단계2초 | numpy2.2.6+pandas1.5.3 조합에서 재현, pandas2.2.3으로 import 복구; 채점 서버의 정확한 pandas 버전은 미확인 |
| temporal 단순화 첫 시도 | 순전파가 같아도 gradient 합산 순서 변경으로 학습 재현성 불일치 | 기존 연산 순서 복원 후 이전 H100 곡선 재현 |
| resolution 최초 추출 | TorchAudio2.9 decoder 부재 | FFmpeg+TorchCodec0.8.1 설치, 실제 decoder parity 확인 후 원래 deadline 안에 복구 |
| resolution 후보 선택 | 동률 EER의 5.55e-17 반올림 차이가 tie-break를 우회 | 12자리 반올림+SD 규칙으로 R1 sigmoid 정정 |
| MERT330 H100 시도 | Jupyter/SSH timeout | 학습 명령 미실행, 결과 없음; 이후 A100 별도 실행과 구분 |
| M330 full 패키지 | A100 music parity 1.91e-7, CPU는 historical A100 대비 약 .00694 차이 | CPU 반복은 동일; 기기·torch/librosa/numpy 차이의 정확한 원인은 미분리 |
| fast H100 자동 종료 | 다운로드 prefix 문제로 즉시 정지 경로 실패 | 올바른 경로로 직접 다운로드·검증 후 정지 기록 |
| B full 첫 acceptance | 학습 allocator cache 잔류 OOM | GPU 메모리 해제 후 실제 패키지 검증 성공 |

배포 검증은 ZIP/CRC/hash, import, 실제 오디오 추론, reference parity, 파일 독립성, 공식 서버 전체 실행을 구분했다. 구조 검사나 CPU 한 파일 성공만으로 L4 전체 제한 준수를 주장하지 않는다. [E04] [E07] [E10] [E22] [E25] [E34] [E35]

## 20. 자원·시간 기록

시간의 포함 범위가 달라 직접 속도 순위로 사용하지 않는다. 잔액 차이는 준비·대기·검증까지 포함할 수 있으며 총 실험 비용의 완전한 장부가 아니다. 완료 기록에 정지가 있는 실행과 단순 준비 기록을 구분했다.

| 실행 | 측정 시간과 범위 | 기록된 크레딧 차이 |
|---|---|---:|
| G0–G4 A100 | 전체 notebook614.35초 / 비교251.06초 | Colab 표시56.68→55.96 |
| V2 최초+정밀도 재검증 | 첫 셀→해제 요청 합23.60분, GPU 할당 약24.23분 | Colab55.96→54.49 |
| AST epoch 연장 | 학습·dev123.476초 / 전체345.715초 | Colab54.49→54.36 |
| AST full1029 | MPS 학습·패키징·검증36.96분 | GPU cloud 비용으로 환산 안 함 |
| V7 A100 | 약1시간, 정확한 GPU 총 runtime 미확보 | 1.415478 |
| C3 final A100 | 실행·검증 기록 보존 | 1.183908 |
| temporal batch16 H100 | notebook 약125.2초 | .569113 |
| temporal batch128 H100 | notebook237.9초 | .343557 |
| augmentation H100 | notebook169.7초 | .325298 |
| simplification H100 | 성공 notebook100.2초; 첫 실패 별도 | .459979 |
| compact H100 | 37fit, 준비·재시도·다운로드 포함 잔액 차이 | .799042 |
| resolution H100 | 추출2296.85초 + 12fit257.54초 | 이 표에서 미집계 |
| MERT330 A100 | 추출4973.93초 + 6fit83.16초 | 2.336719 |
| AF637 A100 | 추출5234.27초 + 6fit·tune86.80초 | 이 표에서 미집계 |
| fullFT8 H100 | 분류8개+분리기400step; export37,601,239,040바이트 | 이 표에서 미집계 |
| fast H100 | 특징약59분9초; 생성→정지확인약71분 | 실제 청구 미확정 |
| redesign A/B H100 | 생성→정지확인1시간34분42초 | 실제 청구 미확정 |
| B full H100 acceptance | 7파일 subprocess38.884초 | 학습 전체 시간과 다름 |

캐시를 쓴 작은 head 학습은 수분 이내인 반면 새 encoder의 특징 추출은 수십 분 걸렸다. 이는 기록된 시간 범위에서의 관찰이며 H100/A100의 순수 하드웨어 우열 비교는 아니다.

## 21. 미실행·취소·준비 단계

| 작업 | 확인한 상태 | 성능 수치 처리 |
|---|---|---|
| V4 train+dev1,286 refit | gate 탈락으로 실행 안 함 | 새 점수 없음 |
| V5 B3/B4 최초 비교 | 공유 전처리 시간 초과 | 미측정 |
| V5 최종 temporal/B4_joint | 전문가와 구분되는 완료 성능 기록 미확보 | 완료로 간주 안 함 |
| MERT330 H100 접속 실패 | 학습 command 미실행 | A100 결과와 중복 집계 안 함 |
| S2 expanded frozen evaluation | 추론 전 취소 | `cancelled_before_inference` |
| S2 group8:1:1 | train630/val88/test90, 그룹49/6/6; AudioLDM 제외110 | 분할만 채택; 잘못 시작한 재학습은 중단·미채택 |
| redesign C/D 혼합 학습 | 검증된 fake accompaniment 부재로 실행 안 함 | A/B에 섞어 쓰지 않음 |
| redesign 음성 적응·raw=stem 용량 대조 | 실행 안 함 | 해당 효과 미확인 |
| fresh music 소규모 probe | FMA12+SONICS12 원본→4/15/60초 총68구간 준비 | 24원본의 파생자료; 새 모델 점수 없음 |
| 후속 six_h100 자료 준비 | 현재 읽은 data_receipt:136부모,823 probe행 | 학습 완료·평가 점수 근거 미확보 |

S2 8:1:1은 원본918개 중 train630(real49/fake581), validation88(6/82), test90(6/84), 제외110을 기록했다. test는 기존 S2 test의 부분집합이며 독립 실제 음악6개뿐이다. 사용자 정정 후 정지한 run을 새 성능 결과로 채택하지 않는다. [E36]

후속 six_h100 자료 기록은 4s136,15s104,60s87,base136,MP3 32kbps136,telephone136,speech_music32,partial_music40,real_voice_fake_music16행으로 총823행이다. 부모 train/dev/test78/29/29, 그룹64/22/22. 이 문서는 해당 시점의 자료 준비만 기록한다. 자료의 `labels_independently_reviewed`는 false다. 새 음성 화자는 네 명뿐이고 dev/test 각 한 명이며 음악 출처의 음성 라벨은 unknown으로 남는다. 다른 작업에서 이후 실행된 결과는 본 문서의 확인 범위에 포함되지 않는다. [E37]

## 22. 실험에서 얻은 방법론과 남은 질문

### 확인된 교훈

1. **선택 기준을 잠근 뒤 평가해야 한다.** AST는 원래 gate에서 탈락했는데 dev proxy 상승만으로 제출했고 공식 점수는 하락했다. A1/A2, C00/C08, cosine/attention처럼 tune 선택과 historical 최고가 다르면 둘 다 기록한다.
2. **분할과 분모를 고정해야 비교가 된다.** 중복 제거만으로 EER이 달라졌고 S0와 S1/S2는 test 구성도 다르다. 원본·아티스트·화자·mixture 부모를 함께 묶고 derived 파일 수를 독립 표본 수로 쓰지 않는다.
3. **작은 분류기를 먼저 비교할 실익이 있었다.** 이번 공통 split에서 414,017→199,169→88,257 파라미터로 단순화하며 개선됐고 C08 공식 점수도 올랐다. 모든 데이터에서 작은 모델이 더 낫다는 일반 법칙은 아니다.
4. **인코더 교체는 matched control이 필요하다.** M95와 M330에 같은 regularized MLP를 새로 학습해 비교했다. AF637은 더 크고 tune이 좋아도 historical test가 악화됐다.
5. **학습량은 완료 epoch·update·노출량으로 보고해야 한다.** 목표60/300과 실제 완료는 다르다. batch나 증강을 바꾸면 epoch의 비용과 update 수도 바뀐다.
6. **음악 head 개선과 File 개선을 함께 봐야 한다.** 공식 File 가중치가 .45로 가장 크다. MLP probability temperature는 Music EER을 유지하면서 File EER을 바꿨다. CPS가 높다고 개별 Presence 오판이 File에 무해한 것도 아니다.
7. **앙상블과 seed 평균은 별도 결과다.** fast H100에서는 두 계산법으로 raw/stem 순위가 뒤집혔다. 가장 좋은 test seed만 고르면 선택 편향이 생긴다.
8. **패키지·의존성·수치 경로도 실험의 일부다.** V5/V6 ABI, TF32, decoder, MPS, GPU cache OOM은 모델 품질과 다른 실패였다. 학습 checkpoint와 실제 ZIP의 예측·해시를 연결해야 한다.

### 아직 확정할 수 없는 것

음악 속 실제/가짜 보컬, 전화 음질, 짧은 위조 구간, 실제/가짜 성분 혼합의 충분한 정답과 독립 원본이 부족하다. full-data 모델은 이전 sealed를 사용했으므로 새 최종 확인셋이 필요하다. 최신 B noisy-OR는 NSynth 포함 여부에 민감하며 bootstrap도 확정적 개선을 지지하지 않았다. raw 분기의 무용성, fullFT의 일반적 우위, 분리기 적응의 실제 stem 품질 개선, 특정 전처리가 공식 점수 하락의 원인이라는 주장은 아직 뒷받침되지 않는다.

후속 계획은 새 원본 그룹과 완전한 성분 라벨을 확보하고, 고정 checkpoint에서 결합식·시간 정렬을 먼저 비교한 뒤, 확인된 오류 경로에만 학습 예산을 배정하는 것이다. 이는 완료된 성과가 아니라 현재 기록에 근거한 후속 검증 방향이다. [E33]

## 23. 부록을 읽는 방법

아래 부록은 저장된 JSON에서 직접 옮긴 후보별 수치다. 소수점은 최대10자리로 표시하며 전체 정밀도, epoch별 loss·LR, 파일별 예측, 해시는 각 원본 링크에 남아 있다. 모든 파생 waveform·매 epoch 로그를 이 문서에 복제하지는 않는다. 후보군·시드별 결과와 설정, 원본을 추적할 위치를 모은 실험 장부다.

`—`는 해당 기록에 값이 없음을 뜻한다. 같은 대조군이 여러 실험에 반복 등장하면 비교 조건을 보존하려고 그대로 기재했다. 부록 행 수는 새로 학습한 모델 수나 독립 실험 수가 아니다. 기존 MERT 대조군 재사용, tuning에 없는 선택 후보의 추가 seed, 진단용 같은 예측의 다른 패널을 구분해야 한다.

## 부록 A. temporal·MERT·분류기 실험의 전체 후보와 seed

각 절의 tune JSON, checkpoint 학습 기록, test 보고서를 연결했다. `test` 열은 모두 재사용한 190개 패널이며 Music EER 분모는 150개다. test 평가를 하지 않은 탐색 후보는 `—`로 남긴다. 최초 batch16/128의 이름에 seed가 없는 T0–T7은 recipe의 seed41을 따른다. AF 실험의 M330 행은 기존 대조군을 재사용했다.

### A1. batch16

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_20260915/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_20260915/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_20260915/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | 41 | 2 | 22 | — | 414017 | 0.4793538985 | 0.7741246391 | — | — |
| T1 | 41 | 2 | 22 | — | 414017 | 0.4793538985 | 0.8749761368 | — | — |
| T2 | 41 | 2 | 22 | — | 414017 | 0.4168488382 | 0.9981868672 | — | — |
| T3 | 41 | 2 | 22 | — | 414017 | 0.3911829002 | 0.9893805497 | — | — |
| T4 | 41 | 1 | 21 | — | 414017 | 0.3627236661 | 0.727493006 | — | — |
| T5 | 41 | 1 | 21 | — | 414017 | 0.3627236661 | 0.7280916837 | — | — |
| T6 | 41 | 1 | 21 | — | 414017 | 0.35993037 | 0.7628644552 | — | — |
| T7 | 41 | 1 | 21 | — | 414017 | 0.3498906971 | 0.7547810927 | 0.5472222222 | 0.3801282051 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3666666667 | 0.3519230769 |
| T7_seed42 | 42 | 2 | 22 | — | 414017 | — | 0.7763515497 | 0.6 | 0.3961538462 |
| T7_seed43 | 43 | 1 | 21 | — | 414017 | — | 0.8239955083 | 0.5527777778 | 0.4 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | 0.0003 | 16 | 0.01 | 0.2 | 0 | 0 | 0.05 |  |
| T1 | 0.0003 | 16 | 0.05 | 0.2 | 0 | 0 | 0.05 |  |
| T2 | 0.0003 | 16 | 0.01 | 0.4 | 0 | 0 | 0.05 |  |
| T3 | 0.0003 | 16 | 0.05 | 0.4 | 0 | 0 | 0.05 |  |
| T4 | 0.0001 | 16 | 0.01 | 0.2 | 0 | 0 | 0.05 |  |
| T5 | 0.0001 | 16 | 0.05 | 0.2 | 0 | 0 | 0.05 |  |
| T6 | 0.0001 | 16 | 0.01 | 0.4 | 0 | 0 | 0.05 |  |
| T7 | 0.0001 | 16 | 0.05 | 0.4 | 0 | 0 | 0.05 |  |

### A2. batch128

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_b128_20260915/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_b128_20260915/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_b128_20260915/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | 41 | 4 | 24 | 192 | 414017 | 0.3030119019 | 0.6570957886 | 0.4666666667 | 0.3679487179 |
| T1 | 41 | 4 | 24 | 192 | 414017 | 0.305805198 | 0.6451871955 | — | — |
| T2 | 41 | 4 | 24 | 192 | 414017 | 0.334264432 | 0.6690421362 | — | — |
| T3 | 41 | 4 | 24 | 192 | 414017 | 0.3443041049 | 0.6795671452 | — | — |
| T4 | 41 | 6 | 26 | 208 | 414017 | 0.3158448709 | 0.6514375756 | — | — |
| T5 | 41 | 6 | 26 | 208 | 414017 | 0.305805198 | 0.651871375 | — | — |
| T6 | 41 | 11 | 31 | 248 | 414017 | 0.3314711359 | 0.6710927504 | — | — |
| T7 | 41 | 11 | 31 | 248 | 414017 | 0.318638167 | 0.6695642557 | — | — |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3666666667 | 0.3519230769 |
| T0_seed42 | 42 | 1 | 21 | 168 | 414017 | — | 0.7081602216 | 0.6333333333 | 0.416025641 |
| T0_seed43 | 43 | 6 | 26 | 208 | 414017 | — | 0.6453942209 | 0.4527777778 | 0.3320512821 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | 0.0003 | 128 | 0.01 | 0.2 | 0 | 0 | 0.05 |  |
| T1 | 0.0003 | 128 | 0.05 | 0.2 | 0 | 0 | 0.05 |  |
| T2 | 0.0003 | 128 | 0.01 | 0.4 | 0 | 0 | 0.05 |  |
| T3 | 0.0003 | 128 | 0.05 | 0.4 | 0 | 0 | 0.05 |  |
| T4 | 0.0001 | 128 | 0.01 | 0.2 | 0 | 0 | 0.05 |  |
| T5 | 0.0001 | 128 | 0.05 | 0.2 | 0 | 0 | 0.05 |  |
| T6 | 0.0001 | 128 | 0.01 | 0.4 | 0 | 0 | 0.05 |  |
| T7 | 0.0001 | 128 | 0.05 | 0.4 | 0 | 0 | 0.05 |  |

### A3. 증강 A0–A2

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_aug_20260915/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_aug_20260915/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_aug_20260915/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_none_seed41 | 41 | 4 | 24 | 192 | 414017 | 0.3030119019 | 0.6570957886 | 0.4666666667 | 0.3679487179 |
| A0_none_seed42 | 42 | 1 | 21 | 168 | 414017 | 0.5362723666 | 0.7081601901 | 0.6333333333 | 0.416025641 |
| A0_none_seed43 | 43 | 6 | 26 | 208 | 414017 | 0.3030119019 | 0.6453941963 | 0.4527777778 | 0.3320512821 |
| A1_crop_seed41 | 41 | 3 | 23 | 184 | 414017 | 0.3030119019 | 0.5659641818 | 0.4333333333 | 0.3320512821 |
| A1_crop_seed42 | 42 | 5 | 25 | 200 | 414017 | 0.318638167 | 0.5588508058 | 0.3861111111 | 0.3358974359 |
| A1_crop_seed43 | 43 | 4 | 24 | 192 | 414017 | 0.3030119019 | 0.5481834483 | 0.4805555556 | 0.316025641 |
| A2_crop_mask_seed41 | 41 | 4 | 24 | 192 | 414017 | 0.292972229 | 0.5698738849 | 0.4194444444 | 0.3358974359 |
| A2_crop_mask_seed42 | 42 | 5 | 25 | 200 | 414017 | 0.321431463 | 0.5655600653 | 0.3861111111 | 0.3320512821 |
| A2_crop_mask_seed43 | 43 | 4 | 24 | 192 | 414017 | 0.318638167 | 0.5578488035 | 0.4527777778 | 0.316025641 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3666666667 | 0.3519230769 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0_none_seed41 | 0.0003 | 128 | 0.01 | 0.2 | 0 | 0 | 0.05 |  |
| A1_crop_seed41 | 0.0003 | 128 | 0.01 | 0.2 | 0.7 | 0 | 0.05 |  |
| A2_crop_mask_seed41 | 0.0003 | 128 | 0.01 | 0.2 | 0.7 | 0.3 | 0.05 |  |

### A4. 단순화

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_simple_20260915/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_simple_20260915/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_simple_20260915/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_deep_crop_seed41 | 41 | 3 | 23 | 184 | 414017 | 0.3030119019 | 0.5659641818 | 0.4333333333 | 0.3320512821 |
| D_deep_crop_seed42 | 42 | 5 | 25 | 200 | 414017 | 0.318638167 | 0.5588508058 | 0.3861111111 | 0.3358974359 |
| D_deep_crop_seed43 | 43 | 4 | 24 | 192 | 414017 | 0.3030119019 | 0.5481834483 | 0.4805555556 | 0.316025641 |
| S_simple_crop_seed41 | 41 | 3 | 23 | 184 | 199169 | 0.2901789329 | 0.576436542 | 0.4194444444 | 0.3519230769 |
| S_simple_crop_seed42 | 42 | 6 | 26 | 208 | 199169 | 0.318638167 | 0.5701571583 | 0.3527777778 | 0.3 |
| S_simple_crop_seed43 | 43 | 9 | 29 | 232 | 199169 | 0.292972229 | 0.5606693848 | 0.3472222222 | 0.316025641 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3666666667 | 0.3519230769 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_deep_crop_seed41 | 0.0003 | 128 | 0.01 | 0.2 | 0.7 | 0 | 0.05 | simplified=False |
| S_simple_crop_seed41 | 0.0003 | 128 | 0.01 | 0.2 | 0.7 | 0 | 0.05 | simplified=True |

### A5. compact C00–C23

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_compact_20260915/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_compact_20260915/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_compact_20260915/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00_seed41 | 41 | 6 | 26 | 208 | 88257 | 0.318638167 | 0.5778319154 | 0.3333333333 | 0.316025641 |
| C01_seed41 | 41 | 12 | 32 | 512 | 88257 | 0.3030119019 | 0.5708488314 | — | — |
| C02_seed41 | 41 | 3 | 23 | 184 | 88257 | 0.3911829002 | 0.6596942444 | — | — |
| C03_seed41 | 41 | 12 | 32 | 512 | 88257 | 0.3627236661 | 0.5882360562 | — | — |
| C04_seed41 | 41 | 88 | 108 | 432 | 88257 | 0.305805198 | 0.5794343546 | — | — |
| C05_seed41 | 41 | 15 | 35 | 280 | 88257 | 0.2873856368 | 0.5755632994 | — | — |
| C06_seed41 | 41 | 5 | 25 | 400 | 88257 | 0.318638167 | 0.6160169323 | — | — |
| C07_seed41 | 41 | 72 | 92 | 368 | 88257 | 0.318638167 | 0.5968585238 | — | — |
| C08_seed41 | 41 | 12 | 32 | 512 | 88257 | 0.2873856368 | 0.5609718889 | 0.3138888889 | 0.316025641 |
| C09_seed41 | 41 | 10 | 30 | 240 | 88257 | 0.321431463 | 0.6023567041 | — | — |
| C10_seed41 | 41 | 28 | 48 | 768 | 88257 | 0.334264432 | 0.5823083758 | — | — |
| C11_seed41 | 41 | 18 | 38 | 304 | 88257 | 0.3314711359 | 0.5829644994 | — | — |
| C12_seed41 | 41 | 5 | 25 | 800 | 88257 | 0.334264432 | 0.6021274504 | — | — |
| C13_seed41 | 41 | 28 | 48 | 1536 | 88257 | 0.318638167 | 0.5969905196 | — | — |
| C14_seed41 | 41 | 34 | 54 | 432 | 88257 | 0.334264432 | 0.6565747787 | — | — |
| C15_seed41 | 41 | 2 | 22 | 704 | 88257 | 0.3627236661 | 0.6381240122 | — | — |
| C16_seed41 | 41 | 51 | 71 | 284 | 88257 | 0.3627236661 | 0.6455718844 | — | — |
| C17_seed41 | 41 | 4 | 24 | 384 | 88257 | 0.318638167 | 0.5386921144 | — | — |
| C18_seed41 | 41 | 2 | 22 | 704 | 88257 | 0.334264432 | 0.6092650281 | — | — |
| C19_seed41 | 41 | 40 | 60 | 240 | 88257 | 0.292972229 | 0.5864405312 | — | — |
| C20_seed41 | 41 | 2 | 22 | 704 | 88257 | 0.347097401 | 0.6071722553 | — | — |
| C21_seed41 | 41 | 7 | 27 | 216 | 88257 | 0.292972229 | 0.5721692135 | — | — |
| C22_seed41 | 41 | 23 | 43 | 172 | 88257 | 0.347097401 | 0.6369162728 | — | — |
| C23_seed41 | 41 | 18 | 38 | 1216 | 88257 | 0.3314711359 | 0.6532138384 | — | — |
| C00_seed42 | 42 | 12 | 32 | 256 | 88257 | 0.2745526678 | 0.5675724715 | 0.2472222222 | 0.3 |
| C00_seed43 | 43 | 11 | 31 | 248 | 88257 | 0.347097401 | 0.5881666308 | 0.2861111111 | 0.316025641 |
| C08_seed42 | 42 | 15 | 35 | 560 | 88257 | 0.2617196988 | 0.5484176151 | 0.2527777778 | 0.316025641 |
| C08_seed43 | 43 | 9 | 29 | 464 | 88257 | 0.2773459639 | 0.5611173175 | 0.3194444444 | 0.3320512821 |
| C05_seed42 | 42 | 4 | 24 | 192 | 88257 | 0.2773459639 | 0.553176605 | — | — |
| C05_seed43 | 43 | 7 | 27 | 216 | 88257 | 0.334264432 | 0.6293883027 | — | — |
| C21_seed42 | 42 | 8 | 28 | 224 | 88257 | 0.2460934337 | 0.549383948 | — | — |
| C21_seed43 | 43 | 9 | 29 | 232 | 88257 | 0.2901789329 | 0.5300326724 | — | — |
| C19_seed42 | 42 | 25 | 45 | 180 | 88257 | 0.2488867298 | 0.5624111438 | — | — |
| C19_seed43 | 43 | 29 | 49 | 196 | 88257 | 0.3158448709 | 0.5516207631 | — | — |
| S128_seed41 | 41 | 3 | 23 | 184 | 199169 | 0.2901789329 | 0.576436542 | 0.4194444444 | 0.3519230769 |
| S128_seed42 | 42 | 6 | 26 | 208 | 199169 | 0.318638167 | 0.5701571583 | 0.3527777778 | 0.3 |
| S128_seed43 | 43 | 9 | 29 | 232 | 199169 | 0.292972229 | 0.5606693848 | 0.3472222222 | 0.316025641 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3666666667 | 0.3519230769 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00_seed41 | 0.0003 | 128 | 0.01 | 0.2 | 0.7 | 0 | 0.05 | simplified=True; compact=True |
| C01_seed41 | 0.0001 | 64 | 0.01 | 0 | 0.7 | 0.15 | 0.025 | simplified=True; compact=True |
| C02_seed41 | 0.001 | 128 | 0.01 | 0.2 | 0 | 0 | 0.1 | simplified=True; compact=True |
| C03_seed41 | 0.0001 | 64 | 0.001 | 0.2 | 0.35 | 0.15 | 0.05 | simplified=True; compact=True |
| C04_seed41 | 0.00003 | 256 | 0.05 | 0.5 | 1 | 0.3 | 0 | simplified=True; compact=True |
| C05_seed41 | 0.001 | 128 | 0.001 | 0.1 | 0.35 | 0 | 0.05 | simplified=True; compact=True |
| C06_seed41 | 0.0003 | 64 | 0.05 | 0.2 | 0.35 | 0 | 0.1 | simplified=True; compact=True |
| C07_seed41 | 0.00003 | 256 | 0 | 0.35 | 0.35 | 0.15 | 0.025 | simplified=True; compact=True |
| C08_seed41 | 0.0001 | 64 | 0 | 0 | 0.7 | 0.3 | 0 | simplified=True; compact=True |
| C09_seed41 | 0.0003 | 128 | 0.01 | 0.1 | 0.35 | 0.3 | 0.05 | simplified=True; compact=True |
| C10_seed41 | 0.00003 | 64 | 0.1 | 0.1 | 0.7 | 0.15 | 0.025 | simplified=True; compact=True |
| C11_seed41 | 0.0001 | 128 | 0.001 | 0.2 | 1 | 0.15 | 0.025 | simplified=True; compact=True |
| C12_seed41 | 0.0003 | 32 | 0.05 | 0 | 1 | 0 | 0.05 | simplified=True; compact=True |
| C13_seed41 | 0.00003 | 32 | 0.1 | 0.5 | 0.7 | 0 | 0 | simplified=True; compact=True |
| C14_seed41 | 0.0001 | 128 | 0.05 | 0.5 | 0 | 0 | 0.1 | simplified=True; compact=True |
| C15_seed41 | 0.0003 | 32 | 0.001 | 0 | 0.7 | 0.3 | 0.1 | simplified=True; compact=True |
| C16_seed41 | 0.00003 | 256 | 0.1 | 0.1 | 0 | 0.3 | 0.1 | simplified=True; compact=True |
| C17_seed41 | 0.001 | 64 | 0 | 0 | 0.7 | 0.3 | 0 | simplified=True; compact=True |
| C18_seed41 | 0.001 | 32 | 0 | 0.2 | 1 | 0 | 0.05 | simplified=True; compact=True |
| C19_seed41 | 0.0001 | 256 | 0.01 | 0.35 | 0.35 | 0.3 | 0.025 | simplified=True; compact=True |
| C20_seed41 | 0.001 | 32 | 0.001 | 0.1 | 0 | 0.15 | 0.025 | simplified=True; compact=True |
| C21_seed41 | 0.0003 | 128 | 0.01 | 0.35 | 1 | 0.15 | 0 | simplified=True; compact=True |
| C22_seed41 | 0.0003 | 256 | 0 | 0.5 | 0 | 0 | 0.05 | simplified=True; compact=True |
| C23_seed41 | 0.00003 | 32 | 0.1 | 0.35 | 0 | 0.15 | 0 | simplified=True; compact=True |
| S128_seed41 | 0.0003 | 128 | 0.01 | 0.2 | 0.7 | 0 | 0.05 | simplified=True; compact=False |

### A6. R1/R8·sigmoid/softmax

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_resolution_20260915/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_resolution_20260915/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_resolution_20260915/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1_sigmoid_seed41 | 41 | 12 | 32 | 512 | 88257 | 0.2773459639 | 0.5577174951 | 0.3333333333 | 0.3320512821 |
| R1_sigmoid_seed42 | 42 | 15 | 35 | 560 | 88257 | 0.2617196988 | 0.5568722648 | 0.2527777778 | 0.316025641 |
| R1_sigmoid_seed43 | 43 | 15 | 35 | 560 | 88257 | 0.2901789329 | 0.5524567386 | 0.3472222222 | 0.3320512821 |
| R1_softmax_seed41 | 41 | 12 | 32 | 512 | 88322 | 0.334264432 | 0.5758205217 | 0.3138888889 | 0.3 |
| R1_softmax_seed42 | 42 | 10 | 30 | 480 | 88322 | 0.2717593717 | 0.5573415424 | 0.3194444444 | 0.3519230769 |
| R1_softmax_seed43 | 43 | 9 | 29 | 464 | 88322 | 0.3030119019 | 0.5499559357 | 0.3805555556 | 0.316025641 |
| R8_sigmoid_seed41 | 41 | 11 | 31 | 496 | 88257 | 0.292972229 | 0.5580211282 | 0.3472222222 | 0.3320512821 |
| R8_sigmoid_seed42 | 42 | 15 | 35 | 560 | 88257 | 0.2460934337 | 0.5223413112 | 0.2805555556 | 0.316025641 |
| R8_sigmoid_seed43 | 43 | 14 | 34 | 544 | 88257 | 0.2901789329 | 0.5209235527 | 0.3 | 0.3320512821 |
| R8_softmax_seed41 | 41 | 12 | 32 | 512 | 88322 | 0.334264432 | 0.5631005523 | 0.3333333333 | 0.3320512821 |
| R8_softmax_seed42 | 42 | 19 | 39 | 624 | 88322 | 0.2332604647 | 0.5100055995 | 0.2805555556 | 0.3 |
| R8_softmax_seed43 | 43 | 14 | 34 | 544 | 88322 | 0.292972229 | 0.5665478221 | 0.2861111111 | 0.3 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3805555556 | 0.3519230769 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1_sigmoid_seed41 | 0.0001 | 64 | 0 | 0 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=False |
| R1_softmax_seed41 | 0.0001 | 64 | 0 | 0 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True |
| R8_sigmoid_seed41 | 0.0001 | 64 | 0 | 0 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=8; softmax_output=False |
| R8_softmax_seed41 | 0.0001 | 64 | 0 | 0 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=8; softmax_output=True |

### A7. M95/M330 MLP

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert330_a100_20260916/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert330_a100_20260916/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert330_a100_20260916/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M95_mlp_seed41 | 41 | 12 | 32 | 512 | 90338 | 0.2617196988 | 0.5505093092 | 0.3194444444 | 0.3320512821 |
| M95_mlp_seed42 | 42 | 15 | 35 | 560 | 90338 | 0.2617196988 | 0.5216010253 | 0.2861111111 | 0.316025641 |
| M95_mlp_seed43 | 43 | 14 | 34 | 544 | 90338 | 0.2901789329 | 0.5440447247 | 0.2861111111 | 0.316025641 |
| M330_mlp_seed41 | 41 | 14 | 34 | 544 | 107234 | 0.2745526678 | 0.5545781044 | 0.2138888889 | 0.2519230769 |
| M330_mlp_seed42 | 42 | 19 | 39 | 624 | 107234 | 0.2460934337 | 0.4950898196 | 0.2333333333 | 0.2801282051 |
| M330_mlp_seed43 | 43 | 17 | 37 | 592 | 107234 | 0.2460934337 | 0.5155753847 | 0.2138888889 | 0.2679487179 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3805555556 | 0.3519230769 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M95_mlp_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=768; mlp_output=True |
| M330_mlp_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True |

### A8. M95/M330 cosine

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/cosine_a100_20260916/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/cosine_a100_20260916/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/cosine_a100_20260916/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M95_cosine_seed41 | 41 | 12 | 32 | 512 | 88320 | 0.318638167 | 0.5746076497 | 0.3333333333 | 0.316025641 |
| M95_cosine_seed42 | 42 | 7 | 27 | 432 | 88320 | 0.2901789329 | 0.5849131262 | 0.3805555556 | 0.316025641 |
| M95_cosine_seed43 | 43 | 9 | 29 | 464 | 88320 | 0.2901789329 | 0.5732926711 | 0.3666666667 | 0.316025641 |
| M330_cosine_seed41 | 41 | 12 | 32 | 512 | 105216 | 0.305805198 | 0.6089903381 | 0.2527777778 | 0.283974359 |
| M330_cosine_seed42 | 42 | 19 | 39 | 624 | 105216 | 0.2745526678 | 0.5332628654 | 0.2194444444 | 0.2801282051 |
| M330_cosine_seed43 | 43 | 15 | 35 | 560 | 105216 | 0.2773459639 | 0.548639552 | 0.2333333333 | 0.2679487179 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3805555556 | 0.3519230769 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M95_cosine_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=768; mlp_output=False; cosine_output=True; label_smoothing=0.05; lr_plateau=True |
| M330_cosine_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=False; cosine_output=True; label_smoothing=0.05; lr_plateau=True |

### A9. M330 분류기 16설정

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_ls0_lr0_seed41 | 41 | 14 | 34 | 544 | 107234 | 0.2745526678 | 0.5545781044 | 0.2138888889 | 0.2519230769 |
| mlp_ls0_lr0_seed42 | 42 | 19 | 39 | 624 | 107234 | 0.2460934337 | 0.4950898196 | 0.2333333333 | 0.2801282051 |
| mlp_ls0_lr0_seed43 | 43 | 17 | 37 | 592 | 107234 | 0.2460934337 | 0.5155753847 | 0.2138888889 | 0.2679487179 |
| mlp_ls0_lr1_seed41 | 41 | 14 | 34 | 544 | 107234 | 0.2745526678 | 0.5545781044 | 0.2138888889 | 0.2519230769 |
| mlp_ls0_lr1_seed42 | 42 | 19 | 39 | 624 | 107234 | 0.2460934337 | 0.4950898196 | 0.2333333333 | 0.2801282051 |
| mlp_ls0_lr1_seed43 | 43 | 17 | 37 | 592 | 107234 | 0.2460934337 | 0.5155753847 | 0.2138888889 | 0.2679487179 |
| mlp_ls5_lr0_seed41 | 41 | 14 | 34 | 544 | 107234 | 0.2745526678 | 0.5529275082 | 0.2138888889 | 0.2519230769 |
| mlp_ls5_lr0_seed42 | 42 | 35 | 55 | 880 | 107234 | 0.2176341997 | 0.476520173 | 0.2194444444 | 0.2679487179 |
| mlp_ls5_lr0_seed43 | 43 | 17 | 37 | 592 | 107234 | 0.2332604647 | 0.51717098 | 0.2194444444 | 0.2519230769 |
| mlp_ls5_lr1_seed41 | 41 | 14 | 34 | 544 | 107234 | 0.2745526678 | 0.5529275082 | 0.2138888889 | 0.2519230769 |
| mlp_ls5_lr1_seed42 | 42 | 19 | 39 | 624 | 107234 | 0.2589264027 | 0.4974943979 | 0.2333333333 | 0.2679487179 |
| mlp_ls5_lr1_seed43 | 43 | 17 | 37 | 592 | 107234 | 0.2332604647 | 0.51717098 | 0.2194444444 | 0.2519230769 |
| cosine_ls0_lr0_seed41 | 41 | 12 | 32 | 512 | 105216 | 0.318638167 | 0.6327856751 | 0.2666666667 | 0.2961538462 |
| cosine_ls0_lr0_seed42 | 42 | 15 | 35 | 560 | 105216 | 0.2460934337 | 0.5787130161 | 0.2333333333 | 0.2679487179 |
| cosine_ls0_lr0_seed43 | 43 | 9 | 29 | 464 | 105216 | 0.2745526678 | 0.5579120932 | 0.2194444444 | 0.2801282051 |
| cosine_ls0_lr1_seed41 | 41 | 12 | 32 | 512 | 105216 | 0.318638167 | 0.6327856751 | 0.2666666667 | 0.2961538462 |
| cosine_ls0_lr1_seed42 | 42 | 15 | 35 | 560 | 105216 | 0.2460934337 | 0.5787130161 | 0.2333333333 | 0.2679487179 |
| cosine_ls0_lr1_seed43 | 43 | 15 | 35 | 560 | 105216 | 0.2745526678 | 0.5609906074 | 0.2333333333 | 0.2679487179 |
| cosine_ls5_lr0_seed41 | 41 | 12 | 32 | 512 | 105216 | 0.305805198 | 0.6089903381 | 0.2527777778 | 0.283974359 |
| cosine_ls5_lr0_seed42 | 42 | 34 | 54 | 864 | 105216 | 0.1735487005 | 0.455745206 | 0.2 | 0.283974359 |
| cosine_ls5_lr0_seed43 | 43 | 31 | 51 | 816 | 105216 | 0.2148409036 | 0.5251992051 | 0.2194444444 | 0.2641025641 |
| cosine_ls5_lr1_seed41 | 41 | 12 | 32 | 512 | 105216 | 0.305805198 | 0.6089903381 | 0.2527777778 | 0.283974359 |
| cosine_ls5_lr1_seed42 | 42 | 19 | 39 | 624 | 105216 | 0.2745526678 | 0.5332628654 | 0.2194444444 | 0.2801282051 |
| cosine_ls5_lr1_seed43 | 43 | 15 | 35 | 560 | 105216 | 0.2773459639 | 0.548639552 | 0.2333333333 | 0.2679487179 |
| prototype_reg0_seed41 | 41 | 12 | 32 | 512 | 105472 | 0.2617196988 | 0.5307471052 | 0.2 | 0.2801282051 |
| prototype_reg0_seed42 | 42 | 6 | 26 | 416 | 105472 | 0.318638167 | 0.6364121734 | 0.3666666667 | 0.3519230769 |
| prototype_reg0_seed43 | 43 | 9 | 29 | 464 | 105472 | 0.318638167 | 0.622854796 | 0.2138888889 | 0.283974359 |
| prototype_reg1_seed41 | 41 | 4 | 24 | 384 | 105472 | 0.305805198 | 0.6097478257 | 0.4 | 0.3519230769 |
| prototype_reg1_seed42 | 42 | 15 | 35 | 560 | 105472 | 0.3030119019 | 0.6310033892 | 0.2527777778 | 0.3038461538 |
| prototype_reg1_seed43 | 43 | 9 | 29 | 464 | 105472 | 0.334264432 | 0.6021462127 | 0.2138888889 | 0.283974359 |
| attention_reg0_seed41 | 41 | 12 | 32 | 512 | 107299 | 0.2332604647 | 0.5299127132 | 0.2194444444 | 0.2679487179 |
| attention_reg0_seed42 | 42 | 15 | 35 | 560 | 107299 | 0.2460934337 | 0.5231347587 | 0.2 | 0.283974359 |
| attention_reg0_seed43 | 43 | 17 | 37 | 592 | 107299 | 0.2176341997 | 0.5294979855 | 0.2194444444 | 0.2519230769 |
| attention_reg1_seed41 | 41 | 12 | 32 | 512 | 107299 | 0.2332604647 | 0.5287053724 | 0.2194444444 | 0.2679487179 |
| attention_reg1_seed42 | 42 | 15 | 35 | 560 | 107299 | 0.2460934337 | 0.5223003292 | 0.2 | 0.2679487179 |
| attention_reg1_seed43 | 43 | 17 | 37 | 592 | 107299 | 0.2176341997 | 0.5204538283 | 0.2194444444 | 0.2519230769 |
| conv_mean_reg0_seed41 | 41 | 12 | 32 | 512 | 105218 | 0.2901789329 | 0.550677976 | 0.1861111111 | 0.283974359 |
| conv_mean_reg0_seed42 | 42 | 19 | 39 | 624 | 105218 | 0.2617196988 | 0.5505292292 | 0.2527777778 | 0.283974359 |
| conv_mean_reg0_seed43 | 43 | 11 | 31 | 496 | 105218 | 0.2617196988 | 0.5609240438 | 0.2138888889 | 0.283974359 |
| conv_mean_reg1_seed41 | 41 | 12 | 32 | 512 | 105218 | 0.2901789329 | 0.5429287568 | 0.1861111111 | 0.283974359 |
| conv_mean_reg1_seed42 | 42 | 19 | 39 | 624 | 105218 | 0.2717593717 | 0.5335322856 | 0.2527777778 | 0.283974359 |
| conv_mean_reg1_seed43 | 43 | 11 | 31 | 496 | 105218 | 0.2745526678 | 0.556872683 | 0.2194444444 | 0.3 |
| conv_lme_reg0_seed41 | 41 | 12 | 32 | 512 | 105218 | 0.3030119019 | 0.5505713702 | 0.2 | 0.283974359 |
| conv_lme_reg0_seed42 | 42 | 19 | 39 | 624 | 105218 | 0.2589264027 | 0.5378465472 | 0.2527777778 | 0.283974359 |
| conv_lme_reg0_seed43 | 43 | 11 | 31 | 496 | 105218 | 0.2901789329 | 0.5632620637 | 0.2333333333 | 0.283974359 |
| conv_lme_reg1_seed41 | 41 | 12 | 32 | 512 | 105218 | 0.2901789329 | 0.5442043755 | 0.2 | 0.283974359 |
| conv_lme_reg1_seed42 | 42 | 19 | 39 | 624 | 105218 | 0.2617196988 | 0.5293423181 | 0.2472222222 | 0.283974359 |
| conv_lme_reg1_seed43 | 43 | 11 | 31 | 496 | 105218 | 0.2901789329 | 0.5628274837 | 0.2333333333 | 0.283974359 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3805555556 | 0.3519230769 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_ls0_lr0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.0; lr_plateau=False; alternative_output= |
| mlp_ls0_lr1_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.0; lr_plateau=True; alternative_output= |
| mlp_ls5_lr0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.05; lr_plateau=False; alternative_output= |
| mlp_ls5_lr1_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.05; lr_plateau=True; alternative_output= |
| cosine_ls0_lr0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=False; cosine_output=True; label_smoothing=0.0; lr_plateau=False; alternative_output= |
| cosine_ls0_lr1_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=False; cosine_output=True; label_smoothing=0.0; lr_plateau=True; alternative_output= |
| cosine_ls5_lr0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=False; cosine_output=True; label_smoothing=0.05; lr_plateau=False; alternative_output= |
| cosine_ls5_lr1_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=False; cosine_output=True; label_smoothing=0.05; lr_plateau=True; alternative_output= |
| prototype_reg0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.0; lr_plateau=False; alternative_output=prototype |
| prototype_reg1_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.05; lr_plateau=True; alternative_output=prototype |
| attention_reg0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.0; lr_plateau=False; alternative_output=attention |
| attention_reg1_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.05; lr_plateau=True; alternative_output=attention |
| conv_mean_reg0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.0; lr_plateau=False; alternative_output=conv_mean |
| conv_mean_reg1_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.05; lr_plateau=True; alternative_output=conv_mean |
| conv_lme_reg0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.0; lr_plateau=False; alternative_output=conv_lme |
| conv_lme_reg1_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.05; lr_plateau=True; alternative_output=conv_lme |

### A10. AF637 대조

근거: [tuning.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/afwhisper_a100_20260916/unpacked/results/tuning.json>) · [test_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/afwhisper_a100_20260916/unpacked/results/test_report.json>) · [selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/afwhisper_a100_20260916/unpacked/results/selection.json>)

| 후보 | seed | 선택 epoch | 완료 epoch | 총 update | 학습 파라미터 | tune Music EER | tune balanced loss | test Music EER | test File EER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M330_mlp_ls0_seed41 | 41 | 14 | 34 | 544 | 107234 | 0.2745526678 | 0.5545781044 | 0.2138888889 | 0.2519230769 |
| M330_mlp_ls0_seed42 | 42 | 19 | 39 | 624 | 107234 | 0.2460934337 | 0.4950898196 | 0.2333333333 | 0.2801282051 |
| M330_mlp_ls0_seed43 | 43 | 17 | 37 | 592 | 107234 | 0.2460934337 | 0.5155753847 | 0.2138888889 | 0.2679487179 |
| M330_mlp_ls5_seed41 | 41 | 14 | 34 | 544 | 107234 | 0.2745526678 | 0.5529275082 | 0.2138888889 | 0.2519230769 |
| M330_mlp_ls5_seed42 | 42 | 35 | 55 | 880 | 107234 | 0.2176341997 | 0.476520173 | 0.2194444444 | 0.2679487179 |
| M330_mlp_ls5_seed43 | 43 | 17 | 37 | 592 | 107234 | 0.2332604647 | 0.51717098 | 0.2194444444 | 0.2519230769 |
| AF637_mlp_ls0_seed41 | 41 | 11 | 31 | 496 | 124130 | 0.2617196988 | 0.51230095 | 0.2666666667 | 0.283974359 |
| AF637_mlp_ls0_seed42 | 42 | 12 | 32 | 512 | 124130 | 0.2176341997 | 0.4663984518 | 0.2527777778 | 0.283974359 |
| AF637_mlp_ls0_seed43 | 43 | 14 | 34 | 544 | 124130 | 0.2176341997 | 0.4650674575 | 0.2527777778 | 0.2641025641 |
| AF637_mlp_ls5_seed41 | 41 | 11 | 31 | 496 | 124130 | 0.2617196988 | 0.5120003355 | 0.2666666667 | 0.283974359 |
| AF637_mlp_ls5_seed42 | 42 | 12 | 32 | 512 | 124130 | 0.2176341997 | 0.4611245363 | 0.2527777778 | 0.283974359 |
| AF637_mlp_ls5_seed43 | 43 | 14 | 34 | 544 | 124130 | 0.2176341997 | 0.4613411975 | 0.2666666667 | 0.2679487179 |
| Official_baseline | — | — | — | — | — | — | — | 0.5527777778 | 0.4320512821 |
| Linear_same_split | — | — | — | — | — | — | — | 0.3805555556 | 0.3519230769 |

설정표는 seed만 다른 동일 recipe를 한 번만 표시한다. crop/mask 키가 없는 초기 recipe는 해당 증강을 적용하지 않은 값 0으로 표시했다. 최대 epoch·patience·min_delta는 원본 recipe에 보존돼 있다.

| 대표 후보 | LR | batch | WD | dropout | crop 확률 | mask 확률 | NSynth mass | 구조·loss·scheduler 옵션 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M330_mlp_ls0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.0; lr_plateau=False; alternative_output= |
| M330_mlp_ls5_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1024; mlp_output=True; cosine_output=False; label_smoothing=0.05; lr_plateau=False; alternative_output= |
| AF637_mlp_ls0_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1280; mlp_output=True; cosine_output=False; label_smoothing=0.0; lr_plateau=False; alternative_output= |
| AF637_mlp_ls5_seed41 | 0.0001 | 64 | 0.01 | 0.1 | 0.7 | 0.3 | 0 | simplified=True; compact=True; temporal_bins=1; softmax_output=True; input_dim=1280; mlp_output=True; cosine_output=False; label_smoothing=0.05; lr_plateau=False; alternative_output= |

## 부록 B. 초기 실험·전체 fine-tuning·후속 비교의 상세 지표

분모는 File/Voice/Music/Voice Presence/Music Presence 순서다. unknown을 제외한 known-subset 결과와 한 클래스뿐이라 계산할 수 없는 AUC는 원본 상태를 유지했다. root 이하 경로는 JSON 내부 panel 위치다. 여러 단계에 같은 패널이 중복 저장된 경우도 원래 문맥을 남긴다.
### G0–G4

근거: [results/pilot_20260912_160943/comparison.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/pilot_20260912_160943/comparison.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/graphs/G0 | 0.1234756098 | 0.041958042 | 0.3078947368 | — | 1 | 65/48/29/52/65 | — |
| root/graphs/G1 | 0.1234756098 | 0.0839160839 | 0.3078947368 | — | 1 | 65/48/29/52/65 | — |
| root/graphs/G2 | 0.1234756098 | 0.041958042 | 0.3078947368 | — | 1 | 65/48/29/52/65 | — |
| root/graphs/G3 | 0.156504065 | 0.041958042 | 0.3842105263 | — | 1 | 65/48/29/52/65 | — |
| root/graphs/G4 | 0.156504065 | 0.041958042 | 0.3078947368 | — | 1 | 65/48/29/52/65 | — |

### AST V2

근거: [results/v2_colab_20260912_174330/comparison.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/v2_colab_20260912_174330/comparison.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/selection/models/G0/metric_report | 0.1557112069 | 0.0304147465 | 0.603030303 | 1 | 0.9232026144 | 122/66/68/90/122 | — |
| root/selection/models/AST_FROZEN/metric_report | 0.2214439655 | 0.166359447 | 0.1766233766 | 1 | 1 | 122/66/68/90/122 | — |
| root/selection/models/AST_LAST_BLOCK/metric_report | 0.1557112069 | 0.1359447005 | 0.1471861472 | 1 | 1 | 122/66/68/90/122 | — |
| root/fresh_test/combined/G0 | 0.2045304233 | 0.0865573186 | 0.4928667823 | 1 | 0.9945838696 | 274/150/148/174/274 | — |
| root/fresh_test/combined/AST_FROZEN | 0.277281746 | 0.1534141558 | 0.3710237131 | 1 | 1 | 274/150/148/174/274 | — |
| root/fresh_test/combined/AST_LAST_BLOCK | 0.2333553792 | 0.1133000535 | 0.3710237131 | 1 | 1 | 274/150/148/174/274 | — |
| root/fresh_test/test_generator/G0 | 0.2209821429 | 0.1410984848 | 0.3224789916 | — | 1 | 176/92/96/96/176 | — |
| root/fresh_test/test_generator/AST_FROZEN | 0.3292410714 | 0.1846590909 | 0.3949579832 | — | 1 | 176/92/96/96/176 | — |
| root/fresh_test/test_generator/AST_LAST_BLOCK | 0.2834821429 | 0.1410984848 | 0.3949579832 | — | 1 | 176/92/96/96/176 | — |
| root/fresh_test/test_source/G0 | 0.1633333333 | 0.0346475508 | 0.6536731634 | 1 | 0.9828595318 | 98/58/52/78/98 | — |
| root/fresh_test/test_source/AST_FROZEN | 0.2041666667 | 0.1039426523 | 0.212143928 | 1 | 1 | 98/58/52/78/98 | — |
| root/fresh_test/test_source/AST_LAST_BLOCK | 0.1020833333 | 0.0692951016 | 0.1731634183 | 1 | 1 | 98/58/52/78/98 | — |

### AST hybrid

근거: [results/v2_colab_20260912_174330/hybrid_comparison.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/v2_colab_20260912_174330/hybrid_comparison.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/fresh_test/combined/G0 | 0.2045304233 | 0.0865573186 | 0.4928667823 | 1 | 0.9945838696 | 274/150/148/174/274 | — |
| root/fresh_test/combined/G0_AST_MUSIC | 0.2045304233 | 0.0865573186 | 0.3710237131 | 1 | 1 | 274/150/148/174/274 | — |
| root/fresh_test/test_generator/G0 | 0.2209821429 | 0.1410984848 | 0.3224789916 | — | 1 | 176/92/96/96/176 | — |
| root/fresh_test/test_generator/G0_AST_MUSIC | 0.2209821429 | 0.1410984848 | 0.3949579832 | — | 1 | 176/92/96/96/176 | — |
| root/fresh_test/test_source/G0 | 0.1633333333 | 0.0346475508 | 0.6536731634 | 1 | 0.9828595318 | 98/58/52/78/98 | — |
| root/fresh_test/test_source/G0_AST_MUSIC | 0.1633333333 | 0.0346475508 | 0.1731634183 | 1 | 1 | 98/58/52/78/98 | — |

### AST epoch 연장

근거: [results/epoch_extension_20260913_013624/comparison.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/epoch_extension_20260913_013624/comparison.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/development/G0 | 0.1557112069 | 0.0304147465 | 0.603030303 | 1 | 0.9232026144 | 122/66/68/90/122 | — |
| root/development/AST_6 | 0.1557112069 | 0.1359447005 | 0.1471861472 | 1 | 1 | 122/66/68/90/122 | — |
| root/development/AST_extended | 0.1064116379 | 0.1359447005 | 0.0883116883 | 1 | 1 | 122/66/68/90/122 | — |
| root/reused_evaluation/combined/G0 | 0.2045304233 | 0.0865573186 | 0.4928667823 | 1 | 0.9945838696 | 274/150/148/174/274 | — |
| root/reused_evaluation/combined/AST_6 | 0.2333553792 | 0.1133000535 | 0.3710237131 | 1 | 1 | 274/150/148/174/274 | — |
| root/reused_evaluation/combined/AST_extended | 0.2302689594 | 0.1133000535 | 0.3710237131 | 1 | 1 | 274/150/148/174/274 | — |
| root/reused_evaluation/combined/hybrid_6 | 0.2045304233 | 0.0865573186 | 0.3710237131 | 1 | 1 | 274/150/148/174/274 | — |
| root/reused_evaluation/combined/hybrid_extended | 0.2045304233 | 0.0865573186 | 0.3710237131 | 1 | 1 | 274/150/148/174/274 | — |
| root/reused_evaluation/test_generator/G0 | 0.2209821429 | 0.1410984848 | 0.3224789916 | — | 1 | 176/92/96/96/176 | — |
| root/reused_evaluation/test_generator/AST_6 | 0.2834821429 | 0.1410984848 | 0.3949579832 | — | 1 | 176/92/96/96/176 | — |
| root/reused_evaluation/test_generator/AST_extended | 0.2834821429 | 0.1410984848 | 0.3949579832 | — | 1 | 176/92/96/96/176 | — |
| root/reused_evaluation/test_generator/hybrid_6 | 0.2209821429 | 0.1410984848 | 0.3949579832 | — | 1 | 176/92/96/96/176 | — |
| root/reused_evaluation/test_generator/hybrid_extended | 0.2209821429 | 0.1410984848 | 0.3949579832 | — | 1 | 176/92/96/96/176 | — |
| root/reused_evaluation/test_source/G0 | 0.1633333333 | 0.0346475508 | 0.6536731634 | 1 | 0.9828595318 | 98/58/52/78/98 | — |
| root/reused_evaluation/test_source/AST_6 | 0.1020833333 | 0.0692951016 | 0.1731634183 | 1 | 1 | 98/58/52/78/98 | — |
| root/reused_evaluation/test_source/AST_extended | 0.1020833333 | 0.0692951016 | 0.1731634183 | 1 | 1 | 98/58/52/78/98 | — |
| root/reused_evaluation/test_source/hybrid_6 | 0.1633333333 | 0.0346475508 | 0.1731634183 | 1 | 1 | 98/58/52/78/98 | — |
| root/reused_evaluation/test_source/hybrid_extended | 0.1633333333 | 0.0346475508 | 0.1731634183 | 1 | 1 | 98/58/52/78/98 | — |

### Recovery V3 개발 선택

근거: [results/recovery_v3/colab_20260913_044621/selection.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/recovery_v3/colab_20260913_044621/selection.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/candidates/G0/development | 0.1557112069 | 0.0304147465 | 0.603030303 | 1 | 0.9232026144 | 122/66/68/90/122 | — |
| root/candidates/SONICS_RAW/development | 0.2621228448 | 0.0304147465 | 0.6619047619 | 1 | 0.9232026144 | 122/66/68/90/122 | — |
| root/candidates/SONICS_STEM/development | 0.2621228448 | 0.0304147465 | 0.5735930736 | 1 | 0.9232026144 | 122/66/68/90/122 | — |
| root/candidates/BLEND_RAW/development | 0.2214439655 | 0.0304147465 | 0.603030303 | 1 | 0.9232026144 | 122/66/68/90/122 | — |
| root/candidates/BLEND_STEM/development | 0.2050107759 | 0.0304147465 | 0.5735930736 | 1 | 0.9232026144 | 122/66/68/90/122 | — |

### Recovery V3 평가

근거: [results/recovery_v3/colab_20260913_044621/comparison.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/recovery_v3/colab_20260913_044621/comparison.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/fresh/G0 | 0.1830357143 | 0 | 0.3333333333 | — | 1 | 88/40/48/40/88 | — |

### V4 본 학습 전 LR probe

각 probe는 60 update·1,916 window exposure로 완료했다. 본 학습의 최적 epoch나 최종 성능과 구분한다. 비교 대상은 같은 dev210개이며 Music EER 대상은170개다.

| probe | 학습 파라미터 | File EER | Music EER | rank score | 실행 초 | 원본 |
|---|---:|---:|---:|---:|---:|---|
| probe_sonics_1e-05 | 17172493 | 0.3714285714 | 0.4121428571 | -0.1890000000 | 48.496284 | [기록](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-retry-first-probe-terminal.json>) |
| probe_ast_1e-05 | 86188033 | 0.3107142857 | 0.3878571429 | -0.1440000000 | 75.655282 | [기록](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-retry-probe_ast_1e-05-terminal.json>) |
| probe_ast_1e-06 | 86188033 | 0.3107142857 | 0.3000000000 | -0.0720000000 | 58.337819 | [기록](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-retry-probe_ast_1e-06-terminal.json>) |
| probe_ast_3e-06 | 86188033 | 0.3428571429 | 0.4171428571 | -0.0450000000 | 70.924026 | [기록](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-retry-probe_ast_3e-06-terminal.json>) |
| probe_sonics_1e-04 | 17172493 | 0.3321428571 | 0.3121428571 | -0.1440000000 | 16.480443 | [기록](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-retry-probe_sonics_1e-04-terminal.json>) |
| probe_sonics_3e-05 | 17172493 | 0.3428571429 | 0.3585714286 | -0.1170000000 | 17.345047 | [기록](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-retry-probe_sonics_3e-05-terminal.json>) |

### V5 원래·중복 제거

근거: [.omo/evidence/v5-colab-runtime/results/analysis.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v5-colab-runtime/results/analysis.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/arms/B0/original | 0.3781746032 | 0.15 | 0.5422222222 | 0.9979166667 | 0.9851315789 | 230/100/190/140/230 | — |
| root/arms/B0/unique_audio | 0.3379807692 | 0.15 | 0.5763888889 | 0.9979166667 | 0.9833823529 | 210/100/170/140/210 | — |
| root/arms/B1/original | 0.3218253968 | 0.1708333333 | 0.3683333333 | 0.9983333333 | 0.9858552632 | 230/100/190/140/230 | — |
| root/arms/B1/unique_audio | 0.3100961538 | 0.1708333333 | 0.3472222222 | 0.9983333333 | 0.9841911765 | 210/100/170/140/210 | — |
| root/arms/B2/original | 0.3781746032 | 0.15 | 0.4577777778 | 0.9695833333 | 0.5935526316 | 230/100/190/140/230 | — |
| root/arms/B2/unique_audio | 0.3759615385 | 0.15 | 0.4645833333 | 0.9695833333 | 0.57875 | 210/100/170/140/210 | — |

### V6

근거: [results/v6_l40s_20260915/v6_run/dev_metrics.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/v6_l40s_20260915/v6_run/dev_metrics.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root | 0.3857142857 | 0.4291666667 | 0.4828571429 | — | 0.9954411765 | 210/100/170/120/210 | — |

### V7 C0–C5

근거: [results/v7_a100_20260915/results/comparison.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/v7_a100_20260915/results/comparison.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/V6_saved/deduplicated | 0.3358974359 | 0.4291666667 | 0.4666666667 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/V6_saved/full | 0.3857142857 | 0.4291666667 | 0.4828571429 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C0_class_balance/full | 0.3857142857 | 0.4291666667 | 0.4828571429 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C0_class_balance/deduplicated | 0.3358974359 | 0.4291666667 | 0.4666666667 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C1_nsynth_5pct/full | 0.3571428571 | 0.4291666667 | 0.4707142857 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C1_nsynth_5pct/deduplicated | 0.316025641 | 0.4291666667 | 0.4666666667 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C2_no_nsynth/full | 0.3857142857 | 0.4291666667 | 0.4414285714 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C2_no_nsynth/deduplicated | 0.3519230769 | 0.4291666667 | 0.4138888889 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C3_equal_windows/full | 0.3571428571 | 0.4291666667 | 0.3171428571 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C3_equal_windows/deduplicated | 0.316025641 | 0.4291666667 | 0.3138888889 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/Official_baseline/full | 0.4392857143 | 0.1708333333 | 0.5292857143 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/Official_baseline/deduplicated | 0.4320512821 | 0.1708333333 | 0.5527777778 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C4_four_seconds_seed41/full | 0.4142857143 | 0.4291666667 | 0.3585714286 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C4_four_seconds_seed41/deduplicated | 0.3961538462 | 0.4291666667 | 0.3472222222 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C5_four_seconds_no_nsynth_seed41/full | 0.3464285714 | 0.4291666667 | 0.3828571429 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C5_four_seconds_no_nsynth_seed41/deduplicated | 0.316025641 | 0.4291666667 | 0.3527777778 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C4_four_seconds_seed42/full | 0.4071428571 | 0.4291666667 | 0.3707142857 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C4_four_seconds_seed42/deduplicated | 0.383974359 | 0.4291666667 | 0.3861111111 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C5_four_seconds_no_nsynth_seed42/full | 0.3714285714 | 0.4291666667 | 0.3878571429 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C5_four_seconds_no_nsynth_seed42/deduplicated | 0.316025641 | 0.4291666667 | 0.3527777778 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C4_four_seconds_seed43/full | 0.3857142857 | 0.4291666667 | 0.3585714286 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C4_four_seconds_seed43/deduplicated | 0.3679487179 | 0.4291666667 | 0.3527777778 | — | 0.9948333333 | 190/100/150/120/190 | — |
| root/C5_four_seconds_no_nsynth_seed43/full | 0.3392857143 | 0.4291666667 | 0.3414285714 | — | 0.9954411765 | 210/100/170/120/210 | — |
| root/C5_four_seconds_no_nsynth_seed43/deduplicated | 0.316025641 | 0.4291666667 | 0.3194444444 | — | 0.9948333333 | 190/100/150/120/190 | — |

### fast H100 전체·부분집합

근거: [results/fast_h100_20260917/downloaded/unpacked/metrics.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/fast_h100_20260917/downloaded/unpacked/metrics.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/validation/M330_stem_max/all | 0.0784615385 | 0.0895721925 | 0.1331653602 | 1 | 0.9956936685 | 227/78/179/94/227 | — |
| root/validation/M330_stem_max/natural | 0.0736453202 | 0.0416666667 | 0.0567033976 | 1 | 0.9944544365 | 187/48/139/54/187 | — |
| root/validation/M330_stem_max/mixed | 0.1 | 0.1 | 0.2 | — | — | 40/30/40/40/40 | — |
| root/validation/M330_stem_max/music_without_nsynth | 0.1659779614 | — | 0.0829889807 | — | — | 133/0/133/0/133 | — |
| root/validation/M330_raw_max/all | 0.1358791209 | 0.0895721925 | 0.2116461366 | 1 | 0.9956936685 | 227/78/179/94/227 | — |
| root/validation/M330_raw_max/natural | 0.0958949097 | 0.0416666667 | 0.1092745638 | 1 | 0.9944544365 | 187/48/139/54/187 | — |
| root/validation/M330_raw_max/mixed | 0.2 | 0.1 | 0.3 | — | — | 40/30/40/40/40 | — |
| root/validation/M330_raw_max/music_without_nsynth | 0.1659779614 | — | 0.1494490358 | — | — | 133/0/133/0/133 | — |
| root/validation/M330_dual_max/all | 0.0784615385 | 0.0895721925 | 0.1331653602 | 1 | 0.9956936685 | 227/78/179/94/227 | — |
| root/validation/M330_dual_max/natural | 0.0736453202 | 0.0416666667 | 0.0567033976 | 1 | 0.9944544365 | 187/48/139/54/187 | — |
| root/validation/M330_dual_max/mixed | 0.1 | 0.1 | 0.2 | — | — | 40/30/40/40/40 | — |
| root/validation/M330_dual_max/music_without_nsynth | 0.1659779614 | — | 0.0829889807 | — | — | 133/0/133/0/133 | — |
| root/validation/M330_stem_direct/all | 0.1569230769 | 0.0895721925 | 0.1331653602 | 1 | 0.9956936685 | 227/78/179/94/227 | — |
| root/validation/M330_stem_direct/natural | 0.1369458128 | 0.0416666667 | 0.0567033976 | 1 | 0.9944544365 | 187/48/139/54/187 | — |
| root/validation/M330_stem_direct/mixed | 0.3 | 0.1 | 0.2 | — | — | 40/30/40/40/40 | — |
| root/validation/M330_stem_direct/music_without_nsynth | 0.2489669421 | — | 0.0829889807 | — | — | 133/0/133/0/133 | — |
| root/validation/M330_dual_direct/all | 0.1569230769 | 0.0895721925 | 0.1331653602 | 1 | 0.9956936685 | 227/78/179/94/227 | — |
| root/validation/M330_dual_direct/natural | 0.1369458128 | 0.0416666667 | 0.0567033976 | 1 | 0.9944544365 | 187/48/139/54/187 | — |
| root/validation/M330_dual_direct/mixed | 0.3 | 0.1 | 0.2 | — | — | 40/30/40/40/40 | — |
| root/validation/M330_dual_direct/music_without_nsynth | 0.2489669421 | — | 0.0829889807 | — | — | 133/0/133/0/133 | — |
| root/test/M330_stem_max/all | 0.1100628931 | 0.0635026738 | 0.2065641026 | 1 | 0.9857723577 | 212/78/164/99/212 | — |
| root/test/M330_stem_max/natural | 0.0930232558 | 0 | 0.1145363409 | 1 | 0.986391129 | 172/48/124/59/172 | — |
| root/test/M330_stem_max/mixed | 0.2 | 0.175 | 0.25 | — | — | 40/30/40/40/40 | — |
| root/test/M330_stem_max/music_without_nsynth | 0.1482142857 | — | 0.1482142857 | — | — | 113/0/113/0/113 | — |
| root/test/M330_raw_max/all | 0.1100628931 | 0.0635026738 | 0.2105641026 | 1 | 0.9857723577 | 212/78/164/99/212 | — |
| root/test/M330_raw_max/natural | 0.0930232558 | 0 | 0.1598997494 | 1 | 0.986391129 | 172/48/124/59/172 | — |
| root/test/M330_raw_max/mixed | 0.2 | 0.175 | 0.4 | — | — | 40/30/40/40/40 | — |
| root/test/M330_raw_max/music_without_nsynth | 0.2535714286 | — | 0.2488095238 | — | — | 113/0/113/0/113 | — |
| root/test/M330_dual_max/all | 0.1100628931 | 0.0635026738 | 0.2065641026 | 1 | 0.9857723577 | 212/78/164/99/212 | — |
| root/test/M330_dual_max/natural | 0.0930232558 | 0 | 0.1145363409 | 1 | 0.986391129 | 172/48/124/59/172 | — |
| root/test/M330_dual_max/mixed | 0.2 | 0.175 | 0.25 | — | — | 40/30/40/40/40 | — |
| root/test/M330_dual_max/music_without_nsynth | 0.1482142857 | — | 0.1482142857 | — | — | 113/0/113/0/113 | — |
| root/test/M330_stem_direct/all | 0.1509433962 | 0.0635026738 | 0.2065641026 | 1 | 0.9857723577 | 212/78/164/99/212 | — |
| root/test/M330_stem_direct/natural | 0.1627906977 | 0 | 0.1145363409 | 1 | 0.986391129 | 172/48/124/59/172 | — |
| root/test/M330_stem_direct/mixed | 0.1166666667 | 0.175 | 0.25 | — | — | 40/30/40/40/40 | — |
| root/test/M330_stem_direct/music_without_nsynth | 0.1386904762 | — | 0.1482142857 | — | — | 113/0/113/0/113 | — |
| root/test/M330_dual_direct/all | 0.1509433962 | 0.0635026738 | 0.2065641026 | 1 | 0.9857723577 | 212/78/164/99/212 | — |
| root/test/M330_dual_direct/natural | 0.1627906977 | 0 | 0.1145363409 | 1 | 0.986391129 | 172/48/124/59/172 | — |
| root/test/M330_dual_direct/mixed | 0.1166666667 | 0.175 | 0.25 | — | — | 40/30/40/40/40 | — |
| root/test/M330_dual_direct/music_without_nsynth | 0.1386904762 | — | 0.1482142857 | — | — | 113/0/113/0/113 | — |

### redesign A/B

근거: [results/redesign_h100_20260917/comparison_A_vs_B.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_h100_20260917/comparison_A_vs_B.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/candidate_metrics | 0.1094771242 | — | 0.0811546841 | — | — | 129/0/129/20/129 | — |
| root/control | 0.1094771242 | — | 0.0811546841 | — | — | 129/0/129/20/129 | — |

### redesign noisy-OR

근거: [results/redesign_h100_20260917/comparison_file_max_vs_noisy_or.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_h100_20260917/comparison_file_max_vs_noisy_or.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/candidate_metrics | 0.0811546841 | — | 0.0811546841 | — | — | 129/0/129/20/129 | — |
| root/control | 0.1094771242 | — | 0.0811546841 | — | — | 129/0/129/20/129 | — |

### redesign OOF relation

근거: [results/redesign_h100_20260917/comparison_file_max_vs_oof_relation.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_h100_20260917/comparison_file_max_vs_oof_relation.json>)

| 패널/후보 | File EER | Voice EER | Music EER | Voice AUC | Music AUC | 유효 분모 F/V/M/VP/MP | 공식 score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| root/candidate_metrics | 0.1045751634 | — | 0.0811546841 | — | — | 129/0/129/20/129 | — |
| root/control | 0.1094771242 | — | 0.0811546841 | — | — | 129/0/129/20/129 | — |

### H100 fullFT8의 정확한 수치

근거: [results/fullft8_h100x1_20260916/final_metrics.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/fullft8_h100x1_20260916/final_metrics.json>)

| 실험 | seed | batch | 선택 epoch | tune Music EER | test Music EER | test 음악 분모 | held-generator EER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01_S0_MERT_frozen | 41 | 1 | 5 | 0.2460934337 | 0.2666666667 | 150 | — |
| 02_S0_MERT_fullft | 41 | 1 | 3 | 0.2304671687 | 0.3194444444 | 150 | — |
| 03_S0_EAT_fullft | 41 | 1 | 1 | 0.2176341997 | 0.3 | 150 | — |
| 04_S1_MERT_fullft | 41 | 1 | 8 | 0.1006944444 | 0.2485955056 | 190 | 0.2708333333 |
| 05_S1_EAT_fullft | 41 | 4 | 6 | 0.0069444444 | 0.2485955056 | 190 | 0.25 |
| 06_S2_MERT_fullft | 41 | 4 | 2 | 0.0972222222 | 0.3323970037 | 190 | 0.3333333333 |
| 07_S2_EAT_fullft | 41 | 4 | 4 | 0.0138888889 | 0.1676029963 | 190 | 0.1666666667 |
| 08_S0_adapted_fullft | 41 | 1 | 6 | 0.1707554044 | 0.2527777778 | 150 | — |

### redesign A/B 시드·확률 앙상블 원시 요약

근거: [results/redesign_h100_20260917/metrics_seed_summary.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_h100_20260917/metrics_seed_summary.json>)

| arm | 구분 | seed | Music EER 또는 평균 | Music SD | File EER 또는 평균 | File SD |
| --- | --- | --- | --- | --- | --- | --- |
| A | seed | 41 | 0.0762527233 | — | 0.1094771242 | — |
| A | seed | 42 | 0.1045751634 | — | 0.114379085 | — |
| A | seed | 43 | 0.1476034858 | — | 0.1525054466 | — |
| B | seed | 41 | 0.1094771242 | — | 0.1094771242 | — |
| B | seed | 42 | 0.1045751634 | — | 0.1094771242 | — |
| B | seed | 43 | 0.1094771242 | — | 0.1476034858 | — |
| A | seed_mean_sd | — | 0.1094771242 | 0.0293343342 | 0.1254538853 | 0.0192327418 |
| B | seed_mean_sd | — | 0.1078431373 | 0.0023108065 | 0.1221859114 | 0.0179729392 |
| A | probability_ensemble | — | 0.0811546841 | — | 0.1094771242 | — |
| B | probability_ensemble | — | 0.0811546841 | — | 0.1094771242 | — |

## 부록 C. 원본 자료와 검증 범위

이 문서는 기존 결과를 모은 기록이다. 이번 작성에서는 원본 JSON과 보고서의 수치를 대조하고, 부록을 JSON에서 직접 생성했으며, 링크 대상과 수치 전사를 검사했다. 전체 오디오를 다시 디코딩하거나 모든 학습·GPU 추론·공식 사이트 검증을 반복하지 않았다. 원본 보고서의 “미제출”은 해당 작성 시점의 상태다. 이후 공식 점수가 있는 파일은 본문 3절의 제출 저장본을 우선한다. V4의 LR probe 여섯 개도 부록 B에 별도로 기재했다.

| 근거 | 파일 |
| --- | --- |
| E01 | [.omo/evidence/model-redesign-20260917/submissions-dom.txt](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/model-redesign-20260917/submissions-dom.txt>) |
| E02 | [.omo/evidence/model-redesign-20260917/evaluation-dom.txt](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/model-redesign-20260917/evaluation-dom.txt>) |
| E03 | [RESULTS.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/RESULTS.md>) |
| E04 | [RESULTS_V2.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/RESULTS_V2.md>) |
| E05 | [RESULTS_EPOCH_EXTENSION.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/RESULTS_EPOCH_EXTENSION.md>) |
| E06 | [results/recovery_v3/failure_analysis.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/recovery_v3/failure_analysis.md>) |
| E07 | [submissions/AST_full1029_epoch17/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/AST_full1029_epoch17/README.md>) |
| E08 | [.omo/evidence/v4-results-report.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-results-report.md>) |
| E09 | [.omo/evidence/v5-colab-runtime/results/RESULTS_KO.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v5-colab-runtime/results/RESULTS_KO.md>) |
| E10 | [.omo/v5-final-submission-notepad.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/v5-final-submission-notepad.md>) |
| E11 | [results/v6_l40s_20260915/v6_run/training.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/v6_l40s_20260915/v6_run/training.json>) |
| E12 | [results/data_audit_20260915/REPORT.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_audit_20260915/REPORT.md>) |
| E13 | [results/v7_a100_20260915/REPORT.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/v7_a100_20260915/REPORT.md>) |
| E14 | [results/c3_final_20260915/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/c3_final_20260915/README.md>) |
| E15 | [C3_EARLY_STOPPING.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/C3_EARLY_STOPPING.md>) |
| E16 | [results/temporal_overfit_20260915/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_overfit_20260915/README.md>) |
| E17 | [results/temporal_h100_20260915/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_20260915/README.md>) |
| E18 | [results/temporal_h100_b128_20260915/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_b128_20260915/README.md>) |
| E19 | [results/temporal_h100_aug_20260915/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_aug_20260915/README.md>) |
| E20 | [results/temporal_h100_simple_20260915/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_simple_20260915/README.md>) |
| E21 | [results/temporal_h100_compact_20260915/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_compact_20260915/README.md>) |
| E22 | [submissions/C08/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/C08/README.md>) |
| E23 | [results/temporal_h100_resolution_20260915/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_resolution_20260915/README.md>) |
| E24 | [results/mert330_a100_20260916/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert330_a100_20260916/README.md>) |
| E25 | [submissions/M330_MLP_full1206/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/M330_MLP_full1206/README.md>) |
| E26 | [results/cosine_a100_20260916/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/cosine_a100_20260916/README.md>) |
| E27 | [results/methods_a100_20260916/COMPARISON.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/COMPARISON.md>) |
| E28 | [results/afwhisper_a100_20260916/COMPARISON.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/afwhisper_a100_20260916/COMPARISON.md>) |
| E29 | [results/fullft8_h100x1_20260916/RESULTS.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/fullft8_h100x1_20260916/RESULTS.md>) |
| E30 | [.omo/evidence/model-redesign-20260917/local-evidence.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/model-redesign-20260917/local-evidence.md>) |
| E31 | [results/fast_h100_20260917/RESULTS.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/fast_h100_20260917/RESULTS.md>) |
| E32 | [results/redesign_h100_20260917/RESULT_SUMMARY.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_h100_20260917/RESULT_SUMMARY.md>) |
| E33 | [results/leaderboard_audit_20260917/experiment_plan.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/leaderboard_audit_20260917/experiment_plan.md>) |
| E34 | [results/redesign_full_20260917/completion_summary.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_full_20260917/completion_summary.json>) |
| E35 | [.omo/evidence/v6-submission-fix/DIAGNOSIS.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v6-submission-fix/DIAGNOSIS.md>) |
| E36 | [results/s2_811_h100_20260916/README.md](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/s2_811_h100_20260916/README.md>) |
| E37 | [results/six_h100_20260917/data_receipt.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/six_h100_20260917/data_receipt.json>) |

### 재현에 필요한 주요 원시 기록
- [results/data_audit_20260915/summary.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_audit_20260915/summary.json>)
- [results/v2_colab_20260912_174330/training_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/v2_colab_20260912_174330/training_report.json>)
- [results/epoch_extension_20260913_013624/training_report.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/epoch_extension_20260913_013624/training_report.json>)
- [results/full_refit_20260913/final_weight_audit.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/full_refit_20260913/final_weight_audit.json>)
- [.omo/evidence/v4-results-summary.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-results-summary.json>)
- [.omo/evidence/v5-final-pool-audit.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v5-final-pool-audit.json>)
- [.omo/evidence/v5-final-pair/packaging_receipt.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v5-final-pair/packaging_receipt.json>)
- [results/c3_final_20260915/unpacked/final_run/training.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/c3_final_20260915/unpacked/final_run/training.json>)
- [results/mlp_full_a100_20260916/unpacked/final_run/training.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/mlp_full_a100_20260916/unpacked/final_run/training.json>)
- [results/fullft8_h100x1_20260916/downloaded/h100x1_fullft8_run/protocol.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/fullft8_h100x1_20260916/downloaded/h100x1_fullft8_run/protocol.json>)
- [results/redesign_h100_20260917/posthoc_raw_file_interaction.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_h100_20260917/posthoc_raw_file_interaction.json>)
- [results/leaderboard_audit_20260917/source_slice_replay.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/leaderboard_audit_20260917/source_slice_replay.json>)
- [results/redesign_full_20260917/downloaded/unpacked/output/fullrefit_receipt.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_full_20260917/downloaded/unpacked/output/fullrefit_receipt.json>)
- [results/redesign_full_20260917/downloaded/unpacked/output/submission_smoke.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_full_20260917/downloaded/unpacked/output/submission_smoke.json>)
- [results/s2_expanded_eval_20260916/status.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/s2_expanded_eval_20260916/status.json>)
- [results/six_h100_20260917/data_receipt.json](</Users/choiwuseck/Documents/ChatGPT/데이콘/results/six_h100_20260917/data_receipt.json>)

### 후보·시드 기록의 수록 범위

다음 수는 JSON 기록 수다. 추가 시드, 재사용 control, 여러 평가 패널이 있으므로 합계를 독립 실험 수로 부르지 않는다.

| 실험 폴더 | tuning 기록 | test/control 기록 | seed 중복 제외 recipe |
| --- | --- | --- | --- |
| temporal_h100_20260915 | 8 | 5 | 8 |
| temporal_h100_b128_20260915 | 8 | 5 | 8 |
| temporal_h100_aug_20260915 | 9 | 11 | 3 |
| temporal_h100_simple_20260915 | 6 | 8 | 2 |
| temporal_h100_compact_20260915 | 37 | 11 | 25 |
| temporal_h100_resolution_20260915 | 12 | 14 | 4 |
| mert330_a100_20260916 | 6 | 8 | 2 |
| cosine_a100_20260916 | 6 | 8 | 2 |
| methods_a100_20260916 | 48 | 50 | 16 |
| afwhisper_a100_20260916 | 12 | 14 | 4 |

부록 B에는 분모가 붙은 지표 보고서 124행과 fullFT8의 8행, redesign A/B 요약 10행을 수록했다. epoch별 전체 곡선과 파일별 예측은 위 원본 파일을 따른다.

[E01]: </Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/model-redesign-20260917/submissions-dom.txt>

[E02]: </Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/model-redesign-20260917/evaluation-dom.txt>

[E03]: </Users/choiwuseck/Documents/ChatGPT/데이콘/RESULTS.md>

[E04]: </Users/choiwuseck/Documents/ChatGPT/데이콘/RESULTS_V2.md>

[E05]: </Users/choiwuseck/Documents/ChatGPT/데이콘/RESULTS_EPOCH_EXTENSION.md>

[E06]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/recovery_v3/failure_analysis.md>

[E07]: </Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/AST_full1029_epoch17/README.md>

[E08]: </Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v4-results-report.md>

[E09]: </Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v5-colab-runtime/results/RESULTS_KO.md>

[E10]: </Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/v5-final-submission-notepad.md>

[E11]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/v6_l40s_20260915/v6_run/training.json>

[E12]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/data_audit_20260915/REPORT.md>

[E13]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/v7_a100_20260915/REPORT.md>

[E14]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/c3_final_20260915/README.md>

[E15]: </Users/choiwuseck/Documents/ChatGPT/데이콘/C3_EARLY_STOPPING.md>

[E16]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_overfit_20260915/README.md>

[E17]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_20260915/README.md>

[E18]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_b128_20260915/README.md>

[E19]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_aug_20260915/README.md>

[E20]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_simple_20260915/README.md>

[E21]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_compact_20260915/README.md>

[E22]: </Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/C08/README.md>

[E23]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/temporal_h100_resolution_20260915/README.md>

[E24]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/mert330_a100_20260916/README.md>

[E25]: </Users/choiwuseck/Documents/ChatGPT/데이콘/submissions/M330_MLP_full1206/README.md>

[E26]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/cosine_a100_20260916/README.md>

[E27]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/methods_a100_20260916/COMPARISON.md>

[E28]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/afwhisper_a100_20260916/COMPARISON.md>

[E29]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/fullft8_h100x1_20260916/RESULTS.md>

[E30]: </Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/model-redesign-20260917/local-evidence.md>

[E31]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/fast_h100_20260917/RESULTS.md>

[E32]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_h100_20260917/RESULT_SUMMARY.md>

[E33]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/leaderboard_audit_20260917/experiment_plan.md>

[E34]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/redesign_full_20260917/completion_summary.json>

[E35]: </Users/choiwuseck/Documents/ChatGPT/데이콘/.omo/evidence/v6-submission-fix/DIAGNOSIS.md>

[E36]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/s2_811_h100_20260916/README.md>

[E37]: </Users/choiwuseck/Documents/ChatGPT/데이콘/results/six_h100_20260917/data_receipt.json>

<!-- HUMANIZE-SUMMARY
원문 107168자 / 윤문 본문 107157자. 줄별 문자 편집량 기준 참고 변경률 0.0280%.
설치된 스킬에 scripts/verify_change_rate.py가 없어 공식 변경률 도구는 실행하지 못했다.
I-2 형식명사 강조 1→0, D-2 의의 강조 1→0; 어휘 "취합한"→"모은" 1건.
수치·코드 식별자·링크 자동 보존 검사, 의미·장르·문체·불확실성 수동 검토 완료.
자체검증: 보존·장르·register·S1·수사 추가 금지 5항 확인; 변경률은 참고 계산으로만 확인.
등급 B: 기술 실험 기록의 근거와 수치를 유지하며 세 문장만 보수적으로 다듬음.
변경: "문제는 … 점이다"→"문제가 반복됐다"; "중요한 학습 범위 차이"→"학습 범위에 차이"; "취합한 기록"→"모은 기록".
-->
