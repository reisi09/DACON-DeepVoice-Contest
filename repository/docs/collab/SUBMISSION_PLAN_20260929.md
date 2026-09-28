# 2026-09-29 제출 계획과 판단 기준

작성 2026-09-28 16:40 KST, 수정 17:45 KST(Codex 15–21라운드 합의 반영). 남은 제출은 3번이고, 마감은 09-29 10:00 KST다.
최종 점수는 대회 종료 시점의 최고 점수 하나다(Private = 종료 시점 Public). 따라서 낮게 나온 제출은 손해가 없고,
쓰지 않은 제출은 마감과 함께 사라진다.

## 1. 기준선과 목표

| 항목 | 값 |
|---|---|
| 현재 최고 | S7, ADS 0.8373968254, 총점 약 0.8535 (총점 = 0.9·ADS + 0.1·CPS, CPS 약 0.998) |
| top-15 기준선 | 총점 0.8737 (09-26 기준). ADS로 약 0.860, 즉 **S7 대비 ADS +0.022 필요** |
| 오프라인 향상폭 (S7 레시피 대비) | E1 +0.003~0.006, E2 +0.03~0.047 |

## 2. 제출 후보 (모두 정상적으로 끝까지 학습한 모델)

| 순서 | 패키지 | 경로 | sha256 | 검증 상태 |
|---|---|---|---|---|
| ① | s7_new (S7 가중치 + 새 데이터 40,214행) | `submissions/DataV4_s7_new/submit.zip` | `fcdd86913360e4dc31caaaa8fa4ee801360f1b7a6e8d901020d324f7597e5162` | Mac CPU, 출력 일치(V 0.00018 / M 0.00089 / F 0.00011), VESSL CUDA 통과 |
| ② | W_newB (W_new + 조기 종료 val에 newval 추가) | `submissions/DataV4_W_newB/submit.zip` | `8d3b62d3ec80a84c6f89d93098b202a22d700f2efea82315304ae2bfd0a5f353` | Mac CPU, 출력 일치(V 0.00055 / M 0.00101 / F 0.00021), VESSL CUDA 통과 |
| ③ 2순위 | s7_new_ob (s7_new + AI-OpenBMAT 400행) | `submissions/DataV4_s7_new_ob/submit.zip` | `e4205dcecbc32989ba899fe445cb455ff510882aade659605232e1fae4ea1070` | Mac CPU, 출력 일치(V 0.00018 / M 0.00028 / F 0.0001). **VESSL CUDA 미실행**(x1 정지 중) |
| ③ 3순위 | W_new (W 가중치 + 같은 새 데이터) | `submissions/DataV4_W_new/submit.zip` | `58e7e3bd460cde93b6cb65aae02da6985b9e2d5c17c5f9f88622f58df04d1c68` | Mac CPU, 출력 일치(V 0.00064 / M 0.00102 / F 0.0002), VESSL CUDA 통과 |

- 모든 패키지는 인코더와 실행 코드가 S7과 같고 head만 다르다. L4 실행 시간(약 30분)과 GPU 메모리도 S7과 같다.
- FILE 규칙은 모두 0/9/1로 고정했다.

## 3. 오프라인 평가 요약 (DB 프로토콜, S7 레시피 대비 ADS 차이)

| 평가셋 | s7_new | W_new | W_newB | s7_new_ob |
|---|---|---|---|---|
| E1 기존 held-out (4,822) | **+0.0064** (CI +0.0013~+0.0111) | +0.0033 | +0.0032 | +0.0049 |
| E2 전체, 누출 행 제외 (12,351) | +0.0285 | +0.0442 | **+0.0467** | +0.0363 |
| E2 공개 데이터, 누출 행 제외 (2,361) | +0.0058 | +0.0058 | **+0.0071** | +0.0049 |
| E2 AT-ADD 새 생성기 (9,990) | +0.0349 | +0.0562 | **+0.0583** | +0.0469 |
| E3 AI-Hub 한국어 real 오탐 (473 가이드보컬) | **0.027** | 0.040 | 0.040 | 측정 안 함 |

- 누출 점검: E2 12,426행 중 75행이 학습 쪽과 계보를 공유했다(AI-OpenBMAT 음악 트랙 65행, AT-ADD 그룹 10행). 제외해도 결론은 같다.
- s7_new_ob의 AI-OpenBMAT 향상 일부는 누출 때문이었다.
- 가중치만 바꾼 W(새 데이터 없음)는 모든 평가셋에서 S7보다 나빠 후보에서 뺐다.
- 원본 결과: `research/top15/work5/newdata/new_eval_newtest.json`, `leak_check.txt`, `e2_subsets.txt`, `new_eval_e3.json`.

## 4. 제출 순서와 결과 읽는 법 (Codex 15–21라운드 합의, 2026-09-28 17:45 KST)

순서는 LB 결과와 무관하게 고정한다. LB 점수로 레시피를 바꾸거나 새 학습을 하지 않는다.

| 순서 | 패키지 | 조건 |
|---|---|---|
| ① | s7_new | 없음 |
| ② | W_newB | 없음 |
| ③ 1순위 | mix: 음성 s7_new / 음악 W_new | presence 보완 검정 통과 + 패키지 검증 통과 (GPU 승인 필요) |
| ③ 2순위 | s7_new_ob | VESSL CUDA 검증 통과 (GPU 승인 필요) |
| ③ 3순위 | W_new | 위 둘이 없을 때. 이미 검증 완료 |

- ③ 분기: mix가 통과하면 s7_new_ob 검증은 생략한다. mix가 탈락했을 때만 s7_new_ob를 검증하고, 실패하거나 GPU 상한
  (모든 추가 작업 합계 0.5 H100시간)을 넘으면 W_new를 낸다. 검증을 통과한 mix를 다른 후보로 바꾸지 않는다.
  보완 검정은 통과했지만 mix 패키지 검증이 실패하면, s7_new_ob 검증 없이 W_new를 낸다(0.57시간이 되어 상한 초과).
- ①을 s7_new로 두는 이유: 두 후보 모두 오프라인 E2 향상이 +0.022를 넘고, s7_new가 E1 향상(+0.0064)이 크고
  E3 한국어 real 오탐이 낮다(0.027 대 0.040). 실제 top-15 도달 확률은 알 수 없다.

결과 읽는 법 (S7 ADS 0.8373968254 대비):
- ① s7_new − S7: 데이터 추가(40,214행), 재학습, CDF 재적합을 포함한 P1 레시피의 LB 차이.
  생성기, 언어, real 구성 효과를 분리하지 못한다.
- W_newB − s7_new: 가중치와 조기 종료(newval 추가) 변경의 합산 효과.
- ③ − s7_new: s7_new_ob면 AI-OpenBMAT 400행 추가 효과, mix면 음악 레시피 교체 효과. W_new면 W_newB와 조기 종료만 다르다.
- 참고: ADS 0.845는 S7 대비 +0.0076이다. top-15에는 약 +0.022가 필요하다.

철회한 해석: 16:40 판의 4.0절은 "s7_new 한 번으로 채점기 fake가 최신 생성기 위주인지 판별한다"고 적었다.
AT-ADD 34,000행은 fake 21,500 + real 12,500이고, 데이터 추가는 real 구성과 학습 가중치도 함께 바꾼다.
그래서 LB 변화폭만으로 채점기의 성격을 판별할 수 없다. 4.1절의 LB 구간별 분기표도 같은 이유로 폐기했다.

## 5. 일정

| 시각 (KST) | 할 일 |
|---|---|
| 09-28 17:35 | 후보 zip 4개의 sha256 재확인 완료(receipt와 일치). GPU·CPU 워크스페이스 모두 정지 |
| 09-29 00:00 | ① s7_new 제출 (업로드 9.75 GB) |
| 약 00:40 | ① 결과 확인. 결과와 관계없이 ②는 W_newB |
| ① 채점 완료 직후 | ② 제출. LB 해석 때문에 기다리지 않는다 |
| ② 채점 완료 직후 | ③ 제출. 목표는 07:30 이전 업로드. 09:00은 목표가 아니라 위험한 최후 시각이다 |

오늘(09-28) 제출 3번은 S6, S7, S8로 이미 썼다. 화면에 남은 횟수가 없으면 00:00 이후에 제출한다.

업로드 직전 확인(매번): 파일 경로, 전체 sha256(5.1절 명령), 그 sha256의 검증 기록(submit.json, verify.log,
runtime_parity.txt)을 대조한다. 제출 후 화면에서 파일명과 접수 상태를 확인하고 나서 재시도 여부를 정한다.
가장 큰 zip도 9,745,305,856 bytes로, 10,000,000,000 bytes까지 254,694,144 bytes 여유가 있다.


## 5.1 제출 파일 열기 (Mac 터미널)

Finder에서 zip이 선택된 상태로 열린다. 업로드 전에 sha256을 확인한다.

```bash
# ① s7_new
open -R ~/Documents/ChatGPT/데이콘/submissions/DataV4_s7_new/submit.zip
shasum -a 256 ~/Documents/ChatGPT/데이콘/submissions/DataV4_s7_new/submit.zip   # fcdd86913360e4dc31caaaa8fa4ee801360f1b7a6e8d901020d324f7597e5162
# ② W_newB
open -R ~/Documents/ChatGPT/데이콘/submissions/DataV4_W_newB/submit.zip
shasum -a 256 ~/Documents/ChatGPT/데이콘/submissions/DataV4_W_newB/submit.zip   # 8d3b62d3ec80a84c6f89d93098b202a22d700f2efea82315304ae2bfd0a5f353
# ③ s7_new_ob (CUDA 검증을 통과한 경우만)
open -R ~/Documents/ChatGPT/데이콘/submissions/DataV4_s7_new_ob/submit.zip
shasum -a 256 ~/Documents/ChatGPT/데이콘/submissions/DataV4_s7_new_ob/submit.zip  # e4205dcecbc32989ba899fe445cb455ff510882aade659605232e1fae4ea1070
# ③ W_new (mix와 s7_new_ob가 없을 때)
open -R ~/Documents/ChatGPT/데이콘/submissions/DataV4_W_new/submit.zip
shasum -a 256 ~/Documents/ChatGPT/데이콘/submissions/DataV4_W_new/submit.zip    # 58e7e3bd460cde93b6cb65aae02da6985b9e2d5c17c5f9f88622f58df04d1c68
# 현재 최고 S7
open -R ~/Documents/ChatGPT/데이콘/submissions/DataV4_S7_SSLAMWavLM/submit.zip
```

## 6. 남은 위험

- **L4 실행 시간·메모리 (60분, 22.4 GiB)**: 같은 인코더와 Engine을 쓰는 S7이 실제로 채점됐고, 새 패키지는 head
  가중치와 표만 다르다(변경 member 19개). 그래서 위험은 낮다고 본다. 다만 새 zip 각각을 L4에서 직접 측정하지는 않았다.
  Mac CPU QA 시간(273초)을 L4 전체 시간으로 환산하지 않는다.
- **DB와 LB의 괴리**: S8은 DB에서 +0.0043(유의)이었는데 LB에서는 −0.0034였다. 이번 후보는 데이터 변경이고 세 평가셋이 같은 방향이지만, LB에서 뒤집힐 가능성은 남아 있다.
- **라이선스 (2차 평가)**: AT-ADD(비상업 학술 목적 한정, YouTube 원천 포함)와 MUSDB18-HQ(재배포 조건 불명확)의 위험이 있다. top-15가 우선이라 감수하기로 했다(`LICENSES.md`).
- **검증 한계**: E2의 80%가 AT-ADD다. E3는 real만 있어 한국어 fake 탐지는 측정하지 못한다. (정정 2026-09-28 18:05) 한국어 fake 음성은 0이 아니다.
  기존 bank에 MLAAD-ko 3,830, SpeechFake-ko 3,000, generated_music-ko 222행이 있다. 부족한 것은 생성기 계열의 다양성,
  전화 채널을 거친 fake, 음성 복제(VC), 한국어 fake 노래다.
- **선택 사항**: AT-ADD Track 1 eval(real·fake 각 1,000개)로 음성 회귀를 확인할 수 있다. HF 로그인이 다시 필요하고, 23시까지 끝나지 않으면 생략한다.
