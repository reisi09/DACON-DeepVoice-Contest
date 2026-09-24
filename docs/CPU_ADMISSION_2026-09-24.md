# 신규 데이터 CPU 검토와 학습 편입 — 2026-09-24

이 문서는 1차 편입 기록이다. 대회 규칙 재검토 후 MLAAD 146개를 추가한 [최신 roles-v6 기록](MLAAD_DECISION_2026-09-24.md)을 함께 확인한다. 아래 보류 판단은 당시 상태로 보존한다.

GPU 없이 **LibriSpeech 449개와 Music Arena 882개, 총 1,331개**를 학습 입력에 편입했다. 새 기준은 `roles-v5`다. 기존 42,947행과 합쳐 44,278행이며 새 자료는 모두 train이다. GPU 학습·특징 추출·공식 제출은 실행하지 않았다.

## 실제 판정과 남은 한계

기존 AST AudioSet safetensors를 오프라인 CPU에서 실행했다. 원본에서 만든 15초 학습 클립 1,380개를 각각 0–10초와 5–15초 구간으로 검사했다. 두 구간 모두 기존에 알려진 Speech 또는 Music 점수가 0.5 이상인 파일을 채택했다. 실제 추론은 636.07초 걸렸다. 원본 전체를 청취한 기록이 아니며, 사람이 청취한 파일 수는 0개다.

AST는 음성·음악 존재를 자동 선별하는 데만 썼다. 진위 정답은 배포자의 출처 기록을 유지했고, 미확인 보컬 존재·진위는 새 정답으로 만들지 않았다. 0.5는 실행 전에 정한 선별 기준이며 목표 도메인의 정확도나 교정된 확률을 보장하지 않는다. 자연스러움·발음·청감상 결함은 이 판정으로 확정하지 않는다. 기존 80개 수동 청취 표본에는 사람 판정을 꾸며 넣지 않았다.

| 원천 | 보관 원본 | 학습 클립 본처리 | 학습 편입 | 이번 학습분 제외 |
|---|---:|---:|---:|---:|
| LibriSpeech | 600 | 452 | 449 | 151 |
| Music Arena | 1,000 | 928 | 882 | 118 |
| Jamendo | 1,000 | 0 | 0 | 1,000 |
| MLAAD | 400 | 0 | 0 | 400 |
| 합계 | 3,000 | 1,380 | 1,331 | 1,669 |

표는 학습용 파생 클립의 본처리 집계다. 사전 CPU 동작·속도 확인에는 네 원천에서 원본 1개씩을 별도로 사용했다. 이 동작 확인이 Jamendo·MLAAD의 학습 허용을 뜻하지는 않는다.

LibriSpeech의 148개는 15초보다 짧아 제외했고 3개는 AST 기준에 못 미쳤다. Music Arena는 기존 기술 검토 표시 또는 생성기 식별 문제로 72개를 먼저 제외했고, 판정한 클립 중 46개가 AST 기준에 못 미쳤다. 기준 미달을 불량 음원의 확정 판정으로 해석하지 않는다. 원본과 제외 사유는 보존한다.

## 이용 조건 결정

- [LibriSpeech](https://www.openslr.org/12/)와 [Music Arena 공식 배포 음원](https://huggingface.co/datasets/music-arena/music-arena-dataset/blob/main/README.md)은 CC BY 4.0에 따라 출처·라이선스·변환 내역을 동봉하는 조건으로 이번 학습과 검증자 제공을 허용했다. Music Arena에서 실제 배포되지 않은 `sao`/`sao-small` 음원은 사용하지 않았다.
- [Jamendo 일반 약관](https://licensing.jamendo.com/en/legal/termsofuse)은 비상업 연구 TDM/ML/AI에도 사전 서면 허가를 요청한다. 개별 곡의 CC BY/BY-SA 기록은 보존하되 별도 조건이 해결되지 않아 이번 학습분에서 제외했다. 개별 CC 허락이 무효라고 판정한 것은 아니다. 외부 연락이나 새 동의는 하지 않았다.
- 사용자는 목적을 **대회 모델 제작용도**로 확인했다. [MLAAD 배포자](https://huggingface.co/datasets/mueller91/MLAAD/blob/main/README.md)는 비상업적 학술연구로 사용을 제한한다. 확인된 목적만으로 추가 범위의 충족을 확정하지 않아 신규 400개를 제외했다. NC 자체를 대회 금지로 해석하지 않는다. [대회 규칙](https://dacon.io/competitions/official/236749/overview/rules)과 [운영진의 NC 사용 답변](https://dacon.io/en/competitions/official/236749/talkboard/417857?category=&dtype=recent&fType=&page=1&ptype=)도 원천의 추가 조건까지 없애지는 않는다.

실행 전 결정은 `rights_decisions.json`, 사용자 목적 확인을 반영한 최종 해석은 `rights_resolution.json`에 따로 보존했다. 이 검토는 이번 신규 후보에 한정하며 기존 데이터 전체의 권리를 새로 승인한 기록은 아니다.

## 원본·라벨·역할 검증

원본을 덮어쓰지 않고 SHA-256과 디코딩 PCM을 대조했다. 모든 신규 클립은 mono 16 kHz PCM16 WAV, 15초로 맞췄다. 최대 진폭이 필요한 경우에만 줄였고 offset·gain·양자화 비율·원본/파생 해시를 기록했다. 측정한 최대 양자화 오차는 `1.52587890625e-05`로, 반 LSB와 float 허용 오차를 합친 `1.53587890625e-05` 이내였다. 출력 형식과 길이를 맞춰도 원래 codec이나 수집 원천의 흔적까지 사라지는 것은 아니다.

기존 42,947행의 오디오·라벨·역할·검토 component ID를 유지했다. 다만 기존 bank의 72행은 검토 manifest와 그룹 식별자가 달라 새 bank에서 맞췄다. 이 행의 기존 캐시가 있다면 행 식별자를 다시 검증해야 한다. 전체 특징 캐시가 확보됐다고 주장하지 않는다.

실제 `full_bank_features.load_rows`로 train/selection/calibration 33,528행과 development 9,842행을 읽었다. 원천 그룹의 역할 중복은 0개이고 신규 평가 행도 0개다. 제외한 1,669개는 bank에 없다. 실제 `head_labels` 경로에서도 LibriSpeech는 voice real, Music Arena는 music fake만 학습하며 반대 성분의 진위는 마스킹된다. 기존 사용자 수정 9개의 SHA-256도 그대로다.

| 역할 | 최종 행 수 |
|---|---:|
| train | 29,650 |
| selection | 2,435 |
| calibration | 2,073 |
| development | 10,120 |

새 통합 검증은 원본 변조, 양자화 오차, 평가 자료 유입, 권리 미허용 자료의 편입, 잘못된 모델·오디오 해시, 누락·잘린 판정 기록, 미확인 라벨의 잘못된 감독을 다뤘다. 실제 1,380개 모델 실행 기록은 별도로 보존한다. 테스트 통과나 자동 판정만으로 탐지 성능·LB 개선·사람의 청취 완료를 주장하지 않는다.

## 전달과 재현

코드 commit: `9c8063cd73391db816f4989685870025948ddc89`.

기존 `qwdkw / objvol-a5gxdg7er70i`의 `lb-improvement-20260924/datasets/admitted-v2/`로 archive와 영수증 두 파일을 업로드했다. 전체 재다운로드 후 크기·SHA-256이 모두 일치했다. `admitted-v2.tar.gz`는 608,398,316 bytes이며 명시한 1,363개 member의 내용 해시를 확인했다. SHA-256은 `9854dd539bf284a41f09020cf96928335de1adf40367afd59fd9ae6a6c5f8471`이다. 마지막 조회에서도 기존 workspace는 `stopped`였다. [전송 검증](evidence/lb_admission_20260924/transfers.json).

`audio/additional-v2/`에는 채택한 1,331개 파생 음원만 담는다. 기존 `audio/00/...`를 가진 동일한 입력 root에 추가해야 한다. 새 archive만으로 기존 전체 bank 오디오가 갖춰지는 것은 아니다. `README.txt`에 prepare → CPU review → finalize 명령과 배치 위치를 적었다. 새 의존성이나 GPU bootstrap은 추가하지 않았다.

근거: [최종 이용 조건](evidence/lb_admission_20260924/rights_resolution.json), [자동 판정 집계](evidence/lb_admission_20260924/automatic_review_summary.json), [실제 입력 통합 검증](evidence/lb_admission_20260924/integration_verification.json), [검증 범위](evidence/lb_admission_20260924/verification_scope.json), [출처 표시](evidence/lb_admission_20260924/THIRD_PARTY_NOTICES.md).
