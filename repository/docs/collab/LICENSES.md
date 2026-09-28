# 새 데이터 라이선스 대장 (2026-09-28 확인)

DACON 규칙 2-1: 누구나 접근 가능한 공개 자원이어야 하고, 최소한 비영리 목적 사용이 허용되어야 한다.
규칙 3과 토크보드 417198: 2차 평가 때 학습에 쓴 데이터 파일 일체와 재현 정보를 제출해야 한다. 생성한 오디오도 포함한다.
아래 표는 각 데이터의 공식 라이선스와, 2차 평가 때 파일을 주최 측에 넘길 수 있는지를 정리한 것이다.

| 데이터 (학습 행) | 라이선스 (출처) | 2차 평가 제출 가능성 | 위험 |
|---|---|---|---|
| AI-OpenBMAT | CC BY 4.0 (Zenodo 20031232) | 가능 (출처 표기) | 낮음 |
| SD-RR | CC BY-SA 4.0 (Zenodo 22169646), 곡별 Jamendo CC. ND 곡은 행 생성 때 제외 | 가능 (출처 표기, SA·NC 조건 유지) | 낮음 |
| HAIM B3·B4 | MIT (ACE-Step 1.5 파생, HAIM DATASET_DESCRIPTION §8) | 가능 | 낮음 |
| HAIM B6 | CC BY-NC 4.0. 저작권 있는 K-pop 가사로 생성했으며 학술 연구 목적이라는 단서가 붙음 | 비상업이면 가능. 가사 저작권 단서 있음 | 중간 |
| GTSinger 한국어 | CC BY-NC-SA 4.0 + 추가 약정 (dataset_license.md) | 비상업이면 가능 | 낮음~중간 |
| MUSDB18-HQ | Zenodo "Other (Non-Commercial)". MedleyDB 46곡은 CC BY-NC-SA 4.0, Easton Ellises 2곡은 CC BY-NC-SA 3.0, 나머지 곡은 조건 미확인 | 불명확 | 중간 |
| AT-ADD Track 2 | AT-ADD Dataset Access Agreement: 비상업 학술 연구·교육 용도만 허용. 재배포도 같은 목적으로 약정서를 첨부할 때만 허용하며, 원천 조건이 우선한다. 원천 코퍼스에 YouTube 기반 MusicCaps와 FortisAVQA가 있음 | 대회가 학술 연구·교육에 해당하는지 불명확. YouTube 원천 음원은 재배포가 사실상 불가 | 높음 |
| D1 재혼합 | MUSDB, GTSinger, AT-ADD에서 파생 | 원천 조건을 그대로 이어받음 | 중간~높음 |

AT-ADD 재검토 (2026-09-28 18:00 KST, AT-ADD Dataset Access Agreement v1.0, 2026-08-16 원문 대조): 사용 가능으로 판단한다.
- §1 허용 용도는 "audio deepfake detection, multimedia forensics 등 분야의 비상업 학술 연구·교육"이다. 이 대회 과제와 같은 분야다.
  AT-ADD 자체도 같은 약정으로 ACM MM 2026 챌린지를 운영하며 업계 참가자도 받는다.
- 대회 규칙 2-1: HF 게이트는 동의 후 자동 승인이다. 가입·약정형인 AI-Hub를 DACON이 허용했다(토크 417844).
  NC 사전학습 모델(417857, 상금이 있다는 이유만으로 제외하지 않음)과 CC BY-NC-SA 데이터(417212)도 허용됐다.
- 지켜야 할 의무:
  - §6 공개 결과에는 AT-ADD 논문을 인용한다. 인용: Y. Xie et al., "AT-ADD: All-Type Audio Deepfake Detection Challenge
    Evaluation Plan", arXiv:2604.08184, 2026. 데이터: AT-ADD Track 2 (https://huggingface.co/datasets/xieyuankun/AT-ADD-Track2).
  - §3 재배포(2차 평가 제출 포함)는 비상업 학술·교육 목적으로만 가능하다. 약정서를 첨부하고, 받는 쪽에 조건을 알린다.
    원 조건이 재배포를 금지하는 원천(YouTube 기반 MusicCaps, FortisAVQA 등)은 파일 대신 ID와 공식 경로로 낸다.
  - §5 접근 통제된 저장소에만 둔다. 오디오와 행 목록은 git에 올리지 않는다(`rows/`는 gitignore).
  - §7 접근 자격은 개인용이다. 공유하지 않는다. 작업에 쓴 HF 토큰은 폐기한다.
  - §3 학습한 가중치는 원천 조건이 허락하고 원 음성을 복원할 수 없을 때만 공유할 수 있다. 우리 head는 고정 인코더
    특징 위의 분류기라 오디오를 복원하지 못한다.
- 남은 위험: 수상하면 저작권이 주최 측에 독점 귀속되고(대회 규정 "수상 산출물에 대한 독점권리"), 수상자는 제한 없는
  라이선스 부여 권리를 표명해야 한다(규정 10(b)). 이때 비상업 학술 조건과 충돌할 수 있다. 규정 11은 라이선스 충돌을
  1주 안에 해결하도록 요구할 수 있다. 대응은 아래 3번(해당 행을 뺀 재학습)이다.

결정 (2026-09-28): top-15 진입이 우선이라 위험을 감수하고 AT-ADD를 포함한 모델을 제출한다(사용자 결정).
2차 평가 대상이 되면 다음 순서로 대응한다.
1. 원천 목록과 약정서를 첨부한다.
2. 재배포가 불가한 원천(MusicCaps, FortisAVQA)은 파일 대신 공식 출처와 파일 목록으로 제출할 수 있는지 주최 측에 문의한다.
3. 필요하면 해당 행을 빼고 재학습한 모델로 재현성을 입증한다.
