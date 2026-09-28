# Round 3 최종 검토: SYNTHESIS_R3 1~5절 (claude_fresh_verifier_r3)

- 검토 대상: `SYNTHESIS_R3.md`, `codex_round11.md`, `fresh_r2.md`, `dataset_scout_r1.md`
- 추가 확인 방법: bank CSV에서 CPU로 가볍게 재집계했다.
- 표기
  - V = VERIFIED
  - I = INFERRED
  - U = UNKNOWN
- 1절(W), 2절(C), 3절(게이트)에는 차단 사유가 없다. 이견은 모두 4절 데이터 추가에 있다.

## (a) BLOCKING

### A1. D1 리믹스 설계가 분리 누설(bleed)로 목표 셀의 라벨을 오염시킨다

**문제**
- 합성안은 다음 두 가지를 제안한다.
  - realV+fakeM의 AI 반주 = "SONICS/HAIM/Echoes fake의 반주 stem"
  - fakeV+realM의 AI 보컬 = "SONICS/HAIM의 AI 보컬 stem"
- SONICS와 HAIM의 fake는 대부분 보컬이 있는 노래이다(V). 예: sonics VP=1 743/800, existing:HAIM VP=1 308.
- HTDemucs로 나눈 반주 stem에는 AI 보컬이, 보컬 stem에는 AI 반주가 남는다(I).
- 그 결과 두 종류의 파일이 생긴다.
  - "VOICE real" 파일인데 AI 보컬 잔향이 들어 있다.
  - "MUSIC real" 파일인데 AI 반주 잔향이 들어 있다.
- 바로 우리가 가르치려는 셀에서 head가 AI 잔향을 무시하도록 학습하게 된다.
- 기존 bank의 constructed_mix와 constructed_song 1,900행은 **음악 부모가 모두 VP=0(반주만 있는 곡)** 이었다(V, parent_ids를 bank와 대조). 새 설계는 이 안전한 관행에서 벗어난다.

**최소 수정**
- AI 반주: VP=0이고 fake이며 라이선스가 기록된 행만 쓴다.
  - existing:HAIM 449, haim fake, generated_music 179, sonics 44, echoes 199(A2 규칙 적용)
- real 반주: VP=0 real 행만 쓴다.
  - openmic 1,363, musan_music 397
- fake 가창: 깨끗한 무반주 AI 가창을 쓴다. 팀이 이미 쓰는 `singmos_pro`의 svs/svc가 여기에 해당하며, CC BY 4.0이다(V: `train_backbone_heads.extra_voice_items`, scout).
  - SingMOS-Pro gt(real)도 같은 비율로 real 보컬 공급원에 넣는다. 그래야 언어·소스 지름길이 생기지 않는다.
- 분리한 AI 보컬을 꼭 써야 한다면: 그 행들은 MUSIC 라벨을 미상(mask=0)으로 두고 VOICE head에만 쓴다.

### A2. 라이선스 조합 규칙이 빠져 있다 (2라운드에서 "모든 학습 파일"을 제출해야 함)

**문제**
- echoes는 CC BY-SA 4.0이다(V). SA의 파생물은 BY-SA로만 배포할 수 있다.
- 따라서 NC 공급원과 섞은 믹스는 어떤 라이선스로도 적법하게 공유할 수 없다(I, CC 호환표 해석이며 법률 자문이 아님). 해당하는 NC 공급원은 csd(BY-NC-SA), SONICS와 HAIM(BY-NC), GTSinger(BY-NC-SA)이다.
- 합성안이 D1 real 반주로 든 "FMA"(existing:FMA)는 bank의 license 칸이 **비어 있다**(V).

**최소 수정**
- echoes는 BY 계열 공급원(vocalset, openmic, musan, singmos_pro)하고만 섞는다.
- license 칸이 빈 `existing:*` 행(FMA, Jamendo, FakeMusicCaps, 레거시 Echoes)과 ND 자료(CtrSVDD)는 D1 공급원에서 제외한다.

**참고 (V, 이미 존재하는 문제)**
- 현재 bank의 constructed_mix에 이미 "CC-BY-NC-4.0 + CC-BY-SA-4.0" 조합이 **111행** 있다.
- 오늘 LB에는 영향이 없지만 2라운드 전에 사용자가 판단해야 한다.

### A3. AI-Hub 변형은 2라운드 규정과 충돌할 수 있다 (조건부 BLOCKING)

**문제**
- 브리핑 §1에는 "2라운드에 학습 데이터 파일 전부 제출"이 요구된다(V).
- 브리핑 §10에는 AI-Hub의 "제3자 제공 제한"이 있다(V).
- 2026-09-21 운영진 허용 답변(417844)이 2라운드 파일 제출까지 다루는지는 U이다.

**최소 수정**
- AI-Hub로 학습한 패키지는 이 점을 확인한 뒤에만 LB에 낸다. 확인 방법은 둘 중 하나이다.
  - 답변 원문
  - 운영진에게 AI-Hub 자료는 "데이터셋 ID 제출로 대체 가능"한지 확인
- 확인이 안 되면 AI-Hub는 국내 audit 용도로만 쓴다.

### A4. CtrSVDD를 추가할 경우의 조건 (추가할 때만 해당)

**문제**
- 일본어 bona fide 4종은 zip에 없다(V, scout).
- 그대로 넣으면 "일본어 가창 = fake"라는 단일 클래스 셀이 새로 생긴다.

**최소 수정**
- bona fide 짝이 실제로 들어가지 않는 원천의 fake 행은 제외한다.
- ACE-Studio bona fide 제외(합성안에 이미 있음)에 더해, **중국어 bona fide와 중국어 원천 fake만** 쓴다.

## (b) 필수 누락 사항 (차단은 아니지만 빠지면 위험)

1. **일정.** "모든 패키지 08:00까지 검증"은 적응형 슬롯 계획과 맞지 않는다.
   - 2번 슬롯의 분기 후보(W+D1, C+D1, S7+D1)는 1번 슬롯 결과가 나오는 약 01:30에 즉시 필요하다.
   - **1번 슬롯 패키지와 2번 슬롯 분기 패키지 전부를 23:30까지 검증**해야 한다. 08:00은 3번 슬롯의 선택 후보에만 적용한다.
   - 업로드(속도 U, 3.3 MB/s라면 약 49분)를 고려하면 마지막 업로드는 09:00 전에 시작해야 한다.
   - 슬롯 2의 규칙 구간이 겹친다(+0.001 이상과 ±0.003). 다음처럼 확정한다.
     - +0.001 이상: W+D1
     - (-0.003, +0.001): 다른 쪽 가중치 + D1
     - -0.003 이하: S7 가중치 + D1
2. **길이 지름길 점검.** duration 배율 clip을 [0.1, 10]에서 [0.5, 2]로 좁히면 길이 보정이 약해진다.
   - 과거에 길이 지름길이 실제로 있었다(V ledger: "dur-only OOF AUC .894").
   - 학습 전에 (head, 층)별로 **가중 적용 후 길이만으로 계산한 AUC**를 로그로 남긴다.
   - 기준값: 0.60 또는 S7 값 중 큰 쪽. 이를 넘으면 그 층은 clip [0.1, 10]을 쓴다고 **사전등록**한다. 모델 결과를 보기 전에 정하는 조건부 규칙이다.
3. **FILE 표.** FILE 가중치 0/9/1을 고정하더라도 **FILE 후보(max, noisy_or, max_nopresence)의 quantile 표는 val로 다시 맞춰야** 한다. head가 바뀌었기 때문이다.
   - 현재 `file_ensemble.fit`은 표와 가중치를 함께 맞춘다(V). 가중치를 고정하는 호출 경로를 하나 추가해야 한다.
4. **D1과 val.** D1 행은 val에 넣지 않는다. val 공급원으로 만든 리믹스는 제외하거나 진단용으로만 쓴다. 그래야 early stopping과 CDF 표가 W와 비교 가능하다.
5. **2라운드 재현성.**
   - D1 믹스 WAV 전부와 레시피(부모 ID, 이득, 오프셋, 코덱, seed, 코드 revision)를 보존한다. 생성물 전부와 재현 정보 제출이 요구된다(scout가 인용한 운영진 답변, 원문은 미확인).
   - bank에서 license 칸이 빈 **3,675행**(V)의 출처와 라이선스를 2라운드 전에 채운다. 모두 music route 레거시 `existing:*`이며, FakeMusicCaps 2,000, Jamendo 400, FMA 455, 레거시 Echoes·suno·udio·constructed_music이다.
