**제 입장은 “S7의 최종 가중치를 바로잡는 실험을 우선하고, 데이터 확보·추출은 병렬로 진행한다”입니다.** 실제 가중치 표는 재가중 실험의 근거를 강화합니다. 다만 그것이 S8 하락의 원인을 입증하거나, 재가중의 LB 개선을 보장하지는 않습니다.

**1. 다른 리뷰에서 수정해야 할 주장**

**[VERIFIED] 실제 표의 MLAAD+ASVspoof 비중은 fake voice loss의 `56.9+20.5+3.6+2.3=83.3%`입니다.** 초기의 `93–99% speech`, `91–97% MLAAD+ASVspoof`는 다른 모집단·보정 단계의 수치입니다. 실제 singing 비중은 fake 약 **14.7%**, real **41.1%**로 읽어야 합니다. `Korean(all)`의 weighted fake share `0.244`를 한국어 **speech만의** 수치로 인용해서도 안 됩니다. 실제 가중치 원표는 제공된 검증 결과로 채택했으며, 개별 cache·weight가 없어 그 계산 전체를 독립 재현하지는 못했습니다. ([actual_weights:3](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/claude_main_r1b_actual_weights.md:3))

주요 반박은 다음과 같습니다.

| 리뷰·위치 | 수정할 주장 |
|---|---|
| `claude_fresh_verifier_r1.md:258–259`, actual 표 20행 | **Kish 11.8/19.0은 가중치 집중도이지 실제 독립 학습 그룹 수가 아닙니다.** CSV에서 P1 music은 fake **3,603**, real **3,063** lineage groups입니다. `artist:FMA:1041`도 여러 legacy source가 연결된 component이지, 모두 한 아티스트의 동질적인 음원이라는 뜻은 아닙니다. |
| Claude fresh 302–309행 | **78%는 반올림된 단일-threshold MUSIC FP 감소분의 비율**입니다. ADS 개선의 78%도, remix 흔적을 학습했다는 증거도 아닙니다. unseen EER에는 공통 real 행이 들어가므로 개선을 HAIM 두 생성기에 귀속할 수도 없습니다. |
| Claude fresh 290–291, 316행 | 격차가 대체로 일정하다고 적응적 과적합을 배제할 수 없습니다. **“약 2σ”도 실제 유의성 검정이 아닙니다.** 알려지지 않은 LB paired SE를 DB와 같다고 둔 가정입니다. |
| Claude fresh 328, 341–345행 | DB의 FILE 효과 `≈0.0013`은 LB 효과의 상한이 아닙니다. 같은 가중치라도 raw-mix 특징이 기존 편향을 더 이용할 수 있습니다. 대상 도메인이 DB에 없으면 개선을 **직접 측정할 수 없는 것**이지, 다른 DB 점수에 반드시 중립인 것은 아닙니다. |
| Claude fresh 213, 324, 416행 | MUSIC의 직접 손실이 크다고 FILE 손실 대부분까지 MUSIC 책임으로 돌릴 수 없습니다. CPS AUC `≈0.998`도 확률 calibration을 보증하지 않습니다. FILE을 고정하고 VOICE·MUSIC을 함께 바꾸면 두 성분의 **합산 효과**만 분리됩니다. |
| Claude fresh 368–375행 | **cap 뒤에 duration 보정을 적용하면 cap이 깨집니다.** 따라서 `loss_weights` 한 곳만 바꾸고 호출부를 유지한다는 구현안은 최종 cap을 보장하지 못합니다. |
| Claude fresh 421행 | 비선형 EER에서 “한 source가 이득의 50% 이하”는 기여도 정의가 없습니다. jackknife 감소분을 합산 기여도로 해석하면 안 됩니다. |
| `fresh_r1.md:65` | fake singing만 25%로 정하면 real singing 41%와의 차이가 남습니다. **양쪽 목표 질량과 최종 cap**을 함께 정해야 합니다. 나머지 핵심 진단에는 대체로 동의합니다. |

[Claude fresh 원문](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/claude_fresh_verifier_r1.md:258), [fresh_r1.md:65](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/fresh_r1.md:65)

**제 이전 권고도 수정합니다.** 기존 가중치와 50:50으로 섞으면, 28.8%였던 그룹이 최소 **14.4%** 남습니다. 이번 집중도를 해결할 주안으로 부족합니다. 모든 리뷰의 “deneb-kr 위치 UNKNOWN”은 이제 오래된 정보입니다. **kr-west는 확인됐고**, 남은 것은 사용자 측 처리 승인·실제 가용 자원·별도 storage 전송입니다. ([브리핑:114](/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad/codex_debate/FULL_BRIEFING_20260928.md:114))

**2. 사전등록할 재가중 수식**

아래는 **[INFERRED: 실험 설계]**입니다. LB 분포를 추정한 값이 아닙니다.

**주안 `W`: DF voice·AD voice·WavLM member·music head를 모두 재학습**합니다. S7 encoder·입력·L·3 seeds는 유지합니다. DF head 복사나 WavLM 재사용으로 일부 member에 옛 가중치를 남기지 않습니다. Presence는 고정하고, fusion 계수 `1/.5/1`, FILE 계수 `0/9/1`도 고정합니다. 필요한 CDF만 같은 val 절차로 다시 적합합니다.

실제로 각 head에 들어가는 필터 완료 행을 대상으로:

- **Voice strata:** `singing`—무반주·반주 포함을 합침, `한국어 speech`, `그 밖의 speech`—언어 미상 포함. source·parent provenance로 매핑을 먼저 고정합니다.
- 각 클래스 안 목표 질량은 각각 **`0.25 / 0.225 / 0.525`**입니다. 즉 singing은 양쪽 클래스 모두 25%, speech 안 한국어는 양쪽 모두 30%입니다.
- **Music은 전체 real/fake를 50:50으로만 맞춥니다.** 보컬 존재·constructed 여부는 검증 층으로 사용합니다. 없는 fake a cappella나 real Korean accompanied-song cell에 억지로 질량을 배정하지 않습니다.
- `speech over music`은 별도 진단 대상으로 유지합니다. 적은 real 행을 독립적으로 1:1 균형화하면 row cap과 충돌할 수 있습니다.

행 \(i\), 클래스 \(y\), stratum \(s\), cell \(c=(y,s,\text{source-family},\text{generator})\)에 대해:

\[
b_i=\max(n_{c_i},20)^{-1/2}.
\]

**행당** inverse-square-root입니다. legacy source alias만 사전에 합치고, generator 이름을 결과에 맞춰 재분류하지 않습니다.

훈련 길이의 10개 log-duration quantile bin에서 \(b\)로 계산한 클래스별 분포를 \(H_y\), 두 클래스의 평균을 \(H_{\rm pool}\)이라 두고:

\[
d_i=\operatorname{clip}\!\left(
\frac{H_{\rm pool}(k_i)}{H_{y_i}(k_i)},\,0.5,\,2
\right),\qquad a_i=b_i d_i .
\]

그다음 클래스별로 정규화한 \(a\)에서 **KL 거리를 최소화하는 최종 가중치 \(p\)**를 구합니다.

\[
\sum_{i:y_i=y}p_i=1,\quad
p_i\le \frac{5}{N_y},\quad
\sum_{i:y_i=y,\ g_i=g}p_i\le0.02.
\]

Voice에는 위 stratum 목표 질량도 제약으로 넣습니다. optimizer에는 \(w_i=Np_i/2\)를 전달합니다. **이 뒤에는 duration 보정을 다시 적용하지 않습니다.** 따라서 cap은 각 클래스 기준으로 최종 row **균등 가중치의 5배**, 최종 lineage group **2%**입니다.

훈련 전 각 head의 실제 admitted rows에서 제약의 가능 여부와 최종 합계를 확인합니다. 불가능하면 결과를 본 뒤 cap을 완화하지 않습니다. **대안은 딱 하나: music에만 같은 수식을 적용하고 voice 세 member는 S7 그대로 유지**합니다. 주안·대안은 val에서 선택하고 audit에는 선택한 하나만 보냅니다.

**3. 제출 전 검증 gate**

**[INFERRED] DB를 버리지 않고, 개발용 안전장치와 새 독립 평가의 역할을 나누겠습니다.** 다음 수치는 결과를 보기 전에 고정할 운영 기준입니다.

- **동일 DB 비교:** 같은 IDs·장치·seeds의 S7 control 대비 RT `ΔADS≥+0.002`, group-bootstrap 양의 차이 비율 `≥0.90`, unseen-generator `Δ≥−0.002`.
- **자연 출처:** `constructed_*` 제외만으로 끝내지 않습니다. legacy remix와 HAIM edit을 별도 표시한 provenance whitelist를 고정하고, 양 클래스가 있는 자연 출처 부분집합에서 `ΔADS≥0`.
- **Source jackknife:** source-family 하나씩 제외해 재계산한 유효 비교의 최솟값 `≥−0.003`. 클래스가 사라지는 경우는 제외 사유와 함께 보고합니다. 이것은 재학습 CV나 독립 증거가 아닙니다.
- **구성 변화:** 원래 구성, speech×2, singing×2, instrumental×2, FLEURS×0.25를 고정합니다. 각 component의 real/fake 안에서 가중치를 정규화하고 **weighted ROC/EER를 다시 계산**합니다. 모든 시나리오에서 `ΔADS≥−0.003`. ADS 계수 `.5/.2/.3`은 유지합니다.
- **새 audit:** 후보 설정 전에 원곡·가수·donor로 분리하고 P1에도 넣지 않습니다. 실제 패키지 기준 `ΔADS≥0`, paired group-bootstrap의 **90% 구간 하한 ≥−0.01**, 미리 지정한 교차 조합의 dev-threshold FPR/FNR 증가 **≤3%p**를 요구합니다. 표본·클래스가 부족한 항목은 통과가 아니라 **UNKNOWN**입니다.

S8을 이 진단에 적용하는 것은 사후 분석입니다. **S8이 계속 통과하더라도 gate를 다시 손보지 않습니다.** 새 audit이 없거나 판단 불가능하면 “독립 검증 통과”라는 이름을 붙이지 않고, 제출하더라도 탐색 후보로 명시해야 합니다.

**4. 22:00까지 가능한 데이터 추가**

**[ESTIMATE] 접근 확인·분할·특징 추출은 재가중 결과를 기다리지 말고 시작할 가치가 있습니다.** 학습 후보 `W+D`는 고정된 `W`에 데이터만 추가해 효과를 분리합니다.

| 자료 | 이번 상한·처리 |
|---|---|
| **AI-Hub 465** | 보유 **311개**를 우선 검사. 가수·곡별로 나눠 **200–250개 안팎**을 학습용으로, 나머지를 dev/audit용으로 확보하되 실제 그룹 구조에 따라 수량 조정 |
| **AI-Hub 473** | 접근이 일찍 되면 **100–200개, 20명 이상**. 465와 원곡 중복 확인 |
| **새 SVS/SVC** | 권리·원본이 확인된 미편입 자료 **real/fake 각 200–300개**. CtrSVDD 등의 접근·가공·2차 제출 조건은 사용자가 확인 |
| **real 완성곡** | 허용된 MUSDB 계열 등 **50–100곡**, 곡당 최대 2구간. 기존 Medley/FMA 계보와 중복 제거 |
| **통제 혼합** | **100 donor 묶음×4조합=400개**부터. 같은 처리·codec·SNR 분포를 적용하고 실제 runtime으로 재분리·추출. AI 생성곡에서 얻은 fake vocal remix를 실제 SVC cover와 동일시하지 않음 |
| **71767 / 71314** | 각각 **50–100개 loop / 준비된 방송 clip 최대 100개**만 보조적으로 사용. 대형 archive 확보는 핵심 일정에서 제외 |

**국내 위치가 확인됐다고 기존 EU cache를 곧바로 쓸 수 있는 것은 아닙니다.** deneb-kr에 모델·코드와 필요한 비AI-Hub feature cache를 **EU→KR**로 옮겨야 합니다. AI-Hub 원음·stem·feature·파생물과 이를 이용한 학습은 KR에 유지합니다.

**[ESTIMATE] 18:00까지 국내 환경·필요 cache 준비, 20:00까지 특징 추출, 22:00까지 head 결과 확보**를 목표로 둡니다. 첫 64개로 전체 pipeline 시간을 측정하십시오. cache 전송이 늦으면 AI-Hub는 우선 국내 audit에 사용하고, 학습 후보는 준비된 비AI-Hub 자료로 제한합니다. 접근 가능성과 처리량은 아직 **UNKNOWN**이므로 22:00 완료를 보장하지는 않습니다.

**5. 세 슬롯**

**[INFERRED] 이번에는 점수 개선을 겨냥한 `W`를 먼저 제출하겠습니다.** 새로운 실제 가중치 증거가 그 우선순위를 높였습니다.

1. **00:00 — `S7-W`**: 위 절차에서 선택한 주안 또는 music-only 대안. 준비되지 않으면 **S8의 FILE 계수만 `0/9/1`로 바꾼 control**을 먼저 제출합니다.
2. **다음 — `W+D`**: 데이터 추가 후보가 새 audit까지 통과했을 때. 실패·미완료이면 아직 사용하지 않은 FILE control.
3. **마지막 — FILE control 또는 검증된 fallback**: 앞에서 FILE control을 사용했다면, 준비된 `S7-noL`, 그다음 기존 lock을 통과한 완성 S9 순으로 탐색 후보를 고려합니다. 동일 후보를 중복 제출하지 않습니다.

모든 후보는 **08:00까지 패키지 검증을 마칩니다.** 첫 LB를 보고 exponent `0.75`나 cap `1%`를 새로 탐색하는 안에는 반대합니다. **S7은 best-ever로 보존하고, 세 장은 재가중·데이터 추가·FILE 변경이라는 서로 다른 가설에 씁니다.**

