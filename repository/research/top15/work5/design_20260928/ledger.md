# DACON 236749 실험/제출 원장 (ledger.md)

작성자: Claude research session 1 (읽기 전용 조사). 작성 시각 기준: 이번 세션 작업 시점(패킷 MANIFEST.md 작성 시각 2026-09-28 21:10 KST 이후).
동반 파일: `ledger.csv` (전체 45개 실험/제출 항목, 컬럼별 근거·라벨 포함).

라벨 정의(패킷 MANIFEST.md/감사 문서와 동일한 기준 사용):
**VERIFIED** = 이번 세션에 내가 직접 파일을 열람/파싱하거나 명령을 실행(재해시, git log 등)해서 확인 / **REPORTED** = 문서·로그·사용자 보고로만 확인, 이번 세션에 재실행하지 않음 / **HYPOTHESIS** / **ESTIMATE** / **UNKNOWN** / **NOT RUN**.

모든 리더보드(LB) 점수는 궁극적으로 사용자가 DACON 페이지를 읽고 보고한 값이며(REPORTED), 이번 세션은 네트워크 접근이 금지되어 있어 DACON을 직접 재조회하지 않았다. 단, sha256 재해시·receipt/verify.log/qa_verification.json 직접 열람·git log 실행 등은 이번 세션에 내가 직접 수행했으므로 VERIFIED로 표기한다.

---

## 1. 핵심 표 (제출된 모델 우선)

### 1.1 DACON 리더보드에 실제 제출된 모델 (시간순, package_sha256은 앞 16자)

| id | 날짜(KST) | sha256(앞16자) | LB ADS | LB total | 비고/결정 |
|---|---|---|---|---|---|
| G0_baseline | 09-13 | UNKNOWN | ~0.6578(ESTIMATE) | 0.6909110688 | 공식 베이스라인 참조값(REPORTED, claim-ledger lane1:L1-F3으로 교차 확인) |
| AST_V2_ep17_full1029 | 09-12/13 | UNKNOWN | - | 0.6453405503 / 0.6174819841(재제출) | invalid-protocol |
| V5_B1B2_experts | 09-14 | UNKNOWN | - | 0.6776485873(1차 오류) | full-FT 실험, 실패 |
| V6_V7_C3 | 09-15 | UNKNOWN | - | 0.6727822646 / 0.6843536931 | 실패, C08로 대체 |
| C08_compact_temporal | 09-15/16 | 0c3c3289(8자만, 미재검증) | ~0.7028(ESTIMATE) | 0.7314822646 | 첫 유의미한 LB 상승 |
| M330_MLP_full1206 | 09-16 | 1f6cfd6c(8자만) | ~0.7119 | 0.7396465503 | - |
| FullFT8_S2EAT | 09-16 | UNKNOWN | ~0.5988 | 0.6378679788 | 전체 미세조정 최악 사례 |
| Redesign_AB_B_full1696 | 09-17 | 1fda295b(8자만) | - | 0.6837822646 | 패키지-점수 연결 **추정**(association inferred) |
| MERT12_full1322 | 09-18 | e7bebbded6fef065...(VERIFIED, 이번 세션 직접 열람) | ~0.7211 | 0.7479394074 | - |
| Expanded_MIL_3659 | 09-18 | 7c7b0c96(8자만) | ~0.7322 | 0.7579036931 | 패키지-점수 링크 **미확인**(원 출처 표기) |
| Balanced7144_MIL | 09-19 | 665cb1b7(8자만) | ~0.7300 | 0.7559347884 | - |
| DataV4_DurationBalanced | 09-23/24 | 05ae786864dc282d...(VERIFIED) | 0.7683174603 | 0.7912534339 | data v4 최초 제출, 최대 단일 상승(+0.036 ADS) |
| E08_Partial5952_Top30 | 09-24 | fc23d780(8자만) | 0.7064761905 | (미기록) | invalid-protocol, 잘못된 기준선과 비교 |
| PANNs_R3_presence_adapt | 09-25/26 | 원본5717f10b.../fix1 6efcb018...(둘 다 VERIFIED) | 0.7674603175 | 0.7901426243 | DurationBalanced 대비 회귀; **점수-아티팩트 연결이 스크린샷 기반 추정**(§2 참고) |
| P1_Refit | 09-26/27 | 1b9ba4411b87c8fc...(VERIFIED) | 0.7700396825 | 0.7928034339 | DACON submission_id 1585448(claim-ledger lane1:L1-F1) |
| S2a_AIHubPhone | 09-27 | c7c8b1c83d242df9(16자, REPORTED만; 폴더 미열람) | 0.7688571429 | 0.7917391481 | P1 대비 -0.0012, 전화채널 데이터는 LB 레버 아님 |
| S3_Backbones | 09-27 | 062fc236bf0d1503...(VERIFIED) | 0.7853015873 | ~0.8065(ESTIMATE) | S6 이전까지 LB 최고 |
| S6_S5LabelCompletion | 09-28 ~02:20 | 7429b1830aaf0333...(VERIFIED) | 0.8220079365 | - | - |
| **S7_SSLAMWavLM** | 09-28 ~04:20 | **5239c912ccbee8c2...**(VERIFIED, **이번 세션 재해시 완료**) | **0.8373968254 (현재 LB 최고)** | ~0.8535(ESTIMATE) | 현재 incumbent |
| S8_MixSSLAM | 09-28 ~07:15 | b0e8744225e9252e...(VERIFIED) | 0.8339682540 | - | S7 대비 -0.0034, 로컬 예측(+0.0043)과 반대 방향 |

### 1.2 빌드·검증됐으나 리더보드에 제출되지 않은 후보

| id | sha256(앞16자) | 로컬 근거 | 제출 여부 근거 |
|---|---|---|---|
| Existing9504_MIL | b19b88de1b23ccb8...(VERIFIED — 모든 DataV4 zip의 base_archive_sha256과 동일) | 모든 DataV4 submit.json에 base_archive_sha256으로 내장 확인 | README "No DACON submission was made" + claim-ledger lane2:L2-F13 이중 확인 |
| S4_MLPHeads | beeda9a6fa785938...(VERIFIED) | CPU QA 108.5s/10파일 PASS | LB 어디에도 없음(UNKNOWN — "대기 또는 미제출") |
| S5_EATMusic | cf17eec9637b15e4...(VERIFIED) | CPU QA 225.8s/10파일 PASS; RT held-out ADS 0.89673 재현 확인(work4/AUDIT.md) | docs/collab §2 표에서 누락, 감사문서 "미제출(REPORTED)" |
| W_Reweight | a17442e6f3e2a830...(VERIFIED) | CPU+parity PASS, VESSL CUDA(REPORTED, memory-notes만) | docs/collab §2 "미제출" |
| S8_v1_twoSSLAM_notsubmitted | 337f172f37872746...(VERIFIED, S8 본채와 다른 sha 확인) | 로컬 검증 파일 없음 | memory-notes: "2nd SSLAM instance was NOT submitted" |

### 1.3 2026-09-29 제출 예정 (MANIFEST.md §3, 21:10 KST 기준 — 가장 최신 계획)

| id | sha256(앞16자) | 슬롯 | E1 ADS(vs s7_ref) | E2 ADS(vs s7_ref) |
|---|---|---|---|---|
| s7_new | fcdd86913360e4dc...(VERIFIED) | 슬롯1 | 0.91540 (+0.0064 [+0.0013,+0.0111]) | 0.92185 (+0.0285) |
| W_newB | 8d3b62d3ec80a84c...(VERIFIED) | 슬롯2 | 0.91221 (+0.0032) | 0.94031 (+0.0470, newval을 학습에 사용했으므로 순환적) |
| W_new | 58e7e3bd460cde93...(VERIFIED) | 슬롯3 또는 대안(§2 불일치 참고) | 0.91231 (+0.0033) | 0.93842 (+0.0451) |
| s7_new_ob | e4205dcecbc32989...(VERIFIED) | 조건부(“CUDA ok면”) | (별도 미분리) | (별도 미분리) |
| broadcast_lowSNR_candidate | 아직 패키지 없음 | 진행중, 결정 23:00 KST | NOT RUN | NOT RUN |

---

## 2. (a) 검증된 최고 공식 점수와 그 근거

**S7 (`DataV4_S7_SSLAMWavLM`)**: LB ADS **0.8373968254**, total ~0.8535(ESTIMATE, MANIFEST.md §2 수치 기반 계산).
근거: `MANIFEST.md` §2(사용자가 DACON 페이지에서 옮겨 적은 값, REPORTED) + `docs/collab/README.md` §2(동일 값, sha256 앞16자 `5239c912ccbee8c2` 일치, 역시 REPORTED — 두 문서 모두 궁극적으로 같은 사용자 관찰에서 나온 값이라 상호 독립적 검증은 아님) + `research/top15/claim-ledger.jsonl`의 `orchestrator:S3-LB` 항목과 `lane1:L1-F1`(2026-09-27 13:33:40 KST에 DACON 리더보드 JSON을 curl로 직접 가져와 확인한 기록, 스크린샷보다 강한 근거지만 이번 세션이 재확인한 것은 아님)이 인접 시점 값들을 뒷받침.
이번 세션이 이 점수 자체를 재조회하지는 않았으므로 최종 라벨은 **REPORTED**이다.

## 3. (b) 해당 제출과 연결되는 로컬 아티팩트와 byte identity

로컬 파일: `submissions/DataV4_S7_SSLAMWavLM/submit.zip` (프로젝트 루트).
이번 세션에서 직접 재해시(`shasum -a 256`, 약 40초 소요)한 결과:
```
5239c912ccbee8c20bef48aa7434cbedf15a8768a1f5ca2cff3a9949cf59a1fa  submit.zip
```
이는 `receipts/DataV4_S7_SSLAMWavLM/submit.json`(패킷)의 `"sha256"` 필드와 **완전히 일치**하며, 파일 크기(9,735,068,819바이트)도 `submit.json`의 `"bytes"` 필드와 일치한다. **byte identity는 VERIFIED** (이번 세션 내 1차 근거).

다만 두 가지 한계가 있다:
1. `verify.log`/`qa_verification.json`(둘 다 직접 열람, VERIFIED)의 `official_l4_validated` 필드는 **false**다. 이는 "이 zip이 실제로 DACON 리더보드에서 채점됐는가"와는 무관한, 로컬 L4/CUDA 에뮬레이션 검증 여부를 나타내는 별개의 플래그다. 즉 S7은 CPU 10파일 오프라인 스모크 테스트(247.4초)만 거쳤고, L4 GPU에서의 실측 런타임(~30분, memory-notes에만 있음, REPORTED)은 영수증 파일 근거가 없다.
2. S7 자체의 offline-vs-runtime parity 파일(`runtime_parity.txt`)이 패킷에 없다. 이 검사는 `s7_new`/`W_new`/`W_newB`/`W_Reweight`/`s7_new_ob`부터 도입된 것으로 보이며, S7 자체가 SSLAM/WavLM 멤버를 추론 시점에 실제로 사용하는지에 대한 직접적 수치 증거는 MEMBERS.txt에 해당 자산이 포함되어 있다는 정황 외에는 이번 패킷에 없다(MANIFEST.md §6 질문3과 동일한 공백).

## 4. (c) 현재 계획된 제출 아티팩트

`MANIFEST.md` §3(21:10 KST, 확보한 소스 중 가장 최신)에 따르면:
- **슬롯1**: `s7_new` (sha `fcdd86913360e4dc...`)
- **슬롯2**: `W_newB` (sha `8d3b62d3ec80a84c...`)
- **슬롯3**: `W_new` (sha `58e7e3bd460cde93...`) 또는 대안

`docs/collab/README.md` §6.3(15:00 KST, 더 이른 시점)은 **①s7_new ②W_new ③(s7_new+AI-OpenBMAT400행 또는 W_newB)** 순서를 제시한다 — **W_new/W_newB의 슬롯 순서가 두 문서 간에 다르다.** 시점상 MANIFEST가 더 최신이므로 이를 현재 계획으로 채택하되, 완전히 해소된 불일치는 아니다(§6 공백 목록 참고).

`docs/collab/README.md` §6.4(16:44–17:45 KST)의 사전 등록 선택 검정에서 "mix(음성 s7_new / 음악 W_new)" 후보는 newval 기준(ΔADS ≥ +0.002 & CI 하한 > 0)을 만족하지 못해(+0.0013 [-0.0020,+0.0057], p>0=0.819) **기각**됐다. 따라서 MANIFEST의 "슬롯3 대안"은 이 mix가 아니라, `research/top15/work5/newdata/bcast_rows.py`(이번 세션 `git status --porcelain`에서 확인된 **유일한 미추적 파일**, VERIFIED)로 진행 중인 별도의 **broadcast 저SNR 후보**를 가리키는 것으로 보인다. 이는 git HEAD(`f20e8b3`, 2026-09-28 18:23:04 KST, 커밋 메시지 "Data deliberation (Codex rounds 24-27): verified gaps, candidate datasets, broadcast low-SNR hypothesis" — 이번 세션 `git log -1`로 VERIFIED)와 메모리 노트의 18:25 KST 항목("user APPROVED one broadcast low-SNR candidate... 23:00 DB decision, 06:30 verified package")이 정확히 교차 확인한다. 이 후보의 23:00 KST 결정은 이번 조사의 근거 수집 시점 이후이므로 상태는 **NOT RUN**이다.

## 5. (d) 가장 강력한 재현 가능 로컬 후보

단일 모델 기준으로는 **`s7_new`**가 가장 방어 가능한 후보다:
- E1(=DB 프로토콜 held-out 4,822행) ADS **0.91540** (+0.0064 vs s7_ref, CI [+0.0013,+0.0111]) — `results/.../new_eval_newtest.json` 직접 열람으로 VERIFIED.
- E2(신규데이터 held-out 12,426행) ADS **0.92185** (+0.0285).
- old val(4,931행)은 조기 종료·CDF 보정에 재사용된 세트이므로 E1/E2가 더 독립적인 대리 지표다(단, E1도 S3~S8 전체 선택 과정에서 반복 참조된 이력이 있어 완전히 새롭지는 않다 — MANIFEST §4 참고).

`W_newB`가 raw E2 수치는 더 높지만(0.94031, +0.0470), newval을 조기 종료(학습)에 직접 사용했으므로 "재현 가능한 held-out 이득"이라는 기준에서는 순환적이라 s7_new보다 근거가 약하다.

`mix(음성 s7_new / 음악 W_newB)`는 E1 점추정치가 가장 높았지만(+0.0103 [+0.0036,+0.0162], `virtual_configs_eval.txt` 직접 열람) 사전 등록된 newval 채택 기준을 통과하지 못해 공식적으로 기각됐다(§4 참고) — 사후적으로 좋아 보이는 조합이라도 이 프로젝트 자체의 사전등록 규칙상 "채택된 로컬 후보"는 아니다.

**결론**: 사전등록 기준을 통과하고 실제 계획(MANIFEST §3 슬롯1)에도 반영된 `s7_new`가 현재 기준 가장 강력하고 방법론적으로 방어 가능한 재현 가능 로컬 후보다.

---

## 6. 공백 목록 (provenance gaps)

1. **영수증은 있지만 점수가 없음**: S4(`beeda9a6...`), S5(`cf17eec9...`) 모두 CPU/parity 검증까지 마쳤지만 어느 소스에도 LB 점수가 없다. `docs/collab/README.md` §2 표에서 완전히 빠져 있고, `MODEL_ARCHITECTURE_ENSEMBLE_DATA_AUDIT_20260928.md` §2.2는 S4="대기 또는 미제출(UNKNOWN)", S5="미제출(REPORTED)"로 명시한다. 09-28 실제 제출 순서(S6→S7→S8)는 실험 ledger가 예정했던 "S4가 09-28 첫 슬롯"과 다르다 — S4/S5는 사실상 건너뛰어졌다.
2. **점수는 있지만 로컬 검증 영수증이 없음**: S8(`b0e87442...`, LB ADS 0.8339682540)은 패킷의 `receipts/DataV4_S8_MixSSLAM/`에 `submit.json`만 있고 `verify.log`/`qa_verification.json`/`runtime_parity.txt`가 전혀 없다. VESSL CUDA 검증 기록(parity V .00082/M .00045/F .00027, peak 21.36GiB)은 memory-notes에만 있으며 이번 세션이 원본 로그로 재확인하지 못했다(REPORTED만).
3. **점수-아티팩트 연결이 추정(inferred)**: PANNs_R3(LB sub 104224, total 0.7901426243)는 `_workspace/lb_cpu_20260924/docs/evidence/panns_leaderboard_20260926/result.json` 스스로가 `"association": "inferred from conversation after the fix; screenshot filename differs"`, `"server_archive_hash_verified": false`라고 명시한다. 어느 zip(원본 `5717f10b...` 또는 fix1 `6efcb018...`)이 실제로 채점됐는지 서버 측에서 확인된 적이 없다.
4. **하드 제약으로 미열람**: S2a(`DataV4_S2a_AIHubPhone`)는 이번 세션 내내 폴더를 열지 않았다(AI-Hub 학습 데이터 포함). sha256은 `docs/collab/README.md`의 16자 접두사(`c7c8b1c83d242df9`)까지만 확보했고 byte-level 재해시는 하지 않았다.
5. **사전-DataV4 구간의 sha256은 8자 접두사만**: C08(`0c3c3289`), M330(`1f6cfd6c`), Redesign(`1fda295b`), Expanded_MIL(`7c7b0c96`), Balanced7144(`665cb1b7`) — 실제 receipt.json/zip이 이 패킷에 없어 이번 세션에서 byte 수준으로 재확인할 수 없었다.
6. **슬롯 순서 불일치**: `W_new`와 `W_newB`의 슬롯 번호가 `MANIFEST.md`(21:10, 슬롯2=W_newB/슬롯3=W_new)와 `docs/collab/README.md` §6.3(15:00, ②=W_new/③=W_newB 또는 대안) 사이에서 다르다. MANIFEST가 더 최신이라 우선했지만 완전히 해소되지 않았다.
7. **bank 분할 카운트 불일치(경미)**: `data/bank/splits.report.json`의 `counts` 필드를 직접 합산하면 train~31,190/val~5,030이 나와 감사 문서(`MODEL_ARCHITECTURE_ENSEMBLE_DATA_AUDIT_20260928.md` §8.1)의 30,658/4,931/6,450과 정확히 일치하지 않는다(원인 미상 — 다른 파이프라인 단계이거나 범주 중복 가능성). val=4,931은 `run_identity.json` 등 다른 모든 소스와 일관되므로 이를 기준값으로 채택했다.
8. **old val의 반복 재사용**: `DataV4_new`부터 `W_newB`까지 거의 모든 variant가 동일한 4,931행 val을 조기 종료 + CDF 보정 + FILE 앙상블 피팅에 반복 사용했다(각 항목의 `local_eval_and_exposure`에 개별 기록). 이 leakage/적응적 과적합 문제 자체는 `claim-ledger.jsonl`의 `v1-leakage`/`v2-statistics` 레인이 이미 별도로 2차 검토(다수 "downgrade"/"reject" 판정 포함) 중이며, 이번 원장 재구성 작업의 범위 밖이다.
9. **runtime_parity.txt 도입 시점**: S3/S4/S5/S6/S7에는 이 파일이 없고 `s7_new` 계열부터 도입됐다. 실제 LB 최고점을 낸 S7 자체의 학습-런타임 수치 일치는 이번 패킷의 직접 증거로 뒷받침되지 않는다(감사 문서의 "학습과 런타임의 일치" 절은 S5/S6 수치만 제시).
10. **L4 실측 전무**: 이번 세션에서 열람한 모든 `verify.log`(S3,S4,S5,S6,S7,s7_new,s7_new_ob,W_new,W_newB,W_Reweight)에서 `official_l4_validated: false`를 직접 확인했다(VERIFIED). 모든 제출이 CPU 10파일 오프라인 스모크 테스트로만 검증됐고, 공식 L4 GPU 런타임/메모리 실측은 어떤 영수증 파일에도 없다(S7의 "L4 30분"은 memory-notes REPORTED뿐).
11. **broadcast 저SNR 후보는 미완성**: 패키지가 아직 없고(스크립트 단계, 미커밋), 23:00 KST 결정은 이번 조사의 근거 수집 시점 이후다 — 상태 NOT RUN.
12. **claim-ledger.jsonl 173줄 중 12줄은 AI-Hub 식별자 포함으로 건너뜀**(작업 지시에 따라 내용을 읽지 않고 스킵). 나머지 161줄 중 90줄을 제출/점수/sha 관련으로 선별해 교차 검증에 사용했다.
