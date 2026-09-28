# 2026-09-28 23:46 – 2026-09-29 00:53 KST: X3 (SONICS AI 노래) 실행 결과

가설: 실제 AI 노래(SONICS train split의 Suno chirp-v3.5 750곡, Udio udio-120s 750곡, 모두 보컬 포함)를 s7_new 레시피에
더하면 노래와 혼합 음원의 FILE EER이 내려간다. 대조는 s7_new이고, 두 쪽 모두 DB 프로토콜로 학습했다(같은 job, WEIGHTING=s7,
NEW_VAL=0). 판정은 결과를 보기 전에 고정한 `newdata/x2_gates.py` MODE=x3로 했다(md5 90bca7cb…). 이 파일은
00:14 KST 커밋 c3d92d3으로 미러에 push했다.

## 데이터 (`newdata/sonics_rows.py`)
- 한 SONICS id에서 한 곡만 쓴다. bank와 new rows의 모든 열에서 참조하는 SONICS id 1,502개는 제외했다. id는 정수로
  맞춰 비교한다(bank는 `fake_02205`, 메타데이터는 `2205`). 원본·크롭 해시 42,039개와도 겹치지 않는다.
- 각 곡에서 30% 지점부터 20초를 FLAC으로 잘랐다. 라벨은 bank의 `sonics` 행과 같은 규칙이다(voice·music 존재, 둘 다 fake).
- 특징 추출 뒤 패키지 presence 추정이 0.5 미만인 크롭 16개는 voice 라벨을 unknown으로 두었다.
- npz 1,500개는 학습에 쓰는 14개 키, 유한값, 윈도 수 일치 검사를 모두 통과했다.

## 판정 (`x3_gates.json`는 x1 볼륨 `/root/x3/`에 있고, 아래 수치는 그 파일에서 옮겼다)
| 관문 | 결과 | 판정 |
|---|---|---|
| T3 표적: E1 voice+music 파일 FILE EER (1,228행, 555그룹) | Δ +0.0065, CI [−0.0054, +0.0165] | 실패 (기준: ≤ −0.01, CI 상한 < 0) |
| T1 방송형 guard | ΔJ +0.0126 (한도 +0.02) | 통과 |
| N1 전역 ADS (E1 + E2 17,158행) | Δ +0.0004, CI [−0.0024, +0.0021] | 실패 (하한 −0.002 미달) |
| N2 격자 (실현 가능 32점) | 최저 −0.0046 | 실패 |
| S seed별 T3 | +0.0065, −0.0024, +0.0164 (3개 중 1개만 음수) | 실패 |
| realV+fakeM MUSIC 놓침 | +0.047 (23그룹) | 실패 |
| real 노래 FILE 오탐 (신설) | +0.0071 (151그룹) | 통과 |
| AT-ADD singing FILE / VOICE (jackknife) | −0.0029 / −0.0013, 한 그룹 제외 시 최대 증가 +0.0002 / −0.0011 | 통과 |
| 그 밖의 보호 층 | music-only FILE EER −0.0073, Pm 0.1–0.5 fake FILE 놓침 −0.056 등 | 통과 |

해석: 사전 등록한 채택 기준을 충족하지 못했다. 표적 층의 점추정은 나빠졌지만 CI가 0을 포함하므로, 유의한 악화가
입증된 것은 아니다. 채택 자격이 없어서 E3(Mac), P1, 잠근 셋, 패키징은 실행하지 않았다. 잠근 셋은 여전히 채점하지 않은 상태다.

## 코드 교차검증 (GPT-6-Astra, reasoning ultra, read-only; 라운드 5–5f)
- 5: 선택 수 검사 없음, 크롭 보컬 미검증, singing 관문이 구조적으로 HOLD, id 앞자리 0 불일치, `.wav` 확장자,
  npz 키 검증 부족. 모두 실행 전에 고쳤다(5b에서 BLOCKING 없음 확인).
- 5c–5e (잠근 셋 채점기와 자동 체인): voice 융합 분기가 runtime과 다름(`w_runtime_parity.py`도 같은 오류), G6 판정 불가가
  실패를 덮음, 1회 사용 보장 부족, 실패가 정상 완료처럼 처리됨, 대기 기한 없음, PGID 기록 오류, 잔액 조회 실패가 통과로
  처리됨, 손자 프로세스 잔존, npz 출처 미증명. 모두 고쳤다.
  - 잠근 셋은 커밋된 manifest(sha256 dcae0c85…)와 byte 단위로 같아야 채점된다.
  - 30행을 새로 추출해 저장된 특징과 대조했다(`locked_recheck.json`, cosine 1.000000, 불일치 0).
  - 이 대조 방식에도 5f에서 결함이 확인됐다. cosine만으로는 척도 차이와 NaN을 막지 못하고, 기록이 npz 해시에 연결되지
    않았다. 그래서 `locked_recheck.py`를 유한값·절대오차 비교와 npz 해시 기록 방식으로 바꿨다. `score_locked.py`는 새 형식의
    기록만 받으므로, 잠근 셋을 쓰기 전에는 재검증을 다시 실행해야 한다.
- `w_runtime_parity.py` 수정본으로 기존 4개 패키지를 다시 돌렸다. 최대 오차가 이전과 같았고 모두 PASS다
  (`submissions/DataV4_*/runtime_parity_fixedbranch.txt`).
- `pkg_chain.sh`: CPU 검증이 `&&` 목록 안에 있어서 실패해도 멈추지 않았다. 고쳤다.

## 비용과 상태
- VESSL 잔액 $18.75 → $11.70. 옵션 B와 X3를 합친 누적 사용액은 $7.05로, 상한 $14 이내다.
- x1은 00:53 KST에 정지했다(state stopped 확인, 과금 0/h). CPU 워크스페이스도 정지 상태다.
- 제출 계획은 그대로다: ① s7_new ② W_newB ③ W_new 또는 s7_new_ob (`docs/collab/SUBMISSION_PLAN_20260929.md`).
