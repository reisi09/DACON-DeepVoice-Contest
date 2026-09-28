**BLOCKING — DB 시작 전 두 가지 확인·수정이 남았습니다.**

1. **[VERIFIED] 계보 교차 검사가 무효입니다.** `group_id`에 split을 넣어 같은 연결요소도 `bcast:train:c0`, `bcast:bpanel:c0`로 나뉩니다. 공유 음악 그룹을 가진 반례가 실제로 assertion을 통과했습니다. **split을 붙이기 전 DSU root로 교차를 검사**하십시오. `INTEGRITY_JSON`도 생략 시 제외 목록이 비워지므로 필수 입력으로 바꿔야 합니다. 실제 생성물의 누수 여부는 UNKNOWN입니다.  
   근거: `round3b/bcast_rows.py:216`, `round3b/bcast_rows.py:248`, `round3b/bcast_rows.py:253`.

2. **[VERIFIED / UNKNOWN] 실행기와 생성기의 CSV 계약이 다릅니다.** 실행기는 `bcast_rows_all.csv`, `locked_rows_all.csv`를 요구하지만 제공된 생성기는 이를 만들지 않습니다. 외부에서 생성했는지는 UNKNOWN입니다. **bcast train+panel과 locked의 실제 CSV를 명시적으로 연결하고 manifest ID 일치를 확인**해야 합니다. 또한 손상된 기존 npz는 todo에 들어가도 extractor가 “파일이 존재한다”는 이유로 건너뜁니다. 해당 파일을 격리한 뒤 재추출해야 합니다.  
   근거: `round3b/run_x2_db.sh:13`, `round3b/bcast_rows.py:259`, `round3b/locked_v2.py:46`, `code/research/top15/work5/newdata/dump_features.py:120`.

**[VERIFIED] 결과를 보기 전 N2도 수정해야 합니다.** 현재 자료에서 AT-ADD와 전화채널의 교집합은 0입니다. 따라서 격자의 `AT-ADD 80% + 전화 30%`는 구현 불가능하며 IPF로 해결되지 않습니다. 구조적으로 불가능한 점을 지금 제외·수정하고 지원되는 격자를 고정하십시오. 이는 학습 계산 자체보다 **채택 판정의 차단 사항**입니다.  
근거: `round3b/x2_gates.py:170`, `round3b/x2_gates.py:180`, `round3b/x2_gates.py:260`.

**T2 변경은 진단·거부 조건으로는 수용할 수 있습니다.** 3개 연결요소의 점추정치를 독립적인 개선 증거로 취급하지 않고, **T1을 주 개선 근거로, T2를 추가 sanity check로 유지**하는 방식을 권고합니다. 결과 전에 변경했다는 사실과 근거 수준의 하향을 기록하십시오.  
근거: `round3b/x2_gates.py:165`, `round3b/x2_gates.py:242`.

**T1에는 다음이 더 필요합니다.**

- **[VERIFIED·수정 권고]** 현재는 전체 20그룹만 검사합니다. 클래스별 연결요소 수와 최대 그룹 비중도 확인하고, 표본 부족·불안정한 CI는 HOLD로 처리하십시오. `ΔJ≤−0.02`, paired CI 상한 `<0` 조건은 유지합니다.  
  근거: `round3b/x2_gates.py:238`.
- **[VERIFIED·수정 권고]** 수정된 OpenBMAT 그룹은 T1에만 적용됩니다. **N1의 `nat`에도 같은 그룹 맵을 적용**해야 합니다.  
  근거: `round3b/x2_gates.py:134`, `round3b/x2_gates.py:163`, `round3b/x2_gates.py:168`.
- **[UNKNOWN]** 음악 트랙만 분리한 T1은 공유 방송 음성과 반복 선택의 영향을 배제하지 못합니다. CI 통과를 독립 최종평가로 해석하지 마십시오.

앞의 입력·계보 검증이 해소되면 **T2의 점추정치 전환 자체는 DB 학습을 막을 이유가 없습니다.**