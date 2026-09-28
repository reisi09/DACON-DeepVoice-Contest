**BLOCKING — 현재 코드로 DB 학습을 자동 시작하는 것은 보류해야 합니다.**

1. **[VERIFIED] bpanel donor가 학습 계보와 분리되지 않습니다.** 실제 CSV에 donor 필터를 적용하면 train·panel 음성 풀에 **4개 그룹, panel 10행**이 겹칩니다. 그런데 생성기는 integrity의 제외 목록을 사용하지 않습니다. 또한 다른 화자의 발화를 보충하면서 `parent_ids`는 8개로 자르고, `group_id`는 최초 화자만 기록합니다. 공유 음악 donor도 bootstrap 그룹에 반영되지 않습니다. **실제 사용한 모든 donor를 기록하고, 전체 연결요소로 train/panel을 분리한 뒤 CI를 계산해야 합니다.** 최종 생성된 panel의 실제 오염 행 수는 UNKNOWN입니다.  
   근거: `round3/bcast_rows.py:156`, `round3/bcast_rows.py:165`, `round3/bcast_rows.py:199`, `round3/x2_gates.py:80`.

2. **[VERIFIED] 사전 명시한 seed 관문 S가 구현되지 않았습니다.** 주석에는 “3개 seed 중 2개 이상 T1 개선”이 있지만, 판정에는 T1/T2/N1/N2/P만 들어갑니다. 따라서 S 실패 여부와 무관하게 `adopt_offline=True`가 가능합니다. S를 구현하고, 별도 Mac E3 결과까지 결합하기 전에는 최종 채택 상태를 내지 않아야 합니다.  
   근거: `round3/x2_gates.py:13`, `round3/x2_gates.py:187`, `round3/x2_gates.py:200`.

3. **[VERIFIED] N2는 명시한 혼합 비율의 격자를 계산하지 않습니다.** 상관된 유형·전화·AT-ADD 축의 주변분포 비율을 곱하면 목표 주변분포가 보존되지 않습니다. 공동 층별 가중치 또는 반복 비례 보정으로 바꾸고, 각 점의 실제 가중 분포·유효 클래스·유한성을 확인해야 합니다. 현재 결과를 “45개 목표 비율 모두 통과”로 해석할 수 없습니다.  
   근거: `round3/x2_gates.py:135`, `round3/x2_gates.py:148`.

4. **[VERIFIED] GPU 실행기가 실패·완료를 잘못 판정합니다.** `finished` 두 줄만 기다리므로 실패 행이 있어도 학습으로 넘어갑니다. 반대로 worker가 죽거나 캐시가 모두 있어 조기 반환하면 무한 대기할 수 있습니다. DB 명령 실패도 뒤의 `echo`가 가립니다. **PID별 종료 상태, 제한시간, 예정 ID 전체의 특징·presence 유효성으로 완료를 판정**해야 합니다. 생성 규칙을 수정하면 같은 ID의 기존 npz를 재사용하지 마십시오.  
   근거: `round3/run_x2_gpu.sh:7`, `round3/run_x2_gpu.sh:14`, `code/research/top15/work5/newdata/dump_features.py:120`, `code/research/top15/work5/newdata/dump_features.py:153`.

두 가지 해석 변경에 대한 판단입니다.

- **[VERIFIED / HYPOTHESIS] N/A는 설계를 완화합니다.** 측정 불가능한 층을 명시적으로 제외하는 정책은 결과 전에 고정할 수 있지만, 해당 층의 안전성이 확인되는 것은 아닙니다. 현재 구현은 이를 넘어, 필터 후 대상이 없으면 모두 N/A로 처리하고 `sequential`은 조사 없이 문자열로 고정합니다. **사전 목록에서 존재하지 않는다고 확인한 층만 N/A**, 누락 예측·라벨·특징 때문에 사라진 층은 HOLD로 처리해야 합니다.  
  근거: `round3/x2_gates.py:168`, `round3/x2_gates.py:171`, `round3/x2_gates.py:200`.

- **[VERIFIED / HYPOTHESIS] OpenBMAT의 음악 트랙만 제외하는 변경은 독립성 기준을 낮춥니다.** 음성이 양쪽 클래스에 나타난다는 사실만으로 학습·평가 의존성이 없어지지는 않습니다. 이 부분은 **음악 트랙만 분리한 조건부 평가**로 보고하고 독립 최종평가와 구분하십시오. 현재 잠금 생성기는 bank를 제외 목록에 넣지 않으며, 새 해시를 기존 사용 자료와 대조하지도 않습니다. OpenBMAT 내부 그룹 역시 트랙 집합 문자열이어서 일부 트랙을 공유하는 행이 서로 다른 그룹으로 남습니다.  
  근거: `round3/locked_rows.py:39`, `round3/locked_rows.py:94`, `round3/locked_rows.py:106`, `round3/locked_rows.py:134`.

**[VERIFIED] 라벨 생성에도 방어가 필요합니다.** `mix()`는 겹치는 음성 또는 음악의 RMS가 0이면 음악을 넣지 않고 반환하지만, 이후 MUSIC_PRESENT=1과 fake 라벨을 기록합니다. 해당 경우는 행 생성을 거절해야 합니다. **[UNKNOWN]** 실제 생성물에서 발생했는지는 제공 자료로 확인되지 않습니다.  
근거: `round3/bcast_rows.py:98`, `round3/bcast_rows.py:130`.

**DB 시작 전 필수 조치:** donor 계보·라벨 문제 수정, 변경 오디오의 캐시 무효화, 학습 입력과 모든 평가 donor의 교차 검사, 실행기의 실패 처리 수정입니다. 관문 S/N2/N/A도 결과를 보기 전에 고정해야 합니다.

**[UNKNOWN]** 패킷의 `round2/DESIGN_KO.md`에는 §5.2가 없고, 생성 CSV·LOCK·전체 `dirty_ids`도 없습니다. 따라서 승인된 T-type 규칙과의 완전한 일치 및 실제 누수 0은 확인할 수 없습니다. 검토는 폴더 내부 코드와 CSV 분석으로 제한했습니다.