# 같은 채팅에서 이어가기

## 설계만 쓰고 끝냈을 때

앞서 첨부한 입력 ZIP과 01_MAIN_PROMPT.md를 기준으로 이제 실제 구현을 완성해라. 설계 설명을 반복하지 말고 cache adapter, feature extraction, grouped meta-model training/evaluation, 저장·로드·추론, 유의미한 검증 코드를 파일로 작성해라. 기존 head를 학습했던 데이터의 예측을 OOF로 쓰지 마라. 확보된 6개 head와 10개 cache로 가능한 CPU 수치 검증을 수행하고, 외부 자산 때문에 실행하지 못한 항목만 NOT_RUN으로 남겨라. 04_RETURN_CONTRACT.md의 산출물을 DACON_Stacking_Return.zip으로 묶어 다운로드할 수 있게 제공해라.

## 응답 길이 또는 실행 시간이 끝났을 때

이미 작성한 파일과 결과를 유지하고 남은 작업부터 계속해라. 먼저 완료 파일과 미완료 항목을 짧게 확인하고, 핵심 함수의 TODO나 생략된 코드부터 완성해라. 실행하지 않은 검증을 통과했다고 쓰지 마라. 마지막에는 변경 파일·실행 로그·미검증 목록이 모두 포함된 DACON_Stacking_Return.zip 하나를 제공해라.

## 최종 자체검사

반환 전에 새 기능을 늘리지 말고 다음 결함을 집중 검사해라: 기존 val/test를 독립 OOF로 오인했는지, source/generator/route/정답/mask/window_truth가 피처로 들어갔는지, 학습·early stopping·전처리에 평가 그룹이 섞였는지, 서로 다른 시간축을 그대로 곱했는지, per-window probability와 MIL logit 집계가 바뀌었는지, 저장 후 추론과 파일 순서·부분집합이 결과를 바꾸는지. 실제 작은 head/cache로 가능한 검증을 실행하고 근거 로그를 남겨라. 미실행은 미실행으로 유지하고 ZIP을 갱신해라.

## Codex에 결과를 돌려줄 때

첨부한 DACON_Stacking_Return.zip은 일반 ChatGPT 채팅에서 만든 구현이다. artifacts/Chat_Stacking_20260921의 원래 요청·입력 manifest·수락 기준과 대조해서 검사해라. 먼저 코드와 데이터 누수, 수치 재현, CPU 통합, 파일별 독립성만 확인하고 필요한 최소 수정을 해라. 새로운 GPU 실행·데이터 대량 수집·대회 제출은 하지 마라. 확인된 것과 미검증을 구분해서 보고해라.
