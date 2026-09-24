# 추가 데이터 수집 결과 — 2026-09-24

후속 [CPU 출처 검토와 캐시 감사](CPU_FOLLOWUP_2026-09-24.md)에서 기존 자료와의 참조 화자 연결을 추가로 확인했다. 아래의 원문 그룹 중복 0을 화자 독립성으로 해석하지 않는다.

CPU로 추가 원본 오디오 **2,800개**, **7,986,807,113 bytes**(약 8.0 GB), **344,462.0419374986초**(약 95.7시간)를 확보했다. 원본 파일은 변환하지 않았다. GPU 실행, 새 토큰 발급, 새 접근 동의, 공식 제출은 하지 않았다. 수집·검증 코드는 `523c1643c4e808d8f0886d7d90d12fc3f73c56b4`에 기록했다.

## 확보한 원본

| 묶음 | 이번 확보 | 범위 |
|---|---:|---|
| Jamendo 실제 음악 후보 | 1,000곡 | 아티스트 ID 1,000개, 메타데이터 발매 연도 2008–2015. 곡별 CC BY 또는 CC BY-SA |
| Music Arena 생성 음악 | 1,000곡 | 생성기 13개, 원문 프롬프트 그룹 559개. 이름에 Lyria가 포함된 생성기 제외 |
| LibriSpeech 실제 발화 | 400개 | 86명, 원본 15.0–35.155초. 기존 200개 파일럿과 파일 중복 0 |
| MLAAD 합성 발화 | 400개 | 시스템 108개, 원문 그룹 371개, 원본 15.2500625–58.5초 |

이 수량은 원본 파일 수다. 계획의 최대 4,400개 파생 학습 행과 다르며, 보컬 분리·crop·혼합 진위 조합은 아직 만들지 않았다. LibriSpeech 400개는 기존 200개 파일럿에 추가한 수량이다. 기존 파일럿과 화자 26명이 겹치므로 독립 평가용 자료로 주장하지 않는다.

원천: [JamendoMaxCaps 메타데이터](https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps), [Music Arena](https://huggingface.co/datasets/music-arena/music-arena-dataset), [OpenSLR LibriSpeech](https://www.openslr.org/12/), [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD).

## 실제 검증과 남은 검토

- 2,800개 전체를 CPU에서 끝까지 디코딩했다. 보존한 파일 바이트와 16 kHz mono float32 PCM의 SHA-256은 이번 묶음 안에서 각각 2,800개로 고유했다. 제공한 기존 manifest·출처 기록과의 정확한 파일/PCM 중복도 발견되지 않았다. 변형된 중복이나 다른 코퍼스의 동일 인물까지 배제한 검사는 아니다.
- Music Arena와 MLAAD는 고정 HF revision의 LFS SHA-256에 맞춰 내려받았다. LibriSpeech 원본 archive는 OpenSLR의 MD5와 대조했다. Jamendo는 공개 원본 체크섬이 없어 응답 크기, 전체 디코딩, 수집 시 SHA-256 기록으로 검증했다. 이를 배포자가 공표한 SHA-256 검증으로 표현하지 않는다.
- MLAAD는 기존 수집 계획의 32,861개 SHA-256과 중복 0이며, 기존 evaluation 역할의 `mailabs_text` 그룹과도 중복 0이다. 새 그룹을 독립적인 화자나 새로운 생성기로 해석하지 않는다.
- Jamendo는 기존 수집·평가 아티스트 ID를 제외했다. 그러나 기존 자료와 정규화된 이름이 같은 아티스트의 곡 3개가 있어 검토 표시를 남겼다. 다른 ID만으로 별개의 인물이라고 단정하지 않는다.
- 검토 표시가 있는 파일은 총 **229개**다. Music Arena 24개는 클리핑 의심, Jamendo는 클리핑 의심 179개·메타데이터 길이 차이 24개·아티스트 이름 일치 3개다. Jamendo의 표시가 겹쳐 해당 파일은 205개다. 이 표시는 결함 확정이나 자동 폐기 판정이 아니다.
- `training_admitted=false`, `role=unassigned`, `sealed=false`를 유지했다. 청취·성분 라벨·계보·대회 이용 및 검증자 제공 조건 검토와 역할 배정이 남았다. 기악 프롬프트를 음성 부재 정답으로 바꾸지 않았다. Music Arena의 익명 preview 시스템 정체도 확인하지 않았으므로 생성기 독립성을 주장하지 않는다. 실제/합성 자료의 길이·코덱 분포를 맞춘 학습 실험은 하지 않았다.

Music Arena의 배포 카드 CC BY 표기, Jamendo 곡별 조건, MLAAD의 비상업 연구 조건을 출처 기록에 보존했다. MLAAD 다운로드에는 이미 접근 가능한 사용자 인증만 사용했다. 새 동의를 대신 승인하지 않았다. FMA 접근 대기와 CtrSVDD·Emilia의 기존 보류 조건은 유지한다.

## 저장 위치와 전달 검증

- 기존 VESSL volume: `objvol-a5gxdg7er70i` (`qwdkw`, `dacon-v6-training`)
- 새 prefix: `lb-improvement-20260924/datasets/additional-v1/`
- 원본·출처·검증 기록을 담은 TAR 6개와 각 영수증, 인증정보가 없는 재수집 계획 `collection_plans.json`을 전달했다.
- TAR마다 명시한 파일만 넣고, archive 내부의 모든 파일 해시가 보존된 원본과 같은지 검사했다. 다운로드 임시 파일·lock·Git 이력·인증정보는 포함하지 않았다.
- 최종 클라우드 상태: **전달 파일 13개 전부 재다운로드 완료, 로컬 원본과 크기·SHA-256 모두 일치**. TAR 6개에 원본 오디오 2,800개가 포함됐음을 영수증과 대조했다.

로컬 수집 자료는 원본 프로젝트의 `_workspace/lb_collect_20260924/` 아래에 있다. `payload/`는 원본과 검증 기록, `delivery/`는 전달 묶음, `readback/`은 클라우드 재다운로드본, `receipts/`는 수집·검증 영수증이다. 기존 원본 checkout의 사용자 수정 9개는 SHA-256으로 보존을 확인했다.

## 코드와 재현에 필요한 입력

`repository/data_v4/prepare_librispeech_pilot.py`는 길이 조건과 기존 파일 제외를 적용한다. `prepare_music_arena_pilot.py`, `prepare_jamendo_expansion.py`, `prepare_mlaad_long.py`는 고정 메타데이터와 기존 역할·출처 목록으로 다운로드 계획을 만든다. 각 파일 첫머리에 실행 인자를 기록했다. 선별을 다시 수행하려면 고정 revision의 메타데이터와 기존 프로젝트의 출처 기록이 필요하다. 수집한 파일을 다시 내려받는 계획은 VESSL의 `collection_plans.json`에 있다.

`verify_acquired_audio.py`는 전체 디코딩과 파일/PCM 중복·길이·검토 표시를 기록한다. `package_acquired_audio.py`는 명시한 파일만 묶고 archive 내부 바이트 보존을 검증한다. 기존 다운로더는 401/403을 대기 상태로 남기며 토큰을 명령행 인자에 넣지 않는다.

회귀 검증에서는 짧은 발화와 기존 파일이 다시 선택되는 문제, 원본 바이트 변경, 평가용 Lyria 포함, 프롬프트 태그를 음성 정답으로 오인하는 문제, 403 거절 시 재시도와 토큰 노출을 확인했다. 실제 수집 파일의 디코딩·체크섬·전송 검증은 별도로 수행했다. 이 결과는 모델 품질이나 리더보드 개선의 증거가 아니다.

근거: [집계](evidence/lb_collection_20260924/collection_summary.json), [선별 감사](evidence/lb_collection_20260924/selection_audit.json), [긴 실제 발화 집계](evidence/lb_collection_20260924/long_speech_summary.json), [클라우드 전송 검증](evidence/lb_collection_20260924/transfers.json).
