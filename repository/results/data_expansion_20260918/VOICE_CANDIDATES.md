# 음성 데이터 뱅크 확보 근거와 범위

2026-09-18. 이 문서는 음성 원본의 확보 상태와 출처를 기록한다. 학습·평가·GPU 실행 결과가 아니다. 모든 bank 행의 분할은 `unassigned`이며, 기존 학습 manifest는 수정하지 않았다.

## 현재 확보 상태

| 자료 | 실제 확보·검증 | 화자 범위 | 용도와 제한 |
|---|---:|---|---|
| 기존 LibriSpeech test-clean | 2,620발화, 원본 archive 346,663,984 bytes, 5.4034668403시간 | 공식 reader ID 40개 | 기존 archive 전체를 재인덱싱했다. 공식 MD5와 모든 발화의 완전 디코드·유한 PCM·출처 해시를 검증했다. |
| ASVspoof 2019 LA 선별 원본 | 4,000발화 확보·검증 완료, 원본 합계 242,884,605 bytes, 3.6987045833시간 | 진짜 원본 화자 ID 107개, 가짜 target 화자 ID 78개, VC input 화자 ID 29개 | 진짜 2,000개·가짜 2,000개. 공식 CM protocol·원본 관계표·ZIP 항목과 일치하며 전체 CRC·완전 디코드·SoundFile와 FFmpeg PCM 해시를 검증했다. |
| LibriSpeech train-clean-100 | 미확보 | 251명 확장 계획은 실행 취소 | 사용자 전역 20GB 상한에 맞춰 미완성 archive를 삭제했다. |

LibriSpeech의 기존 검증 패널에 쓰인 8명은 발화 16개만 제외하지 않고 같은 화자의 전체 494개를 `train_eligible=false`로 표시했다. 나머지 2,126개는 향후 학습 후보로 남겼다. 기존 패널의 16개 원본은 source SHA256와 canonical PCM SHA256가 새 인덱스와 모두 일치했다. 이름 정규화로 확인할 수 없는 화자 별칭까지 해소했다는 뜻은 아니다.

합계 6,620발화, 진짜 4,620개·가짜 2,000개, 9.1021714236시간이다. 원본 SHA256와 PCM SHA256는 각각 6,620개가 모두 고유하며 두 corpus 사이의 정확 PCM 중복도 없었다. LibriSpeech의 40개와 ASV의 107개 화자 ID는 corpus별 기록이며, 이를 더해 서로 다른 사람 147명이라고 단정하지 않는다. ASV의 최종 검증은 2026-09-18 06:08:57 UTC에 끝났다.

## ASV 화자 수를 읽는 방법

공식 원본 CM protocol을 직접 집계한 결과다. 소개 문서의 화자 수와 실제 bona-fide protocol의 화자 수를 하나로 취급하지 않는다.

| 공식 partition | 진짜 발화 | 가짜 발화 | 진짜 protocol 화자 ID | 가짜 target 화자 ID |
|---|---:|---:|---:|---:|
| train | 2,580 | 22,800 | 20 | 20 |
| dev | 2,548 | 22,296 | 20 | 10 |
| eval | 7,355 | 63,882 | 67 | 48 |
| 합계 | 12,483 | 108,978 | 107 | 78 |

원본 관계표 전체는 122,299행이지만 CM protocol에는 121,461행이 있다. 이번 선택은 CM protocol과 관계표·ZIP 항목이 모두 일치하는 행으로 한정했다. 관계표에 있다는 이유만으로 CM 라벨을 부여하지 않았다.

`VCTK_ID`, `TTS_VC_target_speaker`, `VC_source_VCTK_ID`는 서로 다른 정보다. 실제 원본 발화, 합성·변환의 대상 화자, VC에 입력한 발화를 각각 보존한다. 동결한 목록에는 VC 입력 관계가 있는 526행이 있다. 나중에 화자 기준으로 분할할 때 target과 input을 모두 고려해야 한다. ASV 개인별 성별은 공식 매핑을 확보하지 못해 `unknown`으로 남긴다. 집계상의 남녀 수로 개인 성별을 추정하지 않는다.

선택 원본에는 진짜 parent reference 2,000개, 고유 VC 입력 reference 491개, target→input 화자 연결 322개가 있다. target과 VC input이 같은 분할에 속하도록 연결하면 화자 ID 107개가 크기 37·30·24·16의 네 연결요소로 묶인다. 이는 관계표에서 계산한 의존 구조이며, 새 분할을 만들었거나 실제 누출을 측정했다는 뜻은 아니다. target ID만 분리한 평가를 VC 입력 화자까지 독립된 평가로 해석하면 안 된다.

가짜 2,000개는 A01–A05 각각 106개, A06–A19 각각 105개다. 공격 방식마다 화자를 순환 선택했고, 후보 순서는 고정 SHA256로 정했다. 모델 점수나 정답 예측으로 고른 목록이 아니다. 공식 partition은 원본 메타데이터로 보존하며 새로운 train/dev/test 분할에 재사용하지 않았다.

## 기존 MLAAD 규모를 해석할 때 주의할 점

현재 로컬 MLAAD metadata 17개에는 합계 1,700행과 모델명 13개가 기록돼 있다. 이 숫자는 확보한 독립 오디오 수가 아니다. 같은 공개 catalog의 중복 사본과 아직 받지 않은 항목이 섞일 수 있다.

확인한 원본 경로의 화자·책 라벨은 11개다. `auf_zwei_planeten`, `erzaehlungen_poe`는 책 묶음이므로 **11개 라벨을 독립 화자 11명으로 계산하면 안 된다.** 기존 core speech base manifest는 512행, 진짜 256개·가짜 256개이며 기록된 speaker group 10개 역시 실명 확인을 마친 독립 화자 수와 다르다.

로컬 저장과 학습 반영도 구분해야 한다. `v2_sources/originals`에는 MLAAD 진짜 385개·가짜 385개가 있으나 source inventory에 반영된 MLAAD는 진짜 200개·가짜 200개였다. `v4_data/sources`에는 진짜 61개·가짜 61개, `validation_sources/originals`에는 각각 32개가 있다. 이 폴더별 숫자를 합쳐 고유 음성 수로 보고하지 않는다.

## 사용할 수 있는 출처와 권한 근거

**LibriSpeech**는 공식 OpenSLR 배포 페이지가 CC BY 4.0을 명시한다. 기존 test-clean 전체 재사용이 추가 다운로드 없이 화자 범위를 넓히는 첫 선택이다. train-clean-100 전체 추가는 이번 용량 제한에서 취소했다. [OpenSLR 공식 배포](https://www.openslr.org/12/)

**ASVspoof 2019 LA**는 저자들이 공개한 Zenodo 원본이며, 실제 확보한 README와 LICENSE가 ODC-By 1.0 사용 조건을 명시한다. ZIP 전체 7,640,952,520 bytes를 받는 경로는 코드에서 차단했다. 공식 ZIP64 central directory와 소형 CM protocol을 이용해 300MB 이하의 선택 원본만 받았다. 선택 파일의 CRC와 source SHA256를 검증했으며, 전체 archive MD5는 검증하지 않았다. [저자 배포 원본](https://zenodo.org/records/6906306), [공식 원본 관계표 안내](https://www.asvspoof.org/index2019.html)

**MLAAD-tiny 재인덱싱**은 기존 생성기와 원본 관계를 복구하는 후보로 남긴다. 로컬에 보존된 fake license는 CC BY-NC 4.0이며, 원본 M-AILABS 자료에는 별도 라이선스가 있다. 생성 음성의 사용 조건을 원본 음성에 그대로 적용하지 않는다. 전체 MLAAD는 추가 접근 동의를 요구하므로 이를 자동으로 수락하거나 우회하지 않았다. 이번 작업에서 신규 MLAAD 원본은 받지 않았다. [공개 MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny), [전체 MLAAD 접근 조건](https://huggingface.co/datasets/mueller91/MLAAD)

LibriSeVoc는 공개 archive가 약 45GB여서 이번 범위를 벗어난다. WaveFake는 독립 화자 확대 목적에 맞지 않아 수집하지 않는다. 이 후보를 학습 데이터로 확보했다고 계산하지 않는다.

## DF-Arena 중복과 품질 판단

현재 DF-Arena model card의 학습 목록에는 ASVspoof 2019·2024, MLAAD, LibriSeVoc가 명시돼 있다. 이번 ASV bank는 `listed_training_corpus; exact_recording_unknown`으로 기록한다. corpus 사용은 확인했지만 각 발화의 사전학습 포함 여부는 공개 목록만으로 확정할 수 없다. ASV나 MLAAD에서 로컬 분할을 새로 만들었다고 DF-Arena의 사전학습과 독립된 검증이 되지는 않는다. LibriSpeech 자체는 목록에 명시되지 않았지만 LibriSeVoc와 LibriVox 계통의 자료를 공유할 가능성을 배제하지 않았다. [확인한 DF-Arena model card](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1/blob/fb6ce85/README.md)

이 작업에서 가짜 음성의 품질은 출처, source 라벨, 화자·생성기 관계, 디코드 무결성, 실제 음성 커버리지를 뜻한다. 자연스러움·발음·합성 성능을 청취 평가한 결과는 없다. 프레임별 진위 구간도 추정하지 않았다. 음악이 없는 source에서 `MUSIC_FAKE`는 빈 값과 mask 0으로 보존한다.

전체 발화를 다시 읽어 RMS, 절댓값 peak, `abs(sample) >= 0.999` 비율, 정확 0 샘플 비율, 길이를 별도 CSV/JSON에 기록했다. 검토 기준은 전체 무음, RMS `<1e-6`, 길이 `<0.25초`, near-full-scale 샘플 비율 `>=0.001`이다. ASV 4,000개는 이 네 기준에 걸린 파일이 없었다. 길이는 0.6661875–12.1054375초, RMS는 0.0417344975–0.2569611104였다. LibriSpeech 2,620개에는 무음·극단적 저음량·극단적 짧은 발화가 없었고, `2094-142345-0000`, `2094-142345-0005`, `2094-142345-0025`의 near-full-scale 비율만 검토 기준을 넘었다. 이러한 수치는 청취 품질 평가가 아니며, 원본·라벨·학습 후보 여부를 바꾸지 않았다.

현재 은행을 그대로 합치면 LibriSpeech가 진짜 라벨에만 추가되어 corpus와 라벨 사이의 상관이 생긴다. 향후 학습에 반영할 때 source별 샘플링과 평가 범위를 먼저 정해야 한다. 이번 작업은 은행 확보로 끝내며 기존 학습 manifest에는 반영하지 않았다.

## 저장량과 근거 파일

미완성 train-clean-100 1,121,734,656 bytes와 ASV LA 185,761,792 bytes, 합계 1,307,496,448 bytes를 삭제했다. 삭제 후 기존 test-clean과 완료된 메타데이터는 보존했다. 선택 ASV 원본 242,884,605 bytes를 모두 확보했고, 기존 test-clean archive와 메타데이터를 포함한 최종 speech bank는 1GB 미만이다. 실제 apparent/allocated 용량은 `speech_completion_receipt.json`에 남겼다.

처음 81개를 확보한 뒤 HTTP 429로 수집을 멈췄다. 이후 공유 속도 제한 75요청/분, 429 대기 최소 60초와 `Retry-After`·유효한 `X-RateLimit-Reset` 중 긴 시간을 반영하도록 보완했다. 인접한 선택 항목을 최대 4MiB의 공식 HTTP Range로 묶어 요청했고, 첫 실제 응답 4,192,800 bytes에서 선택 원본 5개를 각각 검증했다. 사이에 끼인 비선택 항목의 바이트는 메모리에서 버렸다.

초기 병합 수집 중 curl timeout 28이 한 번 발생했다. 응답 헤더가 없었다는 사실까지 확인했으며 DNS·TLS·서버 중 세부 원인은 확정하지 않았다. 같은 공식 Range `184050196–187194514`를 제한된 재시도 정책으로 다시 받아 HTTP 206·3,144,319 bytes와 선택 원본 4개의 CRC·PCM 해시를 검증했다. 마지막 재개 실행은 1,212요청·3,006,863,735 bytes를 성공적으로 전송하고 종료 코드 0으로 끝났다. 이 수치는 마지막 실행의 성공 전송량이며, 과거 probe·실패한 전송까지 합친 정확한 누적 네트워크량은 아니다. **전송량과 최종 보존 원본 242,884,605 bytes는 별개다.**

- `speech_retention_receipt.json`: 미완성 원본 삭제 경로·크기·사유·부재 확인
- `speech_bank_test-clean.csv`, `speech_bank_test-clean.json`: 검증된 2,620발화와 archive 체크섬
- `speech_asv_protocol_scope.json`: 실제 CM protocol의 라벨·화자 범위
- `speech_asv_partial_check.json`: 초기 81개 검증 당시의 기록과 기존 패널 16개의 원본 일치
- `speech_bank_asv_selected.csv`, `speech_bank_asv_selected.json`: 최종 4,000개 원본과 전체 검증 영수증
- `speech_final_reconciliation.json`: 동결 목록·공식 라벨·원본 관계·mask·중복·보류 화자 대조
- `speech_parent_groups.json`: 실제 원본·target·VC input 관계와 연결요소
- `speech_quality_libri.*`, `speech_quality_asv.*`: 전체 원본에서 측정한 신호 통계와 검토 표시
- `speech_rate_limit_receipt.json`: HTTP 429 시각·quota·reset 값과 재시도 제한
- `speech_asv_coalescing_estimate.json`: 요청 병합의 전송량·요청 수 계산
- `speech_asv_coalesced_probe.json`, `speech_timeout_recovery.json`: 실제 다중 항목 응답과 실패 Range 재검증
- `speech_atomic_retry_regression.json`: 중단·CRC 실패 시 final 파일 보존과 재시도 대기 계산
- `speech_completion_receipt.json`: 최종 용량·프로세스 종료·정리·변경 범위
- `_workspace/data_expansion_20260918/speech/asv_selection.json`: 변경을 막은 4,000개 선택 목록

현재 상태는 **LibriSpeech 2,620개와 ASV 4,000개 확보·검증 완료**다. 모든 행은 `unassigned`이며 DF-Arena 사전학습과 독립된 평가 자료로 주장하지 않는다. 수집 프로세스가 종료됐고 추가 자료·GPU·학습·자동화는 실행하지 않는다. 이 담당 작업에서는 commit하지 않았으며, 코드와 근거 파일의 최종 commit은 상위 작업에서 처리한다.
