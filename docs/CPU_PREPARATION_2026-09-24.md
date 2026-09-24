# DACON CPU 준비 결과 — 2026-09-24

후속 작업으로 진행한 추가 원본 2,800개 수집은 [추가 데이터 수집 기록](ADDITIONAL_DATA_COLLECTION_2026-09-24.md)에 정리했다.

사용자가 승인한 CPU 구현·자료 정리·VESSL 업로드를 수행했다. GPU 학습과 공식 제출은 실행하지 않았다. 원본 checkout의 기존 수정 9개는 SHA-256으로 보존을 확인했다. 구현 commit은 `ad4a6ad918b8d43df4b2e6923c3cd514decbac66`이다. 지정 GitHub 저장소의 `c157c2fe01f2a0707049fe5a7124c38147a82f84`에서 만든 `cpu/lb-improvement-20260924` 브랜치에 변경을 모았다.

[기존 workflow](../repository/docs/superpowers/plans/2026-09-24-dacon-lb-improvement-workflow.md)는 실행 전 계획의 보존본이다. 이 문서가 현재 수행 결과와 남은 제약을 기록한다.

## 구현과 실제 검증

- Echoes 경로에서 TTA/ATA와 생성기를 분리했다. 실제 594행은 12개 생성기로 구분된다. 원곡 참조·동일 파일 hash·명시적 별칭·부모 관계를 component에 반영한다. 구성 전에 부모 역할을 확인하고 자식이 같은 역할을 상속한다.
- 42,947행을 새 `roles-v3`에 배정했다. train 29,643 / selection 2,585 / calibration 2,202 / development 8,517이다. 역할을 넘는 component는 0개다. 기존 분할이 충돌한 40개 component와 원곡 별칭이 미해결인 Echoes 1,238행을 development로 격리했다. 새 sealed는 없다.
- VESSL의 원본 legacy catalog와 로컬 16,042행 catalog의 hash가 일치한다. 이 자료로 기존 manifest의 legacy 8,075행을 모두 연결했다. 콜론이 포함된 ID도 원래 ID를 보존한다. 기존 head 학습·presence 적합·CDF 적합·분석 노출 이력을 component 전체에 전파한다.
- 음성 구간 정답은 시간 정렬이 확인된 경우만 내보낸다. 64,600-sample 반복 입력·마지막 겹침 구간에 맞춰 점유율 target을 만든다. 기존 splice 121행은 codec 정렬 미확인 상태로 유지하며 local loss에 억지로 넣지 않는다.
- FILE 라벨이 없다는 이유로 행 전체를 버리던 경로를 수정했다. 각 지표는 자기 마스크를 적용한다. 과거에 빠진 HAIM 6행은 새 bank에 남으며 MUSIC 정답도 유지한다. 해당 6행의 새 예측은 아직 계산하지 않았다.
- `--experiment-config`로 seed, voice max/LME/top-k, voice/music local-loss 가중치를 연결했다. 학습·평가·패키지 추론이 같은 계산을 쓴다. optimizer는 train만, checkpoint 선택은 selection만, 새 CDF 적합은 calibration만 사용한다. 지원하지 않는 설정은 실패한다.
- CPU에서 실제 classifier gradient와 parameter update, local loss off/on 차이, microbatch 분할 후 업데이트 일치, 1/3-seed 로딩, 파일 순서 독립성을 확인했다. 이는 수치·연결 검증이며 모델 품질 검증이 아니다.
- 캐시는 row/hash뿐 아니라 encoder 자산·전처리·window·dtype·구현 계약도 맞아야 평가에 쓴다. 패키징도 checkpoint 내부 identity와 실제 포장한 자산 hash를 대조한다. 자체 receipt가 유효해도 다른 MERT 가중치를 담은 archive는 거부한다. 출처 계약이 없는 과거 head의 재포장은 자동 허용하지 않는다.

기존 저장 예측 6,444행에서 File/Voice/Music EER 및 presence AUC를 재계산한 결과 저장 보고서와 차이는 **0**이었다. `W_known_subset=0.13685823726783192`이며 공식 점수는 `null`이다. 독립 평가나 성능 개선 결과로 해석하지 않는다. 근거: [baseline lock](evidence/lb_cpu_20260924/baseline_lock.json), [역할 집계](evidence/lb_cpu_20260924/roles_report.json), [데이터 무결성](evidence/lb_cpu_20260924/data_integrity.json).

## 저장소와 수집

목적지는 기존 조직 `qwdkw`의 `objvol-a5gxdg7er70i`, 새 prefix는 `lb-improvement-20260924/`다. 기존 원본·archive는 덮어쓰지 않았다. GitHub에는 코드·문서·작은 검증 영수증만 넣는다.

| 경로 | 실제 내용 | 검증 |
|---|---|---|
| `weights/baseline-assets/` | MERT, DF-Arena, PANNs, HTDemucs 및 설정·코드 26파일, 6,265,302,121 bytes | 원래 자산 lock과 대조, 업로드 후 전 파일 재다운로드 SHA-256 일치 |
| `datasets/roles-v3/` | 원본 42,947행의 새 manifest·역할표·bank·집계 | 전체 component 역할 검사, 업로드 후 readback 영수증 |
| `datasets/librispeech-pilot-v1/` | 원본 발화 200개와 출처·권리·검증 기록, 총 205파일, 46,657,244 bytes | 공식 archive MD5, 전체 audio decode, 전 파일 readback SHA-256 |

`roles-v2`도 작업 중 업로드·readback 검증했으나, 전체 legacy catalog 연결을 기록한 `roles-v3`가 최종본이다. 이전 namespace는 보존하며 다음 학습 입력으로 사용하지 않는다. 기존 DataV4 archive 14개와 생성용 가중치는 목록·크기를 확인했고 다시 전송하지 않았다. 기존 자료에 대해서는 목록 확인을 새 hash 검증으로 주장하지 않는다.

새 LibriSpeech pilot은 33명, 2,590.7750625초다. 원본 FLAC을 변환하지 않았다. 확인한 기존 LibriSpeech 화자 40명과 겹치지 않고, 기존 materialized manifest와 정확한 파일 hash 중복도 없었다. 다른 코퍼스의 동일 화자·backbone 사전학습 노출·청취 품질은 미확인이다. 따라서 `training_admitted=false`, `role=unassigned`이며 sealed로 지정하지 않았다. 긴 발화·가짜 음성·혼합 진위 목표를 이 표본만으로 충족했다고 보지 않는다. [수집 영수증](evidence/lb_cpu_20260924/librispeech_pilot.json), [중복 검사 범위](evidence/lb_cpu_20260924/pilot_overlap.json).

자료는 HF 외 원천도 사용했다. LibriSpeech는 [OpenSLR 공식 배포](https://www.openslr.org/12/)에서 내려받고 [공식 MD5](https://www.openslr.org/resources/12/md5sum.txt)로 검증했다. CC BY 4.0 출처·저자·라이선스를 payload에 보존했다.

| 원천 | 이번 처리 | 남은 조건 |
|---|---|---|
| OpenSLR LibriSpeech dev-other | 314,305,928-byte archive 수집, 200개 pilot 보관·업로드 | cross-corpus identity·청취 검토 후 역할 배정 |
| FMA 원본 CC-BY 후보 | 기존 artist 제외 후보의 직접 링크가 HTTP 403, 다운로드 0 | `pending_access`; 재인증·우회 시도 안 함 |
| Music Arena | 기존 후보 유지, 신규 audio 수집 안 함 | 카드의 CC BY 표기 외 원서비스·검증자 제공 조건 확인 |
| MTG-Jamendo | 기존 확보분 유지, 확대 안 함 | 곡별 라이선스·데이터셋 조건 확인 |
| CtrSVDD | 권리 확인 대기 | 저자 README의 BY-NC-ND와 변환 계획 충돌 해소 |
| Emilia | 접근 동의·권리 확인 대기 | 사용자 동의와 검증자 제공 조건 확인 |

[FMA 원천](https://github.com/mdeff/fma), [Music Arena 카드](https://huggingface.co/datasets/music-arena/music-arena-dataset), [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset), [CtrSVDD 저자 저장소](https://github.com/SVDDChallenge/CtrSVDD_Utils), [Emilia 카드](https://huggingface.co/datasets/amphion/Emilia-Dataset). 새 토큰은 발급하지 않았다. 기존 HF read 인증과 VESSL 접근은 확인했다. 다운로더는 401/403을 재시도하지 않고 대기 상태로 기록하며, HF token을 프로세스 인자나 영수증에 넣지 않는다.

## workflow 상태와 다음 진입 조건

| 작업 | 상태 | 남은 일 |
|---|---|---|
| W00 | CPU 기준점 확인 | 새 모델의 기준 run은 미실행 |
| W01 | 코드·실제 역할표 적용 | Echoes 원곡 별칭 해결; 미해결은 격리 유지 |
| W02 | 공개 원천 pilot 수집·보관 | 미확인 품질·계보·권리 자료는 학습 미승인; GPU 생성·성분쌍 구성 미실행 |
| W03 | 구간·마스크·출처 계약 구현 | 기존 121행 정렬, HAIM 6행 새 예측, 실제 cache 확보·승격 검증 |
| W04 | seed·pooling·local loss CPU 검증 | pair/augmentation loss·joint head는 E00와 pair pilot 이후; 현재 허용 설정 아님 |
| W05–W09 | GPU 대기 | E00–E14 비교, OOF·조합·calibration·sealed 미실행 |
| W10 | packager 매개변수·자산 계약 구현 | 실제 최종 ZIP·전체 encoder parity·offline/Linux·L4 미검증 |
| W11 | 미실행 | 이번 요청 범위에 공식 제출 없음 |

VESSL object storage의 DataV4 results에는 결과 ZIP 2개만 확인했다. 그 목록만으로 feature cache가 보존됐다고 판단할 수 없다. 중지된 workspace를 기동해 조사하지 않았다. 기존 cache의 원래 index·producer 코드·자산·오디오 계약을 확보하기 전에는 metadata hash만 바꿔 재사용하지 않는다. 확보되지 않으면 승인된 GPU 예산에서 재추출 비용부터 측정해야 한다.

실행 코드는 `repository/`에 있다. 현재 환경의 Python으로 다음 CPU 검증을 실행했다.

```sh
python -m pytest -q test_graph_scoring.py test_lb_data_contract.py test_lb_training_contract.py test_lb_acquisition.py
python data_v4/build_manifest.py --self-check
python data_v4/export_bank.py --self-check
python data_v4/file_ensemble.py --self-check
python candidate_data_acquire.py --self-check
```

기존 `test_full_bank_training.py::test_full_bank_training_contract`도 원본 자산의 실제 fc5 정의를 사용해 확인했다. API 거절 처리 테스트의 mock은 분기·token 비노출 회귀 검사용이며 실제 다운로드·업로드 증거는 별도 영수증에 있다. 패키징 검증은 작은 archive fixture와 실제 head serialization을 사용했다. 전체 encoder 추론·음질·LB 개선·L4 적합성의 근거로 쓰지 않는다.

## 판단 기록

- 원본 로컬 history 전체 대신 지정 원격 저장소를 별도 checkout했다. 기존 작업과 미검토 history를 보존하기 위한 선택이다.
- 새 라이브러리·전역 설정·유료 자원을 추가하지 않았다. 추가 payload 한도 20 GiB, 여유 공간 30 GiB를 유지했다.
- GPU 결과가 선행 조건인 실험을 임의로 채택하지 않았다. 지원하지 않는 loss/head 설정은 조용히 무시하지 않고 거부한다.
- 최종 검토에서 캐시가 자기 index만 만족하면 다른 encoder도 받아들이는 문제, head 내부 설정과 포장 자산 연결이 빠진 문제를 고쳤다. 잘못된 encoder·checkpoint 설정을 거부하는 회귀 검증을 추가했다. 최종 검토 결과 남은 중대 지적은 없다.
- 기존 workspace `wsp-9683432r72m7`는 stopped 상태를 유지하며 새 compute를 만들지 않았다. 새 모델 성능·독립 평가·공식 점수 개선은 아직 미확인이다.
