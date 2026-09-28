# S2 EAT 제출 파일

제출할 파일은 이 폴더의 `submit.zip`이다. 크기는 6,239,402,263바이트(약 6.24GB)로 ZIP 10GB 제한 이내다. 압축을 풀면 최상위에 `model/`, `script.py`, `requirements.txt`가 있다.

- 모델: `07_S2_EAT_fullft`, seed 41, tune 기준 선택 epoch 4.
- AudioLDM을 train·tune에서 제외한 기존 실험의 체크포인트를 그대로 사용했다. 재학습하지 않았다.
- 음악: 기존 고정 HTDemucs → 5초 구간 8개 → 전체 학습한 EAT → CompactHead.
- 음성·존재 여부: 기존 DF-Arena 1B·PANNs를 새 입력마다 실행한다. 과거 예측값이나 평가 데이터는 ZIP에 넣지 않았다.
- File 확률: `max(VOICE_PRESENT_PROB * VOICE_FAKE_PROB, MUSIC_PRESENT_PROB * MUSIC_FAKE_PROB)`.
- EAT의 학습된 인코더와 분류기 가중치를 분리 저장했다. 원본 체크포인트의 텐서 314개와 모두 정확히 일치한다.

`package_verification.json`에 ZIP SHA-256, 원본 체크포인트 SHA-256, 학습 식별자와 크기를 기록했다. ZIP의 CRC와 각 멤버 SHA-256을 검증했다.

`runtime_verification.json`과 `offline_cpu.log`에는 빈 HF 캐시·네트워크 차단 상태에서 실행한 실제 WAV·FLAC 추론 결과가 있다. 샘플 제출 양식의 행 순서, 5개 확률의 범위와 File 결합식을 확인했다. 4초·5초·약 11초·60초 입력의 평가 구간은 원본 학습 코드와 정확히 일치했다.

검증 범위는 로컬 CPU 추론이다. L4 GPU에서 1,200개 파일을 60분 안에 처리하는지는 검증하지 않았다. GPU를 새로 켜거나 DACON에 업로드하지 않았다.

S2 로컬 Test Music EER은 0.1676029962546816이며, 공식 제출 총점은 아직 없다.

프로젝트 루트에서 재생성·검증:

```bash
.venv/bin/python package_s2_eat.py
.venv/bin/python verify_s2_submission.py
```
