# 외부 모델 후보

확인일: 2026-09-21 KST. 아래 목록은 선별한 오디오 관련 저장소의 **GitHub 스타 내림차순**이다. 전체 GitHub 순위나 DACON 성능 순위가 아니다. WavLM의 수치는 UniLM 전체, SpeechBrain·S3PRL·fairseq의 수치는 각 도구 모음 전체에 붙은 스타다.

| 모델 / 저장소 | 스타 | 역할 | 가중치·배포 | 검토 메모 |
|---|---:|---|---|---|
| [Whisper](https://github.com/openai/whisper) | 109,400 | 음성 표현 / ASR | [공식 경로](https://huggingface.co/openai/whisper-small) | 신규 비교 후보; ASR 결과 자체는 위조 확률이 아님 |
| [wav2vec 2.0 / XLS-R](https://github.com/facebookresearch/fairseq) | 32,221 | 음성 자기지도 표현 | [공식 경로](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | 신규 비교 후보; 저장소 archived, 환경 고정 필요 |
| [WavLM](https://github.com/microsoft/unilm) | 22,220 | 음성 자기지도 표현 | [공식 경로](https://huggingface.co/microsoft/wavlm-base-plus) | 신규 비교 후보; 스타는 UniLM 전체 수치 |
| [SpeechBrain ECAPA](https://github.com/speechbrain/speechbrain) | 11,828 | 화자 임베딩 / 음성 도구 | [공식 경로](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | 보조 후보; 화자 중복 분석용, 위조 분류기 아님 |
| [HTDemucs](https://github.com/facebookresearch/demucs) | 10,363 | 음성·반주 분리 | [공식 경로](https://github.com/facebookresearch/demucs#pre-trained-models) | 기존 구성 요소; 새로운 detector가 아님 |
| [S3PRL](https://github.com/s3prl/s3prl) | 2,567 | 음성 encoder 비교 도구 | [공식 경로](https://github.com/s3prl/s3prl) | 지원 도구; 라이브러리 자체가 단일 체크포인트는 아님 |
| [CLAP](https://github.com/LAION-AI/CLAP) | 2,278 | 오디오 의미 임베딩 | [공식 경로](https://huggingface.co/laion/clap-htsat-unfused) | 신규 비교 후보; 의미 특징이 생성 흔적을 보장하지 않음 |
| [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) | 1,781 | 음성·음악 존재 / AudioSet 클래스 | [공식 경로](https://zenodo.org/records/3987831) | 기존 구성 요소; presence 확률용 |
| [AST](https://github.com/YuanGongND/ast) | 1,479 | 오디오 분류 표현 | [공식 경로](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | 비교 후보; 전용 위조 head 학습 필요 |
| [MERT330M](https://github.com/yizhilll/MERT) | 511 | 음악 표현 | [공식 경로](https://huggingface.co/m-a-p/MERT-v1-330M) | 기존 음악 encoder; 코드 Apache-2.0와 가중치 CC-BY-NC-4.0 구분 |
| [AASIST / AASIST-L](https://github.com/clovaai/aasist) | 303 | 직접 음성 위조 탐지 | [공식 경로](https://github.com/clovaai/aasist/tree/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/weights) | 신규 비교 후보; 원본 코드와 실제 가중치 2개 포함 |
| [SSL Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) | 181 | SSL 특징 + 위조 탐지 | [공식 경로](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB) | 신규 비교 후보; 사전학습 데이터 노출 확인 필요 |

## 실행 우선순위

신규 음성 비교 후보는 Whisper encoder, WavLM/XLS-R, AASIST/SSL Anti-spoofing이다. 음악에는 CLAP·AST를 기존 MERT와 비교한다. 이는 역할을 고려한 실험 제안이며, 스타 순서를 성능 순서로 해석하지 않는다. 새로운 encoder를 붙이면 피처를 다시 추출하고 기존 meta-model도 다시 학습·검증해야 한다.

Demucs·PANNs·MERT는 현재 프로젝트에 이미 쓰는 구성 요소다. 외부 자원 목록에 등록했다고 새 모델이 기존 추론에 통합되는 것은 아니다. SpeechBrain의 화자 임베딩은 중복 분석용이며 fake 확률로 직접 쓰지 않는다.

가중치 revision과 코드/가중치 라이선스는 MODELS.csv 및 SOURCES.json에서 확인한다. WavLM card에 license 필드가 없어서 미확인으로 남겼다. 코드 저장소의 MIT를 가중치 전체에 자동 적용하지 않는다.

AASIST 공식 ZIP에는 원래 모델 코드, 설정, MIT LICENSE, AASIST.pth와 AASIST-L.pth가 들어 있다. 다운로드·ZIP 무결성·파일 해시만 검사했으며, 이 작업에서 모델을 로드하거나 탐지 성능을 측정하지 않았다.
