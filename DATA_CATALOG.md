# 외부 데이터 후보

GitHub 저장소가 있는 후보는 해당 저장소의 스타순으로 정리했다. GitHub가 공식 배포처가 아닌 데이터는 스타를 0으로 만들지 않고 N/A로 표시했다. HF 좋아요 수를 GitHub 스타와 합산하지 않는다. 아래는 다운로드 카탈로그이며 원본 음원 업로드 목록이 아니다.

| 데이터 | GitHub 스타 | 용도 | 다운로드 | 사용 조건·주의점 |
|---|---:|---|---|---|
| FMA | 2664 | 실제 음악 | [공식 원천](https://github.com/mdeff/fma#data) | 메타데이터 CC BY 4.0; 음원은 트랙별 조건. 기존 사용 원천. artist/track 단위 중복 확인; 보컬 유무 별도 라벨링 |
| ASVspoof 2019/2021 | 256 | 실제·합성 음성 / 전화채널 | [공식 원천](https://www.asvspoof.org/) | 공식 배포별 데이터 조건 확인; 코드 라이선스로 음원 사용권을 추정하지 않음. 스타는 2021 baseline 저장소. 기존 2019 학습/선택 노출 확인; 2021도 자동 독립 아님 |
| WaveFake | 93 | 합성 음성 / vocoder 비교 | [공식 원천](https://zenodo.org/records/5642694) | 원본 Zenodo 레코드와 기반 음원 조건 확인. 가짜 데이터와 원본 real의 문장·화자를 함께 그룹화 |
| SONICS | 58 | AI 생성 노래 | [공식 원천](https://huggingface.co/datasets/awsaf49/sonics) | 데이터 CC BY-NC 4.0; 코드·모델 MIT. 배포 음원은 fake만 포함. real_songs.csv의 YouTube ID는 실제 음원 포함을 뜻하지 않음; 기존 노출 점검 |
| MUSAN | N/A | 음성·음악·잡음 / 강건성 | [공식 원천](https://openslr.org/17/) | CC BY 4.0. 공식 GitHub 스타 없음. fake 라벨 데이터가 아닌 잡음·실제 성분 원천 |
| MLAAD-tiny | N/A | 다양한 TTS와 실제 음성 | [공식 원천](https://huggingface.co/datasets/mueller91/MLAAD-tiny) | 합성 CC BY-NC 4.0; real은 original/LICENSE. 공식 HF 원천. 기존 사용·base encoder 학습 노출 확인; 원문장과 TTS 변형은 같은 그룹 |
| AIME | N/A | 여러 생성기의 음악 | [공식 원천](https://huggingface.co/datasets/disco-eth/AIME) | 생성 음원 CC BY 4.0; 태그 CC BY-NC-SA 4.0; 실제 트랙별 조건. 공식 HF 원천. prompt/generator 단위 중복 확인; 음악 데이터라는 이유로 VOICE_PRESENT=0을 부여하지 않음 |
| In-the-Wild | N/A | 실제 환경의 음성 위조 평가 | [공식 원천](https://huggingface.co/datasets/mueller91/In-The-Wild) | HF card CC BY-SA 4.0. 공식 저자 배포. 인물·발화 그룹 및 기존 모델의 학습 노출 확인 후 평가 용도 결정 |

## 실험에 넣는 절차

원본 ID, SHA-256, source, speaker/artist, track/utterance, generator_family, 부모 ID, 라이선스, 기존 학습·선택 노출을 먼저 기록한다. 원본 단위로 meta-dev와 locked OOD를 나눈 뒤 crop·코덱 변환·혼합물을 만든다. 같은 원본의 파생물이 split을 넘으면 안 된다.

이 프로젝트에서 이미 사용한 ASVspoof2019·FMA·SONICS·MLAAD 등의 파일을 다시 받아도 새 독립 데이터가 되지 않는다. 데이터 전체 이름이 아니라 실제 샘플과 모델 노출 이력을 확인한다. 기존 val/test는 새 meta-dev로 재명명하지 않는다.

음악 원천이라는 이유만으로 보컬 없음, 위조 음악이라는 이유만으로 위조 음성이라고 라벨링하지 않는다. 알 수 없는 성분 라벨은 mask=0으로 남긴다. 공식 점수와 로컬 EER, 파일 저장 검사 결과를 구분한다.

한국어·전화채널·최근 생성기 범위는 이 스타 기반 목록만으로 충족되지 않는다. 실제 목표 분포의 부족분은 별도 수집·라벨 감사로 보완해야 한다.
