**D1 — 위험과 guard에는 동의하지만, 제시한 지표는 수정해야 합니다.**

`fakeV+realM`의 MUSIC 라벨과 `realV+fakeM`의 VOICE 라벨은 모두 real입니다. **단일 클래스 slice에는 EER을 정의할 수 없습니다.**

대신 val에서 같은 fake-component TPR을 목표로 임계값을 정하고 고정한 뒤, 해당 slice의 **MUSIC FPR / VOICE FPR을 seed별로 비교**하십시오. 비열등 허용폭도 결과를 보기 전에 정해야 합니다. EER을 쓰려면 미리 고정한 대응 fake-component 양성군을 포함해야 합니다.

G3/G11은 이 guard를 추가해 조건부 유지합니다. G3 우선순위는 잠정 유지하되, “모든 파일의 VOICE에 융합된다”만으로 G11이 더 위험하다고 단정하지 않겠습니다. G3 역시 모든 파일의 MUSIC과 downstream FILE에 영향을 줍니다.

**D2 — 축소된 G2′는 KEEP입니다. 기존 기각을 수정합니다.**

**EnCodec 24 kHz 하나**, backing stem, 고정된 연속 latent 통계, 기존 head recipe로 제한하십시오. 새 표현을 시험하면서 성분 분리를 유지한다는 장점이 있습니다. 다만 `lyria-3 pro`·`fx_encoder`의 codec 계보와 이득은 **UNKNOWN**입니다.

“WavLM처럼 호출하면 된다”는 구현 추정은 조심해야 합니다. EnCodec의 `encode().audio_codes`는 **이산 코드**입니다. 연속 encoder latent 추출 경로를 확인하고, 코드 번호를 그대로 평균내지 마십시오. [공식 문서](https://huggingface.co/docs/transformers/model_doc/encodec)

구현 30분·추출 20분은 **ESTIMATE**로 두고, pilot부터 시작해 x4 총 **30분/$5 상한**을 제안합니다. 기존 RT·unseen gate와 D1 guard를 통과한 뒤, 실제 ZIP·오프라인 parity·런타임 비용을 확인해야 합니다. S7 완성을 늦추거나 상한을 넘으면 종료하십시오.