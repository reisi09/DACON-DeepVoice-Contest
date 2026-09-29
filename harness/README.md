# 에이전트 하네스: omp, Jev, 플러그인

이 저장소의 기본 에이전트는 omp(Oh My Pi)입니다. Claude Code(클라우드 세션)는 같은 지시문(`AGENTS.md`)과 같은 스킬 폴더(`.claude/skills/`)를 씁니다. 플러그인 목록은 `plugins.json` 한 곳에서 관리합니다.

## 지금 켜져 있는 것

| 항목 | 위치 | 확인한 것 |
|---|---|---|
| 공통 지시문 | `AGENTS.md`(omp·Codex가 직접 읽음), `CLAUDE.md`(`@AGENTS.md` 가져오기) | omp는 루트의 `AGENTS.md`를 읽고, 같은 깊이의 `CLAUDE.md`보다 우선합니다(omp `docs/context-files.md`) |
| omp 프로젝트 설정 | `.omp/config.yml` | 저장소 안에서 `omp config list`를 실행하면 `features.unexpectedStopDetection = smart`, 밖에서는 기본값 `mechanical`로 나옴 |
| Jev(omp 내장) | `.omp/config.yml`의 `unexpectedStopDetection: smart` | 글만 남기고 멈춘 턴을 `judge` 역할(typesafe/jev-latest)이 판정하고, 너무 이른 멈춤이면 이어서 진행합니다. `TYPESAFE_API_KEY`가 없으면 세션 모델이 대신 판정하며, 멈출 때마다 채팅 호출 1회 비용이 듭니다 |

## 플러그인 목록 (`plugins.json`)

아래 7개는 모두 **승인 대기**입니다. 제3자 코드가 매 세션 훅으로 자동 실행되는 일이라, 저장소 소유자가 승인한 뒤에만 넣습니다. 커밋은 모두 고정했습니다.

| 이름 | 하는 일 | 출처 @ 커밋 | 라이선스 | 외부 전송 | Claude Code | omp |
|---|---|---|---|---|---|---|
| ponytail 4.10.0 | 최소 코드 원칙(YAGNI, 표준 라이브러리 우선)을 매 턴과 서브에이전트에 주입 | DietrichGebert/ponytail @ e3ba2aa | MIT | 없음 | 훅 3개(SessionStart, SubagentStart, UserPromptSubmit)와 스킬 6개 | Pi 확장 `pi-extension/index.js`(omp의 Pi 호환 shim), 스킬 |
| caveman 2.7.0 | 짧게 답하는 모드(출력 토큰 약 75% 감소), `/caveman lite\|full\|ultra` | JuliusBrussee/caveman @ 2fd153c | MIT(`skills/`, `src/hooks/`만 가져옴. 엔진 폴더는 BSL이라 제외) | 없음 | 훅 2개(SessionStart, UserPromptSubmit)와 스킬 6개 | 스킬만 |
| jev-belay 0.2.0 | Jev Stop 게이트: 파일을 바꿨는데 그 뒤 통과한 테스트·빌드가 없을 때만 Jev가 마지막 메시지를 판정 | valentynkit/jev-belay @ ef719db | MIT | 키가 있고 검증 안 된 수정이 있을 때만 api.typesafe.ai로 전송(마지막 메시지와 근거 개수, 가림 처리) | Stop 훅, 스킬 2개(`jev-belay-doctor`, `jev-belay-why`) | 해당 없음(omp는 내장 Jev 사용) |
| jevmem | Jev가 기억할 가치가 있는 턴을 `JEVMEM.md`에 한 줄로 저장하고, 프롬프트마다 관련 기억을 불러옴 | Avinash-jetwani/jevmem @ 87c99dc(directory 브랜치), CLI 0.5.9 | MIT | `jevmem enable` 이후 프롬프트, 직전 두 턴, 기억 최대 200줄. CLI 0.5.8 미만은 `PGPASSWORD=` 값을 가리지 않음 | 훅 2개(`jevmem enable` 전에는 아무것도 안 함) | 해당 없음 |
| building-with-typesafe-jev 0.3.0 | Jev API, 확신도 라우팅, 안티패턴, 선행 사례 목록 | aaddrick/building-with-typesafe-jev @ db95c54 | MIT | 키가 있으면 시험 호출 몇 번 | 스킬 | 스킬 |
| omp-utility-skills | `omp-docs`(omp 소스를 새로 받아 최신 문서로 답함), `omp-config` | hkay-dev/omp-utility-skills @ 6c75f09 | MIT | omp 저장소 git clone | 스킬 2개 | 스킬 2개 |
| omp-delegate 0.5.0 | 작업 지시서를 omp headless(`omp --mode json`)에 넘기고, 커밋 전에 diff를 검토 | amElnagdy/delegate-skills @ 6826b36 | MIT | omp가 쓰는 모델 제공자 | 스킬 | 해당 없음 |

넣지 않는 것:
- **Computer(`computer-use@openai-bundled`)**: Codex에 번들된 플러그인이라 Claude Code에는 없습니다. 클라우드 컨테이너에는 조작할 데스크톱도 없습니다.
  - omp에는 브라우저·컴퓨터 도구가 내장돼 있습니다.
  - Claude 데스크톱 앱에서는 claude.ai의 computer-use 스킬을 씁니다.

## 왜 `enabledPlugins`가 아니라 복사(vendoring)인가

- 클라우드 세션은 저장소 `.claude/settings.json`의 `enabledPlugins`와 `extraKnownMarketplaces`를 설치하지 않습니다([cloud-environments 문서](https://code.claude.com/docs/en/cloud-environments)). 저장소의 `.claude/skills/`와 `.claude/settings.json` 훅은 로드됩니다.
- `.claude/skills/<이름>/.claude-plugin/plugin.json` 형태는 `<이름>@skills-dir` 플러그인으로 로드되지만, 작업 공간이 신뢰됐을 때만입니다. 신뢰 기록이 없는 폴더에서는 `claude -p`에서도 건너뛰는 것을 직접 확인했습니다. 신뢰를 기록한 설정으로 같은 폴더를 돌리면 플러그인 훅이 실행됐습니다.
- 그래서 각 플러그인의 훅 스크립트는 `harness/vendor/<이름>/`에, 스킬은 `.claude/skills/`에 복사합니다. `.claude/settings.json`은 `claude-settings.base.json`에 플러그인 훅을 합쳐 생성하고, 각 훅은 `CLAUDE_PLUGIN_ROOT`를 복사본으로 지정해 실행합니다.

## 승인 후 실행

```bash
python3 harness/sync_plugins.py          # 고정 커밋을 받아 vendor, .claude/skills, .claude/settings.json, plugins.lock.json 작성
python3 harness/sync_plugins.py --check  # 오프라인 검사: 복사본이 lock과 같은지, 훅 스크립트가 있고 파싱되는지, settings가 목록과 일치하는지
```

- 플러그인을 빼려면 `plugins.json`에서 지우고 다시 sync합니다(지운 스킬과 vendor 폴더도 정리됨).
- 버전을 올리려면 `commit`을 바꾸고 sync한 뒤 diff를 검토합니다.
- `.claude/settings.json`은 생성 파일이라 직접 고치지 말고 `claude-settings.base.json`을 고칩니다.
- base 설정에는 두 가지가 들어 있습니다.
  - `bin/session-start.sh`: 클라우드 세션에서만 Bun 1.4.2(omp에 1.3.14 이상 필요), omp 18.4.3, jevmem CLI 0.5.9를 설치합니다. 이미 있으면 건너뜁니다.
  - jev-belay 기본값: `JEV_BELAY_SHADOW=1`(막지 않고 기록만), `JEV_BELAY_LOG=1`(`~/.claude/belay/decisions.jsonl`), `JEV_BELAY_TIMEOUT_MS=8000`.

## 소유자가 할 일

1. **플러그인 승인:** 넣을 플러그인을 정합니다. 승인 전에는 아무것도 자동 실행되지 않습니다.
2. **클라우드 환경 설정:** 세션 제목 표시줄의 환경 메뉴 → Edit에서 설정합니다.
   - 환경 변수에 `TYPESAFE_API_KEY`와 omp용 모델 제공자 키(`ANTHROPIC_API_KEY` 등)를 추가합니다.
   - Network access의 허용 도메인에 `api.typesafe.ai`를 추가합니다. 현재는 막혀 있고, npm과 GitHub git은 열려 있습니다.
3. **claude.ai 개인 설정:** `[@ponytail](plugin://…) [@caveman](…) [@Computer](…)`는 Codex 문법이라 Claude Code에서는 동작하지 않으니 지웁니다. 로드는 이 목록이 담당합니다.
4. **Jev 게이트 조정:** 기록이 100–300개 쌓이면 AUROC와 잘못 막은 비율을 보고 나서 `JEV_BELAY_SHADOW`를 끕니다. 한국어 입력은 정확도가 약 6.5%p 낮다는 보고가 있어서 따로 확인합니다.
