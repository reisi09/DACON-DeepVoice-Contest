# Manual QA: actualV4 Colab UI readiness

This attempt records the initial read-only readiness check and the subsequently authorized V4 Colab UI executions. The final matched recheck was run once on A100, completed, and released; no production code was edited.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-UI-R0 | Readiness: open existing Colab tab and verify quota/runtime state | Aside (`at.studio.AsideBrowser`) > existing tab `DACON_236749_recovery_v3.ipynb - Colab` > `Resources` panel | `sky.click({app:"at.studio.AsideBrowser", element_index:366})`; then `sky.get_app_state({app:"at.studio.AsideBrowser"})` | PASS | A1, A2 |
| V4-UI-R1 | Refresh readiness immediately before V4 launch | Aside (`at.studio.AsideBrowser`) > existing V3 Colab tab > `Resources` panel | `sky.get_app_state({app:"at.studio.AsideBrowser"})`; after concurrent foreground change, `sky.click({app:"at.studio.AsideBrowser", element_index:783})`; then `sky.get_app_state({app:"at.studio.AsideBrowser"})` | PASS | A3, A4 |
| V4-UI-R2 | Confirm the first allocation was released before retry | Aside > V4 Colab `Resources` panel | `sky.press_key({app:"at.studio.AsideBrowser",key:"Escape"})`; then `sky.get_app_state({app:"at.studio.AsideBrowser",disableDiff:true})` | PASS | A5, A6 |
| V4-UI-R3 | Reload the updated V4 notebook and launch one fresh A100 run | Aside > `DACON_236749_trainable_v4.ipynb - Colab` | `sky.click({app:"at.studio.AsideBrowser",element_index:6})`; after fresh state, `sky.click({app:"at.studio.AsideBrowser",element_index:79})`; then `sky.click({app:"at.studio.AsideBrowser",element_index:74})` | PASS | A7, A8, A9 |
| V4-UI-R4 | Verify retry runtime and execution heartbeat | Aside > V4 Colab runtime | `sky.get_app_state({app:"at.studio.AsideBrowser",disableDiff:true})` after Drive OAuth and 45-second monitoring intervals | PASS | A10, A11, A12 |
| V4-UI-R5 | Confirm completed retry has released the A100 session | Aside > V4 Colab `Resources` panel | `sky.get_app_state({app:"at.studio.AsideBrowser",disableDiff:true})` after the completed run's runtime release | PASS | A13, A14 |

Observed state in the fresh accessibility tree: `사용 가능: 컴퓨팅 단위 53.34개`; `사용률: 약 0/시간`; `활성 세션이 0개 있습니다.`; `런타임에 연결되어 있지 않습니다.`; A100 action exposed as `다시 연결 ... A100 연결하려면 클릭`.

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-UI-A0 | Readiness must target the named existing notebook | wrong-tab / stale-index | After selecting the current tab, the fresh state identifies `DACON_236749_recovery_v3.ipynb - Colab`; no prior tab index is reused after the action. | PASS | A2 |
| V4-UI-A1 | Readiness must detect an already active runtime before any launch | active-runtime conflict | Resources state must show active sessions and connection state; current state must be zero active sessions and disconnected before launch. | PASS | A2 |
| V4-UI-A2 | Readiness must not consume GPU units | unintended side effect | Before launch, no connect/run action is invoked; the observed utilization remains `약 0/시간`. | PASS | A2 |
| V4-UI-A3 | Refresh must account for concurrent Aside tab changes | concurrent-user / stale-index | If another tab becomes foreground, re-derive the current Colab tab index from a fresh tree before clicking; do not use the prior index. | PASS | A3, A4 |
| V4-UI-A4 | Retry must not reuse a live or stale first-attempt runtime | resource-leak / stale-session | Retry begins only after Resources shows 0 active sessions and disconnected runtime; observed 52.51 units and 0 sessions. | PASS | A5, A6 |
| V4-UI-A5 | Drive authorization can interrupt Run all | authorization-prompt | When Colab presents the Drive access prompt, complete the already-authorized account flow and verify execution resumes on the same A100 runtime. | PASS | A9, A10 |
| V4-UI-A6 | Fresh retry must use the updated notebook source | stale-content / wrong-tab | The selected tab URL is the supplied V4 ID and the title is `DACON_236749_trainable_v4.ipynb`; no V3 execution occurs. | PASS | A7, A8 |
| V4-UI-A7 | Completed run must release the compute session | resource-leak / post-run | After the run finishes, Resources must show 0 active sessions, disconnected runtime, and ~0/hour utilization. | PASS | A13, A14 |

### artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | screenshot | Native Aside screenshot of the Colab Resources panel showing quota, active sessions, disconnected runtime, and A100 connect affordance. | `.omo/evidence/v4-colab-ui/v4-readiness-resources.png` |
| A2 | accessibility-log | Fresh accessibility tree and exact invocation recorded immediately after opening the existing Colab tab. | `.omo/evidence/v4-colab-ui/v4-readiness-accessibility.txt` |
| A3 | screenshot | Native Aside screenshot from the refreshed V3 Colab Resources panel after reselecting it from a fresh tab tree. | `.omo/evidence/v4-colab-ui/v4-readiness-refresh.png` |
| A4 | accessibility-log | Fresh refreshed accessibility tree; records Gmail foreground detection, re-derived Colab tab index, and unchanged resource state. | `.omo/evidence/v4-colab-ui/v4-readiness-refresh-accessibility.txt` |
| A5 | screenshot | Native Aside screenshot of V4 Resources view before retry: 52.51 units, 0 active sessions, disconnected runtime. | `.omo/evidence/v4-colab-ui/v4-retry-preconnect-released.png` |
| A6 | accessibility-log | Fresh V4 Resources state proving first attempt release before retry. | `.omo/evidence/v4-colab-ui/v4-retry-preconnect-released-accessibility.txt` |
| A7 | screenshot | Native Aside screenshot of updated V4 notebook/resources view immediately before retry connection. | `.omo/evidence/v4-colab-ui/v4-retry-preconnect-final.png` |
| A8 | accessibility-log | Fresh updated V4 title and retry controls, including A100 reconnect and Run all affordances. | `.omo/evidence/v4-colab-ui/v4-retry-preconnect-final-accessibility.txt` |
| A9 | accessibility-log | Retry allocation and Run all invocation receipt with Unix timestamps. | `.omo/evidence/v4-colab-ui/v4-retry-a100-connecting-accessibility.txt` |
| A10 | screenshot | Native Aside screenshot showing retry execution on A100 after Drive authorization. | `.omo/evidence/v4-colab-ui/v4-retry-run-heartbeat-1m18.png` |
| A11 | screenshot | Native Aside screenshot at retry heartbeat 40 seconds. | `.omo/evidence/v4-colab-ui/v4-retry-run-heartbeat-40s.png` |
| A12 | screenshot | Native Aside screenshot at retry heartbeat 3 minutes 24 seconds. | `.omo/evidence/v4-colab-ui/v4-retry-run-heartbeat-3m24.png` |
| A13 | screenshot | Native Aside screenshot of completed retry Resources view showing 49.02 units, 0 active sessions, disconnected runtime, and ~0/hour utilization. | `.omo/evidence/v4-colab-ui/v4-retry-release-zero-sessions.png` |
| A14 | accessibility-log | Fresh post-run release state for the V4 tab. | `.omo/evidence/v4-colab-ui/v4-retry-release-zero-sessions-accessibility.txt` |

### matchedRecheckEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-UI-R6 | Matched recheck source must be opened fresh with no embedded outputs | Aside (`at.studio.AsideBrowser`) > `DACON_236749_v4_matched_recheck.ipynb - Colab` | Open the supplied Drive URL in a new tab; `sky.get_app_state({app:"at.studio.AsideBrowser",disableDiff:true})` | PASS | A15, A16 |
| V4-UI-R7 | Requested A100 runtime must be connected before execution | Aside > matched recheck runtime | Record allocation start Unix `1789288464`; click the fresh A100 connect control; `sky.get_app_state({app:"at.studio.AsideBrowser",disableDiff:true})` | PASS | A17, A18 |
| V4-UI-R8 | Run All must execute the matched recheck and handle authorized Drive access | Aside > matched recheck notebook and Google consent flow | Record Run All Unix `1789288485`; click fresh `모두 실행`; click `Google Drive에 연결`, account `gameplayevs@gmail.com`, then fresh `계속`; refresh state after each action | PASS (authorization resumed; run later failed in notebook) | A19, A20, A21, A22 |
| V4-UI-R9 | Notebook failure must be visible and runtime must be released without a retry | Aside > matched recheck traceback, Runtime release confirmation, Resources | Read fresh state after 45-second heartbeat; inspect `FileNotFoundError`; Runtime > `런타임 연결 해제 및 삭제` > `예`; Runtime > `리소스 보기` | PASS for surfaced failure and release; matched recheck execution FAIL | A23, A24, A25, A26 |

### matchedRecheckAdversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-UI-A8 | Matched recheck must not use an old notebook | wrong-tab / stale-content | Fresh state identifies the distinct matched recheck URL/title and retains the explicit no-selection/no-sealed-data text. | PASS | A15, A16 |
| V4-UI-A9 | Allocation timestamp must bind the actual A100 run | timing / allocation provenance | Allocation is recorded immediately before connect and runtime identity is confirmed as `A100(Python 3)` before Run All. | PASS | A17, A18, A19 |
| V4-UI-A10 | Drive authorization interruption must resume the same run | authorization-prompt | Existing authorized account consent completes, then the notebook remains on the same matched recheck runtime and reports execution in progress. | PASS | A20, A21, A22 |
| V4-UI-A11 | Missing runtime path must not be mistaken for completed training | expected-failure / traceback | The exact `FileNotFoundError` is visible; downstream cells remain unexecuted and no training result is claimed. | PASS | A23, A24 |
| V4-UI-A12 | Failed execution must not leak an A100 session | resource-leak / post-failure | After explicit release, Resources shows 0 active sessions, disconnected runtime, and approximately 0/hour utilization. | PASS | A25, A26 |

### finalMatchedRecheckEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-UI-R10 | Final source must be refreshed before launch and match the approved 10-epoch notebook | Aside (`at.studio.AsideBrowser`) > `DACON_236749_v4_matched_recheck.ipynb - Colab` | Fresh reload with `sky.get_app_state({app:"at.studio.AsideBrowser",disableDiff:true})`; verify title, ten-epoch methods text, and `/content/dacon_trainable_v4` source path | PASS | A27, A28 |
| V4-UI-R11 | Final matched recheck must run all cells on the requested A100 and complete authorized Drive setup | Aside > matched recheck notebook | Record allocation start `1789289054` and Run All start `1789289257` (`2026-09-13T08:47:37Z`); click fresh `모두 실행`; complete the already-authorized Drive flow; refresh state after each action | PASS | A29, A30, A31 |
| V4-UI-R12 | Final run must finish both matched arms and persist the result directory | Aside > completed matched recheck notebook | `sky.get_app_state({app:"at.studio.AsideBrowser",disableDiff:true})` after Run All completes; verify saved notebook, mounted Drive, results directory, and `matched_comparison.json` references | PASS | A32, A33 |
| V4-UI-R13 | Final A100 runtime must be released after completion | Aside > matched recheck `Resources` panel and session manager | Open Runtime > `리소스 보기`; then click `세션 관리`; `sky.get_app_state({app:"at.studio.AsideBrowser",disableDiff:true})` | PASS | A34, A35, A36, A37 |

### finalMatchedRecheckAdversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| V4-UI-A13 | Final launch must not execute stale 15-epoch or V3 content | stale-content / wrong-tab | Fresh notebook title and method text identify the final 10-epoch matched recheck; selected URL is the distinct matched-recheck ID. | PASS | A27, A28 |
| V4-UI-A14 | Run All must survive authorized Drive prompts without changing runtime | authorization-prompt / session continuity | Drive mount completes and subsequent cells execute on the same A100-backed notebook; no auth traceback appears. | PASS | A29, A30, A32 |
| V4-UI-A15 | Concurrent Aside foreground changes must not cause stale-index actions | concurrent-user / stale-index | When another tab becomes foreground, reselect the matched Colab tab from a fresh accessibility tree before inspection; final state remains the matched notebook. | PASS | A31, A32 |
| V4-UI-A16 | Completion must not leak GPU resources | resource-leak / post-run | After completion, the Resources view reports 47.54 available units, 0/hour usage, 0 active sessions, and disconnected runtime. | PASS | A34, A35, A36 |

### finalMatchedRecheckArtifactRefs

| id | kind | description | path |
|---|---|---|---|
| A27 | screenshot | Cropped native Aside screenshot immediately before the final 10-epoch launch, showing the matched-recheck title and approved source state. | `.omo/evidence/v4-colab-ui/redacted/v4-matched-recheck-final-10epoch-pre-run.jpg` |
| A28 | accessibility-log | Redacted fresh final source summary recording ten epochs per arm, `/content/dacon_trainable_v4`, A100 control, and no initial run output. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-final-10epoch-pre-run-redacted.txt` |
| A29 | screenshot | Cropped native Aside screenshot immediately after final Run All invocation while the 17-cell run is being allocated. | `.omo/evidence/v4-colab-ui/redacted/v4-matched-recheck-final-10epoch-runall.jpg` |
| A30 | accessibility-log | Redacted final allocation and Run All receipt with `allocation_start_unix=1789289054` and `run_start_unix=1789289257`. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-final-10epoch-runall-redacted.txt` |
| A31 | screenshot | Cropped native Aside active A100 heartbeat after setup and Drive authorization, showing the final matched run executing. | `.omo/evidence/v4-colab-ui/redacted/v4-matched-recheck-final-active-late.jpg` |
| A32 | screenshot | Cropped native Aside completed-run state after both arms finish, showing saved outputs and the disconnected A100 control. | `.omo/evidence/v4-colab-ui/redacted/v4-matched-recheck-final-completed-disconnected.jpg` |
| A33 | accessibility-log | Redacted completed notebook summary recording saved state, Drive mount/results directory, generated source files, and `matched_comparison.json` references. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-final-completed-disconnected-redacted.txt` |
| A34 | screenshot | Cropped native Aside Resources view after completion, showing 47.54 available compute units, 0/hour usage, 0 active sessions, and disconnected runtime. | `.omo/evidence/v4-colab-ui/redacted/v4-matched-recheck-final-resources-zero-session.jpg` |
| A35 | accessibility-log | Redacted Resources summary recording 47.54 available units, 0/hour usage, 0 active sessions, and `런타임에 연결되어 있지 않습니다.` | `.omo/evidence/v4-colab-ui/v4-matched-recheck-final-resources-zero-session-redacted.txt` |
| A36 | screenshot | Cropped native Aside screenshot of the session manager after completion; no active-session row is listed and the termination control is disabled. | `.omo/evidence/v4-colab-ui/redacted/v4-matched-recheck-final-session-manager.jpg` |
| A37 | accessibility-log | Redacted session-manager observation explaining that zero active sessions is taken from the explicit Resources value. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-final-session-manager-redacted.txt` |

### matchedRecheckArtifactRefs

| id | kind | description | path |
|---|---|---|---|
| A15 | screenshot | Native Aside screenshot of the distinct matched recheck notebook before connection, showing title, no embedded result, and the A100 control. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-preconnect.png` |
| A16 | accessibility-log | Fresh matched recheck source/title/no-output accessibility tree and URL. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-preconnect-accessibility.txt` |
| A17 | screenshot | Native Aside screenshot after allocation, showing the connected GPU backend. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-a100-connected.png` |
| A18 | accessibility-log | Allocation timestamp and verified `A100(Python 3)` runtime identity. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-a100-connected-accessibility.txt` |
| A19 | screenshot | Native Aside screenshot after fresh Run All invocation, before Drive consent interruption. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-runall.png` |
| A20 | screenshot | Native Aside Google Drive authorization prompt presented by the running notebook. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-drive-auth-prompt.png` |
| A21 | accessibility-log | Drive authorization prompt and authorized account flow evidence. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-drive-auth-prompt-accessibility.txt` |
| A22 | screenshot | Native Aside screenshot after consent returned to the same notebook, showing execution in progress on A100. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-heartbeat-auth-complete.png` |
| A23 | screenshot | Native Aside heartbeat with the surfaced `FileNotFoundError` traceback. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-heartbeat-3m.png` |
| A24 | accessibility-log | Exact failing cell/path and evidence that later cells were unexecuted. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-error-final-accessibility.txt` |
| A25 | screenshot | Native Aside release confirmation for the failed matched recheck runtime. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-release-confirmation.png` |
| A26 | screenshot | Native Aside Resources view after release: 48.73 compute units, 0 active sessions, disconnected runtime, approximately 0/hour. | `.omo/evidence/v4-colab-ui/v4-matched-recheck-released-resources.png` |
