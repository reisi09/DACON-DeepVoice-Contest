# Lane 1: competition contract and ranking. Evidence log (2026-09-27, KST)

Files in this folder:
- `official_text.txt`: text extracted from `/tmp/dacon_rules.json` → `fetch["CompetitionsName:2"].competitionDetail` (fields description, data_explain, schedule, rule, evaluation_rule, prize_text, agreement, scalars). The seven dumps all hold the same competitionDetail.
- `evaluation_rule_img0.png` / `prize_text_img0.png`: decoded embedded images (the 2nd-round rubric and the prize table).
- `competition_detail_20260927T1333KST.json`: live competitionDetail from https://dacon.io/competitions/official/236749/leaderboard (fetched 2026-09-27 13:33:40 KST). Diff against the 09-26 dump: only m_time (→2026-09-27 11:47:55), now_time, participants (1009→1014), user_count and period_dday changed. The rule, evaluation and schedule text is byte-identical after JS decoding.
- `leaderboard_20260927T1333KST.json`: 465 rows from the leaderboard NUXT payload `data[2].result_full` (ranking, team, score=total, ads=score1, cps=score2, cnt, c_time).
- `report_template/`: the official HWP template zip (61,375 B, linked from the rules tab) and the text I extracted from both HWPs with a minimal OLE/CFB reader (BodyText/Section0, HWPTAG_PARA_TEXT).

Method: `curl` of the official pages, then `node` to eval `window.__NUXT__`. Scripts are in my scratchpad (`lane1/ev*.js`). The talkboard and codeshare were re-fetched at 13:34 KST: same 11 posts on page 1, same reply counts, and codeshare still holds only the baseline (cs 14153).

## Key raw observations
- Live LB 13:33 KST: killerpjw rank 123/465, total 0.7928034339, ADS 0.7700396825, CPS 0.9976771958, cnt 23, c_time 2026-09-27 01:07:13 (sub_id 1585448 = P1). Members goodpjw2008 and reisi09.
- #1 team "55" 0.9004585291 (ADS 0.8896587, 04:13 today). #15 Sia_Le_Blanc 0.8736806667 (ADS 0.8609048, 13:04 today). #16 0.8733253545. #20 0.8693277. #30 0.8597647. #50 0.8409174. #100 0.8016539.
- Rank a given total would get now: 0.80→104, 0.82→76, 0.84→52, 0.86→28, 0.87→20, 0.8737→15, 0.88→10.
- 14 of the top 15 recorded their best within the last 24 h, so the cut is moving.
- Official baseline total is 0.6909110688 (55 teams sit exactly there). 226 teams keep the baseline CPS 0.9893249735, including ranks 10, 11 and 16. CPS is saturated, so ADS is the whole race.
- To match the cut with our CPS: ADS ≥ 0.859903 (+0.08986 over P1), i.e. weighted EER 0.2300 → ≤0.1401.
- 2nd-round performance points = 30·(score/best)^N. At the current cut vs #1 that is 29.1 (N=1) to 25.8 (N=5).
- The FAQ "1일 제출 횟수 초기화는 언제 되나요?" exists as faq_id 93 (dacon.io/more/faq payload). Its body (`faq/93.html`) returned 404 or 403 on cfiles, r2-images, r2-library and the S3 bucket, so the reset time is UNVERIFIED.
- Agreement §4: "시간대는 한국 표준시(UTC + 9)를 기준으로 합니다." LB c_time values are KST: 13:04 is at or before the 13:33 observation, and in UTC it would lie in the future.
- `final_use_type: "SCORE"`, `num_final_sub: 1`, `same_score_standard: 1`, `max_file_per_day: 3`, `max_file: 500`, `submission_end: 2026-09-29 09:59:59`, `container_time_limit: 3600`, `install_time_limit: 600`.

## S3 package checks (read-only; small members extracted to scratchpad)
- The zip top level is exactly `model/`, `script.py`, `requirements.txt` (62 entries, 8,286,530,714 B expanded).
- `script.py` and `requirements.txt` (`torchvision==0.22.1+cu128` + cu128 extra index) have the same sha256 as in P1, which DACON scored. So the install path and entry point are verified on the server.
- Packaged `model/runtime/full_bank_submission.py` (sha 8a7c6d92…, matching the manifest) differs from the worktree copy only in line 20, the `CANDIDATE` string.
- New vs P1, never run on the official L4: AntiDeepfake-1B load and `detector.windows` (voice stem, 50% overlap windows), MERT layers [6, 12], `component_fusion.fuse`, fp32 heads, and the 48-min deadline (`full_bank_submission.py:133,165`).
- verify.log: CPU run of 10 files in 84.9 s, offline guard on, file_independence_max_abs_error 0.0, `official_l4_validated: false`.
