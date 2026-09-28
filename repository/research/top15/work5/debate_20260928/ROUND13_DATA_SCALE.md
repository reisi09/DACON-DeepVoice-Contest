# Round 13 — how much data to add next (2026-09-28 13:45 KST)

Context since round 12: the new-data pipeline ran. Everything below is measured (same evaluation code for every config,
research/top15/work5/newdata/new_eval.py; fusion 1/0.5/1, CDF tables on the old val, FILE fixed 0/9/1).

## 1. Results (DB protocol; E1 = DB held-out 4,822; E2 = new-data held-out 12,426; E3 = AI-Hub Korean real 257, Mac only)

| config | E1 ADS | E2 ADS | E1+E3 ADS | E2 FILE / VOICE / MUSIC EER |
|---|---|---|---|---|
| s7_ref (S7 recipe, LB 0.8374) | 0.90899 | 0.89334 | 0.91001 | 0.1155 / 0.0879 / 0.1045 |
| W_ref (W weights, no new rows) | 0.90615 | 0.88469 | 0.90732 | 0.1248 / 0.1069 / 0.1051 |
| W_new (W + new rows) | 0.91231 | 0.93842 | 0.91439 | 0.0694 / 0.0474 / 0.0579 |
| W_newB (W_new, newval rows in the early-stopping val) | 0.91221 | 0.94031 | 0.91503 | 0.0669 / 0.0483 / 0.0553 |
| s7_new (S7 weights + new rows) | 0.91540 (+0.0064, CI .0013..0.0111) | 0.92185 | 0.91718 | 0.0843 / 0.0638 / 0.0774 |

E2 per source (s7_ref / W_new / W_newB / s7_new): AT-ADD speech FILE EER .161/.058/.056/.062 (real FPR .160/.009/.009/.011);
AT-ADD singing VOICE EER .080/.054/.060/.061, FILE EER .144/.094/.094/.151; AT-ADD music MUSIC EER .111/.057/.055/.083;
MUSDB real FPR .063/.027/.018/.018; SD-RR real FPR .048/.027/.027/.007; AI-OpenBMAT FILE EER .392/.354/.354/.336
(real FPR .378/.300/.311/.200); HAIM B3/B4/B6 fake TPR >= .995 for all.
E3 real FPR at old-val TPR-95 (s7_ref / W_ref / W_new / W_newB / s7_new): AI-Hub 465 singing .067/.067/0/0/0,
473 guide vocals .053/.107/.040/.040/.027, 71544 Korean pop covers 0 everywhere.

## 2. What the new rows were (training pool; newval/newtest held out by lineage group)
AT-ADD T2 train+dev: speech 6,945 (+1,055 newval), singing 7,798 (+1,202), music 5,996 (+1,004); eval pool 10,000 = newtest.
Open data train: HAIM B6 562 / B3 384 / B4 384 (fake), MUSDB 249 (real), AI-OpenBMAT 345, GTSinger-ko 257 + speech 120,
SD-RR 279, D1 remixes 636 (515 fake / 121 real). Total new rows with features 40,214 (train ~24,500).
Bank v4 P1 training rows: 37,108 (real 12,967 / fake 24,116).

## 3. The next step under decision
User plan: rent a Korean GPU (Elice Cloud) ONLY for feature extraction of AI-Hub data (AI-Hub terms: no export abroad,
no third-party provision). AI-Hub features therefore can only be trained on in Korea = this Mac (24 GB RAM, CPU/MPS;
head training of S7-size P1 on the Mac CPU took roughly an hour per final earlier; with ~2x rows it may take 2 h+).
Measured runtime-exact extraction speed: ~18 audio-seconds per second per H100 (fp32, no TF32; 2 workers/GPU).
Available AI-Hub (all REAL, Korean): 71544 music-similarity (Korean pop covers; Validation 1,000 songs + Training ~8,000),
473 guide vocals (a cappella; ~3,600 files), 465 multi-singer singing (~3,000+), 71767 pop loops (instrumental), 71470
gugak, 123/571/100/87 Korean speech (spontaneous, phone, call-center). No AI-Hub fakes exist.
Deadline: slots at 2026-09-29 00:00-10:00 KST (3 submissions left, best-ever counts). Package freeze ~08:00.
Budget: VESSL credit ~$30 (H100x1 $2.98/h); Elice ~KRW 50-80k acceptable.

## 4. Questions (answer in Korean, short, mark VERIFIED / INFERRED / UNKNOWN)
(a) Given the table, which configs deserve LB slots, in which order? (W alone looks dominated.)
(b) How much AI-Hub to add (per dataset counts), and should it be added at all given it is 100% real Korean audio?
    Main risk we see: a "Korean => real" shortcut (the round-10 pathology) unless matched Korean-context fakes exist.
    Proposed mitigation: ~5k Korean-context remixes built in Korea (AI-Hub real vocals over AT-ADD TTM accompaniment =
    realV+fakeM; AT-ADD SVC fake vocals over AI-Hub instrumental loops/covers = fakeV+realM) and a mass cap on AI-Hub
    (<= 25% of the real mass). Is that sufficient / necessary? Better alternative?
(c) Should the non-AI-Hub new data be scaled up further (e.g. more AT-ADD eval-pool rows, more HAIM, AI-OpenBMAT which is
    still weak) instead of / before AI-Hub? What is the highest expected LB gain per hour here?
(d) Any flaw in the evaluation that could make E2 overstate the gain (E2 open-data parts share datasets with training;
    AT-ADD eval generators are unseen) — and a cheap check we should add before spending a slot.
Keep the answer to decisions and numbers; no general advice.
