# Round 5b — fixes for your round-5 findings (2026-09-29 00:15 KST). Same reviewer session. BLOCKING-only check.

Pre-registration: x2_gates.py md5 90bca7cb38b1ac7293f0ea9d5d8fecdc was frozen at 00:12 KST, before any X3 result.
X3 is now running on x1 (rows built, feature dump started 00:09 KST; the 4 DB jobs follow; results expected ~01:00 KST).

Fixes (files in round5b/):
- Pick guard: sonics_rows.py build() asserts 750 unique ids per algorithm and 1,500 in total (build_log.txt: 34,021
  metadata songs, 8,577 found in the two zips, pools 3,287 / 3,384 after exclusion). The metadata filename is zero-padded
  (fake_00031_suno_1, id '31') and equals the zip member stem, so the join is exact.
- Id canonicalisation + lineage: ids compare as int strings; exclusion scans EVERY column of every row of bank_trainval,
  bank_test AND new_rows.csv for fake_<n>_(suno|udio) and sonics_lyrics:<n> (1,502 ids, matching your count). Any picked
  source mp3 or crop whose sha256 equals a bank/new-row audio or source hash aborts (42,039 hashes). Self-check on x1:
  {'2205','6776','31'} from a zero-padded orig_id, a mix parent_ids donor and a lyrics group.
  Checked outside the packet: new_rows.csv has no SONICS/Suno/Udio reference (strict regex, all columns); HAIM B3/B4 in E2
  are ACE-Step + Ozone 9 mastering / pro mixing per the HAIM description, not SONICS audio.
- Crop voice presence: after the dump, `sonics_rows.py relabel` reads each npz 'presence' (VOICE_PRESENT_PROB exactly as
  Engine.predict) and sets VOICE_PRESENT / VOICE_FAKE to unknown (mask 0) when < 0.5; music labels stay; FILE stays fake.
  The DB jobs read only the relabelled sonics_rows.csv; the runner stops unless it holds 1,500 rows.
- Label convention check: the bank's data_v4 'sonics' rows (same HF dataset) carry VOICE_FAKE=1 explicitly (lab(None,1,1,1))
  and are used by WavLM too; the 'existing:SONICS' rows are VF-unknown and completed by rule L for DF/AD only. So vf='1'
  on present-voice crops matches the bank's 'sonics' rows for all members.
- Extension: crops are written as .flac (codec_chain 'flac', as the other open rows).
- npz validation: all 14 trainer keys, finite encoder features (mert, eat, sslam, wavlm, voice/ad embeddings, presence),
  one window count across window_truth/mert/eat/sslam/wavlm (window_truth is NaN by design, not checked for finiteness).
- Gates (x3 only, before any result): (a) new P slice real_song_FILE_FPR = FILE FPR of real files with voice+music
  (E1 + clean E2) at the candidate-specific old-val TPR-95 thresholds, <= +0.01, 20-group rule as other rate slices;
  (b) AT-ADD singing FILE/VOICE EER: the real class has 11 singers for EVERY candidate, so HOLD would reject by construction.
  X3 rule: PASS iff the full-slice increase <= +0.010 AND every leave-one-group-out (smaller class) increase <= +0.010,
  else FAIL. X1/X2 verdicts are untouched (their rejections do not depend on these slices).

Questions (Korean, BLOCKING first, concise, file:line):
1. Are the round-5 BLOCKING items resolved by these changes? Any new code error?
2. Is the singing jackknife rule an acceptable pre-registered replacement for the structural HOLD (decided before any
   result), or must X3 be diagnostic only? If unacceptable, state the minimal acceptable rule.
