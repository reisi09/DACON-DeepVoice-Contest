# Round 25 — cross-examination on data (2026-09-28 18:15 KST). Both reviewers get the same file.

Read the other reviewer's round-24 answer: long = codex_round24.md, fresh = fresh24_r10.md.

## 1. Facts I re-counted from data/bank_trainval.csv + bank_test.csv (VERIFIED by the main agent)
- My brief was wrong twice (long reviewer was right):
  - Phone/codec channels already exist for BOTH classes: codec_chain tel_ulaw 3,024 / tel_alaw 2,133 / tel_gsm 1,721 /
    tel_opus8 1,683 rows (8,561 total), plus mp3_64/128/192, aac_96, opus_32. Korean fake speech rows with tel_*: 1,400;
    Korean real: 1,068.
  - constructed_splice (121) = voice-only partial fakes (VOICE_FAKE=1, MUSIC_PRESENT=0), not speech->music.
- Korean fake speech generators (VOICE_FAKE=1, language ko): SpeechFake EdgeTTS / MeloTTS / CosyVoice / SeedVC / OpenVoice
  (600 each); MLAAD minimax speech-02-turbo (597 under two aliases), bark, xtts_v2, Chatterbox Multilingual, Higgs-Audio-V2,
  Fish-S2-Pro, Qwen3-TTS 0.6B/1.7B, MOSS-TTS 1.7B/8B, VoxCPM2, OmniVoice (~290-300 each).
- Mixed rows (voice and music present) in the S7 bank: 7,618. (VOICE_FAKE, MUSIC_FAKE): real+real 2,267 | fake+fake 2,045 |
  fake voice + real music 1,049 | voice label masked + fake music 2,124 | **real voice + fake music 117** | masked + real 16.
- Sequential speech<->music files: none recorded (constructed_mix has no event intervals; it is an overlay).
- Partial fakes: constructed_splice 121 only.
- New today (not in the S7 bank): AT-ADD singing 12,500 (vocals only), HAIM B3/B4/B6, AI-OpenBMAT (broadcast), D1 remixes 636.

## 2. Questions (Korean, VERIFIED / INFERRED / UNKNOWN, short; verify with web search where needed)
(a) After these facts, merge the two lists into ONE ranked plan of at most 5 data items (external or self-made) for G2,
    with the gap each fills. Drop items whose gap is already covered (e.g. generic phone augmentation, generic Korean TTS).
    Is "real voice + fake music" (117 rows) a real LB risk given the official types? What would fill it (e.g. real speech /
    real vocals over AI instrumentals from HAIM/SONICS/FakeMusicCaps/AT-ADD music; licenses)?
(b) Merge the two G1 panel designs into one (size, strata, channel copies, selection/confirmation split, how to report).
    Can it be built with ZERO new downloads from data already on the VESSL volume and the S7 bank?
(c) Tonight: both of you said no new training. The user's stated objective is top-15 or nothing (needs about ADS +0.022 over
    S7; slot 3 is the only free slot). Given that, is there ANY data action tonight with positive expected value for that
    objective, or does the answer stay "0 GPU tonight"? One line reason. If yes: exact steps, GPU-hours, adoption rule, stop rule.
(d) Point out any error in the other reviewer's round-24 answer (numbers, licenses, links).
