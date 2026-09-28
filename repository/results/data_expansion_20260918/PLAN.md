# Public audio acquisition expansion — 2026-09-18

Status: collection and validation complete; stopped as instructed. There are 11,703 newly acquired originals and 3,466 reused originals, for 15,169 catalogue records. Allocated storage is 15.64 GB for both acquisition roots and 19.38 GB including prepared copies, the upload ZIP and results, below 20,000,000,000 bytes. GPU compute has not started. See [the final report](COLLECTION_REPORT.md) and [completion receipt](collection_summary.json). Do not launch experiments automatically.

The user requested as much high-quality, rule-compliant data as practical before continuing experiments. The earlier 732-original collection was a pilot subset, not the limit of the available data. Its 2,470-row experiment bundle remains a preserved reference and is not the final expanded training population.

## Scope and storage

The new 20 GB cap supersedes the earlier large-bank allocation. Count the current acquisition bank and the earlier pilot acquisition together, counting shared hardlinked bytes once. Include their metadata and partial files. Retain a margin below the limit. Only temporary downloads and redundant archives created by this acquisition may be removed, after recording deletion receipts and verifying any selected originals recovered from them. Existing historical data, submission artifacts and the fixed evaluation panel remain preserved.

| Bank | Acquisition scope | Original-byte safeguard |
|---|---|---:|
| FakeMusicCaps | 512 shared prompt groups across five generators: 2,560 originals | 1.75 GB including metadata; original audio is 1,652,410,368 bytes |
| HAIM | Aim for 800 ACE-Step + 800 MusicGen, preserving reused pilot originals and separately recording rejected duplicates | 2.5 GB originals, approximately 2.52 GB including metadata |
| JamendoMaxCaps | Keep the existing 519 originals and prioritize one additional track per new artist; revised selection 854 tracks / 669 artists | 5 GB originals plus at most 600 MB metadata |
| SONICS | Source-ID-disjoint Suno/Udio subset recovered from four already verified archives; no further bulk download | 2 GB originals, at most 2.2 GB including metadata |
| FMA | Inventory and reuse local originals first, then acquire one track per additional licensed artist | 1.5 GB additional originals, at most 1.55 GB including metadata |
| Speech | Reuse the existing LibriSpeech test-clean archive; at most 4,000 protocol-bound ASVspoof LA originals if the range selection is verified | 300 MB additional audio; no full train-clean-100 or LA archive download |

The earlier pilot also contains FMA/SONICS confirmation originals, SunoCaps, GuitarSet, LibriSpeech and prepared views. Its unique bytes are included in the global measurement; the per-source table alone is not a claim of total compliance. Recheck actual unique allocated and apparent bytes before releasing new audio requests and at completion. Each source's obsolete bulk-download path must remain blocked or capped to the revised allocation.

Prefer archive-member access over duplicate full extraction. Save exact source URLs, repository revisions, source metadata, licenses, original hashes and archive checksums. Track reused files separately so expansion counts do not count the pilot twice.

## Admission and measurement

- A public repository or an open-source code license alone does not establish the rights to its audio. Preserve per-track licenses and attribution. Do not fetch the source YouTube audio referenced by synthetic datasets.
- Preserve original encoding, duration, sample rate and channel count. Record full decode failures, nonfinite samples, silence, unusually short files, clipping statistics and metadata discrepancies. These are integrity/inspection signals, not proof of musical quality or label correctness.
- Keep source-documented labels separate from independently reviewed labels. Unknown voice and presence truth remains unknown. Do not remove synthesis artifacts merely because they distinguish fake audio.
- Preserve the current development, calibration, historical-test and confirmation exclusions in `locked_groups.json`. Same artist, speaker, source recording, MusicCaps prompt or transformed parent must not move across these boundaries.
- Record exact-byte and decoded-audio duplicates plus known cross-corpus references. Acoustic near-duplicates, undocumented artist aliases and pretraining exposure remain separate unresolved risks unless actually measured.
- Bank acquisition does not assign every acquired row to training. Preserve evaluation-only groups and freeze the later experiment manifest before fitting or inspecting held-out predictions.

## Stop condition

The latest user instruction supersedes automatic continuation into the previously discussed A100 experiment: finish acquisition and validation, report the retained data and limitations, then stop. Do not start GPU compute, train a model, submit to DACON, or create a continuation automation. No compute deadline has begun. Preserve the former 2,470-row manifest and its upload receipt as pilot evidence. A later experiment requires a new user instruction.

Rule source inspected again on 2026-09-18: https://dacon.io/competitions/official/236749/overview/rules
