# Validation sources

This is a small engineering validation bank, not a representative estimate of the hidden DACON score. All labels come from source provenance. Missing component labels remain blank. The prepared release has 225 clips: 128 speech/telephone clips, 33 music clips and 64 mixtures; 160 dev and 65 holdout clips share 64 distinct source groups.

## Speech

- Original author release: https://huggingface.co/datasets/mueller91/MLAAD-tiny
- Pinned revision: `9143e5ea709575ebab6bec52840a1043aada7bb1`.
- Eight matched source-utterance pairs per generator: Chatterbox, Edge-TTS, FishTTS, Kyutai-TTS. The source list is sorted by SHA-256 of its repository path before selection. Files shorter than four seconds are rejected after actual decoding.
- Real audio is M-AILABS, with the actual permissive redistribution notice copied from `original/LICENSE`. Fake audio is CC BY-NC 4.0; the actual release `LICENSE` is retained. Attribution: Nicolas M. Müller and MLAAD collaborators; original creators at M-AILABS, copyright 2017–2019.
- Every original utterance, its generator variants and its telephone-channel variants share one source group and split. This establishes source-utterance disjointness; it does not prove speaker-disjointness or disjointness from DF-Arena pretraining (the detector lists MLAAD in training).

## Music

The individual download manifest is archived as `validation/provenance/metadata/music_candidates.json`; selected originals, checksums, authors and licenses are listed in `validation/provenance/music_provenance.json`. Only the first eight real candidates and all 25 fake candidates are included. All original files were actually downloaded and decoded; this is stronger evidence than HTTP HEAD alone.

- Real: eight FMA tracks from eight distinct artists, selected using the original FMA instrumental metadata. Track IDs: `52630`, `24429`, `106568`, `10696`, `62742`, `146019`, `64859`, `58334`. Artists: Jason Shaw, Black Ant, Apache Tomcat, BOPD, Jahzzar, Ars Sonor & Malga Kubiak, Monroeville Music Center, Jono Bacon. Each file retains its recorded CC BY or CC BY-SA license and artist credit. The exact license version/URL is retained per track. Source: https://github.com/mdeff/fma#acknowledgments-and-licenses
- Fake: 25 AIME files, five each from MusicGen Small, MusicGen Large, AudioLDM2 Large, Riffusion and Mustango. These are five model configurations, not five independent model families. Original release: https://huggingface.co/datasets/disco-eth/AIME ; revision `b84d4be5eda830b6eb714998569dba73530f2601`. Generated audio is CC BY 4.0. Description tags are separately CC BY-NC-SA 4.0. Attribution: Florian Groetschla, Ahmet Solak, Luca A. Lanzendoerfer and Roger Wattenhofer. The original model card is copied into the package. Its 500 redistributed human MTG-Jamendo files were not used.
- No music clip received an independent listening audit. Instrumental metadata/protocol alone was not converted into a definitive voice-absence label. Music authenticity and presence have labels; standalone music voice fields remain blank. Real recordings are treated as the dataset's curated real source, not independently proven human-only provenance.
- FMA tracks sharing an artist and AIME files sharing a prompt are grouped before allocating splits. The real/fake source difference remains a confound: FMA versus AIME can reward corpus or codec recognition. This bank cannot establish broad unseen-generator music generalization.

## Mixed audio

Eight selected speech source groups are paired with real/fake music from the same split at 0 and 10 dB speech-to-music RMS SNR, producing 64 mixtures across the four authenticity combinations. File and music labels follow their component sources. Voice presence is known because speech is present. Voice authenticity is blank for real speech mixed with generated music whose vocal content is uncertain. Source group lineage includes both original components, so every reuse remains in its original split.

## Reproducibility and limits

Run `uv run prepare_validation.py`; rerun with `--check` to validate source-group separation, paired labels, decoded audio duration, and measured rejection of a 5 kHz signal by the telephone transformation.

Decoded clips use 16 kHz mono PCM and a center crop of at most ten seconds. Telephone variants apply a fifth-order 300–3400 Hz Butterworth bandpass, 8 kHz resampling, 8-bit mu-law quantization and conversion back to 16 kHz. This is a reproducible channel stress test, not a claim of bit-identical G.711 network simulation.

The manifest has separate `dev` and `holdout` groups. Fit any score fusion only on `dev`. Report `holdout` separately and keep original/telephone samples grouped in confidence intervals. Group allocation is deterministic from the source identifier.

Original downloads, source metadata, actual notices and SHA-256 values remain under `_workspace/validation_sources/`. The upload ZIP contains decoded audio, the manifest, source provenance, this document and actual license texts. Original audio remains local for audit. Cropped/resampled derivatives retain the source-specific licenses, including ShareAlike where applicable; the collection does not replace those licenses.

The AIME rows service supplies expiring signed audio URLs. Its refresh URL, row index and pinned dataset revision are recorded per file. Original downloaded audio plus its checksum is the reproducibility reference. A future fresh acquisition must refresh the URL and verify its checksum rather than assume a stale URL will work. Rerunning locally reuses the saved originals. The script does not silently fetch an alternative dataset or fabricate a missing label.
