---
license: cc-by-nc-4.0
pretty_name: "HAIM: Human-AI Music Dataset"
size_categories:
- 100K<n<1M
task_categories:
- audio-classification
language:
- ko
- en
tags:
- arxiv:2606.01686
- music
- ai-generated-music
- ai-generated-music-detection
- plagiarism-detection
- ace-step
- suno/udio
- mureka
- MTG-Jamendo
configs:
- config_name: A1_real_MTG_audio_preview
  data_files:
  - split: mtg
    path: "A1_real_MTG_audio/mtg-*.parquet"
- config_name: A2_fake_audio_preview
  data_files:
  - split: acestep
    path: "A2_fake_audio/acestep-*.parquet"
  - split: musicgen
    path: "A2_fake_audio/musicgen-*.parquet"
  - split: lyria_pro3
    path: "A2_fake_audio/lyria_pro3-*.parquet"
- config_name: B_hybrid_audio_preview
  data_files:
  - split: B1_ai_mastered_human
    path: "B_hybrid_audio/B1_ai_mastered_human-*.parquet"
  - split: B2_human_mastered_ai_dsp
    path: "B_hybrid_audio/B2_human_mastered_ai_dsp-*.parquet"
  - split: B3_mastered
    path: "B_hybrid_audio/B3_mastered-*.parquet"
  - split: B4_professional_human_mix
    path: "B_hybrid_audio/B4_professional_human_mix-*.parquet"
  - split: B6_human_lyrics_ai_gen
    path: "B_hybrid_audio/B6_human_lyrics_ai_gen-*.parquet"
  - split: B7_variation
    path: "B_hybrid_audio/B7_variation-*.parquet"
  - split: B8_edit
    path: "B_hybrid_audio/B8_edit-*.parquet"
  - split: B9_repaint
    path: "B_hybrid_audio/B9_repaint-*.parquet"
- config_name: C_mixing_audio_preview
  data_files:
  - split: C1_concat
    path: "C_mixing_audio/C1_concat-*.parquet"
  - split: C2_crossfade
    path: "C_mixing_audio/C2_crossfade-*.parquet"
- config_name: A2_fake_links_preview
  data_files:
  - split: suno
    path: "A2_fake_links/suno-*"
  - split: udio
    path: "A2_fake_links/udio-*"
  - split: mureka
    path: "A2_fake_links/mureka-*"
- config_name: B5_youtube_links_preview
  data_files:
  - split: b5
    path: "B5_youtube_links/b5-*"
- config_name: A1_sonics_links_preview
  data_files:
  - split: sonics
    path: "A1_sonics_links/sonics-*"
---


<div align="center">

# HAIM: Human-AI Music Dataset

**Human-AI Music Datasets for AI Music Production Tracking Benchmark**

[![arXiv](https://img.shields.io/badge/arXiv-2606.01686-b31b1b.svg)](https://arxiv.org/abs/2606.01686)
[![HF Papers](https://img.shields.io/badge/HF%20Papers-2606.01686-FFD21E.svg)](https://huggingface.co/papers/2606.01686)
[![GitHub](https://img.shields.io/badge/GitHub-HAIM__dataset-181717.svg?logo=github)](https://github.com/Mippia/HAIM_dataset)
[![License](https://img.shields.io/badge/Data-CC--BY--NC--4.0-lightgrey.svg)](#license)

*Beyond binary "AI-or-human" — granular, role-level tracking of AI intervention across the music production workflow.*

</div>

As generative platforms like Suno and Udio reach human-grade audio quality, AI now touches every stage of music production — not just full-song generation. HAIM moves detection beyond binary "AI-or-human" toward **role-level tracking** (Composer / Lyricist / Vocalist / Audio Engineer), with **153,686 tracks** (67,000 with audio ≈ 229 GB + 86,686 link-based) across **13 categories** that isolate each stage of AI intervention.

> **The viewer configs above are 100-row samples for preview/listening only** — `load_dataset(...)` returns these samples, **not** the full dataset.
> The complete data lives in the **raw folders** (`A_full_generation/`, `B_hybrid/`, `C_mixing/`) — see [Download the full dataset](#quick-start-download-the-full-dataset).

## Preview

In the Data Studio above, pick a config (e.g. `B_hybrid_audio_preview`) → a split (e.g. `B1_ai_mastered_human`) → press ▶ to listen. Audio configs show per-track metadata (prompt, lyrics, mode, …); `*_links` configs show platform/YouTube URL tables.

## What is inside

**A — Full generation.** Fully human music (A1: MTG-Jamendo CC audio, SONICS YouTube links) versus fully AI music (A2: locally generated ACE-Step / MusicGen audio, Lyria Pro 3 via API, plus Suno / Udio / Mureka as URL manifests). The classic real-vs-fake axis, at scale and multi-platform.

**B — Hybrid production.** The interesting middle: AI mastering on human tracks (B1), human mastering/mixing on AI tracks (B2–B4), AI vocal covers on human tracks (B5), AI generation conditioned on human-written lyrics (B6), and AI variation / edit / repaint of real songs (B7–B9). B3/B4 explicitly model how professional human post-production can smooth out AI artifacts and evade detectors.

**C — Temporal mix-sets.** Human and AI segments interleaved (concat) or blended (crossfade) inside one file — for evaluating *where* the AI parts are on a timeline, without boundary-specific training.

## Composition

| Category | Subset | Source | Tracks | Format |
|:---|:---|:---|---:|:---|
| A1 · Full Human | MTG-Jamendo | CC-licensed music | 6,000 | audio |
| A1 · Full Human | SONICS | YouTube | 48,090 | links |
| A2 · Full AI | ACE-Step 1.5 | local generation | 6,000 | audio |
| A2 · Full AI | MusicGen | local generation | 5,000 | audio |
| A2 · Full AI | Lyria Pro 3 | DeepMind API | 6,000 | audio |
| A2 · Full AI | Suno / Udio / Mureka | commercial platforms | 36,556 | links |
| B · Hybrid | B1–B4 mastering & mixing | SonicMaster, FXencoder, DAW | 24,000 | audio |
| B · Hybrid | B5 AI vocal cover | YouTube | 2,040 | links |
| B · Hybrid | B6–B9 lyrics-gen, variation, edit, repaint | ACE-Step 1.5 | 8,000 | audio |
| C · Temporal | C1 concat · C2 crossfade | human+AI segments | 12,000 | audio |

**Totals**: 67,000 audio tracks (~240 GB) + 86,686 link-based = **153,686 tracks**.

## Quick start (download the full dataset)

Download the raw folders directly — **not** via `load_dataset`:

```bash
pip install -U huggingface_hub
hf download mippia/HAIM --repo-type dataset --local-dir HAIM        # everything (~250 GB)
```

```python
from huggingface_hub import snapshot_download
snapshot_download("mippia/HAIM", repo_type="dataset", local_dir="HAIM")            # everything
snapshot_download("mippia/HAIM", repo_type="dataset", local_dir="HAIM",
                  allow_patterns=["B_hybrid/**"])                                  # one category only
```

- Per-folder metadata: each audio folder's `metadata.csv` (`file_name`, `track_id`, + generation params where available)
- Link manifests: `A_full_generation/A2_fake/{suno,udio,mureka}.json` & `*_tracks.csv`, `B_hybrid/B5_ai_vocal_human_track/b5_tracks.csv`, `A_full_generation/A1_real/sonics/real_songs.csv`

## Preview samples via `datasets` (not the full dataset)

```python
from datasets import load_dataset   # pip install "datasets[audio]"
# 100-row preview samples, NOT the full dataset
ds = load_dataset("mippia/HAIM", "B_hybrid_audio_preview", split="B1_ai_mastered_human")
suno = load_dataset("mippia/HAIM", "A2_fake_links_preview", split="suno")
```

<details>
<summary><b>Folder structure</b></summary>

```
├── A_full_generation
│   ├── A1_real
│   │   ├── MTG-Jamendo music subset/   # 6,000 mp3 + metadata.csv
│   │   ├── sonics/real_songs.csv       # 48,090 YouTube links
│   │   └── mtg_tracks.csv
│   └── A2_fake
│       ├── acestep/  lyria-pro3/                   # audio + sidecar JSON + metadata.csv
│       ├── musicgen/                               # audio + metadata.csv (no sidecar)
│       └── suno[.json|_tracks.csv]  udio…  mureka…  # link manifests
├── B_hybrid
│   ├── B1…B4, B6…B9/                   # audio + metadata.csv
│   └── B5_ai_vocal_human_track/        # b5_tracks.csv + metadata.jsonl
├── C_mixing/C1_mixset_concat/  C2_mixset_crossfade/
├── A1_real_MTG_audio/ A2_fake_audio/ B_hybrid_audio/ C_mixing_audio/   # 100-row viewer samples (parquet)
├── A2_fake_links/ B5_youtube_links/ A1_sonics_links/                   # 100-row viewer samples (parquet)
├── scripts/    figures/
├── DATASET_DESCRIPTION.txt
└── SUPPLEMENT_LEGAL_AND_ETHICAL.pdf
```
Large audio folders are sharded into `d0/`, `d1/` subfolders (Hub 10k-files-per-directory limit).
</details>

<details>
<summary><b>Metadata reference (per-file columns)</b></summary>

**`A1_real/sonics/real_songs.csv`** — `filename`, `title`, `artist`, `year`, `lyrics`, `duration`, `youtube_id`, `label`

**`A2_fake/{suno,udio,mureka}_tracks.csv`** — `platform`, `version`, `track_id`, `filename`, `page_url`, `audio_url`

**`B_hybrid/B5_ai_vocal_human_track/b5_tracks.csv`** — `track_id`, `original_title`, `url`, `channel`, `views`, `published`, `category`

**ACE-Step sidecar JSON (A2 acestep, B7–B9)** — `mode` (variation/edit/repaint), `concept.prompt`, `concept.lyrics`, `ace_step.result` (seed, BPM, key, model version)

**`B6_metadata.csv`** — human K-pop lyrics used for conditioning (text only, no original audio; ref [lt_dataset](https://github.com/havenpersona/lt_dataset)): `LID`, `ARTIST`, `EN_TITLE`, `KR_TITLE`, `IS_OFFICIAL`, `URL`
</details>

## Pipeline

![HAIM Pipeline](data_pipeline.png)

## Citation

```bibtex
@article{go2026haim,
  title={HAIM: Human-AI Music Datasets for AI Music Production Tracking Benchmark},
  author={Go, Seonghyeon and Kim, Yumin},
  journal={arXiv preprint arXiv:2606.01686},
  year={2026}
}
```

## License

**Data**: CC-BY-NC-4.0, non-commercial academic research only — commercial-platform content is released **as URL manifests only** per platform ToS. **Code**: MIT ([GitHub](https://github.com/Mippia/HAIM_dataset)). Please credit MTG-Jamendo and SONICS per their licenses. Full analysis: `SUPPLEMENT_LEGAL_AND_ETHICAL.pdf`
