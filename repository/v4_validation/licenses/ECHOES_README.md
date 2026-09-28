---
license: cc-by-sa-4.0
---
## Echoes Dataset

### Dataset Description

This dataset contains **Echoes**, a semantically-aligned **music deepfake detection** dataset built for training and benchmarking detectors under realistic, provider-diverse conditions. Echoes pairs **bona fide** music tracks with **AI-generated** tracks created from multiple popular music generators, using **LLM-produced song descriptors** (and optionally **audio-conditioned generation**) to keep the deepfake semantically close to its reference.
Article describing the dataset: https://arxiv.org/abs/2603.23667
---

## Dataset Details

* **Format:** audio files (e.g., `.mp3`) + **CSV metadata manifest**
* **Size:** **4,468** tracks (AI-generated)
* **Total duration:** **~131 hours**
* **Genres:** Pop, Rock, Electronic
* **Generation types:**

  * **TTA (Text-to-Audio):** generation uses only the LLM textual description
  * **ATA (Audio-to-Audio):** generation uses both the LLM description and a reference audio file (only for providers that support it)

---

## Data Fields (CSV Manifest)

All metadata is provided in a CSV file (e.g., `dataset_manifest.csv`) located at the dataset root. Each row corresponds to one audio file.

Each record contains the following fields:

* `path_in_dataset`: relative path to the audio file inside the dataset directory
* `original_audio`: the bona fide reference track name (string)
* `generator`: provider/model family used to generate the audio 
* `type`: generation type (`TTA` or `ATA`)
* `genre`: genre of the bona fide reference track (Pop/Rock/Electronic)
* `description`: LLM-derived, song-specific keyword descriptors used as the generation prompt
* `duration`: track duration in seconds (float)

---

## Example Record

CSV columns:

```text
path_in_dataset,original_audio,generator,type,genre,description,duration
```

Example row:

```text
TTA/acestep/10000_People_Chanting_Im_an_Individual_Nihilore_acestep_TTA_001.mp3,"10,000 People Chanting, ""I'm an Individual"" - Nihilore",acestep,TTA,Electronic,"cinematic, idm, downtempo, layered, swelling, anthemic, vocal-chants, spacious, emotive, evolving",45.7
```

---

## Dataset Structure

The dataset is organized in directories that encode the **generation type**, **provider**, and **genre**, in a consistent layout.

* **Generation type directory:** `TTA/` or `ATA/`
* **Provider directory:** one folder per generator/provider (e.g., `acestep/`, `suno/`, `audioldm/`, …)
* **Audio files:** stored under these folders; filenames include identifying fields (provider, type, index)

A typical path looks like:

```text
<TYPE>/<PROVIDER>/<FILENAME>.mp3
```

---