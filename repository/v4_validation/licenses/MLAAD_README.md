---
license: cc-by-nc-4.0
task_categories:
- audio-classification
language:
- en
tags:
- deepfake
- deepfake-detection
- audio-deepfake
- audio-deepfake-detection
- mlaad
pretty_name: MLAAD tiny
size_categories:
- 10K<n<100K
---

# Welcome to MLAAD-tiny

**MLAAD-tiny** is a very small subset of the full [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) dataset, designed for **education, prototyping, and debugging**.

Many teaching environments (e.g. Colab, Kaggle, university notebooks -- se [this notebook](https://www.kaggle.com/code/muellermnicolas/audio-deepfake-detection-with-w2v2-and-mlaad-tiny) for example) impose **strict storage limits**, which makes large-scale audio deepfake datasets impractical to use. To address this, we provide **MLAAD-tiny**, a compact yet representative version of MLAAD.

## Download
```
git lfs install
git clone https://huggingface.co/datasets/mueller91/MLAAD-tiny
```

## Dataset composition

### Bona-fide
- Source: [M-AILABS](https://github.com/i-celeste-aurora/m-ailabs-dataset)
- ~6,000 audio files
- ~1.9 GB
- English

### Spoof
- 64 TTS systems  
- 100 samples per system (randomly selected from MLAAD)  
- ~6,400 audio files  
- ~2.3 GB  
- English (for training) and German (for testing)

## License

- **Bona-fide audio** is redistributed from M-AILABS under its original license  
  (see `original/LICENSE`).
- **Spoofed audio** is redistributed under the **MLAAD v8 license (CC BY-NC 4.0)**.

