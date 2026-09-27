---
license: cc-by-nc-sa-4.0
metrics:
- accuracy
- precision
- recall
- f1
- roc_auc
base_model:
- facebook/mms-300m
---
# 🔍 **AntiDeepfake**


The **AntiDeepfake** project provides a series of powerful foundation models post-trained for **deepfake detection**. The AntiDeepfake model can be used for feature extraction for deepfake detection in a zero-shot manner, or it may be further fine-tuned and optimized for a specific database or deepfake-related task.

For technical details and analysis, please refer to our paper [Post-training for Deepfake Speech Detection](https://arxiv.org/abs/2506.21090).

# 🤖 Available Models

All models are released on Hugging Face 🤗 with two variants:
- **Default**: Trained with data augmentation
- **NDA** (No Data Augmentation): Trained without data augmentation

| Model                                    | Variants       |
|-----------------------------------------------|--------------------|
| XLS-R-2B-AntiDeepfake | [Default](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake), [NDA](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake-nda)
| XLS-R-1B-AntiDeepfake | [Default](https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake), [NDA](https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake-nda)
| MMS-1B-AntiDeepfake | [Default](https://huggingface.co/nii-yamagishilab/mms-1b-anti-deepfake), [NDA](https://huggingface.co/nii-yamagishilab/mms-1b-anti-deepfake-nda)
| MMS-300M-AntiDeepfake | [Default](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake), [NDA](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake-nda)
| Wav2Vec2-Large-AntiDeepfake | [Default](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake), [NDA](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda)
| Wav2Vec2-Small-AntiDeepfake | [Default](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake), [NDA](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake-nda)
| Hubert-Extra-Large-AntiDeepfake | [Default](https://huggingface.co/nii-yamagishilab/hubert-xlarge-anti-deepfake), [NDA](https://huggingface.co/nii-yamagishilab/hubert-xlarge-anti-deepfake-nda)

# 🛠️ Training Code & Repository
Explore training scripts, config files, and evaluation utilities in our GitHub repository:🔗 [AntiDeepfake GitHub Repository](https://github.com/nii-yamagishilab/AntiDeepfake)

# 🚀 **Model Spotlight: MMS-300M-AntiDeepfake-NDA**

# 🌟 **Key Features**
- **Architecture**: Wav2Vec 2.0 - [`facebook/mms-300m`](https://huggingface.co/facebook/mms-300m) 🔗.
- **Input**: 16kHz sampled speech with arbitrary length 🎙️.
- **Output**: Binary classification score (<Fake score 🔴 , Real score 🟢>).
- **Training Dataset**: Totally 18k hours of fake speech and 56k hours of real speech.

# 🏗️ **Architecture**
- **Front-end Feature Extractor**: MMS-300M.
- **Back-end Classifier**: A fully connected layer.

# ⚙️ **Training Details**
- **Optimizer**: AdamW with a learning rate of `1e-7`.
- **Batch Size**: Dynamic Batching, maximum length per batch is set to 100 seconds.
- **Data Augmentation**: None
- **Loss Function**: Cross-Entropy Loss.
- **Evaluation Metrics**: Equal Error Rate (EER), ROC AUC, Accuracy, Precision, Recall, F1 Score.

# 🚀 **Inference with PyTorch**
📦 Dependencies:
```
### New conda environments ###
conda create --name antideepfake python==3.9.0
conda activate antideepfake
conda install pip==24.0

### Install packages ###
pip install huggingface-hub==0.31.1 fairseq==0.12.2 safetensors==0.5.3 soundfile==0.13.1
```

🚀 Inference:
```python
import os
import torch
import torchaudio
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from huggingface_hub import PyTorchModelHubMixin

# This is the only part of the script you need to modify.
# Set this to the path where your audio files are stored.
folder_path = "/path/to/folder/contains/wavs/"
audio_formats = (".mp3", ".wav", ".flac", ".m4a")

# === Set device (use GPU if available) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Wrapper for the SSL model ===
class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Model config used to build SSL architecture
        cfg = Wav2Vec2Config(
            quantize_targets=True,
            extractor_mode="layer_norm",
            layer_norm_first=True,
            final_dim=768,
            latent_temp=(2.0, 0.1, 0.999995),
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            conv_bias=True,
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
            feature_grad_mult=1.0
        )
        # Initialize SSL model with random weights
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data):
        # If input has shape (B, T, 1), squeeze the last dim
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        # Extract features
        with torch.no_grad():
            features = self.model(input_data.to(device), mask=False, features_only=True)['x']
        return features

# === Function for reading and pre-processing waveforms ===
def load_wav_and_preprocess(wav_path, target_sr=16000):
    # Load audio file
    wav, sr = torchaudio.load(wav_path)
    # Convert to mono if stereo
    wav = wav.mean(dim=0)
    # Resample to target sampling rate
    wav = torchaudio.functional.resample(wav, sr, new_freq=target_sr)
    # Normalize waveform
    with torch.no_grad():
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
    # Add batch dimension and return
    return wav.unsqueeze(0).to(device)

# === The actual deepfake detection model using SSL frontend + FC backend ===
class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 1024
        self.num_classes = 2

        # Frontend: SSL model
        self.m_ssl = SSLModel()

        # Backend: Pooling + Classification
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(
            in_features=self.ssl_orig_output_dim,
            out_features=self.num_classes,
        )

    def forward(self, wav):
        emb = self.m_ssl.extract_feat(wav)  # [B, T, D]
        emb = emb.transpose(1, 2)           # [B, D, T]
        pooled_emb = self.adap_pool1d(emb)  # [B, D, 1]
        pooled_emb = pooled_emb.squeeze(-1) # [B, D]
        logits = self.proj_fc(pooled_emb)   # [B, 2]
        return logits


# === Load AntiDeepfake model from Hugging Face===
model = DeepfakeDetector.from_pretrained("nii-yamagishilab/mms-300m-anti-deepfake-nda")
model.to(device)
model.eval()

# === Inference on a folder of audio files ===
results = []
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(audio_formats):
            input_path = os.path.join(root, file)
            with torch.no_grad():
                wav = load_wav_and_preprocess(input_path)
                logits = model(wav)
                probs = torch.nn.functional.softmax(logits, dim=1)
                results.append((file, probs.cpu().numpy()[0]))

# Sort results alphabetically by filename
results.sort(key=lambda x: x[0])

# Print formatted results
print("\n=== Deepfake Detection Results ===")
for file_name, prob in results:
    print(f"{file_name}: real prob = {prob[1]:.3f}, fake prob = {prob[0]:.3f}")
```
# 📊 **Performance Metrics**
Results shown below can be reproduced using scripts provided in our [GitHub repository](https://github.com/nii-yamagishilab/AntiDeepfake).

| Test Database        | ROC AUC | Accuracy | Precision | Recall | F1-score | FPR   | FNR   | EER (%) @ Threshold |
|---------------------|---------|----------|-----------|--------|----------|-------|-------|----------------------|
| ADD2023             | 0.962   | 0.917    | 0.936     | 0.952  | 0.944    | 0.183 | 0.048 | 11.22 @ 0.8890       |
| DeepVoice           | 0.996   | 0.874    | 0.494     | 0.991  | 0.660    | 0.142 | 0.009 | 3.06 @ 0.9696        |
| FakeOrReal          | 1.000   | 0.995    | 0.993     | 0.997  | 0.995    | 0.007 | 0.003 | 0.48 @ 0.6142        |
| FakeOrReal-norm     | 0.996   | 0.940    | 0.891     | 1.000  | 0.942    | 0.117 | 0.000 | 2.70 @ 0.9619        |
| In-the-Wild         | 0.997   | 0.978    | 0.989     | 0.975  | 0.982    | 0.018 | 0.025 | 2.00 @ 0.4044        |
| Deepfake-Eval-2024  | 0.760   | 0.732    | 0.742     | 0.904  | 0.815    | 0.594 | 0.006 | 31.39 @ 0.9926       |


You can also fine-tune this model on a specific database, the corresponding code is provided in our [GitHub repository](https://github.com/nii-yamagishilab/AntiDeepfake). Fine-tuning will follow a similar process to training a new model, except that model weights will be initialized as AntiDeepfake checkpoints.

Below are the evaluation results of this model fine-tuned on the Deepfake-Eval-2024 training set and tested on its corresponding test set (as shown in the previous table):

| Test Input Length | ROC AUC | Accuracy | Precision | Recall | F1-score | FPR   | FNR   | EER (%) @ Threshold |
|-------------------|---------|----------|-----------|--------|----------|-------|-------|----------------------|
| 4s                | 0.8965  | 0.8084   | 0.7981    | 0.9466 | 0.8660   | 0.4527| 0.0534| 16.15 @ 0.9706       |
| 10s               | 0.9242  | 0.8651   | 0.8605    | 0.9467 | 0.9015   | 0.2877| 0.0533| 12.86 @ 0.9537       |
| 13s               | 0.9315  | 0.8793   | 0.8772    | 0.9475 | 0.9110   | 0.2482| 0.0525| 11.90 @ 0.9399       |
| 30s               | 0.9380  | 0.8956   | 0.9016    | 0.9413 | 0.9210   | 0.1881| 0.0587| 12.01 @ 0.8516       |
| 50s               | 0.9366  | 0.8881   | 0.9077    | 0.9173 | 0.9125   | 0.1628| 0.0827| 12.60 @ 0.7406       |



# **Training Set**
Below is a breakdown of the training set used for post-training of speech SSL models.

| 📚 Database             | 🌍 Language         | ✅ Genuine (hrs) | ❌ Fake (hrs) |
|----------------------|------------------|----------------|-------------|
| AISHELL3             | zh               | 85.62          | 0           |
| ASVspoof2019-LA      | en               | 11.85          | 97.80       |
| ASVspoof2021-LA      | en               | 16.40          | 116.10      |
| ASVspoof2021-DF      | en               | 20.73          | 487.00      |
| ASVspoof5            | en               | 413.49         | 1808.48     |
| CFAD                 | zh               | 171.25         | 224.55      |
| CNCeleb2             | zh               | 1084.34        | 0           |
| Codecfake            | en, zh           | 129.66         | 808.32      |
| CodecFake            | en               | 0              | 660.92      |
| CVoiceFake           | en, fr, de, it, zh| 315.14         | 1561.16     |
| DECRO                | en, zh           | 35.18          | 102.44      |
| DFADD                | en               | 41.62          | 66.01       |
| Diffuse or Confuse   | en               | 0              | 231.66      |
| DiffSSD              | en               | 0              | 139.73      |
| DSD                  | en, ja, ko       | 100.98         | 60.23       |
| FLEURS               | 102 languages    | 1388.97        | 0           |
| FLEURS-R             | 102 languages    | 0              | 1238.83     |
| HABLA                | es               | 35.56          | 87.83       |
| LibriTTS             | en               | 585.83         | 0           |
| LibriTTS-R           | en               | 0              | 583.15      |
| LibriTTS-Vocoded     | en               | 0              | 2345.14     |
| LJSpeech             | en               | 23.92          | 0           |
| MLAAD                | 38 languages     | 0              | 377.96      |
| MLS                  | 8 languages      | 50558.11       | 0           |
| SpoofCeleb           | Multilingual     | 173.00         | 1916.20     |
| VoiceMOS             | en               | 0              | 448.44      |
| VoxCeleb2            | Multilingual     | 1179.62        | 0           |
| VoxCeleb2-Vocoded    | Multilingual     | 0              | 4721.46     |
| WaveFake             | en, ja           | 0              | 198.65      |
| Train Set            | Over 100 languages| 56370.00       | 18280.00    |

# **Attribution**
All AntiDeepfake models were developed by [Yamagishi Lab](https://yamagishilab.jp/) at the National Institute of Informatics (NII), Japan.

All model weights are the intellectual property of NII and are made available for research and educational purposes under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

# **Acknowledgments**
This project is based on results obtained from project JPNP22007, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).

It is also partially supported by the following grants from the Japan Science and Technology Agency (JST):
- AIP Acceleration Research (Grant No. JPMJCR24U3)
- PRESTO (Grant No. JPMJPR23P9)

This study was carried out using the TSUBAME4.0 supercomputer at Institute of Science Tokyo.