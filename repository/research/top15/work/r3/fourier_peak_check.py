"""Q: at the official 16 kHz rate, does the Afchar et al. (ISMIR 2025, arXiv 2506.19108) Fourier-peak fingerprint
separate local HAIM Lyria-3-Pro fakes from (a) FMA human music and (b) HAIM B1 SonicMaster AI-mastered human music
(REAL under DACON rules, same HAIM mp3 pipeline)? Read-only; ~N files/class, first 30 s, CPU.
Feature: mean |STFT| (n_fft 8192, hop 4096) in dB -> subtract sliding local minimum (31 bins) -> keep 1-8 kHz bins.
Classifier: L2 logistic regression, 5-fold stratified CV AUC. Within-generator CV = optimistic upper bound."""
import glob, sys, numpy as np, soundfile as sf, soxr
from scipy.ndimage import minimum_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/'
H = R + 'candidate_acquisition_20260919/HAIM/'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
sets = {'lyria_fake': sorted(glob.glob(H + 'A_full_generation/A2_fake/lyria-pro3/*/*.mp3'))[:N],
        'sonicmaster_real': sorted(glob.glob(H + 'B_hybrid/B1_ai_mastered_human__MTG_sonicmaster/*/*.mp3'))[:N],
        'fma_real': sorted(glob.glob(R + 'local_sampler_20260918/new_data/music/*.mp3'))[:N]}
NF, SR = 8192, 16000
freqs = np.fft.rfftfreq(NF, 1 / SR); band = (freqs >= 1000) & (freqs <= 8000)
def feat(p):
    sr = sf.info(p).samplerate; x, _ = sf.read(p, frames=sr * 30, dtype='float32', always_2d=True)
    x = x.mean(1)
    x = soxr.resample(x, sr, SR)
    fr = np.lib.stride_tricks.sliding_window_view(x, NF)[::NF // 2] * np.hanning(NF)
    s = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).mean(0) + 1e-8)
    return (s - minimum_filter1d(s, 31))[band]
F = {k: np.array([feat(p) for p in v]) for k, v in sets.items()}
print({k: v.shape for k, v in F.items()})
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for neg in ('fma_real', 'sonicmaster_real'):
    X = np.r_[F['lyria_fake'], F[neg]]; y = np.r_[np.ones(len(F['lyria_fake'])), np.zeros(len(F[neg]))]
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000)), X, y, cv=cv, method='predict_proba')[:, 1]
    print(f'lyria_fake vs {neg}: 5-fold CV AUC {roc_auc_score(y, p):.3f} (n {int(y.sum())}/{int((1-y).sum())})')
# peak structure: bins where the class-mean residual is much higher in Lyria than in both real sets
d = F['lyria_fake'].mean(0) - np.maximum(F['fma_real'].mean(0), F['sonicmaster_real'].mean(0))
top = np.argsort(d)[::-1][:12]
print('top Lyria-excess bins (Hz, dB):', [(int(freqs[band][i]), round(float(d[i]), 2)) for i in sorted(top)])
print('max class-mean residual excess dB:', round(float(d.max()), 2), ' median:', round(float(np.median(d)), 2))
