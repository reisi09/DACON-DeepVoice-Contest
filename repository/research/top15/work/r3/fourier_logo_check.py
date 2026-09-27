"""Q: does the Fourier-peak fingerprint (Afchar et al., arXiv 2506.19108) transfer ACROSS music generators at 16 kHz?
Leave-one-generator-out: train L2 logistic regression on the other generators' fakes vs jamendo+FMA reals,
test on the held-out generator vs HAIM B1 SonicMaster (REAL, unseen real source). Local audio only, read-only, CPU.
Feature identical to fourier_peak_check.py (first 30 s, 16 kHz mono, 8192-pt mean |STFT| dB minus 31-bin local min, 1-8 kHz)."""
import glob, numpy as np, soundfile as sf, soxr
from scipy.ndimage import minimum_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
W = '/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/'
L = W + 'local_sampler_20260918/new_data/'
g = lambda *pats: sorted(sum((glob.glob(W + p, recursive=True) for p in pats), []))
AUD = ('.mp3', '.wav', '.flac')
fakes = {'lyria': g('candidate_acquisition_20260919/HAIM/A_full_generation/A2_fake/lyria-pro3/*/*.mp3')[:200],
         'suno': g('local_sampler_20260918/new_data/sunocaps/original/*', 'v2_sources/originals/echoes/TTA/suno/*'),
         'udio': g('v2_sources/originals/echoes/TTA/udio/*'),
         'acestep': g('local_sampler_20260918/new_data/haim/originals/acestep/*/*', 'v2_sources/originals/echoes/TTA/acestep/*'),
         'musicgen': g('local_sampler_20260918/new_data/haim/originals/musicgen/*/*', 'v2_sources/originals/echoes/TTA/musicgen/*'),
         'audioldm': g('v2_sources/originals/echoes/TTA/audioldm/*'),
         'elevenlabs': g('v2_sources/originals/echoes/TTA/elevenlabs/*')}
reals_train = g('local_sampler_20260918/new_data/additional_training/jamendo/originals/*', 'local_sampler_20260918/new_data/music/*.mp3')
reals_test = g('candidate_acquisition_20260919/HAIM/B_hybrid/B1_ai_mastered_human__MTG_sonicmaster/*/*.mp3')[:200]
NF, SR = 8192, 16000
freqs = np.fft.rfftfreq(NF, 1 / SR); band = (freqs >= 1000) & (freqs <= 8000)
def feat(p):
    sr = sf.info(p).samplerate; x, _ = sf.read(p, frames=sr * 30, dtype='float32', always_2d=True)
    x = soxr.resample(x.mean(1), sr, SR)
    if len(x) < NF: return None
    fr = np.lib.stride_tricks.sliding_window_view(x, NF)[::NF // 2] * np.hanning(NF)
    s = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).mean(0) + 1e-8)
    return (s - minimum_filter1d(s, 31))[band]
def feats(ps):
    out = [feat(p) for p in ps if p.lower().endswith(AUD)]
    return np.array([o for o in out if o is not None])
F = {k: feats(v) for k, v in fakes.items()}; RT = feats(reals_train); RS = feats(reals_test)
print('n', {k: len(v) for k, v in F.items()}, 'reals_train', len(RT), 'reals_test(sonicmaster)', len(RS))
clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000))
for held in F:
    Xtr = np.r_[np.concatenate([F[k] for k in F if k != held]), RT]
    ytr = np.r_[np.ones(sum(len(F[k]) for k in F if k != held)), np.zeros(len(RT))]
    m = clf().fit(Xtr, ytr)
    Xte = np.r_[F[held], RS]; yte = np.r_[np.ones(len(F[held])), np.zeros(len(RS))]
    print(f'held-out generator {held:10s}: AUC vs SonicMaster reals {roc_auc_score(yte, m.predict_proba(Xte)[:, 1]):.3f}')

# Variant: separate "unseen generator" from "SonicMaster hard negative" by testing against held-out halves of the
# jamendo+FMA reals (random half; train on the other half).
rng = np.random.default_rng(0); idx = rng.permutation(len(RT)); a, b = idx[: len(RT) // 2], idx[len(RT) // 2:]
for held in F:
    Xtr = np.r_[np.concatenate([F[k] for k in F if k != held]), RT[a]]
    ytr = np.r_[np.ones(sum(len(F[k]) for k in F if k != held)), np.zeros(len(a))]
    m = clf().fit(Xtr, ytr)
    Xte = np.r_[F[held], RT[b]]; yte = np.r_[np.ones(len(F[held])), np.zeros(len(b))]
    print(f'held-out generator {held:10s}: AUC vs held-out jamendo/FMA reals {roc_auc_score(yte, m.predict_proba(Xte)[:, 1]):.3f}')
