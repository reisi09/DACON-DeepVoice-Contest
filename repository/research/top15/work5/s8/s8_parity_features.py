"""S8 parity reference features on the TRAINING path of the mixsslam member (research/top15/work5/s8/extract_s8.py):
baseline load_audio -> peak cap 0.6 -> runtime windows of the raw mix -> extract_s6.eat_family_features with the fp32
SSLAM-pretrain extraction weights. Output: PARITY/s8_parity/<ID>.npz (mixsslam, sslam_layers).  python s8_parity_features.py PARITY FFMPEG_DIR"""
import csv, os, sys
from pathlib import Path
import numpy as np, torch
P, FF = Path(sys.argv[1]), sys.argv[2]
os.environ['PATH'] = FF + os.pathsep + os.environ.get('PATH', '')
sys.path[:0] = ['/root/ft/code', '/root/newbb/vendor', '/root/s6']
from mert12_submission import baseline_module, pin_torchaudio_loader, window_layout, windows  # noqa: E402
import extract_s6 as X  # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = P / 's8_parity'; OUT.mkdir(exist_ok=True)
dev = torch.device('cpu')
base = baseline_module(Path('/root/bb_assets')); pin_torchaudio_loader(base)
sslam = X.load_eat_family(Path('/root/models/s6/sslam_pt'), dev)
for r in csv.DictReader(open(P / 'parity_manifest.csv')):
    raw = base.load_audio(P / 'parity_audio' / r['path'])
    peak = float(np.abs(raw).max())
    if peak > .6:
        raw = (raw * (.6 / peak)).astype(np.float32)
    starts = window_layout(len(raw))
    wm = np.asarray(windows(np.asarray(raw, np.float32), starts), np.float32)
    with torch.inference_mode():
        np.savez(OUT / f"{r['ID']}.npz", mixsslam=X.eat_family_features(sslam, wm, X.SSLAM_LAYERS, dev), sslam_layers=np.array(X.SSLAM_LAYERS))
    print(r['ID'], len(starts), flush=True)
