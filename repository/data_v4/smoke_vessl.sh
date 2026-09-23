#!/usr/bin/env bash
# First thing to run on the VESSL workspace, before any download or GPU generation.
# Expected env (from past workspace records): H100 SXM x1 (capella-emea) or A100 SXM x1 (betelgeuse-na),
# image quay.io/vessl-ai/torch:2.9.1-cuda13.0.1-py3.13-slim, persistent /root, object volume at /root-1.
# Usage: bash data_v4/smoke_vessl.sh [/root/envs/data/bin/python]
set -euo pipefail
PY=${1:-/root/envs/data/bin/python}
cd "$(dirname "$0")/.."
echo "== host";   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nproc; free -g | head -2
echo "== disk";   df -h /root /root-1 2>/dev/null || df -h /root
avail=$(df --output=avail -B1G /root | tail -1); [ "$avail" -ge 200 ] || echo "WARN: /root has ${avail}G free; budget needs ~200G (DESIGN.md §8)"
echo "== ffmpeg encoders"
for e in libmp3lame libopus libgsm aac pcm_mulaw pcm_alaw flac; do
  ffmpeg -hide_banner -encoders 2>/dev/null | grep -qw "$e" && echo "ok $e" || { echo "MISSING $e"; miss=1; }
done
[ -z "${miss:-}" ] || echo "WARN: drop missing codecs from CODECS in build_manifest.py before materialize (label-independent table must stay valid)"
echo "== self-checks"
for f in sources generate_speech generate_music build_manifest split presence export_bank; do
  case $f in sources) "$PY" data_v4/sources.py check ;; *) "$PY" data_v4/$f.py --self-check ;; esac
done
python3 candidate_data_acquire.py --self-check
echo "== synthetic end-to-end (materialize -> mix -> split)"
"$PY" - <<'EOF'
import json, sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, 'data_v4')
import build_manifest as bm, split as sp
t = Path(tempfile.mkdtemp())
rows = []
for i in range(8):                      # voice-like chirps (real/fake by construction), 3..70 s
    y = np.sin(2*np.pi*(200+50*i)*np.arange(int((3+9*i)*16000))/16000).astype('float32')*.3
    p = t/f'v{i}.wav'; sf.write(p, y, 16000)
    rows.append(bm.item('synthv', str(i), p, bm.lab(1, 0, i % 2, None), 'test', 'test', [f'spk:{i%4}']))
for i in range(8):                      # stereo "music", 44.1 kHz source
    y = (np.random.default_rng(i).standard_normal((44100*20, 2))*.1).astype('float32')
    p = t/f'm{i}.flac'; sf.write(p, y, 44100)
    rows.append(bm.item('synthm', str(i), p, bm.lab(0, 1, None, i % 2), 'test', 'test', [f'trk:{i%4}']))
out = t/'out'
man = [bm.materialize_one(r, out, 1) for r in rows]
for m in man:
    assert (out/m['path']).exists() and m['sha256'] and 3 <= m['crop_len_s'] <= 60.5, m
    info = sf.info(out/m['path']) if not m['path'].endswith(('.mp3',)) else None
    assert info is None or info.samplerate == 16000, (m['codec_chain'], info)
mixes = [bm.mix_one(j, out) for j in bm.plan_mixes(man, 6, 1)]
assert mixes and all(x['VOICE_PRESENT'] == x['MUSIC_PRESENT'] == 1 for x in mixes)
# gap constructions on the same synthetic pool: singing -> songs, same-speaker TTS splices, enhanced real
for m in man:
    if m['source'] == 'synthv':
        m.update(kind='singing' if m['ID'].endswith(('0', '1')) else None)
        if m['VOICE_FAKE'] == 0:
            m['source'] = 'zeroth'
        else:
            m.update(source='generated_speech', kind='tts')
    if m['source'] in ('zeroth', 'generated_speech'):   # utterances 2k (real) and 2k+1 (fake) share a speaker
        m['group_keys'] = [f"speaker:zeroth:{int(m['ID'].split(':')[1]) // 2}"]
noise = dict(bm.materialize_one(bm.item('musan_noise', 'n0', t/'m0.flac', bm.lab(0, 0, None, None), 'test', 'test', ['musan_noise:0']), out, 1))
pool = man + [noise]
built = [bm.mix_one(j, out) for j in bm.plan_songs(pool, 4, 1)] + [bm.splice_one(j, out) for j in bm.plan_splices(pool, 4, 1)]
try:
    from demucs.pretrained import get_model
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    built += [bm.enhance_one(j, out, get_model('htdemucs').to(dev).eval(), dev) for j in bm.plan_enhanced(pool, 1, 1)]
except ImportError:
    print('demucs missing: enhanced construction not exercised')
kinds = sorted({b['source'] for b in built})
assert {'constructed_splice'} <= set(kinds), kinds
for b in built:
    assert (out / b['path']).exists() and b['sha256'] and b['FILE_FAKE'] in (0, 1)
    if b['source'] == 'constructed_splice':
        (s0, s1), = b['fake_voice_segments']; assert 0 < s0 < s1 <= b['crop_len_s'] and b['FILE_FAKE'] == 1
    if b['source'] == 'constructed_enhanced':
        assert b['FILE_FAKE'] == 0 and b['ai_postprocessed_real']
split, rep = sp.apply(man + mixes + built, dict(test_holdout=[], cell_keys=['type', 'FILE_FAKE'], val_fraction=.25))
assert sp.verify(man + mixes + built, split)['ok']
print('e2e ok:', len(man), 'items,', len(mixes), 'mixes,', kinds, 'codecs', sorted({m['codec_chain'] for m in man}))
EOF
echo "== GPU generators (only envs that exist): one sample each"
[ -x /root/envs/data/bin/python ] && /root/envs/data/bin/python -c "
from demucs.pretrained import get_model; import torch; m=get_model('htdemucs').cuda(); print('htdemucs ok', torch.cuda.get_device_name())"
echo "smoke complete"
