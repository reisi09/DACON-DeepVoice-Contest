"""Pinned pretrained-weight sets for data_v4 (called by vessl_fetch.sh; runnable anywhere).

  python data_v4/fetch_weights.py OUT_DIR core|seedvc [--dry-run]
core   -> OUT_DIR/<org>__<name>/ (local_dir layout, loaded by path)
seedvc -> OUT_DIR/hf_cache (HF cache layout: Seed-VC's inference resolves whisper/bigvgan by repo name,
          so HF_HUB_CACHE=OUT_DIR/hf_cache makes it work offline). Revisions pinned, receipt per set.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

SETS = {
    'core': [('FunAudioLLM/CosyVoice2-0.5B', None), ('ACE-Step/ACE-Step-v1-3.5B', None), ('myshell-ai/MeloTTS-Korean', None),
             ('kykim/bert-kor-base', None), ('myshell-ai/OpenVoiceV2', None), ('mispeech/ced-base', None),
             ('MIT/ast-finetuned-audioset-10-10-0.4593', None)],
    # Seed-VC (GitHub Plachtaa/seed-vc, GPL-3.0; HF user is "Plachta"): speech VC model + its runtime dependencies
    'seedvc': [('Plachta/Seed-VC', ['DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth',
                                    'config_dit_mel_seed_uvit_whisper_small_wavenet.yml']),
               ('openai/whisper-small', ['config.json', 'preprocessor_config.json', 'model.safetensors', '*.json', '*.txt']),
               ('nvidia/bigvgan_v2_22khz_80band_256x', ['config.json', 'bigvgan_generator.pt']),
               ('funasr/campplus', ['campplus_cn_common.bin'])],
}
HTDEMUCS = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th'


def plan(name: str) -> list[dict]:
    from huggingface_hub import HfApi
    api, rows = HfApi(), []
    for repo, patterns in SETS[name]:
        info = api.model_info(repo, files_metadata=True)
        import fnmatch
        files = [s for s in info.siblings if patterns is None or any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)]
        rows.append({'repo': repo, 'revision': info.sha, 'patterns': patterns,
                     'files': len(files), 'bytes': sum(s.size or 0 for s in files)})
    return rows


def fetch(out: Path, name: str) -> dict:
    from huggingface_hub import snapshot_download
    receipt = {}
    for row in plan(name):
        try:
            kw = dict(revision=row['revision'], allow_patterns=row['patterns'])
            if name == 'seedvc':
                path = snapshot_download(row['repo'], cache_dir=out / 'hf_cache', **kw)
            else:
                path = snapshot_download(row['repo'], local_dir=out / row['repo'].replace('/', '__'), **kw)
            row['bytes_on_disk'] = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
        except Exception as exc:  # one missing repo must not block the others
            row['error'] = f'{type(exc).__name__}: {exc}'[:300]
        receipt[row['repo']] = row
        print(row['repo'], {k: v for k, v in row.items() if k != 'repo'}, flush=True)
    if name == 'core':
        p = out / 'demucs' / HTDEMUCS.rsplit('/', 1)[1]
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            urllib.request.urlretrieve(HTDEMUCS, p)
        receipt['htdemucs'] = {'url': HTDEMUCS, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
    name_json = 'weights_receipt.json' if name == 'core' else f'weights_receipt_{name}.json'
    (out / name_json).write_text(json.dumps(receipt, indent=1) + '\n')
    return receipt


if __name__ == '__main__':
    out, name = Path(sys.argv[1]), sys.argv[2]
    if '--dry-run' in sys.argv:
        rows = plan(name)
        print(json.dumps(rows, indent=1))
        print('total GB', round(sum(r['bytes'] for r in rows) / 1e9, 2))
    else:
        out.mkdir(parents=True, exist_ok=True)
        fetch(out, name)
