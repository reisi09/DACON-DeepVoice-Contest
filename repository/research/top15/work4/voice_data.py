"""Voice-data study (choice_lock.json): R2 voice recipes retrained per variant under the DB protocol.
  base  original labels (must reproduce research/top15/work/r2/runs voice_df__mlp_long / voice_ad__resmlp_long)
  L     label completion of AI-generated vocals in data v4 TRAIN rows (rule L)
  S     + SingMOS-Pro train rows (gt real, svs/svc fake), features from extract_voice_feats.py ($SINGMOS_FEATS)
  LS    L + S
    python voice_data.py VARIANT MEMBER        MEMBER = df | ad; writes runs/voice_{MEMBER}__{VARIANT}.json
Runs keep the R2 run-file format (val/test ids, labels, per-seed predictions) so eval.py scores them like ads.py.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'work2' / 'backbones'))
import run as R  # noqa: E402  (backbones run.py: train, stats, Standardize, ResMLP, H, T, Item, PINNED, CACHE, LONG)

H, T, Item = R.H, R.T, R.Item
OUT = HERE / 'runs'
SINGMOS = Path(os.environ.get('SINGMOS_FEATS', H.ROOT / '_workspace/singmos_feats'))
FULL_TRACK = ('existing:SONICS', 'existing:suno', 'existing:udio')
FMC = set(json.loads((HERE / 'fmc_caption_vocal.json').read_text())['orig_ids'])


def completes(r) -> bool:
    """Rule L: a present vocal of unknown authenticity inside a fully generated track is AI-generated (fake)."""
    if not (r['VOICE_PRESENT'] == '1' and r['VOICE_PRESENT_mask'] == '1' and r['VOICE_FAKE_mask'] != '1'
            and r['MUSIC_FAKE'] == '1' and r['MUSIC_FAKE_mask'] == '1' and r['event_intervals'] in ('', '[]')):
        return False
    s = r['source']
    return (s in FULL_TRACK or (s == 'existing:HAIM' and r['generator'] in ('acestep', 'musicgen'))
            or (s.startswith('existing:Echoes/') and s.endswith('/TTA')) or (s == 'existing:FakeMusicCaps' and r['orig_id'] in FMC))


def complete(r):
    return {**r, 'VOICE_FAKE': '1', 'VOICE_FAKE_mask': '1'} if completes(r) else r


def load_rows(label_completion: bool):
    rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and r['ID'] in R.PINNED]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    if label_completion:
        rows['train'] = [complete(r) for r in rows['train']]
    return rows


def singmos_items(member: str, split: str):
    """SingMOS-Pro rows as voice-route items (a cappella: VOICE_PRESENT 1, MUSIC_PRESENT 0)."""
    out = []
    for m in csv.DictReader((HERE / 'singmos_manifest.csv').open()):
        p = SINGMOS / f"{m['ID']}.npz"
        if m['split'] != split or not p.exists():
            continue
        with np.load(p) as d:
            f = d['voice_embeddings'].astype(np.float32) if member == 'df' else d['ad_embeddings'].astype(np.float32)
            silent, samples = bool(d['voice_silent']), int(d['samples'])
        fake = 0 if m['type'] == 'gt' else 1
        row = {'ID': m['ID'], 'split': 'train', 'route': 'voice', 'source': 'singmos_pro',
               'generator': 'bonafide' if fake == 0 else m['model_name'], 'group_id': f"singmos:{m['raw_wav_id']}",
               'VOICE_PRESENT': '1', 'VOICE_PRESENT_mask': '1', 'VOICE_FAKE': str(fake), 'VOICE_FAKE_mask': '1',
               'MUSIC_PRESENT': '0', 'MUSIC_PRESENT_mask': '1', 'MUSIC_FAKE': '', 'MUSIC_FAKE_mask': '0',
               'FILE_FAKE': str(fake), 'FILE_FAKE_mask': '1'}
        out.append((Item(row, f, np.full(len(f), np.nan, np.float32), float(fake), silent), samples))
    return out


def every_items(rows, member):
    """Runtime semantics (work2/stacking/run.py every_items): one Item per frozen val/test row whatever its labels;
    silent = old-cache voice stem (the runtime scores VOICE 0 for it)."""
    frozen = set(json.loads((HERE.parent / 'work2' / 'stacking' / 'frozen_rows.json').read_text())['ids'])
    out = []
    for r in rows:
        if r['ID'] not in frozen or not (R.CACHE / f"{r['ID']}.npz").exists():
            continue
        with np.load(H.OLD[r.get('cache_split', r['split'])] / f"{r['ID']}.npz") as o:
            silent = bool(o['voice_silent'])
            f = o['voice_embeddings'].astype(np.float32) if member == 'df' else None
        if member == 'ad':
            with np.load(R.CACHE / f"{r['ID']}.npz") as d:
                f = d['ad_embeddings'].astype(np.float32)
        out.append(Item({**r, 'route': 'voice'}, f, np.full(len(f), np.nan, np.float32), 0., silent))
    return out


def main(variant: str, member: str) -> None:
    torch.set_num_threads(4)
    rows = load_rows('L' in variant)
    data = {s: H.items_for(rows[s], R.CACHE, [12], 'voice', dfarena=member == 'df') for s in rows}
    extra = singmos_items(member, 'train') if 'S' in variant else []
    # rule-L val rows scored for the val_completed metric only (never used for early stopping)
    valc = [(x, n) for x, n in H.items_for([complete(r) for r in rows['val'] if completes(r)], R.CACHE, [12], 'voice', dfarena=member == 'df')]
    sanity = singmos_items(member, 'test') if 'S' in variant else []
    if member == 'df':
        old = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
        fake_index = old['fake_label_index']

        def make(items):
            mean, scale = R.stats(items)
            return nn.Sequential(R.Standardize(mean, scale), nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 2))
    else:
        native = load_file(str(H.AD / 'model.safetensors'))
        init, fake_index = {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}, 0

        def make(items):
            return R.ResMLP(init, *R.stats(items))
    train = data['train'] + extra
    res = {'variant': variant, 'member': member, 'recipe': 'mlp_long' if member == 'df' else 'resmlp_long',
           'n': {'train': len(train), 'train_extra': len(extra), 'train_fake': int(sum(x.label for x, _ in train)),
                 'val': len(data['val']), 'test': len(data['test'])},
           'completed_train_rows': sum(completes(r) for r in load_rows(False)['train']) if 'L' in variant else 0,
           'seeds': {}, 'val_ids': [x.row['ID'] for x, _ in data['val']], 'val_labels': [int(x.label) for x, _ in data['val']],
           'test_ids': [x.row['ID'] for x, _ in data['test']], 'test_labels': [int(x.label) for x, _ in data['test']]}
    val_preds, test_preds, sanity_preds, valc_preds = [], [], [], []
    every = {s: every_items(rows[s], member) for s in ('val', 'test')}
    res['every'] = {s: {'ids': [x.row['ID'] for x in v], 'silent': [bool(x.silent) for x in v], 'preds': []} for s, v in every.items()}
    res['valc_ids'] = [x.row['ID'] for x, _ in valc]
    for seed in (41, 42, 43):
        model, info = R.train('voice', train, data['val'], make, fake_index, seed, **R.LONG)
        val_preds.append(T.predict(model, [x for x, _ in data['val']], 'voice', fake_index, 256, 10 ** 6, 'cpu').tolist())
        test_preds.append(T.predict(model, [x for x, _ in data['test']], 'voice', fake_index, 256, 10 ** 6, 'cpu').tolist())
        for s, v in every.items():
            res['every'][s]['preds'].append(T.predict(model, v, 'voice', fake_index, 256, 10 ** 6, 'cpu').tolist())
        valc_preds.append(T.predict(model, [x for x, _ in valc], 'voice', fake_index, 256, 10 ** 6, 'cpu').tolist())
        if sanity:
            sanity_preds.append(T.predict(model, [x for x, _ in sanity], 'voice', fake_index, 256, 10 ** 6, 'cpu').tolist())
        res['seeds'][seed] = {k: info[k] for k in ('val_eer', 'val_bce', 'best_epoch', 'epochs_run', 'n_train', 'train_seconds')}
        print(variant, member, seed, res['seeds'][seed], flush=True)
    res['val_preds'], res['test_preds'], res['valc_preds'] = val_preds, test_preds, valc_preds
    if sanity:
        y = [int(x.label) for x, _ in sanity]
        res['singmos_test'] = {'n': len(y), 'fake': sum(y), 'ens_eer': float(R._roc_metrics(y, np.mean(sanity_preds, 0).tolist())[0])}
    res['val_ens_eer'] = float(R._roc_metrics(res['val_labels'], np.mean(val_preds, 0).tolist())[0])
    print(variant, member, 'val_ens_eer', round(res['val_ens_eer'], 5), res.get('singmos_test'), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f'voice_{member}__{variant}.json').write_text(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
