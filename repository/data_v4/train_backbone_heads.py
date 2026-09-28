"""Heads on new frozen backbones (data_v4/extract_backbones.py cache), same objective/weights/duration balance/optimizer/
lr/patience/epochs as full_bank_train.py:
  voice: Linear(1280, 2) on AntiDeepfake-1B voice-stem window embeddings, initialized from the AntiDeepfake classifier,
         file = max over windows (the DF-Arena voice head, other backbone)
  music: WindowMIL on MERT hidden layers of the backing stem (layer 12 = the current music feature), log-mean-exp
Screen (DB protocol): train split -> val early stopping -> one-shot held-out test, one seed, then official ADS for
voice/music swaps and CDF fusions with the current heads (FILE ensemble refitted on val for each configuration).

  python data_v4/train_backbone_heads.py --cache BACKBONE_CACHE --out DIR [--layers 12 6 ...] [--final]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data_v4'))
import decision_heads as dh  # noqa: E402
import full_bank_train as T  # noqa: E402
import voice_heads  # noqa: E402
from component_fusion import fuse  # noqa: E402
from file_ensemble import fit as fit_file  # noqa: E402
from full_bank_batches import Item, duration_balance, epoch_order, loss_weights  # noqa: E402
import weighting  # noqa: E402
from full_bank_evaluate import score_rows  # noqa: E402
from graph_scoring import _roc_metrics, official_metrics  # noqa: E402
from localization_models import WindowMIL  # noqa: E402

BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
OLD = {'val': ROOT / '_workspace/cache_v4/cache_tv', 'train': ROOT / '_workspace/cache_v4/cache_tv', 'test': ROOT / '_workspace/cache_v4/cache_test'}
PRESENCE = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json'
AD = ROOT / '_workspace/models/xls-r-1b-anti-deepfake'
DEVICE = os.environ.get('HEAD_DEVICE', 'cpu')   # 'cuda:0' trains heads on GPU in fp32 (full_bank_train autocasts only for 'cuda')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# S8 frozen-feature members (research/top15/work5/s8/extract_s8.py; research/top15/work2/s6/extract_s6.py for hubert/wavlm):
# source -> (npz key, layer key, runtime input, packaged asset folder). voice16/backing16 = fp16-stored stem, mix = raw mix.
SSL_SOURCES = {'wavlm': ('wavlm_voice', 'wavlm_layers', 'voice16', 'wavlm'), 'hubert': ('hubert_voice', 'hubert_layers', 'voice16', 'hubert'),
               'd2v': ('d2v', 'ssl_layers', 'voice16', 'd2v'), 'usat': ('usat', 'ssl_layers', 'voice16', 'usat'),
               'mixwavlm': ('mixwavlm', 'ssl_layers', 'mix', 'wavlm')}
MUSIC_EXTRA = {'mixsslam': ('mixsslam', 'sslam_layers', 'mix', 'sslam_pt'), 'encodec': ('encodec', None, 'backing16', 'encodec'),
               'mixeat': ('mixeat', 'eat_layers', 'mix', 'eat'), 'mixmert': ('mixmert', 'mert_layers', 'mix', 'mert')}   # S9: --s9-cache


def npz_path(r: dict, directory) -> Path:
    """A row's feature file: its own 'npz' (new rows, research/top15/work5/newdata/dump_features.py: every cache key in one
    file) or DIRECTORY/ID.npz."""
    return Path(r['npz']) if r.get('npz') else directory / f"{r['ID']}.npz"


def new_rows() -> list:
    """Rows of the new datasets (env NEW_ROWS: comma-separated bank-format CSVs with an 'npz' column; split = pool 'train' or
    'newtest') whose feature file exists; rows whose dump failed are dropped and counted."""
    out, missing = [], 0
    for f in filter(None, os.environ.get('NEW_ROWS', '').split(',')):
        for r in csv.DictReader(open(f)):
            if Path(r['npz']).exists():
                out.append(r)
            else:
                missing += 1
    if out or missing:
        print('new rows', len(out), 'without features', missing, flush=True)
    return out


def add_new_rows(rows: dict) -> None:
    """DB protocol with new rows (research/top15/work5/newdata): pool 'train' joins the training rows (unless NEW_TRAIN=0),
    pool 'newval' joins the early-stopping val rows only when NEW_VAL=1 (otherwise unused), pool 'newtest' is a held-out
    split scored every-row."""
    added = new_rows()
    if os.environ.get('NEW_TRAIN', '1') == '1':
        rows['train'] += [r for r in added if r['split'] == 'train']
    if os.environ.get('NEW_VAL') == '1':
        rows['val'] += [r for r in added if r['split'] == 'newval']
    if any(r['split'] == 'newtest' for r in added):
        rows['newtest'] = [r for r in added if r['split'] == 'newtest']


def eat_index(caches) -> dict:
    """ID -> npz path over the EAT caches (research/top15/work2/backbones/extract_new_backbones.py, key eat_backing)."""
    return {f.stem: f for c in caches for f in Path(c).glob('*.npz') if not f.name.endswith('.tmp.npz')}


def eat_features(path: Path, layers) -> np.ndarray:
    with np.load(path) as d:
        L = list(d['eat_layers'])
        return d['eat_backing'][:, [L.index(k) for k in layers]].reshape(len(d['eat_backing']), -1).astype(np.float32)


def sslam_features(path: Path, layers) -> np.ndarray:
    """SSLAM pretrain blocks of the backing stem (research/top15/work2/s6/extract_s6.py, key sslam_pt_backing)."""
    with np.load(path) as d:
        L = list(d['sslam_layers'])
        return d['sslam_pt_backing'][:, [L.index(k) for k in layers]].reshape(len(d['sslam_pt_backing']), -1).astype(np.float32)


def cached(path: Path, key: str, layer_key, layers) -> np.ndarray:
    """(windows, dim) float32 features of one npz: `layers` of key (concatenated) or the whole key when layer_key is None."""
    with np.load(path) as d:
        x = d[key]
        if layer_key is not None:
            L = list(d[layer_key])
            x = x[:, [L.index(k) for k in layers]].reshape(len(x), -1)
        return x.astype(np.float32)


def ssl_items(rows, index, source, layers):
    """wavlm_items for any SSL_SOURCES member (layers concatenated)."""
    key, layer_key = SSL_SOURCES[source][:2]
    out = []
    for r in rows:
        p = index.get(r['ID'])
        if p is None:
            continue
        f = cached(p, key, layer_key, layers)
        with np.load(p) as d:
            silent, samples = bool(d['voice_silent']), int(d['samples'])
        label = T.head_labels(r).get('voice')
        if label is None or (r['route'] != 'voice' and silent):
            continue
        out.append((Item({**r, 'route': 'voice'}, f, np.full(len(f), np.nan, np.float32), label, silent), samples))
    return out


def wavlm_items(rows, index, layer):
    """Voice-route items on the WavLM-Large hidden state `layer` (frame mean per runtime window) of the voice stem
    (research/top15/work2/s6 cache); research/top15/work2/backbones/run.py new_items semantics: voice label known,
    silent non-voice-route rows skipped."""
    out = []
    for r in rows:
        p = index.get(r['ID'])
        if p is None:
            continue
        with np.load(p) as d:
            L = list(d['wavlm_layers'])
            f = d['wavlm_voice'][:, [L.index(layer)]].reshape(len(d['wavlm_voice']), -1).astype(np.float32)
            silent, samples = bool(d['voice_silent']), int(d['samples'])
        label = T.head_labels(r).get('voice')
        if label is None or (r['route'] != 'voice' and silent):
            continue
        out.append((Item({**r, 'route': 'voice'}, f, np.full(len(f), np.nan, np.float32), label, silent), samples))
    return out


def extra_voice_items(manifest: Path, feats: Path, dfarena: bool):
    """Voice-route items from an external a cappella manifest (research/top15/work4/singmos_manifest.csv: type gt -> real,
    svs/svc -> fake, other types skipped) with research/top15/work4/extract_voice_feats.py features. A listed row without
    features raises: a package must not silently train on fewer rows than its study."""
    out = []
    for m in csv.DictReader(manifest.open()):
        if m['type'] not in ('gt', 'svs', 'svc'):
            continue
        with np.load(feats / f"{m['ID']}.npz") as d:
            f = (d['voice_embeddings'] if dfarena else d['ad_embeddings']).astype(np.float32)
            silent, samples = bool(d['voice_silent']), int(d['samples'])
        fake = 0 if m['type'] == 'gt' else 1
        row = {'ID': m['ID'], 'split': 'train', 'route': 'voice', 'source': 'singmos_pro',
               'generator': 'bonafide' if fake == 0 else m['model_name'], 'group_id': f"singmos:{m['raw_wav_id']}",
               'VOICE_PRESENT': '1', 'VOICE_PRESENT_mask': '1', 'VOICE_FAKE': str(fake), 'VOICE_FAKE_mask': '1',
               'MUSIC_PRESENT': '0', 'MUSIC_PRESENT_mask': '1', 'MUSIC_FAKE': '', 'MUSIC_FAKE_mask': '0',
               'FILE_FAKE': str(fake), 'FILE_FAKE_mask': '1'}
        out.append((Item(row, f, np.full(len(f), np.nan, np.float32), float(fake), silent), samples))
    return out


def items_for(rows, cache: Path, layers, route_filter, dfarena=False, eat=None, sslam=None, extras=()):
    """dfarena=True: the current DF-Arena voice features from the old cache, restricted to rows with a backbone npz
    (matched-row baseline for the new voice head). eat=(index, layers): music features = MERT layers + EAT blocks
    (the runtime order), rows without an EAT npz skipped."""
    out = []
    for r in rows:
        p = npz_path(r, cache)
        if not p.exists() or (route_filter == 'music' and ((eat and r['ID'] not in eat[0]) or (sslam and r['ID'] not in sslam[0])
                                                       or any(r['ID'] not in ix for ix, _, _ in extras))):
            continue
        with np.load(p) as d:
            silent, L = bool(d['voice_silent']), list(d['layers'])
            feats = {'voice': d['ad_embeddings'].astype(np.float32),
                     'music': d['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(d['mert_layers']), -1).astype(np.float32)}
            samples = int(d['samples'])
        if eat and route_filter == 'music':
            e = eat_features(eat[0][r['ID']], eat[1])
            if len(e) != len(feats['music']):
                raise ValueError(f"EAT/MERT window counts differ for {r['ID']}: {len(e)} vs {len(feats['music'])}")
            feats['music'] = np.concatenate([feats['music'], e], 1)
        if sslam and route_filter == 'music':
            e = sslam_features(sslam[0][r['ID']], sslam[1])
            if len(e) != len(feats['music']):
                raise ValueError(f"SSLAM/MERT window counts differ for {r['ID']}: {len(e)} vs {len(feats['music'])}")
            feats['music'] = np.concatenate([feats['music'], e], 1)
        for ix, source, extra_layers in (extras if route_filter == 'music' else ()):   # S8 music extras, after SSLAM (runtime order)
            key, layer_key = MUSIC_EXTRA[source][:2]
            e = cached(ix[r['ID']], key, layer_key, extra_layers)
            if len(e) != len(feats['music']):
                raise ValueError(f"{source}/MERT window counts differ for {r['ID']}: {len(e)} vs {len(feats['music'])}")
            feats['music'] = np.concatenate([feats['music'], e], 1)
        if dfarena:
            with np.load(npz_path(r, OLD.get(r.get('cache_split', r['split'])))) as o:
                feats['voice'], silent = o['voice_embeddings'].astype(np.float32), bool(o['voice_silent'])
        truth = np.full(len(feats['music']), np.nan, np.float32)
        with np.load(npz_path(r, OLD.get(r.get('cache_split', r['split'])))) as o:   # same runtime windows (parity-checked)
            if len(o['window_truth']) == len(truth):
                truth = o['window_truth'].astype(np.float32)
        for head, label in T.head_labels(r).items():
            if head != route_filter or (head != r['route'] and head == 'voice' and silent):
                continue
            f = feats[head]
            out.append((Item({**r, 'route': head}, f, truth if head == 'music' else np.full(len(f), np.nan, np.float32), label, silent), samples))
    return out


def train(route, training, validation, init=None, fake_index=0, seed=41, max_epochs=50, patience=5, arch='linear'):
    items = [x for x, _ in training]
    weights = weighting.weights(items, [s for _, s in training], route)   # WEIGHTING env: s7 (default = the previous code) | C | W
    if weighting.SCHEME != 's7' and seed == 41:
        print('weights', route, weighting.SCHEME, json.dumps(weighting.describe(items, weights, route), default=float), flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if route == 'music':
        means = np.stack([x.features.mean(0) for x in items])
        model = WindowMIL(torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6)))
    else:   # voice_heads: linear (init = classifier weights), mlp, resmlp (standardized on training file means)
        means = np.stack([x.features.mean(0) for x in items])
        model = voice_heads.build(arch, init, torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6)), width=means.shape[1])
    model.to(DEVICE)
    val = [x for x, _ in validation]
    limit = max(sum(len(x.features) for x in sorted(items, key=lambda x: len(x.features), reverse=True)[:256]), 10 ** 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if route == 'music' else 1e-4, weight_decay=.01)
    labels = np.array([x.label for x in val])
    best, state, stale, epochs = (float('inf'), float('inf')), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        if stale >= patience:
            break
        model.train()
        order = epoch_order(items, seed, epoch)
        for offset in range(0, len(order), 256):
            T.step(model, optimizer, [items[i] for i in order[offset:offset + 256]], route, fake_index, weights, 256, limit, DEVICE)
        pred = T.predict(model, val, route, fake_index, 256, limit, DEVICE)
        eer = float(_roc_metrics([int(v) for v in labels], pred.tolist())[0])
        clipped = np.clip(pred, 1e-7, 1 - 1e-7)
        bce = float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean())
        epochs = epoch
        if (eer, bce) < best:
            best, stale, state = (eer, bce), 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.best_epoch = epoch
        else:
            stale += 1
    model.load_state_dict(state)
    return model.to('cpu'), best[0], epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layers', type=int, nargs='+', default=[3, 6, 9, 12, 15, 18, 21, 24])
    ap.add_argument('--final', action='store_true', help='P1 protocol: train+held-out test as train, 3 seeds, packaged heads')
    ap.add_argument('--music-layers', type=int, nargs='+', default=[6, 12])
    ap.add_argument('--music-eat', type=int, nargs='*', default=[], help='EAT blocks appended to the music features (--final)')
    ap.add_argument('--eat-cache', type=Path, nargs='*', default=[], help='dirs with EAT npz (eat_backing)')
    ap.add_argument('--reuse-voice', type=Path, help='--final: copy voice_* and voice_ad_* from this heads dir instead of retraining')
    ap.add_argument('--music-sslam', type=int, nargs='*', default=[], help='SSLAM pretrain blocks appended after EAT (--final)')
    ap.add_argument('--sslam-cache', type=Path, nargs='*', default=[], help='dirs with S6 npz (sslam_pt_backing)')
    ap.add_argument('--voice-wavlm', type=int, help='--final: add a WavLM-Large member on this hidden state (voice stem)')
    ap.add_argument('--wavlm-cache', type=Path, nargs='*', default=[], help='dirs with S6 npz (wavlm_voice)')
    ap.add_argument('--wavlm-weight', type=float, default=1., help='fusion weight of the WavLM member')
    ap.add_argument('--reuse-music', type=Path, help='--final: copy music_* from this heads dir instead of retraining')
    ap.add_argument('--reuse-wavlm', type=Path, help='--final: copy wavlm_* from this heads dir instead of retraining')
    ap.add_argument('--file-weights', help='--final: fixed FILE ensemble coefficients max,noisy_or,max_nopresence (e.g. 0,9,1)')
    ap.add_argument('--s8-cache', type=Path, nargs='*', default=[], help='dirs with S8 npz (extract_s8.py)')
    ap.add_argument('--s9-cache', type=Path, nargs='*', default=[], help='dirs with S9 npz (extract_s9.py: mixeat, mixmert)')
    ap.add_argument('--voice-ssl', nargs=4, action='append', default=[], metavar=('NAME', 'SOURCE', 'LAYERS', 'WEIGHT'),
                    help='--final: extra voice member NAME on SSL_SOURCES[SOURCE] hidden states LAYERS (e.g. 8 or 6_8_12), fusion WEIGHT')
    ap.add_argument('--music-extra', nargs=2, action='append', default=[], metavar=('SOURCE', 'LAYERS'),
                    help='--final: MUSIC_EXTRA[SOURCE] features appended after SSLAM (LAYERS e.g. 5, or - for encodec)')
    ap.add_argument('--label-completion', action='store_true', help='--final: data_v4/label_rules.py rule L on the training rows')
    ap.add_argument('--extra-voice', type=Path, nargs=2, metavar=('MANIFEST', 'FEATS'), help='--final: external voice rows added to voice training')
    ap.add_argument('--ours', type=Path, default=ROOT / '_workspace/p1_refit_cpu', help='DF-Arena voice heads kept as "ours"')
    ap.add_argument('--ad-weight', type=float, default=.5)
    ap.add_argument('--voice-arch', default='copy', choices=('copy', 'linear', 'mlp'), help='DF-Arena voice heads: copy from --ours or retrain')
    ap.add_argument('--ad-arch', default='linear', choices=('linear', 'mlp', 'resmlp'))
    ap.add_argument('--voice-epochs', type=int, default=50)
    ap.add_argument('--voice-patience', type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = {s: [r for r in csv.DictReader((BANK / f).open()) if r['split'] == s]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    native = load_file(str(AD / 'model.safetensors'))
    init = {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}
    if a.final:
        return final(a, rows, init)
    report, preds = {'rows_with_cache': {}}, {}
    # matched-row baseline: current DF-Arena voice head trained on exactly the rows that have a backbone npz
    old_init = torch.load(OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
    data = {s: items_for(rows[s], a.cache, [12], 'voice', dfarena=True) for s in rows}
    model, val_eer, epochs = train('voice', data['train'], data['val'], old_init['state_dict'], old_init['fake_label_index'])
    for s in ('val', 'test'):
        p = T.predict(model, [x for x, _ in data[s]], 'voice', old_init['fake_label_index'], 256, 10 ** 6, 'cpu')
        preds.setdefault('voice_dfarena', {}).update({x.row['ID']: float(v) for (x, _), v in zip(data[s], p)})
    test_items = [x for x, _ in data['test']]
    report['voice_dfarena'] = {'val_eer': round(val_eer, 5), 'epochs': epochs, 'test_eer_route': round(float(_roc_metrics(
        [int(x.label) for x in test_items], [preds['voice_dfarena'][x.row['ID']] for x in test_items])[0]), 5)}
    print('voice_dfarena (matched baseline)', report['voice_dfarena'], flush=True)
    # new voice head (AntiDeepfake embeddings)
    data = {s: items_for(rows[s], a.cache, [12], 'voice') for s in rows}
    report['rows_with_cache']['voice'] = {s: len(v) for s, v in data.items()}
    model, val_eer, epochs = train('voice', data['train'], data['val'], init)
    for s in ('val', 'test'):
        p = T.predict(model, [x for x, _ in data[s]], 'voice', 0, 256, 10 ** 6, 'cpu')
        preds.setdefault('voice_ad', {}).update({x.row['ID']: float(v) for (x, _), v in zip(data[s], p)})
    test_items = [x for x, _ in data['test']]
    report['voice_ad'] = {'val_eer': round(val_eer, 5), 'epochs': epochs, 'test_eer_route': round(float(_roc_metrics(
        [int(x.label) for x in test_items], [preds['voice_ad'][x.row['ID']] for x in test_items])[0]), 5)}
    print('voice_ad', report['voice_ad'], report['rows_with_cache']['voice'], flush=True)
    torch.save(model.state_dict(), a.out / 'voice_ad_41.pt')
    # music heads per layer
    for k in a.layers:
        data = {s: items_for(rows[s], a.cache, [k], 'music') for s in rows}
        model, val_eer, epochs = train('music', data['train'], data['val'])
        for s in ('val', 'test'):
            p = T.predict(model, [x for x, _ in data[s]], 'music', 0, 256, 10 ** 6, 'cpu')
            preds.setdefault(f'music_L{k}', {}).update({x.row['ID']: float(v) for (x, _), v in zip(data[s], p)})
        test_items = [x for x, _ in data['test']]
        test_eer = float(_roc_metrics([int(x.label) for x in test_items], [preds[f'music_L{k}'][x.row['ID']] for x in test_items])[0])
        report[f'music_L{k}'] = {'val_eer': round(val_eer, 5), 'test_eer_route': round(test_eer, 5), 'epochs': epochs, 'train_rows': len(data['train'])}
        print(f'music_L{k}', report[f'music_L{k}'], flush=True)
        torch.save(model.state_dict(), a.out / f'music_L{k}_41.pt')
    with (a.out / 'predictions.json').open('w') as f:
        json.dump(preds, f)
    (a.out / 'screen_report.json').write_text(json.dumps(report, indent=1) + '\n')


def final(a, rows, init) -> None:
    """Heads for the package: new voice_ad_* and music_* (MERT a.music_layers) trained on train + held-out test rows that have
    a backbone npz (val early stopping, seeds 41/42/43); DF-Arena voice_* copied from a.ours. Writes the component fusion
    (voice = ours + a.ad_weight * ad, val CDFs), the refitted FILE ensemble and a candidate lock over all nine heads."""
    import hashlib
    import shutil
    rows = {'train': rows['train'] + [{**r, 'split': 'train', 'cache_split': 'test'} for r in rows['test']], 'val': rows['val']}
    assert not (a.reuse_voice and (a.label_completion or a.extra_voice)), 'reused voice heads cannot see new voice labels/rows'
    completed = 0
    if a.label_completion:
        import label_rules
        completed = sum(label_rules.completes(r) for r in rows['train'])
        rows['train'] = [label_rules.complete(r) for r in rows['train']]
    old_fake = torch.load(OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)['fake_label_index']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    lock, results, val_scores = {}, {}, {}
    eat = (eat_index(a.eat_cache), a.music_eat) if a.music_eat else None
    sslam = (eat_index(a.sslam_cache), a.music_sslam) if a.music_sslam else None
    wavlm = eat_index(a.wavlm_cache) if a.voice_wavlm is not None else None
    s8 = eat_index(a.s8_cache) if a.s8_cache else {}
    s6 = wavlm if wavlm is not None else (eat_index(a.wavlm_cache) if a.wavlm_cache else {})
    layers_of = lambda t: None if t == '-' else [int(x) for x in t.split('_')]
    ssl_members = [(n, src, layers_of(ls), float(w), s6 if src in ('wavlm', 'hubert') else s8) for n, src, ls, w in a.voice_ssl]
    assert len({n for n, *_ in ssl_members} | {'ours', 'ad', 'wavlm'}) == len(ssl_members) + 3, 'voice member names must be unique'
    s9 = eat_index(a.s9_cache) if a.s9_cache else {}
    extras = [(s9 if src in ('mixeat', 'mixmert') else s8, src, layers_of(ls)) for src, ls in a.music_extra]
    added = new_rows()   # P1: every new row is a training row (newval -> val instead when NEW_VAL=1); its npz serves every index
    new_val = os.environ.get('NEW_VAL') == '1'
    rows['train'] += [{**r, 'split': 'train'} for r in added if not (new_val and r['split'] == 'newval')]
    rows['val'] += [{**r, 'split': 'val'} for r in added if new_val and r['split'] == 'newval']
    for ix in (eat[0] if eat else None, sslam[0] if sslam else None, wavlm):
        if ix is not None:
            ix.update({r['ID']: Path(r['npz']) for r in added})
    val_rows = [r for r in rows['val'] if (a.cache / f"{r['ID']}.npz").exists() and (eat is None or r['ID'] in eat[0])
                and (sslam is None or r['ID'] in sslam[0]) and (wavlm is None or r['ID'] in wavlm)
                and all(r['ID'] in m[4] for m in ssl_members) and all(r['ID'] in ix for ix, _, _ in extras)]

    def reused(name):
        """A voice head of a.reuse_voice, byte-copied (CPU retraining is not bit-reproducible across runs)."""
        shutil.copy(a.reuse_voice / name, a.out / name)
        record = torch.load(a.out / name, map_location='cpu', weights_only=True)
        return voice_heads.load(record).eval(), record['validation_eer'], record['epochs_run']

    def reused_music(name):
        shutil.copy(a.reuse_music / name, a.out / name)
        record = torch.load(a.out / name, map_location='cpu', weights_only=True)
        model = WindowMIL(record['model']['mean'], record['model']['scale'])
        model.load_state_dict(record['model'], strict=True)
        return model.eval(), record['validation_eer'], record['epochs_run']

    def every_row(model, route, layers):
        """Runtime semantics: every file gets both scores (voice stem -> AD heads, backing stem -> music heads)."""
        out = {}
        with torch.no_grad():
            for r in val_rows:
                with np.load(a.cache / f"{r['ID']}.npz") as d:
                    if route == 'voice':
                        out[r['ID']] = float(model(torch.from_numpy(d['ad_embeddings'].astype(np.float32))).softmax(-1)[:, 0].max())
                    else:
                        L = list(d['layers'])
                        f = d['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(d['mert_layers']), -1).astype(np.float32)
                        f = np.concatenate([f, eat_features(eat[0][r['ID']], eat[1])], 1) if eat else f
                        f = np.concatenate([f, sslam_features(sslam[0][r['ID']], sslam[1])], 1) if sslam else f
                        for ix, src, el in extras:
                            f = np.concatenate([f, cached(ix[r['ID']], *MUSIC_EXTRA[src][:2], el)], 1)
                        f = torch.from_numpy(f)
                        logits, _ = model.head((f - model.mean) / model.scale)
                        out[r['ID']] = float((logits.float().logsumexp(0) - np.log(len(f))).sigmoid())
        return out
    for route, key, layers in (('voice', 'voice_ad', [12]), ('music', 'music', a.music_layers)):
        data = {s: items_for(rows[s], a.cache, layers, route, eat=eat, sslam=sslam, extras=extras) for s in rows}
        if route == 'voice' and a.extra_voice:
            data['train'] = data['train'] + extra_voice_items(*a.extra_voice, dfarena=False)
        copied = a.reuse_voice if route == 'voice' else a.reuse_music
        preds, every = [], []
        for seed in (41, 42, 43):
            extra = dict(arch=a.ad_arch, max_epochs=a.voice_epochs, patience=a.voice_patience) if route == 'voice' else {}
            path = a.out / f'{key}_{seed}.pt'
            if copied:
                model, val_eer, epochs = (reused if route == 'voice' else reused_music)(path.name)
            else:
                model, val_eer, epochs = train(route, data['train'], data['val'], init if route == 'voice' else None, 0, seed=seed, **extra)
            if not copied:
                torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'route': route, 'seed': seed,
                        'arch': a.ad_arch if route == 'voice' else 'mil', 'epoch': getattr(model, 'best_epoch', epochs), 'epochs_run': epochs,
                        'validation_eer': val_eer, 'fake_label_index': 0 if route == 'voice' else old_fake,
                        'features': 'antideepfake_xlsr1b_pooled' if route == 'voice' else f'mert_layers_{layers}' + (f'+eat_blocks_{a.music_eat}' if eat else '') + (f'+sslam_blocks_{a.music_sslam}' if sslam else '') + ''.join(f'+{src}_{el}' for _, src, el in extras),
                        'train_ids': [x.row['ID'] for x, _ in data['train']]}, path)
            lock[path.name] = sha(path)
            preds.append(T.predict(model, [x for x, _ in data['val']], route, 0, 256, 10 ** 6, 'cpu'))
            every.append(every_row(model.eval(), route, layers))
            print(key, seed, round(val_eer, 5), epochs, flush=True)
        mean = np.mean(preds, 0)
        val_scores[key] = {i: float(np.mean([e[i] for e in every])) for i in every[0]}
        results[f'{route}_ensemble' if route == 'music' else 'voice_ad_ensemble'] = {'eer': float(_roc_metrics([int(x.label) for x, _ in data['val']], mean.tolist())[0])}
    if a.voice_arch == 'copy':
        for seed in (41, 42, 43):
            shutil.copy(a.ours / f'voice_{seed}.pt', a.out / f'voice_{seed}.pt')
            lock[f'voice_{seed}.pt'] = sha(a.out / f'voice_{seed}.pt')
        results['voice_ensemble'] = json.loads((a.ours / 'results.json').read_text())['voice_ensemble']
    else:   # DF-Arena fc5-input voice heads retrained with a.voice_arch on the same P1 rows (old cache features)
        old_init = torch.load(OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
        data = {s: items_for(rows[s], a.cache, [12], 'voice', dfarena=True) for s in rows}
        if a.extra_voice:
            data['train'] = data['train'] + extra_voice_items(*a.extra_voice, dfarena=True)
        preds, every = [], []
        for seed in (41, 42, 43):
            path = a.out / f'voice_{seed}.pt'
            if a.reuse_voice:
                model, val_eer, epochs = reused(path.name)
            else:
                model, val_eer, epochs = train('voice', data['train'], data['val'], old_init['state_dict'] if a.voice_arch == 'linear' else None,
                                               old_fake, seed=seed, arch=a.voice_arch, max_epochs=a.voice_epochs, patience=a.voice_patience)
            if not a.reuse_voice:
                torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'route': 'voice', 'seed': seed,
                        'arch': a.voice_arch, 'epoch': getattr(model, 'best_epoch', epochs), 'epochs_run': epochs, 'validation_eer': val_eer,
                        'fake_label_index': old_fake, 'voice_module': 'backbone.conformer.fc5', 'features': 'dfarena_fc5_input',
                        'train_ids': [x.row['ID'] for x, _ in data['train']]}, path)
            lock[path.name] = sha(path)
            preds.append(T.predict(model, [x for x, _ in data['val']], 'voice', old_fake, 256, 10 ** 6, 'cpu'))
            model.eval()
            with torch.no_grad():
                sc = {}
                for r in val_rows:
                    with np.load(OLD['val'] / f"{r['ID']}.npz") as o:
                        sc[r['ID']] = 0. if bool(o['voice_silent']) else float(model(torch.from_numpy(o['voice_embeddings'].astype(np.float32))).softmax(-1)[:, old_fake].max())
            every.append(sc)
            print('voice', a.voice_arch, seed, round(val_eer, 5), getattr(model, 'best_epoch', epochs), flush=True)
        val_scores['voice_ours'] = {i: float(np.mean([e[i] for e in every])) for i in every[0]}
        results['voice_ensemble'] = {'eer': float(_roc_metrics([int(x.label) for x, _ in data['val']], np.mean(preds, 0).tolist())[0])}
    if wavlm is not None:   # WavLM-Large voice member (mlp on the frame-mean hidden state), fused with weight a.wavlm_weight
        data = {s: wavlm_items(rows[s], wavlm, a.voice_wavlm) for s in (('val',) if a.reuse_wavlm else rows)}
        preds, every = [], []
        for seed in (41, 42, 43):
            path = a.out / f'wavlm_{seed}.pt'
            if a.reuse_wavlm:   # byte copy (S7 heads) instead of retraining
                shutil.copy(a.reuse_wavlm / path.name, path)
                record = torch.load(path, map_location='cpu', weights_only=True)
                assert record['features'] == f'wavlm_large_hidden_{a.voice_wavlm}_mean', record['features']
                model, val_eer, epochs = voice_heads.load(record).eval(), record['validation_eer'], record['epochs_run']
            else:
                model, val_eer, epochs = train('voice', data['train'], data['val'], None, 0, seed=seed, arch='mlp',
                                               max_epochs=a.voice_epochs, patience=a.voice_patience)
            if not a.reuse_wavlm:
              torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'route': 'voice', 'seed': seed,
                        'arch': 'mlp', 'width': int(data['train'][0][0].features.shape[1]), 'epoch': getattr(model, 'best_epoch', epochs),
                        'epochs_run': epochs, 'validation_eer': val_eer, 'fake_label_index': 0,
                        'features': f'wavlm_large_hidden_{a.voice_wavlm}_mean', 'train_ids': [x.row['ID'] for x, _ in data['train']]}, path)
            lock[path.name] = sha(path)
            preds.append(T.predict(model, [x for x, _ in data['val']], 'voice', 0, 256, 10 ** 6, 'cpu'))
            model.eval()
            with torch.no_grad():
                sc = {}
                for r in val_rows:
                    with np.load(wavlm[r['ID']]) as d:
                        L = list(d['wavlm_layers'])
                        f = torch.from_numpy(d['wavlm_voice'][:, [L.index(a.voice_wavlm)]].reshape(len(d['wavlm_voice']), -1).astype(np.float32))
                    sc[r['ID']] = float(model(f).softmax(-1)[:, 0].max())
            every.append(sc)
            print('voice_wavlm', seed, round(val_eer, 5), getattr(model, 'best_epoch', epochs), flush=True)
        val_scores['voice_wavlm'] = {i: float(np.mean([e[i] for e in every])) for i in every[0]}
        results['voice_wavlm_ensemble'] = {'eer': float(_roc_metrics([int(x.label) for x, _ in data['val']], np.mean(preds, 0).tolist())[0])}
    for name, src, layers, weight, ix in ssl_members:   # S8 voice members: mlp on frame-mean SSL hidden states, fused with `weight`
        key, layer_key = SSL_SOURCES[src][:2]
        data = {s: ssl_items(rows[s], ix, src, layers) for s in rows}
        preds, every = [], []
        for seed in (41, 42, 43):
            model, val_eer, epochs = train('voice', data['train'], data['val'], None, 0, seed=seed, arch='mlp',
                                           max_epochs=a.voice_epochs, patience=a.voice_patience)
            path = a.out / f'{name}_{seed}.pt'
            torch.save({'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'route': 'voice', 'seed': seed,
                        'arch': 'mlp', 'width': int(data['train'][0][0].features.shape[1]), 'epoch': getattr(model, 'best_epoch', epochs),
                        'epochs_run': epochs, 'validation_eer': val_eer, 'fake_label_index': 0, 'device': DEVICE,
                        'features': f'{src}_hidden_{layers}_mean', 'train_ids': [x.row['ID'] for x, _ in data['train']]}, path)
            lock[path.name] = sha(path)
            preds.append(T.predict(model, [x for x, _ in data['val']], 'voice', 0, 256, 10 ** 6, 'cpu'))
            model.eval()
            with torch.no_grad():
                every.append({r['ID']: float(model(torch.from_numpy(cached(ix[r['ID']], key, layer_key, layers))).softmax(-1)[:, 0].max())
                              for r in val_rows})
            print(name, seed, round(val_eer, 5), getattr(model, 'best_epoch', epochs), flush=True)
        val_scores[f'voice_{name}'] = {i: float(np.mean([e[i] for e in every])) for i in every[0]}
        results[f'voice_{name}_ensemble'] = {'eer': float(_roc_metrics([int(x.label) for x, _ in data['val']], np.mean(preds, 0).tolist())[0])}
    # fusion tables and FILE ensemble on val rows that have a backbone npz; ours = the kept DF-Arena voice heads
    scored = {s['ID']: s for s in score_rows(val_rows, OLD['val'], a.ours, 'cpu')}   # DF-Arena Fv + presence; its Fm is unused
    presence = json.loads(PRESENCE.read_text())
    ours = val_scores.get('voice_ours') or {i: s['Fv'] for i, s in scored.items()}
    spec = {'weights': {'ours': 1, 'ad': a.ad_weight},
            'tables': {'ours': dh.quantile_table([ours[r['ID']] for r in val_rows if ours[r['ID']] > 0]),
                       'ad': dh.quantile_table([val_scores['voice_ad'][r['ID']] for r in val_rows if ours[r['ID']] > 0])}}
    if wavlm is not None:
        spec['weights']['wavlm'] = a.wavlm_weight
        spec['tables']['wavlm'] = dh.quantile_table([val_scores['voice_wavlm'][r['ID']] for r in val_rows if ours[r['ID']] > 0])
    for name, _, _, weight, _ in ssl_members:
        spec['weights'][name] = weight
        spec['tables'][name] = dh.quantile_table([val_scores[f'voice_{name}'][r['ID']] for r in val_rows if ours[r['ID']] > 0])
    files = []
    for r in val_rows:
        s = scored[r['ID']]
        vp, mp = dh.presence(presence, s['presence_features'], (s['Pv'], s['Pm']))
        members = {'ours': ours[r['ID']], 'ad': val_scores['voice_ad'][r['ID']], **({'wavlm': val_scores['voice_wavlm'][r['ID']]} if wavlm is not None else {}),
                   **{n: val_scores[f'voice_{n}'][r['ID']] for n, *_ in ssl_members}}
        fv = 0. if ours[r['ID']] == 0. else fuse(spec, members)
        files.append(({**s, 'Pv': vp, 'Pm': mp, 'Fv': fv, 'Fm': val_scores['music'][r['ID']]}, r))
    keep = [(f, r) for f, r in files if r['FILE_FAKE_mask'] == '1']
    ens, ens_report = fit_file([f for f, _ in keep], np.array([int(r['FILE_FAKE']) for _, r in keep]))
    if a.file_weights:   # fixed coefficients (e.g. S7's 0,9,1); the candidate CDF tables stay refit on these val rows
        ens = {**ens, 'weights': dict(zip(('max', 'noisy_or', 'max_nopresence'), (float(x) for x in a.file_weights.split(','))))}
        ens_report = {**ens_report, 'chosen_weights': ens['weights'], 'chosen_rule': 'fixed by --file-weights'}
    preds = [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
              'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f, _ in files]
    metrics = official_metrics([r for _, r in files], preds)
    (a.out / 'file_ensemble.json').write_text(json.dumps(ens) + '\n')
    (a.out / 'component_fusion.json').write_text(json.dumps({'voice': spec, 'music': {'weights': {'ours': 1}, 'tables': {}},
                                                             'fitted_on': 'data v4 val'}) + '\n')
    (a.out / 'music_layers.json').write_text(json.dumps(a.music_layers) + '\n')
    if eat:
        (a.out / 'music_eat_layers.json').write_text(json.dumps(a.music_eat) + '\n')
    if sslam:
        (a.out / 'music_sslam_layers.json').write_text(json.dumps(a.music_sslam) + '\n')
    if wavlm is not None:
        (a.out / 'voice_wavlm_layer.json').write_text(json.dumps([a.voice_wavlm]) + '\n')
    if ssl_members:   # runtime contract (full_bank_submission.py): input voice16 = fp16-rounded voice stem, mix = raw mix
        (a.out / 'voice_ssl_members.json').write_text(json.dumps([{'name': n, 'source': src, 'layers': ls, 'weight': w,
            'input': SSL_SOURCES[src][2], 'asset': SSL_SOURCES[src][3]} for n, src, ls, w, _ in ssl_members]) + '\n')
    if extras:
        (a.out / 'music_extra.json').write_text(json.dumps([{'source': src, 'layers': el, 'input': MUSIC_EXTRA[src][2],
            'asset': MUSIC_EXTRA[src][3]} for _, src, el in extras]) + '\n')
    (a.out / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    (a.out / 'validation_metrics.json').write_text(json.dumps(metrics, indent=1) + '\n')
    (a.out / 'run_identity.json').write_text(json.dumps({'protocol': 'P1 refit: data v4 train + held-out test as train, val early stopping',
        'rows': {'train_with_backbone': sum((a.cache / f"{r['ID']}.npz").exists() for r in rows['train']), 'val_with_backbone': len(val_rows)},
        'voice_ours': str(a.ours) if a.voice_arch == 'copy' else f'retrained {a.voice_arch}', 'ad_arch': a.ad_arch,
        'voice_epochs': a.voice_epochs, 'voice_patience': a.voice_patience,
        'voice_heads_reused_from': a.reuse_voice and str(a.reuse_voice), 'music_heads_reused_from': a.reuse_music and str(a.reuse_music),
        'label_completion': a.label_completion and {'rule': 'data_v4/label_rules.py', 'train_rows_completed': completed},
        'extra_voice': a.extra_voice and [str(x) for x in a.extra_voice],
        'music_layers': a.music_layers, 'music_eat': a.music_eat, 'eat_cache': [str(c) for c in a.eat_cache],
        'music_sslam': a.music_sslam, 'sslam_cache': [str(c) for c in a.sslam_cache],
        'voice_wavlm': a.voice_wavlm, 'wavlm_weight': a.wavlm_weight if a.voice_wavlm is not None else None,
        'wavlm_heads_reused_from': a.reuse_wavlm and str(a.reuse_wavlm), 'voice_ssl': a.voice_ssl, 'music_extra': a.music_extra,
        's8_cache': [str(c) for c in a.s8_cache], 's9_cache': [str(c) for c in a.s9_cache], 'head_device': DEVICE,
        'weighting': weighting.SCHEME, 'file_weights_fixed': a.file_weights,
        'new_rows': {'csv': os.environ.get('NEW_ROWS', ''), 'n': len(added), 'new_val': os.environ.get('NEW_VAL') == '1'},
        'music_rows_with_eat': eat and {s: sum(r['ID'] in eat[0] for r in rows[s] if T.head_labels(r).get('music') is not None) for s in rows},
        'ad_weight': a.ad_weight, 'file_ensemble': ens_report}, indent=1) + '\n')
    (a.out / 'candidate_lock.json').write_text(json.dumps({'checkpoints': lock, 'selection': 'val EER then BCE per seed; seed-mean probabilities'}, indent=1) + '\n')
    print('val official', {k: round(v, 5) for k, v in metrics['metrics'].items() if v is not None}, flush=True)


if __name__ == '__main__':
    main()
