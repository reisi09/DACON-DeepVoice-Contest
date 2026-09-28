"""Locked set (LOCK_v2, 770 rows): scored ONCE, after the final candidate is fixed (DESIGN_KO 4.4, 5.1, 9.4, gate G6).
Features: the packaged-Engine npz of dump_features.py (one extraction for all packages); each package's P1 heads are
applied offline exactly as w_runtime_parity.py (DF/AD/WavLM voice members fused by component_fusion, WindowMIL music,
FILE ensemble on the stored runtime presence). G6, candidate vs each reference: paired group-bootstrap dADS CI lower
>= -0.002, and for every protected layer (source x component with two classes) the EER worsening CI upper <= 0.015;
a defined failure -> FAIL; otherwise any undefined CI (or a layer with < 20 groups in a class) -> HOLD; PASS only when every
criterion is defined and met (review round 5c). Once only: USED (created atomically before any prediction, whatever the
output name) records the candidate, the heads, rows, audio and npz hashes; it is never removed, so a second run stops.
  python score_locked.py LOCKED_ROWS LOCK_JSON OUT_JSON CAND=HEADS REF=HEADS [REF2=HEADS ...]
  LOCK_JSON NONE = unlocked sanity run on other rows: refused if any row is a locked row"""
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, '/root/proj')
import voice_heads  # noqa: E402
from component_fusion import fuse  # noqa: E402
from decision_heads import file_score  # noqa: E402
from full_bank_evaluate import load_selected_heads  # noqa: E402

ROWS, LOCK, OUT = sys.argv[1:4]
HEADS = dict(a.split('=', 1) for a in sys.argv[4:])
CAND = next(iter(HEADS))
USED = Path('/root/x3/LOCKED_SET_USED.json')
# the pre-fixed manifest (IDs, source, labels, audio_sha256, npz path): research/top15/work5/x2_20260928/locked_locked_rows.csv,
# committed and pushed in 39542f1 (2026-09-28 23:36 KST) before any metric; LOCKED_ROWS must be byte-identical (review 5d)
MANIFEST_SHA256 = 'dcae0c85dd0bda424558633a510d797cebe19691742859a20fdc335db65ee40d'
LOCKED_CSV = Path('/root/x2/locked/locked_rows.csv')
LABEL_COLS = ['ID', 'VOICE_PRESENT', 'VOICE_PRESENT_mask', 'MUSIC_PRESENT', 'MUSIC_PRESENT_mask', 'VOICE_FAKE', 'VOICE_FAKE_mask',
              'MUSIC_FAKE', 'MUSIC_FAKE_mask', 'FILE_FAKE', 'FILE_FAKE_mask', 'group_id']
COMP = {'FILE': ('FILE_FAKE', None), 'VOICE': ('VOICE_FAKE', 'VOICE_PRESENT'), 'MUSIC': ('MUSIC_FAKE', 'MUSIC_PRESENT')}


def lab(r, k):
    return int(r[k]) if r[f'{k}_mask'] == '1' and r[k] != '' else None


@torch.no_grad()
def predict(F: Path, rows):
    music, voice, fi, _ = load_selected_heads(F, 'cpu')
    ad = [voice_heads.load(torch.load(F / f'voice_ad_{s}.pt', weights_only=True)).eval() for s in (41, 42, 43)]
    wav = [voice_heads.load(torch.load(F / f'wavlm_{s}.pt', weights_only=True)).eval() for s in (41, 42, 43)]
    spec = json.loads((F / 'component_fusion.json').read_text())['voice']
    ens = json.loads((F / 'file_ensemble.json').read_text())
    layers, eat_l, ssl_l = (json.loads((F / f).read_text()) for f in ('music_layers.json', 'music_eat_layers.json', 'music_sslam_layers.json'))
    (wl,) = json.loads((F / 'voice_wavlm_layer.json').read_text())
    out = []
    for r in rows:
        with np.load(r['npz']) as d:
            pick = lambda key, lk, want: d[key][:, [list(d[lk]).index(k) for k in want]].reshape(len(d[key]), -1).astype(np.float32)
            f = torch.from_numpy(np.concatenate([pick('mert_layers', 'layers', layers), pick('eat_backing', 'eat_layers', eat_l),
                                                 pick('sslam_pt_backing', 'sslam_layers', ssl_l)], 1))
            e = torch.from_numpy(d['voice_embeddings'].astype(np.float32))
            ea = torch.from_numpy(d['ad_embeddings'].astype(np.float32))
            ew = torch.from_numpy(d['wavlm_voice'][:, list(d['wavlm_layers']).index(wl)].astype(np.float32))
            silent, (pv, pm) = bool(d['voice_silent']), map(float, d['presence'])
        fv = 0. if silent else float(np.mean([float(v(e).softmax(-1)[:, fi].max()) for v in voice]))
        fa = float(np.mean([float(m(ea).softmax(-1)[:, 0].max()) for m in ad]))
        fw = float(np.mean([float(m(ew).softmax(-1)[:, 0].max()) for m in wav]))
        fv = 0. if silent else fuse(spec, {'ours': fv, 'ad': fa, 'wavlm': fw})   # as Engine.predict: only silence skips fusion
        fm = float(np.mean([float((m.head((f - m.mean) / m.scale)[0].logsumexp(0) - np.log(len(f))).sigmoid()) for m in music]))
        ff = file_score(ens, dict(Pv=pv, Pm=pm, Fv=fv, Fm=fm))
        assert np.isfinite([fv, fm, ff]).all(), r['ID']
        out.append({'FILE': ff, 'VOICE': fv, 'MUSIC': fm})
    return out


def comp_eer(rows, preds, idx, comp):
    fake, present = COMP[comp]
    ii = [k for k in idx if lab(rows[k], fake) is not None and (present is None or lab(rows[k], present) == 1)]
    y = np.array([lab(rows[k], fake) for k in ii])
    if len(set(y.tolist())) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(y, [preds[k][comp] for k in ii], drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def ads(rows, preds, idx):
    return .5 * (1 - comp_eer(rows, preds, idx, 'FILE')) + .2 * (1 - comp_eer(rows, preds, idx, 'VOICE')) + .3 * (1 - comp_eer(rows, preds, idx, 'MUSIC'))


def paired(rows, pa, pb, idx, stat, reps=2000):
    """stat(b) - stat(a) with a paired bootstrap over group_id; NaN resamples dropped and counted."""
    groups = defaultdict(list)
    for k in idx:
        groups[rows[k]['group_id']].append(k)
    keys, rng, d = sorted(groups), np.random.default_rng(0), []
    point = stat(rows, pb, idx) - stat(rows, pa, idx)
    for _ in range(reps):
        s = [k for x in rng.choice(len(keys), len(keys)) for k in groups[keys[x]]]
        d.append(stat(rows, pb, s) - stat(rows, pa, s))
    d = np.array(d)
    ok = np.isfinite(d)
    return {'delta': round(point, 5) if np.isfinite(point) else None, 'n_groups': len(keys), 'undefined_resamples': int((~ok).sum()),
            'ci95': np.percentile(d[ok], [2.5, 97.5]).round(5).tolist() if ok.any() and np.isfinite(point) else None}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def heads_sha(d) -> str:
    return hashlib.sha256(''.join(f'{f.name}:{sha(f)}' for f in sorted(Path(d).iterdir()) if f.is_file()).encode()).hexdigest()


def verdict(v):
    checks = [(v['ADS']['ci95'], lambda ci: ci[0] >= -.002)] + [(x['ci95'], lambda ci: ci[1] <= .015) for x in v['layers'].values()]
    if any(ci is not None and not ok(ci) for ci, ok in checks):
        return 'FAIL'
    return 'HOLD' if any(ci is None for ci, _ in checks) else 'PASS'


def main():
    assert not Path(OUT).exists(), f'{OUT} exists'
    rows = sorted(csv.DictReader(open(ROWS)), key=lambda r: r['ID'])
    if LOCK != 'NONE':
        lock = json.load(open(LOCK))
        assert hashlib.sha256('\n'.join(r['ID'] for r in rows).encode()).hexdigest() == lock['ids_sha256'], 'ids differ from the lock'
        assert hashlib.sha256(json.dumps([[r[c] for c in LABEL_COLS] for r in rows]).encode()).hexdigest() == lock['labels_sha256'], 'labels differ'
        assert sha(ROWS) == MANIFEST_SHA256, 'LOCKED_ROWS is not the pre-fixed manifest'
        rc = json.load(open('/root/x3/locked_recheck.json'))   # locked_recheck.py: fresh re-extraction of a seeded sample
        assert rc['result'] == 'PASS' and rc['manifest_sha256'] == MANIFEST_SHA256, 'feature provenance recheck missing or failed'
        by_id = {r['ID']: r for r in rows}   # the sampled npz must still be the ones compared (review round 5f)
        assert rc.get('npz_sha256') and all(sha(by_id[i]['npz']) == h['stored'] for i, h in rc['npz_sha256'].items()), 'npz changed since the recheck'
        bad = [r['ID'] for r in rows if sha(r['path']) != r['audio_sha256']]
        assert not bad, f'{len(bad)} locked rows whose audio differs from the manifest audio_sha256'
        import soundfile as sf
        off = []
        for r in rows:   # ID -> npz link: the npz sample count (16 kHz) must match the audio duration
            with np.load(r['npz']) as d:
                off.append(abs(int(d['samples']) / 16000 - sf.info(r['path']).duration))
        assert max(off) <= .1, f'npz/audio duration mismatch up to {max(off):.3f} s'
        record = {'created_kst': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() + 9 * 3600)), 'candidate': CAND,
                  'heads': {n: {'dir': h, 'sha256': heads_sha(h)} for n, h in HEADS.items()}, 'rows_sha256': sha(ROWS),
                  'lock_sha256': sha(LOCK), 'npz_sha256': hashlib.sha256(''.join(sha(r['npz']) for r in rows).encode()).hexdigest(),
                  'out': str(OUT)}
        fd = os.open(USED, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # raises if the locked set was already used (any output)
        with os.fdopen(fd, 'w') as f:
            json.dump(record, f, indent=1)
    else:
        locked = {r['ID'] for r in csv.DictReader(LOCKED_CSV.open())}
        assert not locked & {r['ID'] for r in rows} and not any('/feats/locked/' in r['npz'] for r in rows), 'locked rows in a sanity run'
        print('UNLOCKED sanity run (not the locked set)', flush=True)
    preds = {n: predict(Path(h), rows) for n, h in HEADS.items()}
    idx = list(range(len(rows)))
    res = {'n': len(rows), 'heads': HEADS, 'metrics': {n: {'ADS': ads(rows, p, idx), **{c: comp_eer(rows, p, idx, c) for c in COMP}}
                                                     for n, p in preds.items()}, 'vs': {}}
    layers = defaultdict(list)
    for k, r in enumerate(rows):
        layers[r['source']].append(k)
    for ref in [n for n in HEADS if n != CAND]:
        v = {'ADS': paired(rows, preds[ref], preds[CAND], idx, ads), 'layers': {}}
        for src, ii in sorted(layers.items()):
            for c in COMP:
                if np.isfinite(comp_eer(rows, preds[ref], ii, c)):
                    x = paired(rows, preds[ref], preds[CAND], ii, lambda rr, p, s, c=c: comp_eer(rr, p, s, c))
                    cg = defaultdict(set)
                    for k in ii:
                        y, pr = lab(rows[k], COMP[c][0]), COMP[c][1]
                        if y is not None and (pr is None or lab(rows[k], pr) == 1):
                            cg[y].add(rows[k]['group_id'])
                    x['groups_per_class'] = {str(a): len(b) for a, b in cg.items()}
                    if min(x['groups_per_class'].values()) < 20:   # as the development P rule: too few groups -> undefined
                        x['ci95'] = None
                    v['layers'][f'{src}:{c}_EER'] = x
        v['G6'] = verdict(v)
        res['vs'][ref] = v
    g = [v['G6'] for v in res['vs'].values()]
    res['G6'] = 'FAIL' if 'FAIL' in g else 'HOLD' if 'HOLD' in g or not g else 'PASS'
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({'metrics': res['metrics'], 'G6_by_ref': {k: v['G6'] for k, v in res['vs'].items()},
                      'dADS': {k: v['ADS'] for k, v in res['vs'].items()}}, indent=1))
    print('G6', res['G6'], flush=True)


if __name__ == '__main__':
    main()
