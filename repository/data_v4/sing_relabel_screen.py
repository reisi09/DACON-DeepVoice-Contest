"""V-SING screen: VOICE_FAKE=1 for the vocals of fully generated songs (label completion, in memory), voice heads retrained
with the DataV4 recipe. Music heads, presence head and FILE rule stay the incumbent's; the FILE CDF tables are refit on val.

  python data_v4/sing_relabel_screen.py --primary BANK/bank_trainval.csv CACHE_TV \
      --heldout BANK/bank_test_as_val.csv CACHE_TEST --incumbent HEADS_DIR --presence PRESENCE.json --out OUT
      # screen: train split only; held-out rows are scored once against the incumbent and gated
  ... --refit    # P1 style: held-out rows join training (forced 'train'); no held-out metric, packageable dir only

OUT holds music_4x.pt (copied from --incumbent), voice_4x.pt, file_ensemble.json, results.json, run_identity.json and
the candidate_lock/validation files of full_bank_evaluate.evaluate, so package_datav4_submission.py --heads OUT works.
Manifests stay untouched, so every cache identity check of full_bank_train.read_items still applies. Held-out labels are
never completed: the comparison uses the original masks (graph_scoring.official_metrics).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data_v4'))
import decision_heads as dh  # noqa: E402
import full_bank_train as T  # noqa: E402
from file_ensemble import eer, fit as fit_file  # noqa: E402
from full_bank_batches import duration_balance, epoch_order, loss_weights  # noqa: E402
from full_bank_evaluate import evaluate, score_rows  # noqa: E402
from graph_scoring import _roc_metrics, official_metrics  # noqa: E402
from v6_data import sha256  # noqa: E402

AI_SONGS = ('existing:SONICS', 'existing:suno', 'existing:udio')   # whole song generated, vocal included
SEEDS = (41, 42, 43)
ONE = ('1', '1.0')


def completed(row: dict) -> dict | None:
    """The row with VOICE_FAKE=1 when the rule applies, else None. Never touches a known voice label."""
    if (row.get('source') in AI_SONGS and not T._known(row, 'VOICE_FAKE') and row.get('VOICE_PRESENT') in ONE
            and T._known(row, 'FILE_FAKE') and row['FILE_FAKE'] in ONE and T._known(row, 'MUSIC_FAKE') and row['MUSIC_FAKE'] in ONE):
        return {**row, 'VOICE_FAKE': '1', 'VOICE_FAKE_mask': '1'}
    return None


@contextmanager
def label_completion():
    original = T.head_labels
    T.head_labels = lambda row: original(completed(row) or row)   # read_items looks head_labels up at call time
    try:
        yield
    finally:
        T.head_labels = original


def voice_items(manifest: Path, cache: Path, force_split=None, complete=False) -> list:
    with label_completion() if complete else nullcontext():
        return [x for x in T.read_items(manifest, cache, force_split) if x.row['route'] == 'voice']


def samples_of(cache: Path) -> dict:
    return {k: v['samples'] for k, v in json.loads((cache / 'index.json').read_text())['items'].items()}


def train_voice(training, validation, samples, init, fake_index, max_epochs, patience, limit):
    """full_bank_train's voice route (cell weights x duration balance, AdamW 1e-4, val (EER, BCE) early stopping)."""
    weights = loss_weights(training)
    ids = [x.row['ID'] for x in training]
    factor = duration_balance([x.label for x in training], [samples[i] for i in ids], [weights[i] for i in ids])
    weights = {i: weights[i] * f for i, f in zip(ids, factor, strict=True)}
    labels = np.array([x.label for x in validation])
    heads = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = nn.Linear(1280, 2)
        model.load_state_dict(init)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        best, stale, state, best_epoch, run = (float('inf'), float('inf')), 0, None, 0, 0
        for epoch in range(1, max_epochs + 1):
            if stale >= patience:
                break
            model.train()
            order = epoch_order(training, seed, epoch)
            for offset in range(0, len(order), 256):
                T.step(model, optimizer, [training[i] for i in order[offset:offset + 256]], 'voice', fake_index, weights, 256, limit, 'cpu')
            pred = T.predict(model, validation, 'voice', fake_index, 256, limit, 'cpu')
            e = float(_roc_metrics(labels.astype(int).tolist(), pred.tolist())[0])
            p = np.clip(pred, 1e-7, 1 - 1e-7)
            bce = float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean())
            run = epoch
            if (e, bce) < best:
                best, stale, best_epoch = (e, bce), 0, epoch
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        model.load_state_dict(state)
        heads.append({'model': model.eval(), 'seed': seed, 'epoch': best_epoch, 'epochs_run': run,
                      'validation_eer': best[0], 'validation_bce': best[1]})
        print(f'voice seed={seed} best_epoch={best_epoch}/{run} val_eer={best[0]:.5f}', flush=True)
    return heads


@torch.no_grad()
def voice_scores(rows, cache: Path, models, fake_index) -> dict:
    """Runtime/score_rows voice path in fp32: seed mean of the window-max fake softmax, 0 for a silent stem."""
    out = {}
    for r in rows:
        with np.load(cache / f"{r['ID']}.npz", allow_pickle=False) as d:
            emb, silent = torch.from_numpy(d['voice_embeddings']), bool(d['voice_silent'])
        out[r['ID']] = 0. if silent else float(np.mean([float(m(emb).softmax(-1)[:, fake_index].max()) for m in models]))
    return out


def decision_files(scored, fv, presence):
    files = []
    for s in scored:
        vp, mp = dh.presence(presence, s['presence_features'], (s['Pv'], s['Pm']))
        files.append({**s, 'Pv': vp, 'Pm': mp, 'Fv': fv[s['ID']]})
    return files


def five(files, ens) -> list[dict]:
    return [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
             'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in files]


def fit_ensemble(rows, files):
    keep = [i for i, r in enumerate(rows) if r['FILE_FAKE'] in ('0', '1')]
    return fit_file([files[i] for i in keep], np.array([int(rows[i]['FILE_FAKE']) for i in keep]))


def eer_at(y, s):
    """EER and its operating threshold (DACON nearest point) for the FPR-on-subset diagnostic."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def compare(rows, inc, new, reps=1000, seed=20260927) -> dict:
    """Held-out comparison on official masks, voice subsets, and paired component bootstrap of the differences."""
    vmask = [i for i, r in enumerate(rows) if r['VOICE_PRESENT'] == '1' and r['VOICE_FAKE'] in ('0', '1')]
    fmask = [i for i, r in enumerate(rows) if r['FILE_FAKE'] in ('0', '1')]
    yv = np.array([int(rows[i]['VOICE_FAKE']) for i in vmask])
    yf = np.array([int(rows[i]['FILE_FAKE']) for i in fmask])
    col = lambda p, c, m: np.array([float(p[i][c]) for i in m])
    out = {'official_incumbent': official_metrics(rows, inc)['metrics'], 'official_new': official_metrics(rows, new)['metrics']}
    music = np.array([rows[i]['MUSIC_PRESENT'] == '1' for i in vmask])
    for name, preds in (('incumbent', inc), ('new', new)):
        s = col(preds, 'VOICE_FAKE_PROB', vmask)
        pooled, thr = eer_at(yv, s)
        sub = {'pooled': pooled}
        for tag, m in (('voice_only', ~music), ('with_music', music)):
            sub[tag] = eer(yv[m], s[m]) if len(set(yv[m])) == 2 else None
        real_music = music & (yv == 0)
        sub['real_with_music_fpr_at_pooled_eer'] = float((s[real_music] >= thr).mean()) if real_music.any() else None
        out[f'voice_{name}'] = sub
    comp = {}
    for k, i in enumerate(vmask):
        comp.setdefault(rows[i]['component_id'], ([], []))[0].append(k)
    for k, i in enumerate(fmask):
        comp.setdefault(rows[i]['component_id'], ([], []))[1].append(k)
    keys = sorted(comp)
    rng = np.random.default_rng(seed)
    sv = (col(inc, 'VOICE_FAKE_PROB', vmask), col(new, 'VOICE_FAKE_PROB', vmask))
    sf = (col(inc, 'FILE_FAKE_PROB', fmask), col(new, 'FILE_FAKE_PROB', fmask))
    deltas = {'voice': [], 'file': []}
    for _ in range(reps):
        pick = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
        for name, y, (a, b), slot in (('voice', yv, sv, 0), ('file', yf, sf, 1)):
            idx = np.array([k for c in pick for k in comp[c][slot]], int)
            if len(idx) and len(set(y[idx])) == 2:
                deltas[name].append(eer(y[idx], b[idx]) - eer(y[idx], a[idx]))
    for name, d in deltas.items():
        out[f'delta_{name}_eer'] = {'point': (eer(yv, sv[1]) - eer(yv, sv[0])) if name == 'voice' else (eer(yf, sf[1]) - eer(yf, sf[0])),
                                    'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))], 'valid_resamples': len(d)}
    return out


def gate(c: dict, control: dict | None = None) -> dict:
    """Pre-declared (docs/TOP15_RESEARCH_PLAN_2026-09-27.md, Stage 2 V-SING). Against the incumbent, and against the
    --no-completion control arm when given, so a gain from the longer retrain alone is not credited to the labels and a
    speech loss traded for singing is caught."""
    v, new = c['delta_voice_eer'], c['voice_new']
    checks = {'pooled_voice_gain_ge_0.010': v['point'] <= -.010 and v['ci95'][1] < 0}
    for name, ref in (('incumbent', c['voice_incumbent']), ('control', control)):
        if ref is None:
            continue
        if name == 'control':
            checks['pooled_voice_below_control'] = new['pooled'] < ref['pooled']
        checks[f'voice_only_not_worse_than_{name}+0.003'] = ref['voice_only'] is None or new['voice_only'] <= ref['voice_only'] + .003
        checks[f'real_with_music_fpr_not_above_{name}'] = (ref['real_with_music_fpr_at_pooled_eer'] is None
                                                           or new['real_with_music_fpr_at_pooled_eer'] <= ref['real_with_music_fpr_at_pooled_eer'])
    return {**checks, 'control_compared': control is not None, 'accept': all(checks.values())}


def write_heads(out: Path, heads, fake_index, module, incumbent: Path, identity_note: dict, training_ids) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base_identity = json.loads((incumbent / 'run_identity.json').read_text())
    identity = {**base_identity, 'voice_retrain': identity_note}
    for seed in SEEDS:
        shutil.copy2(incumbent / f'music_{seed}.pt', out / f'music_{seed}.pt')
    for h in heads:
        torch.save({'model': {k: v.detach().cpu() for k, v in h['model'].state_dict().items()}, 'route': 'voice', 'seed': h['seed'],
                    'epoch': h['epoch'], 'validation_eer': h['validation_eer'], 'validation_bce': h['validation_bce'],
                    'identity': identity, 'fake_label_index': fake_index, 'voice_module': module, 'train_ids': training_ids},
                   out / f"voice_{h['seed']}.pt")
    (out / 'run_identity.json').write_text(json.dumps(identity, indent=1) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--primary', nargs=2, type=Path, required=True, metavar=('MANIFEST', 'CACHE'), help='train/val bank and its cache')
    ap.add_argument('--heldout', nargs=2, type=Path, required=True, metavar=('MANIFEST', 'CACHE'), help='held-out bank (split val) and cache')
    ap.add_argument('--incumbent', type=Path, required=True, help='heads dir of the package to beat (music heads are reused)')
    ap.add_argument('--presence', type=Path, required=True, help='presence_head.json packaged with the incumbent')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--refit', action='store_true', help='train on the held-out rows too (P1 style); skips the held-out report')
    ap.add_argument('--max-epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--no-completion', action='store_true', help='control arm: same retrain without label completion')
    ap.add_argument('--control', type=Path, help='sing_screen_report.json of the --no-completion arm, added to the gate')
    a = ap.parse_args()
    torch.set_num_threads(4)
    (pm, pc), (hm, hc) = a.primary, a.heldout
    complete = not a.no_completion
    items = voice_items(pm, pc, complete=complete)
    training = [x for x in items if x.row['split'] == 'train']
    validation = [x for x in items if x.row['split'] == 'val']
    heldout_rows = list(csv.DictReader(hm.open()))
    added = sum(1 for r in csv.DictReader(pm.open()) if r['split'] == 'train' and completed(r))
    if a.refit:
        extra = voice_items(hm, hc, 'train', complete=complete)
        if {x.row['component_id'] for x in extra} & {x.row['component_id'] for x in validation}:
            raise ValueError('Held-out rows share components with validation')   # same guard as full_bank_train --extra-train
        if {x.row['ID'] for x in extra} & {x.row['ID'] for x in training}:
            raise ValueError('Held-out rows duplicate training IDs')
        training += extra
        added += sum(1 for r in heldout_rows if completed(r))
    samples = {**samples_of(pc), **samples_of(hc)}
    init = torch.load(pc / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
    fake_index = init['fake_label_index']
    heldout = [] if a.refit else voice_items(hm, hc)
    longest = sorted(training + validation + heldout, key=lambda x: len(x.features), reverse=True)[:256]
    limit = sum(len(x.features) for x in longest)
    print(json.dumps({'train_voice_items': len(training), 'val_voice_items': len(validation), 'completed_rows': added,
                      'completion': complete, 'refit': a.refit}), flush=True)
    heads = train_voice(training, validation, samples, init['state_dict'], fake_index, a.max_epochs, a.patience, limit)
    note = {'label_completion': {'sources': list(AI_SONGS), 'rows': added} if complete else None, 'refit_heldout': a.refit,
            'max_epochs': a.max_epochs, 'patience': a.patience, 'script_sha256': sha256(Path(__file__)),
            'best_epochs': {h['seed']: [h['epoch'], h['epochs_run']] for h in heads}}
    write_heads(a.out, heads, fake_index, init['module'], a.incumbent, note, [x.row['ID'] for x in training])
    presence = json.loads(a.presence.read_text())
    val_rows = [r for r in csv.DictReader(pm.open()) if r['split'] == 'val']
    models = [h['model'] for h in heads]
    inc_val = score_rows(val_rows, pc, a.incumbent, 'cpu')
    files_val = decision_files(inc_val, voice_scores(val_rows, pc, models, fake_index), presence)
    ens, ens_report = fit_ensemble(val_rows, files_val)
    (a.out / 'file_ensemble.json').write_text(json.dumps(ens) + '\n')
    val_pred = np.mean([T.predict(m, validation, 'voice', fake_index, 256, limit, 'cpu') for m in models], axis=0)
    ve, va = _roc_metrics([int(x.label) for x in validation], val_pred.tolist())
    base_results = json.loads((a.incumbent / 'results.json').read_text())
    (a.out / 'results.json').write_text(json.dumps({'voice_ensemble': {'eer': ve, 'auc': va},
                                                    'music_ensemble': base_results['music_ensemble']}, indent=1) + '\n')
    evaluate(pm, pc, a.out, 'cpu')   # validation_predictions.csv, validation_metrics.json, candidate_lock.json (checkpoint hashes)
    report = {'setup': note, 'val_voice_ensemble_eer': ve, 'file_ensemble': ens_report}
    if not a.refit:
        inc_ho = score_rows(heldout_rows, hc, a.incumbent, 'cpu')
        inc_ens = json.loads((a.incumbent / 'file_ensemble.json').read_text())
        inc_files = decision_files(inc_ho, {s['ID']: s['Fv'] for s in inc_ho}, presence)
        new_files = decision_files(inc_ho, voice_scores(heldout_rows, hc, models, fake_index), presence)
        report['heldout'] = compare(heldout_rows, five(inc_files, inc_ens), five(new_files, ens))
        control = json.loads(a.control.read_text())['heldout']['voice_new'] if a.control else None
        report['gate'] = gate(report['heldout'], control)
    (a.out / 'sing_screen_report.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    print(json.dumps({k: report[k] for k in ('val_voice_ensemble_eer', 'gate') if k in report}, default=float), flush=True)


def self_check() -> None:
    """The rule completes only fully generated songs with a present vocal and an unknown voice label."""
    base = dict(source='existing:SONICS', FILE_FAKE='1', FILE_FAKE_mask='1', MUSIC_FAKE='1', MUSIC_FAKE_mask='1',
                VOICE_PRESENT='1', VOICE_PRESENT_mask='1', VOICE_FAKE='', VOICE_FAKE_mask='0', route='music')
    assert completed(base)['VOICE_FAKE'] == '1'
    assert T.head_labels(base) == {'music': 1.}, 'the unpatched loader must ignore the unknown voice label'
    with label_completion():
        assert T.head_labels(base) == {'music': 1., 'voice': 1.}
    assert T.head_labels(base) == {'music': 1.}, 'patch restored'
    for change in (dict(source='haim'), dict(source='existing:FMA'), dict(VOICE_PRESENT='0'), dict(VOICE_PRESENT=''),
                   dict(FILE_FAKE='0'), dict(MUSIC_FAKE='0'), dict(MUSIC_FAKE='', MUSIC_FAKE_mask='0'),
                   dict(VOICE_FAKE='0', VOICE_FAKE_mask='1')):
        assert completed({**base, **change}) is None, change
    print('sing_relabel_screen self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
