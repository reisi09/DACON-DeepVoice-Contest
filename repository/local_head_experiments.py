"""Bounded local experiments on verified cached features and preserved DataV4 heads.

Outputs are diagnostic: these cached roles were exposed to the reference model.
No encoder extraction, network access, cloud startup, or submission is performed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import zipfile
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from decision_heads import file_score
from eight_fixed_io import atomic_json
from full_bank_batches import Item, loss_weights, duration_balance
from full_bank_contract import pool_odds, config_for
from full_bank_evaluate import load_selected_heads, five_columns
from full_bank_features import load_rows, row_identity, validate_cache
from full_bank_train import head_labels
from graph_scoring import official_metrics, _roc_metrics
from localization_models import masked_logmeanexp
from music_temporal import ResidualTemporalBlock
from v6_data import sha256


def load_reference(archive: Path, destination: Path):
    """Read only allowlisted, receipt-verified members; weights_only loading below."""
    destination.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(archive.with_suffix('.json').read_text())
    names = [f'{route}_{seed}.pt' for route in ('music', 'voice') for seed in (41, 42, 43)]
    names += ['run_identity.json', 'presence_head.json', 'file_ensemble.json']
    with zipfile.ZipFile(archive) as z:
        for name in names:
            member = 'model/trained/' + name
            content = z.read(member)
            if hashlib.sha256(content).hexdigest() != receipt['member_sha256'][member]:
                raise ValueError('Reference archive member differs from receipt: ' + member)
            target = destination / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError('Refusing to replace a different reference artifact')
            if not target.exists():
                target.write_bytes(content)
    return load_selected_heads(destination, 'cpu')


def read_records(manifest: Path, cache: Path):
    rows = load_rows(manifest)
    index = json.loads((cache / 'index.json').read_text())
    key = hashlib.sha256(json.dumps(index['identity'], sort_keys=True).encode()).hexdigest()
    records = []
    for row in rows:
        entry = index['items'][row['ID']]
        path = (cache / entry['path']).resolve()
        if not path.is_relative_to(cache.resolve()):
            raise ValueError('Cache path escapes authorized root')
        validate_cache(path, row_identity(row, key), entry['sha256'])
        with np.load(path, allow_pickle=False) as data:
            records.append(dict(row=row, music=data['mert_windows'].copy(),
                voice=data['voice_embeddings'].copy(), starts=data['starts'].copy(),
                voice_starts=data['voice_starts'].copy(), silent=bool(data['voice_silent']),
                presence=data['presence'].copy(), samples=int(data['samples']),
                presence_features=json.loads(str(data['presence_features']))))
    return records, index


def padded_batch(records, route, device):
    lengths = [len(r[route]) for r in records]
    width = 1024 if route == 'music' else 1280
    values = torch.zeros(len(records), max(lengths), width)
    valid = torch.zeros(len(records), max(lengths), dtype=torch.bool)
    starts = torch.zeros(len(records), max(lengths), dtype=torch.int64)
    for i, (r, length) in enumerate(zip(records, lengths, strict=True)):
        values[i, :length] = torch.from_numpy(r[route])
        valid[i, :length] = True
        starts[i, :length] = torch.from_numpy(r['starts' if route == 'music' else 'voice_starts'])
    return values.to(device), valid.to(device), starts.to(device)


class MusicResidual(nn.Module):
    """Frozen WindowMIL plus a pointwise/temporal correction; keep every valid tail."""
    def __init__(self, base, *, kernel=3, width=64):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.norm = nn.LayerNorm(1024)
        # Both controls see measured adjacent time gaps, including the forced tail.
        self.projection = nn.Conv1d(1026, width, 1)
        self.block = ResidualTemporalBlock(1, dropout=0., channels=width,
                                          layers=3, kernel_size=kernel)
        self.output = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, features, valid, starts):
        features = features.masked_fill(~valid[..., None], 0)
        old = self.base(features, valid)
        pair = valid[:, 1:] & valid[:, :-1]
        gaps = ((starts[:, 1:] - starts[:, :-1]).float() / 32300).masked_fill(~pair, 0)
        if torch.any(gaps[pair] <= 0):
            raise ValueError('Valid music windows must have increasing start times')
        before = F.pad(gaps, (1, 0)); after = F.pad(gaps, (0, 1))
        normalized = self.norm((features - self.base.mean) / self.base.scale)
        x = torch.cat((normalized, before[..., None], after[..., None]), -1)
        mask = ~valid[:, None, :]
        hidden = F.gelu(self.projection(x.transpose(1, 2))).masked_fill(mask, 0)
        delta = self.output(self.block(hidden, mask)).squeeze(1).masked_fill(~valid, 0)
        return masked_logmeanexp(old.window_logits + delta, valid)


class VoiceResidual(nn.Module):
    def __init__(self, base, fake_index, *, width=128):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.fake_index = fake_index
        self.correction = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, width),
                                        nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode=True):
        super().train(mode); self.base.eval()
        return self

    def forward(self, features, valid, starts):
        features = features.masked_fill(~valid[..., None], 0)
        logits = self.base(features)
        odds = logits[..., self.fake_index] - logits[..., 1-self.fake_index]
        odds = odds + self.correction(features).squeeze(-1)
        return odds.masked_fill(~valid, -torch.inf).max(1).values


class VoiceLinear(nn.Module):
    def __init__(self, base, fake_index):
        super().__init__()
        self.base, self.fake_index = base.requires_grad_(True), fake_index

    def forward(self, features, valid, starts):
        logits = self.base(features.masked_fill(~valid[..., None], 0))
        odds = logits[..., self.fake_index] - logits[..., 1-self.fake_index]
        return odds.masked_fill(~valid, -torch.inf).max(1).values


def eligible(record, route):
    row = record['row']
    return route in head_labels(row) and not (route == 'voice' and row['route'] != route and record['silent'])


def weights_for(records, route, *, rebalance=False):
    items = [Item(r['row'], r[route], np.full(len(r[route]), np.nan),
                  head_labels(r['row'])[route], r['silent']) for r in records]
    base = loss_weights(items)
    ids = [r['row']['ID'] for r in records]
    labels = [i.label for i in items]
    weights = np.array([base[i] for i in ids])
    weights *= duration_balance(labels, [r['samples'] for r in records], weights)
    if rebalance:
        # A cached-data weighting diagnostic, not the unavailable new-data experiment.
        domain = np.array([r['row']['MUSIC_PRESENT'] == '1' for r in records])
        for label in (0., 1.):
            selected = np.asarray(labels) == label
            original_mass = weights[selected].sum()
            for present in (False, True):
                cell = selected & (domain == present)
                if cell.any():
                    weights[cell] *= np.clip(original_mass / (2 * weights[cell].sum()), .25, 4.)
            weights[selected] *= original_mass / weights[selected].sum()
    return {i: float(w) for i, w in zip(ids, weights, strict=True)}


@torch.no_grad()
def predict_model(model, records, route, device, batch_size=64):
    model.eval()
    result = []
    for start in range(0, len(records), batch_size):
        group = records[start:start+batch_size]
        values = model(*padded_batch(group, route, device)).sigmoid().cpu().numpy()
        if route == 'voice':
            values = np.where([r['silent'] for r in group], 0., values)
        result.extend(values.tolist())
    return np.asarray(result)


def timed_steps(model, records, route, device, count=8):
    """Real forward/backward; caller passes a disposable copy of the reference."""
    model.to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x = padded_batch(records[:32], route, device)
    y = torch.tensor([head_labels(r['row'])[route] for r in records[:32]], device=device)
    if device == 'mps': torch.mps.synchronize()
    start = time.monotonic()
    for _ in range(count):
        optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(model(*x), y).backward()
        optimizer.step()
    if device == 'mps': torch.mps.synchronize()
    return (time.monotonic() - start) / count


def train_candidate(model, records, route, *, device, seed, epochs, output, rebalance=False):
    training = [r for r in records if r['row']['role'] == 'train' and eligible(r, route)]
    selection = [r for r in records if r['row']['role'] == 'selection' and eligible(r, route)]
    weights = weights_for(training, route, rebalance=rebalance)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-4 if route == 'music' else 1e-4, weight_decay=.01)
    history = []; start = time.monotonic(); steps = 0
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + (epoch+1)*100003).permutation(len(training))
        total = 0.
        for offset in range(0, len(order), 128):
            batch = [training[i] for i in order[offset:offset+128]]
            if any(r['row']['role'] != 'train' for r in batch):
                raise ValueError('Non-training row reached optimizer')
            logits = model(*padded_batch(batch, route, device))
            labels = torch.tensor([head_labels(r['row'])[route] for r in batch], device=device)
            weight = torch.tensor([weights[r['row']['ID']] for r in batch], device=device)
            loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weight).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step(); steps += 1; total += float(loss.detach()) * len(batch)
        predictions = predict_model(model, selection, route, device)
        labels = [head_labels(r['row'])[route] for r in selection]
        metrics = _roc_metrics(labels, predictions.tolist())
        history.append(dict(epoch=epoch+1, steps=steps, loss=total/len(training),
                            selection_eer=float(metrics[0]), selection_auc=float(metrics[1])))
        print(json.dumps(dict(candidate=output.stem, **history[-1])), flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 14*2**30:
            raise RuntimeError('Local process exceeded 14 GiB RSS ceiling')
    model.eval().cpu()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(dict(model=state, seed=seed, steps=steps, epochs=epochs, route=route), output)
    report = dict(seed=seed, route=route, epochs=epochs, steps=steps, train_rows=len(training),
                  trainable_parameters=sum(p.numel() for p in parameters),
                  elapsed_seconds=time.monotonic()-start, history=history,
                  choice='fixed final epoch; no evaluation-based early stopping',
                  training_ids_sha256=hashlib.sha256('\n'.join(r['row']['ID'] for r in training).encode()).hexdigest())
    atomic_json(output.with_suffix('.json'), report)
    return model, report


def known(row, key):
    return row.get(key) in ('0', '1', '0.0', '1.0') and row.get(key+'_mask', '1') not in ('0', '0.0', 'False')


def slice_metric(records, scores, route, *, domain=None):
    key = route.upper() + '_FAKE'
    present = route.upper() + '_PRESENT'
    indices = [i for i, r in enumerate(records) if known(r['row'], key)
               and r['row'].get(present) == '1'
               and (domain is None or (r['row']['MUSIC_PRESENT'] == '1') == domain)]
    labels = [int(float(records[i]['row'][key])) for i in indices]
    result = dict(rows=len(indices), real=labels.count(0), fake=labels.count(1), eer=None, auc=None)
    if set(labels) == {0, 1}:
        result['eer'], result['auc'] = map(float, _roc_metrics(labels, [float(scores[i]) for i in indices]))
    return result


def cluster_difference(records, baseline, candidate, route, repeats=400):
    """Paired resampling of existing closed parent components; no split changes."""
    key = route.upper() + '_FAKE'; present = route.upper() + '_PRESENT'
    groups = {}
    for i, r in enumerate(records):
        if known(r['row'], key) and r['row'].get(present) == '1':
            groups.setdefault(r['row']['component_id'], []).append(i)
    keys = list(groups); rng = np.random.default_rng(20260925); differences = []
    for _ in range(repeats):
        indices = [i for j in rng.integers(0, len(keys), len(keys)) for i in groups[keys[j]]]
        labels = [int(float(records[i]['row'][key])) for i in indices]
        if set(labels) != {0, 1}: continue
        old = _roc_metrics(labels, baseline[indices].tolist())[0]
        new = _roc_metrics(labels, candidate[indices].tolist())[0]
        differences.append(float(new-old))
    return dict(definition='candidate EER minus reference EER; negative is better',
                groups=len(keys), valid_replicates=len(differences),
                interval95=np.quantile(differences, [.025, .975]).tolist() if differences else None,
                independent_generalization=False)


def validate_reference_cache(archive, index):
    receipt = json.loads(archive.with_suffix('.json').read_text())
    members = receipt['member_sha256']
    expected = {'model/assets/'+k: v for k, v in index['identity']['assets'].items()}
    expected.update({'model/runtime/'+k: v for k, v in index['identity']['implementation'].items()
                     if k not in ('full_bank_features.py', 'full_bank_contract.py')})
    if any(members.get(k) != v for k, v in expected.items()):
        raise ValueError('Reference encoder/preprocessing differs from cached producer')


def audit_manifests(records, index, old_banks, new_banks):
    read = lambda paths: [r for p in paths for r in csv.DictReader(p.open())]
    old = read(old_banks); new = read(new_banks); lookup = {r['ID']: r for r in old}
    added = [r for r in new if r['ID'] not in lookup]
    overlap = Counter((r['row']['role'], lookup.get(r['row']['ID'], {}).get('split', 'not_in_old')) for r in records)
    parent_roles = {}
    audio_roles = {}
    for r in records:
        row = r['row']
        parent_roles.setdefault(row['component_id'], set()).add(row['role'])
        audio_roles.setdefault(row['audio_sha256'], set()).add(row['role'])
    if any(len(v) > 1 for v in parent_roles.values()) or any(len(v) > 1 for v in audio_roles.values()):
        raise ValueError('Parent or exact audio crosses current training/evaluation roles')
    return dict(cache_rows=len(records), cached_bytes=sum(r['music'].nbytes+r['voice'].nbytes for r in records),
                roles=dict(Counter(r['row']['role'] for r in records)),
                old_exposure=[dict(current_role=a, old_split=b, rows=n) for (a,b),n in sorted(overlap.items())],
                added_rows=len(added), added_sources=dict(Counter(r['source'] for r in added)),
                added_rows_cached=sum(r['ID'] in index['items'] for r in added),
                cross_role_components=0, cross_role_audio_hashes=0,
                current_roles_disjoint=True, independent_reference_evaluation=False)


def original_scores(heads, records):
    """Use existing ragged inference and exactly the preserved seed/fake mapping."""
    from full_bank_train import predict
    music, voice, fake, _ = heads
    scores = {}
    for route, models in [('music', music), ('voice', voice)]:
        items = [Item(r['row'], r[route], np.full(len(r[route]), np.nan), 0., r['silent']) for r in records]
        scores[route] = np.stack([predict(h, items, route, fake, 64, 65536, 'cpu') for h in models])
    scored = [dict(ID=r['row']['ID'], Pv=float(r['presence'][0]), Pm=float(r['presence'][1]),
                   Fm=float(scores['music'][:, i].mean()), Fv=float(scores['voice'][:, i].mean()),
                   presence_features=r['presence_features']) for i, r in enumerate(records)]
    return scores, scored


def run(args):
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(6)
    if torch.backends.mps.is_available(): torch.mps.set_per_process_memory_fraction(.5)
    config = dict(epochs=args.epochs, seeds=[41,42,43], effective_batch=128,
                  music_lr=3e-4, voice_lr=1e-4, selection='fixed final epoch',
                  source_sha256=sha256(Path(__file__)), temporal_source_sha256=sha256(Path(__file__).with_name('music_temporal.py')),
                  dataset_manifest_sha256=sha256(args.manifest), independent_evaluation=False)
    atomic_json(args.output/'protocol.json', config)
    heads = load_reference(args.archive, args.output/'reference')
    records, index = read_records(args.manifest, args.cache)
    validate_reference_cache(args.archive, index)
    audit = audit_manifests(records, index, args.old_bank, args.new_bank)
    atomic_json(args.output/'data_audit.json', audit)
    evaluation = [r for r in records if r['row']['role'] in ('selection','calibration')]
    seed_scores, scored = original_scores(heads, evaluation)
    reference = args.output/'reference'
    presence = json.loads((reference/'presence_head.json').read_text())
    ensemble = json.loads((reference/'file_ensemble.json').read_text())
    baseline = [five_columns(s, presence, ensemble) for s in scored]
    roles = {role: [i for i,r in enumerate(evaluation) if r['row']['role']==role] for role in ('selection','calibration')}

    def reports(predictions):
        return {role: official_metrics([evaluation[i]['row'] for i in indices], [predictions[i] for i in indices])
                for role, indices in roles.items()}

    results = {'reference': reports(baseline)}
    slices = {}
    for role, indices in roles.items():
        selected = [evaluation[i] for i in indices]
        v = seed_scores['voice'].mean(0)[indices]
        slices[role] = dict(voice_only=slice_metric(selected, v, 'voice', domain=False),
                            voice_with_music=slice_metric(selected, v, 'voice', domain=True),
                            music=slice_metric(selected, seed_scores['music'].mean(0)[indices], 'music'))
    oracle = {}
    for route, score_key in [('music','Fm'),('voice','Fv')]:
        replaced = [dict(s) for s in scored]; touched = 0
        for r,s in zip(evaluation,replaced,strict=True):
            key=route.upper()+'_FAKE'
            if known(r['row'],key) and r['row'][route.upper()+'_PRESENT']=='1':
                s[score_key]=float(r['row'][key]); touched+=1
        oracle[route] = dict(replaced_rows=touched, rule='only known fake truth in present components; presence unchanged',
                             reports=reports([five_columns(s,presence,ensemble) for s in replaced]))
    atomic_json(args.output/'diagnostics.json',dict(reference=results['reference'],slices=slices,oracle=oracle))
    print(json.dumps(dict(phase='diagnostics', data=audit, slices=slices)),flush=True)

    # Pooling diagnostics reuse the exact frozen segment logits; no model fitting.
    pooling = {}
    for mode in ('lme','topk'):
        per_seed=[]
        for head in heads[1]:
            values=[]
            with torch.no_grad():
                for r in evaluation:
                    z=head(torch.from_numpy(r['voice']))
                    odds=z[:,heads[2]]-z[:,1-heads[2]]
                    values.append(0. if r['silent'] else float(pool_odds(odds,config_for({'voice_pooling':mode,'voice_topk_fraction':.3})).sigmoid()))
            per_seed.append(values)
        pooling[mode]=reports([five_columns({**s,'Fv':float(v)},presence,ensemble) for s,v in zip(scored,np.mean(per_seed,0),strict=True)])
    atomic_json(args.output/'pooling_diagnostic.json',pooling)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    probabilities=seed_scores['voice'].mean(0)
    odds=np.log(np.clip(probabilities,1e-6,1-1e-6)/np.clip(1-probabilities,1e-6,1))
    singing=np.array([r['presence_features']['tag_singing_max'] for r in evaluation])
    fit=[i for i in roles['selection'] if known(evaluation[i]['row'],'VOICE_FAKE') and evaluation[i]['row']['VOICE_PRESENT']=='1']
    calibration={}
    for name,features in [('logit_only',odds[:,None]),('logit_plus_singing',np.column_stack((odds,singing)))]:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=41))
        model.fit(features[fit],[float(evaluation[i]['row']['VOICE_FAKE']) for i in fit])
        predicted=model.predict_proba(features)[:,1]
        predicted=np.where([r['silent'] for r in evaluation],0.,predicted)
        calibration[name]=dict(fitted_role='selection; exposed development diagnostic only',
            coefficients=model[-1].coef_.tolist(),reports=reports([five_columns({**s,'Fv':float(v)},presence,ensemble) for s,v in zip(scored,predicted,strict=True)]))
    atomic_json(args.output/'voice_calibration_diagnostic.json',calibration)

    measurements={}
    devices={}
    for route in ('music','voice'):
        training=[r for r in records if r['row']['role']=='train' and eligible(r,route)]
        make=lambda: MusicResidual(copy.deepcopy(heads[0][0]),kernel=3) if route=='music' else VoiceResidual(copy.deepcopy(heads[1][0]),heads[2])
        measurements[route]={}
        for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
            torch.manual_seed(41)
            disposable=make()
            measurements[route][device]=timed_steps(disposable,training,route,device)
            del disposable
            if device=='mps': torch.mps.empty_cache()
        devices[route]=min(measurements[route],key=measurements[route].get)
    atomic_json(args.output/'local_benchmark.json',dict(seconds_per_real_step=measurements,chosen=devices,torch=torch.__version__))
    print(json.dumps(dict(phase='benchmark',seconds_per_real_step=measurements,chosen=devices)),flush=True)

    variants=[('music_k1','music',False),('music_k3','music',False),
              ('voice_linear','voice',False),('voice_linear_reweight','voice',True)]
    mixed=slices['selection']['voice_with_music']
    if mixed['eer'] is not None and mixed['eer']>0:
        variants.append(('voice_mlp','voice',False))
    atomic_json(args.output/'candidate_schedule.json',dict(candidates=variants,
        voice_mlp_condition='observed within music-present voice ranking errors; exploratory only',
        new_data_training='not run: added1477 have no cached features'))
    candidate_scores={}
    training_reports={}
    for name,route,rebalanced in variants:
        predictions=[]; training_reports[name]=[]
        for k,seed in enumerate((41,42,43)):
            torch.manual_seed(seed)
            if route=='music':
                model=MusicResidual(copy.deepcopy(heads[0][k]),kernel=int(name[-1]))
            elif name=='voice_mlp':
                model=VoiceResidual(copy.deepcopy(heads[1][k]),heads[2])
            else:
                model=VoiceLinear(copy.deepcopy(heads[1][k]),heads[2])
            model,report=train_candidate(model,records,route,device=devices[route],seed=seed,
                epochs=args.epochs,output=args.output/f'{name}_{seed}.pt',rebalance=rebalanced)
            predicted=predict_model(model,evaluation,route,'cpu')
            predictions.append(predicted); training_reports[name].append(report)
            # Replay the actual saved parameters, not a mock serialization.
            model.load_state_dict(torch.load(args.output/f'{name}_{seed}.pt',weights_only=True)['model'])
            replay=predict_model(model,evaluation,route,'cpu')
            if not np.array_equal(predicted,replay):raise ValueError('Saved candidate replay changed predictions')
            del model
            if devices[route]=='mps':torch.mps.empty_cache()
        all_scores=np.stack(predictions); candidate_scores[name]=all_scores
        mean=all_scores.mean(0); key='Fm' if route=='music' else 'Fv'
        final=[five_columns({**s,key:float(v)},presence,ensemble) for s,v in zip(scored,mean,strict=True)]
        for a,b in zip(baseline,final,strict=True):
            for column in ('VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB'):
                if a[column]!=b[column]:raise ValueError('Frozen presence output changed')
        result=reports(final)
        result['paired_component_uncertainty']={role: cluster_difference([evaluation[i] for i in ix],seed_scores[route].mean(0)[ix],mean[ix],route) for role,ix in roles.items()}
        result['seed_metrics']={role:[slice_metric([evaluation[i] for i in ix],p[ix],route) for p in all_scores] for role,ix in roles.items()}
        result['slices']={role:dict(voice_only=slice_metric([evaluation[i] for i in ix],mean[ix],route,domain=False),voice_with_music=slice_metric([evaluation[i] for i in ix],mean[ix],route,domain=True)) for role,ix in roles.items()} if route=='voice' else {}
        results[name]=result
        atomic_json(args.output/'comparison.json',results)
        atomic_json(args.output/'training_summary.json',training_reports)
        np.savez(args.output/(name+'_predictions.npz'),IDs=np.array([r['row']['ID'] for r in evaluation]),seed_scores=all_scores)
        print(json.dumps(dict(phase='candidate_complete',candidate=name,metrics={k:v['metrics'] for k,v in result.items() if k in roles})),flush=True)
    np.savez(args.output/'reference_predictions.npz',IDs=np.array([r['row']['ID'] for r in evaluation]),**seed_scores)
    if all(x in candidate_scores for x in ('music_k1','music_k3')):
        atomic_json(args.output/'temporal_vs_pointwise.json',{role:cluster_difference([evaluation[i] for i in ix],candidate_scores['music_k1'].mean(0)[ix],candidate_scores['music_k3'].mean(0)[ix],'music') for role,ix in roles.items()})
    atomic_json(args.output/'completion.json',dict(completed=True,elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,candidates=list(candidate_scores),
        frozen_presence_exact=True,checkpoint_prediction_replay_exact=True,independent_quality_validated=False,
        no_cloud=True,no_submission=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--old-bank',type=Path,nargs='+',required=True)
    parser.add_argument('--new-bank',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    if args.epochs<1:parser.error('epochs must be positive')
    run(args)
