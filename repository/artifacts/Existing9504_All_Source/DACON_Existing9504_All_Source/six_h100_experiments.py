"""Six predeclared experiments using real audio and the existing frozen encoders."""
from __future__ import annotations
import csv
import gc
import json
import shutil
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from graph_scoring import LABEL_COLUMNS, PREDICTION_COLUMNS, official_metrics
from redesign_checkpoint import load_checkpoint
from redesign_features import extract_features, load_records
from redesign_fit import fit_music
from redesign_h100_run import atomic_json, sha256
from redesign_submission import score_ensemble

def fuse(values: list[float], record: dict, mode: str = 'noisy_or') -> list[float]:
    result = list(values)
    _, vf, mf, vp, mp = result
    voice = vp*vf
    if mode == 'aligned':
        valid = record['valid'].astype(bool)
        voice = float((record['voice_probability'][valid]*record['presence'][valid, 0]).max()) if valid.any() else 0.
        if voice > vp*vf + 1e-6:
            raise ValueError('Aligned voice exceeds product of maxima')
    music = mp*mf
    result[0] = max(voice, music) if mode == 'max' else 1-(1-voice)*(1-music)
    if result[1:] != values[1:]:
        raise ValueError('File-only fusion changed another output')
    return result


def metrics(rows: list[dict], predictions: dict) -> dict:
    labels = [{'ID': r['ID'], **{k: r.get(k, '') for k in LABEL_COLUMNS}} for r in rows]
    values = [{'ID': r['ID'], **dict(zip(PREDICTION_COLUMNS, predictions[r['ID']], strict=True))} for r in rows]
    report = official_metrics(labels, values)
    return {'rows': len(rows), 'parent_groups': len({r['group_id'] for r in rows}), **report}


class Experiments:
    """Keep one notebook's pinned inputs, caches, and model candidates together."""

    def __init__(self, config_path: str):
        self.cfg = json.loads(Path(config_path).read_text())
        self.root = Path(self.cfg['root'])
        self.source = self.root/'source'
        self.out = self.root/'output'
        self.previous = Path(self.cfg['previous'])
        self.deadline = self.cfg['pause_deadline_epoch'] - 900
        if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('This notebook requires the authorized H100; CPU fallback is disabled')
        if time.time() >= self.deadline or shutil.disk_usage(self.root).free < 5_000_000_000:
            raise RuntimeError('Insufficient remaining time or disk for this run')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.rows = list(csv.DictReader((self.source/'probes.csv').open()))
        if sha256(self.source/'probes.csv') != self.cfg['manifest_sha256']:
            raise ValueError('Probe manifest changed after upload')
        for row in self.rows:
            row['path'] = str(self.source/row['path'])
        group_split = {}
        for row in self.rows:
            for group in row['group_id'].split('|'):
                if group in group_split and group_split[group] != row['split']:
                    raise ValueError('Parent or speaker leaked across splits')
                group_split[group] = row['split']
        self.predictions = {}
        self.heads = {}
        self.records = {}
        self.mark('preflight', gpu=torch.cuda.get_device_name(0), rows=len(self.rows),
                  absolute_deadline=self.cfg['pause_deadline_epoch'])
        shutil.copy2(self.source/'data_receipt.json', self.out/'data_receipt.json')

    def mark(self, stage: str, **extra) -> None:
        payload = dict(stage=stage, time=time.time(), **extra)
        atomic_json(self.out/'progress.json', payload)
        atomic_json(Path(self.cfg['shared'])/'progress.json', payload)
        print(json.dumps(payload), flush=True)

    def save(self, name: str, report: dict) -> dict:
        atomic_json(self.out/(name+'.json'), report)
        print(name, json.dumps(report)[:2500], flush=True)
        return report
    def extract(self) -> None:
        self.mark('features_pilot')
        canonical, by_hash, aliases = [], {}, {}
        # Identical 15 s/base probes share features; every alias retains its own labels and group.
        for row in self.rows:
            digest = row['audio_sha256']
            if digest not in by_hash:
                canonical.append(row)
                by_hash[digest] = row['ID']
            aliases[row['ID']] = by_hash[digest]
        pilot = [r for r in canonical if r['variant'] == '4s']
        pilot = [next(r for r in pilot if r['source']==source and r['FILE_FAKE']==label)
                 for source, label in [('FMA','0'), ('suno','1'), ('udio','1'), ('MLAAD','0'), ('MLAAD','1')]]
        extraction = extract_features(pilot, self.previous/'assets', self.out/'cache', deadline=self.deadline)
        self.save('timing_pilot', dict(extraction=extraction, unique_audio=len(canonical),
                  remaining_seconds=self.deadline-time.time(), prediction='4 s pilot is not a full-duration runtime benchmark'))
        gc.collect()
        torch.cuda.empty_cache()
        self.mark('features_all', unique_audio=len(canonical))
        extraction = extract_features(canonical, self.previous/'assets', self.out/'cache', deadline=self.deadline)
        records = load_records(canonical, self.out/'cache')
        self.records = {identifier: records[parent] for identifier, parent in aliases.items()}
        self.save('extraction', dict(extraction=extraction, aliases=aliases))
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self, name: str, heads: list) -> None:
        values = {}
        for row in self.rows:
            if time.time() >= self.deadline:
                raise TimeoutError('Prediction deadline reached')
            values[row['ID']] = score_ensemble(heads, self.records[row['ID']], 'cuda')
        self.predictions[name] = values
        self.heads[name] = heads
        self.save('predictions_'+name, values)

    def subset(self, split: str = 'dev', variant: str = 'base') -> list[dict]:
        return [r for r in self.rows if r['split']==split and r['variant']==variant]

    def baseline(self) -> None:
        from music_temporal import load_temporal_head, predict_temporal_music
        self.mark('experiment_1_baselines')
        heads = [load_checkpoint(self.source/f'heads/B_seed{seed}.pt', device='cuda')[0] for seed in (41,42,43)]
        self.predict('B_full', heads)
        self.predict('B_seed41', heads[:1])
        m330 = [load_temporal_head(path, 'cuda') for path in sorted((self.source/'heads/M330').glob('*.pt'))]
        if len(m330) != 5:
            raise ValueError('M330 requires the five submitted heads')
        adapted = {key: dict(music=r['stem'], music_audible=r['stem_audible'], starts=r['starts']) for key,r in self.records.items()}
        music = np.mean([predict_temporal_music(head, self.rows, adapted, 'cuda') for head in m330], axis=0)
        self.predictions['M330'] = {}
        for row, probability in zip(self.rows, music, strict=True):
            identifier = row['ID']
            values = list(self.predictions['B_full'][identifier])
            values[2] = float(probability)
            self.predictions['M330'][identifier] = fuse(values, self.records[identifier], 'max')
        # Actual submitted C08 scorer must agree with the vectorized adapter on real features.
        from c08_submission import score_ensemble as score_m330
        for row in self.rows[:3]:
            identifier = row['ID']
            expected = self.predictions['M330'][identifier]
            actual = score_m330(m330, adapted[identifier], expected[1], expected[3:], 'cuda')
            if not np.allclose(actual, expected, rtol=0, atol=1e-6):
                raise ValueError('M330 shared-feature scoring differs from submitted head API')
        self.save('predictions_M330', self.predictions['M330'])
        self.save('experiment_1', {name: metrics(self.subset(), self.predictions[name]) for name in ('M330','B_full')})

    def fusion(self) -> None:
        self.mark('experiment_2_file_fusion')
        for mode in ('max', 'aligned'):
            self.predictions['B_'+mode] = {key: fuse(values, self.records[key], mode)
                                         for key, values in self.predictions['B_full'].items()}
            self.save('predictions_B_'+mode, self.predictions['B_'+mode])
        self.save('experiment_2', {name: metrics(self.subset(), self.predictions[name]) for name in ('B_full','B_max','B_aligned')})

    def attribution(self) -> None:
        self.mark('experiment_3_component_attribution')
        result = {}
        for column in LABEL_COLUMNS[1:]:
            rows = [r for r in self.subset() if r[column] in ('0','1')]
            if not rows:
                result[column] = {'status': 'unavailable_labels'}
                continue
            changed = {}
            for row in rows:
                values = list(self.predictions['B_full'][row['ID']])
                values[list(LABEL_COLUMNS).index(column)] = float(row[column])
                changed[row['ID']] = fuse(values, self.records[row['ID']])
            result[column] = dict(before=metrics(rows, self.predictions['B_full']), oracle=metrics(rows, changed),
                                  interpretation='same known-label subset; diagnostic only; source labels not independently reviewed')
        self.save('experiment_3', result)

    def train(self, mixtures: bool = False) -> None:
        self.mark('experiment_5_mixture_training' if mixtures else 'experiment_4_hard_real_training')
        if not hasattr(self, 'old_rows'):
            self.old_rows = list(csv.DictReader((Path(self.cfg['old_run'])/'source/full_training.csv').open()))
            music_count = sum(r['MUSIC_PRESENT']=='1' and r['MUSIC_FAKE'] in ('0','1') for r in self.old_rows)
            if len(self.old_rows) != 1696 or music_count != 1216:
                raise ValueError('Original music training cohort changed')
            old_records = load_records(self.old_rows, self.previous/'cache')
            self.training_records = old_records | self.records
        added = [r for r in self.rows if r['split']=='train' and r['variant']=='base' and r['source']!='MLAAD']
        if mixtures:
            added += [r for r in self.rows if r['split']=='train' and r['source']=='constructed']
        name = 'mixture_seed41' if mixtures else 'hard_real_seed41'
        rows = self.old_rows + added
        result = fit_music(rows, self.training_records, [], self.out/'models'/name, arm='B', seed=41,
                           epochs=17, patience=1, batch_size=64, device='cuda', deadline=self.deadline, fixed_epochs=True)
        if result.report['status'] != 'complete' or result.report['optimizer_updates'] <= 0:
            raise RuntimeError('Head training did not complete real optimization')
        self.predict(name, [result.model])
        self.save('experiment_5' if mixtures else 'experiment_4', dict(training=result.report,
                  baseline=metrics(self.subset(), self.predictions['B_seed41']),
                  candidate=metrics(self.subset(), self.predictions[name]),
                  changes='frozen encoders, same B architecture/seed/epochs; only training rows changed',
                  hard_negative_policy='preselected fresh real training sources; no dev/test mining'))
        setattr(self, 'mixture_train' if mixtures else 'hard_real_train', rows)

    def finish(self) -> dict:
        self.mark('experiment_6_robustness_and_seed_gate')
        dev = self.subset()
        control = metrics(dev, self.predictions['B_seed41'])['metrics']
        candidates = []
        for name in ('hard_real_seed41', 'mixture_seed41'):
            measured = metrics(dev, self.predictions[name])['metrics']
            keys = ('file_eer', 'music_eer')
            if all(measured[k] is not None and control[k] is not None and measured[k] <= control[k] for k in keys):
                gain = .45*(control['file_eer']-measured['file_eer']) + .27*(control['music_eer']-measured['music_eer'])
                if gain > 0:
                    candidates.append((gain, name))
        selected = max(candidates)[1] if candidates else None
        self.save('selection_locked', dict(selected=selected, based_on='dev base File/Music EER only',
                  test_used=False, official_score=None, promotion_allowed=False,
                  reason='pilot source bias and unreviewed clip labels prevent a leaderboard claim'))
        if selected:
            rows = self.mixture_train if selected.startswith('mixture') else self.hard_real_train
            heads = list(self.heads[selected])
            for seed, epochs in ((42,17),(43,8)):
                result = fit_music(rows, self.training_records, [], self.out/'models'/f'selected_seed{seed}',
                                   arm='B', seed=seed, epochs=epochs, patience=1, batch_size=64,
                                   device='cuda', deadline=self.deadline, fixed_epochs=True)
                self.predict(f'selected_seed{seed}', [result.model])
                heads.append(result.model)
            self.predict('selected_ensemble', heads)
        reports = {}
        variants = sorted({r['variant'] for r in self.rows})
        # Final test is opened only after the candidate/seed decision has been written.
        for split in ('dev', 'test'):
            reports[split] = {}
            for variant in variants:
                rows = self.subset(split, variant)
                if rows:
                    reports[split][variant] = {name: metrics(rows, values) for name, values in self.predictions.items()}
            common = set.intersection(*[{r['parent_id'] for r in self.subset(split, v)} for v in ('4s','15s','60s')])
            reports[split]['matched_lengths'] = {
                variant: {name: metrics([r for r in self.subset(split,variant) if r['parent_id'] in common], values)
                          for name,values in self.predictions.items()} for variant in ('4s','15s','60s')}
        self.save('experiment_6', reports)
        self.mark('complete', selected=selected, official_score=None)
        return self.save('summary', dict(status='complete', experiments=[1,2,3,4,5,6],
                    rows=len(self.rows), variants=dict(Counter(r['variant'] for r in self.rows)),
                    selected_for_more_validation=selected, official_score=None,
                    independently_reviewed_labels=False, leaderboard_improvement_verified=False))
