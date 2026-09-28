"""Audit emitted manifests and demonstrate that parent leakage/unknown labels are rejected.
Run: .venv/bin/python verify_reconstructed_bank.py
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from reconstruct_candidate_bank import OUTPUT, LABELS, SPLITS, verify, sha, SplitError


def main():
    rows = list(csv.DictReader((OUTPUT / 'selected.csv').open()))
    ledger = list(csv.DictReader((OUTPUT / 'combined_catalog.csv').open()))
    report = json.loads((OUTPUT / 'verification.json').read_text())
    verify(rows, ledger, report['split_class_counts'])
    if sha(OUTPUT / 'selected.csv') != report['selected_sha256']:
        raise SplitError('Frozen manifest hash drift')
    for split in SPLITS:
        written = list(csv.DictReader((OUTPUT / (split + '.csv')).open()))
        if written != [r for r in rows if r['split'] == split]:
            raise SplitError('Split file differs from selected manifest')
    damaged = deepcopy(rows)
    training = next(r for r in damaged if r['split'] == 'train')
    validation = next(r for r in damaged if r['split'] == 'val')
    validation['group_id'] += '|id:' + training['ID']
    try:
        verify(damaged, ledger, report['split_class_counts'])
    except SplitError as error:
        if 'Cross-partition' not in str(error):
            raise
    else:
        raise AssertionError('A parent derivative leaked across splits without rejection')
    damaged = deepcopy(rows)
    unknown = next((r, f) for r in damaged for f in LABELS if r.get(f) == '')
    unknown[0][unknown[1] + '_mask'] = '1'
    try:
        verify(damaged, ledger, report['split_class_counts'])
    except SplitError as error:
        if 'Unknown label unmasked' not in str(error):
            raise
    else:
        raise AssertionError('Unknown component truth was accepted as a training target')
    diagnostics = {}
    for route in ('voice', 'music'):
        training = [r for r in rows if r['split'] == 'train' and r['route'] == route]
        source_labels = defaultdict(list)
        for row in training:
            source_labels[row['source']].append(int(row['FILE_FAKE']))
        for split in ('val', 'test'):
            subset = [r for r in rows if r['split'] == split and r['route'] == route]
            labels = [int(r['FILE_FAKE']) for r in subset]
            source_scores = [np.mean(source_labels[r['source']]) if r['source'] in source_labels else .5 for r in subset]
            # Fix direction from training labels only; no tuning on validation/test scores.
            real = [float(r['duration_seconds']) for r in training if r['FILE_FAKE'] == '0']
            fake = [float(r['duration_seconds']) for r in training if r['FILE_FAKE'] == '1']
            sign = 1 if np.median(fake) >= np.median(real) else -1
            diagnostics[route + '_' + split] = dict(source_only_auc=float(roc_auc_score(labels, source_scores)),
                duration_only_auc=float(roc_auc_score(labels, [sign * float(r['duration_seconds']) for r in subset])))
    cells = Counter((r['source'], r['generator'], r['FILE_FAKE'], r['split']) for r in rows)
    weak = [{'source':s,'generator':g,'label':l,'split':p,'rows':n} for (s,g,l,p),n in sorted(cells.items()) if p != 'train' and n < 10]
    eligible = [r for r in ledger if r['admission'] == 'admitted']
    available_groups = defaultdict(set)
    for row in eligible:
        available_groups[row['source'], row['generator'], row['FILE_FAKE']].add(row['component_id'])
    missing = [dict(source=s, generator=g, label=l, split=p, rows=0,
                    available_groups=len(groups), reason=('intentional_training_only' if s=='CtrSVDD' and p!='train'
                        and report.get('policy',{}).get('singing_train_only') else
                        'fewer_than_three_groups' if len(groups)<3 else 'selection_gap'))
               for (s,g,l), groups in sorted(available_groups.items()) for p in SPLITS if not cells[s,g,l,p]]
    result = dict(parent_derivative_cross_split_rejected=True, unknown_component_target_rejected=True,
                  emitted_partition_membership_matches=True, frozen_manifest_hash_matches=True,
                  counts={s:dict(Counter(r['route'] + '_' + r['FILE_FAKE'] for r in rows if r['split']==s)) for s in SPLITS},
                  new_selected=dict(Counter(r['generator'] for r in rows if r['new_collection']=='True')),
                  new_selected_by_split={s:dict(Counter(r['generator'] for r in rows if r['split']==s and r['new_collection']=='True')) for s in SPLITS},
                  unknown_labels={s:{f:sum(r[f+'_mask']=='0' for r in rows if r['split']==s) for f in LABELS} for s in SPLITS},
                  weak_evaluation_cells=weak, metadata_shortcut_diagnostics=diagnostics,
                  missing_source_generator_cells=missing,
                  all_source_generator_cells_present=not missing,
                  diagnostics_are_not_model_performance=True, test_covariates_inspected=True,
                  independent_test_claim=False)
    (OUTPUT / 'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output.resolve()
    main()
