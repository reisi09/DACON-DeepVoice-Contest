"""Replay verified metrics and diagnose partial-fake probes against their real parents.

Run: .venv/bin/python analyze_six_h100.py
"""
import csv
import json
from pathlib import Path

from redesign_evaluation_common import components
from six_h100_experiments import metrics


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root/'results/six_h100_20260917'
    data = output/'downloaded/unpacked/output'
    verification = json.loads((output/'download_verification.json').read_text())
    assert verification['status']=='exported' and verification['files_verified']==768
    rows = list(csv.DictReader((output/'probes.csv').open()))
    predictions = {p.stem.removeprefix('predictions_'):json.loads(p.read_text())
                   for p in data.glob('predictions_*.json')}
    original = json.loads((data/'experiment_6.json').read_text())
    replayed = 0
    for split in ('dev','test'):
        for variant, candidates in original[split].items():
            if variant=='matched_lengths':
                continue
            subset = [r for r in rows if r['split']==split and r['variant']==variant]
            for candidate, expected in candidates.items():
                assert metrics(subset,predictions[candidate])==expected, (split,variant,candidate)
                replayed += 1
    analyses = {}
    for variant in ('partial_voice','partial_music'):
        fake = [r for r in rows if r['split']=='test' and r['variant']==variant]
        parents = {json.loads(r['parent_ids'])[1] for r in fake}
        real = [r for r in rows if r['split']=='test' and r['variant']=='base' and r['parent_id'] in parents]
        assert len(real)==len(parents) and all(r['FILE_FAKE']=='0' for r in real)
        joined = fake+real
        connected = components({'groups':{r['ID']:r['group_id'].split('|') for r in joined}})
        analyses[variant] = dict(post_hoc=True, real_controls='same real background parents, unique IDs',
                                fake_files=len(fake), real_files=len(real),
                                independent_connected_groups=len(set(connected.values())),
                                metrics={name:metrics(joined,values) for name,values in predictions.items()})
    new_data = original['dev']['base']
    tie = new_data['hard_real_seed41']['metrics']==new_data['mixture_seed41']['metrics']
    assert tie
    mixing = []
    for row in rows:
        if row['split']=='test' and row['variant']=='speech_music':
            parent = json.loads(row['parent_ids'])[0]+'_base'
            pure, mixed = predictions['B_full'][parent], predictions['B_full'][row['ID']]
            mixing.append(dict(ID=row['ID'], parent=parent, file_label=row['FILE_FAKE'],
                               pure_voice_fake=pure[1], mixed_voice_fake=mixed[1],
                               mixed_voice_presence=mixed[3], mixed_music_fake=mixed[2]))
    report = dict(metric_reports_replayed_exactly=replayed, matched_parent_partial_probes=analyses,
                  speech_mixing_evidence=mixing,
                  dev_selection_tie=True, actual_tie_break='max((gain,name)) chose mixture lexicographically',
                  mixture_superiority_proven_by_selection=False, official_score=None,
                  promote_candidate=False, additional_gpu_training_authorized_by_this_analysis=False)
    (output/'analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'replayed':replayed,'dev_tie':tie,'partial_probes':{
        variant:{'groups':values['independent_connected_groups'],
                 'file_eer':{name:report['metrics']['file_eer'] for name,report in values['metrics'].items()
                             if name in ('M330','B_full','mixture_seed41','selected_ensemble')}}
        for variant,values in analyses.items()}}))


if __name__=='__main__':
    main()
