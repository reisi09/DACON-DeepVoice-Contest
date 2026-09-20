# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Check real frozen manifests and known split-policy regressions on CPU.

Run: .venv/bin/python verify_existing_bank.py --output results/existing_only_20260919
"""
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

from rebuild_existing_bank import INPUT, verify_policy
from reconstruct_candidate_bank import SPLITS, sha, SplitError


def read(path: Path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main(output: Path) -> None:
    """Reject real prior dominance, cross-split parent leakage and unmasked unknowns."""
    selected,ledger = read(output/'selected.csv'),read(output/'combined_catalog.csv')
    report=json.loads((output/'verification.json').read_text())
    accounting=report['split_class_counts']
    verify_policy(selected,ledger,accounting)
    provenance=json.loads((output/'provenance.json').read_text())
    for name,digest in provenance['csv_sha256'].items():
        if sha(output/name)!=digest: raise SplitError('Frozen CSV hash changed: '+name)
    for split in SPLITS:
        expected=[r for r in selected if r['split']==split]
        if read(output/(split+'.csv'))!=expected: raise SplitError('Partition membership mismatch')
        if split!='train':
            for flag,suffix in [('True','_historical.csv'),('False','_no_recorded_exposure.csv')]:
                if read(output/(split+suffix))!=[r for r in expected if r['historical_development_exposure']==flag]:
                    raise SplitError('Exposure ledger membership mismatch')
    old_report=json.loads((INPUT/'verification.json').read_text())
    try:
        verify_policy(read(INPUT/'selected.csv'),read(INPUT/'combined_catalog.csv'),old_report['split_class_counts'])
    except SplitError:
        pass
    else:
        raise AssertionError('Prior singing-dominated evaluation split was accepted')
    previous=INPUT.parent/'existing_only_20260919'
    if previous.exists():
        prior=json.loads((previous/'verification.json').read_text())
        try:
            verify_policy(read(previous/'selected.csv'),read(previous/'combined_catalog.csv'),prior['split_class_counts'])
        except SplitError as error:
            if 'Validation/test subtype mix' not in str(error): raise
        else:
            raise AssertionError('Known validation/test subtype drift was accepted')
    confounded=INPUT.parent/'existing_only_20260919_balanced_eval'
    if confounded.exists():
        prior=json.loads((confounded/'verification.json').read_text())
        try:
            verify_policy(read(confounded/'selected.csv'),read(confounded/'combined_catalog.csv'),prior['split_class_counts'])
        except SplitError as error:
            if 'Speech evaluation corpus reveals class' not in str(error): raise
        else:
            raise AssertionError('Corpus-confounded speech evaluation was accepted')
    damaged=deepcopy(selected)
    training=next(r for r in damaged if r['split']=='train')
    validation=next(r for r in damaged if r['split']=='val')
    validation['group_id']+='|id:'+training['ID']
    try:
        verify_policy(damaged,ledger,accounting)
    except SplitError as error:
        if 'Cross-partition' not in str(error): raise
    else:
        raise AssertionError('Parent leakage accepted')
    damaged=deepcopy(selected)
    unknown=next(r for r in damaged if r['VOICE_FAKE']=='')
    unknown['VOICE_FAKE_mask']='1'
    try:
        verify_policy(damaged,ledger,accounting)
    except SplitError as error:
        if 'Unknown label unmasked' not in str(error): raise
    else:
        raise AssertionError('Unknown component label accepted')
    result=dict(real_policy_pass=True,csv_hashes_match=True,partitions_match=True,exposure_subsets_match=True,
                prior_dominance_rejected=True,prior_evaluation_mix_drift_rejected=previous.exists(),
                prior_speech_corpus_confound_rejected=confounded.exists(),
                injected_parent_leak_rejected=True,unknown_mask_rejected=True)
    (output/'policy_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
