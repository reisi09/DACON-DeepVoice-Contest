# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Rebalance already audited originals without downloads or GPU execution.

Run: .venv/bin/python rebuild_existing_bank.py --output results/existing_only_20260919
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from balanced_bank_split import check_audio
from expanded_data import Row, SplitError
from full_bank_split import write_csv
from reconstruct_candidate_bank import ROOT, SPLITS, RATIOS, rank, sha, verify

INPUT = ROOT / 'results/reconstructed_bank_20260919_ctrsvdd_dev'
Cell = tuple[str, str, str, str]


def subtype(row: Row) -> str:
    return 'singing' if row['source'] == 'CtrSVDD' else ('speech' if row['route'] == 'voice' else 'music')


def cell(row: Row) -> Cell:
    return subtype(row), row['FILE_FAKE'], row['source'], row['generator']


def select(rows: list[Row]):
    """Maximize retained rows under hard balance/coverage gates, then minimize exposure."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['component_id']].append(row)
    patterns = defaultdict(list)
    for group, members in groups.items():
        signature = tuple(sorted(Counter(cell(r) for r in members).items()))
        exposed = any(r['historical_development_exposure'] == 'True' for r in members)
        patterns[signature, exposed].append(group)
    ordered = sorted(patterns)
    edges = [(p, c, n) for p, (pattern, _) in enumerate(ordered) for c, n in pattern]
    cells = sorted({cell(r) for r in rows})
    capacity = Counter(cell(r) for r in rows)
    P, E, C = len(ordered), len(edges), len(cells)
    total_var, quota_start = 3 * (P + E), 3 * (P + E) + 1
    slack_start = quota_start + 3
    size = slack_start + 6 * C
    upper, integer = np.full(size, np.inf), np.ones(size)
    integer[slack_start:] = 0
    rr, cc, vv, low, high = [], [], [], [], []
    def constraint(terms, lower, higher):
        for column, value in terms:
            rr.append(len(low)); cc.append(column); vv.append(value)
        low.append(lower); high.append(higher)
    def gv(s, p): return s * P + p
    def rv(s, e): return 3 * P + s * E + e
    def terms(s, predicate):
        return [(rv(s,e), 1) for e, (_,c,_) in enumerate(edges) if predicate(c)]
    for p, (pattern, _) in enumerate(ordered):
        count = len(patterns[ordered[p]])
        for s in range(3):
            upper[gv(s,p)] = 0 if s and any(c[0]=='singing' for c,n in pattern) else count
        constraint([(gv(s,p),1) for s in range(3)], count, count)
    for e,(p,c,n) in enumerate(edges):
        for s in range(3):
            constraint([(rv(s,e),1),(gv(s,p),-n)], -np.inf, 0)
    constraint([(quota_start+s,1) for s in range(3)]+[(total_var,-1)],0,0)
    cost = np.zeros(size)
    for s in range(3):
        constraint([(quota_start+s,1),(total_var,-RATIOS[s])],-.5,.5)
        for label in ('0','1'):
            constraint(terms(s,lambda c:c[1]==label)+[(quota_start+s,-1)],0,0)
        for kind in ('speech','singing','music'):
            constraint([(i,1) for i,_ in terms(s,lambda c:c[:2]==(kind,'0'))]+
                       [(i,-1) for i,_ in terms(s,lambda c:c[:2]==(kind,'1'))],0,0)
            if kind != 'singing':
                constraint(terms(s,lambda c:c[:2]==(kind,'1')),1,np.inf)
        if s:
            # Shared-corpus real controls prevent corpus identity from revealing speech truth.
            for source in sorted({c[2] for c in cells if c[0]=='speech'}):
                constraint(terms(s,lambda c:c[0]=='speech' and c[1]=='0' and c[2]==source)+
                           [(i,-1) for i,_ in terms(s,lambda c:c[0]=='speech' and c[1]=='1' and c[2]==source)],0,0)
        for k,current in enumerate(cells):
            chosen = terms(s,lambda c:c==current)
            if not s:
                constraint(chosen,min(20,capacity[current]),np.inf)
            denominator = sum(n for c,n in capacity.items() if c[:2]==current[:2])
            fraction = capacity[current] / denominator
            target = [(i,-fraction) for i,_ in terms(s,lambda c:c[:2]==current[:2])]
            plus = slack_start + 2*(s*C+k)
            constraint(chosen+target+[(plus,-1),(plus+1,1)],0,0)
            cost[plus:plus+2] = 1/max(20,capacity[current])
    speech_terms = [item for s in range(3) for item in terms(s,lambda c:c[:2]==('speech','1'))]
    for kind in ('speech','music'):
        constraint(terms(1,lambda c:c[:2]==(kind,'1'))+
                   [(i,-1) for i,_ in terms(2,lambda c:c[:2]==(kind,'1'))],-1,1)
    constraint(terms(0,lambda c:c[:2]==('singing','1'))+
               [(i,-1) for i,_ in terms(0,lambda c:c[:2]==('speech','1'))],-np.inf,0)
    def solve(objective):
        matrix = coo_matrix((vv,(rr,cc)),shape=(len(low),size)).tocsc()
        result = milp(objective,integrality=integer,bounds=Bounds(np.zeros(size),upper),
                      constraints=LinearConstraint(matrix,low,high),options={'time_limit':120,'mip_rel_gap':0})
        if result.x is None:
            raise SplitError('Existing-only policy infeasible: '+result.message)
        return result
    maximum = np.zeros(size); maximum[total_var] = -1
    first = solve(maximum)
    per_class = int(round(first.x[total_var]))
    constraint([(total_var,1)],per_class,per_class)
    retain_speech = np.zeros(size)
    for i,_ in speech_terms: retain_speech[i]=-1
    retention = solve(retain_speech)
    speech_count = int(round(sum(retention.x[i] for i,_ in speech_terms)))
    constraint(speech_terms,speech_count,speech_count)
    for e,(p,c,n) in enumerate(edges):
        if ordered[p][1]:
            for s in (1,2): cost[rv(s,e)] = 100000
    second = solve(cost)
    counts = np.rint(second.x[:slack_start]).astype(int)
    selected, assignments = [], {}
    for p, key in enumerate(ordered):
        available = sorted(patterns[key],key=rank)
        offset = 0
        for s,split in enumerate(SPLITS):
            chosen = available[offset:offset+counts[gv(s,p)]]; offset += len(chosen)
            assignments.update({g:split for g in chosen})
            for e,(pi,current,n) in enumerate(edges):
                if pi != p: continue
                candidates = sorted((r for g in chosen for r in groups[g] if cell(r)==current),key=lambda r:rank(r['ID']))
                selected.extend(dict(r,split=split) for r in candidates[:counts[rv(s,e)]])
    accounting = dict(per_class=per_class,speech_fake_selected=speech_count,
                      per_split_per_class={s:int(counts[quota_start+i]) for i,s in enumerate(SPLITS)})
    return sorted(selected,key=lambda r:r['ID']),assignments,accounting,dict(
        maximum_rows_proven=first.status==0,first_message=first.message,first_gap=float(first.mip_gap),
        speech_retention_proven=retention.status==0,speech_retention_gap=float(retention.mip_gap),
        second_message=second.message,second_gap=float(second.mip_gap))


def verify_policy(selected: list[Row], ledger: list[Row], accounting) -> None:
    """Reject subtype dominance, unlearned evaluated generators, or singing evaluation."""
    verify(selected,ledger,accounting)
    counts = Counter((r['split'],subtype(r),r['FILE_FAKE']) for r in selected)
    for split in SPLITS:
        for kind in ('speech','singing','music'):
            if counts[split,kind,'0'] != counts[split,kind,'1']:
                raise SplitError('Subtype class imbalance')
    if counts['train','singing','1'] > counts['train','speech','1']:
        raise SplitError('Training singing dominates speech')
    if any(subtype(r)=='singing' and r['split']!='train' for r in selected):
        raise SplitError('Unresolved singing content used for evaluation')
    available = Counter(cell(r) for r in ledger if r['admission']=='admitted')
    training = Counter(cell(r) for r in selected if r['split']=='train')
    if any(training[c] < min(20,n) for c,n in available.items()):
        raise SplitError('Training source/generator coverage missing')
    for kind in ('speech','music'):
        if abs(counts['val',kind,'1']-counts['test',kind,'1'])>1:
            raise SplitError('Validation/test subtype mix mismatch')
    speech_sources = Counter((r['split'],r['source'],r['FILE_FAKE']) for r in selected if subtype(r)=='speech')
    for split,source,label in speech_sources:
        if split!='train' and speech_sources[split,source,'0']!=speech_sources[split,source,'1']:
            raise SplitError('Speech evaluation corpus reveals class')
    actual = sum(cell(r)[:2]==('speech','1') for r in selected)
    if actual != accounting.get('speech_fake_selected',actual):
        raise SplitError('Optimized speech retention count changed')
    for s,n in accounting['per_split_per_class'].items():
        if abs(n-accounting['per_class']*RATIOS[SPLITS.index(s)]) > .50001:
            raise SplitError('70/15/15 rounding tolerance exceeded')


def main(output: Path) -> None:
    if output.exists(): raise SplitError('Choose a new output directory; snapshots are immutable')
    provenance = json.loads((INPUT/'provenance.json').read_text())
    source = INPUT/'combined_catalog.csv'
    if sha(source) != provenance['csv_sha256'][source.name]: raise SplitError('Input ledger changed')
    with source.open(newline='') as stream: ledger = list(csv.DictReader(stream))
    eligible = [r for r in ledger if r['admission']=='admitted']
    selected,assignments,accounting,solver = select(eligible)
    verify_policy(selected,ledger,accounting)
    integrity = check_audio(selected)
    output.mkdir(parents=True)
    chosen = {r['ID']:r['split'] for r in selected}
    for row in ledger:
        row.update(split=chosen.get(row['ID'],''),selected=str(row['ID'] in chosen),
                   assigned_group_split=assignments.get(row['component_id'],''),subtype=subtype(row))
        if row['admission']=='admitted': row['reason']='' if row['ID'] in chosen else 'existing_only_balance_reserve'
    write_csv(output/'combined_catalog.csv',ledger)
    write_csv(output/'selected.csv',selected)
    for split in SPLITS:
        subset = [r for r in selected if r['split']==split]
        write_csv(output/(split+'.csv'),subset)
        if split!='train':
            for exposed in (False,True):
                write_csv(output/(split+('_historical.csv' if exposed else '_no_recorded_exposure.csv')),
                          [r for r in subset if (r['historical_development_exposure']=='True')==exposed])
    composition = Counter((r['split'],*cell(r)) for r in selected)
    write_csv(output/'composition.csv',[dict(split=s,subtype=t,label=l,source=o,generator=g,rows=n) for (s,t,l,o,g),n in sorted(composition.items())])
    report = dict(status='existing_only_reconstruction',training_ready=False,gpu_started=False,new_downloads=False,
        input_catalog_sha256=sha(source),selected_sha256=sha(output/'selected.csv'),selected_rows=len(selected),
        combined_rows=len(ledger),admitted_pool_rows=len(eligible),split_class_counts=accounting,solver=solver,
        selected_audio_integrity=integrity,independent_test_claim=False,
        historical_exposure_by_split={s:sum(r['historical_development_exposure']=='True' for r in selected if r['split']==s) for s in SPLITS},
        policy={'singing_train_only':True,'fake_speech_retained':accounting['speech_fake_selected'],
                'fake_speech_available':sum(cell(r)[:2]==('speech','1') for r in eligible),
                'validation_test_subtype_difference_at_most_one':True,'subtype_class_balance_per_split':True,
                'speech_evaluation_real_fake_matched_by_corpus':True,
                'singing_not_above_train_speech':True,'minimum_training_cell':20},
        remaining_gates=['singing_generalization_not_evaluated','historical_speech_evaluation_not_independent',
            'source_duration_bias','loader_notebook_integration','A100_runtime_VRAM_not_measured'])
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(csv_sha256={p.name:sha(p) for p in output.glob('*.csv')},
        code_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
