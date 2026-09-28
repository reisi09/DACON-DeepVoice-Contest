"""Separate commands for dev, immutable candidate lock, and one-shot locked OOD."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from .common import ContractError, check_hash, digest, read_json, sha256, write_json, contained
from .features import read_extended, feature_frame, FEATURE_SETS
from .provenance import validate_manifest, registry_from_input, validate_ood_domains, lineage_tokens
from .models import save_bundle, load_bundle, predict_estimator
from .evaluation import evaluate_dev, fit_final_dev, metrics_report, baseline_predictions, replace_file, paired_bootstrap, condition_reports
from ._snapshot_scoring import SUBMISSION_COLUMNS, validate_predictions


def read_rows(path):
    with Path(path).open(encoding='utf8') as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    return rows


def load_records(index_path,rows):
    index_path=Path(index_path);index=read_json(index_path)
    if index.get('schema')!='dacon.stacking.cache_index.v1': raise ContractError('Unknown cache index schema')
    items=index['items']
    if set(items)!={r['ID'] for r in rows}: raise ContractError('Cache index must match exact requested ID set')
    output=[]
    for row in rows:
        item=items[row['ID']]
        record=read_extended(contained(index_path.parent,item['path']),item['sidecar_sha256'],item['identity'])
        if record['ID']!=row['ID']: raise ContractError('Cache ID mismatch')
        if record['provenance']['audio_sha256']!=row['audio_sha256']: raise ContractError('Audio hash differs from feature cache')
        if record['provenance']['base_exposure']!=row['base_exposure']: raise ContractError('Cache and manifest exposure provenance differ')
        output.append(record)
    # A single experiment cannot combine mixed model revisions/preprocessing contracts.
    identities={digest({k:r['provenance'][k] for k in ('code_sha256','checkpoint_sha256','numeric_mode','feature_schema')}) for r in output}
    if len(identities)!=1: raise ContractError('Mixed encoder/head/code/numeric identities in one experiment')
    return output


def run_dev(manifest,index,config_path,input_root,out):
    rows=read_rows(manifest);registry=registry_from_input(input_root)
    audit=validate_manifest(rows,registry,'meta_dev');records=load_records(index,rows);config=read_json(config_path)
    report,predictions=evaluate_dev(rows,records,config)
    report.update(data_audit=audit,input_manifest_sha256=sha256(manifest),cache_index_sha256=sha256(index),config_sha256=sha256(config_path))
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise ContractError('Refuse to overwrite dev evidence')
    out.mkdir(parents=True,exist_ok=True);write_json(out/'dev_report.json',report)
    for name,p in predictions.items():write_csv(out/(name.replace('/','_')+'.csv'),p)
    return report


def lock_candidate(manifest,index,config_path,input_root,dev_report_path,kind,features,ood_manifest_sha256,ood_cache_index_sha256,out):
    """OOD supplied ONLY as preregistered digests; no OOD file/labels are opened."""
    for h in (ood_manifest_sha256,ood_cache_index_sha256):
        if len(h)!=64 or any(c not in '0123456789abcdef' for c in h): raise ContractError('Require exact preregistered OOD SHA-256 digests')
    rows=read_rows(manifest);validate_manifest(rows,registry_from_input(input_root),'meta_dev')
    records=load_records(index,rows);config=read_json(config_path);dev_report=read_json(dev_report_path)
    if dev_report.get('input_manifest_sha256')!=sha256(manifest) or dev_report.get('cache_index_sha256')!=sha256(index) or dev_report.get('config_sha256')!=sha256(config_path):
        raise ContractError('Dev report inputs/config do not match final fit')
    if dev_report['systems'].get(features+'/'+kind,{}).get('status')!='OK': raise ContractError('Cannot lock an unevaluated system')
    out=Path(out)
    if out.exists(): raise ContractError('Candidate directory already exists; no overwrite')
    model,audit=fit_final_dev(rows,records,kind,features,config)
    provenance={'purpose':'production_candidate','dev_manifest_sha256':sha256(manifest),'config_sha256':sha256(config_path),'dev_report_sha256':sha256(dev_report_path),'audit':audit}
    bundle_hash=save_bundle(out/'bundle',model,provenance)
    lock={'schema':'dacon.stacking.lock.v1','purpose':'production_candidate','bundle_manifest_sha256':bundle_hash,'candidate':kind,'feature_set':features,
          'dev_manifest_sha256':sha256(manifest),'dev_cache_index_sha256':sha256(index),'ood_manifest_sha256':ood_manifest_sha256,'ood_cache_index_sha256':ood_cache_index_sha256,
          'dev_report_sha256':sha256(dev_report_path),'config_sha256':sha256(config_path),
          'feature_extraction_identity':{k:records[0]['provenance'][k] for k in ('code_sha256','checkpoint_sha256','numeric_mode','feature_schema')},
          'dev_lineage_tokens':sorted(set().union(*(lineage_tokens(r) for r in rows))),
          'dev_domains':{key:sorted({p[key] for r in rows for p in r['parents'] if p[key]!='real'}) for key in ('source','generator_family')},
          'base_registry_digest':digest(sorted(registry_from_input(input_root))),
          'bootstrap':config['bootstrap'],'seed':config['seed'],'threshold_selection':'none'}
    write_json(out/'candidate_lock.json',lock)
    return {'lock_sha256':sha256(out/'candidate_lock.json'),'bundle_manifest_sha256':bundle_hash}


def verify_ood_lock(path,expected_lock_sha256,ood_manifest_sha256):
    if not Path(path).exists(): raise ContractError('Candidate lock is required before locked OOD evaluation')
    check_hash(path,expected_lock_sha256);lock=read_json(path)
    if lock.get('schema')!='dacon.stacking.lock.v1' or lock.get('purpose')!='production_candidate': raise ContractError('Only a production candidate lock can open OOD')
    if lock['ood_manifest_sha256']!=ood_manifest_sha256: raise ContractError('locked OOD manifest changed')
    return lock


def evaluate_ood(manifest,index,input_root,lock_path,lock_sha256,out):
    """No fit/scaler/calibration/threshold call; fail if an evaluation receipt exists."""
    lock_path=Path(lock_path);lock=verify_ood_lock(lock_path,lock_sha256,sha256(manifest))
    if sha256(index)!=lock['ood_cache_index_sha256']: raise ContractError('locked OOD cache index changed')
    marker=lock_path.parent/'OOD_OPENED.json'
    # Exclusive creation is a process guard, not protection against a malicious user deleting it.
    with marker.open('x',encoding='utf8') as stream:
        json.dump({'lock_sha256':lock_sha256,'status':'OPENED; failures are retained and do not authorize tuning'},stream)
    rows=read_rows(manifest);registry=registry_from_input(input_root)
    if digest(sorted(registry))!=lock['base_registry_digest']: raise ContractError('Base exposure registry changed')
    audit=validate_manifest(rows,registry|set(lock['dev_lineage_tokens']),'locked_ood')
    for key,values in lock['dev_domains'].items():
        if {p[key] for r in rows for p in r['parents'] if p[key]!='real'}&set(values):
            raise ContractError(f'OOD {key} overlaps dev')
    records=load_records(index,rows)
    for r in records:
        if {k:r['provenance'][k] for k in lock['feature_extraction_identity']}!=lock['feature_extraction_identity']:
            raise ContractError('OOD feature extraction differs from locked dev extraction')
    model=load_bundle(lock_path.parent/'bundle',lock['bundle_manifest_sha256'])
    baseline=baseline_predictions(rows,records);p=predict_estimator(model,feature_frame(records,lock['feature_set']))
    candidate=replace_file(baseline,p)
    result={'data_audit':audit,'baseline':metrics_report(rows,baseline),'candidate':metrics_report(rows,candidate),
            'paired_bootstrap':paired_bootstrap(rows,baseline,candidate,lock['bootstrap'],lock['seed']),
            'conditions':condition_reports(rows,baseline,candidate),'lock_sha256':lock_sha256,'note':'One prelocked evaluation; do not revise the candidate based on this result.'}
    out=Path(out)
    if out.exists(): raise ContractError('Refuse to overwrite OOD evidence')
    out.mkdir(parents=True);write_json(out/'ood_report.json',result);write_csv(out/'predictions.csv',candidate)
    write_json(marker,{'lock_sha256':lock_sha256,'status':'COMPLETE','result_sha256':sha256(out/'ood_report.json')})
    return result


def infer_record(record,model):
    """One file in, one file out. No labels, cohort statistics, or other records."""
    import pandas as pd
    missing=set(model['order'])-record['features'].keys()
    if missing: raise ContractError(f'Required inference features absent: {sorted(missing)}')
    x=pd.DataFrame([[record['features'][n] for n in model['order']]],columns=model['order'])
    p=float(predict_estimator(model,x)[0])
    return replace_file([{'ID':record['ID'],**record['baseline']}],[p])[0]


def write_csv(path,predictions):
    validate_predictions(predictions,[r['ID'] for r in predictions])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=SUBMISSION_COLUMNS);writer.writeheader();writer.writerows(predictions)
