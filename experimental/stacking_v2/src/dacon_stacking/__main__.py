"""python -m dacon_stacking --help. Local-only entrypoints; no remote service APIs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .common import ContractError,NotRunError,read_json,write_json,sha256,check_hash


def main():
    parser=argparse.ArgumentParser(description='Current frozen-expert File-only stacking; local execution only')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('audit');p.add_argument('--input-root',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('smoke');p.add_argument('--out',required=True)
    p=sub.add_parser('validate-data');p.add_argument('--manifest',required=True);p.add_argument('--input-root',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('split-parents');p.add_argument('--parents',required=True);p.add_argument('--reservations',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('extract');p.add_argument('--manifest',required=True);p.add_argument('--input-root',required=True);p.add_argument('--source-root',required=True);p.add_argument('--out',required=True);p.add_argument('--raw-comparison',action='store_true')
    p=sub.add_parser('dev')
    for name in ('manifest','index','config','input-root','out'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('lock')
    for name in ('manifest','index','config','input-root','dev-report','kind','features','ood-manifest-sha256','ood-index-sha256','out'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('ood')
    for name in ('manifest','index','input-root','lock','lock-sha256','out'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('infer-cache')
    for name in ('cache','sidecar-sha256','identity','bundle','bundle-sha256','out'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('infer-audio')
    for name in ('audio','id','source-root','lock','lock-sha256','out'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('stage-runtime');p.add_argument('--source-root',required=True);p.add_argument('--return-root',required=True)
    args=parser.parse_args()
    try:
        if args.command=='audit':
            from .audit import audit_input
            report=audit_input(args.input_root,args.out)
            print(json.dumps({'verified_members':report['manifest_members_verified'],'real_cache_files':len(report['caches']),'max_abs_reference_error':max(report['max_reference_abs_errors_by_probability'].values()),'new_audio_acquired':0},indent=2))
        elif args.command=='smoke':
            from .demo import run_smoke
            print(json.dumps(run_smoke(args.out),indent=2))
        elif args.command=='validate-data':
            from .workflow import read_rows
            from .provenance import validate_manifest,registry_from_input,validate_ood_domains
            rows=read_rows(args.manifest);report=validate_manifest(rows,registry_from_input(args.input_root))
            dev=[r for r in rows if r['split']=='meta_dev'];ood=[r for r in rows if r['split']=='locked_ood']
            if dev and ood: validate_ood_domains(dev,ood)
            write_json(args.out,report);print(json.dumps(report,indent=2))
        elif args.command=='split-parents':
            from .provenance import assign_parent_splits
            reservations=read_json(args.reservations)
            result=assign_parent_splits(read_json(args.parents),reservations['reserved_sources'],reservations['reserved_generators'])
            write_json(args.out,result);print(json.dumps({'originals':len(result),'output':args.out}))
        elif args.command=='extract':
            from .workflow import read_rows
            from .provenance import validate_manifest,registry_from_input
            from .gpu import extract_audio
            rows=read_rows(args.manifest);validate_manifest(rows,registry_from_input(args.input_root))
            print(json.dumps({'files':len(extract_audio(rows,args.source_root,args.out,args.raw_comparison)['items'])}))
        elif args.command=='dev':
            from .workflow import run_dev
            report=run_dev(args.manifest,args.index,args.config,args.input_root,args.out)
            print(json.dumps({k:v['status'] for k,v in report['systems'].items()},indent=2))
        elif args.command=='lock':
            from .workflow import lock_candidate
            print(json.dumps(lock_candidate(args.manifest,args.index,args.config,args.input_root,args.dev_report,args.kind,args.features,args.ood_manifest_sha256,args.ood_index_sha256,args.out),indent=2))
        elif args.command=='ood':
            from .workflow import evaluate_ood
            result=evaluate_ood(args.manifest,args.index,args.input_root,args.lock,args.lock_sha256,args.out)
            print(json.dumps({'candidate_metrics':result['candidate']['official'],'paired_bootstrap':result['paired_bootstrap']},indent=2))
        elif args.command=='infer-cache':
            from .features import read_extended
            from .models import load_bundle
            from .workflow import infer_record,write_csv
            record=read_extended(args.cache,args.sidecar_sha256,args.identity)
            model=load_bundle(args.bundle,args.bundle_sha256)
            write_csv(args.out,[infer_record(record,model)]);print(json.dumps({'output':args.out,'files':1}))
        elif args.command=='infer-audio':
            from .gpu import load_local_engine,load_deployment,predict_engine_file
            from .workflow import write_csv
            model,contract=load_deployment(args.lock,args.lock_sha256,args.source_root)
            engine=load_local_engine(args.source_root)
            write_csv(args.out,[predict_engine_file(engine,Path(args.audio),args.id,model,contract)]);print(json.dumps({'output':args.out,'files':1}))
        elif args.command=='stage-runtime':
            from .stage import stage_runtime
            print(json.dumps(stage_runtime(args.source_root,args.return_root),indent=2))
    except NotRunError as exc:
        print(json.dumps({'status':'NOT_RUN','reason':str(exc)}),file=sys.stderr);return 3
    except (ContractError,ValueError,FileNotFoundError,FileExistsError,KeyError) as exc:
        print(json.dumps({'status':'FAILED_CONTRACT','reason':str(exc)}),file=sys.stderr);return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
