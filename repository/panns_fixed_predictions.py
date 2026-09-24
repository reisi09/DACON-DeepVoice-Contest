"""Score immutable DataV4 fake heads once, without changing their weights."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from panns_adaptation_data import sha256, validate_config, write_json, identity
from panns_adaptation_cache import load_demucs


def fixed_predictions(config, device):
    import torch
    package = validate_config(config)
    reference = json.loads(Path(config['reference']).read_text())
    for name, expected in reference['members'].items():
        if sha256(package/name) != expected:
            raise ValueError('Frozen runtime asset changed: '+name)
    # Only the previously audited, hash-pinned local package is executable.
    runtime = package/'model/runtime'
    sys.path.insert(0,str(runtime))
    spec = importlib.util.spec_from_file_location('preserved_full_bank', runtime/'full_bank_submission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_base = module.baseline_module
    def safe_base(assets):
        base = original_base(assets)
        base.load_htdemucs_model = lambda: load_demucs(package)
        def verified_labels():
            target = Path.home()/'panns_data/class_labels_indices.csv'
            if sha256(target) != sha256(package/'model/assets/baseline/model/panns/class_labels_indices.csv'):
                raise ValueError('PANNs label table differs; use the prepared task HOME')
        base.prepare_panns_labels = verified_labels
        return base
    module.baseline_module = safe_base
    original_load = torch.load
    def restricted_load(*args, **kwargs):
        kwargs['weights_only'] = True
        return original_load(*args, **kwargs)
    torch.load = restricted_load
    try:
        engine = module.Engine(package/'model/assets',package/'model/trained',device=device)
    finally:
        torch.load = original_load
    panel = set(config['panel_ids'])
    rows = [r for r in map(json.loads,Path(config['rows']).read_text().splitlines()) if r['role'] in ('development','selection') and r['ID'] in panel]
    out = Path(config['output'])/'fixed_predictions'
    out.mkdir(parents=True,exist_ok=True)
    contract = dict(reference_sha256=sha256(config['reference']), rows_sha256=sha256(config['rows']),
                    producer_sha256=sha256(__file__), device=device, torch_version=str(torch.__version__))
    entries=[]
    with Path(config['archive']).open('rb') as archive:
        for row in rows:
            path = out/(hashlib.sha256(row['ID'].encode()).hexdigest()+'.json')
            key = identity(dict(contract=contract, ID=row['ID'], audio_sha256=row['verified_audio']['sha256']))
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['identity']!=key or saved['entry_sha256'] != identity(saved['entry']):
                    raise ValueError('Frozen prediction resume identity mismatch')
                entries.append(saved['entry'])
                continue
            audio=row['verified_audio']
            if 'local_path' in audio:
                source=Path(audio['local_path']).resolve()
                if not source.is_relative_to(Path(config['recovered_audio_root']).resolve()):
                    raise ValueError('Recovered audio outside allowed root')
                payload=source.read_bytes()
            else:
                archive.seek(audio['tar_offset']);payload=archive.read(audio['bytes'])
            if hashlib.sha256(payload).hexdigest()!=audio['sha256']:
                raise ValueError('Frozen prediction input hash mismatch')
            with tempfile.TemporaryDirectory(dir=out) as directory:
                source=Path(directory)/('input'+Path(row['path']).suffix)
                source.write_bytes(payload)
                prediction=engine.predict(source)
            entry=dict(ID=row['ID'],audio_sha256=audio['sha256'],**prediction)
            write_json(path,dict(identity=key,entry_sha256=identity(entry),entry=entry))
            entries.append(entry)
            print(json.dumps(dict(fixed_fake_predictions=len(entries),total=len(rows))),flush=True)
    write_json(config['fixed_predictions'],dict(**contract,entries=entries))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    fixed_predictions(json.loads(args.run.read_text()),args.device)
