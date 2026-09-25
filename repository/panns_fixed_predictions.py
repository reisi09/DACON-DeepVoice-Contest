"""Score immutable DataV4 fake heads once, without changing their weights."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import tempfile
from panns_adaptation_data import cache_provenance, sha256, validate_config, write_json, identity
from panns_adaptation_cache import NORMALIZED_WAV_DECODER, load_demucs, pin_normalized_wav_loader


def configure_numerics(device: str) -> None:
    import torch
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested frozen prediction device unavailable')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def fixed_prediction_contract(config, device):
    import torch
    runtime = cache_provenance()
    runtime['code']['panns_fixed_predictions.py'] = sha256(__file__)
    runtime['versions'].update({name: version(name) for name in (
        'transformers', 'safetensors', 'scipy', 'einops', 'huggingface-hub')})
    runtime.update(python=platform.python_version(), system=platform.system(), architecture=platform.machine(),
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    hardware = dict(device=device)
    if device == 'cuda':
        hardware.update(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    return dict(config_identity=config['identity'], reference_sha256=sha256(config['reference']),
        rows_sha256=sha256(config['rows']), producer_sha256=sha256(__file__), device=device,
        runtime=runtime, hardware=hardware, preprocessing='preserved Engine gain .6, soxr_hq, Demucs shifts=0',
        demucs_input_decoder=NORMALIZED_WAV_DECODER,
        numerics=dict(threads=torch.get_num_threads(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            fake_head_autocast='bfloat16' if device == 'cuda' else 'disabled'))


def load_fixed_predictions(config, device):
    configure_numerics(device)
    saved = json.loads(Path(config['fixed_predictions']).read_text())
    actual = {key: value for key, value in saved.items() if key not in ('entries', 'entries_sha256')}
    if actual != fixed_prediction_contract(config, device):
        raise ValueError('Frozen fake predictions have incompatible provenance')
    if saved.get('entries_sha256') != identity(saved['entries']):
        raise ValueError('Frozen prediction payload hash mismatch')
    from graph_scoring import validate_predictions
    panel = set(config['panel_ids'])
    expected_ids = [row['ID'] for row in map(json.loads, Path(config['rows']).read_text().splitlines())
                    if row['role'] in ('development', 'selection') and row['ID'] in panel]
    validate_predictions([{key: value for key, value in entry.items() if key != 'audio_sha256'}
                          for entry in saved['entries']], expected_ids)
    return saved


def fixed_predictions(config, device):
    import torch
    configure_numerics(device)
    package = validate_config(config)
    reference = json.loads(Path(config['reference']).read_text())
    for name, expected in reference['members'].items():
        if sha256(package/name) != expected:
            raise ValueError('Frozen runtime asset changed: '+name)
    # Only the previously audited, hash-pinned local package is executable.
    runtime = package/'model/runtime'
    for name, expected in reference['members'].items():
        if name.startswith('model/runtime/') and name.endswith('.py'):
            loaded = sys.modules.get(Path(name).stem)
            if loaded is not None and sha256(loaded.__file__) != expected:
                raise ValueError('Conflicting already-loaded frozen runtime module: '+name)
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
    module.pin_torchaudio_loader = pin_normalized_wav_loader
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
    contract = fixed_prediction_contract(config, device)
    entries=[]
    with Path(config['archive']).open('rb') as archive:
        for row in rows:
            path = out/(hashlib.sha256(row['ID'].encode()).hexdigest()+'.json')
            key = identity(dict(contract=contract, ID=row['ID'], audio_sha256=row['verified_audio']['sha256']))
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['identity']!=key or saved['entry_sha256'] != identity(saved['entry']):
                    raise ValueError('Frozen prediction resume identity mismatch')
                module.validate_predictions([{name:value for name,value in saved['entry'].items() if name!='audio_sha256'}], [row['ID']])
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
            with tempfile.TemporaryDirectory() as directory:
                source=Path(directory)/('input'+Path(row['path']).suffix)
                source.write_bytes(payload)
                prediction=engine.predict(source)
            entry=dict(ID=row['ID'],audio_sha256=audio['sha256'],**prediction)
            module.validate_predictions([{name:value for name,value in entry.items() if name!='audio_sha256'}], [row['ID']])
            write_json(path,dict(identity=key,entry_sha256=identity(entry),entry=entry))
            entries.append(entry)
            print(json.dumps(dict(fixed_fake_predictions=len(entries),total=len(rows))),flush=True)
    write_json(config['fixed_predictions'],dict(**contract,entries=entries,entries_sha256=identity(entries)))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    fixed_predictions(json.loads(args.run.read_text()),args.device)
