"""Package the existing C08 seed41 checkpoint, original baseline assets and frozen MERT."""

import hashlib
import json
from pathlib import Path
import zipfile

from music_temporal import load_temporal_head
from package_submission import _model_members
from v6_data import sha256, write_json


RUNTIME = ('c08_submission.py', 'music_temporal.py', 'temporal_augmentation.py', 'v7_ablation.py',
           'v6_core.py', 'v6_data.py', 'v6_train.py', 'v6_runtime.py', 'graph_scoring.py')
REQUIREMENTS = '''--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.7.1+cu128
torchaudio==2.7.1+cu128
torchvision==0.22.1+cu128
'''
ENTRYPOINT = '''import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'model/runtime'))
from c08_submission import main
if __name__ == '__main__':
    main(ROOT)
'''


def package(root: Path, output: Path, *, final_run: Path | None = None,
            mert_assets: Path | None = None) -> dict:
    """Hash the bytes actually written and verify the completed archive before promotion."""
    run = root/'results/temporal_h100_compact_20260915/unpacked/results'
    trial = final_run/'fold0' if final_run is not None else run/'C08_seed41'
    trace = json.loads((trial/'training.json').read_text())
    checkpoint = trial/'temporal_music.pt'
    if sha256(checkpoint) != trace['checkpoint_sha256']:
        raise ValueError('C08 checkpoint differs from its H100 training receipt')
    head = load_temporal_head(checkpoint)
    if final_run is not None:
        plan = json.loads((final_run/'full_data_plan.json').read_text())
        cv = json.loads((final_run/'training.json').read_text())
        approved = [r for r in plan['rows'] if r['music_training'] == '1']
        if (cv['training_mode'] != 'grouped_5fold_early_stopping'
                or cv['train_ids'] != [r['ID'] for r in approved]
                or len(approved) != 1206 or len(cv['folds']) != 5):
            raise ValueError('Expected the approved 1206-file early-stopped MERT330 MLP ensemble')
        for fold in cv['folds']:
            trained = final_run/f"fold{fold['fold']}"
            receipt = json.loads((trained/'training.json').read_text())
            model = load_temporal_head(trained/'temporal_music.pt')
            if (sha256(trained/'temporal_music.pt') != receipt['checkpoint_sha256']
                    or receipt['train_ids'] != fold['train_ids']
                    or receipt['validation_ids'] != fold['validation_ids']
                    or receipt['recipe']['nsynth_mass'] != .05
                    or receipt['recipe']['label_smoothing'] != 0 or receipt['recipe']['lr_plateau']
                    or model.architecture != 'mert-residual-64x3-max-d1024-mlp32-softmax-v1'):
                raise ValueError('Fold checkpoint or training recipe does not match the approved MLP')
        for row in approved:
            if (sum(row['ID'] in f['train_ids'] for f in cv['folds']) != 4
                    or sum(row['ID'] in f['validation_ids'] for f in cv['folds']) != 1):
                raise ValueError('Incomplete all-data cross-validation coverage')
        if mert_assets is None:
            raise ValueError('Full MLP packaging requires the pinned MERT330 assets')
    elif (head.architecture != 'mert-residual-64x3-max-v1'
            or trace['recipe']['seed'] != 41 or trace['parameters'] != 88257
            or trace['selected_epoch'] != 12
            or trace['checkpoint_sha256'] != 'a32d8b51337e35a3395e83bdcc0d367d28429e7159f015a307db4369559c64b3'
            or json.loads((run/'selection.json').read_text())['winner']['name'] != 'C08'):
        raise ValueError('Expected the tune-selected original C08 seed41 model')
    mert = mert_assets if mert_assets is not None else root/'_workspace/v6_assets/mert'
    if final_run is not None:
        meta = json.loads((final_run/'mert330_metadata.json').read_text())
        sealed = {'mert/'+name: digest for name, digest in meta['source_sha256'].items()}
        sealed['mert/pytorch_model.bin'] = meta['weight_sha256']
    else:
        sealed = json.loads((mert.parent/'assets.json').read_text())['sha256']
    members = {f'model/runtime/{name}': root/name for name in RUNTIME}
    members.update({f'model/assets/mert/{name}': mert/name for name in (
        'config.json','configuration_MERT.py','modeling_MERT.py','preprocessor_config.json',
        'pytorch_model.bin','README.md')})
    members['model/trained/c08.pt'] = checkpoint
    if final_run is not None:
        members.update({f'model/trained/fold{fold}.pt': final_run/f'fold{fold}'/'temporal_music.pt'
                        for fold in range(1,5)})
    for name, path in members.items():
        if name.startswith('model/assets/mert/') and sha256(path) != sealed['mert/'+path.name]:
            raise ValueError(f'MERT asset differs from the sealed training assets: {path.name}')
    choice = dict(candidate='M330_MLP_full1206' if final_run is not None else 'C08',
                  seed=trace['recipe']['seed'], selected_epoch=trace['selected_epoch'], recipe=trace['recipe'],
                  checkpoint_sha256=trace['checkpoint_sha256'], parameters=trace['parameters'],
                  full_data_retraining=False, ensemble=final_run is not None)
    if final_run is not None:
        choice.update(training_mode=cv['training_mode'], all_data_used_across_folds=True,
                      selected_epochs=[f['selected_epoch'] for f in cv['folds']],
                      checkpoints=['model/trained/c08.pt']+[f'model/trained/fold{i}.pt' for i in range(1,5)])
    hashes = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    with zipfile.ZipFile(root/'baseline_original/baseline_submit.zip') as original:
        baseline_members = _model_members(original)
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
            for member in baseline_members:
                if member.is_dir():
                    continue
                name = 'model/assets/baseline/'+member.filename
                digest = hashlib.sha256()
                with original.open(member) as incoming, archive.open(name, 'w', force_zip64=True) as outgoing:
                    while chunk := incoming.read(8*1024*1024):
                        digest.update(chunk)
                        outgoing.write(chunk)
                hashes[name] = digest.hexdigest()
            text = {'script.py': ENTRYPOINT.encode(), 'requirements.txt': REQUIREMENTS.encode(),
                    'model/assets/baseline/script.py': original.read('script.py'),
                    'model/trained/c08.json': json.dumps(choice, indent=2).encode()}
            for name, path in members.items():
                archive.write(path, name)
                hashes[name] = sha256(path)
            for name, content in text.items():
                archive.writestr(name, content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr('model/manifest.json', json.dumps(dict(choice=choice,sha256=hashes),indent=2))
    with zipfile.ZipFile(temporary) as archive:
        expanded = sum(m.file_size for m in archive.infolist())
        if temporary.stat().st_size > 10_000_000_000 or expanded > 32_000_000_000:
            raise ValueError('Archive exceeds competition size limits')
        if archive.testzip() is not None:
            raise ValueError('Submission archive CRC verification failed')
    temporary.replace(output)
    report = dict(file=output.name, sha256=sha256(output), bytes=output.stat().st_size,
                  expanded_bytes=expanded, choice=choice, crc_verified=True,
                  gpu_end_to_end_verified=False, official_environment='Python 3.11, torch 2.7.1 CUDA 12.8',
                  dependency_source='https://dacon.io/competitions/official/236749/overview/evaluation')
    write_json(output.with_suffix('.json'), report)
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    print(json.dumps(package(root, root/'submissions/C08/C08_submit.zip'), indent=2))
