"""Final all-permitted-local-data refit of the unchanged B1–B4 architectures."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from v5_dataset import LABELS, Example, _read_rows
from v5_reporting import write_json


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_final_pool(data: Path) -> tuple[list[Example], dict]:
    """Merge licensed local train/dev/sealed, preserving unknowns and full sources."""
    data = data.resolve()
    sources = json.loads((data / 'provenance.json').read_text())['sources']
    provenance = {item['ID']: item for item in sources}
    if len(provenance) != len(sources):
        raise ValueError('Duplicate provenance identifiers')
    rows, receipts, hashes, split_counts = [], [], {}, {}
    for split in ('train', 'dev', 'sealed'):
        split_rows = _read_rows(data, split)
        split_counts[split] = len(split_rows)
        for row in split_rows:
            source = provenance.get(row.id)
            if source is None or not source.get('license'):
                raise ValueError(f'Missing licensed provenance: {row.id}')
            for name, value in zip(LABELS, row.labels, strict=True):
                recorded = source.get(name, '')
                if recorded not in ('', '0', '1') or (recorded == '') != math.isnan(value) or (recorded and float(recorded) != value):
                    raise ValueError(f'Manifest/provenance label mismatch: {row.id}/{name}')
            selected = (data / (source.get('full_path') or source['path'])).resolve()
            if not selected.is_relative_to(data) or not selected.is_file():
                raise ValueError(f'Missing or escaped final training source: {row.id}')
            audio, rate = sf.read(selected, dtype='float32', always_2d=True)
            if rate != 16000 or audio.shape[1] != 1 or not len(audio) or not np.isfinite(audio).all():
                raise ValueError(f'Invalid prepared audio: {row.id}')
            pcm_hash = hashlib.sha256(np.asarray(audio[:, 0], dtype='<f4').tobytes()).hexdigest()
            known = {name: value for name, value in zip(LABELS, row.labels, strict=True) if math.isfinite(value)}
            previous = hashes.setdefault(pcm_hash, {})
            if any(name in previous and previous[name] != value for name, value in known.items()):
                raise ValueError(f'Conflicting labels for identical audio: {row.id}')
            previous.update(known)
            rows.append(replace(row, path=selected))
            receipts.append({'ID': row.id, 'original_split': split, 'path': str(selected.relative_to(data)),
                             'sha256': digest(selected), 'pcm_sha256': pcm_hash, 'seconds': len(audio) / rate,
                             'license': source['license'], 'labels': {name: source.get(name, '') for name in LABELS}})
    if len({row.id for row in rows}) != len(rows) or set(provenance) != {row.id for row in rows}:
        raise ValueError('Final data identifiers do not match complete provenance')
    audit = {'protocol': 'final refit: all licensed local train+dev+sealed; no DACON evaluation data',
             'manifest_rows': len(rows), 'unique_pcm_files': len(hashes), 'split_counts': split_counts,
             'independent_holdout': False, 'selection_on_merged_data': False,
             'original_full_source_preferred': True, 'duplicate_weighting': 'preserved manifest IDs and original exposure weights',
             'label_counts': {name: dict(Counter('unknown' if math.isnan(row.labels[i]) else str(int(row.labels[i])) for row in rows)) for i, name in enumerate(LABELS)},
             'manifests': {split: digest(data / f'{split}.csv') for split in split_counts},
             'provenance_sha256': digest(data / 'provenance.json'), 'audio': receipts}
    return rows, audit


def release(model: torch.nn.Module) -> None:
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def verify_packages(packages, rows, work, output, deadline):
    """Run real decoded audio through fresh ZIPs with network disabled and compare references."""
    import subprocess
    import sys
    import tempfile
    import zipfile

    import imageio_ffmpeg

    from graph_scoring import SUBMISSION_COLUMNS
    from v5_experts import load_expert
    from v5_features import assemble_features, encode_expert_views, prepare_views
    from v5_fusion import TemporalFusion
    from v5_fusion_training import _batch
    from v5_reporting import combine_experts, predict_expert

    qa = work / 'package_qa'
    qa.mkdir(exist_ok=True)
    evidence = output / 'package_qa'
    evidence.mkdir(exist_ok=True)
    specs = [('voice_real', 4, '.wav', lambda row: row.panel == 'speech' and row.labels[1] == 0),
             ('voice_fake', 10, '.flac', lambda row: row.panel == 'speech' and row.labels[1] == 1),
             ('music_real', 30, '.mp3', lambda row: row.panel == 'music_real'),
             ('music_fake', 60, '.wav', lambda row: row.panel == 'music_seen'),
             ('partial_fake', 30, '.wav', lambda row: row.panel == 'splice' and row.labels[0] == 1)]
    probes = []
    for name, seconds, suffix, predicate in specs:
        source = next(row for row in rows if predicate(row))
        audio, rate = sf.read(source.path, dtype='float32')
        audio = np.tile(audio, math.ceil(seconds * rate / len(audio)))[:seconds * rate]
        wav = qa / f'{name}.wav'
        # Stereo is deliberate decoder coverage; both channels retain the same mono signal.
        sf.write(wav, np.stack((audio, audio), axis=1) if name == 'music_real' else audio, rate, subtype='PCM_16')
        destination = qa / f'{name}{suffix}'
        if suffix == '.flac':
            sf.write(destination, audio, rate)
            wav.unlink()
        elif suffix == '.mp3':
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(wav), '-b:a', '128k', str(destination)], check=True)
            wav.unlink()
        probes.append(Example(name, destination, (source.id,), (math.nan,) * 5, 'probe'))
    silent = qa / 'silence.wav'
    sf.write(silent, np.zeros(64000, dtype=np.float32), 16000)
    probes.append(Example('silence', silent, ('silence',), (math.nan,) * 5, 'probe'))
    write_json(evidence / 'inputs.json', [{'ID': row.id, 'source_ids': row.groups, 'sha256': digest(row.path), 'path': row.path.name} for row in probes])
    views = prepare_views(probes, work / 'baseline', qa / 'views', deadline, training=False)
    expected, roots = {}, []
    for arm in ('B1', 'B2'):
        components = {}
        for kind in ('voice', 'music', 'presence'):
            model = load_expert(kind, work / 'baseline', work / 'sonics_assets', 'cuda')
            try:
                state = torch.load(output / 'jobs' / f'{arm}_{kind}' / 'final.pt', map_location='cpu', weights_only=True)
                model.load_state_dict(state, strict=True)
                del state
                components[kind] = predict_expert(model, probes, deadline)
                if arm == 'B2':
                    root = qa / 'features' / kind
                    encode_expert_views(model, probes, views, kind, root, deadline)
                    roots.append(root)
            finally:
                release(model)
                del model
        expected[arm] = combine_experts(*(components[kind] for kind in ('voice', 'music', 'presence')))
    features = assemble_features(probes, roots)
    for arm, joint in (('B3', False), ('B4', True)):
        model = TemporalFusion(features[0].features.shape[1], joint).cuda().eval()
        model.load_state_dict(torch.load(output / arm / 'final.pt', map_location='cpu', weights_only=True), strict=True)
        values, padding, times, _ = _batch(features, list(range(len(features))), torch.device('cuda'))
        with torch.inference_mode():
            expected[arm] = model(values, padding, times).probabilities.float().cpu().numpy()
        release(model)
        del model, values, padding, times
    guard = qa / 'offline_run.py'
    guard.write_text('''import json, resource, runpy, socket, time, torch
from pathlib import Path
def blocked(*args, **kwargs):
    raise RuntimeError("NETWORK_ACCESS_FORBIDDEN_DURING_INFERENCE")
socket.create_connection = blocked
socket.getaddrinfo = blocked
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
torch.set_num_threads(6)
torch.cuda.set_per_process_memory_fraction(min(1.0, 22.0 * 1024**3 / torch.cuda.get_device_properties(0).total_memory))
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
runpy.run_path("script.py", run_name="__main__")
torch.cuda.synchronize()
Path("output/runtime.json").write_text(json.dumps({"seconds": time.perf_counter()-start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "device": torch.cuda.get_device_name(0), "network_blocked": True}))
''')
    def write_sample(root, selected):
        with (root / 'data' / 'sample_submission.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
            writer.writeheader()
            writer.writerows({'ID': row.id, **{column: 0.0 for column in SUBMISSION_COLUMNS[1:]}} for row in selected)
    receipts = []
    for index, archive in enumerate(packages):
        arm = f'B{index + 1}'
        actual_expected = {row.id: expected[arm][i] for i, row in enumerate(probes)}
        write_json(evidence / f'{arm}_reference.json', {key: value.tolist() for key, value in actual_expected.items()})
        with tempfile.TemporaryDirectory(prefix=f'{arm}_unpacked_', dir=work) as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(archive) as packed:
                unpacked_bytes = sum(item.file_size for item in packed.infolist())
                if archive.stat().st_size > 10_000_000_000 or unpacked_bytes > 32_000_000_000:
                    raise ValueError('Submission archive exceeds official size limits')
                if any(Path(item.filename).is_absolute() or '..' in Path(item.filename).parts for item in packed.infolist()):
                    raise ValueError('Unsafe submission archive path')
                packed.extractall(root)
            audio_root = root / 'data' / 'test'
            audio_root.mkdir(parents=True)
            for row in probes:
                shutil.copy2(row.path, audio_root / row.path.name)
            runtime_records, first_predictions, maximum_delta = [], {}, 0.0
            for pass_index, selected in enumerate((probes, list(reversed(probes[:-1])))):
                if pass_index:
                    (audio_root / 'silence.wav').unlink()
                write_sample(root, selected)
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError('Package QA deadline reached')
                with (evidence / f'{arm}_pass{pass_index}.log').open('w') as log:
                    subprocess.run([sys.executable, str(guard)], cwd=root, check=True, timeout=remaining,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6'))
                with (root / 'output' / 'submission.csv').open(newline='') as stream:
                    reader = csv.DictReader(stream)
                    predictions = list(reader)
                    if reader.fieldnames != list(SUBMISSION_COLUMNS):
                        raise ValueError('Packaged output columns differ from competition contract')
                if [row['ID'] for row in predictions] != [row.id for row in selected]:
                    raise ValueError('Packaged output IDs/order differ from input template')
                for row in predictions:
                    values = np.asarray([float(row[column]) for column in SUBMISSION_COLUMNS[1:]])
                    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                        raise ValueError('Invalid packaged probabilities')
                    delta = float(np.max(np.abs(values - actual_expected[row['ID']])))
                    maximum_delta = max(maximum_delta, delta)
                    if delta > 1e-4:
                        raise ValueError(f'{arm}/{row["ID"]} packaged/reference mismatch: {delta}')
                    if pass_index and not np.allclose(values, first_predictions[row['ID']], atol=1e-6, rtol=0):
                        raise ValueError('Prediction depends on file order or companion files')
                    first_predictions[row['ID']] = values
                runtime = json.loads((root / 'output' / 'runtime.json').read_text())
                if runtime['peak_rss_bytes'] > 28_000_000_000:
                    raise ValueError('Packaged runtime exceeds evaluator host RAM')
                runtime_records.append(runtime)
                shutil.copy2(root / 'output' / 'submission.csv', evidence / f'{arm}_pass{pass_index}.csv')
            receipt = {'arm': arm, 'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive),
                       'unpacked_bytes': unpacked_bytes, 'offline_validation': {'status': 'passed', 'files': len(probes),
                       'max_reference_delta': maximum_delta, 'file_order_and_companion_independent': True,
                       'runs': runtime_records, 'L4_full_1200_file_runtime': 'not measured; A100 measurements do not prove the L4 limit'}}
            write_json(evidence / f'{arm}_verification.json', receipt)
            receipts.append(receipt)
            print('PACKAGE_OFFLINE_QA_PASS', arm, 'delta', maximum_delta, flush=True)
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    from package_v5_submission import build_v5_submissions
    from v5_dataset import SOURCE_PANELS
    from v5_experts import load_expert
    from v5_features import (
        add_compositions,
        assemble_features,
        encode_expert_views,
        prepare_views,
    )
    from v5_final_refit import (
        FinalExpertConfig,
        FinalFusionConfig,
        train_final_expert,
        train_final_fusion,
    )

    work, output = args.work.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = work / 'v4_validation'
    original_rows, audit = load_final_pool(data)
    write_json(output / 'data_audit.json', audit)
    print('FINAL_DATA', audit['manifest_rows'], 'unique_audio', audit['unique_pcm_files'], flush=True)
    link_root = work / 'final_pool' / 'train'
    link_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in original_rows:
        target = link_root / f'{row.id}.wav'
        if not target.exists():
            os.link(row.path, target)
        elif digest(target) != digest(row.path):
            raise ValueError(f'Stale final-pool link: {row.id}')
        rows.append(replace(row, path=target))
    work_deadline = args.deadline - 1200
    weights = {}
    for kind in ('voice', 'music', 'presence'):
        for arm in ('B1', 'B2'):
            if time.time() >= work_deadline:
                raise TimeoutError('Final expert refit cannot continue inside allocation budget')
            random.seed(20260913)
            np.random.seed(20260913)
            torch.manual_seed(20260913)
            torch.cuda.manual_seed_all(20260913)
            local = work / 'trials' / 'jobs' / f'{arm}_{kind}'
            print('FINAL_TRAIN_START', arm, kind, flush=True)
            model = load_expert(kind, work / 'baseline', work / 'sonics_assets', 'cuda')
            try:
                metadata = train_final_expert(model, rows, FinalExpertConfig(arm, kind, work / 'trials', work_deadline))
            finally:
                release(model)
                del model
            destination = output / 'jobs' / f'{arm}_{kind}'
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local / 'training.json', destination / 'training.json')
            if metadata['status'] != 'complete':
                raise TimeoutError(f'Incomplete final refit: {arm}/{kind}; no submission exported')
            shutil.copy2(local / 'final.pt', destination / 'final.pt')
            if digest(destination / 'final.pt') != metadata['state_sha256']:
                raise ValueError('Drive weight copy mismatch')
            if arm == 'B2':
                weights[kind] = local / 'final.pt'
            print('FINAL_TRAIN_COMPLETE', arm, kind, metadata['epochs_completed'], flush=True)
            checkpoint = local / 'checkpoint.pt'
            if checkpoint.is_file():
                checkpoint.unlink()

    temporal_rows, ledger = add_compositions(rows, work / 'compositions' / 'train')
    write_json(output / 'composition_ledger.json', ledger)
    source_rows = [row for row in temporal_rows if row.panel in SOURCE_PANELS]
    derived_rows = [row for row in temporal_rows if row.panel not in SOURCE_PANELS]
    views = prepare_views(source_rows, work / 'baseline', work / 'views' / 'sources', work_deadline, training=True)
    views.update(prepare_views(derived_rows, work / 'baseline', work / 'views' / 'derived', work_deadline, training=False))
    roots = []
    for kind in ('voice', 'music', 'presence'):
        model = load_expert(kind, work / 'baseline', work / 'sonics_assets', 'cuda')
        try:
            model.load_state_dict(torch.load(weights[kind], map_location='cpu', weights_only=True), strict=True)
            root = work / 'features' / kind
            encode_expert_views(model, temporal_rows, views, kind, root, work_deadline)
            roots.append(root)
        finally:
            release(model)
            del model
    features = assemble_features(temporal_rows, roots)
    write_json(output / 'feature_receipt.json', {'rows': len(features), 'weights': {kind: digest(path) for kind, path in weights.items()},
               'source_context_limit_seconds': 20, 'derived_partial_fake_full_file': True, 'independent_holdout': False})
    for name, joint in (('B3', False), ('B4', True)):
        local = work / 'trials' / name
        metadata = train_final_fusion(features, FinalFusionConfig(name, joint, work / 'trials', work_deadline))
        destination = output / name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local / 'training.json', destination / 'training.json')
        if metadata['status'] != 'complete':
            raise TimeoutError(f'Incomplete final temporal refit: {name}; no submission exported')
        shutil.copy2(local / 'final.pt', destination / 'final.pt')
        if digest(destination / 'final.pt') != metadata['state_sha256']:
            raise ValueError('Drive fusion copy mismatch')
        print('FINAL_TRAIN_COMPLETE', name, metadata['epochs_completed'], flush=True)
    packages = build_v5_submissions(work / 'baseline', work / 'sonics_assets', output, output / 'submissions')
    receipts = verify_packages(packages, original_rows, work, output, args.deadline)
    write_json(output / 'final_completion.json', {'status': 'ready_not_submitted', 'submitted': False,
               'packages': receipts, 'data_rows': len(rows), 'independent_holdout': False})
    print('FOUR_FINAL_SUBMISSION_ZIPS_READY', output / 'submissions', flush=True)


if __name__ == '__main__':
    main()
