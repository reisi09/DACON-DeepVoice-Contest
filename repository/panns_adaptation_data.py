"""Audio admission and immutable cache artifacts for PANNs adaptation."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np
import soundfile as sf
from data_v4.build_manifest import components
TARGETS = ('VOICE_PRESENT', 'MUSIC_PRESENT')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def quarantine_roles(rows):
    component = components(rows)
    roles = defaultdict(set)
    for row in rows:
        roles[component[row['ID']]].add(row['role'])
    return {row['ID']: dict(component_id=component[row['ID']],
                           quarantined=len(roles[component[row['ID']]]) != 1)
            for row in rows}


def safe_member(member):
    path = PurePosixPath(member.name)
    if not member.isfile() or path.is_absolute() or '..' in path.parts or str(path) != member.name or '\\' in member.name:
        raise ValueError(f'Unsafe archive member: {member.name}')
    if member.size > 64 * 1024**2:
        raise ValueError(f'Archive member exceeds resource limit: {member.name}')


def audit_archive(archive, rows):
    by_path = {row['path']: row for row in rows}
    if len(by_path) != len(rows) or len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('Duplicate manifest IDs or paths')
    verified, seen = {}, set()
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            safe_member(member)
            if member.name in seen:
                raise ValueError('Duplicate archive member')
            seen.add(member.name)
            if member.name == 'SHA256SUMS.json':
                continue
            if member.name not in by_path:
                raise ValueError(f'Unlisted archive member: {member.name}')
            row = by_path[member.name]
            payload = tar.extractfile(member).read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != row['sha256'] or ('bytes' in row and len(payload) != row['bytes']):
                raise ValueError(f'Audio hash/size mismatch: {row["ID"]}')
            try:
                with sf.SoundFile(io.BytesIO(payload)) as source:
                    if not 8000 <= source.samplerate <= 192000 or not 1 <= source.channels <= 2 or source.frames/source.samplerate > 600:
                        raise ValueError('Audio exceeds decode bounds')
                    sr = source.samplerate
                    audio = source.read(dtype='float32', always_2d=True)
            except Exception as exc:
                raise ValueError(f'Audio decode failed: {row["ID"]}: {exc}') from exc
            if not len(audio) or len(audio)/sr > 600 or not np.isfinite(audio).all():
                raise ValueError(f'Invalid decoded audio: {row["ID"]}')
            verified[row['ID']] = dict(sha256=digest, bytes=len(payload), sample_rate=sr,
                decoded_samples=len(audio), channels=audio.shape[1], seconds=len(audio)/sr,
                tar_offset=member.offset_data, path=member.name)
            if len(verified) % 5000 == 0:
                print(f'Validated {len(verified)} audio payloads', flush=True)
    return verified, [r['ID'] for r in rows if r['path'] not in seen]


def admit_noise(rows, verified, manifest, expected_hash):
    manifest = Path(manifest)
    if sha256(manifest) != expected_hash:
        raise ValueError('Recovered manifest hash mismatch')
    recovered = [json.loads(line) for line in manifest.open()]
    old = {r['ID']: r for r in rows if r['source'] == 'musan_noise'}
    if len(recovered) != len(old) or {r['ID'] for r in recovered} != set(old):
        raise ValueError('Recovered noise ID coverage mismatch')
    replacements = {}
    for row in recovered:
        original = old[row['ID']]
        fields = ('role', 'source', 'VOICE_PRESENT', 'MUSIC_PRESENT', 'VOICE_FAKE', 'MUSIC_FAKE',
                  'FILE_FAKE', 'parent_ids', 'historical_exposure', 'historical_split',
                  'crop_start_s', 'crop_len_s', 'gain_db', 'channels', 'codec_chain')
        if (any(row.get(k) != original.get(k) for k in fields)
                or row.get('original_processed_sha256') != original['sha256']
                or not set(original['group_keys']).issubset(row['group_keys'])):
            raise ValueError('Recovered audio changed historical identity, role or labels')
        if row['ID'] in verified or row['VOICE_PRESENT'] != 0 or row['MUSIC_PRESENT'] != 0:
            raise ValueError('Recovery must only fill missing, both-absent noise')
        path = (manifest.parent/row['path']).resolve()
        if not path.is_relative_to(manifest.parent.resolve()) or Path(row['path']).is_absolute():
            raise ValueError('Recovered audio escapes its namespace')
        if path.stat().st_size != row['bytes'] or row['bytes'] > 64*1024**2 or sha256(path) != row['sha256']:
            raise ValueError('Recovered audio hash/size mismatch')
        with sf.SoundFile(path) as source:
            if source.samplerate != 16000 or source.frames/16000 > 600 or source.channels not in (1, 2):
                raise ValueError('Recovered audio exceeds decode bounds')
            signal = source.read(dtype='float32', always_2d=True)
        if not len(signal) or not np.isfinite(signal).all():
            raise ValueError('Invalid recovered waveform')
        verified[row['ID']] = dict(local_path=str(path), sha256=row['sha256'], bytes=row['bytes'],
            sample_rate=16000, decoded_samples=len(signal), channels=signal.shape[1], seconds=len(signal)/16000)
        replacements[row['ID']] = row
    return [replacements.get(row['ID'], row) for row in rows]


def audit(config):
    out = Path(config['output'])
    rows = [json.loads(line) for line in Path(config['manifest']).open()]
    role_rows = list(csv.DictReader(Path(config['roles']).open()))
    roles = {r['ID']: r for r in role_rows}
    if len(roles) != len(role_rows) or set(roles) != {r['ID'] for r in rows}:
        raise ValueError('Role manifest identity mismatch')
    for row in rows:
        if row['role'] != roles[row['ID']]['role']:
            raise ValueError('Conflicting role assignment')
    actual = sha256(config['archive'])
    if actual != config['archive_sha256']:
        raise ValueError('Archive hash mismatch')
    verified, missing = audit_archive(config['archive'], rows)
    original_missing = len(missing)
    if config.get('recovered_noise_manifest'):
        rows = admit_noise(rows, verified, config['recovered_noise_manifest'], config['recovered_noise_sha256'])
        missing = [rid for rid in missing if rid not in verified]
    closure = quarantine_roles(rows)
    selected, excluded = [], Counter()
    for row in rows:
        reason = ('Dplus_not_D0' if row['source'] in ('LibriSpeech', 'MLAAD', 'MusicArena') else
                  'missing_audio' if row['ID'] not in verified else
                  'cross_role_component' if closure[row['ID']]['quarantined'] else None)
        if reason:
            excluded[f'{reason}|{row["source"]}|{row["role"]}'] += 1
            continue
        if any(row[k] not in (0, 1, None) for k in TARGETS):
            raise ValueError('Unexpected presence target')
        if not any(row[k] is not None for k in TARGETS):
            excluded['both_targets_unknown'] += 1
            continue
        selected.append({**row, 'verified_audio': verified[row['ID']], **closure[row['ID']]})
    train_rows = [r for r in selected if r['role'] == 'train']
    fraction = [sum(r[k] is not None for r in train_rows)/len(train_rows) for k in TARGETS]
    if any({r[k] for r in train_rows if r[k] is not None} != {0, 1} for k in TARGETS):
        raise ValueError('Training needs known positives and negatives for both targets')
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out/'presence_rows.jsonl'
    if rows_path.exists():
        raise FileExistsError('Use a fresh data audit namespace')
    rows_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
    missing_set = set(missing)
    report = dict(rows=len(rows), verified=len(verified), missing=len(missing),
        recovered_noise=original_missing-len(missing), recovered_noise_sha256=config.get('recovered_noise_sha256'),
        missing_by_source=dict(Counter(r['source'] for r in rows if r['ID'] in missing_set)),
        eligible_D0=len(selected), excluded=dict(excluded), known_fraction=fraction,
        presence_by_role=dict(Counter(f'{r["role"]}|{r[TARGETS[0]]}|{r[TARGETS[1]]}' for r in selected)),
        input_hashes={key: sha256(config[key]) for key in ('manifest', 'roles')},
        archive_sha256=actual, rows_sha256=sha256(rows_path),
        complete_presence_panel=any(r['VOICE_PRESENT']==0 and r['MUSIC_PRESENT']==0 for r in selected),
        independent_evaluation_available=False, soundfile_version=sf.__version__)
    write_json(out/'data_audit.json', report)
    print(json.dumps(report), flush=True)
    return report


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_config(config):
    if identity({k: v for k, v in config.items() if k != 'identity'}) != config['identity']:
        raise ValueError('Run manifest identity mismatch')
    for path, digest in config['inputs'].items():
        if sha256(path) != digest:
            raise ValueError(f'Pinned input hash mismatch: {path}')
    reference = json.loads(Path(config['reference']).read_text())
    package = Path(reference['package'])
    for name, digest in reference['members'].items():
        if '/panns/' in name or '/htdemucs/' in name or name in (
                'model/trained/presence_head.json', 'model/assets/baseline/script.py'):
            if sha256(package/name) != digest:
                raise ValueError(f'Reference asset hash mismatch: {name}')
    return package


def cache_provenance():
    from importlib.metadata import version
    root = Path(__file__).resolve().parent
    return dict(code={name: sha256(root/name) for name in (
        'panns_adaptation_cache.py', 'panns_adaptation_data.py', 'panns_adaptation.py', 'mert12_submission.py',
        'decision_heads.py', 'data_v4/build_manifest.py')},
        versions={name: version(name) for name in (
            'torch', 'torchaudio', 'numpy', 'librosa', 'soundfile', 'soxr',
            'panns-inference', 'demucs', 'torchlibrosa')},
        libsndfile=sf.__libsndfile_version__)


def read_cache_index(config):
    path = Path(config['output'])/'cache/index.json'
    if sha256(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Cache index hash mismatch')
    index = json.loads(path.read_text())
    if index.get('provenance') != cache_provenance():
        raise ValueError('Cache preprocessing/runtime provenance mismatch')
    if index['config_identity'] != config['identity']:
        raise ValueError('Cache/config identity mismatch')
    if [e['ID'] for e in index['entries']] != config['panel_ids']:
        raise ValueError('Cache panel IDs or order differ from locked manifest')
    return index


def load_cache(entry):
    if sha256(entry['path']) != entry['sha256']:
        raise ValueError('Cache hash mismatch')
    with np.load(entry['path'], allow_pickle=False) as data:
        if str(data['identity'].item()) != entry['identity']:
            raise ValueError('Cache identity mismatch')
        result = {key: data[key].copy() for key in data.files}
    if result['block5'].dtype != np.float32 or not np.isfinite(result['block5']).all():
        raise ValueError('Invalid cached activations')
    return result


