"""Export verified cached features on the source cluster using a CPU-only job."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_features(rows: list[dict[str, str]], old: Path, final: Path, target: Path) -> None:
    """Bind each portable cache to its verified source audio and original cache ID."""
    previous = json.loads((old / 'training.json').read_text())
    split = json.loads((old / 'split.json').read_text())
    old_ids = {r['ID'] for part in ('expert', 'fusion', 'dev') for r in split[part]}
    records = {}
    if len({r['ID'] for r in rows}) != len(rows):
        message = 'Duplicate export IDs'
        raise ValueError(message)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED) as archive:
        for index, row in enumerate(rows):
            assert digest(Path(row['path'])) == row['audio_sha256'], row['ID']
            identity = row['audio_sha256'] + ':' + previous['cache_id']
            directory = old / 'cache' if row['ID'] in old_ids else final / 'final_run/new_cache'
            source = directory / (hashlib.sha256(identity.encode()).hexdigest() + '.npz')
            name = 'features/' + row['ID'] + '.npz'
            records[row['ID']] = dict(path=name, identity=identity, sha256=digest(source),
                                       audio_sha256=row['audio_sha256'])
            archive.write(source, name)
            if (index + 1) % 100 == 0:
                print('EXPORTED', index + 1, 'of', len(rows), flush=True)
        archive.writestr('features_manifest.json', json.dumps(dict(records=records,
            cache_id=previous['cache_id'], asset_id=previous['asset_id'],
            verification='Source audio SHA256 verified before export; original cache bytes preserved')))
    temporary.replace(target)


if __name__ == '__main__':
    directory = Path('/root-1/temporal/h100-transfer')
    rows = json.loads((directory / 'selected_rows.json').read_text())
    target = Path('/root/Temporal_Feature_Export/features.zip')
    export_features(rows, Path('/root/V6_L40S/v6_run'), Path('/root/C3_Final'), target)
    shutil.copyfile(target, directory / target.name)
    report = dict(file=target.name, bytes=target.stat().st_size, sha256=digest(target), files=len(rows))
    (directory / 'features_download_manifest.json').write_text(json.dumps(report))
    print('FEATURES_READY', json.dumps(report), flush=True)
