"""Restore the Drive backup parts, verifying hashes before publishing the ZIP."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def restore(folder: Path) -> Path:
    manifest = json.loads((folder / 'split_manifest.json').read_text())
    names = [manifest['output_filename'], *(part['name'] for part in manifest['parts'])]
    if any(Path(name).name != name or name in {'', '.', '..'} for name in names):
        raise ValueError('backup filenames must remain inside their folder')
    output = folder / manifest['output_filename']
    def checksum(path: Path) -> str:
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    if output.exists():
        if checksum(output) != manifest['sha256']:
            raise FileExistsError('existing ZIP has a different hash; preserve it')
        return output
    temporary = output.with_suffix('.zip.partial')
    stream = temporary.open('xb')
    try:
        with stream as target:
            for part in manifest['parts']:
                path = folder / part['name']
                if path.stat().st_size != part['size'] or checksum(path) != part['sha256']:
                    raise ValueError(f"corrupt backup part: {path.name}")
                with path.open('rb') as source:
                    shutil.copyfileobj(source, target, 8 * 1024**2)
        if temporary.stat().st_size != manifest['size'] or checksum(temporary) != manifest['sha256']:
            raise ValueError('restored ZIP failed verification')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == '__main__':
    print(restore(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent))
