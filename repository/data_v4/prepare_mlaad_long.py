#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub"]
# ///
# Run: python data_v4/prepare_mlaad_long.py METADATA_INDEX ROLES_MANIFEST OLD_PLAN_DIRECTORY OUTPUT_DIRECTORY
"""Select existing long synthetic speech using the user's existing corpus access only."""
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from urllib.parse import quote

from huggingface_hub import HfApi, hf_hub_download


def main(index: Path, manifest: Path, old_plans: Path, output: Path) -> None:
    """Exclude prior paths and evaluation text groups; preserve uncertain admission."""
    metadata = json.loads(index.read_text())
    prior_paths = set()
    for path in old_plans.glob('mlaad_*.json'):
        for row in json.loads(path.read_text()):
            prior_paths.add('fake/' + row['path'].split('/fake/', 1)[1])
    held = set()
    for line in manifest.open():
        row = json.loads(line)
        if row['role'] != 'train':
            held.update(g for g in row.get('group_keys', []) if g.startswith('mailabs_text:'))
    groups = defaultdict(list)
    for group in metadata['groups']:
        for row in group.get('eligible', []):
            path = row['path'].removeprefix('./')
            text_group = 'mailabs_text:' + PurePosixPath(row['original_file']).stem
            if path in prior_paths or text_group in held or not 15.25 <= float(row['duration']) <= 59.5:
                continue
            p = PurePosixPath(path)
            if p.is_absolute() or '..' in p.parts or p.parts[:2] != ('fake', 'en') or p.suffix != '.wav':
                raise ValueError('Unexpected corpus audio path')
            groups[group['folder']].append(dict(ID='mlaad:' + '/'.join(p.parts[1:]).removesuffix('.wav'),
                path=path, source='MLAAD', generator=group['folder'].split('/')[-1],
                model_name=row['model_name'], language=row['language'], original_file=row['original_file'],
                group_keys=[text_group], reference_speaker=row.get('reference_speaker'),
                duration_metadata_seconds=float(row['duration']), metadata_sha256=group['metadata_sha256'],
                source_url=f'https://huggingface.co/datasets/{metadata["repo_id"]}/resolve/{metadata["revision"]}/{quote(path, safe="/")}',
                revision=metadata['revision'], license='https://creativecommons.org/licenses/by-nc/4.0/',
                FILE_FAKE=1, VOICE_FAKE=1, VOICE_PRESENT=1, MUSIC_PRESENT=0, MUSIC_FAKE=None,
                label_basis='publisher synthetic-speech metadata; existing released audio, no new synthesis',
                role='unassigned', training_admitted=False, listening_verified=False,
                rights_review='noncommercial and competition/provision review pending',
                corpus_overlap_review='prior download paths and evaluation source-text groups excluded; aliases pending'))
    queues = {key: deque(sorted(rows, key=lambda r: hashlib.sha256(r['ID'].encode()).hexdigest()))
              for key, rows in sorted(groups.items())}
    chosen = []
    while queues and len(chosen) < 400:
        for key in list(queues):
            chosen.append(queues[key].popleft())
            if not queues[key]:
                del queues[key]
            if len(chosen) == 400:
                break
    info = HfApi().get_paths_info(metadata['repo_id'], [r['path'] for r in chosen],
        repo_type='dataset', revision=metadata['revision'])
    files = {f.path: f for f in info}
    for row in chosen:
        file = files[row['path']]
        if not file.lfs or not 0 < file.size <= 20 * 2**20:
            raise ValueError('Missing source checksum or oversized speech file')
        row.update(sha256=file.lfs.sha256, bytes=file.size)
    if sum(r['bytes'] for r in chosen) > 2 * 2**30:
        raise ValueError('Long-speech batch exceeds 2 GiB')
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in chosen))
    plan = [dict(dataset='MLAAD_long', path=r['path'], url=r['source_url'], size=r['bytes'],
        checksum='sha256:' + r['sha256'], license=r['license'], source_split='unassigned') for r in chosen]
    (output/'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    card = hf_hub_download(metadata['repo_id'], 'README.md', repo_type='dataset', revision=metadata['revision'])
    (output/'README.upstream.md').write_bytes(Path(card).read_bytes())
    (output/'ATTRIBUTION.md').write_text('MLAAD, Nicolas Mueller et al.\n'
        'https://huggingface.co/datasets/mueller91/MLAAD\n'
        'CC BY-NC 4.0. Original released WAV files retained without modification.\n'
        'Existing account access used; no new consent or authentication was granted.\n'
        'Competition admission and downstream provision review remain pending.\n')
    print(json.dumps(dict(selected=len(chosen), generators=len({r['generator'] for r in chosen}),
        bytes=sum(r['bytes'] for r in chosen), held_text_groups=len(held), training_admitted=False)))


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
