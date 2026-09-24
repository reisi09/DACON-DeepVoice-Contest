#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: python data_v4/prepare_music_arena_pilot.py TREE METADATA_DIRECTORY OUTPUT_DIRECTORY
"""Select public, checksum-bound music originals without admitting them to training."""
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def main(tree_path: Path, metadata: Path, output: Path) -> None:
    """Balance generator/prompt-type cells; exclude held-out Lyria and duplicate audio."""
    tree = json.loads(tree_path.read_text())
    files = {row['path']: row for row in tree['files']}
    revision = tree['revision']
    groups = defaultdict(list)
    for path in sorted(metadata.glob('battle_data/*/*.json')):
        raw = path.read_bytes()
        battle = json.loads(raw)
        prompt = ' '.join(str(battle.get('prompt', '')).casefold().split())
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        for side in ('a', 'b'):
            audio = battle.get('audio_' + side)
            generator = battle.get('system_' + side, '')
            duration = battle.get('duration_' + side, 0)
            if not audio or not generator or 'lyria' in generator.casefold() or not 15 <= duration <= 360:
                continue
            member = files.get(audio)
            if not member or not member.get('lfs') or not 0 < member['size'] <= 20 * 2**20:
                continue
            p = PurePosixPath(audio)
            if p.is_absolute() or '..' in p.parts or p.parts[0] != 'audio_files' or p.suffix != '.mp3':
                raise ValueError('Unexpected public audio path')
            groups[(generator, bool(battle.get('is_instrumental')))].append(dict(
                ID='music_arena:' + battle['battle_uuid'] + ':' + side,
                path=audio, sha256=member['lfs']['sha256'], bytes=member['size'],
                source='MusicArena', generator=generator, battle_id=battle['battle_uuid'],
                prompt_sha256=prompt_hash, group_keys=['prompt:music_arena:' + prompt_hash,
                    'battle:music_arena:' + battle['battle_uuid']],
                source_url=f'https://huggingface.co/datasets/{tree["repo_id"]}/resolve/{revision}/{audio}',
                revision=revision, license='https://creativecommons.org/licenses/by/4.0/',
                metadata_sha256=hashlib.sha256(raw).hexdigest(), duration_metadata_seconds=duration,
                prompt_is_instrumental=bool(battle.get('is_instrumental')), FILE_FAKE=1, MUSIC_FAKE=1,
                VOICE_FAKE=None, VOICE_PRESENT=None, MUSIC_PRESENT=1,
                label_basis='publisher generation metadata; vocal presence is not inferred from prompts',
                role='unassigned', training_admitted=False, rights_review='competition and downstream provision pending',
                listening_verified=False, corpus_overlap_review='pending'))
    queues = {key: deque(sorted(rows, key=lambda r: hashlib.sha256(r['ID'].encode()).hexdigest()))
              for key, rows in sorted(groups.items())}
    selected, seen = [], set()
    while queues and len(selected) < 1000:
        for key in list(queues):
            row = queues[key].popleft()
            if not queues[key]:
                del queues[key]
            if row['sha256'] not in seen:
                seen.add(row['sha256'])
                selected.append(row)
            if len(selected) == 1000:
                break
    if sum(row['bytes'] for row in selected) > 4 * 2**30:
        raise ValueError('Music pilot exceeds the 4 GiB download budget')
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in selected))
    plan = [dict(dataset='MusicArena', path=row['path'], url=row['source_url'], size=row['bytes'],
        checksum='sha256:' + row['sha256'], license=row['license'], source_split='unassigned') for row in selected]
    (output/'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (output/'README.upstream.md').write_bytes((metadata/'README.md').read_bytes())
    (output/'ATTRIBUTION.md').write_text('Music Arena Dataset, Yonghyun Kim et al. (2025).\n'
        'https://huggingface.co/datasets/music-arena/music-arena-dataset\n'
        'Publisher license: CC BY 4.0. Original released MP3 files, no transformations.\n'
        'Competition-use review pending; these files are not admitted to training.\n')
    print(json.dumps(dict(selected=len(selected), bytes=sum(r['bytes'] for r in selected),
        generators=sorted({r['generator'] for r in selected}), training_admitted=False)))


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
