"""Prepare grouped real-audio probes before starting paid compute.

Run: .venv/bin/python prepare_six_h100.py speech|probes
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import imageio_ffmpeg
import numpy as np

from prepare_v2_sources import MLAAD_REV, fetch
from v2_data import decode_audio, pcm_sha256, sha256_file, stable_key
from v4_data_acquisition import SPEECH_PROVIDERS, _speaker_id

ROOT = Path(__file__).resolve().parent
DATA = ROOT / '_workspace/six_h100_20260917'
OUT = ROOT / 'results/six_h100_20260917'
LABELS = ('FILE_FAKE', 'VOICE_FAKE', 'MUSIC_FAKE', 'VOICE_PRESENT', 'MUSIC_PRESENT')


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def speech() -> None:
    prior = json.loads((ROOT / 'v4_validation/provenance.json').read_text())['sources']
    excluded = {r.get('speaker_id') for r in prior}
    old_hashes = {r.get(k) for r in prior for k in ('source_sha256', 'source_pcm_sha256', 'pcm_sha256')}
    candidates = []
    for provider in SPEECH_PROVIDERS:
        path = ROOT / f'_workspace/v4_data/sources/metadata/mlaad_de_{stable_key(provider)[:12]}.csv'
        for row in csv.DictReader(path.open(), delimiter='|'):
            parts = Path(row['original_file']).parts
            if len(parts) < 5 or parts[2] == 'mix':
                continue
            speaker = _speaker_id(row['original_file'])
            if speaker not in excluded and float(row['duration']) >= 4:
                candidates.append(row | {'provider': provider, 'speaker_id': speaker})
    speakers = sorted({r['speaker_id'] for r in candidates}, key=stable_key)
    if len(speakers) != 4:
        raise ValueError(f'Expected four unused named German speakers, found {len(speakers)}')
    selected = []
    for index, speaker in enumerate(speakers):
        used = set()
        for provider in SPEECH_PROVIDERS:
            cohort = sorted((r for r in candidates if r['speaker_id'] == speaker and r['provider'] == provider),
                            key=lambda r: stable_key(r['original_file']))
            item = next(r for r in cohort if r['original_file'] not in used)
            used.add(item['original_file'])
            selected.append(item | {'split': ('train', 'train', 'dev', 'test')[index]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'speech_selection.json').write_text(json.dumps(selected, indent=2))
    base = f'https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{MLAAD_REV}/'
    jobs = []
    for item in selected:
        filename = Path(item['path']).name
        for fake in (0, 1):
            remote = f"fake/de/{item['provider']}/{filename}" if fake else f'original/de/{filename}'
            path = DATA / 'speech' / f'{stable_key(remote)[:20]}.wav'
            jobs.append((item, fake, base + quote(remote, safe='/'), path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda job: fetch(job[2], job[3]), jobs))
    rows, seen = [], set()
    for item, fake, url, path in jobs:
        audio = decode_audio(path, limit_seconds=None)
        digest, pcm = sha256_file(path), pcm_sha256(audio)
        if len(audio) < 64000 or digest in old_hashes or pcm in old_hashes or pcm in seen:
            raise ValueError(f'Speech duration or duplicate check failed: {path}')
        seen.add(pcm)
        rows.append(dict(ID='speech_' + path.stem, path=str(path.relative_to(ROOT)),
                         group_id=item['speaker_id'], split=item['split'], source='MLAAD',
                         provider=item['provider'] if fake else 'original', source_url=url,
                         revision=MLAAD_REV, source_sha256=digest, pcm_sha256=pcm,
                         duration_seconds=len(audio)/16000, FILE_FAKE=str(fake), VOICE_FAKE=str(fake),
                         MUSIC_FAKE='', VOICE_PRESENT='1', MUSIC_PRESENT='0',
                         label_status='source-annotated speech; no independent listening review',
                         original_file=item['original_file']))
    write_rows(OUT / 'fresh_speech_manifest.csv', rows)
    print(json.dumps({'speech_files': len(rows), 'independent_speakers': len(speakers)}), flush=True)


def save_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-f', 'f32le',
                    '-ar', '16000', '-ac', '1', '-i', 'pipe:0', '-c:a', 'pcm_f32le', str(path)],
                   input=audio.astype('<f4').tobytes(), check=True, capture_output=True)
    if not np.array_equal(decode_audio(path, limit_seconds=None), audio.astype(np.float32)):
        raise ValueError('Lossless probe write changed samples')


def probes() -> None:
    sources = []
    for name in ('fma', 'sonics', 'speech'):
        cohort = list(csv.DictReader((OUT / f'fresh_{name}_manifest.csv').open()))
        if name != 'speech':
            for provider in sorted({r['source'] for r in cohort}):
                subset = sorted((r for r in cohort if r['source'] == provider), key=lambda r: stable_key(r['group_id']))
                for index, row in enumerate(subset):
                    row['split'] = 'train' if index < int(len(subset)*.6) else 'dev' if index < int(len(subset)*.8) else 'test'
        sources.extend(cohort)
    groups = {s: {r['group_id'] for r in sources if r['split'] == s} for s in ('train', 'dev', 'test')}
    assert not (groups['train'] & groups['dev'] or groups['train'] & groups['test'] or groups['dev'] & groups['test'])
    rows, base_clips, unavailable = [], {}, []

    def add(row: dict, audio: np.ndarray, variant: str, parents: list[dict] | None = None) -> dict:
        identifier = row['ID'] + '_' + variant
        path = DATA / 'audio' / (identifier + '.wav')
        save_audio(path, audio)
        parents = parents or [row]
        assert len({p['split'] for p in parents}) == 1
        output = {**{k: row.get(k, '') for k in LABELS}, 'ID': identifier,
                  'path': 'audio/' + path.name, 'audio_sha256': sha256_file(path),
                  'group_id': '|'.join(sorted({p['group_id'] for p in parents})),
                  'parent_id': row['ID'], 'parent_ids': json.dumps([p['ID'] for p in parents]),
                  'split': row['split'], 'source': row['source'], 'variant': variant,
                  'duration_seconds': len(audio)/16000,
                  'label_status': 'source metadata / construction labels; independent listening pending'}
        rows.append(output)
        return output

    for row in sources:
        source = ROOT / row['path']
        assert sha256_file(source) == row['source_sha256']
        audio = decode_audio(source, limit_seconds=None)
        start = max(0, (len(audio) - 60*16000)//2)
        for seconds in (4, 15, 60):
            if len(audio) < seconds*16000:
                unavailable.append({'ID': row['ID'], 'seconds': seconds})
                continue
            add(row, audio[start:start+seconds*16000], f'{seconds}s')
        clip = audio[start:start+15*16000]
        base_clips[row['ID']] = clip
        # Codec comparisons use exactly the same source interval; speech can be shorter than 15 s.
        clean = add(row, clip, 'base')
        input_path = DATA / clean['path']
        for codec, args, suffix in (
            ('mp3_32k', ['-c:a', 'libmp3lame', '-b:a', '32k'], '.mp3'),
            ('telephone', ['-af', 'highpass=f=300,lowpass=f=3400', '-ar', '8000', '-c:a', 'pcm_mulaw'], '.wav'),
        ):
            temp = DATA / 'encoded' / (row['ID'] + '_' + codec + suffix)
            temp.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-i', str(input_path),
                            *args, str(temp)], check=True, capture_output=True)
            transformed = decode_audio(temp, limit_seconds=None)
            if abs(len(transformed)-len(clip)) > 1600:
                raise ValueError('Codec introduced an unexpected timing change')
            result = add(row, transformed, codec)
            result['encoded_source_sha256'] = sha256_file(temp)
            result['codec_scope'] = 'decoded waveform degradation; native codec runtime separately untested'
    for split in ('train', 'dev', 'test'):
        real = [r for r in sources if r['split'] == split and r['source'] == 'FMA']
        fake_music = [r for r in sources if r['split'] == split and r['MUSIC_FAKE'] == '1']
        voices = [r for r in sources if r['split'] == split and r['source'] == 'MLAAD']
        pairs = [(a, real[i % len(real)], 'speech_music') for i, a in enumerate(voices)]
        pairs += [(a, real[i % len(real)], 'partial_voice') for i, a in enumerate(voices) if a['VOICE_FAKE']=='1']
        pairs += [(a, real[i % len(real)], 'partial_music') for i, a in enumerate(fake_music)]
        real_voice = [r for r in voices if r['VOICE_FAKE'] == '0']
        pairs += [(a, fake_music[i % len(fake_music)], 'real_voice_fake_music') for i, a in enumerate(real_voice)]
        for first, second, kind in pairs:
            a, b = base_clips[first['ID']], base_clips[second['ID']]
            b = b.copy()
            a = a[:min(len(a), len(b))]
            if kind in ('partial_music', 'partial_voice'):
                a = a[:3*16000]
            for value in (a, b):
                if float(np.sqrt(np.mean(value**2))) < 1e-5:
                    raise ValueError('Cannot construct an audible mixture from silence')
            # Fixed equal RMS; no model-driven mixture selection.
            a = a * (.10 / np.sqrt(np.mean(a**2)))
            b *= .10 / np.sqrt(np.mean(b**2))
            offset = (len(b)-len(a))//2
            b[offset:offset+len(a)] += a
            b *= min(1., .98 / max(float(np.abs(b).max()), 1e-6))
            labels = dict(FILE_FAKE=str(max(int(first['FILE_FAKE']), int(second['FILE_FAKE']))),
                          MUSIC_FAKE='0' if kind in ('speech_music','partial_voice') else '1', MUSIC_PRESENT='1',
                          VOICE_PRESENT='1' if kind != 'partial_music' else '',
                          VOICE_FAKE='1' if first['VOICE_FAKE'] == '1' else '')
            mixed = first | labels | {'ID': first['ID'] + '_' + second['ID'], 'source': 'constructed'}
            result = add(mixed, b, kind, [first, second])
            result['event_intervals'] = json.dumps([dict(component='music' if kind == 'partial_music' else 'voice',
                                                         start=offset/16000, end=(offset+len(a))/16000,
                                                         label=int(first['FILE_FAKE']))])
    assert len({r['ID'] for r in rows}) == len(rows)
    write_rows(OUT / 'probes.csv', rows)
    write_rows(OUT / 'sources.csv', sources)
    receipt = dict(rows=len(rows), parents=len(sources), variants=dict(Counter(r['variant'] for r in rows)),
                   splits={s: dict(parents=sum(r['split']==s for r in sources), groups=len(groups[s])) for s in groups},
                   unavailable_lengths=unavailable, manifest_sha256=sha256_file(OUT/'probes.csv'),
                   labels_independently_reviewed=False, official_score=None,
                   limitations=['FMA versus SONICS source confounding remains',
                                'Only four new speech speakers; one dev and one test speaker',
                                'Unknown music-source voice labels remain unknown in mixtures',
                                'MP3 and telephone probes measure decoded degradations, not native codec ingestion'])
    (OUT / 'data_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ('rows', 'parents', 'variants', 'splits')}), flush=True)


if __name__ == '__main__':
    {'speech': speech, 'probes': probes}[sys.argv[1]]()
