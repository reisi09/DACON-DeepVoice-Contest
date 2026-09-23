"""Data v4 source catalog -> file-level acquisition plan for candidate_data_acquire.py.

Run: python data_v4/sources.py build --out data_v4/plan   (network; HF_TOKEN for gated)
     python data_v4/sources.py list
Every entry records license/access evidence checked 2026-09-23. Plans pin a revision
and carry upstream checksums; nothing here downloads audio.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import quote, unquote
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    kind: str                 # hf | zenodo | url | modelscope
    ref: str                  # repo id / record id / url
    license: str
    evidence: str             # where the license/access was checked
    role: str                 # what gap it fills
    include: tuple[str, ...] = ('*',)
    exclude: tuple[str, ...] = ()
    per_dir_cap: int = 0      # max items (stem groups) per directory; 0 = all
    enabled: bool = True
    note: str = ''
    extra: dict = field(default_factory=dict)


# Gaps (see DESIGN.md §1): Korean real/fake speech, telephone channel, modern TTS/VC,
# real singing, real/fake songs with vocals, new music generators, non-speech/music audio.
CATALOG: tuple[Source, ...] = (
    # --- speech: real ---
    Source('zeroth_korean', 'hf', 'kresnik/zeroth_korean', 'CC-BY-4.0',
           'HF card 2026-09-23', 'Korean real speech; text+speaker source for TTS/VC generation'),
    Source('fleurs_ko', 'hf', 'google/fleurs', 'CC-BY-4.0', 'HF tag license:cc-by-4.0',
           'Korean real read speech, second domain', include=('data/ko_kr/*',)),
    Source('vocalset', 'zenodo', '1193957', 'CC-BY-4.0', 'Zenodo API license id',
           'Real a cappella singing (VOICE_PRESENT=1, MUSIC_PRESENT=0)'),
    Source('csd', 'zenodo', '4916302', 'CC-BY-NC-SA-4.0', 'Zenodo API license id',
           'Real Korean/English children songs a cappella', include=('CSD.zip',)),
    # --- speech: fake / channel ---
    Source('asvspoof2021_la', 'zenodo', '4837263', 'ODC-By-1.0', 'Zenodo API license id',
           'Real telephony transmission (a-law, G.722, PSTN, VoIP) of bona fide+spoof',
           extra={'keys': 'https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz'}),
    Source('asvspoof2021_df', 'zenodo', '4835108', 'ODbL-1.0', 'Zenodo API license id',
           'Compression codecs (mp3/m4a/ogg) + 100+ vocoders; one part only',
           include=('ASVspoof2021_DF_eval_part00.tar.gz',),
           extra={'keys': 'https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz'}),
    Source('mlaad_ko', 'hf', 'mueller91/MLAAD', 'CC-BY-NC-4.0', 'HF card; gated=auto (click-through)',
           'Korean TTS fakes, all systems', include=('fake/ko/*',),
           note='needs HF_TOKEN after accepting terms on the dataset page'),
    Source('mlaad_other', 'hf', 'mueller91/MLAAD', 'CC-BY-NC-4.0', 'HF card; gated=auto',
           'Generator diversity: 40 items per (language, system) dir', include=('fake/*',),
           exclude=('fake/ko/*',), per_dir_cap=40),
    Source('speechfake', 'modelscope', 'inclusionAI/SPEECHFAKE', 'CC-BY-4.0',
           'GitHub README badge + LICENSE; real part from Common Voice CC0',
           'Korean fakes from 30 open generators (CosyVoice, Fish, GPT-SoVITS, MeloTTS, Seed-VC...)',
           note='Korean subset via HTTP Range from the 115.7 GB split zip: vessl_fetch_speechfake.sh / zip_range_extract.py'),
    Source('rtcfake', 'hf', 'JunXueTech/RTCFake', 'CC-BY-NC-4.0',
           'HF tag; gated=manual (author approval)', 'Deepfakes transmitted through real RTC apps',
           enabled=False, note='manual approval -> "publicly accessible" is uncertain; enable only if approved and noted in report'),
    # --- music ---
    Source('haim', 'hf', 'mippia/HAIM@371be01815b65274762d6d0a33e2cb70e24b9baa', 'CC-BY-NC-4.0 + MTG/SONICS credits',
           'HF card; audio subsets only, link subsets excluded',
           'Real MTG music, Lyria/ACE-Step/MusicGen fakes, AI-mastered real (B1), human-mastered AI, partial edits, mixsets',
           include=('A_full_generation/*', 'B_hybrid/*', 'C_mixing/*'),
           exclude=('*links*', '*sonics*'), per_dir_cap=150),
    Source('echoes', 'hf', 'Octavian97/Echoes', 'CC-BY-SA-4.0', 'HF tag license:cc-by-sa-4.0',
           '12 music generators TTA/ATA incl. vocal songs; replaces 8 s legacy crops'),
    Source('sonics', 'hf', 'awsaf49/sonics', 'CC-BY-NC-4.0', 'HF tag',
           'Suno/Udio fake songs with vocals (full length)',
           include=('*.csv', 'metadata.json', 'fake_songs/part_01.zip', 'fake_songs/part_02.zip')),
    Source('openmic', 'zenodo', '1432913', 'CC-BY-4.0', 'Zenodo API license id',
           'Real music 10 s clips with voice annotation mask (vocal vs instrumental)'),
    Source('musan', 'url', 'https://openslr.trmal.net/resources/17/musan.tar.gz', 'CC-BY-4.0',
           'openslr.org/17 page', 'Real music (5 corpora), real speech, noise (none/none rows)'),
)

# Deliberately not collected (DESIGN.md §2.3): AI-Hub (nationality/approval gate), KSS (Kaggle login),
# Common Voice direct (Mozilla Data Collective registration; SpeechFake carries CC0 CV reals),
# SingFake/WildSVDD/SONICS-real (links only), ASVspoof5 (142 GB English), Codecfake (ND + label conflict).


def cap_items(paths: list[str], cap: int) -> list[str]:
    """Keep sidecars together: an item is dir + stem. Evenly spaced over sorted stems per dir."""
    if cap <= 0:
        return sorted(paths)
    groups: dict[tuple[str, str], list[str]] = {}
    for p in paths:
        d, _, base = p.rpartition('/')
        groups.setdefault((d, base.split('.')[0]), []).append(p)
    by_dir: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(groups):
        by_dir.setdefault(key[0], []).append(key)
    keep = []
    for keys in by_dir.values():
        n = len(keys)
        picks = keys if n <= cap else [keys[i * n // cap] for i in range(cap)]
        for k in picks:
            keep += groups[k]
    return sorted(keep)


def selected(src: Source, path: str) -> bool:
    return (any(fnmatch.fnmatch(path, g) for g in src.include)
            and not any(fnmatch.fnmatch(path, g) for g in src.exclude))


def _json(url: str):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'dacon-data-v4'}), timeout=120))


def _head_size(url: str) -> int:
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'dacon-data-v4'})
    return int(urllib.request.urlopen(req, timeout=120).headers['Content-Length'])


def plan_hf(src: Source) -> list[dict]:
    from huggingface_hub import HfApi
    repo, _, rev = src.ref.partition('@')
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    rev = rev or api.dataset_info(repo).sha
    files = [f for f in api.list_repo_tree(repo, repo_type='dataset', revision=rev, recursive=True)
             if hasattr(f, 'size') and selected(src, f.path)]
    info = {f.path: f for f in files}
    rows = []
    for path in cap_items(list(info), src.per_dir_cap):
        f = info[path]
        checksum = 'sha256:' + f.lfs.sha256 if f.lfs else 'git-sha1:' + f.blob_id
        if '*' in checksum:  # gated repo listed without access: HF masks hashes -> plan would be unverifiable
            raise RuntimeError(f'{repo}: masked checksum; set HF_TOKEN with granted access and rebuild')
        rows.append(dict(dataset=src.name, path=f'{src.name}/{path}', size=f.size, checksum=checksum,
                         url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{quote(path)}',
                         license=src.license, source_split='unassigned'))
    return rows


def plan_zenodo(src: Source) -> list[dict]:
    record = _json(f'https://zenodo.org/api/records/{src.ref}')
    rows = [dict(dataset=src.name, path=f"{src.name}/{f['key']}", size=f['size'], checksum=f['checksum'],
                 url=f"https://zenodo.org/api/records/{src.ref}/files/{f['key']}/content",
                 license=src.license, source_split='unassigned')
            for f in record['files'] if selected(src, f['key'])]
    if 'keys' in src.extra:
        rows.append(plan_url(src, src.extra['keys'])[0])
    return rows


def plan_url(src: Source, url: str | None = None) -> list[dict]:
    url = url or src.ref
    return [dict(dataset=src.name, path=f"{src.name}/{url.rsplit('/', 1)[1]}", size=_head_size(url),
                 checksum='size-only:', url=url, license=src.license, source_split='unassigned')]


def build(out: Path, names: set[str] | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for src in CATALOG:
        if not src.enabled or (names and src.name not in names):
            continue
        if src.kind == 'modelscope':
            summary[src.name] = {'manual': f'modelscope download --dataset {src.ref} (inspect subsets first)'}
            continue
        try:
            rows = {'hf': plan_hf, 'zenodo': plan_zenodo, 'url': plan_url}[src.kind](src)
        except Exception as exc:  # gated/offline sources must not block the others
            summary[src.name] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
            continue
        (out / f'{src.name}.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False) + '\n')
        summary[src.name] = {'files': len(rows), 'GB': round(sum(r['size'] for r in rows) / 1e9, 2)}
    summary['TOTAL_GB'] = round(sum(v.get('GB', 0) for v in summary.values() if isinstance(v, dict)), 2)
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


def self_check() -> None:
    paths = [f'd{d}/{i:03d}.{e}' for d in range(2) for i in range(10) for e in ('mp3', 'json')] + ['d2/x.wav']
    kept = cap_items(paths, 3)
    assert len(kept) == 3 * 2 * 2 + 1, kept                       # 3 items x 2 sidecars x 2 dirs + 1-file dir
    assert {p.split('.')[0] for p in kept if p.startswith('d0')} == {'d0/000', 'd0/003', 'd0/006'}
    s = Source('t', 'hf', 'x', 'l', 'e', 'r', include=('a/*',), exclude=('*links*',))
    assert selected(s, 'a/b.wav') and not selected(s, 'a/links/b.wav') and not selected(s, 'b/c.wav')
    print('sources self-check ok')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'list', 'check'])
    ap.add_argument('--out', type=Path, default=Path('data_v4/plan'))
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    if a.cmd == 'check':
        self_check()
    elif a.cmd == 'list':
        for s in CATALOG:
            print(f"{'ON ' if s.enabled else 'OFF'} {s.name:18} {s.license:30} {s.role}")
    else:
        build(a.out, set(a.only) if a.only else None)
    sys.exit(0)
