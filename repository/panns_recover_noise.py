"""Regenerate noise in a new namespace, retaining each historical codec choice."""
import hashlib
import json
import subprocess
import argparse
import time
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from panns_adaptation_data import recovered_noise_id

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work', type=Path, required=True, help='Verified raw recovery directory')
parser.add_argument('--manifest', type=Path, required=True, help='Historical roles manifest')
parser.add_argument('--output', type=Path, required=True, help='Fresh derived namespace within work')
args = parser.parse_args()
root = args.work.resolve()
if not args.output.resolve().is_relative_to(root):
    raise ValueError('Output must remain inside recovery work directory')
from data_v4 import build_manifest as bm

binary = imageio_ffmpeg.get_ffmpeg_exe()
run = subprocess.run
bm.subprocess = types.SimpleNamespace(run=lambda args, **kwargs: run(
    [binary if args[0]=='ffmpeg' else args[0], *args[1:]], timeout=120, **kwargs))
with args.manifest.open() as stream:
    rows = [r for r in map(json.loads, stream) if r['source']=='musan_noise']
with (root/'recovered_sources.jsonl').open() as stream:
    rawmeta = {r['ID']:r for r in map(json.loads, stream)}
codec_by_stem = {hashlib.sha256(r['ID'].encode()).hexdigest()[:20]: r['codec_chain'] for r in rows}
original_encode = bm.encode

def encode_historical(src: Path, stem: Path, _drawn: str) -> Path:
    codec = codec_by_stem[stem.name]
    if codec != 'tel_gsm':
        return original_encode(src, stem, codec)
    # Installed libsndfile supplies GSM610; FFmpeg supplies the original filters/resampling.
    pcm = stem.with_suffix('.filtered.wav')
    args = bm.FFMPEG['tel_gsm'][0].copy()
    args[args.index('libgsm_ms')] = 'pcm_s16le'
    run([binary, '-v', 'error', '-i', str(src), *args, str(pcm)], check=True, timeout=120)
    signal, rate = sf.read(pcm, dtype='int16')
    gsm = stem.with_suffix('.gsm.wav')
    sf.write(gsm, signal, rate, subtype='GSM610')
    if sf.info(gsm).subtype != 'GSM610':
        raise ValueError('GSM encoder did not emit GSM610')
    final = stem.with_suffix('.flac')
    run([binary, '-v', 'error', '-i', str(gsm), '-ar', '16000', str(final)], check=True, timeout=120)
    pcm.unlink(); gsm.unlink()
    return final

bm.encode = encode_historical
bm.usable_codecs()
out = args.output; out.mkdir(exist_ok=False)

def rebuild(row: dict) -> dict:
    source = Path(rawmeta[row['ID']]['path']).resolve()
    if not source.is_relative_to(root/'raw'):
        raise ValueError('Raw source outside verified recovery directory')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=rawmeta[row['ID']]['sha256']:
        raise ValueError('Raw source hash mismatch')
    result = bm.materialize_one(row | {'path':str(source)}, out, 20260923)
    result['codec_chain'] = row['codec_chain']
    signal, rate = sf.read(out/result['path'], dtype='float32', always_2d=True)
    if rate!=16000 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError('Invalid regenerated audio')
    for field in ('crop_start_s','crop_len_s','codec_chain','gain_db','channels'):
        if result[field]!=row[field]:
            raise ValueError(f'Historical preprocessing changed: {row["ID"]} {field}')
    result.update(ID=recovered_noise_id(row['ID'], result['sha256']), original_id=row['ID'],
        parent_ids=sorted(set(row.get('parent_ids', [])) | {row['ID']}),
        original_processed_sha256=row['sha256'], recovered_raw_sha256=rawmeta[row['ID']]['sha256'],
        recovery_method='seed20260923 with exact historical codec; new processed bytes',
        gsm_encoder='libsndfile GSM610' if row['codec_chain']=='tel_gsm' else None,
        full_audio_decode_verified=True, decoded_seconds=len(signal)/rate)
    result['group_keys'] = sorted(set(result['group_keys']+[f"bytes:{result['sha256']}"]))
    return result

started=time.monotonic(); result=[]
with ThreadPoolExecutor(max_workers=4) as pool, (out/'manifest.jsonl').open('w') as stream:
    for row in pool.map(rebuild, rows):
        result.append(row); stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
report=dict(rows=len(result), historical_preprocessing_matched=True,
    gsm_backend_rows=sum(r['gsm_encoder'] is not None for r in result),
    byte_identical=sum(r['sha256']==r['original_processed_sha256'] for r in result),
    role_counts=dict(Counter(r['role'] for r in result)),
    seconds=time.monotonic()-started, soundfile_version=sf.__version__,
    libsndfile_version=sf.__libsndfile_version__, ffmpeg_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
    source_archive_sha256=json.loads((root/'raw_recovery.json').read_text())['archive_sha256'],
    historical_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    materializer_sha256=hashlib.sha256(Path(bm.__file__).read_bytes()).hexdigest(),
    recovery_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    all_decoded=True, original_overwritten=False, automatic_admission=False)
(out/'recovery.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)
