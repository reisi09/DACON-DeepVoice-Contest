"""Runtime-exact training features for new audio rows (research/top15/work5/FINAL_PLAN_20260928.md, data additions).
The packaged Engine (PKG/model/runtime/full_bank_submission.py with the S7 assets) runs the Engine.predict path up to the
heads; one npz per row holds the union of the trainer cache keys (data_v4/train_backbone_heads.py, work2/backbones/run.py):
  backbone_cache  ad_embeddings f16, mert_layers f16 (windows, layers, 1024) + layers, voice_silent, samples
  cache_v4        voice_embeddings f32 (DF-Arena fc5 input per segment), window_truth f32 (NaN: no localisation labels)
  EAT / S6 caches eat_backing + eat_layers, sslam_pt_backing + sslam_layers, wavlm_voice + wavlm_layers (f16 as stored there)
  python dump_features.py PKG MANIFEST OUT [--shard I/N] [--device cuda:0]      MANIFEST: csv with columns ID, path
  workers share OUT: each claims a row with an exclusive ID.lock (delete stale *.lock before restarting killed workers)
Resumable: rows whose npz exists are skipped; every npz is written as ID.tmp.npz and renamed."""
import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def mert_windows(e, S, backing, starts):
    """(windows, len(music_layers), 1024): the inline MERT loop of the S7 Engine.predict (per-window soxr_hq 16->24 kHz)."""
    import librosa
    out = []
    for offset in range(0, len(starts), e.batch):
        audio = [librosa.resample(x, orig_sr=16000, target_sr=24000, res_type='soxr_hq') for x in S.windows(backing, starts[offset:offset + e.batch])]
        values = e.processor(audio, sampling_rate=24000, return_tensors='pt', padding=True).input_values.to(e.device)
        hidden = e.mert(values, output_hidden_states=True).hidden_states
        out.append(torch.stack([hidden[k].mean(1).float() for k in e.music_layers], 1))
    return torch.cat(out)


@torch.inference_mode()
def features(e, S, path: Path) -> dict:
    """Engine.predict (preserve_gain=False) up to the heads: the same load, peak cap, separation, windows and encoders."""
    import soundfile as sf
    raw = e.base.load_audio(path)
    peak = float(np.abs(raw).max())
    if peak > .6:
        raw = (raw * (.6 / peak)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='dump_features_') as temporary:
        normalized = Path(temporary) / 'normalized.wav'
        sf.write(normalized, raw, 16000, subtype='FLOAT')
        voice, backing = e.base.separate_voice_and_music(normalized, e.demucs, e.device)
    voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice))))
    backing = np.pad(backing[:len(raw)], (0, max(0, len(raw) - len(backing))))
    starts = S.window_layout(len(raw))
    n = len(starts)
    out = {'samples': np.int64(len(raw)), 'window_truth': np.full(n, np.nan, np.float32),
           'layers': np.array(e.music_layers, np.int64),
           'mert_layers': mert_windows(e, S, backing, starts).half().cpu().numpy()}
    for (model, blocks, fp16), (key, layer_key) in zip(e.eat_family, (('eat_backing', 'eat_layers'), ('sslam_pt_backing', 'sslam_layers'))):
        chunks = S.windows(S.fp16_round(backing) if fp16 else backing, starts)
        out[key] = e.eat_features(model, blocks, chunks).reshape(n, len(blocks), -1).half().cpu().numpy()
        out[layer_key] = np.array(blocks, np.int64)
    out['wavlm_voice'] = e.wavlm_features(S.windows(S.fp16_round(voice), starts))[:, None].half().cpu().numpy()
    out['wavlm_layers'] = np.array([e.wavlm_layer], np.int64)
    embeddings = []
    for start in e.base.get_segment_starts(len(raw)):
        e.df(input_values=torch.from_numpy(e.base.extract_segment(voice, start)).to(e.device))
        embeddings.append(e.embedding)
    out['voice_embeddings'] = torch.cat(embeddings).float().cpu().numpy()
    out['ad_embeddings'] = np.asarray(e.detector.windows(voice)[0]).astype(np.float16)
    out['voice_silent'] = np.bool_(e.base.calculate_rms(voice) < e.base.SILENCE_RMS)
    out['presence'] = presence(e, S, raw, voice)
    return out


def presence(e, S, raw, voice) -> np.ndarray:
    """(VOICE_PRESENT_PROB, MUSIC_PRESENT_PROB) exactly as Engine.predict (PANNs segments + packaged presence heads)."""
    dh = S.decision_heads
    probs = e.panns.inference(e.base.make_panns_segments(raw))[0]
    legacy = dh.legacy_presence(probs, e.panns_labels, e.label_groups)
    feats = dh.presence_features(probs, e.panns_labels, e.label_groups, raw, voice) if e.presence_heads else None
    return np.array(dh.presence(e.presence_heads, feats, legacy), np.float32)


@torch.inference_mode()
def backfill(e, S, path: Path) -> np.ndarray:
    """presence only, for npz written before the presence key existed (same load, peak cap and separation)."""
    import soundfile as sf
    raw = e.base.load_audio(path)
    peak = float(np.abs(raw).max())
    if peak > .6:
        raw = (raw * (.6 / peak)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='dump_features_') as temporary:
        normalized = Path(temporary) / 'normalized.wav'
        sf.write(normalized, raw, 16000, subtype='FLOAT')
        voice, _ = e.base.separate_voice_and_music(normalized, e.demucs, e.device)
    voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice))))
    return presence(e, S, raw, voice)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('pkg', type=Path)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--backfill', action='store_true', help='add the presence key to existing npz that lack it')
    a = ap.parse_args()
    sys.path.insert(0, str(a.pkg / 'model/runtime'))
    import full_bank_submission as S   # also exposes window_layout, windows, fp16_round
    decoder = a.pkg / 'model/runtime/bin/ffmpeg'
    if sys.platform == 'linux' and decoder.exists():   # the packaged decoder (as run()); elsewhere the system ffmpeg
        decoder.chmod(decoder.stat().st_mode | 0o111)
        os.environ['PATH'] = str(decoder.parent) + os.pathsep + os.environ.get('PATH', '')
    torch.set_num_threads(4)
    tf32 = os.environ.get('DUMP_TF32') == '1'   # speed option; the parity check (parity.py) decides whether it may be used
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    i, n = map(int, a.shard.split('/'))
    rows = list(csv.DictReader(a.manifest.open()))
    rows = rows[len(rows) * i // n:] + rows[:len(rows) * i // n]   # worker i starts at its own offset and claims rows (ID.lock)
    a.out.mkdir(parents=True, exist_ok=True)
    def needs(r):
        f = a.out / f"{r['ID']}.npz"
        if not a.backfill:
            return not f.exists()
        if not f.exists():
            return False
        with np.load(f) as d:
            return 'presence' not in d.files
    todo = [r for r in rows if needs(r)]
    print(f'worker {a.shard}: {len(rows)} rows, {len(todo)} without npz', flush=True)
    if not todo:
        return
    e = S.Engine(a.pkg / 'model/assets', a.pkg / 'model/trained', a.device)
    assert e.detector is not None and e.wavlm is not None and len(e.eat_family) == 2, 'S7 package expected (AD, WavLM, EAT, SSLAM)'
    started, done, failed, oom = time.time(), 0, [], 0
    for r in todo:
        target, lock = a.out / f"{r['ID']}.npz", a.out / f"{r['ID']}.{'plock' if a.backfill else 'lock'}"
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))   # another worker has it (or a killed one had it: rerun with fresh locks)
        except FileExistsError:
            continue
        if target.exists() != a.backfill:
            continue
        try:
            if a.backfill:
                with np.load(target) as d:
                    f = {k: d[k] for k in d.files}
                f['presence'] = backfill(e, S, Path(r['path']))
            else:
                f = features(e, S, Path(r['path']))
            tmp = a.out / f"{r['ID']}.tmp.npz"
            np.savez(tmp, **f)
            os.replace(tmp, target)
        except Exception as x:   # a bad file must not stop the worker; failures are listed and the rows left out
            if 'out of memory' in str(x).lower():   # GPU shared with other workers: release the row for a retry, back off
                torch.cuda.empty_cache()
                os.remove(lock)
                oom += 1
                if oom > 30:
                    print('too many CUDA OOM, worker stops', flush=True)
                    break
                time.sleep(10)
                continue
            failed.append({'ID': r['ID'], 'path': r['path'], 'error': f'{type(x).__name__}: {x}'[:300]})
        done += 1
        if done % 50 == 0 and str(e.device).startswith('cuda'):
            torch.cuda.empty_cache()
        if done % 100 == 0:
            print(f'{done} {done / (time.time() - started):.2f} rows/s failed {len(failed)} oom {oom}', flush=True)
            (a.out / f'_failed_{i}of{n}.json').write_text(json.dumps(failed, indent=1))
    print(f'finished {done} rows, failed {len(failed)}', flush=True)
    (a.out / f'_failed_{i}of{n}.json').write_text(json.dumps(failed, indent=1))

if __name__ == '__main__':
    main()
