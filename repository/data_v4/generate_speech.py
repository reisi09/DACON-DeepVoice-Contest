"""Korean fake speech from open TTS/VC models, paired with Zeroth-Korean real speech.

Run inside the backend's own venv (see envs.sh), on the VESSL GPU workspace:
  /root/envs/xtts/bin/python data_v4/generate_speech.py --backend xtts \
      --zeroth /root/data_v4/raw/zeroth_korean/data --out /root/data_v4/gen/speech --per-speaker 8

Lineage rule (keeps split components small, DESIGN.md §5):
  TTS: speaker S's voice reads S's *other* transcripts  -> links only speaker S.
  VC : fixed speaker pairs (A<->B); A's utterance converted to B's voice -> links {A, B} only.
Rows are appended to <out>/<backend>/rows.jsonl; finished IDs are skipped on restart.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

LABELS = dict(VOICE_PRESENT=1, MUSIC_PRESENT=0, VOICE_FAKE=1, MUSIC_FAKE=None, FILE_FAKE=1)
LICENSES = {  # model weights / code license, recorded per generated row
    'xtts': ('coqui/XTTS-v2', 'CPML (non-commercial)'),
    'chatterbox': ('ResembleAI/chatterbox multilingual', 'MIT'),
    'cosyvoice2': ('FunAudioLLM/CosyVoice2-0.5B', 'Apache-2.0'),
    'melo': ('myshell-ai/MeloTTS-Korean', 'MIT'),
    'seedvc': ('Plachtaa/seed-vc', 'GPL-3.0 code; weights per repo card'),
}


def load_zeroth(root: Path) -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(root.glob('*.parquet')):
        t = pq.read_table(f).to_pylist()
        missing = {'speaker_id', 'text', 'audio'} - set(t[0])
        if missing:
            raise SystemExit(f'{f}: Zeroth parquet lacks {missing}; inspect schema before generating')
        for r in t:
            rows.append(dict(uid=str(r.get('id') or r['audio'].get('path')), speaker=str(r['speaker_id']),
                             text=r['text'].strip(), audio=r['audio']['bytes'], split_file=f.name))
    return rows


def plan_jobs(rows: list[dict], backend: str, per_speaker: int, seed: int) -> list[dict]:
    """Deterministic job list. Generation randomness only; splits are decided later from lineage."""
    rng = random.Random(f'{seed}|{backend}')
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        by_spk.setdefault(r['speaker'], []).append(r)
    speakers = sorted(s for s, u in by_spk.items() if len(u) >= 4)
    jobs = []
    if backend == 'seedvc':
        for a, b in zip(speakers[0::2], speakers[1::2]):        # fixed pairs -> components of 2 speakers
            for src_spk, tgt_spk in ((a, b), (b, a)):
                src = rng.sample(by_spk[src_spk], min(per_speaker, len(by_spk[src_spk])))
                ref = by_spk[tgt_spk][0]
                for u in src:
                    jobs.append(dict(source=u, ref=ref, text=None, speakers=[src_spk, tgt_spk]))
        return jobs
    for s in speakers:
        utts = by_spk[s][:]
        rng.shuffle(utts)
        ref, pool = utts[0], utts[1:]
        for i in range(min(per_speaker, len(pool))):
            k = rng.choice([1, 1, 2, 3, 5])                     # 1 sentence ~5 s ... 5 sentences ~30 s
            texts = [pool[(i + j) % len(pool)] for j in range(k)]
            jobs.append(dict(source=None, ref=ref, text=' '.join(t['text'] for t in texts),
                             text_ids=[t['uid'] for t in texts], speakers=[s]))
    return jobs


class Backend:
    def __init__(self, name: str, device: str, work: Path):
        self.name, self.work = name, work
        if name == 'xtts':
            from TTS.api import TTS                                   # needs COQUI_TOS_AGREED=1 set by operator
            self.m = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
        elif name == 'chatterbox':
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self.m = ChatterboxMultilingualTTS.from_pretrained(device=device)
        elif name == 'cosyvoice2':
            from cosyvoice.cli.cosyvoice import CosyVoice2            # repo on PYTHONPATH (envs.sh)
            self.m = CosyVoice2(str(work / 'CosyVoice2-0.5B'), load_jit=False, load_trt=False, fp16=True)
        elif name == 'melo':
            from melo.api import TTS
            self.m = TTS(language='KR', device=device)
            self.spk = self.m.hps.data.spk2id['KR']
        elif name == 'seedvc':
            import subprocess  # seed-vc ships a CLI; model download happens on first call
            self.run = subprocess.run
        else:
            raise SystemExit(f'unknown backend {name}')

    def synth(self, job: dict, ref_wav: Path, src_wav: Path | None, out: Path) -> None:
        import soundfile as sf
        if self.name == 'xtts':
            self.m.tts_to_file(text=job['text'], speaker_wav=str(ref_wav), language='ko', file_path=str(out))
        elif self.name == 'chatterbox':
            wav = self.m.generate(job['text'], language_id='ko', audio_prompt_path=str(ref_wav))
            sf.write(out, wav.squeeze(0).cpu().numpy(), self.m.sr)
        elif self.name == 'cosyvoice2':
            import torch
            from cosyvoice.utils.file_utils import load_wav
            parts = [x['tts_speech'] for x in self.m.inference_zero_shot(
                job['text'], job['ref']['text'], load_wav(str(ref_wav), 16000), stream=False)]
            sf.write(out, torch.cat(parts, 1).squeeze(0).numpy(), self.m.sample_rate)
        elif self.name == 'melo':
            self.m.tts_to_file(job['text'], self.spk, str(out), speed=random.uniform(0.9, 1.15))
        elif self.name == 'seedvc':
            tmp = out.parent / (out.stem + '_vc')
            self.run([sys.executable, str(self.work / 'seed-vc/inference.py'), '--source', str(src_wav),
                      '--target', str(ref_wav), '--output', str(tmp), '--diffusion-steps', '30',
                      '--fp16', 'True'], check=True, cwd=self.work / 'seed-vc')
            produced = sorted(tmp.glob('*.wav'))
            if len(produced) != 1:
                raise RuntimeError(f'seed-vc produced {len(produced)} files')
            produced[0].rename(out)
            tmp.rmdir()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=sorted(LICENSES))
    ap.add_argument('--zeroth', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--work', type=Path, default=Path('/root/gen_models'))
    ap.add_argument('--per-speaker', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--deadline-epoch', type=float, default=0, help='stop cleanly at this time (GPU budget)')
    a = ap.parse_args()

    rows = load_zeroth(a.zeroth)
    jobs = plan_jobs(rows, a.backend, a.per_speaker, a.seed)
    out = a.out / a.backend
    (out / 'wav').mkdir(parents=True, exist_ok=True)
    (out / 'ref').mkdir(exist_ok=True)
    ledger = out / 'rows.jsonl'
    done = {json.loads(l)['ID'] for l in ledger.open()} if ledger.exists() else set()
    model = Backend(a.backend, a.device, a.work)
    repo, lic = LICENSES[a.backend]
    failures = 0
    with ledger.open('a') as log:
        for job in jobs:
            if a.deadline_epoch and time.time() > a.deadline_epoch:
                print('deadline reached; stopping cleanly', flush=True)
                break
            key = json.dumps([a.backend, job['speakers'], job.get('text_ids'), job['source'] and job['source']['uid']])
            rid = f'gen_{a.backend}_' + hashlib.sha256(key.encode()).hexdigest()[:16]
            if rid in done:
                continue
            ref_wav = out / 'ref' / f"{hashlib.sha256(job['ref']['uid'].encode()).hexdigest()[:16]}.flac"
            if not ref_wav.exists():
                ref_wav.write_bytes(job['ref']['audio'])  # Zeroth audio bytes are FLAC
            src_wav = None
            if job['source']:
                src_wav = out / 'ref' / f"src_{hashlib.sha256(job['source']['uid'].encode()).hexdigest()[:16]}.flac"
                src_wav.write_bytes(job['source']['audio'])
            path = out / 'wav' / f'{rid}.wav'
            try:
                model.synth(job, ref_wav, src_wav, path)
            except Exception as exc:  # one bad text must not kill a paid GPU run
                failures += 1
                print(f'fail {rid}: {type(exc).__name__}: {exc}'[:300], flush=True)
                if failures > 50 and failures > len(done) // 5:
                    raise SystemExit('failure rate too high; fix backend before spending more GPU')
                continue
            parents = [job['ref']['uid']] + (job.get('text_ids') or []) + ([job['source']['uid']] if job['source'] else [])
            row = dict(ID=rid, path=str(path.relative_to(a.out)), source='generated_speech', subset=a.backend,
                       generator=a.backend, generator_repo=repo, license=lic, language='ko',
                       speaker_ids=[f'zeroth:{s}' for s in job['speakers']], parent_ids=[f'zeroth:{p}' for p in parents],
                       text=job['text'], kind='vc' if job['source'] else 'tts', label_evidence='construction', **LABELS)
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            done.add(rid)
    print(json.dumps({'backend': a.backend, 'jobs': len(jobs), 'done': len(done), 'failures': failures}))


def self_check() -> None:
    rows = [dict(uid=f'{s}_{i}', speaker=str(s), text=f't{s}{i}', audio=b'') for s in range(4) for i in range(6)]
    tts = plan_jobs(rows, 'xtts', 3, 1)
    assert all(len(j['speakers']) == 1 for j in tts)
    assert all(j['ref']['speaker'] == j['speakers'][0] and all(t.startswith(j['speakers'][0] + '_') for t in j['text_ids']) for j in tts)
    assert all(j['ref']['uid'] not in j['text_ids'] for j in tts), 'reference utterance must not be read back'
    vc = plan_jobs(rows, 'seedvc', 2, 1)
    assert {tuple(sorted(j['speakers'])) for j in vc} == {('0', '1'), ('2', '3')}, 'VC must stay within fixed pairs'
    assert plan_jobs(rows, 'xtts', 3, 1) == tts, 'job plan must be reproducible'
    print('generate_speech self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
