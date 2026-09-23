"""AI songs (vocal+accompaniment) and instrumentals with ACE-Step, Korean-first.

  /root/envs/acestep/bin/python data_v4/generate_music.py --out /root/data_v4/gen/music --songs 300 --instrumentals 200
Lyrics below are original text written for this project (no third-party lyrics).
Vocal presence is a construction label; build_manifest.py re-checks it with the separator
and masks VOICE_* when the stem disagrees (DESIGN.md §4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

GENRES = ['k-pop, dance, synth, female vocal', 'ballad, piano, strings, male vocal', 'indie rock, guitar, drums',
          'hip hop, boom bap, rap', 'r&b, soul, smooth', 'trot, accordion, festive', 'city pop, bass, 80s',
          'acoustic folk, guitar, warm', 'edm, house, energetic', 'jazz, lounge, saxophone', 'lo-fi, chill, mellow',
          'rock ballad, electric guitar', 'children song, bright, simple', 'ost, orchestral, emotional',
          'reggae, offbeat, relaxed', 'metal, distorted guitar, fast']
INSTRUMENTAL = ['ambient pad, slow', 'classical piano solo', 'string quartet', 'cafe bossa nova', 'cinematic trailer',
                'lo-fi beat, vinyl', 'techno, 128 bpm', 'marching band', 'traditional korean gayageum', 'funk groove, bass',
                'hold music, soft synth', 'retail background music, upbeat']
KO = ['아침 햇살 창가에 번지면\n너의 목소리가 떠올라\n조용한 거리 위를 걸으며\n오늘도 너를 불러',
      '바람이 불어오는 저녁\n우리의 약속을 기억해\n멀어진 시간 속에서도\n나는 여기 있을게',
      '버스 창밖으로 흐르는 불빛\n이어폰 속 우리 노래\n내일은 조금 더 가까이\n한 걸음 더 다가가',
      '비 내리는 골목 끝에서\n우산 하나 나눠 쓰던 날\n그 온기가 아직 남아서\n나는 웃고 있어',
      '달리자 끝까지 달려\n숨이 차도 멈추지 마\n우리의 노래가 울리면\n세상이 춤을 춰',
      '작은 방 안의 꿈들이\n별처럼 반짝이는 밤\n눈을 감고 소원을 빌어\n내일은 빛날 거야',
      '시장 골목 따뜻한 국물\n할머니의 웃음소리\n고향의 냄새가 그리워\n오늘은 전화를 걸어',
      '파도 소리 들리는 바다\n모래 위에 쓴 이름\n지워져도 괜찮아\n마음에 새겨 두었어',
      '월요일 아침 알람 소리\n커피 한 잔에 힘을 내\n지친 하루 끝에는\n네가 기다려 줘',
      '눈 내리는 겨울 거리\n하얀 숨결이 피어나\n손을 잡고 걸어가면\n추위도 잊혀져']
EN = ['city lights are calling out my name\nevery road leads back to you again',
      'we were young and the summer was long\nsinging loud to our favorite song',
      'hold on tight when the night gets cold\nstories written that were never told']


def plan(songs: int, instrumentals: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    jobs = []
    for i in range(songs):
        ko = rng.random() < 0.75
        lines = rng.sample(KO if ko else EN, 2 if ko else 2)
        jobs.append(dict(kind='song', prompt=rng.choice(GENRES), language='ko' if ko else 'en',
                         lyrics='[verse]\n' + lines[0] + '\n\n[chorus]\n' + lines[1],
                         duration=rng.choice([15, 20, 30, 45, 60]), seed=rng.randrange(2**31)))
    for i in range(instrumentals):
        jobs.append(dict(kind='instrumental', prompt=rng.choice(INSTRUMENTAL + GENRES) + ', instrumental, no vocals',
                         language=None, lyrics='[instrumental]', duration=rng.choice([10, 20, 30, 45, 60]),
                         seed=rng.randrange(2**31)))
    return jobs


def labels(kind: str) -> dict:
    if kind == 'song':
        return dict(VOICE_PRESENT=1, MUSIC_PRESENT=1, VOICE_FAKE=1, MUSIC_FAKE=1, FILE_FAKE=1)
    return dict(VOICE_PRESENT=0, MUSIC_PRESENT=1, VOICE_FAKE=None, MUSIC_FAKE=1, FILE_FAKE=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--songs', type=int, default=300)
    ap.add_argument('--instrumentals', type=int, default=200)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--deadline-epoch', type=float, default=0)
    a = ap.parse_args()
    from acestep.pipeline_ace_step import ACEStepPipeline   # checkpoint downloads on first use (Apache-2.0)
    pipe = ACEStepPipeline(dtype='bfloat16', torch_compile=False)
    out = a.out / 'acestep'
    (out / 'wav').mkdir(parents=True, exist_ok=True)
    ledger = out / 'rows.jsonl'
    done = {json.loads(l)['ID'] for l in ledger.open()} if ledger.exists() else set()
    with ledger.open('a') as log:
        for job in plan(a.songs, a.instrumentals, a.seed):
            if a.deadline_epoch and time.time() > a.deadline_epoch:
                print('deadline reached; stopping cleanly', flush=True)
                break
            rid = 'gen_acestep_' + hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]
            if rid in done:
                continue
            path = out / 'wav' / f'{rid}.wav'
            try:
                pipe(audio_duration=job['duration'], prompt=job['prompt'], lyrics=job['lyrics'],
                     infer_step=a.steps, guidance_scale=15, manual_seeds=[job['seed']], save_path=str(path))
            except Exception as exc:
                print(f'fail {rid}: {type(exc).__name__}: {exc}'[:300], flush=True)
                continue
            # prompt group = genre text: songs sharing a prompt are one split component (DESIGN.md §5)
            row = dict(ID=rid, path=str(path.relative_to(a.out)), source='generated_music', subset='acestep',
                       generator='acestep', generator_repo='ace-step/ACE-Step', license='Apache-2.0 (model); lyrics original',
                       kind=job['kind'], language=job['language'], prompt=job['prompt'], lyrics=job['lyrics'],
                       group_keys=[f"prompt:{job['prompt']}"], label_evidence='construction', **labels(job['kind']))
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            done.add(rid)
    print(json.dumps({'done': len(done)}))


def self_check() -> None:
    jobs = plan(20, 10, 1)
    assert plan(20, 10, 1) == jobs
    assert sum(j['kind'] == 'song' for j in jobs) == 20 and all(j['lyrics'] == '[instrumental]' for j in jobs[20:])
    assert labels('instrumental')['VOICE_FAKE'] is None and labels('song')['FILE_FAKE'] == 1
    print('generate_music self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
