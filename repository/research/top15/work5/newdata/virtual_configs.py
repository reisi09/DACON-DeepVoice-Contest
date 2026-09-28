"""Prediction-level candidates for the integrated model (CPU only): every member of a virtual config is built from the
every-row predictions of already-trained DB-protocol configs, so new_eval.py scores it with the same tables / FILE rule.
  ens:A+B      each member = equal-weight mean over the seeds of A and B (6 predictions)
  mix:VA/MB    voice members (df, ad, wavlm) from config VA, music member from config MB
  python virtual_configs.py RUNS OUT SPEC [SPEC ...]      e.g. ens:s7_new+W_newB  mix:s7_new/W_newB"""
import json
import sys
from pathlib import Path

FILES = {'df': 'voice_df__L.json', 'ad': 'voice_ad__L.json', 'wavlm': 'voice_new__wavlm8.json', 'music': 'music_new__mert6_12+eat16+sslampt5.json'}


def combine(parts):
    """Per split, rows common to every run (aligned by ID; W_newB's val also holds newval rows), per-seed prediction
    lists concatenated -> new_eval's mean over axis 0 is the equal-weight mean over all seeds of all runs."""
    base = json.loads(json.dumps(parts[0]))
    for split, e in base['every'].items():
        others = [set(p['every'][split]['ids']) for p in parts[1:]]
        ids = [i for i in e['ids'] if all(i in s for s in others)]
        preds, silent = [], dict(zip(e['ids'], e['silent']))
        for p in parts:
            pos = {i: k for k, i in enumerate(p['every'][split]['ids'])}
            preds += [[seed[pos[i]] for i in ids] for seed in p['every'][split]['preds']]
        e.update(ids=ids, silent=[silent[i] for i in ids], preds=preds)
    return base


def main():
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    for spec in sys.argv[3:]:
        kind, body = spec.split(':', 1)
        name = spec.replace(':', '_').replace('+', '_').replace('/', '_')
        (out / name).mkdir(parents=True, exist_ok=True)
        for member, f in FILES.items():
            if kind == 'ens':
                srcs = body.split('+')
            else:
                voice, music = body.split('/')
                srcs = [music if member == 'music' else voice]
            parts = [json.loads((runs / s / f).read_text()) for s in srcs]
            (out / name / f).write_text(json.dumps(combine(parts)))
        print(name, 'written', flush=True)


if __name__ == '__main__':
    main()
