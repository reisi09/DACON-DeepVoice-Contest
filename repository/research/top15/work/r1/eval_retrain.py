"""R1 (c) evaluation: retrained 1-seed voice heads (retrain_<pool>/windows_voice.npz) through the (a) and (b) protocols.
Music Fm = DB 3-seed LME (unchanged); FILE = val-refit ensemble (file_ensemble.fit); presence = DB decision columns.
  python eval_retrain.py max lme mean   -> c.json, b_retrain_<pool>.json"""
import json, sys
import numpy as np
import agg

POOL2AGG = {'max': 'max', 'lme': 'lme', 'mean': 'mean_logit'}


def main(pools):
    b, _, mz = agg.load()
    m = agg.masks(b)
    val, test = (b.split == 'val').values, (b.split == 'test').values
    fk = m['file'].values
    vzs = {}
    for p in pools:
        w = np.load(agg.OUT / f'retrain_{p}/windows_voice.npz')
        vzs[p] = np.split(w['vz'], w['vo'][1:-1])
    res, base = {}, None
    for p in pools:
        h = json.load(open(agg.OUT / f'retrain_{p}/history.json'))
        for inf in dict.fromkeys([POOL2AGG[p], 'max', 'lme', 'mean_logit']):
            fv, fm = agg.score(vzs[p], mz, b.silent.values, inf, 'lme')
            ens, rep = agg.fit(agg.files(fv[val & fk], fm[val & fk], b[val & fk]), b.FILE_FAKE[val & fk].astype(int).values)
            sc = {'file': agg.file_vec(ens, fv, fm, b.VOICE_PRESENT_PROB.values, b.MUSIC_PRESENT_PROB.values), 'voice': fv, 'music': fm}
            r = {'best_epoch': h['best_epoch'], 'epochs_run': len(h['history']), 'file_weights': rep['chosen_weights']}
            for s, sm in (('val', val), ('test', test)):
                e = {k: round(agg.eer(b.loc[sm & m[k], k.upper() + '_FAKE'].astype(int).values, sc[k][sm & m[k].values]), 5) for k in m}
                e['ADS'] = round(.5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), 5)
                r[s] = e
            for k, lab in (('voice', 'VOICE_FAKE'), ('file', 'FILE_FAKE')):
                for cls in ('0', '1'):
                    c = test & m[k].values & (b[lab] == cls).values
                    r.setdefault('len_rho_within_source_test', {})[f'{k}_{"real" if cls == "0" else "fake"}'] = \
                        agg.within_source_rho(sc[k][c], b.source.values[c], b.dur.values[c])
            if base is None:
                base = sc   # control: max-trained, max-inference
            else:
                for k, lab in (('voice', 'VOICE_FAKE'), ('file', 'FILE_FAKE')):
                    mm = test & m[k].values
                    r.setdefault('boot_delta_test_vs_max_control', {})[k] = agg.boot_delta(
                        b[lab].values[mm].astype(int), base[k][mm], sc[k][mm], b.group_id.values[mm])
            res[f'train={p}|infer={inf}'] = r
            print(p, inf, r['val'], r['test'], r['len_rho_within_source_test'], r.get('boot_delta_test_vs_max_control'), flush=True)
    json.dump(res, open(agg.OUT / 'c.json', 'w'), indent=1)
    for p in pools:   # synthetic length protocol, matched train/inference pooling
        a = POOL2AGG[p]
        agg._SETS.clear()
        agg.part_b(vz=vzs[p], tasks=[('voice', 'VOICE_FAKE', a), ('file', 'FILE_FAKE', a)], out_name=f'b_retrain_{p}.json')


if __name__ == '__main__':
    main(sys.argv[1:])
