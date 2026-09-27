"""R1 (c4): 3-seed grid, voice pooling {max, lme, mean} x music pooling {lme, mean}; every head trained with the
pooling it is scored with (CPU retrains; max/lme controls reproduce the DB ensemble). FILE = val-refit ensemble.
  python eval_grid.py -> grid.json, b_grid_<cfg>.json"""
import json
import numpy as np, pandas as pd
import agg

VA = {'max': 'max', 'lme': 'lme', 'mean': 'mean_logit'}


def seeds(route, p):
    pre = f'retrain_{p}' if route == 'voice' else f'retrain_music_{p}'
    k = route[0]
    zs = [np.load(agg.OUT / d / f'windows_{route}.npz') for d in (pre, pre + '_s42', pre + '_s43')]
    return np.split(np.concatenate([z[k + 'z'] for z in zs], 1), zs[0][k + 'o'][1:-1])


def main():
    b, _, _ = agg.load()
    m = agg.masks(b)
    val, test = (b.split == 'val').values, (b.split == 'test').values
    fk = m['file'].values
    V = {p: seeds('voice', p) for p in VA}
    M = {p: seeds('music', p) for p in ('lme', 'mean')}
    res, base = {}, {}
    for vp in VA:
        for mp in M:
            name = f'voice_{vp}|music_{mp}'
            fv, fm = agg.score(V[vp], M[mp], b.silent.values, VA[vp], mp)
            ens, rep = agg.fit(agg.files(fv[val & fk], fm[val & fk], b[val & fk]), b.FILE_FAKE[val & fk].astype(int).values)
            sc = {'file': agg.file_vec(ens, fv, fm, b.VOICE_PRESENT_PROB.values, b.MUSIC_PRESENT_PROB.values), 'voice': fv, 'music': fm}
            r = {'file_weights': rep['chosen_weights']}
            for s, sm in (('val', val), ('test', test)):
                e = {k: round(agg.eer(b.loc[sm & m[k], k.upper() + '_FAKE'].astype(int).values, sc[k][sm & m[k].values]), 5) for k in m}
                e['ADS'] = round(.5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), 5)
                r[s] = e
            for k, lab in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE')):
                mm = test & m[k].values
                for cls in ('0', '1'):
                    c = mm & (b[lab] == cls).values
                    r.setdefault('len_rho_within_source_test', {})[f'{k}_{"real" if cls == "0" else "fake"}'] = \
                        agg.within_source_rho(sc[k][c], b.source.values[c], b.dur.values[c])
                y, s_ = b[lab].values[mm].astype(int), sc[k][mm]
                thr, _ = agg.eer_thr(y, s_)
                bins = pd.cut(b.dur.values[mm], [0, 4, 8, 15, 30, 61], right=False, labels=['<4', '4-8', '8-15', '15-30', '30-60'])
                t = pd.DataFrame({'y': y, 'hit': s_ >= thr, 'bin': bins})
                r.setdefault('err_by_duration_test', {})[k] = {
                    'fpr_real': {str(q): [round(float(g.hit.mean()), 3), len(g)] for q, g in t[t.y == 0].groupby('bin', observed=True)},
                    'fnr_fake': {str(q): [round(float(1 - g.hit.mean()), 3), len(g)] for q, g in t[t.y == 1].groupby('bin', observed=True)}}
                if base:
                    for s, sm in (('test', test), ('val', val)):
                        mk = sm & m[k].values
                        r.setdefault(f'boot_delta_{s}_vs_control', {})[k] = agg.boot_delta(
                            b[lab].values[mk].astype(int), base[k][mk], sc[k][mk], b.group_id.values[mk])
            if not base:
                base = sc   # control = voice max | music lme (current DB design)
            res[name] = r
            print(name, 'val', r['val'], 'test', r['test'], r['file_weights'], flush=True)
    json.dump(res, open(agg.OUT / 'grid.json', 'w'), indent=1)
    for vp, mp in (('max', 'lme'), ('max', 'mean'), ('lme', 'mean'), ('lme', 'lme')):
        agg.part_b(vz=V[vp], mz=M[mp], magg=mp, tasks=[('voice', 'VOICE_FAKE', VA[vp]), ('file', 'FILE_FAKE', VA[vp])],
                   out_name=f'b_grid_voice_{vp}_music_{mp}.json')


if __name__ == '__main__':
    main()
