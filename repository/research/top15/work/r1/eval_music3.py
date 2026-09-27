"""R1 (c3): 3-seed music heads, LME (matched control, reproduces DB seeds 42/43 exactly) vs mean pooling.
Voice = DB 3-seed max (unchanged). FILE = val-refit ensemble. -> music3.json, b_music3_<pool>.json"""
import json
import numpy as np
import agg

DIRS = {p: [f'retrain_music_{p}', f'retrain_music_{p}_s42', f'retrain_music_{p}_s43'] for p in ('lme', 'mean')}


def seeds(p):
    zs = [np.load(agg.OUT / d / 'windows_music.npz') for d in DIRS[p]]
    return np.split(np.concatenate([z['mz'] for z in zs], 1), zs[0]['mo'][1:-1])


def main():
    b, vz, mz_db = agg.load()
    m = agg.masks(b)
    val, test = (b.split == 'val').values, (b.split == 'test').values
    fk = m['file'].values
    M = {'DB': (mz_db, 'lme'), 'lme3': (seeds('lme'), 'lme'), 'mean3': (seeds('mean'), 'mean')}
    res, base = {}, None
    for name, (mz, ma) in M.items():
        fv, fm = agg.score(vz, mz, b.silent.values, 'max', ma)
        ens, rep = agg.fit(agg.files(fv[val & fk], fm[val & fk], b[val & fk]), b.FILE_FAKE[val & fk].astype(int).values)
        sc = {'file': agg.file_vec(ens, fv, fm, b.VOICE_PRESENT_PROB.values, b.MUSIC_PRESENT_PROB.values), 'voice': fv, 'music': fm}
        r = {'file_weights': rep['chosen_weights']}
        for s, sm in (('val', val), ('test', test)):
            e = {k: round(agg.eer(b.loc[sm & m[k], k.upper() + '_FAKE'].astype(int).values, sc[k][sm & m[k].values]), 5) for k in m}
            e['ADS'] = round(.5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), 5)
            r[s] = e
            # per-seed music EER (each seed alone)
            mm = sm & m['music'].values
            r[f'{s}_music_eer_per_seed'] = [round(agg.eer(b.MUSIC_FAKE.values[mm].astype(int),
                                             np.array([agg.MAGG[ma](z[:, [j]]).mean() for z, t in zip(mz, mm) if t])), 5) for j in range(3)]
        for k, lab in (('music', 'MUSIC_FAKE'), ('file', 'FILE_FAKE')):
            mm = test & m[k].values
            for cls in ('0', '1'):
                c = mm & (b[lab] == cls).values
                r.setdefault('len_rho_within_source_test', {})[f'{k}_{"real" if cls == "0" else "fake"}'] = \
                    agg.within_source_rho(sc[k][c], b.source.values[c], b.dur.values[c])
            if name == 'mean3':
                r.setdefault('boot_delta_test_vs_lme3', {})[k] = agg.boot_delta(b[lab].values[mm].astype(int), base[k][mm], sc[k][mm], b.group_id.values[mm])
                mv = val & m[k].values
                r.setdefault('boot_delta_val_vs_lme3', {})[k] = agg.boot_delta(b[lab].values[mv].astype(int), base_v[k][mv], sc[k][mv], b.group_id.values[mv])
        if name == 'lme3':
            base = base_v = sc
        res[name] = r
        print(name, r['val'], r['test'], 'seeds val', r['val_music_eer_per_seed'], 'test', r['test_music_eer_per_seed'],
              r['len_rho_within_source_test'], r.get('boot_delta_test_vs_lme3'), r.get('boot_delta_val_vs_lme3'), flush=True)
    json.dump(res, open(agg.OUT / 'music3.json', 'w'), indent=1)
    for name in ('lme3', 'mean3'):
        mz, ma = M[name]
        agg.part_b(vz=vz, mz=mz, magg=ma, tasks=[('music', 'MUSIC_FAKE', ma), ('file', 'FILE_FAKE', 'max')], out_name=f'b_music3_{name}.json')


if __name__ == '__main__':
    main()
