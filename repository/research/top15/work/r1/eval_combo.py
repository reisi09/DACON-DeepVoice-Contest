"""R1 (c2): matched 1-seed configurations (voice pooling x music pooling), each head trained with its pooling.
  python eval_combo.py -> combo.json, b_combo_<name>.json"""
import json
import numpy as np, pandas as pd
import agg


def split_npz(path, key):
    w = np.load(path)
    return np.split(w[key + 'z'], w[key + 'o'][1:-1])


def main():
    b, vz_db, mz_db = agg.load()
    m = agg.masks(b)
    val, test = (b.split == 'val').values, (b.split == 'test').values
    fk = m['file'].values
    V = {p: split_npz(agg.OUT / f'retrain_{p}/windows_voice.npz', 'v') for p in ('max', 'lme', 'mean')}
    M = {p: split_npz(agg.OUT / f'retrain_music_{p}/windows_music.npz', 'm') for p in ('lme', 'mean')}
    cfgs = {'DB_3seed': (vz_db, 'max', mz_db, 'lme'), 'ctrl_1seed': (V['max'], 'max', M['lme'], 'lme'),
            'music_mean': (V['max'], 'max', M['mean'], 'mean'), 'voice_mean': (V['mean'], 'mean_logit', M['lme'], 'lme'),
            'voice_lme': (V['lme'], 'lme', M['lme'], 'lme'), 'both_mean': (V['mean'], 'mean_logit', M['mean'], 'mean'),
            'voice_lme_music_mean': (V['lme'], 'lme', M['mean'], 'mean')}
    res, base = {}, None
    for name, (vz, va, mz, ma) in cfgs.items():
        fv, fm = agg.score(vz, mz, b.silent.values, va, ma)
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
            if name not in ('DB_3seed', 'ctrl_1seed'):
                r.setdefault('boot_delta_test_vs_ctrl', {})[k] = agg.boot_delta(y, base[k][mm], s_, b.group_id.values[mm])
        if name == 'ctrl_1seed':
            base = sc
        res[name] = r
        print(name, r['val'], r['test'], r['file_weights'], r.get('boot_delta_test_vs_ctrl'), flush=True)
    json.dump(res, open(agg.OUT / 'combo.json', 'w'), indent=1)
    for name in ('ctrl_1seed', 'music_mean', 'both_mean', 'voice_lme_music_mean'):
        vz, va, mz, ma = cfgs[name]
        tasks = [('music', 'MUSIC_FAKE', ma), ('file', 'FILE_FAKE', va)]
        agg.part_b(vz=vz, mz=mz, magg=ma, tasks=tasks, out_name=f'b_combo_{name}.json')


if __name__ == '__main__':
    main()
