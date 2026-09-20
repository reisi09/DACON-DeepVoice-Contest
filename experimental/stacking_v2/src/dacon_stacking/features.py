"""Current-head CPU replay and finite, explicitly ordered small feature sets.
Metadata/labels are never concatenated into the estimator's input DataFrame.
"""
from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import pandas as pd
import torch
from .common import ContractError, check_hash, contained, digest, finite, read_json, sha256, write_json
from . import _snapshot_math as math
from ._snapshot_scoring import PREDICTION_COLUMNS

F0 = ['Pv', 'Pm', 'Fv', 'Fm']
STATS = ['mean','max','std','q25','q50','q90','top3mean']
F1 = F0 + [f'{b}_seed{s}' for b in ('Fv','Fm') for s in math.SEEDS]
F1 += [f'{b}_seed_{s}' for b in ('Fv','Fm') for s in ('var','std','range')]
F1 += ['df_original'] + [f'{b}_window_{s}' for b in ('voice','music') for s in STATS] + ['df_minus_Fv']
F2 = F1 + [f'{b}_clip_{s}' for b in ('Pv','Pm') for s in ('mean','std','q90')]
F2 += ['pann_entropy_mean', 'pann_entropy_std']
F2 += [f'{b}_aligned_{s}' for b in ('voice','music') for s in ('mean','max','coverage','longest_fraction')]
F2R = F2 + ['raw_voice_minus_Fv','raw_music_minus_Fm','voice_raw_energy_ratio','music_raw_energy_ratio']
SHORTCUT = ['log_duration','source_peak','normalization_gain','normalized_rms','voice_silent','music_padding_fraction']
FEATURE_SETS = {'F0':F0, 'F1':F1, 'F2':F2, 'F2R':F2R, 'F1S':F1+SHORTCUT}
ALLOWLIST = frozenset(sum(FEATURE_SETS.values(), []))
SCHEMA = 'dacon.file_stacking.features.v1'


def strict_frame(frame: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    if not order or len(set(order)) != len(order) or not set(order) <= ALLOWLIST:
        raise ContractError('Feature allowlist violation')
    if list(frame.columns) != list(order):
        raise ContractError('Feature order/columns mismatch (no silent reordering or extra columns)')
    if frame.empty:
        raise ContractError('Empty feature frame')
    a = finite(frame.to_numpy(), 'features')
    return pd.DataFrame(a, columns=order, index=frame.index)


def feature_frame(records: list[dict], set_name: str) -> pd.DataFrame:
    order = FEATURE_SETS[set_name]
    for record in records:
        absent = set(order) - record['features'].keys()
        if absent:
            raise ContractError(f'Required measured features absent: {sorted(absent)}')
    return strict_frame(pd.DataFrame([[r['features'][n] for n in order] for r in records], columns=order), order)


def time_layout(samples: int, branch: str) -> dict:
    if samples <= 0 or branch not in ('music','voice','pann'):
        raise ContractError('Invalid sample count/timebase branch')
    starts = math.window_layout(samples).tolist() if branch == 'music' else math.get_segment_starts(samples)
    return {'sr':16000, 'unit':'samples', 'starts':[int(x) for x in starts],
            'valid_samples':[min(64600, samples-int(x)) for x in starts],
            'window_samples':64600, 'nominal_hop_samples':32300 if branch=='music' else 64600,
            'tail':'append exact final start when not on hop grid',
            'short_padding':'zero' if branch=='music' else 'repeat entire short signal'}


def _intervals(starts, lengths, scores, samples):
    starts, lengths = np.asarray(starts), np.asarray(lengths)
    scores = finite(scores, 'window probabilities', True)
    if starts.ndim != 1 or lengths.shape != starts.shape or scores.shape != starts.shape:
        raise ContractError('Timebase length mismatch')
    if (not np.issubdtype(starts.dtype,np.integer) or not np.issubdtype(lengths.dtype,np.integer)
        or np.any(lengths<=0) or np.any(starts<0) or np.any(starts+lengths>samples)
        or len(set(starts.tolist())) != len(starts)):
        raise ContractError('Invalid or padded-outside-valid time interval')
    return starts, starts+lengths, scores


def aligned_summary(p_starts, p_lengths, p_values, f_starts, f_lengths, f_values, samples: int, sr=16000) -> dict:
    """Integrate piecewise window means over actual overlap, not window indexes.
    At each elementary interval, average overlapping windows in each branch.
    Every real sample must be covered. Padding/repeated virtual audio is excluded.
    """
    if samples<=0 or sr<=0:
        raise ContractError('Invalid recording duration')
    ps,pe,pv = _intervals(p_starts,p_lengths,p_values,samples)
    fs,fe,fv = _intervals(f_starts,f_lengths,f_values,samples)
    edges = np.unique(np.concatenate(([0,samples],ps,pe,fs,fe)))
    values, weights = [], []
    longest = run = 0
    for start,end in zip(edges[:-1],edges[1:]):
        pi,fi = (ps<=start)&(pe>=end),(fs<=start)&(fe>=end)
        if not pi.any() or not fi.any():
            raise ContractError('uncovered real-time interval in aligned features')
        value=float(pv[pi].mean()*fv[fi].mean()); weight=int(end-start)
        values.append(value);weights.append(weight)
        run = run+weight if value>=.5 else 0; longest=max(longest,run)
    values=np.asarray(values);weights=np.asarray(weights)
    return {'mean':float(np.average(values,weights=weights)), 'max':float(values.max()),
            'coverage':float(weights[values>=.5].sum()/samples),'longest_fraction':float(longest/samples)}


def _summaries(values):
    a=finite(values,'window summary',True)
    if a.ndim!=1:
        raise ContractError('Summary expects 1-D probabilities')
    return dict(zip(STATS,[float(a.mean()),float(a.max()),float(a.std()),*np.quantile(a,[.25,.5,.9]).tolist(),float(np.sort(a)[-min(3,len(a)):].mean())],strict=True))


def from_measurements(baseline: Mapping, measurements: Mapping) -> dict:
    """Single-file adapter shared by CPU replay and GPU telemetry.
    All listed F1 measurements are mandatory; F2 is added only if present in full.
    """
    finite([baseline[n] for n in PREDICTION_COLUMNS],'baseline',True)
    result=dict(zip(F0,[baseline['VOICE_PRESENT_PROB'],baseline['MUSIC_PRESENT_PROB'],baseline['VOICE_FAKE_PROB'],baseline['MUSIC_FAKE_PROB']],strict=True))
    for b,name in [('Fv','voice'),('Fm','music')]:
        seed=finite(measurements[name+'_seed'],'seed',True)
        if seed.shape!=(3,): raise ContractError('Exactly seeds 41,42,43 required in order')
        result.update({f'{b}_seed{s}':float(v) for s,v in zip(math.SEEDS,seed,strict=True)})
        result.update({f'{b}_seed_var':float(seed.var()),f'{b}_seed_std':float(seed.std()),f'{b}_seed_range':float(np.ptp(seed))})
        window=finite(measurements[name+'_window'],'window',True)
        if window.ndim!=2 or window.shape[0]!=3:
            raise ContractError('Window probability matrix must be [3 seeds, windows]')
        result.update({f'{name}_window_{k}':v for k,v in _summaries(window.mean(0)).items()})
    result['df_original']=float(finite([measurements['df_original']],'original DF',True)[0])
    result['df_minus_Fv']=result['df_original']-result['Fv']
    if 'pann_classes' in measurements:
        probs=finite(measurements['pann_classes'],'PANN clip classes',True)
        pv=finite(measurements['pann_presence'],'PANN per clip',True)
        samples=int(measurements['samples']); v=time_layout(samples,'voice');m=time_layout(samples,'music')
        if probs.ndim!=2 or probs.shape!=(len(v['starts']),527) or pv.shape!=(len(v['starts']),2):
            raise ContractError('PANN is 527 clip-level classes on the voice grid, not music frames')
        for j,b in enumerate(('Pv','Pm')):
            for s,val in [('mean',pv[:,j].mean()),('std',pv[:,j].std()),('q90',np.quantile(pv[:,j],.9))]:
                result[f'{b}_clip_{s}']=float(val)
        # An uncertainty descriptor for multilabel scores, NOT categorical truth.
        sums=probs.sum(1,keepdims=True)
        if np.any(sums<=0): raise ContractError('Cannot normalize zero PANN class mass')
        q=probs/sums; entropy=-(q*np.log(np.maximum(q,1e-12))).sum(1)
        result['pann_entropy_mean']=float(entropy.mean());result['pann_entropy_std']=float(entropy.std())
        for j,b,layout in [(0,'voice',v),(1,'music',m)]:
            summary=aligned_summary(v['starts'],v['valid_samples'],pv[:,j],layout['starts'],layout['valid_samples'],np.asarray(measurements[b+'_window']).mean(0),samples)
            result.update({f'{b}_aligned_{k}':val for k,val in summary.items()})
        if 'raw_voice_fake' in measurements:
            rawv,rawm=finite([measurements['raw_voice_fake'],measurements['raw_music_fake']],'raw branch',True)
            result['raw_voice_minus_Fv']=float(rawv-result['Fv']);result['raw_music_minus_Fm']=float(rawm-result['Fm'])
            for n in ('voice_raw_energy_ratio','music_raw_energy_ratio'):
                val=float(finite([measurements[n]],n)[0])
                if val<0: raise ContractError('Negative energy ratio')
                result[n]=val
    if all(k in measurements for k in ('samples','source_peak','normalization_gain','normalized_rms','voice_silent')):
        samples=int(measurements['samples']); ml=time_layout(samples,'music')
        result.update(log_duration=float(np.log1p(samples/16000)),source_peak=float(measurements['source_peak']),normalization_gain=float(measurements['normalization_gain']),normalized_rms=float(measurements['normalized_rms']),voice_silent=float(measurements['voice_silent']),music_padding_fraction=1-sum(ml['valid_samples'])/(64600*len(ml['starts'])))
    finite(list(result.values()),'generated features')
    return result


class CacheAdapter:
    def __init__(self, heads_dir: str | Path, head_hashes: dict[str,str]):
        self.head_hashes=dict(head_hashes)
        expected={f'{r}_{s}.pt' for r in ('music','voice') for s in math.SEEDS}
        if set(head_hashes)!=expected: raise ContractError('Require exactly six externally pinned head hashes')
        for n,h in head_hashes.items(): check_hash(Path(heads_dir)/n,h)
        self.heads=(*math.load_heads(Path(heads_dir),'cpu'),dict(head_hashes))

    @torch.inference_mode()
    def replay_arrays(self, arrays: Mapping) -> tuple[dict,dict]:
        required=['mert_windows','voice_embeddings','presence','voice_original','voice_silent','samples','starts','valid_samples']
        if set(required)-arrays.keys(): raise ContractError('Missing current cache arrays')
        samples=int(arrays['samples']);ml=time_layout(samples,'music');vl=time_layout(samples,'voice')
        if not np.array_equal(arrays['starts'],ml['starts']) or not np.array_equal(arrays['valid_samples'],ml['valid_samples']):
            raise ContractError('Stale or wrong music window layout')
        for key,shape in [('mert_windows',(len(ml['starts']),1024)),('voice_embeddings',(len(vl['starts']),1280)),('presence',(2,))]:
            a=finite(arrays[key],key, key=='presence')
            if a.shape!=shape: raise ContractError(f'{key}: wrong shape/timebase {a.shape}, expected {shape}')
        if np.asarray(arrays['mert_windows']).dtype!=np.float32 or np.asarray(arrays['voice_embeddings']).dtype!=np.float32:
            raise ContractError('Current embedding caches must be float32')
        diagnostic=math.score_arrays(arrays,self.heads,'cpu')
        music,voice,fake,_=self.heads
        x=torch.from_numpy(np.asarray(arrays['mert_windows']));z=torch.from_numpy(np.asarray(arrays['voice_embeddings']))
        mw=np.stack([h.head((x-h.mean)/h.scale)[0].float().sigmoid().numpy() for h in music])
        vw=np.stack([h(z).float().softmax(-1)[:,fake].numpy() for h in voice])
        if bool(arrays['voice_silent']): vw=np.zeros_like(vw) # measured, predeclared silence gate, NOT missing imputation
        ms={f'{r}_seed':[diagnostic[f'{r.upper()}_FAKE_PROB_seed{s}'] for s in math.SEEDS] for r in ('voice','music')}
        ms.update(music_window=mw,voice_window=vw,df_original=float(arrays['voice_original']))
        for k in ('samples','source_peak','normalization_gain','normalized_rms','voice_silent'):
            if k in arrays: ms[k]=np.asarray(arrays[k]).item()
        baseline={n:diagnostic[n] for n in PREDICTION_COLUMNS}
        return diagnostic,from_measurements(baseline,ms)

    def replay_file(self, path: str | Path, expected_hash: str, expected_identity: str):
        check_hash(path,expected_hash)
        with np.load(path,allow_pickle=False) as archive: arrays={k:archive[k] for k in archive.files}
        if str(arrays.get('identity',''))!=expected_identity:
            raise ContractError('Cache identity mismatch')
        return self.replay_arrays(arrays)


class FixtureAdapter(CacheAdapter):
    def __init__(self, root: str | Path):
        self.root=Path(root); manifest=read_json(self.root/'INPUT_MANIFEST.json')
        # The manifest is input provenance, not an assertion that these data are new.
        for entry in manifest['files']: check_hash(contained(self.root,entry['path']),entry['sha256'])
        self.index=read_json(self.root/'fixtures/index.json')
        hashes={Path(e['path']).name:e['sha256'] for e in manifest['files'] if e['path'].startswith('fixtures/heads/')}
        super().__init__(self.root/'fixtures/heads',hashes)

    def replay(self):
        records=[]
        for entry in self.index:
            p=contained(self.root,entry['cache']);check_hash(p,entry['sha256'])
            with np.load(p,allow_pickle=False) as a: arrays={k:a[k] for k in a.files}
            diag,feat=self.replay_arrays(arrays)
            records.append({'ID':entry['ID'],'baseline':{n:diag[n] for n in PREDICTION_COLUMNS},'diagnostic':diag,'features':feat,'cache_identity':str(arrays['identity']), 'cache_sha256':entry['sha256'],'purpose':'diagnostic_only; previously exposed; NOT independent OOF'})
        return records


def write_extended(path: str | Path, record: dict, provenance: dict) -> str:
    """Non-object NPZ + JSON, immutable identity binds all required provenance.
    No labels or source descriptors are included in the feature matrix.
    """
    required={'audio_sha256','code_sha256','checkpoint_sha256','timebase','preprocessing','base_exposure','numeric_mode','feature_schema'}
    if required-provenance.keys(): raise ContractError('Incomplete extended-cache provenance')
    if provenance['feature_schema']!=SCHEMA: raise ContractError('Wrong extended-cache schema')
    order=list(record['features'])
    strict_frame(pd.DataFrame([[record['features'][n] for n in order]],columns=order),order)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    meta={'schema':SCHEMA,'ID':record['ID'],'feature_order':order,'provenance':provenance,'identity':digest(provenance)}
    with path.open('wb') as f:
        np.savez_compressed(f,values=np.asarray([record['features'][n] for n in order],np.float64),baseline=np.asarray([record['baseline'][n] for n in PREDICTION_COLUMNS],np.float64))
    meta['npz_sha256']=sha256(path);write_json(path.with_suffix('.json'),meta)
    return sha256(path.with_suffix('.json'))


def read_extended(path: str | Path, expected_sidecar_sha256: str, expected_identity: str) -> dict:
    path=Path(path); side=path.with_suffix('.json');check_hash(side,expected_sidecar_sha256);meta=read_json(side)
    if meta['schema']!=SCHEMA or meta['identity']!=digest(meta['provenance']) or meta['identity']!=expected_identity:
        raise ContractError('Extended-cache identity/schema mismatch')
    check_hash(path,meta['npz_sha256'])
    with np.load(path,allow_pickle=False) as a: values=a['values'];baseline=a['baseline']
    strict_frame(pd.DataFrame([values],columns=meta['feature_order']),meta['feature_order'])
    finite(baseline,'cached baseline',True)
    if baseline.shape!=(5,): raise ContractError('Expected five fixed base outputs')
    return {'ID':meta['ID'],'features':dict(zip(meta['feature_order'],values.tolist(),strict=True)), 'baseline':dict(zip(PREDICTION_COLUMNS,baseline.tolist(),strict=True)), 'provenance':meta['provenance']}
