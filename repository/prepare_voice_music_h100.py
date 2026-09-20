"""Prepare a bounded follow-up before paid compute; preserve source labels."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import sys

import imageio_ffmpeg
import numpy as np

from prepare_six_h100 import save_audio, write_rows
from v2_data import decode_audio, sha256_file, stable_key

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'_workspace/voice_music_h100_20260917'
OUT = ROOT/'results/voice_music_h100_20260917'
PRIOR = ROOT/'results/six_h100_20260917'


def acquire() -> None:
    import prepare_fresh_fma as fma
    import prepare_fresh_sonics as sonics
    OUT.mkdir(parents=True, exist_ok=True)
    old = list(csv.DictReader((PRIOR/'fresh_fma_manifest.csv').open()))
    prior = OUT/'prior_fma.json'
    prior.write_text(json.dumps({'sources': old}))
    fma.OUTPUT, fma.EVIDENCE = DATA/'fma', OUT
    fma.PRIOR = (*fma.PRIOR, prior)
    fma.main(64)
    sonics.OUTPUT, sonics.EVIDENCE = DATA/'sonics', OUT
    metadata = sonics.OUTPUT/'metadata'
    shutil.copytree(ROOT/'_workspace/fresh_sonics_20260917/metadata', metadata, dirs_exist_ok=True)
    sonics.main(19, PRIOR/'fresh_sonics_manifest.csv')


def probes() -> None:
    source = DATA/'source'
    source.mkdir(parents=True, exist_ok=True)
    rows, originals = [], []

    def add(parent, audio, variant, **extra):
        identifier = parent['ID']+'_'+variant
        path = source/'audio'/(identifier+'.wav')
        save_audio(path, audio)
        row = {k: parent.get(k, '') for k in ('FILE_FAKE','VOICE_FAKE','MUSIC_FAKE','VOICE_PRESENT','MUSIC_PRESENT')}
        row.update(ID=identifier, path='audio/'+path.name, audio_sha256=sha256_file(path),
            parent_id=parent['ID'], group_id=parent['group_id'], split=parent['split'],
            source=parent['source'], variant=variant, duration_seconds=len(audio)/16000,
            kind='speech' if parent['source']=='MLAAD' else 'music',
            domain='telephone' if variant=='telephone' else 'clean',
            event_intervals=json.dumps([dict(component='voice',start=0,end=len(audio)/16000,
                label=int(parent['VOICE_FAKE']))]) if parent.get('VOICE_FAKE') in ('0','1') else '[]',
            label_status='source metadata; independent listening pending', **extra)
        rows.append(row)
        return row

    speech = list(csv.DictReader((PRIOR/'fresh_speech_manifest.csv').open()))
    training_speakers = sorted({r['group_id'] for r in speech if r['split']=='train'})
    for row in speech:
        if row['group_id']==training_speakers[1]:
            row['split']='calibration'
    music = []
    for name in ('fma','sonics'):
        cohort = list(csv.DictReader((OUT/f'fresh_{name}_manifest.csv').open()))
        for provider in sorted({r['source'] for r in cohort}):
            subset = sorted((r for r in cohort if r['source']==provider), key=lambda r:stable_key(r['group_id']))
            for index, row in enumerate(subset):
                fraction = index/len(subset)
                row['split'] = 'train' if fraction<.5 else 'calibration' if fraction<.7 else 'dev' if fraction<.85 else 'test'
        music.extend(cohort)
    for row in speech+music:
        original = ROOT/row['path']
        if sha256_file(original)!=row['source_sha256']:
            raise ValueError('Source changed since acquisition')
        audio = decode_audio(original, limit_seconds=None)
        if len(audio)<64000:
            raise ValueError('Source too short')
        row['source_pcm_crop'] = ''
        clip = audio[:min(15*16000,len(audio))]
        if row['source']=='MLAAD':
            clip = audio[:64000]
            clip = clip * (.05/max(float(np.sqrt(np.mean(clip**2))),1e-8))
            if np.abs(clip).max()>.95:
                raise ValueError('Matched voice crop clips at fixed gain')
        clean = add(row,clip,'base')
        for variant,args,suffix in (
            ('telephone',['-af','highpass=f=300,lowpass=f=3400','-ar','8000','-c:a','pcm_mulaw'],'.wav'),
            ('mp3_32k',['-c:a','libmp3lame','-b:a','32k'],'.mp3')):
            encoded = DATA/'encoded'/(row['ID']+'_'+variant+suffix)
            encoded.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(source/clean['path']),
                            *args,str(encoded)],check=True,capture_output=True)
            degraded=decode_audio(encoded,limit_seconds=None)
            if abs(len(degraded)-len(clip))>1600:
                raise ValueError('Codec changed matched duration')
            add(row,degraded[:len(clip)],variant)
        if row['source']!='MLAAD':
            add(row,clip[:64000],'4s')
        originals.append(row)
    for row in speech:
        voice=decode_audio(source/next(r['path'] for r in rows if r['parent_id']==row['ID'] and r['variant']=='base'),limit_seconds=None)
        candidates=[r for r in music if r['split']==row['split'] and r['source']=='FMA']
        # Select background by parent identity, independently of classifier scores.
        background=min(candidates,key=lambda r:stable_key(row['original_file']+'|'+r['ID']))
        backing=decode_audio(ROOT/background['path'],limit_seconds=None)[:len(voice)]
        backing*=.05/max(float(np.sqrt(np.mean(backing**2))),1e-8)
        for ratio,variant in ((.5,'mix_minus6dB'),(1.,'mix_0dB')):
            combined=voice+ratio*backing
            if np.abs(combined).max()>.99:
                raise ValueError('Matched mixture clips; do not silently change gain')
            parent=row | {'ID':row['ID'], 'group_id':row['group_id']+'|'+background['group_id'],
                'MUSIC_FAKE':'0','MUSIC_PRESENT':'1',
                'VOICE_FAKE':'1' if row['VOICE_FAKE']=='1' else ''}
            add(parent,combined,variant,background_id=background['ID'],paired_voice_id=row['ID']+'_base')
    assignments={}
    for row in rows:
        for group in row['group_id'].split('|'):
            if group in assignments and assignments[group]!=row['split']:
                raise ValueError('Source leaked across splits')
            assignments[group]=row['split']
    old_music={r['group_id'] for name in ('fma','sonics') for r in csv.DictReader((PRIOR/f'fresh_{name}_manifest.csv').open())}
    if old_music & {r['group_id'] for r in music}:
        raise ValueError('New music overlaps the previous run')
    write_rows(source/'probes.csv',rows)
    write_rows(OUT/'sources.csv',originals)
    receipt=dict(rows=len(rows), parents=len(originals), new_music_parents=len(music),
        independent_groups=len(assignments), manifest_sha256=sha256_file(source/'probes.csv'),
        purpose='cause-isolation pilot; below the 300-independent-source promotion gate',
        promotion_allowed=False, official_score=None, labels_independently_reviewed=False,
        limitations=['Speech speakers reused from the opened six-experiment diagnostic panel',
            'FMA/SONICS corpus bias remains; pretraining overlap is unknown',
            'Music-source voice labels remain unknown, including real-voice mixtures',
            'No verified singing labels; residual fitting is a speech-only diagnostic',
            'Fresh music groups were excluded from all previous project music cohorts'])
    (source/'data_receipt.json').write_text(json.dumps(receipt,indent=2))
    shutil.copy2(source/'probes.csv',OUT/'probes.csv')
    shutil.copy2(source/'data_receipt.json',OUT/'data_receipt.json')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    {'acquire':acquire,'probes':probes}[sys.argv[1]]()
