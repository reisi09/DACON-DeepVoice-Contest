"""CPU checks for source leakage, matched controls, and actual head gradients."""
from pathlib import Path
import csv
import json

import numpy as np
import torch

from prepare_voice_music_h100 import DATA, OUT, PRIOR, ROOT
from redesign_models import VoiceResidual
from v2_data import decode_audio, sha256_file
from voice_music_h100_selection import seed_gate


def main():
    rows=list(csv.DictReader((DATA/'source/probes.csv').open()))
    groups={}; hashes={}
    for row in rows:
        path=DATA/'source'/row['path']
        assert sha256_file(path)==row['audio_sha256'], row['ID']
        for group in row['group_id'].split('|'):
            assert groups.setdefault(group,row['split'])==row['split'], group
        assert hashes.setdefault(row['audio_sha256'],row['split'])==row['split']
    sources={r['ID']:r for r in csv.DictReader((OUT/'sources.csv').open())}
    matched=0
    for row in rows:
        if not row.get('paired_voice_id'):
            continue
        voice=decode_audio(DATA/'source/audio'/(row['paired_voice_id']+'.wav'),limit_seconds=None)
        mixture=decode_audio(DATA/'source'/row['path'],limit_seconds=None)
        background=decode_audio(ROOT/sources[row['background_id']]['path'],limit_seconds=None)[:len(voice)]
        background*=.05/np.sqrt(np.mean(background**2))
        ratio=.5 if row['variant']=='mix_minus6dB' else 1.
        assert len(voice)==len(mixture)==64000
        assert np.allclose(mixture,voice+ratio*background,atol=1e-7,rtol=0)
        assert row['VOICE_FAKE']==('1' if row['FILE_FAKE']=='1' else '')
        matched+=1
    output=PRIOR/'downloaded/unpacked/output'
    manifest=json.loads((output/'cache/manifest.json').read_text())
    prior_rows=list(csv.DictReader((PRIOR/'probes.csv').open()))
    selected=[next(r for r in prior_rows if r['source']=='MLAAD' and r['variant']=='base' and r['VOICE_FAKE']==str(label)) for label in (0,1)]
    records=[]
    for row in selected:
        key=row['ID']
        if key not in manifest['records']:
            aliases=json.loads((output/'extraction.json').read_text())['aliases'];key=aliases[key]
        with np.load(output/'cache'/manifest['records'][key]['record'],allow_pickle=False) as record:
            records.append((record['voice_embedding'][0].copy(),record['voice_logits'][0].copy()))
    embeddings=torch.tensor(np.stack([r[0] for r in records]))[:,None]
    logits=torch.tensor(np.asarray([r[1] for r in records]))[:,None]
    valid=torch.ones(2,1,dtype=torch.bool); targets=torch.tensor([0.,1.])
    torch.manual_seed(41);model=VoiceResidual();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    first=model(embeddings,logits,valid)[0].detach().clone()
    for _ in range(2):
        optimizer.zero_grad();bag,_=model(embeddings,logits,valid)
        torch.nn.functional.binary_cross_entropy_with_logits(bag,targets).backward()
        optimizer.step()
    grad=float(model.input.weight.grad.norm());movement=float((model(embeddings,logits,valid)[0]-first).abs().max())
    assert grad>0 and movement>0, 'Residual hidden layer disconnected from loss'
    baseline=json.loads((output/'predictions_B_seed41.json').read_text())
    old=[r|{'kind':'music'} for r in prior_rows if r['source'] in ('FMA','suno','udio')]
    assert not seed_gate(old,{'music_control':baseline,'music_new':baseline})['additional_seeds']
    result=dict(source_groups_disjoint=True,audio_hashes_checked=len(rows),matched_mixtures_checked=matched,
        real_audio_hidden_gradient_norm=grad,real_audio_logit_movement=movement,tie_does_not_spend_gpu=True)
    (OUT/'preflight_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    main()
