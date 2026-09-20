"""Protect all-data coverage and independent early stopping in the submission ensemble."""
from dataclasses import replace
from pathlib import Path
import json
import time

from temporal_backbone import REGULARIZED_C08
from temporal_full_refit import train_cv
from test_temporal_augmentation import samples
from v6_data import assert_disjoint
import numpy as np
import torch
from music_temporal import TemporalMusic
from c08_submission import score_ensemble


def test_all_files_train_four_folds_without_entering_their_own_validation(tmp_path: Path) -> None:
    rows, records = samples()
    rows = [dict(r, music_training='1') for r in rows]
    plan = dict(rows=rows, quality={r['ID']:dict(starts=records[r['ID']]['starts'].tolist(),
        usable_indices=list(range(len(records[r['ID']]['starts'])))) for r in rows})
    recipe = replace(REGULARIZED_C08, epochs=5, patience=1, min_delta=100.)
    result = train_cv(plan, records, tmp_path, time.time()+60, recipe=recipe, device='cpu')
    by_id = {r['ID']:r for r in rows}
    for identifier in by_id:
        assert sum(identifier in f['train_ids'] for f in result['folds']) == 4
        assert sum(identifier in f['validation_ids'] for f in result['folds']) == 1
    for fold in result['folds']:
        assert_disjoint([by_id[k] for k in fold['train_ids']], [by_id[k] for k in fold['validation_ids']])
        assert fold['epochs_completed'] == 2
        trace=json.loads((tmp_path/f"fold{fold['fold']}"/'training.json').read_text())
        assert trace['best_validation_loss'] == min(trace['validation_losses'])


def test_ensemble_averages_music_before_file_fusion() -> None:
    heads=[]
    for p in (.1,.9):
        model=TemporalMusic(simplified=True,compact=True,input_dim=1024,
                            mlp_output=True,softmax_output=True).eval()
        with torch.no_grad():
            model.output[-1].weight.zero_()
            model.output[-1].bias.copy_(torch.tensor([0.,np.log(p/(1-p))]))
        heads.append(model)
    record=dict(music=np.zeros((3,1024),dtype='float32'),music_audible=np.ones(3),
                starts=np.arange(3)*64600)
    actual=score_ensemble(heads,record,.5,(1.,1.),'cpu')
    np.testing.assert_allclose(actual,[.5,.5,.5,1.,1.],atol=1e-6)
