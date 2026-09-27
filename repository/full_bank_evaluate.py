"""Evaluate all five scores from the six validation-selected classifier states."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eight_fixed_io import atomic_json
from full_bank_features import load_rows
from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, official_metrics, validate_predictions
from localization_models import WindowMIL
import voice_heads as voice_heads_module
from v6_data import sha256
import decision_heads


def load_selected_heads(output: Path, device: str):
    music_heads, voice_heads, checkpoints = [], [], {}
    fake_index = None
    for seed in (41, 42, 43):
        music = torch.load(output / f'music_{seed}.pt', map_location='cpu', weights_only=True)
        head = WindowMIL(music['model']['mean'], music['model']['scale'])
        head.load_state_dict(music['model'])
        music_heads.append(head.to(device).eval())
        voice = torch.load(output / f'voice_{seed}.pt', map_location='cpu', weights_only=True)
        head = voice_heads_module.load(voice)   # Linear, MLP or ResMLP by the record's 'arch'
        voice_heads.append(head.to(device).eval())
        if fake_index is not None and fake_index != voice['fake_label_index']:
            raise ValueError('Voice fake-label mapping differs across seeds')
        fake_index = voice['fake_label_index']
        for route in ('music', 'voice'):
            checkpoints[f'{route}_{seed}.pt'] = sha256(output / f'{route}_{seed}.pt')
    return music_heads, voice_heads, fake_index, checkpoints


@torch.no_grad()
def score_rows(rows, cache: Path, output: Path, device: str = 'cuda') -> list[dict]:
    """Per-file Pv/Pm (legacy PANNs max), Fv/Fm (seed means) and cached presence features; any split."""
    music_heads, voice_heads, fake_index, _ = load_selected_heads(output, device)
    scored = []
    for row in rows:
        with np.load(cache / f"{row['ID']}.npz", allow_pickle=False) as data:
            features = torch.from_numpy(data['mert_windows']).to(device)
            embeddings = torch.from_numpy(data['voice_embeddings']).to(device)
            presence = data['presence']
            silent = bool(data['voice_silent'])
            feats = json.loads(str(data['presence_features'])) if 'presence_features' in data.files else None
        music_scores, voice_scores = [], []
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            for head in music_heads:
                logits, _ = head.head((features - head.mean) / head.scale)
                music_scores.append(float((logits.float().logsumexp(0)-np.log(len(features))).sigmoid()))
            for head in voice_heads:
                voice_scores.append(0. if silent else float(head(embeddings).float().softmax(-1)[:, fake_index].max()))
        scored.append({'ID': row['ID'], 'Pv': float(presence[0]), 'Pm': float(presence[1]),
                       'Fv': float(np.mean(voice_scores)), 'Fm': float(np.mean(music_scores)), 'presence_features': feats})
    return scored


def five_columns(s: dict, presence_heads=None, ensemble=None) -> dict:
    """Same per-file decision layer as the submission runtime (decision_heads)."""
    vp, mp = decision_heads.presence(presence_heads, s['presence_features'], (s['Pv'], s['Pm']))
    file = decision_heads.file_score(ensemble, {**s, 'Pv': vp, 'Pm': mp})
    return {'ID': s['ID'], **dict(zip(PREDICTION_COLUMNS, (file, s['Fv'], s['Fm'], vp, mp), strict=True))}


@torch.no_grad()
def evaluate(manifest: Path, cache: Path, output: Path, device: str = 'cuda') -> None:
    """No fitted calibration or cross-file aggregation; file-max graph is unchanged."""
    rows = [row for row in load_rows(manifest) if row['split'] == 'val']
    checkpoints = load_selected_heads(output, device)[3]
    predictions = [five_columns(s) for s in score_rows(rows, cache, output, device)]
    validated = validate_predictions(predictions, [row['ID'] for row in rows])
    with (output / 'validation_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(validated)
    atomic_json(output / 'validation_metrics.json', official_metrics(rows, validated))
    atomic_json(output / 'candidate_lock.json', {'selection': 'per-route validation EER then BCE; seeds41/42/43 equal probability mean',
                'checkpoints': checkpoints, 'validation_manifest_sha256': sha256(manifest),
                'predictions_sha256': sha256(output / 'validation_predictions.csv'),
                'official_submission': False, 'independent_test': False})
