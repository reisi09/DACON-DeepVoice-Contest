"""Evaluate all five scores from the six validation-selected classifier states."""
from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eight_fixed_io import atomic_json
from full_bank_features import load_rows, validate_cache, row_identity
from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, official_metrics, validate_predictions
from localization_models import WindowMIL
from v6_data import sha256
import decision_heads
from full_bank_contract import saved_config, check_config, pool_voice, role, validate_feature_contract


def load_selected_heads(output: Path, device: str):
    music_heads, voice_heads, checkpoints = [], [], {}
    fake_index = None
    config = saved_config(output)
    identity = json.loads((output / 'run_identity.json').read_text())
    for seed in config['seeds']:
        music = torch.load(output / f'music_{seed}.pt', map_location='cpu', weights_only=True)
        check_config(music, config, identity)
        if music.get('seed') != seed or music.get('route') != 'music':
            raise ValueError('Music checkpoint route or seed mismatch')
        head = WindowMIL(music['model']['mean'], music['model']['scale'])
        head.load_state_dict(music['model'])
        music_heads.append(head.to(device).eval())
        voice = torch.load(output / f'voice_{seed}.pt', map_location='cpu', weights_only=True)
        check_config(voice, config, identity)
        if voice.get('seed') != seed or voice.get('route') != 'voice':
            raise ValueError('Voice checkpoint route or seed mismatch')
        head = nn.Linear(1280, 2)
        head.load_state_dict(voice['model'])
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
    config = saved_config(output)
    index = json.loads((cache / 'index.json').read_text())
    validate_feature_contract(json.loads((output / 'run_identity.json').read_text()), index['identity'])
    identity_key = hashlib.sha256(json.dumps(index['identity'], sort_keys=True).encode()).hexdigest()
    scored = []
    for row in rows:
        entry = index['items'][row['ID']]
        path = (cache / entry['path']).resolve()
        if not path.is_relative_to(cache.resolve()):
            raise ValueError('Feature path escapes cache')
        validate_cache(path, row_identity(row, identity_key), entry['sha256'])
        with np.load(path, allow_pickle=False) as data:
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
                voice_scores.append(0. if silent else float(pool_voice(head(embeddings), fake_index, config).sigmoid()))
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
    rows = [row for row in load_rows(manifest) if role(row) == 'selection']
    checkpoints = load_selected_heads(output, device)[3]
    predictions = [five_columns(s) for s in score_rows(rows, cache, output, device)]
    validated = validate_predictions(predictions, [row['ID'] for row in rows])
    with (output / 'validation_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(validated)
    atomic_json(output / 'validation_metrics.json', official_metrics(rows, validated))
    atomic_json(output / 'candidate_lock.json', {'selection': 'selection role EER then BCE; configured seeds equal probability mean', 'config': saved_config(output),
                'checkpoints': checkpoints, 'validation_manifest_sha256': sha256(manifest),
                'predictions_sha256': sha256(output / 'validation_predictions.csv'),
                'official_submission': False, 'independent_test': False})
