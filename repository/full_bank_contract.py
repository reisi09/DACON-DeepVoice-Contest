"""Training settings shared by checkpoint selection and packaged per-file inference."""
from __future__ import annotations

import json
import math
from pathlib import Path



def role(row):
    return row.get('role') or {'train': 'train', 'val': 'selection', 'test': 'development'}[row['split']]


def config_for(values=None):
    config = dict(seeds=[41, 42, 43], voice_pooling='max', voice_topk_fraction=.25,
                  voice_local_loss_weight=0., music_local_loss_weight=.25)
    values = values or {}
    if set(values) - set(config):
        raise ValueError('Unsupported experiment settings: ' + ', '.join(sorted(set(values) - set(config))))
    config.update(values)
    seeds = config['seeds']
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    if config['voice_pooling'] not in ('max', 'lme', 'topk'):
        raise ValueError('Unsupported voice pooling')
    for key in ('voice_local_loss_weight', 'music_local_loss_weight', 'voice_topk_fraction'):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('Invalid setting: ' + key)
    if not 0 < config['voice_topk_fraction'] <= 1:
        raise ValueError('Top-k fraction must be in (0, 1]')
    return config


def saved_config(directory: Path):
    path = directory / 'run_identity.json'
    identity = json.loads(path.read_text()) if path.exists() else {}
    return config_for(identity.get('config', {'seeds': identity.get('seeds', [41, 42, 43])}))


def check_config(record, config, expected_identity=None):
    identity = record.get('identity', {})
    if expected_identity is not None and identity != expected_identity:
        raise ValueError('Checkpoint training identity differs from run identity')
    if config_for(identity.get('config', {'seeds': identity.get('seeds', [41, 42, 43])})) != config:
        raise ValueError('Checkpoint inference settings differ from run identity')


def pool_odds(odds, config):
    mode = config['voice_pooling']
    if mode == 'max':
        return odds.max()
    if mode == 'lme':
        return odds.logsumexp(0) - math.log(len(odds))
    return odds.topk(max(1, math.ceil(len(odds) * config['voice_topk_fraction']))).values.mean()


def pool_voice(values, fake_index, config):
    return pool_odds(values.float()[:, fake_index] - values.float()[:, 1-fake_index], config)


def feature_contract(identity):
    return {k: v for k, v in identity.items() if k != 'manifest_sha256'}


def validate_feature_contract(run_identity, cache_identity):
    expected = run_identity.get('feature_contract')
    if expected is None or expected != feature_contract(cache_identity):
        raise ValueError('Cache encoder/assets/preprocessing provenance differs or is unavailable')
