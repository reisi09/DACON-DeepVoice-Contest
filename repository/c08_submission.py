"""Offline C08 music inference with the original baseline voice and presence models."""

import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from graph_scoring import SUBMISSION_COLUMNS, validate_predictions
from music_temporal import MusicRecord, TemporalMusic, load_temporal_head, predict_temporal_music
from v6_data import sha256
from v6_runtime import baseline_module


def score_record(head: TemporalMusic, record: MusicRecord, voice: float,
                 presence: tuple[float, float], device: str) -> list[float]:
    """Keep the original single-head inference path compatible."""
    return score_ensemble([head], record, voice, presence, device)


def score_ensemble(heads: list[TemporalMusic], record: MusicRecord, voice: float,
                   presence: tuple[float, float], device: str) -> list[float]:
    """Average music probabilities before applying the baseline file-max combination."""
    if not heads:
        raise ValueError('At least one music classifier is required')
    music = float(np.mean([predict_temporal_music(head, [dict(ID='file')],
                          {'file': record}, device)[0] for head in heads]))
    vp, mp = presence
    return [max(vp*voice, mp*music), voice, music, vp, mp]


def main(root: Path) -> None:
    """Infer each supplied file independently and write only a complete submission CSV."""
    assets = root/'model/assets'
    base = baseline_module(assets)
    args = base.parse_arguments()
    device = base.select_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((root/'model/manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError(f'Packaged asset or runtime hash mismatch: {name}')
    columns, rows = base.read_sample_submission(args.sample_submission)
    if columns != list(SUBMISSION_COLUMNS):
        raise ValueError('Unexpected sample-submission columns')
    files = base.order_audio_files(base.find_audio_files(args.test_dir), rows)
    presence = base.predict_presence_for_all_files(files, device)
    df, fake_index = base.load_df_arena_model(device)
    demucs = base.load_htdemucs_model()
    processor = AutoFeatureExtractor.from_pretrained(str(assets/'mert'), local_files_only=True)
    mert, info = AutoModel.from_pretrained(str(assets/'mert'), trust_remote_code=True,
                                          local_files_only=True, output_loading_info=True)
    if any(info.values()):
        raise ValueError(f'MERT checkpoint did not load exactly: {info}')
    mert.to(device).eval().requires_grad_(False)
    paths = manifest['choice'].get('checkpoints', ['model/trained/c08.pt'])
    if any(name not in manifest['sha256'] for name in paths):
        raise ValueError('Ensemble checkpoints must be covered by the package manifest')
    heads = [load_temporal_head(root/name, str(device)) for name in paths]
    for head in heads:
        if (head.architecture not in ('mert-residual-64x3-max-v1',
                                     'mert-residual-64x3-max-d1024-mlp32-softmax-v1')
                or head.input_dim != mert.config.hidden_size):
            raise ValueError('Unsupported C08 head or mismatched pretrained feature dimension')
    for index, (row, path) in enumerate(zip(rows, files, strict=True), 1):
        audio = base.load_audio(path)
        with torch.inference_mode():
            vocal, backing = base.separate_voice_and_music(path, demucs, device)
            voice = base.predict_fake(df, fake_index, vocal, device)
            backing = np.pad(backing[:len(audio)], (0, max(0, len(audio)-len(backing))))
            starts = np.asarray(base.get_segment_starts(len(audio)))
            vectors, audible = [], []
            for start in starts:
                segment = base.extract_segment(backing, int(start))
                audible.append(float(base.calculate_rms(segment) >= base.SILENCE_RMS))
                resampled = librosa.resample(segment, orig_sr=16000, target_sr=24000, res_type='soxr_hq')
                inputs = processor(resampled, sampling_rate=24000, return_tensors='pt')
                vectors.append(mert(inputs.input_values.to(device)).last_hidden_state.mean(1)[0].float().cpu().numpy())
            record = dict(music=np.asarray(vectors), music_audible=np.asarray(audible), starts=starts)
            values = score_ensemble(heads, record, voice, presence[path.stem], str(device))
        row.update({key: round(value, 10) for key, value in zip(base.PREDICTION_COLUMNS, values, strict=True)})
        print(f'C08 {index}/{len(files)} {path.stem}', flush=True)
    validate_predictions(rows, [r['ID'] for r in rows])
    temporary = args.output.with_suffix('.csv.partial')
    base.save_submission(temporary, columns, rows)
    temporary.replace(args.output)
    print(f'Saved {len(rows)} predictions to {args.output}', flush=True)
