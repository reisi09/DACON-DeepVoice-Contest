"""Offline full-bank inference with the unchanged encoders and trained heads."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn

from mert12_submission import (AutoFeatureExtractor, AutoModel, SEEDS, PREDICTION_COLUMNS,
    SUBMISSION_COLUMNS, baseline_module, pin_torchaudio_loader, windows, window_layout,
    sha256, validate_predictions, SubmissionError)
from localization_models import WindowMIL

CANDIDATE = 'existing9504_MERT330M_WindowMIL_DFfc5_701515'


def load_heads(root: Path, device):
    music, voice = [], []
    fake_index = None
    for seed in SEEDS:
        for route, collection in (('music', music), ('voice', voice)):
            record = torch.load(root/f'{route}_{seed}.pt', map_location='cpu', weights_only=True)
            if record['route'] != route or record['seed'] != seed:
                raise SubmissionError('Checkpoint route or seed mismatch')
            state = record['model']
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise SubmissionError('Nonfinite checkpoint')
            model = WindowMIL(state['mean'], state['scale']) if route == 'music' else nn.Linear(1280, 2)
            model.load_state_dict(state, strict=True)
            collection.append(model.to(device).eval().requires_grad_(False))
            if fake_index is not None and record['fake_label_index'] != fake_index:
                raise SubmissionError('Inconsistent fake-label index')
            fake_index = record['fake_label_index']
    return music, voice, fake_index


class Engine:
    def __init__(self, assets: Path, trained: Path, device, batch=8):
        self.device, self.batch = torch.device(device), batch
        self.base = baseline_module(assets)
        pin_torchaudio_loader(self.base)
        self.demucs = self.base.load_htdemucs_model().to(self.device).eval().requires_grad_(False)
        self.df, original_fake = self.base.load_df_arena_model(self.device)
        self.df.requires_grad_(False)
        self.panns, self.vi, self.mi = self.base.load_panns_model(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(assets/'mert', local_files_only=True)
        self.mert, info = AutoModel.from_pretrained(assets/'mert', trust_remote_code=True,
                                                  local_files_only=True, output_loading_info=True)
        if any(info.values()):
            raise SubmissionError('MERT weights did not load exactly')
        self.mert.to(self.device).eval().requires_grad_(False)
        self.music_heads, self.voice_heads, self.fake_index = load_heads(trained, self.device)
        if original_fake != self.fake_index:
            raise SubmissionError('DF-Arena fake label mapping changed')
        self.embedding = None
        def capture(_module, inputs):
            self.embedding = inputs[0].detach()
        self.hook = self.df.backbone.conformer.fc5.register_forward_pre_hook(capture)

    @torch.inference_mode()
    def predict(self, path: Path, *, preserve_gain=False, collect_features=False):
        import librosa
        import soundfile as sf
        raw = self.base.load_audio(path)
        peak = float(np.abs(raw).max())
        if not preserve_gain and peak > .6:
            raw = (raw*(.6/peak)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix='fullbank_inference_') as temporary:
            normalized = Path(temporary)/'normalized.wav'
            sf.write(normalized, raw, 16000, subtype='FLOAT')
            voice, backing = self.base.separate_voice_and_music(normalized, self.demucs, self.device)
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw)-len(voice))))
        backing = np.pad(backing[:len(raw)], (0, max(0, len(raw)-len(backing))))
        starts = window_layout(len(raw))
        features = []
        for offset in range(0, len(starts), self.batch):
            audio = [librosa.resample(x, orig_sr=16000, target_sr=24000, res_type='soxr_hq')
                     for x in windows(backing, starts[offset:offset+self.batch])]
            values = self.processor(audio, sampling_rate=24000, return_tensors='pt', padding=True).input_values.to(self.device)
            features.append(self.mert(values, output_hidden_states=True).hidden_states[12].mean(1).float())
        features = torch.cat(features)
        music = []
        music_windows = []
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            for head in self.music_heads:
                logits, _ = head.head((features-head.mean)/head.scale)
                music.append(float((logits.float().logsumexp(0)-np.log(len(features))).sigmoid()))
                if collect_features:
                    music_windows.append(logits.float().sigmoid().cpu().numpy())
        embeddings = []
        original_scores = []
        panns = self.panns
        if collect_features:
            from dacon_stacking.gpu import RecordingPanns
            panns = RecordingPanns(self.panns)
        presence = np.zeros(2)
        for start in self.base.get_segment_starts(len(raw)):
            original = self.df(input_values=torch.from_numpy(self.base.extract_segment(voice, start)).to(self.device))
            if collect_features:
                original_scores.append(float(original['logits'].float().softmax(-1)[0, self.fake_index]))
            embeddings.append(self.embedding)
            presence = np.maximum(presence, self.base.predict_presence(panns, self.vi, self.mi,
                                                                     self.base.extract_segment(raw, start)))
        embeddings = torch.cat(embeddings)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            voices, voice_windows = [], []
            for head in self.voice_heads:
                probabilities = head(embeddings).float().softmax(-1)[:, self.fake_index]
                voices.append(float(probabilities.max()))
                if collect_features:
                    voice_windows.append(probabilities.cpu().numpy())
        vp, mp = presence.tolist()
        vf = 0. if self.base.calculate_rms(voice) < self.base.SILENCE_RMS else float(np.mean(voices))
        mf = float(np.mean(music))
        prediction = dict(zip(PREDICTION_COLUMNS, (max(vp*vf, mp*mf), vf, mf, vp, mp), strict=True))
        if not collect_features:
            return prediction
        silent = self.base.calculate_rms(voice) < self.base.SILENCE_RMS
        vw = np.stack(voice_windows)
        classes = np.concatenate(panns.clip_predictions, axis=0)
        per_clip_presence = np.stack((classes[:, self.vi].max(axis=1),
                                      classes[:, self.mi].max(axis=1)), axis=1)
        measurements = {
            'music_seed': music, 'voice_seed': [0.]*len(voices) if silent else voices,
            'music_window': np.stack(music_windows),
            'voice_window': np.zeros_like(vw) if silent else vw,
            'df_original': 0. if silent else max(original_scores),
            'pann_classes': classes, 'pann_presence': per_clip_presence,
            'samples': len(raw), 'source_peak': peak,
            'normalization_gain': .6/peak if not preserve_gain and peak > .6 else 1.,
            'normalized_rms': self.base.calculate_rms(raw), 'voice_silent': silent,
            '_normalized_audio': raw, '_voice_audio': voice, '_music_audio': backing,
        }
        return prediction, measurements


def run(root: Path):
    started = time.monotonic()
    manifest = json.loads((root/'model/manifest.json').read_text())
    if manifest['choice']['candidate'] != CANDIDATE:
        raise SubmissionError('Unexpected full-bank package')
    for name, expected in manifest['sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise SubmissionError('Package member changed: '+name)
    decoder = root/'model/runtime/bin/ffmpeg'
    decoder.chmod(decoder.stat().st_mode | 0o111)
    os.environ['PATH'] = str(decoder.parent)+os.pathsep+os.environ.get('PATH', '')
    base = baseline_module(root/'model/assets')
    args = base.parse_arguments()
    device = base.select_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    columns, rows = base.read_sample_submission(args.sample_submission)
    if columns != list(SUBMISSION_COLUMNS):
        raise SubmissionError('Unexpected submission columns')
    files = base.order_audio_files(base.find_audio_files(args.test_dir), rows)
    engine = Engine(root/'model/assets', root/'model/trained', device)
    stacking_lock = os.environ.get('DACON_STACKING_LOCK')
    stacking_sha = os.environ.get('DACON_STACKING_LOCK_SHA256')
    if bool(stacking_lock) != bool(stacking_sha):
        raise SubmissionError('Supply both stacking lock path and out-of-band SHA-256')
    if stacking_lock:
        from dacon_stacking.gpu import load_deployment, predict_engine_file
        meta_model, meta_contract = load_deployment(Path(stacking_lock), stacking_sha, root)
        predictions = [predict_engine_file(engine, path, path.stem, meta_model, meta_contract)
                       for path in files]
    else:
        predictions = [dict(ID=path.stem, **engine.predict(path)) for path in files]
    predictions = validate_predictions(predictions, [path.stem for path in files])
    import csv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)
    (args.output.parent/'runtime_receipt.json').write_text(json.dumps({'candidate': CANDIDATE,
        'files': len(files), 'seconds': time.monotonic()-started, 'device': str(device), 'file_combiner': 'stacking' if stacking_lock else 'max',
        'stacking_lock_sha256': stacking_sha}, indent=2)+'\n')
