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
import decision_heads

CANDIDATE = 'datav4_s3_p1voice_adhead_fusion_mert6x12_music'


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
        from panns_inference import labels as panns_labels
        self.panns_labels = list(panns_labels)
        self.label_groups = json.loads((self.base.PANNS_DIR / 'component_labels.json').read_text())
        # optional per-file decision layer (decision_heads); absent files -> legacy PANNs max and FILE = max rule
        read = lambda name: json.loads((trained / name).read_text()) if (trained / name).exists() else None
        self.presence_heads, self.ensemble = read('presence_head.json'), read('file_ensemble.json')
        # optional: MERT layers of the music heads (music_layers.json, default layer 12) and component fusion
        # (component_fusion.json): VOICE = val-CDF fusion of our DF-Arena heads with heads trained on AntiDeepfake-1B
        # voice-stem embeddings (voice_ad_*.pt)
        self.music_layers = read('music_layers.json') or [12]
        self.fusion, self.detector, self.ad_heads = read('component_fusion.json'), None, []
        self.fusion_deadline = float('inf')   # run() sets it from the 60-minute competition limit
        if self.fusion and self.fusion['voice']['weights'].get('ad'):
            from antideepfake_runtime import AntiDeepfake
            self.detector = AntiDeepfake(assets/'antideepfake'/'ad', str(self.device))
            for seed in SEEDS:
                record = torch.load(trained/f'voice_ad_{seed}.pt', map_location='cpu', weights_only=True)
                if record['route'] != 'voice' or record['seed'] != seed or record['fake_label_index'] != 0:
                    raise SubmissionError('AntiDeepfake head mismatch')
                head = nn.Linear(1280, 2)
                head.load_state_dict(record['model'], strict=True)
                self.ad_heads.append(head.to(self.device).eval().requires_grad_(False))
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
    def predict(self, path: Path, *, preserve_gain=False):
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
            hidden = self.mert(values, output_hidden_states=True).hidden_states
            features.append(torch.cat([hidden[k].mean(1).float() for k in self.music_layers], -1))
        features = torch.cat(features)
        music = []   # heads in fp32, as trained and as the fusion tables were fitted
        for head in self.music_heads:
            logits, _ = head.head((features-head.mean)/head.scale)
            music.append(float((logits.float().logsumexp(0)-np.log(len(features))).sigmoid()))
        embeddings = []
        for start in self.base.get_segment_starts(len(raw)):
            self.df(input_values=torch.from_numpy(self.base.extract_segment(voice, start)).to(self.device))
            embeddings.append(self.embedding)
        embeddings = torch.cat(embeddings)
        voices = [float(head(embeddings).float().softmax(-1)[:, self.fake_index].max())
                  for head in self.voice_heads]
        probs = self.panns.inference(self.base.make_panns_segments(raw))[0]   # same segments as the legacy loop
        legacy = decision_heads.legacy_presence(probs, self.panns_labels, self.label_groups)
        feats = (decision_heads.presence_features(probs, self.panns_labels, self.label_groups, raw, voice)
                 if self.presence_heads else None)
        vp, mp = decision_heads.presence(self.presence_heads, feats, legacy)
        silent = self.base.calculate_rms(voice) < self.base.SILENCE_RMS
        vf = 0. if silent else float(np.mean(voices))
        mf = float(np.mean(music))
        if self.fusion and not silent:
            import component_fusion as cf
            members = {'ours': vf}
            if self.detector and time.monotonic() < self.fusion_deadline:   # ponytail: past the budget, later files use ours
                emb = torch.from_numpy(self.detector.windows(voice)[0]).to(self.device)
                members['ad'] = float(np.mean([float(h(emb).softmax(-1)[:, 0].max()) for h in self.ad_heads]))
            vf = cf.fuse(self.fusion['voice'], members)
        file = decision_heads.file_score(self.ensemble, dict(Pv=vp, Pm=mp, Fv=vf, Fm=mf))
        return dict(zip(PREDICTION_COLUMNS, (file, vf, mf, vp, mp), strict=True))


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
    engine.fusion_deadline = started + 48*60   # leaves ~12 min of the 60-min limit for the rest without fusion members
    predictions = [dict(ID=path.stem, **engine.predict(path)) for path in files]
    predictions = validate_predictions(predictions, [path.stem for path in files])
    import csv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)
    (args.output.parent/'runtime_receipt.json').write_text(json.dumps({'candidate': CANDIDATE,
        'files': len(files), 'seconds': time.monotonic()-started, 'device': str(device)}, indent=2)+'\n')
