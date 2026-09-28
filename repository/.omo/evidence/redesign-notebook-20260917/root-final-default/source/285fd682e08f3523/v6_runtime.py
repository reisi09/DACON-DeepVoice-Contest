"""Frozen final encoders, aligned windows and content-addressed feature cache."""

import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from v6_core import VERSION
from v6_data import sha256


def load_cache(path: Path, identity: str) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if archive["identity"].item() != identity:
            return None
        values = {k: archive[k] for k in archive.files if k != "identity"}
    for k in ("presence", "voice", "music", "raw", "music_audible", "starts"):
        if k not in values or not np.isfinite(values[k]).all():
            raise ValueError(f"Corrupted cache {path}: {k}")
    n = len(values["starts"])
    shapes = dict(presence=(n, 2), voice=(n,), music=(n, 768), raw=(n,),
                  music_audible=(n,), starts=(n,))
    if not n or any(values[k].shape != shape for k, shape in shapes.items()):
        raise ValueError(f"Misaligned cached features: {path}")
    return values


def asset_identity(assets: Path, verify: bool = True) -> str:
    manifest = json.loads((assets / "assets.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("Asset manifest version mismatch")
    if verify:
        for name, digest in manifest["sha256"].items():
            if sha256(assets / name) != digest:
                raise ValueError(f"Asset digest mismatch: {name}")
    return sha256(assets / "assets.json")


def baseline_module(assets: Path):
    spec = importlib.util.spec_from_file_location("v6_official_baseline", assets / "baseline" / "script.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the bundled official baseline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Engine:
    def __init__(self, assets: Path, device: str = "cuda") -> None:
        import torch
        import onnxruntime as ort
        from transformers import AutoFeatureExtractor, AutoModel

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.asset_id = asset_identity(assets)
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            ort.preload_dlls()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("Install onnxruntime-gpu==1.23.2; CPU fallback is not allowed")
            providers = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested", "use_tf32": "0"})]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.voice = ort.InferenceSession(str(assets / "voice" / "w2v2-aasist.onnx"),
                                         sess_options=options, providers=providers)
        if device == "cuda" and self.voice.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("Voice model did not initialize CUDAExecutionProvider")
        self.voice.disable_fallback()
        self.baseline = baseline_module(assets)
        self.panns, self.voice_indices, self.music_indices = self.baseline.load_panns_model(self.device)
        self.demucs = self.baseline.load_htdemucs_model()
        self.raw, self.fake_index = self.baseline.load_df_arena_model(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(str(assets / "mert"), local_files_only=True)
        self.mert, info = AutoModel.from_pretrained(str(assets / "mert"), trust_remote_code=True,
                                                  local_files_only=True, output_loading_info=True)
        if any(info.values()):
            raise ValueError(f"MERT checkpoint was not loaded exactly: {info}")
        self.mert.to(self.device).eval().requires_grad_(False)
        self.raw.requires_grad_(False)
        versions = {name: importlib.metadata.version(name) for name in (
            "transformers", "librosa", "numpy", "demucs", "panns-inference", "soundfile")}
        identity = [VERSION, self.asset_id, sha256(Path(__file__)),
                    device, torch.__version__, ort.__version__, versions]
        self.cache_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def extract(self, path: Path) -> dict[str, np.ndarray]:
        import librosa
        import torch
        from scipy.special import expit

        base = self.baseline
        audio = base.load_audio(path)
        starts = base.get_segment_starts(len(audio))
        voice_audio, music_audio = base.separate_voice_and_music(path, self.demucs, self.device)
        # Resampler rounding may differ by one sample. Preserve the raw timeline.
        voice_audio = np.pad(voice_audio[:len(audio)], (0, max(0, len(audio) - len(voice_audio))))
        music_audio = np.pad(music_audio[:len(audio)], (0, max(0, len(audio) - len(music_audio))))
        presence, voice, music, raw, audible = [], [], [], [], []
        with torch.inference_mode():
            for start in starts:
                original = base.extract_segment(audio, start)
                vocal = base.extract_segment(voice_audio, start)
                backing = base.extract_segment(music_audio, start)
                p, _ = self.panns.inference(base.make_panns_segments(original))
                presence.append([p[:, self.voice_indices].max(), p[:, self.music_indices].max()])
                if base.calculate_rms(vocal) < base.SILENCE_RMS:
                    voice.append(0.)
                else:
                    logits = self.voice.run(["logits"], {"wav": vocal[None]})[0][0]
                    voice.append(float(expit(logits[0] - logits[1])))
                raw.append(base.predict_fake(self.raw, self.fake_index, original, self.device))
                audible.append(float(base.calculate_rms(backing) >= base.SILENCE_RMS))
                resampled = librosa.resample(backing, orig_sr=16000, target_sr=24000, res_type="soxr_hq")
                inputs = self.processor(resampled, sampling_rate=24000, return_tensors="pt")
                embedding = self.mert(inputs.input_values.to(self.device)).last_hidden_state.mean(dim=1)
                music.append(embedding[0].float().cpu().numpy())
        result = dict(presence=np.asarray(presence, dtype="float32"), voice=np.asarray(voice),
                      music=np.asarray(music, dtype="float32"), raw=np.asarray(raw),
                      music_audible=np.asarray(audible), starts=np.asarray(starts))
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError(f"Nonfinite model output: {path}")
        return result

    def cached(self, path: Path, cache: Path) -> dict[str, np.ndarray]:
        identity = sha256(path) + ":" + self.cache_id
        target = cache / (hashlib.sha256(identity.encode()).hexdigest() + ".npz")
        existing = load_cache(target, identity)
        if existing is not None:
            return existing
        result = self.extract(path)
        cache.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, identity=np.array(identity), **result)
        temporary.replace(target)
        return result
