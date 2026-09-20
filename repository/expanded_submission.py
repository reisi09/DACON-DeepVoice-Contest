"""Offline inference for the Dev-selected expanded-data MIL ensemble."""
from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

# Import the established runtime first: it enables offline mode before model loading.
from mert12_submission import (
    DEMUCS_INPUT_DECODER, SEEDS, SubmissionError, AutoFeatureExtractor, AutoModel,
    PREDICTION_COLUMNS, SUBMISSION_COLUMNS, baseline_module, combine, extract_mert12,
    pin_torchaudio_loader, sha256, validate_predictions,
)
from localization_models import WindowMIL

CANDIDATE = "expanded3659_MERT12_MIL_3seed"


def load_heads(paths: Sequence[Path], device: str) -> list[WindowMIL]:
    """Load the unchanged selected checkpoints, including fitted scaler buffers."""
    if len(paths) != len(SEEDS):
        raise SubmissionError("Exactly three selected MIL checkpoints are required")
    heads: list[WindowMIL] = []
    for path, seed, update in zip(paths, SEEDS, (96, 72, 72), strict=True):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        state = saved["model"]
        if (saved["arm"] != "mil" or saved["seed"] != seed
                or saved["selected_update"] != update or state["mean"].shape != (1024,)
                or any(not torch.isfinite(value).all() for value in state.values())):
            raise SubmissionError(f"Invalid selected MIL checkpoint: {path.name}")
        model = WindowMIL(state["mean"], state["scale"])
        model.load_state_dict(state, strict=True)
        heads.append(model.to(device).eval().requires_grad_(False))
    return heads


def predict_music(heads: Sequence[WindowMIL], feature_windows: np.ndarray, device: str) -> float:
    values = np.asarray(feature_windows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 1024 or not len(values) or not np.isfinite(values).all():
        raise SubmissionError("Expected finite non-empty 1024-wide MERT windows")
    features = torch.as_tensor(values, device=device)[None]
    mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
    probabilities: list[float] = []
    with torch.inference_mode():
        for model in heads:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(features, mask).file_logits
            probabilities.append(float(logits.float().sigmoid()[0].cpu()))
    return float(np.mean(probabilities))


def verify_package(root: Path) -> dict:
    manifest = json.loads((root / "model/manifest.json").read_text())
    choice = manifest["choice"]
    if (choice["candidate"] != CANDIDATE or choice["seeds"] != list(SEEDS)
            or choice["file_fusion"] != "max" or choice["selection"] != "dev_only"
            or choice["demucs_input_decoder"] != DEMUCS_INPUT_DECODER
            or choice["checkpoints"] != [f"model/trained/mil_{seed}.pt" for seed in SEEDS]):
        raise SubmissionError("Package choice differs from the selected expanded MIL recipe")
    for name, expected in manifest["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise SubmissionError(f"Packaged member hash mismatch: {name}")
    lock = json.loads((root / "model/trained/candidate_lock.json").read_text())
    if lock["winner_by_dev_only"] != "mil":
        raise SubmissionError("Candidate is not the frozen Dev winner")
    for name in choice["checkpoints"]:
        if manifest["sha256"].get(name) != lock["checkpoint_sha256"][Path(name).name]:
            raise SubmissionError("Checkpoint does not match the pre-test selection lock")
    return manifest


def run(root: Path) -> None:
    began = time.monotonic()
    root = root.resolve()
    manifest = verify_package(root)
    assets = root / "model/assets"
    base = baseline_module(assets)
    pin_torchaudio_loader(base)
    args = base.parse_arguments()
    device = base.select_device(args.device)
    if str(device) == "cuda" and not torch.cuda.is_available():
        raise SubmissionError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if str(device) == "cuda":
        torch.cuda.reset_peak_memory_stats()
    columns, rows = base.read_sample_submission(args.sample_submission)
    if columns != list(SUBMISSION_COLUMNS):
        raise SubmissionError("Unexpected sample-submission columns")
    files = base.order_audio_files(base.find_audio_files(args.test_dir), rows)
    presence = base.predict_presence_for_all_files(files, device)
    df, fake_index = base.load_df_arena_model(device)
    demucs = base.load_htdemucs_model()
    processor = AutoFeatureExtractor.from_pretrained(assets / "mert", local_files_only=True)
    mert, info = AutoModel.from_pretrained(assets / "mert", trust_remote_code=True,
                                           local_files_only=True, output_loading_info=True)
    if any(info.values()):
        raise SubmissionError(f"MERT checkpoint did not load exactly: {info}")
    mert.to(device).eval().requires_grad_(False)
    heads = load_heads([root / name for name in manifest["choice"]["checkpoints"]], str(device))
    total_windows = 0
    for index, (row, path) in enumerate(zip(rows, files, strict=True), 1):
        audio = base.load_audio(path)
        with torch.inference_mode():
            vocal, backing = base.separate_voice_and_music(path, demucs, device)
            voice = base.predict_fake(df, fake_index, vocal, device)
        backing = np.pad(backing[:len(audio)], (0, max(0, len(audio) - len(backing))))
        features, starts = extract_mert12(backing, processor, mert, str(device))
        music = predict_music(heads, features, str(device))
        values = combine(voice, music, presence[path.stem])
        total_windows += len(starts)
        row.update({key: round(value, 10) for key, value in zip(PREDICTION_COLUMNS, values, strict=True)})
        print(f"Expanded MIL {index}/{len(files)} {path.stem}", flush=True)
    validate_predictions(rows, [row["ID"] for row in rows])
    temporary = args.output.with_suffix(".csv.partial")
    base.save_submission(temporary, columns, rows)
    temporary.replace(args.output)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {"files": len(rows), "windows": total_windows, "device": args.device,
               "elapsed_seconds": time.monotonic() - began,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if str(device) == "cuda" else 0,
               "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if str(device) == "cuda" else 0,
               "peak_rss_bytes": int(peak_rss if sys.platform == "darwin" else peak_rss * 1024),
               "submission_sha256": sha256(args.output), "manifest_sha256": sha256(root / "model/manifest.json"),
               "official_l4_validated": False, "window_pooling": "window_logmeanexp_temperature1_three_seed_probability_mean"}
    receipt["demucs_input_decoder"] = DEMUCS_INPUT_DECODER
    (args.output.parent / "runtime_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

