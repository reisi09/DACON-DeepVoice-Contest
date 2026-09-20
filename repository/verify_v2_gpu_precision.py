"""Diagnose A100 package parity without changing the 1e-5 acceptance limit."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

from ast_multitask import PREDICTION_COLUMNS, load_checkpoint
from graph_features import atomic_json, digest
from package_v2_submission import CHECKPOINT_FILES
from v2_inference import ordered_audio_files, predict_audio_file


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _reference(checkpoint: Path, inputs: Path, output: Path, deadline: float) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU precision diagnostic")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mode = json.loads((checkpoint / "multitask_config.json").read_text())["mode"]
    model, processor = load_checkpoint(checkpoint, mode)
    model = model.cuda().eval()
    requested = [row["ID"] for row in _rows(inputs / "sample_submission.csv")]
    paths = ordered_audio_files(inputs / "test", requested)
    records = []
    for path in paths:
        if time.time() >= deadline:
            raise TimeoutError("reference prediction deadline reached")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        probabilities = predict_audio_file(model, processor, path)
        torch.cuda.synchronize()
        records.append({"ID": path.stem, **dict(zip(PREDICTION_COLUMNS, probabilities, strict=True)),
                        "seconds": time.perf_counter() - started,
                        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return records


def _child_code(cudnn: bool | None, matmul: bool | None) -> str:
    assignments = []
    if cudnn is not None:
        assignments.append(f"torch.backends.cudnn.allow_tf32={cudnn}")
    if matmul is not None:
        assignments.append(f"torch.backends.cuda.matmul.allow_tf32={matmul}")
    return ";".join([
        "import json,runpy,torch,time",
        *assignments,
        "torch.cuda.reset_peak_memory_stats()",
        "started=time.perf_counter()",
        "runpy.run_path('script.py',run_name='__main__')",
        "torch.cuda.synchronize()",
        "print('GPU_DIAGNOSTIC '+json.dumps({'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'seconds':time.perf_counter()-started,'max_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'max_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}))",
    ])


def _run_variant(name: str, flags: tuple[bool | None, bool | None], root: Path,
                 work: Path, deadline: float) -> tuple[list[dict[str, str]], dict]:
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError(f"deadline reached before {name}")
    result = subprocess.run(
        [sys.executable, "-c", _child_code(*flags)], cwd=root,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        text=True, capture_output=True, timeout=remaining, check=False,
    )
    (work / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    receipt = json.loads(next(line.removeprefix("GPU_DIAGNOSTIC ") for line in result.stdout.splitlines()
                              if line.startswith("GPU_DIAGNOSTIC ")))
    destination = work / f"{name}.csv"
    shutil.copy2(root / "output" / "submission.csv", destination)
    return _rows(destination), receipt


def _classify(variants: dict) -> tuple[str, dict]:
    delta = {name: value["max_delta"] for name, value in variants.items()}
    default = variants["default"]
    precision_state = delta["default"] > 1e-5 and delta["both_false"] <= 1e-5
    cudnn_signature = (default["cudnn_tf32"] is True and default["matmul_tf32"] is False
                       and delta["cudnn_true_matmul_false"] > 1e-5)
    diagnosis = ("confirmed_cudnn_tf32" if precision_state and cudnn_signature
                 else "confirmed_process_precision_state" if precision_state else "inconclusive")
    return diagnosis, {"cudnn_false_matmul_true_max_delta": delta["cudnn_false_matmul_true"],
                       "passes_1e-5": delta["cudnn_false_matmul_true"] <= 1e-5}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "package", "inputs", "work", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    reference = _reference(args.checkpoint, args.inputs, args.work / "reference.csv", args.deadline)
    unpacked = args.work / "unpacked"
    unpacked.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.package) as archive:
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in archive.namelist()):
            raise ValueError("unsafe package member")
        archive.extractall(unpacked)
    shutil.copytree(args.inputs, unpacked / "data", dirs_exist_ok=True)
    hashes = {name: {"reference": digest(args.checkpoint / name),
                     "package": digest(unpacked / "model" / "ast" / name)} for name in CHECKPOINT_FILES}
    if any(value["reference"] != value["package"] for value in hashes.values()):
        raise ValueError("package checkpoint assets differ from reference")
    variants = {"default": (None, None), "both_false": (False, False),
                "cudnn_false_matmul_true": (False, True),
                "cudnn_true_matmul_false": (True, False)}
    report = {"asset_sha256": hashes, "reference": reference, "variants": {}}
    for name, flags in variants.items():
        rows, receipt = _run_variant(name, flags, unpacked, args.work, args.deadline)
        by_id = {row["ID"]: row for row in rows}
        deltas = {row["ID"]: max(abs(float(row[column]) - float(by_id[row["ID"]][column]))
                                      for column in PREDICTION_COLUMNS) for row in reference}
        report["variants"][name] = {**receipt, "per_file_max_delta": deltas,
                                     "max_delta": max(deltas.values()), "csv": str(args.work / f"{name}.csv")}
        atomic_json(args.output, report)
    delta = {name: value["max_delta"] for name, value in report["variants"].items()}
    report["diagnosis"], report["secondary_sensitivity"] = _classify(report["variants"])
    report["threshold"] = 1e-5
    atomic_json(args.output, report)
    print(json.dumps({"diagnosis": report["diagnosis"], "deltas": delta}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
