"""Bounded VESSL H100 execution for the portable redesign notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import stat
import time
import traceback
import zipfile
from pathlib import Path

ROOT_DEFAULT = Path("/root/redesign_h100_20260917")
SHARED_DEFAULT = Path("/shared/redesign-h100-20260917")
ASSET_SOURCE = Path("/root/fast_h100_20260917/assets")
PYTHON_DEFAULT = Path("/root/fullft8/.venv/bin/python")
CAP_SECONDS, RESERVE_SECONDS = 4 * 3600, 15 * 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def one(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def safe_extract(archive: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as packed:
        infos = packed.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("input ZIP contains duplicate members")
        for item in infos:
            relative, target = Path(item.filename), destination / item.filename
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(item.external_attr >> 16) or not target.resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe input ZIP member: {item.filename}")
        if packed.testzip() is not None:
            raise RuntimeError("input ZIP CRC verification failed")
        try:
            declared = json.loads(packed.read("INPUT_SHA256.json"))
        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("input ZIP lacks a valid INPUT_SHA256.json") from exc
        files = {item.filename for item in infos if not item.is_dir() and item.filename != "INPUT_SHA256.json"}
        if not isinstance(declared, dict) or set(declared) != files:
            raise RuntimeError("INPUT_SHA256.json does not cover every other file exactly")
        for item in infos:
            if item.filename not in files:
                continue
            digest = hashlib.sha256()
            with packed.open(item) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != declared[item.filename]:
                raise RuntimeError(f"input member SHA-256 mismatch: {item.filename}")
        packed.extractall(destination)


def materialize_assets(manifest_path: Path, destination: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("asset_hashes.json is empty or invalid")
    methods = {"existing": 0, "hardlink": 0, "copy": 0}
    destination.mkdir(parents=True, exist_ok=True)
    for relative, expected in manifest.items():
        source = ASSET_SOURCE / relative
        target = destination / relative
        if not source.is_file() or sha256(source) != expected:
            raise RuntimeError(f"frozen asset source mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or sha256(target) != expected:
                raise RuntimeError(f"existing destination asset mismatch: {relative}")
            methods["existing"] += 1
            continue
        try:
            os.link(source.resolve(), target)
            methods["hardlink"] += 1
        except OSError:
            shutil.copy2(source.resolve(), target)
            methods["copy"] += 1
        if target.is_symlink() or sha256(target) != expected:
            raise RuntimeError(f"materialized asset mismatch: {relative}")
    shutil.copy2(manifest_path, destination / "asset_hashes.json")
    return {"members": len(manifest), "methods": methods, "manifest_sha256": sha256(manifest_path)}


def patch_notebook(source: Path, output: Path, args: argparse.Namespace) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    flags = {
        "prepare_data": False, "prepare_assets": False, "extract_features": True,
        "train_experiments": True, "fit_file_relation": True, "compare_and_lock": False,
        "open_sealed": False, "export_results": True, "cpu_integration": False,
    }
    config = f'''import os\nfrom pathlib import Path\nROOT = Path(os.environ["DACON_REDESIGN_ROOT"])\nEXISTING_LOCAL_WORKSPACE = Path("/nonexistent")\nINPUT_MANIFEST = Path(os.environ["DACON_INPUT_MANIFEST"])\nBASELINE_ZIP = None\nBASELINE_ZIP_SHA256 = ""\nASSETS = Path(os.environ["DACON_ASSETS"])\nDATA = Path(os.environ["DACON_DATA"])\nCACHE = Path(os.environ["DACON_CACHE"])\nOUTPUT = Path(os.environ["DACON_OUTPUT"])\nSEEDS = {tuple(args.seeds)!r}\nARMS = {tuple(args.arms)!r}\nCANDIDATE = {args.candidate!r}\nLOCKED_CHECKPOINTS = ()\nLOCKED_VOICE_CHECKPOINTS = ()\nGPU_DEVICE = "cuda"\nINSTALL_DEPENDENCIES = False\nPLOT_RESULTS = False\nEPOCHS = {args.epochs}\nPATIENCE = {args.patience}\nBATCH_SIZE = {args.batch_size}\nFOLDS = {args.folds}\nFLAGS = {flags!r}\nROOT.mkdir(parents=True, exist_ok=True)\nprint({{"root": str(ROOT), "flags": FLAGS, "seeds": SEEDS, "arms": ARMS, "diagnostic_only": True}})\n'''
    notebook["cells"][2]["source"] = config.splitlines(keepends=True)
    found_train = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        if "run_experiments(DATA, CACHE" in text:
            text = text.replace("epochs=300, patience=20, batch_size=64", "epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE")
            text = text.replace("arms=ARMS, folds=5", "arms=ARMS, folds=FOLDS")
            found_train = True
        cell["source"] = text.splitlines(keepends=True)
    if not found_train:
        raise RuntimeError("notebook training knob injection failed")
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def run_pilot(python: Path, source_root: Path, root: Path, deadline: float) -> dict:
    script = root / "pilot.py"
    script.write_text(
        "import json,time\nfrom pathlib import Path\nfrom redesign_data import read_rows\n"
        "from redesign_features import extract_features\n"
        f"root=Path({str(root)!r}); rows=read_rows(root/'data/train.csv')+read_rows(root/'data/validation.csv')\n"
        "sample=rows[:min(100,len(rows))]; began=time.time()\n"
        f"result=extract_features(sample,root/'assets',root/'cache',device='cuda',deadline={deadline!r})\n"
        "elapsed=time.time()-began; remaining=len(rows)-len(sample)\n"
        "receipt={'sample_rows':len(sample),'total_rows':len(rows),'elapsed_seconds':elapsed,"
        "'estimated_total_feature_seconds':elapsed/len(sample)*len(rows),'remaining_rows':remaining,'extract':result}\n"
        "(root/'output/pilot.json').write_text(json.dumps(receipt,indent=2,default=str)); print(json.dumps(receipt))\n",
        encoding="utf-8",
    )
    env = os.environ | {"PYTHONPATH": str(source_root)}
    log = root / "output/pilot.log"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run([str(python), str(script)], env=env, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"100-row feature pilot failed with exit {process.returncode}")
    receipt = json.loads((root / "output/pilot.json").read_text(encoding="utf-8"))
    if time.time() + float(receipt["estimated_total_feature_seconds"]) * 1.25 >= deadline:
        raise RuntimeError("pilot estimate does not fit before the export reserve")
    return receipt


def mark_diagnostic(output: Path) -> None:
    forbidden = list(output.rglob("locked_candidate.json")) + list(output.rglob("sealed_report.json"))
    if forbidden:
        raise RuntimeError(f"diagnostic run produced forbidden lock/sealed artifacts: {forbidden}")
    for path in output.rglob("comparison*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["diagnostic_metric_status"] = value.get("status")
        value["status"] = "diagnostic_only_not_accepted"
        value["diagnostic_only"] = True
        atomic_json(path, value)


def diagnostic_reports(source_root: Path, output: Path) -> dict:
    sys.path.insert(0, str(source_root))
    from redesign_evaluate import compare
    predictions = output / "experiments/predictions"
    reports = {}
    pairs = {
        "A_vs_B": (predictions / "A_ensemble.json", predictions / "B_ensemble.json"),
        "file_max_vs_noisy_or": (output / "predictions/B_ensemble_max.json", output / "predictions/B_ensemble_noisy_or.json"),
        "file_max_vs_oof_relation": (output / "predictions/B_ensemble_max.json", output / "predictions/B_ensemble_oof_relation.json"),
    }
    for name, (control, candidate) in pairs.items():
        target = output / f"comparison_{name}.json"
        reports[name] = compare(control, candidate, target, bootstrap=2000, seed=41)
    summary = json.loads((output / "experiments/summary.json").read_text(encoding="utf-8"))
    scope = {"diagnostic_only": True, "accepted": False, "historical_reused_data": True,
             "run_arms": ["A", "B"], "skipped_arms": {"C": "no verified paired pure/mixed slots; C duplicates A exposure",
             "D": "no verified paired pure/mixed slots; D duplicates B exposure"},
             "capacity_control": {"status": "not_run", "reason": "optional post-main control was not placed ahead of bounded export"},
             "voice": summary.get("voice"), "reports": list(reports)}
    atomic_json(output / "diagnostic_scope.json", scope)
    return scope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--shared", type=Path, default=SHARED_DEFAULT)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--created-unix", type=float, required=True)
    parser.add_argument("--seeds", type=lambda raw: tuple(map(int, raw.split(","))), default=(41, 42, 43))
    parser.add_argument("--arms", type=lambda raw: tuple(raw.split(",")), default=("A", "B"))
    parser.add_argument("--candidate", default="B_ensemble")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    root, shared = args.root, args.shared
    output, evidence = root / "output", root / "evidence"
    root.mkdir(parents=True, exist_ok=True); output.mkdir(exist_ok=True); evidence.mkdir(exist_ok=True); shared.mkdir(parents=True, exist_ok=True)
    hard, stage = args.created_unix + CAP_SECONDS, args.created_unix + CAP_SECONDS - RESERVE_SECONDS
    state = {"schema_version": 1, "status": "running", "archive_ready": False, "started_unix": args.created_unix,
             "hard_deadline_unix": hard, "stage_deadline_unix": stage, "updated_unix": time.time(), "finished_unix": None,
             "error": None, "archive": str(shared / "results.zip"), "archive_sha256": None, "archive_bytes": None,
             "archive_verified": False, "export_member_count": 0, "notebook_exit_code": None}
    atomic_json(shared / "run_state.json", state)
    status, error = "failed", None
    try:
        if time.time() >= stage:
            raise RuntimeError("workspace reached export reserve before setup")
        if sha256(args.input_zip) != args.input_sha256:
            raise RuntimeError("input ZIP differs from pinned sidecar SHA-256")
        source_root = root / "input"
        safe_extract(args.input_zip, source_root)
        manifest, notebook, sources = one(source_root, "asset_hashes.json"), one(source_root, "DACON_Redesign_H100_Experiments.ipynb"), one(source_root, "sources.csv")
        sys.path.insert(0, str(source_root))
        from redesign_notebook_runtime import require_storage, storage_plan
        from redesign_assets import validate_assets
        from redesign_data import prepare_data
        asset_receipt = materialize_assets(manifest, root / "assets")
        validate_assets(root / "assets")
        atomic_json(evidence / "assets.json", asset_receipt)
        audit = prepare_data(sources, root / "data", seed=41, diagnostic=True, make_mixtures=True)
        atomic_json(evidence / "data_audit_copy.json", audit)
        plan = storage_plan(root, sources, None); require_storage(plan); atomic_json(evidence / "storage_plan.json", plan)
        atomic_json(output / "gpu_budget.json", {"started_unix": args.created_unix, "hard_deadline_unix": hard, "stage_deadline_unix": stage})
        imports = "import torch,torchaudio,torchvision,numpy,scipy,sklearn,soundfile,transformers,demucs,panns_inference,librosa,timm,diffq,einops,nbformat,nbclient,ipykernel; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
        checked = subprocess.run([str(args.python), "-c", imports], capture_output=True, text=True)
        atomic_json(evidence / "cuda_import_check.json", {"returncode": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr})
        if checked.returncode:
            raise RuntimeError("frozen runtime import/CUDA check failed")
        run_pilot(args.python, source_root, root, stage)
        patched, executed = root / "DACON_Redesign_H100_Experiments.run.ipynb", root / "DACON_Redesign_H100_Experiments.executed.ipynb"
        patch_notebook(notebook, patched, args)
        kernel = root / "jupyter/kernels/redesign-h100"; kernel.mkdir(parents=True)
        atomic_json(kernel / "kernel.json", {"argv": [str(args.python), "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "redesign-h100", "language": "python"})
        driver = root / "execute_notebook.py"
        driver.write_text("import nbformat,sys\nfrom nbclient import NotebookClient\np=nbformat.read(sys.argv[1],4)\ntry: NotebookClient(p,timeout=None,kernel_name='redesign-h100').execute()\nfinally: nbformat.write(p,sys.argv[2])\n", encoding="utf-8")
        env = os.environ | {"PYTHONPATH": str(source_root), "DACON_REDESIGN_ROOT": str(root), "DACON_INPUT_MANIFEST": str(sources),
                            "DACON_ASSETS": str(root / "assets"), "DACON_DATA": str(root / "data"), "DACON_CACHE": str(root / "cache"), "DACON_OUTPUT": str(output),
                            "JUPYTER_PATH": str(root / "jupyter")}
        with (output / "notebook_execution.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run([str(args.python), str(driver), str(patched), str(executed)], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, stage - time.time()))
                state["notebook_exit_code"] = completed.returncode
                if completed.returncode:
                    raise RuntimeError(f"notebook execution failed with exit {completed.returncode}")
            except subprocess.TimeoutExpired as exc:
                state["notebook_exit_code"] = 124
                raise RuntimeError("notebook exceeded the training-stage deadline") from exc
        diagnostic_reports(source_root, output)
        status = "exported"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (evidence / "failure.txt").write_text(error + "\n" + traceback.format_exc(), encoding="utf-8")
    finally:
        try:
            mark_diagnostic(output)
        except BaseException as diagnostic_error:
            status = "failed"
            error = f"{error}; diagnostic guard: {diagnostic_error}" if error else f"diagnostic guard: {diagnostic_error}"
        atomic_json(evidence / "run_state_before_export.json", state | {"status": status, "error": error, "updated_unix": time.time()})
        try:
            sys.path.insert(0, str(root / "input"))
            from redesign_notebook_runtime import export_results
            archive = shared / "results.zip"
            receipt = export_results(tuple(path for path in (output, evidence, root / "DACON_Redesign_H100_Experiments.run.ipynb", root / "DACON_Redesign_H100_Experiments.executed.ipynb") if path.exists()), archive)
            archive_status = "exported" if status == "exported" else "partial_exported"
            export_receipt = {"schema_version": 1, "archive": str(archive), "archive_sha256": receipt["archive_sha256"],
                              "archive_bytes": receipt["bytes"], "archive_ready": True, "archive_verified": receipt["verified"],
                              "export_member_count": receipt["member_count"], "exported_unix": time.time(), "status": archive_status}
            atomic_json(shared / "export_receipt.json", export_receipt)
            (shared / "results.zip.sha256").write_text(f'{receipt["archive_sha256"]}  results.zip\n', encoding="utf-8")
            state.update(archive_ready=True, archive_sha256=receipt["archive_sha256"], archive_bytes=receipt["bytes"],
                         archive_verified=receipt["verified"], export_member_count=receipt["member_count"])
        except BaseException as export_error:
            error = f"{error}; export: {type(export_error).__name__}: {export_error}" if error else f"export: {export_error}"
        state.update(status=status if state["archive_ready"] else "failed", error=error, updated_unix=time.time(), finished_unix=time.time())
        atomic_json(shared / "run_state.json", state)
    return 0 if status == "exported" and state["archive_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
