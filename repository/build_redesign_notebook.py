#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5.10,<6"]
# ///
# noqa: SIZE_OK

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run build_redesign_notebook.py
# 3. Or make executable and run:
#      chmod +x build_redesign_notebook.py && ./build_redesign_notebook.py
# ──────────────────
"""Build the portable, source-embedded DACON redesign experiment notebook."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import textwrap
import zipfile
from pathlib import Path
from typing import Final

import nbformat as nb

ROOT: Final = Path(__file__).resolve().parent
NOTEBOOK: Final = ROOT / "DACON_Redesign_H100_Experiments.ipynb"
ENTRY_SOURCES: Final = (
    "graph_scoring.py",
    "music_temporal.py",
    "redesign_assets.py",
    "redesign_data.py",
    "redesign_evaluate.py",
    "redesign_features.py",
    "redesign_file.py",
    "redesign_notebook_integration.py",
    "redesign_notebook_results.py",
    "redesign_notebook_runtime.py",
    "redesign_train.py",
)


def code(source: str) -> nb.NotebookNode:
    """Dedent and syntax-check a generated Python cell."""
    clean = textwrap.dedent(source).strip()
    compile(clean, "redesign_notebook_cell", "exec")
    return nb.v4.new_code_cell(clean)


def source_payload(entry_sources: tuple[str, ...] = ENTRY_SOURCES) -> tuple[str, str, tuple[str, ...]]:
    """Compress the complete local redesign source closure without artifacts."""
    pending: list[str] = list(entry_sources)
    names: list[str] = []
    missing: list[str] = []
    while pending:
        name = pending.pop(0)
        if name in names:
            continue
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
            continue
        names.append(name)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        modules = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            for alias in (node.names if isinstance(node, ast.Import) else ())
        }
        modules.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        pending.extend(f"{module}.py" for module in sorted(modules) if (ROOT / f"{module}.py").is_file())
    if missing:
        raise FileNotFoundError("notebook source closure incomplete: " + ", ".join(sorted(set(missing))))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as packed:
        for name in names:
            source = (ROOT / name).read_bytes()
            compile(source, name, "exec")
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            packed.writestr(info, source)
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), hashlib.sha256(payload).hexdigest(), tuple(names)


def build() -> nb.NotebookNode:
    """Return the validated notebook document."""
    payload, payload_hash, source_names = source_payload()
    cells: list[nb.NotebookNode] = [
        nb.v4.new_markdown_cell(
            "# DACON 236749 재설계 H100 실험\n\n"
            "이 노트북은 검토된 A/B/C/D 실험과 File 관계식 검증을 실행한다. 기본 `Run All`은 CPU 사전 점검만 하고 "
            "학습·sealed 평가·제출은 실행하지 않는다. CPU 통합 모드는 알고리즘 연결을 확인하는 QA이며 성능 근거가 아니다. "
            "H100·L4와 실제 라벨 데이터 검증은 아직 수행되지 않았다."
        ),
        nb.v4.new_markdown_cell("## 1. 경로와 실행 플래그\n\n비용이 드는 단계는 사용자가 해당 플래그를 직접 켜야 한다."),
        code(
            """
            import os
            from pathlib import Path

            ROOT = Path(os.environ.get("DACON_REDESIGN_ROOT", "/workspace/dacon_redesign")).expanduser()
            EXISTING_LOCAL_WORKSPACE = Path("/Users/choiwuseck/Documents/ChatGPT/데이콘")  # 로컬에서만 선택적으로 사용
            INPUT_MANIFEST = Path(os.environ.get("DACON_INPUT_MANIFEST", str(ROOT / "config" / "sources.csv")))
            BASELINE_ZIP = Path(os.environ["DACON_BASELINE_ZIP"]) if os.environ.get("DACON_BASELINE_ZIP") else None
            BASELINE_ZIP_SHA256 = os.environ.get("DACON_BASELINE_ZIP_SHA256", "1f6cfd6c9feacead01d4ebcfd217d10c1796969b1f863ba4b0283d46e7432c7e")
            ASSETS = Path(os.environ.get("DACON_ASSETS", str(ROOT / "assets")))
            DATA = Path(os.environ.get("DACON_DATA", str(ROOT / "data")))
            CACHE = Path(os.environ.get("DACON_CACHE", str(ROOT / "cache")))
            OUTPUT = Path(os.environ.get("DACON_OUTPUT", str(ROOT / "results")))
            SEEDS = (41, 42, 43)
            ARMS = ("A", "B", "C", "D")
            CANDIDATE = os.environ.get("DACON_CANDIDATE", "D_ensemble")
            LOCKED_CHECKPOINTS = tuple(Path(item) for item in os.environ.get("DACON_LOCKED_CHECKPOINTS", "").split(",") if item)
            LOCKED_VOICE_CHECKPOINTS = tuple(Path(item) for item in os.environ.get("DACON_LOCKED_VOICE_CHECKPOINTS", "").split(",") if item)
            GPU_DEVICE = os.environ.get("DACON_DEVICE", "cuda")
            INSTALL_DEPENDENCIES = False
            PLOT_RESULTS = False
            FLAGS = {
                "prepare_data": False,
                "prepare_assets": False,
                "extract_features": False,
                "train_experiments": False,
                "fit_file_relation": False,
                "compare_and_lock": False,
                "open_sealed": False,
                "export_results": False,
                "cpu_integration": False,
            }
            ROOT.mkdir(parents=True, exist_ok=True)
            print({"root": str(ROOT), "flags": FLAGS, "seeds": SEEDS, "arms": ARMS})
            """
        ),
        nb.v4.new_markdown_cell("## 2. 내장 소스 복원\n\n가중치·오디오·비밀값은 포함하지 않았다. 압축 소스의 SHA-256을 확인한 뒤 안전하게 푼다."),
        code(
            f"""
            import base64, hashlib, io, stat, sys, zipfile

            SOURCE_PAYLOAD_B64 = {payload!r}
            SOURCE_PAYLOAD_SHA256 = {payload_hash!r}
            SOURCE_MEMBERS = {source_names!r}
            source_bytes = base64.b64decode(SOURCE_PAYLOAD_B64)
            assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_PAYLOAD_SHA256
            SOURCE_ROOT = ROOT / "source" / SOURCE_PAYLOAD_SHA256[:16]
            SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(source_bytes)) as packed:
                infos = packed.infolist()
                names = [item.filename for item in infos]
                if tuple(names) != SOURCE_MEMBERS or len(names) != len(set(names)):
                    raise RuntimeError("embedded source member contract mismatch")
                for item in infos:
                    relative = Path(item.filename)
                    target = SOURCE_ROOT / relative
                    if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(item.external_attr >> 16) or not target.resolve().is_relative_to(SOURCE_ROOT.resolve()):
                        raise RuntimeError(f"unsafe embedded member: {{item.filename}}")
                for item in infos:
                    target = SOURCE_ROOT / item.filename
                    expected = packed.read(item)
                    if target.exists() and target.read_bytes() != expected:
                        raise RuntimeError(f"existing versioned source differs: {{item.filename}}")
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(expected)
            stale = [name for name, module in sys.modules.items() if (name in {"graph_scoring", "music_temporal"} or name.startswith("redesign_")) and getattr(module, "__file__", None) and not Path(module.__file__).resolve().is_relative_to(SOURCE_ROOT.resolve())]
            if stale:
                raise RuntimeError(f"source payload changed after imports; restart the kernel: {{stale}}")
            if str(SOURCE_ROOT) not in sys.path:
                sys.path.insert(0, str(SOURCE_ROOT))
            print({{"source_sha256": SOURCE_PAYLOAD_SHA256, "members": len(SOURCE_MEMBERS)}})
            """
        ),
        nb.v4.new_markdown_cell("## 3. 재현 가능한 의존성 확인\n\n새 H100 환경에서만 `INSTALL_DEPENDENCIES=True`로 바꿔 실행한다. CUDA 12.8의 torch·torchaudio·torchvision은 한 묶음으로 고정한다."),
        code(
            """
            import importlib.util, subprocess

            TORCH_COMMAND = [
                sys.executable, "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu128",
                "torch==2.7.1+cu128", "torchaudio==2.7.1+cu128", "torchvision==0.22.1+cu128",
            ]
            RUNTIME_COMMAND = [
                sys.executable, "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu128",
                "torch==2.7.1+cu128", "torchaudio==2.7.1+cu128", "torchvision==0.22.1+cu128",
                "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1", "matplotlib==3.10.3",
                "librosa==0.11.0", "soundfile==0.13.1", "transformers==4.57.6",
                "demucs==4.0.1", "panns-inference==0.1.1", "diffq==0.2.4", "einops==0.8.1",
                "timm==1.0.15", "fvcore==0.1.5.post20221221",
            ]
            DEPENDENCY_PROVENANCE = "_workspace/mert330_mlp_input/requirements-backbone.txt + verified M330 submission torch/torchaudio/torchvision cu128 pins"
            if INSTALL_DEPENDENCIES:
                subprocess.run(TORCH_COMMAND, check=True)
                subprocess.run(RUNTIME_COMMAND, check=True)
                raise RuntimeError("dependencies installed; restart the kernel, keep INSTALL_DEPENDENCIES=False, and Run All")
            required_modules = ["numpy", "scipy", "sklearn", "torch"]
            if FLAGS["extract_features"] or FLAGS["prepare_assets"] or FLAGS["open_sealed"]:
                required_modules.extend(("soundfile", "torchaudio", "torchvision", "transformers", "demucs", "panns_inference", "librosa", "timm", "diffq", "einops"))
            missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
            if missing_modules:
                raise RuntimeError(f"missing runtime dependencies: {missing_modules}; set INSTALL_DEPENDENCIES=True once")
            print({"dependency_check": "passed", "install_executed": False, "provenance": DEPENDENCY_PROVENANCE})
            """
        ),
        nb.v4.new_markdown_cell("## 4. CPU 출처·저장소 점검\n\n이 단계는 GPU 4시간 예산을 시작하지 않는다. 빈 manifest 템플릿에는 계약 컬럼만 기록한다."),
        code(
            """
            import json, platform, time
            from redesign_data import write_source_template
            from redesign_notebook_runtime import record_stage, sha256_file, storage_plan, storage_receipt, write_json

            PROGRESS = OUTPUT / "progress.json"
            OUTPUT.mkdir(parents=True, exist_ok=True)
            if not INPUT_MANIFEST.exists():
                write_source_template(INPUT_MANIFEST)
            provenance = {
                "status": "preflight_only",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
                "manifest": str(INPUT_MANIFEST),
                "manifest_sha256": sha256_file(INPUT_MANIFEST),
                "baseline_zip": str(BASELINE_ZIP) if BASELINE_ZIP else None,
                "baseline_zip_sha256": sha256_file(BASELINE_ZIP) if BASELINE_ZIP else None,
                "actual_gpu_training_verified": False,
                "l4_runtime_verified": False,
                "checked_unix": time.time(),
            }
            write_json(OUTPUT / "preflight.json", provenance)
            write_json(OUTPUT / "storage.json", storage_receipt(ROOT))
            plan = storage_plan(ROOT, INPUT_MANIFEST, BASELINE_ZIP)
            write_json(OUTPUT / "storage_plan.json", plan)
            record_stage(PROGRESS, "preflight", "completed", detail="CPU provenance, hashes, and writable storage checked")
            print(json.dumps({"provenance": provenance, "storage_plan": plan}, indent=2, ensure_ascii=False))
            """
        ),
        nb.v4.new_markdown_cell("## 4. 데이터와 자산 준비\n\n실제 출처·라이선스·checksum이 있는 manifest만 처리한다. historical 자료는 `diagnostic=True`일 때만 진단용으로 둔다."),
        code(
            """
            from redesign_data import prepare_data
            from redesign_assets import prepare_assets
            from redesign_notebook_runtime import record_stage, require_storage

            if FLAGS["prepare_data"]:
                require_storage(plan)
                data_receipt = prepare_data(INPUT_MANIFEST, DATA, seed=41, diagnostic=False, make_mixtures=True)
                record_stage(PROGRESS, "prepare_data", "completed", detail=json.dumps(data_receipt, default=str))
            else:
                record_stage(PROGRESS, "prepare_data", "not_run", detail="flag is False")
            if FLAGS["prepare_assets"]:
                if BASELINE_ZIP is None:
                    raise RuntimeError("DACON_BASELINE_ZIP is required for asset preparation")
                if not BASELINE_ZIP_SHA256:
                    raise RuntimeError("DACON_BASELINE_ZIP_SHA256 pinned receipt is required")
                require_storage(plan)
                asset_receipt = prepare_assets(BASELINE_ZIP, ASSETS, expected_sha256=BASELINE_ZIP_SHA256)
                record_stage(PROGRESS, "prepare_assets", "completed", detail=json.dumps(asset_receipt, default=str))
            else:
                record_stage(PROGRESS, "prepare_assets", "not_run", detail="flag is False")
            """
        ),
        nb.v4.new_markdown_cell("## 5. 특징 추출과 A/B/C/D 학습\n\nGPU 예산은 첫 비용 단계에서 한 번만 시작하며 재실행해도 초기 시각을 유지한다. sealed는 이 단계에서 읽지 않는다."),
        code(
            """
            from redesign_data import read_rows
            from redesign_features import extract_features
            from redesign_notebook_runtime import Budget, NotebookContractError, export_results, record_stage, require_storage, write_json
            from redesign_train import run_experiments

            costly = FLAGS["extract_features"] or FLAGS["train_experiments"]
            budget = Budget.load_or_start(OUTPUT / "gpu_budget.json") if costly else None
            active_stage = "costly_stage"
            try:
                if FLAGS["extract_features"]:
                    active_stage = "extract_features"
                    require_storage(plan)
                    budget.require_stage_time()
                    rows = read_rows(DATA / "train.csv") + read_rows(DATA / "validation.csv")
                    feature_receipt = extract_features(rows, ASSETS, CACHE, device=GPU_DEVICE, deadline=budget.stage_deadline_unix, views=("stem", "raw"))
                    record_stage(PROGRESS, "extract_features", "completed", detail=json.dumps(feature_receipt, default=str))
                else:
                    record_stage(PROGRESS, "extract_features", "not_run", detail="flag is False; no fixture fallback")
                if FLAGS["train_experiments"]:
                    active_stage = "train_experiments"
                    budget.require_stage_time()
                    train_receipt = run_experiments(DATA, CACHE, OUTPUT / "experiments", seeds=SEEDS, epochs=300, patience=20, batch_size=64, device=GPU_DEVICE, deadline=budget.stage_deadline_unix, arms=ARMS, folds=5)
                    record_stage(PROGRESS, "train_experiments", "completed", detail=json.dumps(train_receipt, default=str))
                else:
                    record_stage(PROGRESS, "train_experiments", "not_run", detail="flag is False; A/B/C/D/capacity/voice not claimed")
            except (TimeoutError, KeyboardInterrupt, NotebookContractError) as error:
                record_stage(PROGRESS, active_stage, "interrupted", detail=f"{type(error).__name__}: {error}")
                interrupted_export = export_results((OUTPUT,), ROOT / "exports" / "interrupted_results.zip")
                write_json(ROOT / "exports" / "interrupted_readback.json", interrupted_export)
                print(json.dumps({"interrupted_stage": active_stage, "partial_export": interrupted_export}, indent=2))
                raise
            """
        ),
        nb.v4.new_markdown_cell("## 6. 후보별 결과 읽기\n\nseed별 값, seed 평균±SD, probability ensemble의 EER를 분리한다. 계산할 수 없는 공식 총점은 `None`으로 남긴다."),
        code(
            """
            from redesign_notebook_results import summarize_predictions

            prediction_directory = OUTPUT / "experiments" / "predictions"
            result_rows = summarize_predictions(prediction_directory) if prediction_directory.is_dir() else []
            if result_rows:
                print(json.dumps(result_rows, indent=2, ensure_ascii=False))
                if PLOT_RESULTS:
                    import matplotlib.pyplot as plt
                    ensemble_rows = [row for row in result_rows if row["kind"] == "probability_ensemble" and row.get("music_eer") is not None]
                    if ensemble_rows:
                        plt.figure(figsize=(7, 3.5))
                        plt.bar([row["candidate"] for row in ensemble_rows], [row["music_eer"] for row in ensemble_rows])
                        plt.ylabel("Music EER")
                        plt.title("Validation probability ensemble Music EER")
                        plt.xticks(rotation=30, ha="right")
                        plt.tight_layout()
                        plt.show()
            else:
                print({"status": "not_run", "reason": "prediction artifacts are absent"})
            """
        ),
        nb.v4.new_markdown_cell("## 7. File max·noisy-OR·OOF 관계식과 grouped 비교\n\n관계식은 train OOF에만 맞추고 validation 보고서가 실제 라벨 gate를 통과할 때만 후보를 잠근다."),
        code(
            """
            from redesign_evaluate import compare, lock_candidate
            from redesign_file import apply_relation, fit_relation
            from redesign_notebook_runtime import record_stage, write_file_control

            prediction_path = OUTPUT / "experiments" / "predictions" / f"{CANDIDATE}.json"
            reference_path = OUTPUT / "experiments" / "predictions" / "A_ensemble.json"
            oof_path = OUTPUT / "experiments" / "oof" / f"{CANDIDATE}.json"
            relation_path = OUTPUT / "file_relation.json"
            reference_max_path = OUTPUT / "predictions" / "A_ensemble_max.json"
            max_path = OUTPUT / "predictions" / f"{CANDIDATE}_max.json"
            noisy_path = OUTPUT / "predictions" / f"{CANDIDATE}_noisy_or.json"
            related_path = OUTPUT / "predictions" / f"{CANDIDATE}_oof_relation.json"
            noisy_report_path = OUTPUT / "comparison_noisy_or.json"
            relation_report_path = OUTPUT / "comparison_oof_relation.json"
            report_path = OUTPUT / "comparison_primary_vs_A.json"
            lock_path = OUTPUT / "locked_candidate.json"
            if FLAGS["fit_file_relation"]:
                if not reference_path.is_file():
                    raise RuntimeError("A_ensemble reference is required; candidate cannot be locked without the fixed control")
                reference_receipt = write_file_control(reference_path, reference_max_path, noisy_or=False)
                max_receipt = write_file_control(prediction_path, max_path, noisy_or=False)
                noisy_receipt = write_file_control(prediction_path, noisy_path, noisy_or=True)
                relation_receipt = fit_relation(oof_path, relation_path)
                apply_receipt = apply_relation(prediction_path, relation_path, related_path)
                record_stage(PROGRESS, "fit_file_relation", "completed", detail=json.dumps({"reference_max": reference_receipt, "max": max_receipt, "noisy_or": noisy_receipt, "fit": relation_receipt, "apply": apply_receipt}, default=str))
            else:
                record_stage(PROGRESS, "fit_file_relation", "not_run", detail="max/noisy-OR/OOF fitting not run")
            if FLAGS["compare_and_lock"]:
                data_audit = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))
                panel_identity = data_audit["sealed_panel_identity"]
                compare(max_path, noisy_path, noisy_report_path, bootstrap=2000, seed=41)
                relation_comparison = compare(max_path, related_path, relation_report_path, bootstrap=2000, seed=41)
                architecture_pairs = (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"))
                architecture_reports = {}
                for control_arm, candidate_arm in architecture_pairs:
                    control_artifact = OUTPUT / "experiments" / "predictions" / f"{control_arm}_ensemble.json"
                    candidate_artifact = OUTPUT / "experiments" / "predictions" / f"{candidate_arm}_ensemble.json"
                    if not control_artifact.is_file() or not candidate_artifact.is_file():
                        raise RuntimeError(f"missing causal comparison artifact: {control_arm}->{candidate_arm}")
                    architecture_reports[f"{control_arm}_to_{candidate_arm}"] = compare(
                        control_artifact, candidate_artifact,
                        OUTPUT / f"comparison_{control_arm}_to_{candidate_arm}.json",
                        bootstrap=2000, seed=41,
                    )
                comparison = compare(reference_max_path, related_path, report_path, bootstrap=2000, seed=41)
                eligible = comparison.get("status") in {"accepted", "provisional"} and comparison.get("comparison", {}).get("statistical_eligible") is True and comparison.get("comparison", {}).get("validation_eligible") is True
                if eligible:
                    lock = lock_candidate(related_path, report_path, lock_path, sealed_panel_identity=panel_identity)
                    detail = {"primary_comparison": comparison, "file_relation_diagnostic": relation_comparison, "architecture_reports": architecture_reports, "lock": lock}
                else:
                    detail = {"primary_comparison": comparison, "file_relation_diagnostic": relation_comparison, "architecture_reports": architecture_reports, "lock": None, "reason": "candidate is not eligible to lock against A_ensemble"}
                record_stage(PROGRESS, "compare_and_lock", "completed", detail=json.dumps(detail, default=str))
            else:
                record_stage(PROGRESS, "compare_and_lock", "not_run", detail="no candidate accepted or locked")
            """
        ),
        nb.v4.new_markdown_cell("## 7. 선택적 sealed 실행\n\n잠긴 checkpoint로 예측한 sealed 결과만 한 번 연다. 이 플래그는 기본적으로 꺼져 있다."),
        code(
            """
            from redesign_evaluate import open_sealed
            from redesign_features import load_records
            from redesign_notebook_runtime import record_stage
            from redesign_train import predict_ensemble_artifact

            if FLAGS["open_sealed"]:
                if not lock_path.is_file():
                    raise RuntimeError("locked_candidate.json is required before sealed evaluation")
                if not LOCKED_CHECKPOINTS:
                    raise RuntimeError("DACON_LOCKED_CHECKPOINTS must list the locked ensemble checkpoints")
                locked = json.loads(lock_path.read_text(encoding="utf-8"))
                data_audit = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))
                panel_identity = data_audit["sealed_panel_identity"]
                if panel_identity != locked.get("sealed_panel_identity"):
                    raise RuntimeError("configured sealed panel does not match the candidate lock")
                locked_validation = json.loads(related_path.read_text(encoding="utf-8"))
                identity = locked_validation["upstream_identity"]
                identity_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if sha256_file(related_path) != locked["prediction_sha256"] or identity_sha256 != locked["model_identity_sha256"]:
                    raise RuntimeError("locked validation artifact hash or model identity differs")
                music_hashes = [sha256_file(path) for path in LOCKED_CHECKPOINTS]
                if music_hashes != identity.get("seed_checkpoint_sha256"):
                    raise RuntimeError("configured music checkpoints do not match the locked ordered weights")
                expected_voice_hashes = identity.get("seed_voice_checkpoint_sha256")
                if expected_voice_hashes is not None:
                    voice_hashes = [sha256_file(path) for path in LOCKED_VOICE_CHECKPOINTS]
                    if voice_hashes != expected_voice_hashes:
                        raise RuntimeError("configured voice checkpoints do not match the locked ordered weights")
                elif LOCKED_VOICE_CHECKPOINTS:
                    raise RuntimeError("voice checkpoints were supplied for a music-only lock")
                sealed_budget = Budget.load_or_start(OUTPUT / "gpu_budget.json")
                sealed_budget.require_stage_time()
                sealed_rows = read_rows(DATA / "sealed.csv")
                extract_features(sealed_rows, ASSETS, CACHE, device=GPU_DEVICE, deadline=sealed_budget.stage_deadline_unix, views=("stem", "raw"))
                sealed_records = load_records(sealed_rows, CACHE)
                train_manifest_sha256 = sha256_file(DATA / "train.csv")
                records_identity = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))["receipt_sha256"]
                raw_sealed_path = OUTPUT / "sealed_predictions_raw.json"
                sealed_prediction_path = OUTPUT / "sealed_predictions.json"
                sealed_predictions = predict_ensemble_artifact(LOCKED_CHECKPOINTS, sealed_rows, sealed_records, training_manifest_sha256=train_manifest_sha256, records_identity=records_identity, candidate=locked["candidate"], device=GPU_DEVICE, voice_checkpoints=LOCKED_VOICE_CHECKPOINTS or None)
                raw_sealed_path.write_text(json.dumps(sealed_predictions, indent=2, sort_keys=True), encoding="utf-8")
                apply_relation(raw_sealed_path, relation_path, sealed_prediction_path)
                sealed_identity = json.loads(sealed_prediction_path.read_text(encoding="utf-8"))["upstream_identity"]
                sealed_identity_sha256 = hashlib.sha256(json.dumps(sealed_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if sealed_identity_sha256 != locked["model_identity_sha256"]:
                    raise RuntimeError("sealed prediction identity differs; access registry was not consumed")
                sealed_receipt = open_sealed(lock_path, sealed_prediction_path, OUTPUT / "sealed_report.json", panel_identity=panel_identity, access_registry=DATA / ".sealed_access_registry")
                record_stage(PROGRESS, "open_sealed", "completed", detail=json.dumps(sealed_receipt, default=str))
                print(json.dumps(sealed_receipt, indent=2, ensure_ascii=False))
            else:
                record_stage(PROGRESS, "open_sealed", "not_run", detail="explicit sealed flag is False")
            """
        ),
        nb.v4.new_markdown_cell("## 8. CPU 통합 QA와 결과 내보내기\n\nCPU fixture는 결정적 labeled features로 실제 저장·재로딩·예측·OOF·평가·ZIP readback을 검사한다. 성능 선택에는 쓰지 않는다."),
        code(
            """
            from redesign_notebook_integration import run_cpu_integration
            from redesign_notebook_runtime import export_results, record_stage, write_json

            if FLAGS["cpu_integration"]:
                integration = run_cpu_integration(ROOT / "cpu_integration")
                integration_results = summarize_predictions(ROOT / "cpu_integration" / "experiments" / "predictions")
                print(json.dumps({"integration": integration, "candidate_seed_ensemble_metrics": integration_results}, indent=2, ensure_ascii=False))
                record_stage(PROGRESS, "cpu_integration", "completed", detail=json.dumps(integration, default=str))
            else:
                record_stage(PROGRESS, "cpu_integration", "not_run", detail="flag is False; algorithm QA not run")
            if FLAGS["export_results"]:
                export = export_results((OUTPUT, ROOT / "cpu_integration"), ROOT / "exports" / "redesign_results.zip")
                write_json(ROOT / "exports" / "readback.json", export)
                record_stage(PROGRESS, "export_results", "completed", detail=json.dumps(export, default=str))
                print(json.dumps(export, indent=2))
            else:
                record_stage(PROGRESS, "export_results", "not_run", detail="flag is False")
            print("외부 workspace 종료 명령은 자동 실행하지 않습니다. 결과 ZIP과 readback SHA-256을 확인한 뒤 직접 종료하세요.")
            """
        ),
        nb.v4.new_markdown_cell(
            "## 판정 경계\n\n"
            "기본 실행의 완료는 사전 점검뿐이다. CPU 통합 모드가 통과해도 `algorithm_qa_only`이며 실제 성능, H100 실행, L4 제한, "
            "DACON 제출 ZIP을 증명하지 않는다. 실제 라벨 gate, accepted weights, sealed 1회 검증, L4 package 검증을 모두 통과하기 전에는 최종 후보로 표기하지 않는다."
        ),
    ]
    notebook = nb.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK.name, "gpuType": "H100", "provenance": []},
        },
    )
    nb.validate(notebook)
    return notebook


def main() -> None:
    """Build and validate the notebook."""
    notebook = build()
    nb.write(notebook, NOTEBOOK)
    nb.validate(nb.read(NOTEBOOK, as_version=4))
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
