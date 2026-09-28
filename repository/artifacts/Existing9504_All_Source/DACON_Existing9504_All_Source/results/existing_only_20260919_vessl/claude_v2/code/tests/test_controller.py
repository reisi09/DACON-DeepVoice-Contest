"""Two-controller / orphan-reclaim safety of the notebook controller (standard library only, no GPU).

    python tests/test_controller.py --work <scratch dir>

What it protects against:
  * a second live controller (another kernel) touching the job: it must refuse
    before writing anything and must not kill the first controller or its stage;
  * a StageLock holder being killed by a second stage process (it must refuse);
  * an unrecorded stage process for the job being killed (must refuse clearly);
  * an orphan left by a dead controller blocking or corrupting the next run
    (the next controller must terminate exactly that recorded process);
  * re-running the bootstrap cell in the same kernel: ownership moves to the new
    object, the old object refuses, and other processes stay locked out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONTROLLER_A = r"""
import json, sys, time
sys.path.insert(0, sys.argv[2])
from dacon9504a100.orchestrator import Pipeline
pipe = Pipeline(config=json.load(open(sys.argv[1])), report_every=1.0)
command = [sys.executable, "-c", "import time; time.sleep(600)", "-m", "dacon9504a100.features", "full",
           "--config", str(pipe.config_path)]
child = pipe._spawn("features", command, pipe.job / "logs" / "features.log")
print(json.dumps({"child": child.pid}), flush=True)
time.sleep(600)
"""

STAGE_HOLDER = r"""
import sys, time
sys.path.insert(0, sys.argv[2])
from dacon9504a100.common import StageLock
lock = StageLock(sys.argv[1], "dacon9504a100.features")
lock.acquire()
print("held", flush=True)
time.sleep(600)
"""

SECOND_PROCESS = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
from dacon9504a100.orchestrator import Pipeline
Pipeline(config=json.load(open(sys.argv[1])))
"""


def snapshot(*roots: Path) -> dict[str, tuple[int, int, str]]:
    result = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                result[str(path)] = (path.stat().st_size, path.stat().st_mtime_ns,
                                     hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def alive(pid: int) -> bool:
    from dacon9504a100.common import process_start_ticks
    return process_start_ticks(pid) is not None


def expect_error(function, needle: str, *args, **kwargs) -> str:
    try:
        function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - message asserted
        assert needle in str(error), f"expected {needle!r} in {error}"
        return str(error)
    raise AssertionError(f"expected an error containing {needle!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.work.exists():
        shutil.rmtree(args.work)
    deploy, job = args.work / "deploy", args.work / "job"
    deploy.mkdir(parents=True)
    from dacon9504a100.common import StageLock
    from dacon9504a100.orchestrator import Pipeline, build_config
    config = build_config(deploy, job, ROOT, None, None)
    config_file = args.work / "config.json"
    config_file.write_text(json.dumps(config))
    (job / "env").mkdir(parents=True)
    (job / "env" / "receipt.json").write_text("{}")
    processes: list[subprocess.Popen] = []
    try:
        print("controller A starts and launches a (recorded) long stage process")
        a = subprocess.Popen([sys.executable, "-c", CONTROLLER_A, str(config_file), str(ROOT)],
                             stdout=subprocess.PIPE, text=True)
        processes.append(a)
        child = json.loads(a.stdout.readline())["child"]
        time.sleep(0.5)

        print("second live controller must refuse without writing or killing anything")
        before = snapshot(job, deploy)
        expect_error(Pipeline, "Another live controller", config=config)
        assert snapshot(job, deploy) == before, "refused controller modified files"
        assert a.poll() is None and alive(child), "refused controller killed the live controller or its stage"

        print("a live StageLock holder is never killed by a second acquirer")
        holder = subprocess.Popen([sys.executable, "-c", STAGE_HOLDER, str(job / "locks" / "probe.lock"), str(ROOT)],
                                  stdout=subprocess.PIPE, text=True)
        processes.append(holder)
        assert holder.stdout.readline().strip() == "held"
        expect_error(StageLock(job / "locks" / "probe.lock", "dacon9504a100.features").acquire,
                     "refusing to kill it", wait_seconds=1)
        assert holder.poll() is None, "StageLock killed its live holder"
        holder.kill()
        holder.wait()

        print("controller A dies; its stage process survives as an orphan")
        a.kill()
        a.wait()
        time.sleep(0.5)
        assert alive(child)

        print("an unrecorded stage process for this job -> refusal, nothing killed, lock not kept")
        unknown = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "-m",
                                    "dacon9504a100.parity", "--config", str(job / "config" / "run_config.json")])
        processes.append(unknown)
        time.sleep(0.5)
        expect_error(Pipeline, "Unrecorded dacon9504a100 stage processes", config=config)
        assert unknown.poll() is None and alive(child), "refusal killed a process"
        unknown.kill()
        unknown.wait()

        print("next controller reclaims exactly the recorded orphan of the dead controller")
        b = Pipeline(config=config, report_every=1.0)
        assert not alive(child), "orphan was not reclaimed"
        reclaimed = json.loads((job / "status" / "reclaimed_children.json").read_text())
        assert [r["pid"] for r in reclaimed] == [child] and reclaimed[0]["controller"]["pid"] == a.pid
        assert json.loads((job / "status" / "children.json").read_text())["children"] == []

        print("bootstrap cell re-run in the same kernel: ownership moves, old object refuses")
        c = Pipeline(config=config, report_every=1.0)
        expect_error(b.run, "superseded", "preflight")
        assert c._controller.owns()
        other = subprocess.run([sys.executable, "-c", SECOND_PROCESS, str(config_file), str(ROOT)],
                               capture_output=True, text=True)
        assert other.returncode != 0 and "Another live controller" in other.stderr, other.stderr[-2000:]
        c._controller.release()
        again = subprocess.run([sys.executable, "-c", SECOND_PROCESS, str(config_file), str(ROOT)],
                               capture_output=True, text=True)
        assert again.returncode == 0, "a released controller lock must be acquirable: " + again.stderr[-2000:]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        try:
            os.kill(child, 9)
        except (ProcessLookupError, UnboundLocalError, NameError):
            pass
    print("CONTROLLER CHECKS PASSED")


if __name__ == "__main__":
    main()
