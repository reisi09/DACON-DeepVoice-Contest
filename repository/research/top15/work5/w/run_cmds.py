"""Run 'NAME|ENV ASSIGNMENTS|COMMAND' lines on VESSL GPUs, PAR at a time, round-robin over NGPU; per-job log + exit code in
STATUS (default /root/w_runs/status.json); exit 1 if any job failed.  python run_cmds.py JOBS"""
import json, os, shlex, subprocess, sys, time
from pathlib import Path

jobs = [l.split('|', 2) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip() and not l.startswith('#')]
PAR, NGPU = int(os.environ.get('PAR', 12)), int(os.environ.get('NGPU', 4))
status_path = Path(os.environ.get('STATUS', '/root/w_runs/status.json')); status_path.parent.mkdir(parents=True, exist_ok=True)
logs = status_path.parent / 'logs'; logs.mkdir(exist_ok=True)
status = json.loads(status_path.read_text()) if status_path.exists() else {}
todo = [j for j in jobs if status.get(j[0], {}).get('exit') != 0]
running, k = [], 0
while todo or running:
    while todo and len(running) < PAR:
        name, env, cmd = todo.pop(0)
        e = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(k % NGPU), **dict(x.split('=', 1) for x in shlex.split(env))}
        p = subprocess.Popen(cmd, shell=True, env=e, stdout=open(logs / f'{name}.log', 'w'), stderr=subprocess.STDOUT)
        running.append((p, name, time.time())); k += 1
    time.sleep(5)
    for item in [r for r in running if r[0].poll() is not None]:
        running.remove(item); p, name, t0 = item
        status[name] = {'exit': p.returncode, 'minutes': round((time.time() - t0) / 60, 1)}
        status_path.write_text(json.dumps(status, indent=1)); print(name, status[name], flush=True)
failed = {n: s for n, s in status.items() if s['exit'] != 0}
print('failed', failed, flush=True)
sys.exit(1 if failed else 0)
