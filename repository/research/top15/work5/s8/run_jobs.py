"""Run run.py head jobs (one 'GROUP VARIANT' per line of JOBS) on VESSL GPUs, PAR at a time, round-robin over NGPU GPUs,
fp32 heads (HEAD_DEVICE=cuda:0 under CUDA_VISIBLE_DEVICES), every-row RT predictions (EVERY=1). Records each job's exit
code and log in /root/s8_runs/status.json; exit 1 if any job failed.  python run_jobs.py JOBS"""
import json, os, subprocess, sys, time
from pathlib import Path

jobs = [l.split() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip() and not l.startswith('#')]
PAR, NGPU = int(os.environ.get('PAR', 16)), int(os.environ.get('NGPU', 4))
OUT, LOGS = Path('/root/s8_runs'), Path('/root/s8_runs/logs')
LOGS.mkdir(parents=True, exist_ok=True)
base = {**os.environ, 'NEWCACHE': '/root/newbackbone_cache', 'S6CACHE': '/root/s6_cache', 'S8CACHE': '/root/s8_cache',
        'RUNS_OUT': str(OUT), 'HEAD_DEVICE': 'cuda:0', 'EVERY': '1', 'OMP_NUM_THREADS': '6', 'MKL_NUM_THREADS': '6'}
status_path = OUT / 'status.json'
status = json.loads(status_path.read_text()) if status_path.exists() else {}
todo = [j for j in jobs if status.get(' '.join(j), {}).get('exit') != 0]
running, k = [], 0
while todo or running:
    while todo and len(running) < PAR:
        g, v = todo.pop(0)
        log = open(LOGS / f'{g}__{v}.log', 'w')
        p = subprocess.Popen([sys.executable, 'run.py', g, v], env={**base, 'CUDA_VISIBLE_DEVICES': str(k % NGPU)},
                             stdout=log, stderr=subprocess.STDOUT, cwd='/root/proj/research/top15/work2/backbones')
        running.append((p, f'{g} {v}', time.time())); k += 1
    time.sleep(5)
    for item in [r for r in running if r[0].poll() is not None]:
        p, name, t0 = item
        running.remove(item)
        status[name] = {'exit': p.returncode, 'minutes': round((time.time() - t0) / 60, 1)}
        status_path.write_text(json.dumps(status, indent=1))
        print(name, status[name], flush=True)
failed = {n: s for n, s in status.items() if s['exit'] != 0}
print('failed', failed, flush=True)
sys.exit(1 if failed else 0)
