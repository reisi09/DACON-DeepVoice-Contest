# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///
# Run remotely: .venv/bin/python benchmark_fullft8_batching.py
"""Measure real pretrained training throughput before deploying larger batches."""
import gc
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import batch_worker as worker

ROOT = Path('/root/fullft8/h100x1_fullft8_run')


def main() -> None:
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    config = json.loads((ROOT/'runtime.json').read_text())
    if os.environ.get('BENCH_WAIT_PID'):
        status = Path('/proc')/os.environ['BENCH_WAIT_PID']/'stat'
        while status.exists() and status.read_text().split(') ')[1].split()[0] != 'Z':
            worker.check_deadline(config)
            time.sleep(2)
    rows = [r for r in config['splits']['S1']['train']
            if json.loads((ROOT/'audio_cache'/f'{r["ID"]}.json').read_text())['usable_train']][:72]
    assert len(rows) == 72
    folder = ROOT/'performance'
    folder.mkdir(exist_ok=True)
    results = []
    for kind in ['mert', 'eat']:
        for batch, accumulation, workers in [(1, 8, 0), (4, 2, 4)]:
            torch.manual_seed(config['seed'])
            current = dict(config, batch_size=batch, accumulation=accumulation, loader_workers=workers)
            model = worker.MusicClassifier(kind, False, Path(config['model_root'])).cuda().train()
            optimizer = torch.optim.AdamW(worker.optimization_groups(model, current))
            data = worker.AudioBags(rows, current, True)
            loader = worker.batch_loader(data, current)
            begun = time.perf_counter()
            sample_file = folder/f'{kind}_batch{batch}_gpu.csv'
            with sample_file.open('w') as samples:
                monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits', '--loop-ms=200'], stdout=samples)
                try:
                    measured_at, warm_seconds, updates = 0., 0., 0
                    for index, (bags, labels) in enumerate(loader):
                        worker.check_deadline(current)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            loss = torch.nn.functional.cross_entropy(
                                model(bags.cuda(non_blocking=True)).float(), labels.cuda(non_blocking=True))
                        if not torch.isfinite(loss):
                            raise RuntimeError('Benchmark nonfinite loss')
                        (loss * worker.batch_loss_weight(index, len(bags), len(data), current)).backward()
                        if (index+1) % accumulation == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            updates += 1
                            if updates == 1:
                                torch.cuda.synchronize()
                                measured_at = time.perf_counter()
                                warm_seconds = measured_at - begun
                                torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - measured_at
                    assert updates == 9
                finally:
                    monitor.terminate()
                    monitor.wait()
            record = dict(encoder=kind, batch_size=batch, accumulation=accumulation,
                          loader_workers=workers, measured_songs=64, seconds=seconds,
                          songs_per_second=64/seconds, warmup_and_spawn_seconds=warm_seconds,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            print(json.dumps(record), flush=True)
            results.append(record)
            (folder/'batch_benchmark.json').write_text(json.dumps(results, indent=2))
            del model, optimizer, loss, bags, labels, loader, data
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
