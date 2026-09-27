"""CPU smoke of finetune_ad.py / score_ft.py on real HTDemucs stems (built by data_v4/dump_stems.py --device cpu).

  python smoke_test.py --stems STEMS --model AD_FP32_DIR --code-root PROJECT_ROOT --work SCRATCH --report smoke/report.json
Proves: augmentation filters do what they claim; the trainable set at the default freeze is layers 32..47 + encoder LN +
classifier; a tiny overfit lowers the loss through the real torchrun/DDP(gloo) entry point; only intended tensors
change; the fp16 native-layout checkpoint reloads in antideepfake_runtime.AntiDeepfake and reproduces predictions;
score_ft.py equals AntiDeepfake.predict. Also counts FLOPs (meta device) for the H100 estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import finetune_ad as ft  # noqa: E402

# (stem id, route, VOICE_PRESENT, VOICE_FAKE) -- labels are synthetic: the smoke checks mechanics, not accuracy
ROWS = [('asvspoof2019_la_LA_E_5923854__3148a95d8c', 'voice', '1', '1'),   # 3.1 s: repeat-pad crop, unpadded eval
        ('asvspoof2019_la_LA_T_5517266__5a374f6261', 'voice', '1', '0'),
        ('ctrsvdd_CtrSVDD_0059_D_0002454__146540f32b', 'voice', '1', '1'),
        ('librispeech_test-clean_1580-141083-0008__a832b3a4a5', 'voice', '1', '0'),
        ('fake_16900_udio_1__81062a7b5a', 'music', '1', '1'),                 # mixed row: voice head via head_labels
        ('haim_musicgen_musicgen_small_00029__f7f6acf7a5', 'music', '1', '0'),
        ('fakemusiccaps_mustango_b1j-hD9zs6Q__59053ec13e', 'music', '0', ''),  # no voice label -> excluded
        ('core_63ab76f979000832863b_base__2b1a70e367', 'music', '', '')]      # excluded


def run(cmd: list[str], log: Path) -> dict:
    started = time.time()
    with log.open('w') as stream:
        code = subprocess.call(cmd, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '4'})
    if code:
        raise SystemExit(f'{cmd[-1]} failed ({code}); see {log}\n' + log.read_text()[-3000:])
    return {'seconds': round(time.time() - started, 1)}


def check_augment() -> dict:
    t = torch.arange(ft.WINDOW) / ft.SR
    tone = lambda f: torch.sin(2 * math.pi * f * t)[None].repeat(2, 1)  # noqa: E731
    gain = lambda y, x: float(10 * torch.log10(y[:, 2000:-2000].pow(2).mean() / x[:, 2000:-2000].pow(2).mean()))  # noqa: E731
    lp = lambda x: ft.fir(x, ft.fir_lowpass(torch.tensor([3400., 3400.]), ft.SR))  # noqa: E731
    out = {'lowpass3400_1k_db': gain(lp(tone(1000)), tone(1000)), 'lowpass3400_6k_db': gain(lp(tone(6000)), tone(6000)),
           'lowpass8000_identity_err': float((ft.fir(tone(6000), ft.fir_lowpass(torch.tensor([8000., 8000.]), ft.SR)) - tone(6000)).abs().max()),
           'tel_1k_db': gain(ft.telephone(tone(1000)), tone(1000)), 'tel_100_db': gain(ft.telephone(tone(100)), tone(100)),
           'tel_5k_db': gain(ft.telephone(tone(5000)), tone(5000))}
    x = tone(440)
    for pink in (False, True):
        noisy = ft.add_noise(x.clone(), torch.tensor([20., 20.]), torch.tensor([pink, pink]))
        out[f'snr20_{"pink" if pink else "white"}_db'] = float(10 * torch.log10(x.pow(2).mean() / (noisy - x).pow(2).mean()))
    batch = torch.randn(64, ft.WINDOW) * .1
    y = ft.augment(batch.clone())
    out['augment_shape_ok'] = list(y.shape) == [64, ft.WINDOW] and bool(torch.isfinite(y).all())
    assert out['lowpass3400_1k_db'] > -.5 and out['lowpass3400_6k_db'] < -30 and out['lowpass8000_identity_err'] < 1e-5, out
    assert out['tel_1k_db'] > -1 and out['tel_100_db'] < -15 and out['tel_5k_db'] < -40, out
    assert all(abs(out[k] - 20) < .5 for k in ('snr20_white_db', 'snr20_pink_db')) and out['augment_shape_ok'], out
    return out


def check_freeze_and_flops(model_dir: Path) -> dict:
    """Default freeze on a meta-device replica (no weights): names and counts, and FLOPs per 4.04 s window."""
    from torch.utils.flop_counter import FlopCounterMode
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    config = Wav2Vec2Config.from_json_file(str(model_dir / 'antideepfake_config.json'))
    config.apply_spec_augment, config._attn_implementation = False, 'eager'
    with torch.device('meta'):
        model = torch.nn.Module()
        model.encoder, model.classifier = Wav2Vec2Model(config), torch.nn.Linear(config.hidden_size, 2)
        x = torch.randn(1, ft.WINDOW)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model.classifier(model.encoder(torch.nn.functional.layer_norm(x, x.shape[-1:])).last_hidden_state.mean(1))
    forward = counter.get_total_flops()
    names = [n for n, _ in model.named_parameters() if ft.trainable(n, 32)]
    layers = sorted({int(m[1]) for n in names for m in [re.match(r'encoder\.encoder\.layers\.(\d+)\.', n)] if m})
    assert layers == list(range(32, 48)) and {n for n in names if 'layers.' not in n} == {
        'encoder.encoder.layer_norm.weight', 'encoder.encoder.layer_norm.bias', 'classifier.weight', 'classifier.bias'}, names
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
    with FlopCounterMode(display=False) as counter:
        model.classifier(model.encoder(torch.nn.functional.layer_norm(x, x.shape[-1:])).last_hidden_state.mean(1)).sum().backward()
    return {'trainable_tensors': len(names), 'trainable_params': sum(p.numel() for n, p in model.named_parameters() if n in names),
            'total_params': sum(p.numel() for p in model.parameters()), 'fwd_gflops_per_window': forward / 1e9,
            'train_gflops_per_window': counter.get_total_flops() / 1e9}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stems', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--code-root', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(4)
    report = {'augment': check_augment(), 'freeze_flops': check_freeze_and_flops(a.model)}
    print(json.dumps(report), flush=True)

    stems, run_dir = a.work / 'stems', a.work / 'run'
    stems.mkdir(parents=True, exist_ok=True)
    fields = ['ID', 'split', 'route', 'component_id', 'VOICE_PRESENT', 'VOICE_FAKE', 'VOICE_FAKE_mask', 'source', 'generator']
    rows = []
    for split in ('train', 'val', 'test'):
        for stem, route, present, fake in ROWS:
            rid = f'{split}_{stem}'
            rows.append(dict(ID=rid, split=split, route=route, component_id='c_' + rid, VOICE_PRESENT=present,
                             VOICE_FAKE=fake, VOICE_FAKE_mask='1' if fake else '0', source=stem.split('_')[0], generator=''))
            link = stems / f'{rid}.npz'
            if not link.exists():
                link.symlink_to(a.stems / f'{stem}.npz')
    # silent voice stems: a mixed train row (must be dropped) and a voice-route val row (kept, scored 0)
    silent = np.zeros(80000, np.float16)
    for rid, split, route in (('train_silent_mixed', 'train', 'music'), ('val_silent_voice', 'val', 'voice')):
        np.savez(stems / f'{rid}.npz', voice=silent, backing=silent, samples=np.array(80000), voice_silent=np.array(True))
        rows.append(dict(ID=rid, split=split, route=route, component_id='c_' + rid, VOICE_PRESENT='1', VOICE_FAKE='0',
                         VOICE_FAKE_mask='1', source='silent', generator=''))
    rows.append(dict(ID='val_no_stem', split='val', route='voice', component_id='c_ns', VOICE_PRESENT='1', VOICE_FAKE='1',
                     VOICE_FAKE_mask='1', source='x', generator=''))
    manifest = a.work / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # static rendezvous on 127.0.0.1: macOS --standalone hangs resolving the host name
    torchrun = lambda port: [sys.executable, '-m', 'torch.distributed.run', '--nnodes', '1', '--nproc_per_node', '1',  # noqa: E731
                             '--master_addr', '127.0.0.1', '--master_port', str(port)]
    report['train_run'] = run(torchrun(29571) + [
        str(HERE / 'finetune_ad.py'), '--manifest', str(manifest), '--stems', str(stems), '--model', str(a.model),
        '--code-root', str(a.code_root), '--out', str(run_dir), '--device', 'cpu', '--first-trainable-layer', '46',
        '--batch', '4', '--max-steps', str(a.steps), '--eval-every', str(a.steps // 2), '--log-every', '2',
        '--lr-backbone', '1e-4', '--lr-head', '1e-3', '--warmup', '.1', '--patience', '10', '--no-augment',
        '--workers', '2', '--ref-files', '3', '--eval-batch', '4'], a.work / 'train.log')   # CPU RAM cap: 4-window eval batches
    records = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines()]
    setup = next(r for r in records if r['kind'] == 'setup')
    losses = [r['loss'] for r in records if r['kind'] == 'train']
    result = json.loads((run_dir / 'result.json').read_text())
    report['train'] = {'setup': {k: setup[k] for k in ('train_items', 'train_fake', 'val_items', 'val_fake', 'missing_stems',
                                                    'trainable_params', 'trainable_tensors')},
                       'losses': losses, 'evals': [(e['step'], e['val_eer'], e['val_bce']) for e in records if e['kind'] == 'eval'],
                       'result': result}
    assert setup['train_items'] == 6 and setup['val_items'] == 7 and setup['missing_stems'] == [0, 1], setup
    assert np.mean(losses[-2:]) < .5 * np.mean(losses[:2]), losses
    assert not result['params']['changed_outside_trainable'] and not result['params']['trainable_unchanged'], result['params']
    assert result['roundtrip']['ok'] and result['roundtrip']['weights_exact'], result['roundtrip']

    out = a.work / 'ft_members.csv'
    report['score_run'] = run(torchrun(29572) + [
        str(HERE / 'score_ft.py'), '--ckpt', str(run_dir / 'best'), '--manifest', str(manifest), '--splits', 'val', 'test',
        '--stems', str(stems), '--code-root', str(a.code_root), '--out', str(out), '--device', 'cpu', '--batch', '4'], a.work / 'score.log')
    scored = {r['ID']: r for r in csv.DictReader(out.open())}
    assert len(scored) == 2 * len(ROWS) + 1 and scored['val_silent_voice']['voice_silent'] == '1', len(scored)
    best = {r['ID']: r for r in csv.DictReader((run_dir / 'val_best.csv').open())}
    sys.path.insert(0, str(a.code_root))
    from antideepfake_runtime import AntiDeepfake
    runtime = AntiDeepfake(run_dir / 'best', 'cpu')
    vs_predict, vs_prefp16 = 0., 0.
    for rid in [f'test_{ROWS[0][0]}', f'test_{ROWS[4][0]}', f'test_{ROWS[2][0]}']:   # 3.1 s, 32.8 s, 4.6 s
        expected = runtime.predict(ft.load_voice(stems / f'{rid}.npz').astype(np.float32), float('inf'))
        got = [float(scored[rid][k]) for k in ('ft_max', 'ft_top30', 'ft_mean')]
        vs_predict = max(vs_predict, *(abs(g - e) for g, e in zip(got, [expected[0], expected[1], float(np.mean(expected[2:]))])))
    drift = {rid: max(abs(float(r[k]) - float(scored[rid][k])) for k in ('ft_max', 'ft_top30', 'ft_mean')) for rid, r in best.items()}
    vs_prefp16 = max(d for rid, d in drift.items() if best[rid]['voice_silent'] == '0')
    report['score'] = {'rows': len(scored), 'max_abs_vs_runtime_predict': vs_predict,
                       'max_abs_fp16ckpt_vs_prefp16_val_best_nonsilent': vs_prefp16,
                       'all_zero_silent_stem_drift (runtime scores it 0)': drift['val_silent_voice'],
                       'best_step': json.loads((run_dir / 'best/meta.json').read_text())['step']}
    assert vs_predict < 1e-5 and vs_prefp16 <= 1e-3, report['score']

    f = report['freeze_flops']
    report['h100_estimate'] = {
        'train_tflop_per_step_per_gpu': 32 * f['train_gflops_per_window'] / 1e3,
        'steps_per_s_per_gpu_at_mfu': {m: round(989e12 * m / (32 * f['train_gflops_per_window'] * 1e9), 2) for m in (.25, .4)},
        'note': 'H100 SXM dense bf16 peak 989 TFLOP/s; eager attention + elementwise CNN/LN ops not in the FLOP count'}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print('SMOKE OK')


if __name__ == '__main__':
    main()
