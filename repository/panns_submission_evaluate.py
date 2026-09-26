# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
"""Evaluate R0/R1/R3 on every selection row using verified frozen fake scores."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from decision_heads import file_score
from graph_scoring import official_metrics, validate_predictions
from panns_adaptation import presence_logits
from panns_adaptation_cache import readout_assets
from panns_adaptation_data import identity, load_cache, read_cache_index, sha256, validate_config, write_json
from panns_adaptation_run import model_for_arm
from panns_cached_fake import before_deadline
from panns_fixed_predictions import configure_numerics
from panns_submission import Selection


def choose_model(reports) -> str:
    baseline = reports['R0']
    if baseline['ads'] is None or baseline['cps'] is None:
        raise ValueError('Selection baseline has incomplete metric classes')
    eligible = [arm for arm, report in reports.items() if report['ads'] is not None
                and report['cps'] is not None and report['ads'] >= baseline['ads']
                and report['cps'] >= baseline['cps']]
    return max(eligible, key=lambda arm: (reports[arm]['ads'], reports[arm]['cps'], arm == 'R0'))


def summarize(labels, predictions):
    report = official_metrics(labels, predictions)
    metrics = report['metrics']
    detection = ('file_eer', 'voice_eer', 'music_eer')
    presence = ('voice_presence_auc', 'music_presence_auc')
    ads = (sum(weight*(1-metrics[name]) for weight, name in zip((.5, .2, .3), detection, strict=True))
           if all(metrics[name] is not None for name in detection) else None)
    cps = (sum(metrics[name] for name in presence)/2
           if all(metrics[name] is not None for name in presence) else None)
    return dict(ads=ads, cps=cps, report=report, scope='known-label selection diagnostic; not leaderboard')


def evaluate(options: argparse.Namespace) -> None:
    configure_numerics(options.device)
    config = json.loads(options.run.read_text())
    package = validate_config(config)
    rows = {row['ID']: row for row in map(json.loads, Path(config['rows']).read_text().splitlines())}
    expected = [rid for rid in config['panel_ids'] if rows[rid]['role'] == 'selection']
    fixed = json.loads(options.fixed.read_text())
    if (fixed['role'] != 'selection' or fixed['device'] != options.device
            or fixed['config_identity'] != config['identity']
            or fixed['rows_sha256'] != sha256(config['rows'])
            or fixed['reference_sha256'] != sha256(config['reference'])
            or fixed['producer_sha256'] != sha256(Path(__file__).with_name('panns_cached_fake.py'))
            or fixed['entries_sha256'] != identity(fixed['entries'])
            or fixed['legacy_used'] and not fixed['gate']['passed']):
        raise ValueError('Frozen fake prediction provenance differs')
    validate_predictions([{k: v for k, v in e.items() if k != 'audio_sha256'}
                          for e in fixed['entries']], expected)
    fake = {entry['ID']: entry for entry in fixed['entries']}
    if any(fake[rid]['audio_sha256'] != rows[rid]['verified_audio']['sha256'] for rid in expected):
        raise ValueError('Frozen predictions bind different audio')
    entries = [entry for entry in read_cache_index(config)['entries'] if entry['role'] == 'selection']
    if [entry['ID'] for entry in entries] != expected:
        raise ValueError('Selection activation cache coverage differs')
    exports = {}
    for item in options.models:
        arm, separator, directory = item.partition('=')
        if not separator or arm not in ('R1', 'R3') or arm in exports:
            raise ValueError('Models must be unique R1=EXPORTDIR and R3=EXPORTDIR')
        exports[arm] = Path(directory).resolve()
    if set(exports) != {'R1', 'R3'}:
        raise ValueError('Both declared structural candidates are required')
    heads, labels, groups = readout_assets(package)
    ensemble = json.loads((package/'model/trained/file_ensemble.json').read_text())
    reports, provenance = {}, {}
    for arm in ('R0', 'R1', 'R3'):
        before_deadline(options.deadline_epoch)
        model = model_for_arm(package, arm, 41, options.device)
        expected_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        model.eval().requires_grad_(False)
        if arm in exports:
            export = exports[arm]
            selection = Selection.read(export/'selection.json')
            if (selection.arm != arm or selection.config_identity != config['identity']
                    or selection.reference_sha256 != sha256(config['reference'])
                    or selection.weights_sha256 != sha256(export/'best_model.pt')):
                raise ValueError('Export identity differs from the original training contract')
            state = torch.load(export/'best_model.pt', map_location='cpu', weights_only=True)
            if set(state) != expected_names:
                raise ValueError('Export trainable tensor names differ')
            parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in state.items():
                    parameter = parameters[name]
                    if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                        raise ValueError('Invalid exported tensor: '+name)
                    parameter.copy_(value)
            provenance[arm] = dict(export=str(export), selection_sha256=sha256(export/'selection.json'),
                                   weights_sha256=selection.weights_sha256)
            del state, parameters, parameter
        predictions = []
        with torch.inference_mode():
            for entry in entries:
                before_deadline(options.deadline_epoch)
                features = torch.from_numpy(load_cache(entry)['block5'])
                tags = torch.cat([model.tags_from_block5(part.to(options.device)).cpu()
                                  for part in features.split(config['microbatch'])])
                vp, mp = presence_logits(tags, entry['stems'], heads, labels, groups).sigmoid().tolist()
                prior = fake[entry['ID']]
                vf, mf = prior['VOICE_FAKE_PROB'], prior['MUSIC_FAKE_PROB']
                predictions.append(dict(ID=entry['ID'], VOICE_FAKE_PROB=vf, MUSIC_FAKE_PROB=mf,
                    VOICE_PRESENT_PROB=vp, MUSIC_PRESENT_PROB=mp,
                    FILE_FAKE_PROB=file_score(ensemble, dict(Pv=vp, Pm=mp, Fv=vf, Fm=mf))))
        validate_predictions(predictions, expected)
        reports[arm] = summarize([rows[rid] for rid in expected], predictions)
        write_json(options.output/f'evaluation_{arm}_41.json', dict(arm=arm, seed=41,
            config_identity=config['identity'], fixed_predictions_sha256=sha256(options.fixed),
            rows_sha256=sha256(config['rows']), predictions=predictions, selection=reports[arm],
            export=provenance.get(arm), independent_evaluation=False))
        print(json.dumps(dict(arm=arm, rows=len(predictions), ads=reports[arm]['ads'], cps=reports[arm]['cps'])), flush=True)
        del model
        gc.collect()
        if options.device == 'cuda':
            torch.cuda.empty_cache()
    chosen = choose_model(reports)
    trial = max(exports, key=lambda arm: (reports[arm]['ads'], reports[arm]['cps']))
    write_json(options.output/'chosen.json', dict(selected_arm=chosen,
        selected_export=str(exports[chosen]) if chosen in exports else None,
        preferred_trial_arm=trial, preferred_trial_export=str(exports[trial]),
        reports=reports, exports=provenance, config_identity=config['identity'],
        selection_rows=len(expected), ranking='selection ADS maximum; CPS >= R0; ADS >= R0; exact tie retains R0',
        development_used_for_selection=False, independent_evaluation=False,
        official_submission=False, quality_claim='known-label diagnostic only'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--fixed', type=Path, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    options = parser.parse_args()
    if not time.time() < options.deadline_epoch <= time.time()+12*3600:
        parser.error('Deadline must be in the next 12 hours')
    evaluate(options)


if __name__ == '__main__':
    main()
