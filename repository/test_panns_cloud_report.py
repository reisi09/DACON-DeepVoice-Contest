"""Protect the parent-bootstrap calculation against weighted/tied ROC errors."""
import json
import numpy as np
import pytest
import panns_cloud_report as cloud_report
from graph_scoring import _roc_metrics, official_metrics
from panns_adaptation_data import identity, sha256, write_json
from panns_cloud_report import curve


def test_weighted_curve_matches_actual_parent_resampling():
    rng=np.random.default_rng(20260925)
    for _ in range(100):
        y=rng.integers(0,2,40)
        scores=rng.integers(0,6,40)/5
        parents=np.repeat(np.arange(10),4)
        counts=np.bincount(rng.integers(0,10,10),minlength=10)
        weights=counts[parents]
        selected=np.repeat(np.arange(len(y)),weights)
        expected=_roc_metrics(y[selected].tolist(),scores[selected].tolist())
        np.testing.assert_allclose(curve(y,scores,weights),expected,rtol=0,atol=1e-12)


def test_seed_mean_uses_the_same_parent_draw_before_computing_intervals():
    # Given: three models on the same parent panel, with correlated bootstrap draws.
    rng = np.random.default_rng(20260925)
    labels = np.tile([[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1]], (6, 1))
    baseline = rng.random(labels.shape)
    candidates = [rng.random(labels.shape) for _ in range(3)]
    parents = np.repeat(np.arange(6), 4)
    samples = [np.repeat(np.arange(len(labels)), np.bincount(
        rng.integers(6, size=6), minlength=6)[parents]) for _ in range(80)]

    def metrics(probabilities, selected):
        return np.array([_roc_metrics(labels[selected, k].tolist(), probabilities[selected, k].tolist())[
            0 if k == 0 else 1] for k in range(3)])

    points = [metrics(p, np.arange(len(labels))) - metrics(baseline, np.arange(len(labels)))
              for p in candidates]
    draws = [np.array([metrics(p, sample) - metrics(baseline, sample) for sample in samples])
             for p in candidates]
    # An undefined seed/draw must invalidate that shared draw, not drop just one seed.
    draws[1][7, 1] = np.nan

    # When: the report summarizes the three seeds.
    actual = cloud_report.delta_summary(list(zip(points, draws)))

    # Then: each draw equals explicit seed averaging, not pooled seeds or mean endpoints.
    point_mean = sum(points) / 3
    draw_mean = (draws[0] + draws[1] + draws[2]) / 3
    valid = np.isfinite(draw_mean).all(axis=1)
    for name, values, point in (
        ('G', draw_mean @ [-.45, .05, .05], point_mean @ [-.45, .05, .05]),
        ('voice_auc', draw_mean[:, 1], point_mean[1]),
        ('music_auc', draw_mean[:, 2], point_mean[2]),
    ):
        key = name if name == 'G' else name + '_delta'
        np.testing.assert_allclose(actual[key], point, rtol=0, atol=1e-15)
        np.testing.assert_allclose(actual[name + '_interval95'], np.quantile(values[valid], [.025, .975]),
                                   rtol=0, atol=1e-15)
    assert actual['invalid_bootstrap_fraction'] == 1 - valid.mean()


def test_development_ranking_enforces_gates_and_unrounded_tie_breaks():
    # Given: gains with rounding collisions, exact ties, and failed/undefined gates.
    candidates = [dict(arm=arm, G=gain, voice_auc_delta=voice, music_auc_delta=music,
                       new_parameters=count) for arm, gain, voice, music, count in (
        ('high', .10000000001, 0., 0., 200), ('small', .1, 0., 0., 2),
        ('large', .1, 0., 0., 100), ('zero', 0., 1., 1., 0),
        ('voice_regression', .9, -.001, 1., 0), ('music_regression', .9, 1., -.001, 0),
        ('undefined', None, 1., 1., 0), ('nonfinite', float('nan'), 1., 1., 0),
    )]

    # When: input order changes, selection still follows exact gain then module size.
    actual = cloud_report.rank_candidates(list(reversed(candidates)))

    # Then: no rounding lets the smaller but lower-gain model win; failed gates stay out.
    assert actual == ['high', 'small', 'large']


def synthetic_run(tmp_path, rows):
    from panns_cloud_run import training_identity
    from panns_training_budget import budget_steps, profile_binding
    rng = np.random.default_rng(41)
    rows_path = tmp_path / 'rows.jsonl'
    rows_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    reference_path = tmp_path / 'reference.json'
    write_json(reference_path, dict(package=str(tmp_path / 'package'), members={}))
    fixed_path = tmp_path / 'fixed.json'
    write_json(fixed_path, dict(entries=[dict(ID=row['ID'], VOICE_FAKE_PROB=.2, MUSIC_FAKE_PROB=.7) for row in rows]))
    write_json(tmp_path / 'cache/index.json', dict(synthetic_arithmetic_only=True))
    arms = ['R1', 'R2', 'R3', 'R5']
    seeds = [41, 42, 43]
    config = dict(output=str(tmp_path), rows=str(rows_path), arms=arms, seeds=seeds,
                  panel_ids=[row['ID'] for row in rows], reference=str(reference_path), fixed_predictions=str(fixed_path),
                  inputs={str(rows_path): sha256(rows_path), str(reference_path): sha256(reference_path)})
    config['identity'] = identity(config)
    binding = profile_binding(config, 'cpu')
    seconds = dict(R1=1., R2=2., R3=3., R5=4.)
    profile_path = tmp_path / 'throughput_profile.json'
    write_json(profile_path, dict(binding=binding, train_files=5, seconds_per_step=seconds, synthetic_arithmetic_only=True))
    steps = budget_steps(seconds, 5, 1.)
    budget = dict(binding=binding, profile_sha256=sha256(profile_path), training_hours=1., training_steps=steps, steps_per_pass=2)
    budget['identity'] = identity(budget)
    write_json(tmp_path / 'training_budget.json', budget)
    evaluations = {}
    for arm, seed in [('R0', 41), *[(arm, seed) for arm in arms for seed in seeds]]:
        predictions = [dict(ID=row['ID'], FILE_FAKE_PROB=float(rng.random()),
                            VOICE_FAKE_PROB=.2, MUSIC_FAKE_PROB=.7,
                            VOICE_PRESENT_PROB=float(rng.random()), MUSIC_PRESENT_PROB=float(rng.random()))
                       for row in rows]
        evaluations[arm, seed] = predictions
        run_identity = checkpoint_sha256 = None
        if arm != 'R0':
            folder = tmp_path / f'train_{arm}_{seed}'
            folder.mkdir()
            checkpoint = folder / 'checkpoint.pt'
            checkpoint.write_bytes(b'Synthetic hash input; never deserialized as a model.')
            run_identity, checkpoint_sha256 = training_identity(config, arm, seed, 'cpu'), sha256(checkpoint)
            write_json(folder / 'complete.json', dict(steps=steps, run_identity=run_identity, checkpoint_sha256=checkpoint_sha256))
        write_json(tmp_path / f'evaluation_{arm}_{seed}.json', dict(arm=arm, seed=seed, predictions=predictions,
            device='cpu', config_identity=config['identity'], rows_sha256=sha256(rows_path),
            fixed_predictions_sha256=sha256(fixed_path), run_identity=run_identity, checkpoint_sha256=checkpoint_sha256))
    write_json(tmp_path / 'run.json', config)
    return config, evaluations


@pytest.mark.parametrize('panel_case', ['complete', 'masked', 'one_fake_class'])
def test_report_uses_official_masks_and_keeps_diagnostics_without_a_g_panel(tmp_path, panel_case):
    rows = [dict(ID=str(i), role='development', source='synthetic', component_id=str(i // 4),
                 FILE_FAKE=i % 2, VOICE_PRESENT=(i % 4 in (1, 2)), MUSIC_PRESENT=(i % 4 in (0, 3)),
                 VOICE_FAKE=i % 2, MUSIC_FAKE=i % 2) for i in range(24)]
    rows[0].update(VOICE_FAKE=None, VOICE_FAKE_mask=0)
    rows[1].update(VOICE_FAKE_mask=0)
    rows[2].update(VOICE_PRESENT_mask=0)
    rows[3].update(FILE_FAKE_mask=0)
    for row in rows:
        if panel_case == 'masked':
            row['FILE_FAKE_mask'] = 0
        if panel_case == 'one_fake_class':
            row['VOICE_FAKE'] = 0
    config, evaluations = synthetic_run(tmp_path, rows)

    result = cloud_report.report(config, repeats=32)

    role = result['roles']['development']
    panel = [] if panel_case == 'masked' else [i for i in range(24) if i not in (1, 2, 3)]
    assert role['complete_G_rows'] == len(panel)
    assert role['excluded_from_G_by_source'] == {'synthetic': len(rows) - len(panel)}
    assert result['deployment_seed'] == 41
    assert result['selected_candidate'] is None
    assert 'candidate_selection' not in role
    assert len(role['structural_contrasts']) == 6
    assert {item['contrast'] for item in role['structural_seed_means']} == {'R3-R1', 'R5-R1'}
    for comparison in role['comparisons']:
        assert comparison['presence']['VOICE_PRESENT']['known'] == 23
        assert comparison['presence']['MUSIC_PRESENT']['known'] == 24
        assert comparison['presence']['VOICE_PRESENT']['auc'] is not None
        assert comparison['strata']['source|synthetic']['candidate']['counts']['voice_presence_auc'] == 23
    if panel_case == 'complete':
        reference = official_metrics([rows[i] for i in panel], [evaluations['R0', 41][i] for i in panel])
        for mean in role['seed_means']:
            expected = sum(official_metrics([rows[i] for i in panel], [evaluations[mean['arm'], seed][i]
                           for i in panel])['score'] - reference['score'] for seed in config['seeds']) / 3
            np.testing.assert_allclose(mean['G'], expected, rtol=0, atol=1e-15)
    else:
        assert all(mean['G'] is None for mean in role['seed_means'])
    assert result['independent_evaluation'] is False
    assert result['adoption_authorized'] is False


@pytest.mark.parametrize('corruption', ['evaluation', 'rows', 'checkpoint', 'fake_head'])
def test_report_rejects_stale_or_changed_run_inputs(tmp_path, corruption):
    rows = [dict(ID=str(i), role='development', source='synthetic', component_id=str(i),
                 FILE_FAKE=i % 2, VOICE_PRESENT=i % 2, MUSIC_PRESENT=i % 2,
                 VOICE_FAKE=i % 2, MUSIC_FAKE=i % 2) for i in range(4)]
    config, _ = synthetic_run(tmp_path, rows)
    path = tmp_path / 'evaluation_R3_41.json'
    evaluation = json.loads(path.read_text())
    if corruption == 'evaluation':
        evaluation['config_identity'] = 'stale'
        write_json(path, evaluation)
    elif corruption == 'fake_head':
        evaluation['predictions'][0]['VOICE_FAKE_PROB'] = .9
        write_json(path, evaluation)
    elif corruption == 'rows':
        (tmp_path / 'rows.jsonl').write_text(json.dumps(rows[0]) + '\n')
    else:
        (tmp_path / 'train_R3_41/checkpoint.pt').write_bytes(b'Changed checkpoint')
    with pytest.raises(ValueError):
        cloud_report.report(config, repeats=2)


def test_only_selection_role_selects_candidates_when_development_prefers_another_arm(tmp_path):
    rows = [dict(ID=str(i), role='development' if i < 24 else 'selection', source='synthetic',
                 component_id=str(i // 4), FILE_FAKE=i % 2,
                 VOICE_PRESENT=int(i % 4 in (1, 2)), MUSIC_PRESENT=int(i % 4 in (0, 3)),
                 VOICE_FAKE=i % 2, MUSIC_FAKE=i % 2) for i in range(48)]
    config, evaluations = synthetic_run(tmp_path, rows)
    for (arm, seed), predictions in evaluations.items():
        path = tmp_path / f'evaluation_{arm}_{seed}.json'
        evaluation = json.loads(path.read_text())
        for row, prediction in zip(rows, predictions, strict=True):
            improved = (row['role'] == 'development' and arm == 'R1') or (row['role'] == 'selection' and arm == 'R5')
            for target in ('FILE_FAKE', 'VOICE_PRESENT', 'MUSIC_PRESENT'):
                prediction[target + '_PROB'] = .9 if row[target] == improved else .1
        evaluation['predictions'] = predictions
        write_json(path, evaluation)

    result = cloud_report.report(config, repeats=8)

    assert result['selection_role'] == 'selection'
    assert result['selected_candidate'] == 'R5'
    assert 'candidate_selection' not in result['roles']['development']
    selection = result['roles']['selection']
    assert selection['candidate_selection']['ranking'] == ['R5']
    assert selection['candidate_selection']['selected_arm'] == 'R5'
    assert selection['candidate_selection']['deployment_seed'] == 41
    assert selection['candidate_selection']['r6_combination'] == []
