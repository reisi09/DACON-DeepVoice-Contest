"""Protect the observed H100 endpoint format and cleanup after failed launches."""
from unittest.mock import patch
import localization_a100_ops as ops


def test_h100_uses_recorded_cluster_endpoint_without_accepting_other_hosts():
    with patch.object(ops, 'ENDPOINT_SCHEME', 'http', create=True), patch.object(ops, 'ENDPOINT_HOST', 'capella.cloud.vessl.ai', create=True):
        assert ops.endpoint_allowed('http://capella.cloud.vessl.ai:32053/?token=fixture')
        assert not ops.endpoint_allowed('http://capella.cloud.vessl.ai.attacker.test:32053/?token=fixture')
        assert not ops.endpoint_allowed('http://attacker.test:32053/?token=fixture')


def test_pause_waits_for_stopped_state_when_phase_already_standby(tmp_path):
    import json
    import redesign_h100_watch as watch
    states = iter(['stopping', 'stopped'])
    observed = []
    def cloud(args):
        if args[:2] == ['workspace', 'show']:
            state = next(states)
            observed.append(state)
            return json.dumps({'workspace': {'state': state, 'phase': 'standby'}})
        return ''
    with patch.object(watch, '_call', side_effect=cloud), patch.object(watch.time, 'sleep'), patch.object(watch, 'OUTPUT', tmp_path):
        watch._pause_and_verify({'workspace': 'wsp-fixture'}, 'launch_failure')
    assert observed == ['stopping', 'stopped']
    assert json.loads((tmp_path / 'resource_stop.json').read_text())['pause_verified']
