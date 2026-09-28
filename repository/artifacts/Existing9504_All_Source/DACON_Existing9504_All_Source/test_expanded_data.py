"""Protect cross-split prompt, transformed-parent and exact-audio isolation."""
import json

import pytest

from expanded_data import SplitError, audit_groups


@pytest.mark.parametrize('field,value', [('group_id', 'prompt:shared'), ('source_sha256', 'same-original'), ('parent_ids', json.dumps(['same-parent']))])
def test_audit_rejects_shared_identity_across_splits(field: str, value: str) -> None:
    rows = [dict(ID='one', group_id='one', split='train'), dict(ID='two', group_id='two', split='test')]
    for row in rows:
        row[field] = value
    with pytest.raises(SplitError, match='Cross-split identity'):
        audit_groups(rows)
    rows[1]['split'] = 'train'
    audit_groups(rows)
