import pytest
from panns_training_state import EpochSampler


def test_shuffle_covers_every_file_and_resume_keeps_next_batches():
    sampler = EpochSampler(11, 41)
    first = [sampler.next_batch() for _ in range(3)]
    assert sorted(sum(first, [])) == list(range(11))
    assert [len(batch) for batch in first] == [4, 4, 3]
    sampler.next_batch()
    saved = sampler.state_dict()
    expected = [sampler.next_batch() for _ in range(8)]
    restored = EpochSampler(11, 41)
    restored.load_state_dict(saved)
    assert [restored.next_batch() for _ in range(8)] == expected
    broken = {**saved, 'order': [0] * 11}
    with pytest.raises(ValueError, match='permutation'):
        restored.load_state_dict(broken)
