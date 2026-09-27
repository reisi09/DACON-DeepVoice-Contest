"""Label completion rule L (research/top15/work4/choice_lock.json): a present vocal of unknown authenticity inside a
fully generated track is AI-generated, so VOICE_FAKE := 1 (competition definition: any AI-generated component is fake).
Only whole-track generators qualify; partial edits (HAIM crossfade/concat, event intervals), constructed mixes and
reference-conditioned Echoes ATA rows stay unknown. FakeMusicCaps rows qualify only when their MusicCaps caption
mentions vocals (fmc_caption_vocal.json, google/MusicCaps rev 0a51889b).
"""
from __future__ import annotations

import json
from pathlib import Path

FULL_TRACK = ('existing:SONICS', 'existing:suno', 'existing:udio')
FMC = frozenset(json.loads((Path(__file__).resolve().parent / 'fmc_caption_vocal.json').read_text())['orig_ids'])


def completes(r) -> bool:
    if not (r['VOICE_PRESENT'] == '1' and r['VOICE_PRESENT_mask'] == '1' and r['VOICE_FAKE_mask'] != '1'
            and r['MUSIC_FAKE'] == '1' and r['MUSIC_FAKE_mask'] == '1' and r['event_intervals'] in ('', '[]')):
        return False
    s = r['source']
    return (s in FULL_TRACK or (s == 'existing:HAIM' and r['generator'] in ('acestep', 'musicgen'))
            or (s.startswith('existing:Echoes/') and s.endswith('/TTA')) or (s == 'existing:FakeMusicCaps' and r['orig_id'] in FMC))


def complete(r):
    return {**r, 'VOICE_FAKE': '1', 'VOICE_FAKE_mask': '1'} if completes(r) else r


if __name__ == '__main__':
    base = dict(VOICE_PRESENT='1', VOICE_PRESENT_mask='1', VOICE_FAKE='', VOICE_FAKE_mask='0', MUSIC_FAKE='1', MUSIC_FAKE_mask='1',
                event_intervals='', source='existing:SONICS', generator='suno', orig_id='x')
    assert complete(base)['VOICE_FAKE'] == '1'
    assert not completes({**base, 'event_intervals': '[[0, 1, 1]]'}), 'partial tracks stay unknown'
    assert not completes({**base, 'source': 'haim', 'generator': 'crossfade'})
    assert not completes({**base, 'source': 'existing:Echoes/suno/ATA'})
    assert not completes({**base, 'source': 'existing:FakeMusicCaps'}), 'FakeMusicCaps needs a vocal caption'
    assert not completes({**base, 'VOICE_FAKE': '0', 'VOICE_FAKE_mask': '1'}), 'known labels are never changed'
    print('label_rules self-check ok')
