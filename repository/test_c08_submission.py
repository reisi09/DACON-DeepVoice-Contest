"""Protect the deployed C08 probabilities against checkpoint, pooling and fusion drift."""
import io
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np

from c08_submission import score_record
from music_temporal import load_temporal_head


class C08SubmissionChecks(unittest.TestCase):
    def test_deployed_scores_reproduce_h100_c08_on_real_cached_audio(self):
        # Given: actual C08 seed41 and its immutable H100 evaluation records.
        root=Path(__file__).resolve().parent
        run=root/'results/temporal_h100_compact_20260915/unpacked'
        manifest=json.loads((run/'features_manifest.json').read_text())
        baseline=json.loads((run/'results/Official_baseline_test_predictions.json').read_text())
        expected=json.loads((run/'results/C08_seed41_test_predictions.json').read_text())
        head=load_temporal_head(run/'results/C08_seed41/temporal_music.pt')
        with zipfile.ZipFile(root/'_workspace/h100_portable_backup/features.zip') as archive:
            actual=[]
            for row in baseline:
                item=manifest['records'][row['ID']]
                with np.load(io.BytesIO(archive.read(item['path'])),allow_pickle=False) as z:
                    record={k:z[k] for k in ('music','music_audible','starts')}
                # When: use the submission's real music-head and file-fusion path.
                actual.append(score_record(head,record,row['VOICE_FAKE_PROB'],
                    (row['VOICE_PRESENT_PROB'],row['MUSIC_PRESENT_PROB']),'cpu'))
        # Then: all five probabilities agree with recorded H100 results.
        fields=['FILE_FAKE_PROB','VOICE_FAKE_PROB','MUSIC_FAKE_PROB','VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB']
        np.testing.assert_allclose(actual,[[r[k] for k in fields] for r in expected],atol=1e-6,rtol=0)


if __name__=='__main__':
    unittest.main()
