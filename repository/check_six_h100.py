"""Preflight checks that protect actual split, label, and fusion correctness.

Run: .venv/bin/python check_six_h100.py
"""
import csv
import json
from pathlib import Path

import nbformat
import numpy as np

from six_h100_experiments import fuse, metrics
from v2_data import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = list(csv.DictReader((root/'results/six_h100_20260917/probes.csv').open()))
    groups, hashes, parents = {}, {}, {}
    for row in rows:
        assert sha256_file(root/'_workspace/six_h100_20260917'/row['path']) == row['audio_sha256']
        for mapping, identifiers in ((groups, row['group_id'].split('|')),
                                      (hashes, [row['audio_sha256']]),
                                      (parents, json.loads(row['parent_ids']))):
            for identifier in identifiers:
                assert mapping.setdefault(identifier, row['split']) == row['split'], 'cross-split leakage'
        if row['source'] in ('FMA','suno','udio'):
            assert row['VOICE_FAKE']==row['VOICE_PRESENT']=='', 'unverified voice label was invented'
        if row['variant']=='partial_voice':
            event=json.loads(row['event_intervals'])[0]
            assert abs(event['end']-event['start']-3.) < 1e-9 and row['VOICE_FAKE']=='1'
    # Different windows supply the two maxima: File must decrease without changing the four outputs.
    record={'valid':np.array([True,True]),'voice_probability':np.array([.1,.9]),
            'presence':np.array([[.9,.8],[.1,.7]])}
    values=[0.,.9,.2,.9,.8]
    aligned, independent=fuse(values,record,'aligned'),fuse(values,record)
    assert aligned[0] < independent[0] and aligned[1:]==independent[1:]==values[1:]
    labels=[dict(ID='r',FILE_FAKE='0',MUSIC_FAKE='0',MUSIC_PRESENT='1',group_id='a'),
            dict(ID='f',FILE_FAKE='1',MUSIC_FAKE='1',MUSIC_PRESENT='1',group_id='b')]
    report=metrics(labels,{'r':[.1,.2,.1,.8,.9],'f':[.9,.8,.9,.8,.9]})
    assert report['score'] is None and report['metrics']['file_eer']==report['metrics']['music_eer']==0
    notebook=nbformat.read(root/'DACON_Six_Experiments_H100.ipynb',as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            compile(cell.source,'H100 notebook','exec')
    receipt=dict(audio_files_verified=len(rows),group_hash_parent_split_overlap=0,
                 partial_voice_seconds=3, fusion_invariant='four component outputs preserved',
                 unknown_labels_masked=True, notebook_executed_on_gpu=False)
    (root/'results/six_h100_20260917/local_preflight.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
