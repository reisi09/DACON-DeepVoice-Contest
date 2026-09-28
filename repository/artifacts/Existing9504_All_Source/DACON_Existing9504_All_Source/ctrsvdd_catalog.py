"""Join verified CtrSVDD originals to protocols and fully decode every available clip.

Run: .venv/bin/python ctrsvdd_catalog.py --split dev
SVC input-singer relationships are unresolved; those clips remain diagnostic.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

from candidate_data_acquire import BANK, OUT, ROOT, SourceFile, verify_file
from candidate_data_catalog import safe_member
from full_bank_split import write_csv
from reconstruct_candidate_bank import audio_quality, receipts, sha


def catalog(split):
    output = OUT / ('ctrsvdd_' + split)
    if (output / 'audit.json').exists():
        raise RuntimeError('Completed audit exists; preserve this snapshot')
    completed = receipts()
    archive_name, protocol_name = f'CtrSVDD/{split}_set.zip', f'CtrSVDD/{split}.txt'
    for name in (archive_name, protocol_name):
        receipt = completed[name]
        job = SourceFile(**{key: receipt[key] for key in SourceFile.__dataclass_fields__})
        if verify_file(BANK / name, job) != receipt['sha256']:
            raise RuntimeError('Verified original changed: ' + name)
    protocol = {}
    for line in (BANK / protocol_name).read_text().splitlines():
        fields = line.split()
        if len(fields) != 6 or fields[5] not in ('bonafide', 'deepfake') or fields[2] in protocol:
            raise RuntimeError('Malformed/duplicate protocol row: ' + line)
        protocol[fields[2]] = fields
    output.mkdir(parents=True, exist_ok=True)
    originals = BANK / 'CtrSVDD' / (split + '_originals')
    originals.mkdir(exist_ok=True)
    rows = []
    with zipfile.ZipFile(BANK / archive_name) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if any(not safe_member(m.filename) or Path(m.filename).suffix != '.flac' for m in members):
            raise RuntimeError('Unexpected archive entry')
        ids = [Path(m.filename).stem for m in members]
        if len(ids) != len(set(ids)) or set(ids) - protocol.keys():
            raise RuntimeError('Archive IDs repeat or are absent from protocol')
        def inspect(member):
            identifier = Path(member.filename).stem
            source_corpus, singer, _, _, attack, label = protocol[identifier]
            fake = str(int(label == 'deepfake'))
            row = dict(ID='ctrsvdd_' + identifier, source='CtrSVDD', generator=attack if fake=='1' else 'REAL',
                       source_corpus=source_corpus, target_singer_id=singer, official_partition=split,
                       route='voice', kind='a_cappella_singing', FILE_FAKE=fake, VOICE_FAKE=fake,
                       VOICE_PRESENT='1', MUSIC_PRESENT='0', MUSIC_FAKE='', FILE_FAKE_mask='1',
                       VOICE_FAKE_mask='1', VOICE_PRESENT_mask='1', MUSIC_PRESENT_mask='1', MUSIC_FAKE_mask='0',
                       license='CC-BY-NC-ND-4.0', license_url='https://creativecommons.org/licenses/by-nc-nd/4.0/',
                       source_url=completed[archive_name]['url'], archive_sha256=completed[archive_name]['sha256'],
                       archive_path=str((BANK/archive_name).relative_to(ROOT)), archive_member=member.filename,
                       protocol_sha256=completed[protocol_name]['sha256'],
                       group_id='singer:ctrsvdd:' + singer, new_collection='True', input_kind='original_full_duration',
                       historical_development_exposure='False', prior_seen='False', parent_ids='[]',
                       label_status='source_annotation; no_independent_listening', path='', split='')
            flags = ['svc_input_singer_and_recording_unresolved'] if attack in ('A06','A07','A08') else []
            try:
                payload = archive.read(member)  # Full ZIP read verifies the member's CRC too.
                row['source_sha256'] = hashlib.sha256(payload).hexdigest()
                row['source_bytes'] = str(len(payload))
                row['zip_crc_verified'] = 'True'
                quality = audio_quality(io.BytesIO(payload))
                row['group_id'] += '|' + quality.pop('group_pcm')
                row.update(quality)
                if row['sample_rate'] != '16000' or row['channels'] != '1':
                    flags.append('unexpected_audio_layout')
                if not flags:
                    target = originals / (identifier + '.flac')
                    if target.exists():
                        if sha(target) != row['source_sha256']:
                            raise RuntimeError('Extracted original changed: ' + identifier)
                    else:
                        partial = target.with_suffix('.flac.part')
                        partial.write_bytes(payload)
                        partial.rename(target)
                    row['path'] = str(target.relative_to(ROOT))
            except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as error:
                flags.append('audio_integrity_or_decode:' + str(error).split(': <_io.BytesIO')[0])
            row.update(admission='diagnostic' if flags else 'admitted', reason=';'.join(flags))
            return row
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, row in enumerate(pool.map(inspect, members), 1):
                rows.append(row)
                if i % 2000 == 0:
                    print('Fully decoded', i, '/', len(members), flush=True)
    missing = [dict(source_corpus=protocol[key][0], singer=protocol[key][1], ID=key, label=protocol[key][-1],
                    reason='not_distributed_in_official_archive') for key in sorted(protocol.keys()-set(ids))]
    write_csv(output/'catalog.csv', rows)
    write_csv(output/'missing_audio.csv', missing)
    audit = dict(source_split=split, archive_sha256=completed[archive_name]['sha256'],
                 protocol_rows=len(protocol), actual_audio_rows=len(rows), missing_protocol_audio=len(missing),
                 missing_by_source_and_label=dict(Counter(x['source_corpus']+':'+x['label'] for x in missing)),
                 actual_by_label=dict(Counter(x['FILE_FAKE'] for x in rows)),
                 rows_by_admission=dict(Counter(x['admission'] for x in rows)),
                 diagnostic_reasons=dict(Counter(x['reason'] for x in rows if x['reason'])),
                 full_decode_pass=sum(x.get('decode_status')=='full_decode_pass' for x in rows),
                 checksum_and_crc_pass=sum(x.get('zip_crc_verified')=='True' for x in rows),
                 admitted_singer_groups=len({x['target_singer_id'] for x in rows if x['admission']=='admitted'}),
                 selected_for_training=False, gpu_started=False,
                 parent_policy='SVS and bona fide grouped by all clips of target singer; SVC not admitted without input parent mapping',
                 sources=['https://github.com/SVDDChallenge/CtrSVDD_Utils','https://arxiv.org/html/2406.02438v2'],
                 catalog_sha256=sha(output/'catalog.csv'))
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('train','dev'),default='dev')
    catalog(parser.parse_args().split)
