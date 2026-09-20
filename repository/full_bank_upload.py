# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3"]
# ///
# Run: .venv/bin/python full_bank_upload.py
"""Stage every admitted original in bounded shards while the H100 stays stopped."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import tarfile
import zipfile
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig

import redesign_h100_watch as cloud

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/full_bank_mil_8020_20260918'
PREFIX = 'full-bank-mil-8020-20260918/data/'
PART = 16 * 1024 * 1024


def client():
    token = json.loads(cloud._call(['volume', 'token', 'objvol-a5gxdg7er70i', '-o', 'json']))['s3Token']
    connection = boto3.client('s3', region_name=token['region'], aws_access_key_id=token['accessKeyId'],
                              aws_secret_access_key=token['secretAccessKey'], aws_session_token=token['sessionToken'])
    return connection, token['bucketName'], token['prefix'].rstrip('/') + '/'


def upload(path: Path, key: str) -> dict:
    s3, bucket, base = client()
    sha, blocks = hashlib.sha256(), []
    with path.open('rb') as stream:
        while block := stream.read(PART):
            sha.update(block)
            blocks.append(hashlib.md5(block).digest())
    size = path.stat().st_size
    expected = (hashlib.md5(b''.join(blocks)).hexdigest() + '-' + str(len(blocks))
                if size >= PART else blocks[0].hex())
    s3.upload_file(str(path), bucket, base + key,
                   ExtraArgs={'Metadata': {'sha256': sha.hexdigest()}},
                   Config=TransferConfig(multipart_threshold=PART, multipart_chunksize=PART, max_concurrency=4))
    remote = s3.head_object(Bucket=bucket, Key=base + key)
    if remote['ContentLength'] != size or remote['ETag'].strip('"') != expected:
        raise ValueError('Uploaded object differs from exact local bytes: ' + key)
    return dict(key=key, bytes=size, sha256=sha.hexdigest(), etag=remote['ETag'], verification='S3 ETag matches every uploaded part')


def write_source(archive: zipfile.ZipFile, row: dict[str, str], data: bytes) -> dict[str, str]:
    digest = hashlib.sha256(data).hexdigest()
    if digest != row['source_sha256']:
        raise ValueError('Original bytes changed: ' + row['ID'])
    suffix = Path(row.get('path') or row['archive_member']).suffix.lower()
    member = 'audio/' + digest + suffix
    archive.writestr(zipfile.ZipInfo(member), data)
    return dict(ID=row['ID'], path=member, audio_sha256=digest)


def main() -> None:
    with (OUTPUT / 'all_sources.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if len({row['source_sha256'] for row in rows}) != len(rows):
        raise ValueError('Exact duplicate source bytes remain in admitted dataset')
    receipt_path = OUTPUT / 'audio_upload.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'shards': [], 'items': [], 'gpu_started': False}
    completed = {row['audio_sha256'] for row in receipt['items']}
    direct = [row for row in rows if row['path'] and row['source_sha256'] not in completed]
    members = {row['archive_member']: row for row in rows if not row['path'] and row['source_sha256'] not in completed}
    work = ROOT / '_workspace/full_bank_mil_8020_20260918'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='shard-', dir=work) as temporary:
        path = Path(temporary) / 'current.zip'
        archive = zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED)
        pending = []
        def flush() -> None:
            nonlocal archive, pending
            if not pending:
                return
            archive.close()
            key = PREFIX + f"audio_{len(receipt['shards']):03d}.zip"
            receipt['shards'].append(upload(path, key))
            receipt['items'].extend(pending)
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({'shards': len(receipt['shards']), 'files': len(receipt['items']), 'bytes': sum(x['bytes'] for x in receipt['shards'])}), flush=True)
            archive = zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED)
            pending = []
        for row in direct:
            pending.append(write_source(archive, row, (ROOT / row['path']).read_bytes()))
            if archive.fp.tell() >= 256 * 1024 * 1024:
                flush()
        if members:
            archive_paths = {row['archive_path'] for row in members.values()}
            if len(archive_paths) != 1:
                raise ValueError('Expected one recorded LibriSpeech archive')
            with tarfile.open(ROOT / archive_paths.pop(), 'r|gz') as original:
                for member in original:
                    if member.name not in members:
                        continue
                    incoming = original.extractfile(member)
                    if incoming is None:
                        raise ValueError('Expected audio file member')
                    pending.append(write_source(archive, members.pop(member.name), incoming.read()))
                    if archive.fp.tell() >= 256 * 1024 * 1024:
                        flush()
            if members:
                raise ValueError('LibriSpeech archive is missing declared originals')
        flush()
        archive.close()
    if {row['audio_sha256'] for row in receipt['items']} != {row['source_sha256'] for row in rows}:
        raise ValueError('Upload did not account for all admitted originals')
    receipt['all_sources_uploaded'] = True
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
