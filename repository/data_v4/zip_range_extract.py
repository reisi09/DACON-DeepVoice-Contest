"""Extract selected members from a remote (optionally split/spanned, ZIP64) zip via HTTP Range requests.

Used for SpeechFake (ModelScope): MD/ is a 115.7 GB split zip (MD.z01, MD.z02, MD.zip); only the Korean
members (~85k files) are needed. Parts are treated as one concatenated stream (absolute offset =
sum of earlier part sizes + per-disk offset). Every extracted member is checked against its CRC32.

  python data_v4/zip_range_extract.py --parts URL1 URL2 URL3 --sizes S1 S2 S3 --include 'MD/*/ko/*' --out DIR
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Remote:
    def __init__(self, urls: list[str], sizes: list[int]):
        self.urls, self.sizes = urls, sizes
        self.starts = [sum(sizes[:i]) for i in range(len(sizes))]
        self.total = sum(sizes)

    def _get(self, url: str, lo: int, hi: int) -> bytes:   # inclusive hi
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={lo}-{hi}', 'User-Agent': 'dacon-data-v4'})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if r.status != 206 or len(data) != hi - lo + 1:
                    raise IOError(f'range {lo}-{hi}: status {r.status}, {len(data)} bytes')
                return data
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))

    def read(self, off: int, n: int) -> bytes:
        """Absolute offset over the concatenated parts; a read may cross a part boundary."""
        out, end = [], off + n
        for i, (s, size) in enumerate(zip(self.starts, self.sizes)):
            lo, hi = max(off, s), min(end, s + size)
            if lo < hi:
                out.append(self._get(self.urls[i], lo - s, hi - s - 1))
        data = b''.join(out)
        if len(data) != n:
            raise IOError(f'short read at {off}: {len(data)}/{n}')
        return data


def central_directory(rem: Remote) -> list[dict]:
    n = min(rem.total, 1 << 20)
    tail = rem.read(rem.total - n, n)
    i = tail.rfind(b'PK\x05\x06')
    if i < 0:
        raise ValueError('EOCD not found')
    _, _, cd_disk, _, n_entries, cd_size, cd_off, _ = struct.unpack('<IHHHHIIH', tail[i:i + 22])
    if 0xFFFF in (n_entries, cd_disk) or 0xFFFFFFFF in (cd_size, cd_off):     # ZIP64
        j = tail.rfind(b'PK\x06\x07', 0, i)
        _, z64_disk, z64_off, _ = struct.unpack('<IIQI', tail[j:j + 20])
        z = rem.read(rem.starts[z64_disk] + z64_off, 56)
        if z[:4] != b'PK\x06\x06':
            raise ValueError('ZIP64 EOCD not found')
        _, _, _, _, _, cd_disk, _, n_entries, cd_size, cd_off = struct.unpack('<IQHHIIQQQQ', z)
    raw = rem.read(rem.starts[cd_disk] + cd_off, cd_size)
    entries, p = [], 0
    while p < len(raw):
        if raw[p:p + 4] != b'PK\x01\x02':
            raise ValueError(f'bad central header at {p}')
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, xlen, clen, disk,
         _, _, loff) = struct.unpack('<IHHHHHHIIIHHHHHII', raw[p:p + 46])
        name = raw[p + 46:p + 46 + nlen].decode('utf-8' if flags & 0x800 else 'cp437')
        extra = raw[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):                                        # ZIP64 extended info
            hid, hlen = struct.unpack('<HH', extra[q:q + 4])
            if hid == 1:
                vals, r = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack('<Q', vals[r:r + 8])[0]; r += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack('<Q', vals[r:r + 8])[0]; r += 8
                if loff == 0xFFFFFFFF:
                    loff = struct.unpack('<Q', vals[r:r + 8])[0]; r += 8
                if disk == 0xFFFF:
                    disk = struct.unpack('<I', vals[r:r + 4])[0]
            q += 4 + hlen
        entries.append(dict(name=name, method=method, crc=crc, csize=csize, usize=usize, abs=rem.starts[disk] + loff))
        p += 46 + nlen + xlen + clen
    if len(entries) != n_entries:
        raise ValueError(f'central directory has {len(entries)} entries, EOCD says {n_entries}')
    return entries


def runs(members: list[dict], gap: int = 1 << 20, cap: int = 256 << 20) -> list[list[dict]]:
    """Merge members that sit close together into one ranged read (members of one folder are contiguous)."""
    out, cur = [], []
    for m in sorted(members, key=lambda m: m['abs']):
        if cur and (m['abs'] - (cur[-1]['abs'] + cur[-1]['csize'] + 30 + 1024) > gap or m['abs'] - cur[0]['abs'] > cap):
            out.append(cur); cur = []
        cur.append(m)
    return out + ([cur] if cur else [])


def extract_run(rem: Remote, run: list[dict], out: Path) -> list[dict]:
    lo = run[0]['abs']
    hi = run[-1]['abs'] + 30 + 65535 * 2 + run[-1]['csize']                # local header upper bound
    buf = rem.read(lo, min(hi, rem.total) - lo)
    done = []
    for m in run:
        o = m['abs'] - lo
        if buf[o:o + 4] != b'PK\x03\x04':
            raise ValueError(f"bad local header: {m['name']}")
        nlen, xlen = struct.unpack('<HH', buf[o + 26:o + 30])
        data = buf[o + 30 + nlen + xlen:o + 30 + nlen + xlen + m['csize']]
        if m['method'] == 8:
            data = zlib.decompress(data, -15)
        elif m['method'] != 0:
            raise ValueError(f"unsupported method {m['method']}: {m['name']}")
        if len(data) != m['usize'] or zlib.crc32(data) != m['crc']:
            raise ValueError(f"CRC/size mismatch: {m['name']}")
        target = (out / m['name']).resolve()
        if not target.is_relative_to(out.resolve()):
            raise ValueError(f"unsafe member path: {m['name']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        done.append({'name': m['name'], 'bytes': len(data), 'crc32': m['crc']})
    return done


def modelscope_parts(dataset: str, paths: list[str]) -> tuple[list[str], list[int]]:
    """Resolve ModelScope dataset files to CDN URLs (Range-capable) and exact sizes via the public API."""
    base = f'https://www.modelscope.cn/api/v1/datasets/{dataset}'
    def opened(req):  # transient DNS/HTTP errors were seen; retry like Remote._get
        for attempt in range(6):
            try:
                return urllib.request.urlopen(req, timeout=120)
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
    urls, sizes = [], []
    for path in paths:
        folder = '/' + path.rsplit('/', 1)[0] if '/' in path else '/'
        q = f'{base}/repo/tree?Revision=master&Root={folder}&Recursive=false&PageNumber=1&PageSize=500'
        files = json.load(opened(urllib.request.Request(q, headers={'User-Agent': 'x'})))['Data']['Files']
        sizes.append(next(f['Size'] for f in files if f['Path'] == path))
        req = urllib.request.Request(f'{base}/repo?Revision=master&FilePath={path}', headers={'User-Agent': 'x', 'Range': 'bytes=0-0'})
        with opened(req) as r:
            urls.append(r.geturl())
    return urls, sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', help='part URLs in disk order (or use --modelscope)')
    ap.add_argument('--sizes', nargs='+', type=int)
    ap.add_argument('--modelscope', nargs='+', metavar=('DATASET', 'PATH'), help='DATASET PART_PATH [PART_PATH ...]')
    ap.add_argument('--include', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='test: first N selected members')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    parts, sizes = modelscope_parts(a.modelscope[0], a.modelscope[1:]) if a.modelscope else (a.parts, a.sizes)
    rem = Remote(parts, sizes)
    entries = central_directory(rem)
    sel = [e for e in entries if not e['name'].endswith('/') and any(fnmatch.fnmatch(e['name'], g) for g in a.include)]
    sel = sorted(sel, key=lambda e: e['abs'])[:a.limit or None]
    print(json.dumps({'entries': len(entries), 'selected': len(sel), 'selected_GB': round(sum(e['csize'] for e in sel) / 1e9, 2)}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    receipt, failed = [], []
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(extract_run, rem, r, a.out): r for r in runs(sel)}
        for i, f in enumerate(futs):
            try:
                receipt += f.result()
            except Exception as exc:
                failed += [m['name'] for m in futs[f]]
                print(f'run failed ({len(futs[f])} members): {type(exc).__name__}: {exc}'[:300], flush=True)
            if i % 20 == 0:
                print(f'runs {i + 1}/{len(futs)} files {len(receipt)}', flush=True)
    (a.out / 'extract_receipt.json').write_text(json.dumps({'selected': len(sel), 'extracted': len(receipt), 'failed': failed,
                                                           'files': receipt}) + '\n')
    print(json.dumps({'extracted': len(receipt), 'failed': len(failed)}), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
