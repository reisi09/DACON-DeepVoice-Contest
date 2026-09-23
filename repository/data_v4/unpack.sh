#!/usr/bin/env bash
# Unpack data_v4 archives (object volume) into RAW/<source>/ in the layout build_manifest's adapters expect.
# Parallel over sources (per-source mkdir lock), pigz for .gz. Idempotent: RAW/<source>/.unpacked marks a
# finished source; inner archives are removed only after a successful extraction.
set -euo pipefail
export ARCH=${ARCH:-/shared/data-v4-20260923/archives} RAW=${RAW:-/data/data_v4/raw} PY=${PY:-python3}
JOBS=${JOBS:-6}
mkdir -p "$RAW"
x() { echo "[$(date -u +%T)] $*"; }
unz() { $PY -m zipfile -e "$1" "$2" && rm -f "$1"; }            # no unzip binary on the slim image
untgz() { tar -I pigz -xf "$1" -C "$2" && rm -f "$1"; }          # gzip -d was single-thread bound (~9 MB/s)
unpack_one() {
  local src=$1 d="$RAW/$1"
  [ -f "$ARCH/$src.tar.done" ] || { x "skip $src (no .done marker)"; return 0; }
  [ -f "$d/.unpacked" ] && { x "skip $src (unpacked)"; return 0; }
  mkdir "$RAW/.lock_$src" 2>/dev/null || { x "skip $src (locked by another worker)"; return 0; }
  x "unpack $src"
  tar -xf "$ARCH/$src.tar" -C "$RAW"
  case "$src" in
    asvspoof2021_la|asvspoof2021_df) for t in "$d"/*.tar.gz; do untgz "$t" "$d"; done ;;
    vocalset|csd|echoes)             for z in "$d"/*.zip; do unz "$z" "$d"; done ;;
    sonics)                          for z in "$d"/fake_songs/*.zip; do unz "$z" "$d/fake_songs"; done ;;
    openmic|musan)                   for t in "$d"/*.tgz "$d"/*.tar.gz; do [ -f "$t" ] && untgz "$t" "$d"; done ;;
    fleurs_ko)                       for t in "$d"/data/ko_kr/audio/*.tar.gz; do untgz "$t" "$d/data/ko_kr/audio"; done ;;
  esac
  touch "$d/.unpacked"; rmdir "$RAW/.lock_$src"
  x "done $src ($(du -sh "$d" | cut -f1))"
}
export -f x unz untgz unpack_one
ls "$ARCH"/*.tar | xargs -n1 basename | sed 's/\.tar$//' | xargs -P "$JOBS" -I{} bash -c 'unpack_one {}'
# legacy bank (16,042 rows): 56 zip shards -> RAW/existing/audio/<sha256>.<ext>
if [ ! -f "$RAW/existing/.unpacked" ]; then
  mkdir -p "$RAW/existing"
  ls /shared/full-bank-mil-8020-20260918/data/audio_*.zip | xargs -P 8 -I{} "$PY" -m zipfile -e {} "$RAW/existing"
  touch "$RAW/existing/.unpacked"; x "done existing ($(du -sh "$RAW/existing" | cut -f1))"
fi
missing=$(for t in "$ARCH"/*.tar; do s=$(basename "$t" .tar); [ -f "$RAW/$s/.unpacked" ] || echo "$s"; done)
[ -z "$missing" ] || { x "NOT UNPACKED: $missing"; exit 1; }
df -h "$RAW" | tail -1
