#!/usr/bin/env bash
# Exit when the remote backbone cache reaches $1 files, a shard logs a Traceback, or DONE exists.
target=$1
while true; do
  read -r n tb done_ < <(ssh -n -o ConnectTimeout=15 -p 31082 root@capella.cloud.vessl.ai \
    'echo $(ls /root/backbone_cache 2>/dev/null | grep -c "\.npz$") $(grep -l Traceback /shared/backbones-20260927/logs/shard_*.log 2>/dev/null | wc -l) $(test -f /shared/backbones-20260927/logs/DONE && echo 1 || echo 0)' 2>/dev/null)
  if [ "${tb:-0}" -gt 0 ] || [ "${done_:-0}" = 1 ] || [ "${n:-0}" -ge "$target" ]; then
    echo "npz=$n tracebacks=$tb done=$done_ $(TZ=Asia/Seoul date '+%T KST')"; exit 0
  fi
  sleep 30
done
