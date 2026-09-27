#!/usr/bin/env bash
# Auto-pause idle GPU workspaces: H100x4 once stems are done; H100x1 once MuQ/EAT extraction is done and fully pulled.
V=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/bin/vesslctl
L=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/newbackbone_cache
x4=0; x1=0
while [ $x4 = 0 ] || [ $x1 = 0 ]; do
  if [ $x4 = 0 ] && ssh -n -o ConnectTimeout=15 -p 30229 root@capella.cloud.vessl.ai 'test -f /shared/ft-20260927/logs/STEMS_DONE' 2>/dev/null; then
    ssh -n -p 30229 root@capella.cloud.vessl.ai 'cat /shared/ft-20260927/logs/STEMS_DONE; grep -l Traceback /shared/ft-20260927/logs/stems_*.log | wc -l'
    $V workspace pause wsp-9v8efltansxx >/dev/null 2>&1 && echo "paused H100x4 $(TZ=Asia/Seoul date '+%T KST')"; x4=1
  fi
  if [ $x1 = 0 ] && ssh -n -o ConnectTimeout=15 -p 31082 root@capella.cloud.vessl.ai 'test -f /shared/newbackbones-20260927/logs/DONE' 2>/dev/null; then
    r=$(ssh -n -p 31082 root@capella.cloud.vessl.ai 'ls /root/newbackbone_cache | grep "\.npz$" | grep -v "\.tmp\.npz$" | wc -l'); l=$(ls $L 2>/dev/null | grep -c "\.npz$")
    if [ "$r" -gt 0 ] && [ "$l" -ge "$r" ]; then
      $V workspace pause wsp-9683432r72m7 >/dev/null 2>&1 && echo "paused H100x1 remote=$r local=$l $(TZ=Asia/Seoul date '+%T KST')"; x1=1
    fi
  fi
  sleep 60
done
$V billing show 2>&1 | grep -E "Balance|Burn"
