#!/usr/bin/env bash
# Pull finished /root/s6_cache npz (never *.tmp.npz) from H100x4 until DONE, verify count, then pause the workspace.
V=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/bin/vesslctl
L=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/s6_cache; mkdir -p $L
T=/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad
pull() {
  ssh -n -p 30229 root@capella.cloud.vessl.ai 'ls /root/s6_cache' | grep "\.npz$" | grep -v "\.tmp\.npz$" | sort > $T/s6_remote.txt
  ls $L | sort > $T/s6_local.txt
  comm -23 $T/s6_remote.txt $T/s6_local.txt > $T/s6_new.txt
  [ -s $T/s6_new.txt ] && ssh -p 30229 root@capella.cloud.vessl.ai 'cd /root/s6_cache && tar cf - -T -' < $T/s6_new.txt | tar xf - -C $L
  echo "$(TZ=Asia/Seoul date '+%T') pulled $(wc -l < $T/s6_new.txt) local $(ls $L | grep -c '\.npz$')"
}
until ssh -n -p 30229 root@capella.cloud.vessl.ai 'test -f /root/s6_logs/DONE || ! pgrep -f "extract_s[6]" >/dev/null'; do pull; sleep 120; done
sleep 5; pull
r=$(ssh -n -p 30229 root@capella.cloud.vessl.ai 'ls /root/s6_cache | grep "\.npz$" | grep -vc "\.tmp\.npz$"; cat /root/s6_logs/DONE; grep -l Traceback /root/s6_logs/shard_*.log')
echo "remote: $r"
l=$(ls $L | grep -c '\.npz$'); first=$(echo "$r" | head -1)
if [ "$l" -ge "$first" ]; then $V workspace pause wsp-9v8efltansxx >/dev/null 2>&1 && echo "paused H100x4 local=$l $(TZ=Asia/Seoul date '+%T KST')"; else echo "NOT paused: local $l < remote $first"; fi
