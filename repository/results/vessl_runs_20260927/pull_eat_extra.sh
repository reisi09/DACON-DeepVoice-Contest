#!/usr/bin/env bash
# Pull finished /root/eat_extra_cache npz (never *.tmp.npz) from H100x1 until DONE, verify count, then pause the workspace.
V=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/bin/vesslctl
L=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/eat_extra_cache; mkdir -p $L
T=/private/tmp/claude-501/-Users-choiwuseck/a25211a1-372e-4ca8-a15c-ca4634f5b407/scratchpad
pull() {
  ssh -n -p 31082 root@capella.cloud.vessl.ai 'ls /root/eat_extra_cache' | grep "\.npz$" | grep -v "\.tmp\.npz$" | sort > $T/eat_remote.txt
  ls $L | sort > $T/eat_local.txt
  comm -23 $T/eat_remote.txt $T/eat_local.txt > $T/eat_new.txt
  [ -s $T/eat_new.txt ] && ssh -p 31082 root@capella.cloud.vessl.ai 'cd /root/eat_extra_cache && tar cf - -T -' < $T/eat_new.txt | tar xf - -C $L
  echo "$(TZ=Asia/Seoul date '+%T') pulled $(wc -l < $T/eat_new.txt) local $(ls $L | grep -c '\.npz$')"
}
until ssh -n -p 31082 root@capella.cloud.vessl.ai 'test -f /root/eat_extra_logs/DONE || ! pgrep -f "extract_new_backbone[s]" >/dev/null'; do pull; sleep 120; done
sleep 5; pull
r=$(ssh -n -p 31082 root@capella.cloud.vessl.ai 'ls /root/eat_extra_cache | grep "\.npz$" | grep -vc "\.tmp\.npz$"; cat /root/eat_extra_logs/DONE; grep -l Traceback /root/eat_extra_logs/shard_*.log')
echo "remote: $r"
l=$(ls $L | grep -c '\.npz$'); first=$(echo "$r" | head -1)
if [ "$l" -ge "$first" ]; then $V workspace pause wsp-9683432r72m7 >/dev/null 2>&1 && echo "paused H100x1 local=$l $(TZ=Asia/Seoul date '+%T KST')"; else echo "NOT paused: local $l < remote $first"; fi
