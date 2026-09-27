#!/usr/bin/env bash
# Incrementally pull finished backbone npz files (never *.tmp.npz) until the remote DONE marker, then one final pass.
L=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/backbone_cache; mkdir -p $L
pull() {
  ssh -n -p 31082 root@capella.cloud.vessl.ai 'ls /root/backbone_cache' | grep "\.npz$" | grep -v "\.tmp\.npz$" | sort > /tmp/bb_remote.txt
  ls $L | sort > /tmp/bb_local.txt
  comm -23 /tmp/bb_remote.txt /tmp/bb_local.txt > /tmp/bb_new.txt
  [ -s /tmp/bb_new.txt ] && ssh -p 31082 root@capella.cloud.vessl.ai 'cd /root/backbone_cache && tar cf - -T -' < /tmp/bb_new.txt | tar xf - -C $L
  echo "$(TZ=Asia/Seoul date '+%T') pulled $(wc -l < /tmp/bb_new.txt) local $(ls $L | wc -l)"
}
while ! ssh -n -p 31082 root@capella.cloud.vessl.ai 'test -f /shared/backbones-20260927/logs/DONE'; do pull; sleep 600; done
sleep 5; pull; echo FINAL
