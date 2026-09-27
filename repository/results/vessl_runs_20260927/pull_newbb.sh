#!/usr/bin/env bash
# Incrementally pull finished backbone npz files (never *.tmp.npz) until the remote DONE marker, then one final pass.
L=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/newbackbone_cache; mkdir -p $L
pull() {
  ssh -n -p 31082 root@capella.cloud.vessl.ai 'ls /root/newbackbone_cache' | grep "\.npz$" | grep -v "\.tmp\.npz$" | sort > /tmp/nbb_remote.txt
  ls $L | sort > /tmp/nbb_local.txt
  comm -23 /tmp/nbb_remote.txt /tmp/nbb_local.txt > /tmp/nbb_new.txt
  [ -s /tmp/nbb_new.txt ] && ssh -p 31082 root@capella.cloud.vessl.ai 'cd /root/newbackbone_cache && tar cf - -T -' < /tmp/nbb_new.txt | tar xf - -C $L
  echo "$(TZ=Asia/Seoul date '+%T') pulled $(wc -l < /tmp/nbb_new.txt) local $(ls $L | wc -l)"
}
while ! ssh -n -p 31082 root@capella.cloud.vessl.ai 'test -f /shared/newbackbones-20260927/logs/DONE'; do pull; sleep 600; done
sleep 5; pull; echo FINAL
