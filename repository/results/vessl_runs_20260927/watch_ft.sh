#!/usr/bin/env bash
# Pause the H100x4 when run $1 finishes (DONE) or its processes die; print the last evals.
RUN=$1; V=/Users/choiwuseck/Documents/ChatGPT/데이콘/_workspace/bin/vesslctl
while true; do
  st=$(ssh -n -o ConnectTimeout=15 -p 30229 root@capella.cloud.vessl.ai "test -f $RUN/DONE && echo done || (pgrep -f 'finetune_a[d]|score_f[t]|run_f[t]' >/dev/null && echo running || echo dead)" 2>/dev/null)
  [ "$st" = done ] || [ "$st" = dead ] && break
  sleep 60
done
ssh -n -p 30229 root@capella.cloud.vessl.ai "grep -h '\"eval\"' $RUN/metrics.jsonl | tail -12 | cut -c1-220; tail -3 $RUN/score.log 2>/dev/null; cat $RUN/DONE 2>/dev/null; ls $RUN" 2>&1
$V workspace pause wsp-9v8efltansxx >/dev/null 2>&1 && echo "paused H100x4 ($st) $(TZ=Asia/Seoul date '+%T KST')"
