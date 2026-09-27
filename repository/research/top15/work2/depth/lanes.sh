#!/bin/zsh
# two CPU lanes (4 torch threads each); every process hard-capped at 20 min via SIGALRM
cd /Users/choiwuseck/Documents/ChatGPT/데이콘/research/top15/work2/depth
export PYTHONDONTWRITEBYTECODE=1
PY=/Users/choiwuseck/Documents/ChatGPT/데이콘/.venv/bin/python
run() { local log=$1; shift; perl -e 'alarm 1200; exec @ARGV' /usr/bin/time -l $PY run.py "$@" > logs/$log.log 2>&1; echo "$log exit $?" >> logs/lanes.status; }
if [[ $1 == A ]]; then
  run df_a fullvoice_db mlp256 mlp512 mlp3 mlp4
  run df_b fullvoice_db resln resln_d1 resln_d3
  run music music mil128_r2 mil128 wide512 deep deep_d1 deep_d3 resln resln_d1
else
  run ad_a voice_ad resmlp mlp256 mlp512 mlp3 mlp4
  run ad_b voice_ad resln resln_d1 resln_d3 res_mlp3 res_resln_d1
fi
