#!/bin/bash
# run_dump.sh MANIFEST OUT "GPU..." PER_GPU : PER_GPU workers per listed GPU, all sharing OUT (row claims by lock files)
M=$1; O=$2; G=($3); K=$4; N=$(( ${#G[@]} * K )); i=0; mkdir -p $O/_logs
for g in "${G[@]}"; do for k in $(seq 1 $K); do
  nohup /root/envs/data/bin/python /root/proj/research/top15/work5/newdata/dump_features.py /root/s7pkg/x $M $O --shard $i/$N --device cuda:$g > $O/_logs/w$i.log 2>&1 < /dev/null &
  i=$((i+1)); done; done; echo "launched $N workers"
