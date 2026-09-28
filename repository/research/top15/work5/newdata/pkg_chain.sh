#!/bin/bash
# pkg_chain.sh NAME CANDIDATE "HEADS NOTE"  (Mac, project root): fetch P1 heads from x1, package, CPU verify, runtime parity,
# then stage the changed members + receipt on the VESSL volume for the CUDA check (verify_s8_vessl.py).
set -euo pipefail
N=$1; CAND=$2; NOTE=$3; P=/Users/choiwuseck/Documents/ChatGPT/데이콘; H=_workspace/final_${N}_20260928; O=submissions/DataV4_${N}
PORT=${PORT:-31082}; SSH="ssh -n -o ConnectTimeout=20 -p $PORT root@capella.cloud.vessl.ai"; S=/private/tmp/claude-501/pkg_${N}
cd $P
mkdir -p $H $O
scp -q -P $PORT "root@capella.cloud.vessl.ai:/root/proj/$H/*" $H/
scp -q -P $PORT root@capella.cloud.vessl.ai:/root/proj/_workspace/final_${N}_20260928.log root@capella.cloud.vessl.ai:/root/proj/_workspace/final_${N}L_20260928.log _workspace/
.venv/bin/python package_datav4_submission.py --heads $H --output $O/submit.zip --candidate $CAND --heads-note "$NOTE" \
  --split-note "train 30,658 + held-out 6,450 + new rows (research/top15/work5/newdata/rows) / val 4,931" --fusion $H \
  --antideepfake ad=_workspace/models/xls-r-1b-anti-deepfake-fp16 > $O.package.log 2>&1
grep -E '"(sha256|bytes)"' $O.package.log | head -2
.venv/bin/python verify_datav4_submission.py --out $O > $O/verify.log 2>&1 && echo "cpu verify ok"
OMP_NUM_THREADS=6 HEADS=$H VERIFICATION=$O/qa/verification.json RECEIPT=$O/submit.json .venv/bin/python research/top15/work5/w/w_runtime_parity.py 2>&1 | grep -v -i warn > $O/runtime_parity.txt
tail -1 $O/runtime_parity.txt
rm -rf $S && mkdir -p $S/changed && .venv/bin/python - "$O" "$S/changed" <<'EOF'
import json, sys, zipfile
o, dst = sys.argv[1:3]
a = json.load(open('submissions/DataV4_S7_SSLAMWavLM/submit.json'))['member_sha256']
b = json.load(open(f'{o}/submit.json'))['member_sha256']
changed = sorted(n for n, d in b.items() if a.get(n) != d)
assert not set(a) - set(b), 'members removed'
with zipfile.ZipFile(f'{o}/submit.zip') as z:
    for n in changed:
        z.extract(n, dst)
print(len(changed), 'changed members')
EOF
(cd $S && COPYFILE_DISABLE=1 tar czf changed.tgz -C changed .)
$SSH "mkdir -p /root/pkg_${N}"
scp -q -P $PORT $S/changed.tgz $O/submit.json root@capella.cloud.vessl.ai:/root/pkg_${N}/
echo "staged for CUDA check: /root/pkg_${N}"
