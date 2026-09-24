import os,signal,json,time
from pathlib import Path
checks={2513:b'/root/lb-a100-20260924-v1/prepared/code/data_v4/gpu_experiments.py',2520:b'/root/lb-a100-20260924-v1/prepared/code/full_bank_features.py',159:b'/opt/vessl/userdata.sh'}
for p,expected in checks.items():assert expected in Path(f'/proc/{p}/cmdline').read_bytes().split(b'\0')
os.kill(159,signal.SIGSTOP)
os.kill(2513,signal.SIGSTOP)
os.kill(2520,signal.SIGTERM)
print('AUTHORIZED_EXTRACTION_STOP_REQUESTED',time.time())
