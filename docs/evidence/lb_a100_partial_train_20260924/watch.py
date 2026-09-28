"""Replacement guard for the user-authorized 30-minute training window; creates no compute."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DEADLINE = 1790257450
CLI = ['/Users/choiwuseck/.local/bin/vesslctl', '--org', 'qwdkw']
WORKSPACE = 'wsp-nvxqw1jf9zdv'
OUT = Path('../lb_a100_run_20260924_v1/monitor_partial')
OUT.mkdir(exist_ok=False)

def call(args, timeout=40):
    return subprocess.run(CLI+args, capture_output=True, text=True, check=True, timeout=timeout).stdout

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')

try:
    w=json.loads(call(['workspace','show',WORKSPACE,'-o','json']))['workspace']
    assert w['name']=='lb-e00-e04-a100-20260924-v1'
    old=subprocess.check_output(['ps','-p','16643','-o','command='],text=True)
    assert 'a100_workspace_watch.py --request ../lb_a100_run_20260924_v1/launch_request.json' in old
    os.kill(16643,signal.SIGTERM)
    save('controller.json',{'workspace':WORKSPACE,'deadline_epoch':DEADLINE,'previous_controller_replaced':True})
    print('REPLACEMENT_GUARD_ACTIVE',flush=True)
    while time.time()<DEADLINE-45:
        call(['volume','download','objvol-a5gxdg7er70i',str(OUT/'downloaded'),
              '--remote-prefix','lb-improvement-20260924/a100-run-v1/', '--overwrite',
              '--exclude','bootstrap_bundle.zip','--exclude','results.zip'],min(90,max(1,DEADLINE-time.time()-45)))
        receipt=OUT/'downloaded/export_receipt.json'
        if receipt.exists():
            r=json.loads(receipt.read_text())
            call(['volume','download','objvol-a5gxdg7er70i',str(OUT/'downloaded'),
                  '--remote-prefix','lb-improvement-20260924/a100-run-v1/', '--overwrite',
                  '--exclude','bootstrap_bundle.zip'],min(180,max(1,DEADLINE-time.time()-45)))
            sys.path.insert(0,str(Path('repository').resolve()))
            from redesign_h100_watch import verify_archive
            v=verify_archive(OUT/'downloaded/results.zip',r['archive_sha256'],OUT/'unpacked')
            save('verified_export.json',{'verification':v,'receipt':r})
            print('RESULTS_VERIFIED',flush=True)
            break
        if (OUT/'downloaded/partial_training_exit.json').exists():break
        time.sleep(max(0,min(300,DEADLINE-time.time()-45)))
except Exception as e:
    save('error.json',{'type':type(e).__name__})
    print('GUARD_ERROR',type(e).__name__,flush=True)
finally:
    for attempt in range(3):
        try:
            w=json.loads(call(['workspace','show',WORKSPACE,'-o','json']))['workspace']
            assert w['name']=='lb-e00-e04-a100-20260924-v1'
            if w['state'] not in ('stopped','stopping','terminating'):call(['workspace','pause',WORKSPACE])
            w=json.loads(call(['workspace','show',WORKSPACE,'-o','json']))['workspace']
            save('pause.json',{k:w[k] for k in ('slug','state','phase')})
            if w['state'] in ('stopped','stopping','terminating'):
                print('PAUSE_CONFIRMED',flush=True);break
        except Exception as e:print('PAUSE_RETRY',type(e).__name__,flush=True)
        time.sleep(5)
