import csv, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'data_v4')
from full_bank_evaluate import score_rows, five_columns
from sklearn.metrics import roc_curve
B='results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
H=Path(sys.argv[1])
pres=json.load(open('results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json'))
ens=json.load(open(H/'file_ensemble.json'))
for split,bank,cache in (('val','bank_trainval.csv','_workspace/cache_v4/cache_tv'),('test','bank_test.csv','_workspace/cache_v4/cache_test')):
    rows=[r for r in csv.DictReader(open(B+bank)) if r['split']==split]
    S=score_rows(rows,Path(cache),H,'cpu'); P=[five_columns(s,pres,ens) for s in S]
    for comp,col,mask in (('FILE','FILE_FAKE_PROB','FILE_FAKE_mask'),('VOICE','VOICE_FAKE_PROB','VOICE_FAKE_mask'),('MUSIC','MUSIC_FAKE_PROB','MUSIC_FAKE_mask')):
        idx=[i for i,r in enumerate(rows) if r[mask]=='1' and (comp=='FILE' or r[comp+'_PRESENT']=='1')]
        y=np.array([int(rows[i][comp+'_FAKE']) for i in idx]); s=np.array([P[i][col] for i in idx])
        fpr,tpr,th=roc_curve(y,s,drop_intermediate=False); k=int(np.argmin(np.abs(fpr-(1-tpr)))); t=th[k]
        err={}
        for i in idx:
            r=rows[i]; key=(r['source'],r[comp+'_FAKE']); e=(P[i][col]>=t) if r[comp+'_FAKE']=='0' else (P[i][col]<t)
            err.setdefault(key,[0,0]); err[key][0]+=int(e); err[key][1]+=1
        top=sorted(err.items(), key=lambda kv:-kv[1][0])[:8]
        print(split,comp,'EER',round((fpr[k]+1-tpr[k])/2,4),'n',len(idx),'|',' '.join(f"{s}/{'F' if l=='1' else 'R'}:{a}/{b}" for (s,l),(a,b) in top),flush=True)
