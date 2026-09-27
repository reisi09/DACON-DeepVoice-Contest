# Question: is the 74% backbone_cache coverage (S3 heads + DB-protocol held-out ADS 0.8644->0.8727) balanced across
# split / route / FILE label / source, or is the S3 comparison on a biased subset? Read-only; CSV + directory listing.
import csv, os, collections
B='results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
have={f[:-4] for f in os.listdir('_workspace/backbone_cache') if f.endswith('.npz')}
rows=[r for f in ('bank_trainval.csv','bank_test.csv') for r in csv.DictReader(open(f'{B}/{f}'))]
print('rows',len(rows),'cached',sum(r['ID'] in have for r in rows))
def tab(key):
    c=collections.defaultdict(lambda:[0,0])
    for r in rows:
        k=key(r); c[k][0]+=1; c[k][1]+=r['ID'] in have
    return c
for name,key in (('split',lambda r:r['split']),('split/route',lambda r:(r['split'],r['route'])),
                 ('split/FILE_FAKE',lambda r:(r['split'],r['FILE_FAKE'] if r['FILE_FAKE_mask']=='1' else 'unk'))):
    print('==',name)
    for k,(n,h) in sorted(tab(key).items()): print(' ',k,n,h,f'{h/n:.2f}')
print('== test sources with coverage <0.5 or >0.95 (n>=50)')
for k,(n,h) in sorted(tab(lambda r:(r['split'],r['source'])).items(), key=lambda kv:kv[1][1]/kv[1][0]):
    if n>=50 and k[0] in('test','val') and (h/n<0.5 or h/n>0.95): print(' ',k,n,h,f'{h/n:.2f}')
