"""Q: with the packaged S3 runtime modules, does a head whose val scores include exactly 1.0 make VOICE > 1 (crash)?
And can the shipped S3 tables ever produce VOICE > 1?  numpy only, <1 s."""
import sys, json, numpy as np
sys.path.insert(0, 'pkg/model/runtime')
import decision_heads as dh, component_fusion as cf
rng = np.random.default_rng(0)
val_ours = np.r_[rng.random(2700), np.ones(128)]          # R2: 128 val seed-mean scores == 1.0
val_ad = rng.random(2845)
spec = {'weights': {'ours': 1, 'ad': .5}, 'tables': {'ours': dh.quantile_table(val_ours), 'ad': dh.quantile_table(val_ad)}}
worst = cf.fuse(spec, {'ours': 1.0, 'ad': 1.0})
share = np.mean([cf.fuse(spec, {'ours': 1.0, 'ad': a}) > 1 for a in rng.random(1000)])
print('saturated: table top %.12f  fused(1,1)=%.12f  >1: %s  share of saturated-ours files >1 over random ad: %.3f' % (spec['tables']['ours'][-1], worst, worst > 1, share))
s3 = json.load(open('pkg/model/trained/component_fusion.json'))['voice']
print('S3 shipped: fused(1,1)=%.12f  <=1: %s' % (cf.fuse(s3, {'ours': 1.0, 'ad': 1.0}), cf.fuse(s3, {'ours': 1.0, 'ad': 1.0}) <= 1))
