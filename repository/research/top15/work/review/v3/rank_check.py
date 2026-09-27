"""Q: where does S3 (ADS 0.7853015873, CPS 0.9976771958) rank on the 13:33 KST snapshot, and what ADS gain reaches #15?"""
import json
rows = json.load(open('../../lane1/leaderboard_20260927T1333KST.json'))
others = sorted([float(r['score']) for r in rows if r['team'] != 'killerpjw'], reverse=True)
cps = 0.9976771958; rank = lambda t: 1 + sum(s > t for s in others)
s3 = 0.9 * 0.7853015873 + 0.1 * cps
print('S3 total %.6f rank %d of %d; #15 %.7f needs +%.5f ADS' % (s3, rank(s3), len(others) + 1, others[14], (others[14] - s3) / .9))
for d in (0.004, 0.009, 0.014, 0.025, 0.045, 0.075): print('S3 +%.3f ADS -> rank %d' % (d, rank(s3 + .9 * d)))
