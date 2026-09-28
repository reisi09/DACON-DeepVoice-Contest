# ACTUAL trained loss weights (VERIFIED: P1 rows exactly as the trainer builds them - rule L, backbone-npz filter, silent-stem
# filter from the old cache, loss_weights x duration_balance with the cached sample counts)
## voice head (DF/AD recipe), 31,126 items
FAKE: rows 20,659, effective rows (Kish) 6,891, effective lineage groups 257.8; sources mlaad 56.9%, asvspoof2021_df 20.5%,
      asvspoof2019_la 3.6%, asvspoof2021_la 2.3%; top group artist:FMA:1041 2.6%.
REAL: rows 10,467, effective rows 2,113, effective lineage groups 88.1; top groups artist:FMA:1041 7.3%, csd:en001a 3.8%.
| stratum | rows | row fake share | weighted fake share | share of FAKE weight | share of REAL weight |
|---|---|---|---|---|---|
| speech | 23,956 | 0.691 | 0.604 | 0.848 | 0.555 |
| speech over music | 1,342 | 0.856 | 0.138 | 0.006 | 0.035 |
| singing a cappella | 934 | 0.000 | 0.000 | 0.000 | 0.095 |
| singing in song | 4,894 | 0.605 | 0.317 | 0.147 | 0.316 |
| Korean (all) | 10,840 | 0.569 | 0.244 | 0.057 | - |
## music head, 12,218 items
FAKE: rows 6,638, effective rows 2,077, EFFECTIVE LINEAGE GROUPS 11.8; artist:FMA:1041 28.8% of fake weight.
REAL: rows 5,580, effective rows 857, EFFECTIVE LINEAGE GROUPS 19.0; artist:FMA:1041 22.0%; sources haim 24.8%,
      existing:FMA/FMA 12.9%, existing:constructed_music 12.9%, openmic 8.6%.
Reading (INFERRED): the weighting, not the raw counts, makes (a) singing lean "real" (41% of real weight vs 15% of fake),
(b) Korean speech lean "real" (weighted fake 0.24 vs row 0.57; Korean = 5.7% of fake weight), (c) the music head learn
from ~12-19 effective lineage groups dominated by one legacy lineage group.
