# Workflow 2 (2026-09-27 afternoon): deeper heads, data-diversity verification, new frozen backbones, add-on stacking

Reference baseline: S3 (LB ADS 0.7853015873). Leading candidate S4 = S3 with R2's val-chosen voice heads
(DF-Arena fc5 inputs -> 2-layer MLP, AntiDeepfake embeddings -> ResMLP; 150 epochs / patience 10), held-out ADS
0.87267 -> 0.88643 (+0.0138, group CI 0.0092-0.0174). User directives: GPU budget = time only (money no concern),
multi-GPU allowed, "consider deeper networks, verify data diversity, do everything". Deadline 2026-09-29 09:59:59 KST;
slots: 3 on 09-28, 3 on 09-29 before 10:00 (reset time unverified).
Folders: depth/, diversity/, backbones/, stacking/ (one owner each).
