# Fresh-result descriptive analysis

This is a post-lock description of the 274 already-opened external test clips. It joins the locked manifest to G0 and AST_LAST_BLOCK prediction CSVs, preserves empty labels as masks, and does not tune, select, replay, or read DACON/leaderboard audio.

The fresh result differs by evaluation slice. On `test_generator`, G0 has lower file/music EER (0.221/0.322) than AST_LAST_BLOCK (0.283/0.395), with equal voice EER (0.141). On `test_source`, AST_LAST_BLOCK has lower file/music EER (0.102/0.173) than G0 (0.163/0.654), while its voice EER is higher (0.069 vs 0.035).

For music authenticity, the raw diagnostic excludes NSynth and reports mixtures separately. G0 gives high music-fake scores to both FMA-real and Echoes-TTA: means 0.620/0.706 in generator-held-out and 0.663/0.798 in source-held-out. AST_LAST_BLOCK separates them weakly in generator-held-out (0.210/0.435; 43/80 correct at 0.5) but more strongly in source-held-out (0.515/0.854; 16/20). The latter FMA-real slice contains only five clips, so it is descriptive rather than stable evidence.

NSynth is a material source-slice contrast: G0's music-fake mean is 0.912 on its 16 real acoustic notes, while AST_LAST_BLOCK's is 0.003. Mixtures show a different distribution and remain separate in the JSON artifact. These measurements identify source/generator variation; they do not establish a causal mechanism or justify changing the locked selection.
