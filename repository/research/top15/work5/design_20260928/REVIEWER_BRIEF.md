# Independent review brief (GPT-6-Astra, reasoning effort ultra, read-only)

You are the independent reviewer of a competition model (DACON 236749, audio deepfake detection). You work alone and form
your conclusions from the source evidence in THIS folder only (review_packet_v1). Do not read files outside this folder.
Do not run anything that uses a network, GPU, or paid service; you may read files and run local read-only analysis
(e.g. python/pandas on the CSV/JSON files here).

Start with MANIFEST.md, then inspect the code (code/, runtime_pkg/), receipts (receipts/), results (results/), data
manifests (data/) and history (history/). The coordinator's recommendations are intentionally not in this packet.

Deliver, in Korean, with every claim labelled VERIFIED / REPORTED / HYPOTHESIS / ESTIMATE / UNKNOWN / NOT RUN and
file:line citations:
1. Leakage and adaptive-overfitting findings (MANIFEST section 6, question 1), ranked by likely impact.
2. Evaluation validity (question 2): what each local set can and cannot support; single-class and all-positive slices;
   score-scale comparability.
3. Implementation and integration findings (question 3), each as: evidence -> affected path -> expected -> observed ->
   likely impact -> proposed correction -> the check that would verify the correction. Separate confirmed defects from
   risks and hypotheses.
4. Improvement options (question 4): without new data first, then with data; the evidence behind each; overfitting risk.
5. Independent final evaluation (question 5): whether any set here qualifies, and what would be required.
6. Coverage: which files/paths you reviewed fully, partially, or not at all.
Keep it concise and specific; no generic advice.
