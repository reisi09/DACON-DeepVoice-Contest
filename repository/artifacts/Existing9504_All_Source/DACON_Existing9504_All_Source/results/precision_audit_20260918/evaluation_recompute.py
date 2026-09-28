# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn"]
# ///
# How to run: .venv/bin/python results/precision_audit_20260918/evaluation_recompute.py
"""Recompute saved evaluation outcomes and check meaningful scoring contracts."""
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from graph_scoring import LABEL_COLUMNS, PREDICTION_COLUMNS, _roc_metrics, official_metrics
from eight_metrics import binary_summary, eer, threshold_at_fpr
from v2_scoring import _roc

out = Path(__file__).parent
base = ROOT / "results/eight_h100_20260917/downloaded/unpacked/output"
report = {"eight": {}, "six": {}, "contracts": {}, "input_sha256": {}}
parity_deltas = []
for name in ("E1_music", "E1_voice", "E3_music", "E3_voice", "E4", "E6_MERT", "E6_Anti", "E8"):
    path = base / f"predictions_{name}.csv"
    report["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    cohorts = {"all": rows}
    if name.startswith("E3"):
        cohorts = {codec: [r for r in rows if r["codec"] == codec] for codec in sorted({r["codec"] for r in rows})}
    if name == "E8":
        cohorts = {split: [r for r in rows if r["split"] == split] for split in ("dev", "test")}
    scores = [k for k in rows[0] if "seed" in k or k == "distance"]
    report["eight"][name] = {}
    for cohort, selected in cohorts.items():
        y = np.array([int(r["label"]) for r in selected])
        item = {"rows": len(selected), "components": len({r["component_id"] for r in selected}),
                "real": int(sum(y == 0)), "fake": int(sum(y == 1)), "scores": {}, "ensembles": {}}
        for key in scores:
            s = np.array([float(r[key]) for r in selected])
            fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
            idx = np.argmin(abs(fpr - (1-tpr)))
            expected = float((fpr[idx] + 1-tpr[idx])/2)
            graph_eer, graph_auc = _roc_metrics(y.tolist(), s.tolist())
            v2 = _roc(y.tolist(), s.tolist())
            observed = eer(y, s)["eer"]
            parity_deltas += [abs(graph_eer-expected), abs(observed-expected), abs(v2["eer"]-expected), abs(graph_auc-roc_auc_score(y,s))]
            item["scores"][key] = expected
        for arm in sorted({key.rsplit("_seed", 1)[0] for key in scores if "_seed" in key}):
            seed_keys = [k for k in scores if k.startswith(arm+"_seed")]
            ensemble = np.mean([[float(r[k]) for r in selected] for k in seed_keys], axis=0)
            item["ensembles"][arm] = {"seed_mean_eer": float(np.mean([item["scores"][k] for k in seed_keys])),
                                         "probability_ensemble_eer": eer(y, ensemble)["eer"]}
        report["eight"][name][cohort] = item
assert max(parity_deltas) < 1e-14
report["contracts"]["saved_scoring_parity"] = {"comparisons": len(parity_deltas), "max_abs_delta": max(parity_deltas)}

six = ROOT / "results/six_h100_20260917"
with (six / "probes.csv").open() as stream:
    probes = list(csv.DictReader(stream))
sixbase = six / "downloaded/unpacked/output"
for name in ("B_seed41", "hard_real_seed41", "mixture_seed41", "B_full", "M330", "selected_ensemble"):
    path = sixbase / f"predictions_{name}.json"
    saved = json.loads(path.read_text())
    report["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["six"][name] = {}
    for split in ("dev", "test"):
        selected = [r for r in probes if r["split"] == split and r["variant"] == "base"]
        labels = [{"ID":r["ID"], **{k:r[k] for k in LABEL_COLUMNS}} for r in selected]
        predictions = [{"ID":r["ID"], **dict(zip(PREDICTION_COLUMNS,saved[r["ID"]],strict=True))} for r in selected]
        result = official_metrics(labels, predictions)
        assert result["score"] is None
        report["six"][name][split] = result
for experiment in (4, 5):
    saved = json.loads((sixbase / f"experiment_{experiment}.json").read_text())
    name = "hard_real_seed41" if experiment == 4 else "mixture_seed41"
    assert report["six"][name]["dev"] == {k:v for k,v in saved["candidate"].items() if k not in ("rows", "parent_groups")}
    report["six"][name]["training"] = {k:saved["training"][k] for k in ("epochs", "optimizer_updates", "batch_size")}
    report["six"][name]["training"]["rows"] = len(saved["training"]["train_ids"])
control = report["six"]["B_seed41"]["dev"]["metrics"]
gains = {}
for name in ("hard_real_seed41", "mixture_seed41"):
    candidate = report["six"][name]["dev"]["metrics"]
    gains[name] = .45*(control["file_eer"]-candidate["file_eer"])+.27*(control["music_eer"]-candidate["music_eer"])
assert len(set(gains.values())) == 1
report["six"]["selection"] = {"gains": gains, "tuple_max_selected":max((gain,name) for name,gain in gains.items())[1],
    "actual_lock":json.loads((sixbase/"selection_locked.json").read_text())}

# Reversed presence predictions must not change ground-truth component eligibility.
selected = [r for r in probes if r["split"] == "test" and r["variant"] == "base"]
saved = json.loads((sixbase / "predictions_M330.json").read_text())
labels = [{"ID":r["ID"], **{k:r[k] for k in LABEL_COLUMNS}} for r in selected]
predictions = [{"ID":r["ID"], **dict(zip(PREDICTION_COLUMNS,saved[r["ID"]],strict=True))} for r in selected]
original = official_metrics(labels,predictions)
changed = copy.deepcopy(predictions)
for p in changed:
    p["VOICE_PRESENT_PROB"] = 1-p["VOICE_PRESENT_PROB"]
    p["MUSIC_PRESENT_PROB"] = 1-p["MUSIC_PRESENT_PROB"]
masked = official_metrics(labels, changed)
assert original["counts"] == masked["counts"]
assert all(original["metrics"][k] == masked["metrics"][k] for k in ("voice_eer","music_eer"))
report["contracts"]["ground_truth_presence_mask"] = {"passed":True,"counts":original["counts"],"unknown_counts":original["unknown_counts"]}
# A complete mixed-presence case must be eligible; one missing relevant truth blocks the composite.
synthetic = [{"ID":str(i),"FILE_FAKE":i%2,"VOICE_FAKE":i%2,"MUSIC_FAKE":i%2,"VOICE_PRESENT":int(i<4),"MUSIC_PRESENT":int(i>=2)} for i in range(6)]
perfect = [{"ID":r["ID"],**{k+"_PROB":r[k] for k in LABEL_COLUMNS}} for r in synthetic]
assert official_metrics(synthetic,perfect)["score"] == 1
unknown = copy.deepcopy(synthetic); unknown[0]["VOICE_FAKE"] = ""
partial = official_metrics(unknown,perfect)
assert partial["score"] is None and partial["unknown_counts"]["voice_eer"] == 1
report["contracts"]["composite_eligibility"] = {"complete_score":1,"missing_relevant_label_score":partial["score"]}
report["contracts"]["fractional_label_coercion_reproduced"] = binary_summary([0.5,1.5],[0.1,0.9])

with (base/"predictions_E5.csv").open() as stream:
    rows = list(csv.DictReader(stream))
e5saved=json.loads((base/"experiment_5.json").read_text())
cal=[r for r in rows if r["split"]=="calibration"]
test=[r for r in rows if r["split"]=="test"]
report["eight"]["E5"]={"split_components":{s:len({r["component_id"] for r in rows if r["split"]==s}) for s in ("train","calibration","dev","test")},"test":{}}
for key,threshold in e5saved["thresholds"].items():
    fresh=threshold_at_fpr([int(r["label"]) for r in cal],[float(r[key]) for r in cal],.05)
    assert fresh==threshold
    measured=binary_summary([int(r["label"]) for r in test],[float(r[key]) for r in test],threshold)
    assert measured==e5saved["sealed_test"][key]
    report["eight"]["E5"]["test"][key]=measured
for source in (ROOT/"graph_scoring.py", ROOT/"eight_metrics.py", ROOT/"v2_scoring.py", six/"probes.csv", base/"predictions_E5.csv", base/"experiment_5.json"):
    report["input_sha256"][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
report["scope"]="CPU rescoring only; no refit, no official hidden labels, no production changes"
(out/"evaluation_recompute.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
print(json.dumps({"parity":report["contracts"]["saved_scoring_parity"],"six_selection":report["six"]["selection"],"e5_components":report["eight"]["E5"]["split_components"]},indent=2))
