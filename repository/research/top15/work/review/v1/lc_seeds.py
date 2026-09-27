"""Q: is the learning-curve reading 'music held-out flat beyond 50% of train groups, voice still improving'
(_workspace/audit_20260927/learning_curve.json, ONE seed, ONE group order) robust to the model seed and to the group draw?
Reuses learning_curve.run/subset unchanged; adds model seeds 42/43 and a second nested group order (subset seed 1).
  python lc_seeds.py"""
import importlib.util, json, sys, time
from pathlib import Path
import torch
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('lc', ROOT / '_workspace/audit_20260927/learning_curve.py')
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
T = L.T
torch.set_num_threads(4)
t0 = time.time()
items = T.read_items(L.BANK / 'bank_trainval.csv', L.CACHE / 'cache_tv')
test_items = T.read_items(L.BANK / 'bank_test_as_val.csv', L.CACHE / 'cache_test')
samples = {k: v['samples'] for c in ('cache_tv', 'cache_test')
           for k, v in json.loads((L.CACHE / c / 'index.json').read_text())['items'].items()}
init = torch.load(L.CACHE / 'cache_tv' / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
out = []
for route in ('music', 'voice'):
    training = [x for x in items if x.row['route'] == route and x.row['split'] == 'train']
    validation = [x for x in items if x.row['route'] == route and x.row['split'] == 'val']
    test = [x for x in test_items if x.row['route'] == route]
    for subset_seed, model_seeds in ((20260927, (42, 43)), (1, (41,))):
        for fraction in (.5, 1.):
            if subset_seed == 1 and fraction == 1.:
                continue   # full set is the same for any group order
            part, ng = L.subset(training, fraction, seed=subset_seed)
            for s in model_seeds:
                r = L.run(route, part, validation, test, samples, init['state_dict'], init['fake_label_index'], seed=s)
                r = {k: v for k, v in r.items() if k != 'val_history'} | dict(route=route, fraction=fraction, groups=ng,
                                                                          subset_seed=subset_seed, model_seed=s,
                                                                          minutes=round((time.time() - t0) / 60, 1))
                out.append(r)
                print(json.dumps(r), flush=True)
                (Path(__file__).resolve().parent / 'lc_seeds.json').write_text(json.dumps(out, indent=1))
