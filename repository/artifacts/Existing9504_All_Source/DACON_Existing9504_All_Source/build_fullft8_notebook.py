# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat==5.10.4"]
# ///
# Run: uv run build_fullft8_notebook.py
"""Extend the sealed seven-trial notebook with the agreed separator comparison."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
NAME = 'MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb'


def build() -> None:
    notebook = nbformat.read(ROOT / 'MERT_EAT_FullFT_H100x1_7Experiments.ipynb', as_version=4)
    notebook.cells[0].source += '''

## 추가 범위: HTDemucs 재혼합 적응

기존 01~07을 유지하고, S0 train만으로 HTDemucs 학생 모델 전체를 학습합니다.
고정 교사의 네 가지 추정 stem을 서로 다른 원본 그룹끼리 재혼합합니다.
학생 checkpoint는 S0 train 안에서 따로 떼어 놓은 원본 그룹의 pseudo-target
복원 손실로 선택합니다. S0 tune/test는 분리기 학습에 사용하지 않습니다.

08은 02·03의 tune Music EER → balanced BCE로 선택한 encoder를 같은 pretrained
초기값에서 새로 학습합니다. 기존 head와 분류 학습 조건은 유지합니다.
실행 순서: 01~03 → HTDemucs 적응 → 08 → 04~07.

이는 RemixIT에서 착안한 고정 교사 재혼합 실험이며, 교사 갱신을 포함한 원 논문
전체 재현은 아닙니다. pseudo-target 손실 감소는 실제 분리 품질 향상의 증거가 아닙니다.
음성 판별 경로는 기존 frozen HTDemucs·DF-Arena 예측을 유지하고 음악 경로만 비교합니다.
추가 seed·head 탐색·분류 loss를 분리기까지 전달하는 공동 학습은 포함하지 않습니다.
'''
    notebook.cells[1].source += '\n추가 실험 설정도 기본 False입니다. 원본 7개 노트북은 별도로 보존합니다.\n'
    notebook.cells[2].source = notebook.cells[2].source.replace(
        'MERT_EAT_FullFT_H100x1_7Experiments.ipynb', NAME).replace(
        "OUTPUT = PROJECT_ROOT / 'h100x1_fullft7_run'", "OUTPUT = PROJECT_ROOT / 'h100x1_fullft8_run'")
    notebook.cells[2].source += '\nRUN_SEPARATOR = False\nSEPARATOR_CONFIG = dict(separator_steps=400, separator_lr=1e-5, separator_seconds=5.0)\n'
    notebook.cells[2].source += '\nTRAIN_CONFIG.update(batch_size=4, accumulation=2, loader_workers=4)\n'
    notebook.cells[0].source += '\n처리량 개선: 4곡씩 2회 누적하여 optimizer update당 8곡을 유지합니다. 로더 4개는 spawn으로 실행하며 epoch마다 다시 만들어 crop seed를 반영합니다. 이전 1곡 배치와 dropout 난수·부동소수점 연산 순서까지 같지는 않습니다.\n'
    source = notebook.cells[12].source
    anchor = "    model = model.eval().requires_grad_(False).to('cuda')"
    source = source.replace(anchor, '''    if config.get('adapted_separator_path'):
        checkpoint = Path(config['adapted_separator_path'])
        if file_hash(checkpoint) != config['separator_checkpoint_sha256']:
            raise RuntimeError('Adapted separator checkpoint changed')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
''' + anchor)
    notebook.cells[12].source = source
    notebook.cells[15].source = notebook.cells[15].source.replace(
        '        self.encoder = AutoModel.from_pretrained',
        '''        # Local HF snapshots need transitive Python imports copied into the module cache.
        from transformers.dynamic_module_utils import get_cached_module_file
        for source in sorted((model_root/encoder).glob('*.py')):
            get_cached_module_file(str(model_root/encoder), source.name, local_files_only=True)
        self.encoder = AutoModel.from_pretrained''')
    source = notebook.cells[17].source
    source = source.replace('def predict(', '''def batch_loader(data: AudioBags, config, sampler=None) -> DataLoader:
    workers = config['loader_workers']
    return DataLoader(data, batch_size=config['batch_size'], sampler=sampler,
        num_workers=workers, pin_memory=True, persistent_workers=False,
        multiprocessing_context='spawn' if workers else None,
        generator=torch.Generator().manual_seed(config['seed'] + data.epoch))


def batch_loss_weight(index: int, batch_count: int, total_count: int, config) -> float:
    effective = config['batch_size'] * config['accumulation']
    group_start = (index * config['batch_size']) // effective * effective
    return batch_count / min(effective, total_count - group_start)


def predict(''')
    source = source.replace('DataLoader(data, batch_size=1, shuffle=False, num_workers=0)', 'batch_loader(data, config)')
    source = source.replace("'accumulation','audio_cache_fingerprint'", "'accumulation','batch_size','loader_workers','audio_cache_fingerprint'")
    source = source.replace("    random.seed(config['seed']);", "    save_json(folder/'execution_config.json', config)\n    random.seed(config['seed']);")
    source = source.replace("math.ceil(len(train)/config['accumulation'])", "math.ceil(len(train)/(config['batch_size']*config['accumulation']))")
    source = source.replace('DataLoader(train,batch_size=1,sampler=sampler,num_workers=0)', 'batch_loader(train, config, sampler)')
    source = source.replace("            group_start = index//config['accumulation']*config['accumulation']\n            divisor = min(config['accumulation'],len(loader)-group_start)", "            weight = batch_loss_weight(index, len(bags), len(train), config)")
    source = source.replace('(loss/divisor).backward()', '(loss*weight).backward()')
    source = source.replace('loss_sum += float(loss.detach())', 'loss_sum += float(loss.detach()) * len(bags)')
    source = source.replace("'train_loss':loss_sum/len(loader)", "'train_loss':loss_sum/len(train)")
    source = source.replace('bags.cuda()', 'bags.cuda(non_blocking=True)').replace('labels.cuda()', 'labels.cuda(non_blocking=True)')
    notebook.cells[17].source = source
    source = notebook.cells[18].source.replace(
        "                trial = next(t for t in TRIALS if t.name==os.environ['FULLFT_TRIAL'])",
        "                trial = Trial(**config['extra_trial']) if config.get('extra_trial') else next(t for t in TRIALS if t.name==os.environ['FULLFT_TRIAL'])")
    source = source.replace("            case 'train':", "            case 'separator':\n                separator_module.adapt_separator(splits['S0']['train'], config)\n            case 'train':")
    notebook.cells[18].source = source
    source = notebook.cells[23].source.replace(
        "scope='7 configurations, one seed, no separator training, no official total'",
        "scope='8 classifiers plus fixed-teacher train-only separator adaptation; one seed; no official total'")
    source = source.replace("        check_deadline(config)", "        config.update(SEPARATOR_CONFIG)\n        config['separator_code_sha256'] = SEPARATOR_CODE_SHA256\n        check_deadline(config)", 1)
    source = source.replace(
        "            order=range(len(TRIALS))\n            run_jobs([('train',TRIALS[i].name) for i in order],config)",
        '''            run_jobs([('train', trial.name) for trial in TRIALS[:3]], config)
            if RUN_SEPARATOR:
                candidates = []
                for trial in TRIALS[1:3]:
                    selection = json.loads((OUTPUT/trial.name/'selection.json').read_text())
                    candidates.append((selection['metrics']['eer'], selection['metrics']['balanced_bce'], trial.name, trial.encoder))
                selected = min(candidates)
                save_json(OUTPUT/'separator_encoder_selection.json', {
                    'criterion':'S0 tune EER, balanced BCE, trial name', 'test_used':False,
                    'selected_trial':selected[2], 'encoder':selected[3], 'candidates':candidates})
                run_jobs([('separator','S0_train')], config)
                checkpoint = OUTPUT/'separator_adaptation'/'best.pt'
                adapted_output = OUTPUT/'adapted'
                adapted_output.mkdir(exist_ok=True)
                adapted = dict(config, output=str(adapted_output.resolve()), splits={'S0':SPLITS['S0']},
                    adapted_separator_path=str(checkpoint.resolve()), separator_checkpoint_sha256=file_hash(checkpoint),
                    audio_recipe='adapted_htdemucs_native_to16k_5s_pool8_v1',
                    extra_trial=dict(name='08_S0_adapted_fullft',split='S0',encoder=selected[3],frozen=False))
                (adapted_output/'worker.py').write_text(worker_source)
                save_json(adapted_output/'runtime.json', adapted)
                run_jobs([('prepare','all')], adapted)
                s0_rows = {r['ID']:r for part in SPLITS['S0'].values() for r in part}
                for row in s0_rows.values():
                    validate_audio_cache(row, adapted)
                adapted['audio_cache_fingerprint'] = stable_hash([
                    file_hash(adapted_output/'audio_cache'/f'{identifier}.json') for identifier in sorted(s0_rows)])
                save_json(adapted_output/'runtime.json', adapted)
                run_jobs([('train','08_S0_adapted_fullft')], adapted)
            run_jobs([('train', trial.name) for trial in TRIALS[3:]], config)''')
    notebook.cells[23].source = source
    notebook.cells[25].source = notebook.cells[25].source.replace(
        'for row in summary:', '''extra_folder = OUTPUT/'adapted'/'08_S0_adapted_fullft'
if (extra_folder/'complete.json').exists():
    extra_report = json.loads((extra_folder/'evaluation.json').read_text())
    extra_selection = json.loads((extra_folder/'selection.json').read_text())
    summary.append(dict(trial='08_S0_adapted_fullft',status='complete',epoch=extra_selection['epoch'],
        tune_eer=extra_selection['metrics']['eer'],test_eer=extra_report['overall']['eer']))
else:
    summary.append(dict(trial='08_S0_adapted_fullft',status='미완료/미실행'))
for row in summary:''').replace("'/ 7'", "'/ 8'")
    module = (ROOT / 'fullft_separator.py').read_text()
    import hashlib
    notebook.cells.insert(18, nbformat.v4.new_code_cell(
        "import types, sys\nseparator_module = types.ModuleType('fullft_separator')\n"
        "sys.modules['fullft_separator'] = separator_module\n"
        f"SEPARATOR_CODE_SHA256 = {hashlib.sha256(module.encode()).hexdigest()!r}\n"
        f"exec(compile({module!r}, 'fullft_separator.py', 'exec', dont_inherit=True), separator_module.__dict__)\n",
        metadata={'tags':['worker']}))
    notebook.cells[-1].source += '\n재혼합 방법 참고: https://arxiv.org/abs/2202.08862 . 추가 코드의 GPU 학습 결과는 실행 후 별도로 확인합니다.\n'
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / NAME)


if __name__ == '__main__':
    build()
