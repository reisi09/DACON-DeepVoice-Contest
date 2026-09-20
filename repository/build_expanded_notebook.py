# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat"]
# ///
# Run: .venv/bin/python build_expanded_notebook.py
"""Build the source-embedded H100 notebook; local inspection never starts compute."""
from __future__ import annotations

import nbformat as nb

from build_redesign_notebook import ROOT, code, source_payload


def build() -> nb.NotebookNode:
    payload, digest, _ = source_payload(('expanded_experiment.py', 'localization_features.py', 'localization_export.py'))
    cells = [nb.v4.new_markdown_cell('''# 신규 데이터 · H100 실험

새 그룹 분할에서 **기존 831개 기준선 → 데이터 확장 → 시간별 MIL → AST(ViT 계열)**를 비교한다. H100 1대, 시작부터 최대 4시간, seed 41·42·43이다. 최종 결과는 아래 실행 출력에서 확인한다.

로컬 `Run All`은 분할과 입력 기록을 확인한다. 실제 GPU 실행은 `expanded_h100_ops.py launch --release`가 검증된 업로드와 절대 종료 시각을 전달한다. 가중치·음원은 노트북에 중복 포함하지 않으며 `input.zip`, `audio_upload.json`, 원격 shard와 기존 모델 archive의 해시로 연결한다. 실험은 자동 제출하지 않는다.'''),
             nb.v4.new_markdown_cell('## 1. 실행 조건'), code('''
        import os, sys, json, csv, time
        from pathlib import Path
        from collections import Counter
        EXECUTE_GPU = os.environ.get('EXPANDED_EXECUTE_GPU') == '1'
        JOB = Path(os.environ.get('EXPANDED_JOB_ROOT', '_workspace/expanded_h100_20260918')).resolve()
        INPUT = JOB / 'input' if EXECUTE_GPU else JOB
        OUTPUT = JOB / 'output'
        OUTPUT.mkdir(parents=True, exist_ok=True)
        ASSETS = Path('/root/redesign_h100_20260917/assets')
        DEADLINE = float(os.environ['DEADLINE_EPOCH']) - 600 if EXECUTE_GPU else None
        print({'gpu_execution': EXECUTE_GPU, 'device': 'H100 x1', 'effective_batch': 256,
               'updates': 96, 'seeds': [41,42,43], 'deadline_epoch': DEADLINE})
        '''), nb.v4.new_markdown_cell('## 2. 고정 소스 복원'), code(f'''
        import base64, hashlib, io, zipfile
        payload = base64.b64decode({payload!r})
        assert hashlib.sha256(payload).hexdigest() == {digest!r}
        SOURCE = JOB / 'notebook_source'
        SOURCE.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as packed:
            for name in packed.namelist():
                assert name == Path(name).name and name.endswith('.py')
                (SOURCE / name).write_bytes(packed.read(name))
        sys.path.insert(0, str(SOURCE))
        print('Source SHA256:', {digest!r})
        '''), nb.v4.new_markdown_cell('''## 3. 새 분할과 라벨

아티스트·음원 원본·생성 프롬프트·변환 입력 관계를 묶어 분리했다. Mustango와 해당 프롬프트의 다른 생성기 출력은 학습에서 제외했다. MusicGen의 미공개 프롬프트, 알려지지 않은 별칭·유사 음원, 사전학습 중복은 미해결이다.

기존 학습 959개(원본 831개 + partial/sham 128개)와 새 학습 2,700개를 합친다. 새 dev 240개만 선택에 사용한다. calibration 160개, test 240개, Mustango 평가 200개는 선택에서 제외한다. ASV 128개는 기존 음성 분기 진단이며 독립 사전학습 평가가 아니다.'''), code('''
        from eight_contract import read_manifest, sha256_file
        rows = read_manifest(INPUT / 'dataset/manifest.csv')
        summary_path = INPUT / 'split_summary.json'
        if not summary_path.exists():
            summary_path = Path('results/expanded_h100_20260918/split_summary.json')
        summary = json.loads(summary_path.read_text())
        assert len(rows) == summary['total_rows']
        print('Manifest SHA256:', sha256_file(INPUT / 'dataset/manifest.csv'))
        print('Split counts:', dict(Counter(r['split'] for r in rows)))
        print('Unknown voice labels are preserved:', sum(r['VOICE_FAKE'] == '' for r in rows))
        '''), code('''
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.figsize': (10,4), 'font.size': 10})
        splits = ['train','dev','calibration','test','confirmation_external','confirmation_presence']
        real = [sum(r['split']==s and r['MUSIC_FAKE']=='0' for r in rows) for s in splits]
        fake = [sum(r['split']==s and r['MUSIC_FAKE']=='1' for r in rows) for s in splits]
        unknown = [sum(r['split']==s and r['MUSIC_FAKE']=='' for r in rows) for s in splits]
        x = list(range(len(splits)))
        fig, ax = plt.subplots()
        ax.bar(x, real, label='Real music', color='#2463a5')
        ax.bar(x, fake, bottom=real, label='Synthetic music', color='#e17c35')
        ax.bar(x, unknown, bottom=[a+b for a,b in zip(real,fake)], label='Unknown music truth', color='#909090')
        ax.set_xticks(x, ['Train','Dev','Calibration','Test','Mustango holdout','Speech diagnostic'])
        ax.set_ylabel('Audio files'); ax.set_title('Frozen split: known music labels and unknowns')
        totals = [a+b+c for a,b,c in zip(real,fake,unknown)]
        ax.set_ylim(0, max(totals)*1.13)
        for i, total in enumerate(totals):
            ax.text(i, total+max(totals)*.015, str(total), ha='center')
        ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT/'split_counts.png', dpi=160); plt.show()
        '''), nb.v4.new_markdown_cell('''## 4. H100 처리량 측정

대표 자료의 Demucs·MERT·AST 처리와 실제 AST optimizer step을 측정한다. 배치 32·64·128·256 중 메모리 여유와 처리량을 만족하는 값을 선택한다. 전체 작업 예상 시간이 남은 예산을 넘으면 중단한다. 추정은 완료 보장이 아니다.'''), code('''
        if EXECUTE_GPU:
            from expanded_experiment import canary, command, feature_arguments
            capacity = canary(INPUT, JOB/'cache', OUTPUT, ASSETS, DEADLINE)
            print({k:v for k,v in capacity.items() if k != 'ast_trials'})
        else:
            print('Local inspection only. No GPU or model training started by this execution.')
        '''), nb.v4.new_markdown_cell('## 5. 같은 전처리로 특징 추출'), code('''
        if EXECUTE_GPU:
            command(feature_arguments(INPUT/'dataset', JOB/'cache', OUTPUT, ASSETS,
                                      INPUT/'ast_assets', DEADLINE), OUTPUT/'features.log', DEADLINE)
            features = json.loads((OUTPUT/'feature_cache_receipt.json').read_text())
            print({'files':len(features['items']), 'windows':features['windows'],
                   'elapsed_seconds':features['elapsed_seconds'], 'stage_seconds':features['stage_seconds']})
        '''), nb.v4.new_markdown_cell('''## 6. 세 seed의 네 조건 비교

유효 배치 256·96 updates를 사용한다. 기준선과 확대 평균 MLP는 파일 BCE, MIL/AST는 파일 BCE + 알려진 음악 구간 BCE 0.25다. AST는 마지막 block·정규화·head만 학습한다. dev EER, 동률이면 dev BCE로 checkpoint를 고른 뒤 전체 후보를 잠근다. 이후에만 최종 평가를 연다.'''), code('''
        if EXECUTE_GPU:
            command([str(SOURCE/'localization_train.py'), '--manifest', str(INPUT/'dataset/manifest.csv'),
                     '--cache-dir',str(JOB/'cache'), '--ast-assets',str(INPUT/'ast_assets'),
                     '--output-dir',str(OUTPUT/'train'), '--deadline-epoch',str(DEADLINE),
                     '--effective-batch','256','--steps','96','--eval-every','24','--no-evaluate-training'],
                    OUTPUT/'training.log', DEADLINE)
            print(json.loads((OUTPUT/'train/candidate_lock.json').read_text()))
        '''), nb.v4.new_markdown_cell('''## 7. 결과

Music EER는 낮을수록 좋다. 같은 평가 집합의 확률 평균 결과끼리 비교하며 seed EER 평균과 구분한다. 데이터가 바뀌었으므로 이전 실험의 EER와 바로 우열을 판단하지 않는다. 전체 공식 점수에 필요한 라벨이 없는 집합은 `null`이며, 이 결과는 DACON 제출 점수가 아니다.'''), code('''
        result_path = OUTPUT/'train/results.json'
        if result_path.exists():
            results = json.loads(result_path.read_text())
            print('Status:', results['status'])
            print('Dev-only selection:', results.get('winner_by_dev_only'))
            if results['status'] == 'completed':
                arms = ['original831_mean','mean','mil','ast']
                for split in ['dev','test','confirmation_external']:
                    values = [results['metrics'][arm][split]['music_eer'] for arm in arms]
                    print(split, dict(zip(arms, values)))
                fig, axes = plt.subplots(1, 2, figsize=(11,4), sharey=True)
                for ax, split in zip(axes, ['test','confirmation_external']):
                    values = [results['metrics'][arm][split]['music_eer'] for arm in arms]
                    ax.bar(range(4), values, color=['#999999','#2463a5','#288a6d','#e17c35'])
                    ax.set_xticks(range(4), ['Old data','Expanded','MIL','AST'])
                    ax.set_ylim(0, 1.12*max(results['metrics'][a][s]['music_eer'] for a in arms for s in ['test','confirmation_external'])); ax.set_title(split); ax.set_ylabel('Music EER')
                fig.tight_layout(); fig.savefig(OUTPUT/'music_eer.png', dpi=160); plt.show()
        else:
            print('No training results yet. Split verification does not establish model quality.')
        ''')]
    notebook = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name':'expanded-h100','display_name':'Python (expanded H100)','language':'python'}})
    nb.validate(notebook)
    return notebook


if __name__ == '__main__':
    nb.write(build(), ROOT / 'DACON_Expanded_H100_Experiments.ipynb')
