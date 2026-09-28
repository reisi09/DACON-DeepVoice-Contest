"""Build the approved paired architecture experiment from the existing notebook."""

from pathlib import Path

import nbformat as nb


def main() -> None:
    notebook = nb.read('Temporal_H100_Augmentation.ipynb', as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.cells[0].source = '''# H100 음악 분류기 단순화 비교

실행 전 노트북입니다. 기존 6층 Conv1D + gated attention/max와 2층 Conv1D + max를 비교합니다.
입력 MERT, LayerNorm, 128차원 투영, 데이터 분할, crop 70%, batch 128,
AdamW(lr 3e-4, weight decay 0.01), Dropout 0.2, clipping 1.0을 유지합니다.
최대 300 epoch, patience 20, min_delta 1e-4이며 최저 튜닝 BCE 가중치를 복원합니다.

두 구조 × seed 41·42·43을 모두 보고합니다. 기존 공개 평가 자료를 재사용하며
Music EER·File EER을 함께 비교합니다. 전체 공식 점수는 정답 누락으로 계산할 수 없습니다.
파일·음악 가중 부분합은 공식 총점이 아니며, 평가 결과로 seed를 선택하지 않습니다.
튜닝 Music EER로 기록한 선택은 기존 절차 재현용이며 최종 제출 모델 선정은 아닙니다.
'''
    notebook.cells[5].source = '''## 3. 두 구조 × 세 seed 학습
학습 원본·crop·튜닝 입력을 고정합니다. 기존 구조의 학습 곡선을 이전 H100 결과와 대조합니다.
구조를 여러 부분에서 함께 줄이므로 개별 삭제 항목의 효과를 분리한 실험은 아닙니다.'''
    notebook.cells[6].source = '''search = run_augmentation_search(
    SPLIT['train'], SPLIT['tune'], records, masked, OUT, DEADLINE,
    batch_size=BATCH_SIZE, compare_architectures=True)
assert CACHE_BEFORE == (feature_hash(records), feature_hash(masked)), 'Feature cache mutated'
write_json(OUT/'cache_integrity.json', dict(unchanged=True, original_sha256=CACHE_BEFORE))
reference = json.loads((ROOT/'reference_deep_crop.json').read_text())
for seed in (41,42,43):
    current = json.loads((OUT/f'D_deep_crop_seed{seed}'/'training.json').read_text())
    previous = reference[str(seed)]
    for field in ('train_losses', 'validation_losses'):
        np.testing.assert_allclose(current[field], previous[field], atol=1e-6, rtol=0)
    assert current['selected_epoch'] == previous['selected_epoch']
write_json(OUT/'deep_reproduction.json', dict(passed=True, tolerance=1e-6, seeds=[41,42,43]))
selection = json.loads((OUT/'selection.json').read_text())
lines = ['| 모델 | 파라미터 | 튜닝 EER | BCE | best epoch | 학습 epoch | 초 |',
         '|---|---:|---:|---:|---:|---:|---:|']
for r in search:
    lines.append(f"| {r['name']} | {r['parameters']:,} | {r['music_eer']:.4f} | {r['balanced_bce']:.4f} | {r['selected_epoch']} | {r['epochs_completed']} | {r['seconds']:.1f} |")
display(Markdown('\\n'.join(lines)))
'''
    notebook.cells[7].source = '''## 4. 고정한 여섯 모델의 기존 평가 자료 비교
평가 190개 중 음악 EER 대상은 150개입니다. 음성·존재 확률과 파일 max 결합식을 유지합니다.
가중 부분합은 0.45×(1−File EER)+0.27×(1−Music EER)이며 높을수록 좋습니다.
이 값은 전체 공식 점수도, 새 독립 평가 결과도 아닙니다.'''
    notebook.cells[8].source = '''reports = evaluate_locked_test(SPLIT['test'], records, BASELINE, OUT)
assert CACHE_BEFORE == (feature_hash(records), feature_hash(masked))
for r in reports:
    r['file_eer'] = r['metrics']['metrics']['file_eer']
    r['file_music_partial'] = .45*(1-r['file_eer']) + .27*(1-r['music_eer'])
test_summaries = []
for s in selection['summaries']:
    matched = [r for r in reports if r['name'] in s['candidates']]
    summary = dict(name=s['name'])
    for metric in ('music_eer','file_eer','file_music_partial','balanced_bce'):
        values = [r[metric] for r in matched]
        summary[metric+'_mean'] = float(np.mean(values))
        summary[metric+'_std'] = float(np.std(values))
    test_summaries.append(summary)
write_json(OUT/'comparison_summary.json', dict(tune=selection['summaries'],
    historical_test=test_summaries, per_seed=reports,
    partial_score_definition='.45*(1-file_eer)+.27*(1-music_eer); not full official score',
    no_seed_selected_using_test=True))
lines = ['| 모델 | Music EER ↓ | File EER ↓ | 파일·음악 가중 부분합 ↑ |',
         '|---|---:|---:|---:|']
for r in reports:
    lines.append(f"| {r['name']} | {r['music_eer']:.4f} | {r['file_eer']:.4f} | {r['file_music_partial']:.4f} |")
display(Markdown('\\n'.join(lines)))
lines = ['| 구조 | 평균 Music EER | 평균 File EER | 평균 가중 부분합 |', '|---|---:|---:|---:|']
for r in test_summaries:
    lines.append(f"| {r['name']} | {r['music_eer_mean']:.4f} | {r['file_eer_mean']:.4f} | {r['file_music_partial_mean']:.4f} |")
display(Markdown('\\n'.join(lines)))
'''
    notebook.cells[9].source = '''## 5. 구조별 결과와 학습 곡선
오차 막대는 seed 3개의 모집단 표준편차이며 신뢰구간이 아닙니다.
각 seed의 튜닝 손실을 모두 표시합니다. 학습·튜닝 BCE의 가중치가 달라 절대값을 직접 비교하지 않습니다.'''
    notebook.cells[10].source = '''import matplotlib.pyplot as plt
from IPython.display import Image
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
colors = ['#2463a5','#b16b1b']
fig, axes = plt.subplots(2,3,figsize=(16,9),layout='constrained')
for ax,metric,title in zip(axes[0],['music_eer','file_eer','file_music_partial'],
                         ['Historical music EER (lower)','Historical file EER (lower)','File/music partial score (higher)']):
    values = [r[metric+'_mean'] for r in test_summaries]
    spread = [r[metric+'_std'] for r in test_summaries]
    ax.bar([0,1], values, yerr=spread, capsize=5, color=colors)
    ax.set(xticks=[0,1],xticklabels=['6 conv + gates','2 conv + max'],title=title,ylim=(0,1))
    for i,value in enumerate(values):
        ax.text(i,value+spread[i]+.02,f'{value:.4f}',ha='center')
    ax.axhline(reports[1][metric],color='#555555',linestyle='--',label='Linear control')
    ax.legend(fontsize=9)
for ax,seed in zip(axes[1],(41,42,43)):
    for setting,color in zip(('D_deep_crop','S_simple_crop'),colors):
        trace = json.loads((OUT/f'{setting}_seed{seed}'/'training.json').read_text())
        y = trace['validation_losses']
        ax.plot(np.arange(1,len(y)+1),y,color=color,label=setting)
        ax.scatter([trace['selected_epoch']],[trace['best_validation_loss']],color=color)
    ax.set(xlabel='Epoch',ylabel='Tune balanced BCE',title=f'Seed {seed}')
    ax.legend(fontsize=9)
fig.savefig(OUT/'comparison.png',dpi=150)
plt.close(fig)
display(Image(filename=str(OUT/'comparison.png')))
display(Markdown('여섯 모델의 가중치·곡선·예측을 저장했습니다. 기존 평가 결과로 seed를 선택하지 않았습니다.'))
'''
    nb.validate(notebook)
    nb.write(notebook, Path('Temporal_H100_Simplification.ipynb'))


if __name__ == '__main__':
    main()
