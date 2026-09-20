"""Reuse the verified feature-loading and evaluation notebook for compact tuning."""

from pathlib import Path

import nbformat as nb


def main() -> None:
    notebook = nb.read('Temporal_H100_Simplification.ipynb', as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.cells[0].source = '''# 64채널·Conv1D 3층 하이퍼파라미터 튜닝

실행 전 노트북입니다. GELU, residual 연결, max pooling과 MERT 입력은 유지합니다.
입력 투영 Conv1D(768→64, kernel 1) 뒤에 시간축 Conv1D(kernel 3) 3층을 둡니다.
학습률·weight decay·dropout·batch size·crop·특징 마스킹·NSynth 가중치를 탐색합니다.
24개 설정을 seed 41로 비교하고 상위 4개를 seed 42·43에서도 학습합니다.
튜닝 Music EER 3개 seed 평균→표준편차→BCE 순으로 최종 후보를 선택합니다.
최대 300 epoch, patience 20이며 최저 튜닝 BCE 가중치를 복원합니다.

선택을 고정한 뒤 기존 128채널·2층, 튜닝 전 64채널·3층, 선택한 64채널·3층을 비교합니다.
기존 공개 평가 자료를 재사용합니다. 전체 공식 점수는 정답 누락으로 계산할 수 없습니다.
파일·음악 가중 부분합은 공식 총점이 아니며, 평가 결과로 설정이나 seed를 다시 고르지 않습니다.
'''
    notebook.cells[2].source = notebook.cells[2].source.replace(
        'from temporal_augmentation_experiment import run_augmentation_search',
        'from temporal_tuning import run_compact_search')
    notebook.cells[5].source = '''## 3. 튜닝 데이터로 후보 선택
탐색 설정과 shortlist를 저장한 뒤 seed 반복을 진행합니다.
원본 캐시 불변성과 기존 128채널 모델의 학습 곡선 재현을 확인합니다.'''
    notebook.cells[6].source = '''search = run_compact_search(
    SPLIT['train'], SPLIT['tune'], records, masked, OUT, DEADLINE)
assert CACHE_BEFORE == (feature_hash(records), feature_hash(masked)), 'Feature cache mutated'
write_json(OUT/'cache_integrity.json', dict(unchanged=True, original_sha256=CACHE_BEFORE))
reference = json.loads((ROOT/'reference_simple.json').read_text())
for seed in (41,42,43):
    current = json.loads((OUT/f'S128_seed{seed}'/'training.json').read_text())
    previous = reference[str(seed)]
    for field in ('train_losses', 'validation_losses'):
        np.testing.assert_allclose(current[field], previous[field], atol=1e-6, rtol=0)
    assert current['selected_epoch'] == previous['selected_epoch']
write_json(OUT/'prior_reproduction.json', dict(passed=True, tolerance=1e-6, seeds=[41,42,43]))
selection = json.loads((OUT/'selection.json').read_text())
lines = ['| 설정 | 3-seed 튜닝 EER | 표준편차 | BCE |', '|---|---:|---:|---:|']
for r in selection['summaries']:
    lines.append(f"| {r['name']} | {r['music_eer_mean']:.4f} | {r['music_eer_std']:.4f} | {r['balanced_bce_mean']:.4f} |")
display(Markdown('\\n'.join(lines)))
print('Selected using tune only:', selection['winner']['name'], selection['winner']['recipe'])
'''
    notebook.cells[7].source = '''## 4. 고정한 후보와 대조군 평가
평가 190개 중 음악 EER 대상은 150개입니다. 음성·존재 확률과 파일 max 결합식을 유지합니다.
가중 부분합은 0.45×(1−File EER)+0.27×(1−Music EER)이며 전체 공식 점수가 아닙니다.'''
    notebook.cells[8].source = notebook.cells[8].source.replace(
        "for s in selection['summaries']:",
        "for s in selection['summaries']:\n    if s['name'] not in selection['comparison_groups']:\n        continue")
    notebook.cells[9].source = '''## 5. 평가 결과와 학습 곡선
오차 막대는 seed 3개의 모집단 표준편차이며 신뢰구간이 아닙니다.
튜닝 전후를 구분하고 모든 seed의 튜닝 BCE를 표시합니다.'''
    notebook.cells[10].source = '''import matplotlib.pyplot as plt
from IPython.display import Image
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
winner = selection['winner']['name']
settings = list(dict.fromkeys(['S128','C00',winner]))
labels = {'S128':'128ch / 2 conv', 'C00':'64ch / 3 conv (untuned)'}
if winner != 'C00':
    labels[winner] = '64ch / 3 conv (tuned)'
colors = ['#2463a5','#888888','#b16b1b'][:len(settings)]
lookup = {r['name']:r for r in test_summaries}
fig, axes = plt.subplots(2,3,figsize=(17,9),layout='constrained')
for ax,metric,title in zip(axes[0],['music_eer','file_eer','file_music_partial'],
                         ['Historical music EER (lower)','Historical file EER (lower)','File/music partial score (higher)']):
    values = [lookup[s][metric+'_mean'] for s in settings]
    spread = [lookup[s][metric+'_std'] for s in settings]
    ax.bar(range(len(settings)), values, yerr=spread, capsize=5, color=colors)
    ax.set(xticks=range(len(settings)),xticklabels=[labels[s].replace(' / ','\\n') for s in settings],
           title=title,ylim=(0,1))
    for i,value in enumerate(values):
        ax.text(i,value+spread[i]+.02,f'{value:.4f}',ha='center')
    ax.axhline(reports[1][metric],color='#555555',linestyle='--',label='Linear control')
    ax.legend(fontsize=9)
for ax,seed in zip(axes[1],(41,42,43)):
    for setting,color in zip(settings,colors):
        trace = json.loads((OUT/f'{setting}_seed{seed}'/'training.json').read_text())
        y = trace['validation_losses']
        ax.plot(np.arange(1,len(y)+1),y,color=color,label=labels[setting])
        ax.scatter([trace['selected_epoch']],[trace['best_validation_loss']],color=color)
    ax.set(xlabel='Epoch',ylabel='Tune balanced BCE',title=f'Seed {seed}')
    ax.legend(fontsize=9)
fig.savefig(OUT/'comparison.png',dpi=150)
plt.close(fig)
display(Image(filename=str(OUT/'comparison.png')))
display(Markdown('모든 탐색 가중치·곡선·튜닝 예측을 저장했습니다. 평가 결과로 후보를 다시 선택하지 않았습니다.'))
'''
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook-cell>', 'exec')
    nb.validate(notebook)
    nb.write(notebook, Path('Temporal_H100_Compact_Tuning.ipynb'))


if __name__ == '__main__':
    main()
