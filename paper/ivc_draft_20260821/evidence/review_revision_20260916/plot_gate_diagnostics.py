"""Render gate diagnostics from compact frozen evidence, without sample selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / 'figures/journal'
OUT.mkdir(parents=True, exist_ok=True)
conditions = ['clean', 'left_zero', 'right_zero', 'switch_left_to_right', 'switch_right_to_left']
labels = ['Clean', 'Left zero', 'Right zero', 'L to R', 'R to L']
data = [json.loads((ROOT/'gate_unity'/f'{c}.json').read_text()) for c in conditions]
joints = ['Eye L','Eye R','Shoulder L','Shoulder R','Elbow L','Elbow R',
          'Hip L','Hip R','Knee L','Knee R','Ankle L','Ankle R','Wrist R','Wrist L','Neck']
plt.rcParams.update({'font.size':9, 'pdf.fonttype':42, 'ps.fonttype':42})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained', gridspec_kw={'width_ratios':[1,1.35]})
d = data[0]['gate_diagnostics']
edges = np.asarray(d['histogram_edges']); counts=np.asarray(d['histogram_counts'])
axes[0].bar((edges[1:]+edges[:-1])/2, counts/counts.sum(), width=.048, color='#0072B2')
axes[0].axvline(.5, color='gray', linestyle='--', linewidth=1)
axes[0].axvline(d['alpha_mean'],color='#D55E00',label=f"Mean = {d['alpha_mean']:.4f}")
axes[0].set(xlabel='Left-candidate weight',ylabel='Joint-frame fraction',xlim=(0,1),title='(a) Clean full-split distribution')
axes[0].legend(frameon=False)
values=np.stack([x['gate_diagnostics']['alpha_mean_by_time_joint'] for x in data]).mean(axis=1)
im=axes[1].imshow(values.T, vmin=0, vmax=1, cmap='viridis', aspect='auto')
axes[1].set(xticks=range(5),xticklabels=labels,yticks=range(15),yticklabels=joints,title='(b) Full-split joint means')
fig.colorbar(im,ax=axes[1],label='Left-candidate weight')
fig.savefig(OUT/'gate_diagnostics_full.pdf'); plt.close(fig)

fig, axes = plt.subplots(2,5,figsize=(12,5.8),layout='constrained')
for row in range(2):
    for col,d in enumerate(data):
        a=np.asarray(d['fixed_first_two_exemplars']['alpha'])[row,:,:,0].T
        im=axes[row,col].imshow(a,vmin=0,vmax=1,cmap='viridis',aspect='auto',origin='upper',extent=[.5,30.5,14.5,-.5])
        axes[row,col].set(xticks=[1,15,30],xlabel='Frame',title=f'Index {row}: {labels[col]}')
        if col==0:
            axes[row,col].set(yticks=range(15),yticklabels=joints)
        else:
            axes[row,col].set(yticks=[])
        if col>=3:
            axes[row,col].axvline(15.5,color='white',ls='--',lw=.8)
fig.colorbar(im,ax=axes.ravel().tolist(),label='Left-candidate weight',shrink=.8)
fig.savefig(OUT/'gate_diagnostics_exemplars.pdf'); plt.close(fig)
