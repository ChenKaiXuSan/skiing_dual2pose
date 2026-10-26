"""Export traceable drawing materials; no training, no model-weight inference.

Run from repository root: python3 paper/ivc_figure_handoff_20260917/build_assets.py
Frozen gate components reconstruct the output. Canonicalization and Sim3 are
replayed on the original first batch (256 samples) on CPU for input parity.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from dual2pose.eval.main_baseline_data import canonical_batch
from dual2pose.trainer.canonicalize import canonicalize_pose_torch
from dual2pose.models.sim3 import align_points_sim3
from dual2pose.models.crossview_fusion import rotation_matrix_to_rotvec, compute_velocity
from dual2pose.map_config import FILTER_SKELETON_CONNECTIONS, FILTERED_KPTS_MAPPING

CACHE = ROOT/'logs/ivc_mmsports_extension/main_baselines/20260906/cache/unity/test'
EVID = ROOT/'paper/ivc_draft_20260821/evidence/review_revision_20260916/gate_unity'
PAPER = ROOT/'paper/ivc_draft_20260821'
for sub in ['assets', 'data', 'reference']:
    (OUT/sub).mkdir(exist_ok=True)
torch.set_num_threads(2)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'svg.fonttype':'none','pdf.fonttype':42})
COL = {'left':'#587C9E','right':'#5C9488','fused':'#D18B47',
       'gt':'#575D66','difference':'#A06BA5'}
JT = ['Eye L','Eye R','Shoulder L','Shoulder R','Elbow L','Elbow R',
      'Hip L','Hip R','Knee L','Knee R','Ankle L','Ankle R','Wrist R','Wrist L','Neck']
raw = [torch.from_numpy(np.load(CACHE/f'{n}.npy',mmap_mode='r')[:256].copy())
       for n in ['left','right','target']]
l,r,g = canonical_batch(*raw,dataset='unity')
rl,info_rl = align_points_sim3(r,l)
lr,info_lr = align_points_sim3(l,r)
source = json.loads((EVID/'clean.json').read_text())
config = json.loads((EVID/'config.json').read_text())
assert json.loads((CACHE/'manifest.json').read_text()) == config['cache_manifest']
e = {k:np.asarray(v) for k,v in source['fixed_first_two_exemplars'].items()}
parity = {'base_l':float(np.max(np.abs((.5*(l+rl))[:2].numpy()-e['base_l']))),
          'base_r':float(np.max(np.abs((.5*(r+lr))[:2].numpy()-e['base_r'])))}
assert max(parity.values()) < 1e-6, parity
sample = json.loads((CACHE/'samples.jsonl').open().readline())
frames = np.load(CACHE/'frames.npy',mmap_mode='r')[0].copy()
TI = 14  # Fixed before inspecting the local joint errors; zero-based.
base = e['alpha']*e['base_l']+(1-e['alpha'])*e['base_r']
fused = base+e['output_residual']
arrays = {'raw_left':raw[0][0].numpy(),'raw_right':raw[1][0].numpy(),
          'left':l[0].numpy(),'right':r[0].numpy(),'right_to_left':rl[0].numpy(),
          'left_to_right':lr[0].numpy(),'gt_canonical':g[0].numpy(),
          'difference_left':(rl-l)[0].numpy(),'velocity_left':compute_velocity(l)[0].numpy(),
          'rotation_right_to_left':rotation_matrix_to_rotvec(info_rl['rotation'])[0].numpy(),
          'rotation_matrix_right_to_left':info_rl['rotation'][0].numpy(),
          'alpha':e['alpha'][0], 'base_left':e['base_l'][0], 'base_right':e['base_r'][0],
          'weighted_base':base[0], 'output_residual':e['output_residual'][0],
          'fused':fused[0], 'canonical_average':(.5*(l+r))[0].numpy(),'frame_ids':frames}
assert all(np.isfinite(a).all() for a in arrays.values())
np.savez_compressed(OUT/'data/same_sample_arrays.npz',**arrays)
with (OUT/'data/poses.csv').open('w') as h:
    w=csv.writer(h); w.writerow(['representation','window_index_0based','source_frame','joint_index_0based','joint','x','y','z'])
    for name,a in arrays.items():
        if a.shape == (30,15,3):
            for t in range(30):
                for j in range(15):
                    w.writerow([name,t,int(frames[t]),j,JT[j],*a[t,j]])
with (OUT/'data/gate_alpha.csv').open('w') as h:
    w=csv.writer(h); w.writerow(['window_index_0based','source_frame',*JT])
    for t in range(30): w.writerow([t,int(frames[t]),*arrays['alpha'][t,:,0]])
with (OUT/'data/joints_and_edges.json').open('w') as h:
    json.dump({'joint_names':FILTERED_KPTS_MAPPING,'edges':FILTER_SKELETON_CONNECTIONS,
               'note':'15 nodes; eyes are isolated points in the repository topology. Do not invent extra predicted joints.'},h,indent=2)
shutil.copy2(EVID/'clean.json',OUT/'data/source_gate_clean.json')
shutil.copy2(EVID/'config.json',OUT/'data/source_gate_config.json')
for name in ['study_overview','method_pipeline','feature_details']:
    shutil.copy2(PAPER/f'figures/journal/{name}.pdf',OUT/f'reference/{name}.pdf')

catalog=[]
def save(fig,name,description,kind='measured'):
    for ext in ['svg','pdf','png']:
        fig.savefig(OUT/f'assets/{name}.{ext}',bbox_inches='tight',pad_inches=.10,
                    transparent=True,dpi=300)
    plt.close(fig)
    catalog.append({'id':name,'description':description,'kind':kind})

# A single fixed oblique projection is used for all canonical pose assets.
# This is a drawing projection, NOT either physical camera projection.
def proj(p):
    return np.stack([p[...,0]+.45*p[...,2],p[...,1]+.15*p[...,2]],axis=-1)

canon_limits=proj(np.concatenate([arrays[n].reshape(-1,3) for n in
              ['left','right','right_to_left','base_left','base_right','fused','gt_canonical']]))
lo=canon_limits.min(0); hi=canon_limits.max(0); center=(lo+hi)/2
span=max(hi-lo)*1.12
def setup(ax,title='',fixed=True):
    ax.set_aspect('equal'); ax.axis('off')
    if fixed:
        ax.set_xlim(center[0]-span/2,center[0]+span/2)
        ax.set_ylim(center[1]-span/2,center[1]+span/2)
    ax.set_title(title,pad=10)

def skeleton(ax,p,color,lw=2.1,alpha=1.,ls='-',label=None):
    q=proj(p)
    for k,(a,b) in enumerate(FILTER_SKELETON_CONNECTIONS):
        ax.plot(q[[a,b],0],q[[a,b],1],color=color,lw=lw,alpha=alpha,ls=ls,
                label=label if k==0 else None)
    ax.scatter(q[:,0],q[:,1],s=10,color=color,alpha=alpha,zorder=3)

def arrows(ax,p,d,color=COL['difference']):
    a,b=proj(p),proj(p+d)
    for j in [4,5,10,11,12,13]:
        ax.annotate('',xy=b[j],xytext=a[j],arrowprops={'arrowstyle':'->','color':color,'lw':1.5})

for name,label,col,fixed in [
    ('raw_left','Input 3D pose L',COL['left'],False),
    ('raw_right','Input 3D pose R',COL['right'],False),
    ('left','Own-view pose',COL['left'],True),
    ('right_to_left','Aligned other view',COL['right'],True),
    ('base_left','Base candidate B L',COL['left'],True),
    ('base_right','Base candidate B R',COL['right'],True),
    ('fused','Refined canonical pose',COL['fused'],True),
    ('gt_canonical','Canonicalized Unity GT',COL['gt'],True)]:
    fig,ax=plt.subplots(figsize=(3.2,3.4)); skeleton(ax,arrays[name][TI],col)
    setup(ax,label,fixed)
    if name.startswith('raw_'):
        ax.invert_yaxis()  # Camera-space y points down; display only.
    save(fig,name,f'{label}; sample 0, window index 14 / source frame {frames[TI]}')

fig,axes=plt.subplots(1,2,figsize=(6.4,3.5))
for ax,n,title in [(axes[0],'right','Before sequence alignment'),(axes[1],'right_to_left','After R to L alignment')]:
    skeleton(ax,arrays['left'][TI],COL['left'],label='L')
    skeleton(ax,arrays[n][TI],COL['right'],ls='--',label='R' if n=='right' else 'R to L')
    setup(ax,title);ax.legend(loc='lower left',frameon=False,fontsize=8)
save(fig,'alignment_before_after','Both panels share limits. R to L only; L to R is symmetric in construction.')

fig,ax=plt.subplots(figsize=(3.2,3.4))
skeleton(ax,arrays['left'][TI],COL['left'],alpha=.65)
skeleton(ax,arrays['right_to_left'][TI],COL['right'],alpha=.65,ls='--')
arrows(ax,arrays['left'][TI],arrays['difference_left'][TI]);setup(ax,'Cross-view difference')
save(fig,'cue_difference','True displacement arrows from own pose to aligned opposite pose; no magnification.')

fig,ax=plt.subplots(figsize=(3.2,3.4))
skeleton(ax,arrays['left'][TI-1],COL['left'],alpha=.25,ls='--')
skeleton(ax,arrays['left'][TI],COL['left'])
arrows(ax,arrays['left'][TI-1],arrays['velocity_left'][TI]);setup(ax,'Temporal first difference')
save(fig,'cue_velocity','Actual adjacent resampled positions; arrows are not magnified; no division by time.')

fig,ax=plt.subplots(figsize=(3.8,3.4))
p=arrays['raw_left'][0]; skeleton(ax,p,COL['left'],alpha=.6)
_,tf=canonicalize_pose_torch(raw[0]);c=tf['pelvis'].numpy();R=tf['R'].numpy()
q=proj(c); ax.scatter(*q,color=COL['gt'],s=35,zorder=5)
for i,label in enumerate(['ex (hip)','ey (neck)','ez (cross product)']):
    end=proj(c+.3*R[:,i]);ax.annotate('',xy=end,xytext=q,arrowprops={'arrowstyle':'->','lw':2,'color':['#587C9E','#5C9488','#A06BA5'][i]})
    ax.annotate(label,xy=end,xytext=[(-35,-25),(10,0),(-70,20)][i],textcoords='offset points',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':1})
ax.annotate('Hip midpoint',xy=q,xytext=(15,-20),textcoords='offset points',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':1})
setup(ax,'Body reference: first anchor pose',False)
ax.invert_yaxis()
save(fig,'body_reference','Actual first raw left anchor; fixed pelvis and landmark axes. Axis lengths scaled for legibility.')

fig=plt.figure(figsize=(3.4,3.4));ax=fig.add_subplot(111,projection='3d')
rot=arrays['rotation_matrix_right_to_left']
for k,col in enumerate([COL['left'],COL['right'],COL['difference']]):
    v=np.eye(3)[:,k];u=rot@v
    ax.quiver(0,0,0,*v,color=col,linestyle='dashed',linewidth=1)
    ax.quiver(0,0,0,*u,color=col,linewidth=2)
ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1));ax.set_axis_off();ax.set_title('Alignment rotation vector')
ax.text2D(.05,.02,'One 3D cue per sequence; broadcast to T x J',transform=ax.transAxes,fontsize=8)
save(fig,'cue_rotation','Actual sequence alignment rotation, shown as basis rotation; no camera extrinsics.')

fig,ax=plt.subplots(figsize=(6,3.5))
q1=arrays['left'][TI].copy();q2=arrays['right'][TI].copy()
q1[:,0]-=.65;q2[:,0]+=.65
skeleton(ax,q1,COL['left']);skeleton(ax,q2,COL['right'])
for a,b in [(4,4),(4,13),(14,2),(10,10)]:
    u,v=proj(q1[a]),proj(q2[b]);ax.annotate('',xy=v,xytext=u,
        arrowprops={'arrowstyle':'<->','color':'#9298A1','lw':1,'linestyle':'--'})
setup(ax,'Cross-view attention over joints',False)
ax.text(.5,-.02,'Connections are illustrative; no measured attention weights shown.',transform=ax.transAxes,ha='center',fontsize=8)
save(fig,'attention_connections','Conceptual connections on real poses. Uniform lines; all-joint attention, not only matching-joint links.','schematic')

fig,axes=plt.subplots(1,3,figsize=(7,3))
for ax,t in zip(axes,[13,14,15]):
    skeleton(ax,arrays['left'][t],COL['left']);setup(ax,f'Window {t+1} / source {frames[t]}')
save(fig,'temporal_poses','Three representative actual poses for illustration; refiner kernel size remains FIVE.')

fig,ax=plt.subplots(figsize=(6,4.2))
im=ax.imshow(arrays['alpha'][:,:,0].T,aspect='auto',cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
ax.set(yticks=range(15),yticklabels=JT,xticks=[0,9,19,29],xticklabels=['1','10','20','30'],
       xlabel='Resampled window position',ylabel='Joint',title='Measured joint-frame gate (sample 0)')
fig.colorbar(im,ax=ax,label='Weight of B L (alpha); B R uses 1 - alpha')
save(fig,'gate_heatmap','Measured alpha; fixed 0-1 color scale; do not stretch contrast to imply binary switching.')

fig,ax=plt.subplots(figsize=(3.4,3.5))
skeleton(ax,arrays['weighted_base'][TI],COL['gt'],alpha=.5,ls='--',label='Weighted base')
skeleton(ax,arrays['fused'][TI],COL['fused'],label='Refined')
arrows(ax,arrays['weighted_base'][TI],arrays['output_residual'][TI]);setup(ax,'Residual correction')
ax.legend(loc='lower left',frameon=False,fontsize=8)
save(fig,'residual_correction','True output correction arrows, no magnification; different from cross-view feature difference.')

fig,axes=plt.subplots(1,2,figsize=(6.4,3.5))
for ax,n,label,col in [(axes[0],'canonical_average','Canonical average',COL['left']),
                        (axes[1],'fused','CanonFuse3D',COL['fused'])]:
    skeleton(ax,arrays['gt_canonical'][TI],COL['gt'],ls='--',label='Unity GT')
    skeleton(ax,arrays[n][TI],col,label=label);setup(ax,label)
    ax.legend(loc='lower left',frameon=False,fontsize=8)
save(fig,'output_comparison','Same sample/frame/projection/limits; GT canonicalized with original independent batch anchor; no per-panel fitted alignment.')

err_avg=np.linalg.norm(arrays['canonical_average'][TI]-arrays['gt_canonical'][TI],axis=-1)
err_fused=np.linalg.norm(arrays['fused'][TI]-arrays['gt_canonical'][TI],axis=-1)
with (OUT/'data/local_errors.csv').open('w') as h:
    w=csv.writer(h);w.writerow(['joint','canonical_average_error','fused_error','coordinate_units'])
    for j in range(15):w.writerow([JT[j],err_avg[j],err_fused[j],'canonical coordinate units'])

rgb_sources=[]
for view,key in [('left','cam1_id'),('right','cam2_id')]:
    src=ROOT.parent/'data/skiing_unity_dataset/data_pole_ski'/sample['person_id']/sample['action_id']/'frames'/sample[key]/f'frame_{int(frames[TI]):06d}.png'
    assert src.is_file(),src
    dest=OUT/f'assets/rgb_{view}_source_frame_{frames[TI]:06d}.png'
    shutil.copy2(src,dest);rgb_sources.append(str(src))
    catalog.append({'id':dest.stem,'description':f'Unmodified RGB; {sample[key]}; same physical frame as the poses.','kind':'measured_rgb'})

metadata={'sample':sample,'selection':'Fixed sample index 0, fixed window index 14; not selected by gain.',
          'window_index_0based':TI,'source_frame':int(frames[TI]),'frame_ids':frames.tolist(),
          'canonical_batch_size':256,'base_parity_max_abs_error':parity,
          'output_source':'Reconstructed from archived alpha/base_l/base_r/output_residual; no new network inference.',
          'alpha_range_sample_0':[float(arrays['alpha'].min()),float(arrays['alpha'].max())],
          'frame_all15_error':{'canonical_average':float(err_avg.mean()),'fused':float(err_fused.mean()),'unit':'canonical coordinate units'},
          'projection':'Canonical poses: drawing-only oblique projection u=x+0.45z, v=y+0.15z. Raw poses and body-reference display invert v because camera y points down. Neither is a physical camera projection.',
          'rgb_sources':rgb_sources,'checkpoint_sha256':config['checkpoint_sha256'],
          'limitations':['Single fixed sample, not aggregate performance evidence.',
                         'View separation is 10 degrees; do not redraw cameras as a wide-baseline example.',
                         'Resampled positions repeat source frames; window position is not physical time.',
                         'Body references are separate; output does not recover a shared physical world frame.',
                         'Attention links are schematic; no attention weights have been exported.'],
          'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [EVID/'clean.json',EVID/'config.json',PAPER/'main.tex',ROOT/'dual2pose/models/crossview_fusion.py',ROOT/'dual2pose/trainer/canonicalize.py',ROOT/'dual2pose/models/sim3.py',CACHE/'manifest.json']}}
(OUT/'data/provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
(OUT/'data/asset_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(metadata,ensure_ascii=False,indent=2))
