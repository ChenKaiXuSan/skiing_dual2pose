"""Export four traceable sample bundles per dataset on CPU.

Keeps the original single-sample assets. Selection uses metadata and clip
midpoints, never model error. Does not train or overwrite experiment logs.
"""
from pathlib import Path
import base64
import csv
import hashlib
import html
import io
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from dual2pose.eval.main_baseline_data import canonical_batch
from dual2pose.models.sim3 import align_points_sim3
from dual2pose.models.crossview_fusion import compute_velocity, rotation_matrix_to_rotvec
from dual2pose.trainer.canonicalize import canonicalize_pose_torch
from dual2pose.experiments.run_main_baselines import CHECKPOINTS, ReferenceModel
from dual2pose.map_config import FILTER_SKELETON_CONNECTIONS, filter_sam3d_body_kpts

DEST=OUT/'multi_samples'
DEST.mkdir(exist_ok=True)
CACHE=ROOT/'logs/ivc_mmsports_extension/main_baselines/20260906/cache'
DATA=ROOT.parent/'data'
JN=['Eye L','Eye R','Shoulder L','Shoulder R','Elbow L','Elbow R','Hip L','Hip R',
    'Knee L','Knee R','Ankle L','Ankle R','Wrist R','Wrist L','Neck']
COL=['#587C9E','#5C9488','#D18B47','#575D66','#A06BA5']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
torch.set_num_threads(2)
torch.manual_seed(42)
MODELS={k:ReferenceModel(v).model.eval() for k,v in CHECKPOINTS.items()}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def proj(p):return np.stack([p[...,0]+.45*p[...,2],p[...,1]+.15*p[...,2]],axis=-1)

# Gate record parity verifies checkpoint and forward conventions before export.
parity={}
for ds,bs in [('unity',256),('ski',4)]:
    raw=[torch.from_numpy(np.load(CACHE/ds/'test'/f'{n}.npy',mmap_mode='r')[:bs].copy()) for n in ['left','right','target']]
    l,r,_=canonical_batch(*raw,dataset=ds)
    with torch.no_grad():_,aux=MODELS[ds](l,r,return_components=True)
    frozen=json.loads((ROOT/f'paper/ivc_draft_20260821/evidence/review_revision_20260916/gate_{ds}/clean.json').read_text())['fixed_first_two_exemplars']
    parity[ds]={k:float(np.max(np.abs(aux[k][:2].numpy()-np.asarray(frozen[k])))) for k in ['alpha','base_l','base_r','output_residual']}
    assert max(parity[ds].values())<2e-5,parity
write(DEST/'checkpoint_parity.json',parity)

rows={ds:[json.loads(x) for x in (CACHE/ds/'test/samples.jsonl').read_text().splitlines()] for ds in ['unity','ski']}
groups=list(dict.fromkeys((r['person_id'],r['action_id']) for r in rows['unity']))
selection=[]
for n,(group,angle) in enumerate(zip(groups,[30,60,120,180]),1):
    row=next(r for r in rows['unity'] if (r['person_id'],r['action_id'])==group and r['cam1_id']=='capture_L0_A000' and r['cam2_id']==f'capture_L0_A{angle:03d}')
    selection.append({'id':f'U{n:02d}','dataset':'unity','index':row['index'],'azimuth_gap_degrees':angle})
for n,index in enumerate([0,9,15,24],1):selection.append({'id':f'S{n:02d}','dataset':'ski','index':index})
for n,run in enumerate(['pro_1','pro_2','run_3','run_4'],1):selection.append({'id':f'A{n:02d}','dataset':'author','run':run,'scene':'Outdoor' if run.startswith('pro') else 'Indoor'})
write(DEST/'selection_plan.json',{'rule':'Four Unity action groups with predefined camera gaps; two Ski sequences with two camera pairs each; four author runs. Main display at window position 15 (cached datasets) or clip midpoint (author). No gain-based filtering.','samples':selection})

def infer(l,r,key):
    with torch.no_grad():
        output,aux=MODELS[key](l,r,return_components=True)
    for name in ['attn_l','attn_r']:
        aux[name]=aux[name].reshape(l.shape[0],l.shape[1],l.shape[2],l.shape[2])
        assert torch.allclose(aux[name].sum(-1),torch.ones_like(aux[name].sum(-1)),atol=2e-6)
    return output,aux


def load_sample(item):
    ds=item['dataset'];meta=dict(item)
    if ds in ['unity','ski']:
        c=CACHE/ds/'test';idx=item['index'];bs=256 if ds=='unity' else 4
        start=idx//bs*bs;raw=[torch.from_numpy(np.load(c/f'{n}.npy',mmap_mode='r')[start:start+bs].copy()) for n in ['left','right','target']]
        l,r,target=canonical_batch(*raw,dataset=ds);off=idx-start
        fused,aux=infer(l,r,ds)
        l,r=l[off:off+1],r[off:off+1];fused=fused[off:off+1];aux={k:v[off:off+1] for k,v in aux.items()}
        raw_l,raw_r=raw[0][off:off+1],raw[1][off:off+1]
        ref=target[off].numpy();frames=np.load(c/'frames.npy',mmap_mode='r')[idx].copy();t=14;row=rows[ds][idx]
        meta.update(sample=row,batch_size=bs,batch_start_index=start,reference_kind='Unity avatar GT' if ds=='unity' else 'Constructed canonical reference',cache_manifest_sha256=sha(c/'manifest.json'))
        if ds=='unity':
            rgb=[DATA/'skiing_unity_dataset/data_pole_ski'/row['person_id']/row['action_id']/'frames'/row[k]/f'frame_{frames[t]:06d}.png' for k in ['cam1_id','cam2_id']]
        else:
            manifest=json.loads((c/'manifest.json').read_text());mapping=json.loads(Path(manifest['index']).read_text())['test']
            cfg=next(r for r in mapping if int(r['sequence_id'])==row['seq'] and int(r['cam1_id'])==row['cam1'] and int(r['cam2_id'])==row['cam2'])
            rgb=[Path(cfg[k].replace('/home/kaixu_chen/data/skiing','/home/kaixu_chen/skiing/data'))/f'image_{frames[t]:06d}.png' for k in ['cam1_frames_dir','cam2_frames_dir']]
        imgs=[np.asarray(Image.open(p).convert('RGB')) for p in rgb]
        meta['rgb_sources']=[str(p) for p in rgb];meta['rgb_source_sha256']=[sha(p) for p in rgb]
        ckpt=ds
    else:
        run=DATA/'sam3d_body_results/person'/item['run']
        files=[{int(p.stem.split('_')[1]):p for p in (run/v).glob('*.npz')} for v in ['left','right']]
        shared=sorted(set(files[0])&set(files[1]));assert len(shared)>=30
        mid=len(shared)//2;start=max(0,min(mid-15,len(shared)-30));frames=np.array(shared[start:start+30]);t=mid-start
        poses=[];imgs=[];source_files=[]
        for fs in files:
            pp=[]
            for fid in frames:
                p=fs[int(fid)]
                with np.load(p,allow_pickle=True) as archive:
                    if 'pred_keypoints_3d' in archive:
                        z=archive
                    else:
                        values=archive['outputs'] if 'outputs' in archive else archive['output']
                        assert values.size==1, 'Ambiguous multiple-person archive'
                        z=values.item() if values.ndim==0 else values.reshape(-1)[0]
                    if 'frame_idx' in z:
                        assert int(z['frame_idx'])==fid
                    pp.append(filter_sam3d_body_kpts(z['pred_keypoints_3d']))
                    if fid==frames[t]:imgs.append(np.asarray(z['frame']).copy())
                source_files.append({'path':str(p),'sha256':sha(p)})
            poses.append(torch.from_numpy(np.stack(pp)).float()[None])
        raw_l,raw_r=poses;l=canonicalize_pose_torch(raw_l)[0];r=canonicalize_pose_torch(raw_r)[0]
        ckpt='unity'  # Fixed choice, not selected on appearance or error.
        fused,aux=infer(l,r,ckpt)
        ref=None;meta.update(reference_kind='No GT: qualitative only',batch_size=1,window_start_position=start,available_common_frames=len(shared),source_npz=source_files,
                             rgb_sources=[str(fs[int(frames[t])])+'::frame' for fs in files],sync_scope='Matching filenames; embedded frame_idx additionally checked when available. No new synchronization estimation.')
    rl,info=align_points_sim3(r,l);lr,_=align_points_sim3(l,r)
    a={'raw_left':raw_l[0].numpy(),'raw_right':raw_r[0].numpy(),'left':l[0].numpy(),'right':r[0].numpy(),
       'right_to_left':rl[0].numpy(),'left_to_right':lr[0].numpy(),'difference_left':(rl-l)[0].numpy(),
       'velocity_left':compute_velocity(l)[0].numpy(),'rotation_right_to_left':rotation_matrix_to_rotvec(info['rotation'])[0].numpy(),
       'rotation_matrix_right_to_left':info['rotation'][0].numpy(),'fused':fused[0].numpy(),'canonical_average':(.5*(l+r))[0].numpy(),
       'frame_ids':frames}
    for k,new in [('alpha','alpha'),('base_l','base_left'),('base_r','base_right'),('output_residual','output_residual'),('attn_l','attention_left_to_right'),('attn_r','attention_right_to_left')]:a[new]=aux[k][0].numpy()
    a['weighted_base']=a['alpha']*a['base_left']+(1-a['alpha'])*a['base_right']
    if ref is not None:a['reference']=ref
    assert all(np.isfinite(v).all() for v in a.values())
    err=float(np.max(np.abs(a['fused']-a['weighted_base']-a['output_residual'])))
    assert err<1e-6
    meta.update(window_index_0based=int(t),source_frame=int(frames[t]),checkpoint=str(CHECKPOINTS[ckpt]),checkpoint_sha256=sha(CHECKPOINTS[ckpt]),
                inference='New frozen-checkpoint CPU inference, float32, eval mode; not retraining.',num_joints=int(l.shape[2]),reconstruction_max_abs_error=err,
                alpha_min=float(a['alpha'].min()),alpha_max=float(a['alpha'].max()),projection='Canonical: u=x+0.45z,v=y+0.15z; raw display flips v. Display only, not RGB projection.',
                limitations=['Selected illustration, not aggregate or independently sampled evidence.','Reference is never a model input.','No 2D skeleton overlay exported.'])
    return a,imgs,meta

def render(item,a,imgs,meta):
    folder=DEST/item['dataset']/item['id'];folder.mkdir(parents=True,exist_ok=True)
    t=meta['window_index_0based'];J=meta['num_joints'];names=JN if J==15 else JN[2:]
    edges=FILTER_SKELETON_CONNECTIONS if J==15 else [(u-2,v-2) for u,v in FILTER_SKELETON_CONNECTIONS]
    np.savez_compressed(folder/'arrays.npz',**a)
    with (folder/'poses.csv').open('w') as h:
        w=csv.writer(h);w.writerow(['representation','window_index_0based','source_frame','joint_index','joint','x','y','z'])
        for k,v in a.items():
            if v.shape==(30,J,3):
                for ti in range(30):
                    for j in range(J):w.writerow([k,ti,int(a['frame_ids'][ti]),j,names[j],*v[ti,j]])
    with (folder/'gate_alpha.csv').open('w') as h:
        w=csv.writer(h);w.writerow(['window_index_0based','source_frame',*names])
        for ti in range(30):w.writerow([ti,int(a['frame_ids'][ti]),*map(float,a['alpha'][ti,:,0])])
    write(folder/'joints_and_edges.json',{'joint_names':names,'edges':edges})
    for view,img in zip(['left','right'],imgs):Image.fromarray(img).save(folder/f'rgb_{view}.png')
    limits=proj(np.concatenate([a[k].reshape(-1,3) for k in ['left','right','right_to_left','fused','canonical_average']+(['reference'] if 'reference' in a else [])]))
    center=(limits.min(0)+limits.max(0))/2;span=max(np.ptp(limits,axis=0))*1.15
    def setup(ax,title='',fixed=True):
        ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=9,pad=8)
        if fixed:ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2)
    def sk(ax,p,col,ls='-',alpha=1,label=None):
        q=proj(p)
        for n,(u,v) in enumerate(edges):ax.plot(q[[u,v],0],q[[u,v],1],color=col,lw=1.8,ls=ls,alpha=alpha,label=label if n==0 else None)
        ax.scatter(q[:,0],q[:,1],s=8,color=col,alpha=alpha)
    catalog=[]
    def save(fig,n):
        for ext in ['svg','pdf','png']:fig.savefig(folder/f'{n}.{ext}',bbox_inches='tight',pad_inches=.1,transparent=True,dpi=300)
        plt.close(fig);catalog.append(n)
    def pose_panel(ax,key,title):sk(ax,a[key][t],COL[2] if key=='fused' else (COL[1] if key=='right' else COL[0]));setup(ax,title)
    for key,col,title in [('raw_left',COL[0],'Raw input L'),('raw_right',COL[1],'Raw input R'),('left',COL[0],'Own-view pose'),('right_to_left',COL[1],'Aligned opposite pose'),('base_left',COL[0],'Base candidate B L'),('base_right',COL[1],'Base candidate B R'),('fused',COL[2],'Refined canonical output')]+([('reference',COL[3],meta['reference_kind'])] if 'reference' in a else []):
        fig,ax=plt.subplots(figsize=(3.2,3.5));sk(ax,a[key][t],col);setup(ax,title,not key.startswith('raw'))
        if key.startswith('raw'):ax.invert_yaxis()
        save(fig,key)
    fig,axs=plt.subplots(1,2,figsize=(6.4,3.5))
    for ax,key,title in zip(axs,['right','right_to_left'],['Before alignment','After R to L alignment']):
        sk(ax,a['left'][t],COL[0]);sk(ax,a[key][t],COL[1],ls='--');setup(ax,title)
    save(fig,'alignment_before_after')
    for key,start,end,title in [('cross_view_difference',a['left'][t],a['right_to_left'][t],'Cross-view difference'),('temporal_velocity',a['left'][t-1],a['left'][t],'Temporal first difference'),('residual_correction',a['weighted_base'][t],a['fused'][t],'Output residual correction')]:
        fig,ax=plt.subplots(figsize=(3.2,3.5));sk(ax,start,COL[0],alpha=.55);sk(ax,end,COL[1] if key!='residual_correction' else COL[2],ls='--');setup(ax,title)
        for j in ([4,5,10,11,12,13] if J==15 else [2,3,8,9,10,11]):ax.annotate('',xy=proj(end[j]),xytext=proj(start[j]),arrowprops={'arrowstyle':'->','color':COL[4],'lw':1.3})
        save(fig,key)
    fig=plt.figure(figsize=(3.8,3.5));ax=fig.add_subplot(111,projection='3d')
    for i in range(3):
        v=np.eye(3)[:,i];u=a['rotation_matrix_right_to_left']@v
        ax.quiver(0,0,0,*v,color=COL[i],linestyle='dashed');ax.quiver(0,0,0,*u,color=COL[i],linewidth=2)
    ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1));ax.set_axis_off();ax.set_title('Alignment rotation (R to L)')
    ax.text2D(0,0,'3D rotation vector; broadcast over frames and joints',transform=ax.transAxes,fontsize=8);save(fig,'alignment_rotation')
    fig,axs=plt.subplots(1,3,figsize=(7.5,3.4))
    for ax,ti in zip(axs,[t-1,t,t+1]):sk(ax,a['left'][ti],COL[0]);setup(ax,f'Position {ti+1} / frame {a["frame_ids"][ti]}')
    save(fig,'temporal_poses')
    def heat(ax,small=False):
        im=ax.imshow(a['alpha'][:,:,0].T,aspect='auto',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
        ax.set(xticks=[0,14,29],xticklabels=['1','15','30'],xlabel='Window position',yticks=range(J),yticklabels=names if not small else [],title='Measured gate alpha')
        return im
    fig,ax=plt.subplots(figsize=(5.8,4));im=heat(ax);fig.colorbar(im,ax=ax,label='Weight of B L; B R uses 1 - alpha');save(fig,'gate_heatmap')
    fig,ax=plt.subplots(figsize=(5.2,4.5));att=a['attention_left_to_right'][t]
    assert att.shape==(J,J),att.shape
    im=ax.imshow(att,vmin=0,vmax=1,cmap='viridis',interpolation='nearest');ax.set(xticks=range(J),yticks=range(J),xticklabels=names,yticklabels=names,xlabel='Right key joint',ylabel='Left query joint',title='Measured cross-view attention (head average)');ax.tick_params(axis='x',labelrotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=7);fig.colorbar(im,ax=ax,label='Attention weight');save(fig,'attention_heatmap')
    def comparison(ax,key,title):
        if 'reference' in a:sk(ax,a['reference'][t],COL[3],ls='--',label='Unity GT' if item['dataset']=='unity' else 'Constructed reference')
        sk(ax,a[key][t],COL[2] if key=='fused' else COL[0],label=title);setup(ax,title);ax.legend(loc='lower left',fontsize=7,frameon=False)
    fig,axs=plt.subplots(1,2,figsize=(6.4,3.5))
    for ax,key,title in zip(axs,['canonical_average','fused'],['Canonical average','CanonFuse3D']):comparison(ax,key,title)
    save(fig,'output_comparison')
    if 'reference' in a:
        ev={k:np.linalg.norm(a[k][t]-a['reference'][t],axis=-1) for k in ['canonical_average','fused']}
        with (folder/'local_errors.csv').open('w') as h:
            w=csv.writer(h);w.writerow(['joint','average_error','fused_error','reference_kind','unit'])
            for j in range(J):w.writerow([names[j],ev['canonical_average'][j],ev['fused'][j],meta['reference_kind'],'dataset canonical coordinate units'])
        meta['frame_errors']={k:float(v.mean()) for k,v in ev.items()}
    fig,axs=plt.subplots(2,4,figsize=(12,6.7),layout='constrained')
    for ax,img,title in zip(axs[0,:2],imgs,['RGB L','RGB R']):ax.imshow(img);ax.axis('off');ax.set_title(title)
    for ax,key,title in zip(axs[0,2:],['left','right'],['Canonical input L','Canonical input R']):pose_panel(ax,key,title)
    sk(axs[1,0],a['left'][t],COL[0]);sk(axs[1,0],a['right_to_left'][t],COL[1],ls='--');setup(axs[1,0],'Aligned R to L')
    comparison(axs[1,1],'canonical_average','Canonical average');comparison(axs[1,2],'fused','CanonFuse3D');im=heat(axs[1,3],True);fig.colorbar(im,ax=axs[1,3],shrink=.7)
    fig.suptitle(f'{item["id"]} | {item["dataset"]} | frame {meta["source_frame"]} | {meta["reference_kind"]}',fontsize=12)
    save(fig,'sample_preview')
    meta['assets']=catalog;write(folder/'provenance.json',meta)
    cards=''.join(f'<article><h3>{n}</h3><a href="{n}.svg">SVG</a> · <a href="{n}.pdf">PDF</a><img loading="lazy" src="{n}.png"></article>' for n in catalog)
    (folder/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{item["id"]} 素材</title><style>body{{font-family:Arial;margin:24px}}main{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img{{width:100%;max-height:420px;object-fit:contain}}article{{border:1px solid #ddd;padding:15px}}</style><h1>{item["id"]} / {meta["reference_kind"]}</h1><p>同样本独立素材。CSV/NPZ 保存三维数据，骨架图不能直接覆盖到 RGB 上。新增注意力热图是真实输出、各头平均；自采结果无 GT。</p><main>{cards}</main>')
    print('EXPORTED',item['id'],item['dataset'],meta['source_frame'],len(catalog),'vector asset sets',flush=True)
    return meta

all_meta=[]
for item in selection:
    completed=DEST/item['dataset']/item['id']/'provenance.json'
    if completed.exists():
        meta=json.loads(completed.read_text())
        required=[completed.parent/f'{n}.{ext}' for n in meta['assets'] for ext in ['svg','pdf','png']]
        if all(p.is_file() for p in required):
            if '--rerender' in sys.argv:
                with np.load(completed.parent/'arrays.npz') as z:
                    a={k:z[k] for k in z.files}
                imgs=[np.asarray(Image.open(completed.parent/f'rgb_{v}.png')) for v in ['left','right']]
                meta=render(item,a,imgs,meta)
            all_meta.append(meta);print('REUSED',item['id'],flush=True);continue
    a,imgs,meta=load_sample(item);all_meta.append(render(item,a,imgs,meta))
write(DEST/'manifest.json',{'sample_count':12,'by_dataset':{'unity':4,'ski':4,'author':4},'samples':all_meta,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'dual2pose/models/crossview_fusion.py',ROOT/'dual2pose/trainer/canonicalize.py']}})

original=OUT/'ASSET_GALLERY_SINGLE.html'
if not original.exists():shutil.copy2(OUT/'ASSET_GALLERY.html',original)
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>IVC 三数据集多样本素材</title><style>body{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:30px;background:#f5f7f8;color:#26313c}main{max-width:1200px;margin:auto}article{background:white;padding:20px;margin:22px 0;border:1px solid #dbe0e5;border-radius:8px}img{width:100%}p{line-height:1.7}h2{margin-top:36px}</style><main><h1>IVC 三数据集多样本素材 · 每类 4 组</h1><p>点击每组“全部素材”打开可编辑 SVG/PDF 与 PNG。Unity 有 avatar GT；Ski 是构造参考；自采视频无 GT，仅作定性示例。样本先按动作、视角和片段选定，没有按改善幅度筛选。原始单样本完整机制示例仍保留：<a href="ASSET_GALLERY_SINGLE.html">原版图册</a>。</p>']
for ds,title in [('unity','Unity：4 个动作 / 30、60、120、180 度'),('ski','Ski-PTZ-Pose：2 段序列 × 2 组视角'),('author','自采双视角：2 段室外 + 2 段室内')]:
    parts.append(f'<h2>{title}</h2>')
    for m in [x for x in all_meta if x['dataset']==ds]:
        rel=Path('multi_samples')/ds/m['id'];im=Image.open(OUT/rel/'sample_preview.png').convert('RGB');im.thumbnail((1300,900));buf=io.BytesIO();im.save(buf,format='PNG');b64=base64.b64encode(buf.getvalue()).decode()
        detail=m.get('sample',m.get('run'));parts.append(f'<article><h3>{m["id"]} · frame {m["source_frame"]} · {m["reference_kind"]}</h3><p>{html.escape(str(detail))}</p><a href="{rel}/index.html">全部素材</a> · <a href="{rel}/poses.csv">3D 坐标 CSV</a> · <a href="{rel}/gate_alpha.csv">门控 CSV</a><img src="data:image/png;base64,{b64}"></article>')
parts.append('</main></html>');(OUT/'ASSET_GALLERY.html').write_text(''.join(parts))

lines=['# 三个数据集的多样本素材索引','', '每类 4 组，共 12 组。所有新样本使用固定检查点在 CPU 上重新推理；原版单样本素材仍保留在根目录 assets/。先打开 ASSET_GALLERY.html 选择，再进入 multi_samples/数据集/编号/ 取素材。','', '|编号|数据集|样本|视角|主帧|参考类型|','|---|---|---|---|---|---|']
for m in all_meta:
    r=m.get('sample',{});desc=r.get('action_id',f'seq_{r.get("seq", "")}') if r else m['run'];views=f'{r["cam1_id"]} / {r["cam2_id"]}' if m['dataset']=='unity' else (f'cam {r["cam1"]} / {r["cam2"]}' if r else 'left / right')
    lines.append(f'|{m["id"]}|{m["dataset"]}|{desc}|{views}|{m["source_frame"]}|{m["reference_kind"]}|')
lines+=['','## 使用说明','', '- 每组包含双视角原图、输入与候选骨架、对齐、五路特征相关素材、时间示例、门控和真实注意力热图、输出对照、完整 30 帧数据。','- 注意力热图是当前帧、多头平均的真实权重，区别于原版中的示意连线；横轴为右流 key 关节，纵轴为左流 query 关节。','- Unity 为 15 关节；Ski 为 13 个共同关节，不补画缺失的双眼；自采为 15 关节。','- Unity 与 Ski 保留原测试批次锚点（256 / 4），不能把这些结果解释为各样本独立首帧归一化。自采采用单窗口首帧锚点。','- 自采统一使用 Unity 检查点，按片段中间时刻选样；没有按外观或效果选择模型。相同帧号为已有双视角对应，本次未重新估计时间同步。','- 自采数据没有 GT，未导出误差文件；其输出外观不能证明绝对三维精度提高。Ski 的参考必须写 Constructed reference，不能改成独立实测 GT。','- 这些是预选示例，不是独立受试者样本或全数据集统计。Ski 的四组来自同一受试者的两段序列，存在相关性。','- 所有热图采用固定 0–1 色标；输出/参考在同一示例内共用投影与范围。三维骨架展示投影不能直接叠到 RGB 上。','- 原任务书中的固定 sample 0 / frame 10 描述只适用于原版素材；新增样本的帧号、关节数和来源以本索引及每组 provenance.json 为准。']
(OUT/'MULTI_SAMPLE_INDEX.md').write_text('\n'.join(lines)+'\n')
print('COMPLETE: 12 bundles and gallery',flush=True)
