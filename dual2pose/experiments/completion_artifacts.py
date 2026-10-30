"""Report-only supplement artifacts; every writer targets a fresh run directory."""
from __future__ import annotations
import csv
from pathlib import Path
from dual2pose.runtime_paths import input_path
from types import SimpleNamespace
import numpy as np
import torch
from dual2pose.experiments import run_dependency_workflow as flow
from dual2pose.experiments.workflow_completion import read, shared_shift_mask, qualitative_window
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL, common13_pose, COMMON13_NAMES

BONES=((12,0),(0,2),(2,11),(12,1),(1,3),(3,10),(12,4),(12,5),(4,5),(4,6),(6,8),(5,7),(7,9))


def finish(directory, **extra):
    files={str(p.relative_to(directory)):flow.sha(p) for p in directory.rglob('*') if p.is_file()}
    flow.atomic_json(directory/'manifest.json',dict(evaluation_protocol=dict(EVALUATION_PROTOCOL),files=files,**extra))


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def pocket_data():
    from dual2pose.pocket3 import evaluate as ep, train as tr
    return tr.load_frames(SimpleNamespace(dataset_root=ep.DEFAULT_ROOT,session='chen_4',frame_height=1080.))


def checkpoints(ctx):
    state=read(Path(ctx['old_root'])/'state.json')
    return {d:Path(state['metadata']['main_'+d]['checkpoint']) for d in ('unity','ski')}


def pocket_shift(ctx):
    from dual2pose.pocket3 import evaluate as ep, train as tr
    out=Path(ctx['root'])/'pocket_shift';out.mkdir(exist_ok=False)
    torch.set_num_threads(4); device=torch.device('cuda:0')
    L,R,pair,gt,usable,anchor=pocket_data()
    starts=ep.build_windows(usable,30,30,anchor);ids=np.stack([np.arange(s,s+30) for s in starts])
    data=tr.gather(L,R,pair,gt,starts,30);ep.validate_targets(data[2],data[3])
    ckpts=checkpoints(ctx);models={d:ep.load_model(p,device) for d,p in ckpts.items()}
    predictions=ep.run_methods(*data[:3],models,device)
    methods=ep.BASE_METHODS+ep.fusion_methods(list(models))
    shifts=list(range(-4,5));mask=shared_shift_mask(ids,gt['reliable'],ep.ANCHOR_JOINTS,shifts)
    targets=[]
    for shift in shifts:
        index=np.clip(ids+shift,0,len(gt['joints'])-1)
        raw=gt['joints'][index];rel=gt['reliable'][index]
        raw=np.where(rel[...,None],raw,0).astype(np.float32)
        canon,transform=ep.comparison_canonical(torch.from_numpy(raw))
        canon=common13_pose(canon).numpy()
        mask &= (~transform['degenerate'].numpy())[...,None] & np.isfinite(canon).all(-1)
        targets.append(canon)
    if not mask.any():raise ValueError('No common valid joint-frames across reference shifts')
    rows=[]
    for shift,target in zip(shifts,targets):
        for method in methods:
            pred=predictions[method]
            score,n=ep.masked_mpjpe(pred,target,mask,ep.COMMON13)
            aligned,_=ep.sequence_aligned_mpjpe(pred,target,mask,ep.COMMON13)
            rows.append(dict(reference_shift_frames=shift,method=method,mpjpe_mm=score*1000,
                             window_aligned_mpjpe_mm=aligned*1000,point_count=n))
    with (out/'metrics_mm.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez(out/'evidence.npz',frame_indices=ids,reference_shifts=shifts,common_mask=mask,
             targets=np.stack(targets),**{k:predictions[k] for k in methods})
    plt=pyplot();fig,ax=plt.subplots(figsize=(9,5))
    for method in methods:
        ax.plot(shifts,[r['mpjpe_mm'] for r in rows if r['method']==method],marker='o',label=method)
    ax.set(xlabel='Reference shift (frames; positive = later reference)',ylabel='Direct common-13 MPJPE (mm)')
    ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'reference_shifts.pdf');fig.savefig(out/'reference_shifts.png',dpi=140);plt.close(fig)
    finish(out,shifts=shifts,methods=methods,common_point_count=int(mask.sum()),window_starts=starts,
           selection='Full recording baseline windows; fixed intersection of valid joint-frames across all nine shifts; no model selection',
           checkpoint_sha256={d:flow.sha(p) for d,p in ckpts.items()},
           shift_definition='Positive k compares unchanged prediction at input frame t to reference frame t+k',
           primary_metric='mpjpe_mm; output not recanonicalized', auxiliary_metric='window_aligned_mpjpe_mm')


def person_index(directory):
    files={}
    for p in sorted(directory.glob('*.npz')):
        try:files[int(p.stem.split('_')[1])]=p
        except (ValueError,IndexError):continue
    return files


def load_person(path):
    with np.load(path,allow_pickle=True) as z:
        if 'pred_keypoints_3d' in z.files:
            return np.asarray(z['pred_keypoints_3d'],np.float32),np.asarray(z['pred_keypoints_2d'],np.float32)
        o=z['outputs' if 'outputs' in z.files else 'output'];o=o.item() if o.shape==() else o[0]
        if isinstance(o,(list,tuple)):o=o[0]
        return np.asarray(o['pred_keypoints_3d'],np.float32),np.asarray(o['pred_keypoints_2d'],np.float32)


def mask_heads(rgb,kp):
    """Anonymize the tracked head and other detected faces, using CPU OpenCV."""
    import cv2
    rgb=np.ascontiguousarray(rgb.copy());h,w=rgb.shape[:2];boxes=[]
    eyes=kp[[1,2]];neck=kp[69];c=eyes.mean(0)
    if np.isfinite(eyes).all() and np.isfinite(neck).all():
        r=max(1.4*np.linalg.norm(c-neck),2.5*np.linalg.norm(eyes[0]-eyes[1]),16)
        boxes.append((c[0]-r,c[1]-1.2*r,2*r,2.4*r))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    for filename in ('haarcascade_frontalface_default.xml','haarcascade_profileface.xml'):
        cascade=cv2.CascadeClassifier(cv2.data.haarcascades+filename)
        for flip in (False,True):
            img=cv2.flip(gray,1) if flip else gray
            for x,y,bw,bh in cascade.detectMultiScale(img,1.1,4,minSize=(20,20)):
                if flip:x=w-x-bw
                boxes.append((x-.25*bw,y-.3*bh,1.5*bw,1.6*bh))
    # Conservative person-head masking includes faces missed by the cascades.
    hog=cv2.HOGDescriptor();hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    scale=min(1.,960/w);small=cv2.resize(rgb,None,fx=scale,fy=scale)
    for x,y,bw,bh in hog.detectMultiScale(small,winStride=(8,8),padding=(8,8),scale=1.05)[0]:
        boxes.append(((x-.1*bw)/scale,(y-.04*bh)/scale,1.2*bw/scale,.3*bh/scale))
    for x,y,bw,bh in boxes:
        x0,y0=max(0,int(x)),max(0,int(y));x1,y1=min(w,int(x+bw)),min(h,int(y+bh))
        if x1>x0 and y1>y0:
            rgb[y0:y1,x0:x1]=cv2.resize(cv2.resize(rgb[y0:y1,x0:x1],(8,8)),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
    return rgb,len(boxes)


def realworld(ctx):
    # OpenCV 5 removed cascade/HOG APIs; use isolated pinned renderer dependency.
    import sys
    runtime=Path(ctx['opencv_runtime'])
    if not (runtime/'cv2').is_dir():raise FileNotFoundError(runtime/'cv2')
    sys.path.insert(0,str(runtime))
    import cv2
    if not hasattr(cv2,'CascadeClassifier') or not hasattr(cv2,'HOGDescriptor'):
        raise RuntimeError('Renderer requires the pinned OpenCV 4 runtime')
    from dual2pose.pocket3 import evaluate as ep, train as tr
    from dual2pose.map_config import filter_sam3d_body_kpts
    out=Path(ctx['root'])/'realworld';out.mkdir(exist_ok=False);(out/'frames').mkdir()
    repo=Path(ctx['repo']);old=Path(ctx['old_root']);paper=repo/'paper/ieee_access_20261027'
    device=torch.device('cuda:0');torch.set_num_threads(4)
    frozen_ckpt=checkpoints(ctx)['unity'];frozen=ep.load_model(frozen_ckpt,device)
    person=input_path('/home/kaixu_chen/skiing/data/sam3d_body_results/person')
    rows=[];allposes=[];images=[];input_hashes={}
    for run,prefix,fid,start in [('pro_2','a',1231,1216),('run_4','b',180,165)]:
        seqs=[];pics=[]
        for side in ('left','right'):
            index=person_index(person/run/side);files=[index[i] for i in range(start,start+30)]
            seqs.append(np.stack([filter_sam3d_body_kpts(load_person(p)[0]) for p in files]))
            input_hashes.update({str(p):flow.sha(p) for p in files})
            source=paper/f'evidence/pocket3_chen4/qualitative_frames/{prefix}_{run}_{side}.png'
            pic=cv2.cvtColor(cv2.imread(str(source)),cv2.COLOR_BGR2RGB);pics.append(pic)
            cv2.imwrite(str(out/f'frames/{run}_{side}.png'),cv2.cvtColor(pic,cv2.COLOR_RGB2BGR))
        with torch.no_grad():
            left=frozen.canonicalize(torch.tensor(seqs[0][None],device=device))[0]
            right=frozen.canonicalize(torch.tensor(seqs[1][None],device=device))[0]
            fused,_=frozen.models(left,right)
        poses=np.stack([common13_pose(x)[0].cpu().numpy() for x in [left,right,(left+right)/2,fused]])
        allposes.append(poses);images.append(pics)
        rows.append(dict(run=run,window_start=start,midpoint=fid,model='frozen Unity',checkpoint=str(frozen_ckpt),checkpoint_sha256=flow.sha(frozen_ckpt),rgb='Reused anonymized images of the same archived frame'))
    summary=read(old/'pocket/pocket_forward_unity_seed0/summary.json');start=qualitative_window(summary)
    selected=old/'pocket/pocket_forward_unity_seed0/selected.pt'
    if flow.sha(selected)!=summary['checkpoint_sha256']:raise ValueError('Pocket selected weight mismatch')
    trained=ep.load_model(selected,device);L,R,pair,gt,usable,anchor=pocket_data()
    data=tr.gather(L,R,pair,gt,[start],30);ep.validate_targets(data[2],data[3])
    pred=ep.run_methods(*data[:3],{'trained':trained,'frozen':frozen},device)
    keys=['left','right','canonical_average','canonfuse3d[trained]']
    allposes.append(np.stack([pred[k][0] for k in keys]));pics=[];mid=start+15
    for side,cam,frame in [('left','pocket_1',mid),('right','pocket_2',int(pair[mid]))]:
        cap=cv2.VideoCapture(str(ep.DEFAULT_ROOT/f'sessions/chen_4/pocket/{cam}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,bgr=cap.read();cap.release()
        if not ok:raise ValueError('Missing Pocket RGB frame')
        rawfile=ep.DEFAULT_ROOT/f'sam3d_body_results/chen_4/{cam}/{frame:06d}_sam3d_body.npz'
        kp=load_person(rawfile)[1];rgb,_=mask_heads(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),kp);pics.append(rgb)
        cv2.imwrite(str(out/f'frames/chen_4_{side}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        input_hashes[str(rawfile)]=flow.sha(rawfile)
    images.append(pics)
    errors={k:ep.masked_mpjpe(pred[k],pred['gt'],data[3],ep.COMMON13)[0]*1000 for k in keys+['canonfuse3d[frozen]']}
    rows.append(dict(run='chen_4',window_start=start,midpoint=mid,right_frame=int(pair[mid]),
        test_block=summary['blocks']['test'],selection='Middle current test window, forward Unity initialization seed0, chosen without scores',
        checkpoint=str(selected),checkpoint_sha256=flow.sha(selected),window_direct_mpjpe_mm=errors,
        rgb='Tracked head, Haar frontal/profile faces and HOG person-head regions mosaicked'))
    poses=np.stack(allposes)*1000;np.savez(out/'poses_mm.npz',poses=poses)
    plt=pyplot();fig=plt.figure(figsize=(16,10.5));grid=fig.add_gridspec(3,6,width_ratios=[1.25,1.25,1,1,1,1],hspace=.65,wspace=.18)
    displayed=poses[:,:,15,:][:,:,:,[0,2,1]];pts=displayed.reshape(-1,3)
    center=(pts.max(0)+pts.min(0))/2;radius=max(float(np.ptp(pts,axis=0).max())*.56,500)
    titles=['Left RGB','Right RGB','Left 3D','Right 3D','Canonical average','CanonFuse3D']
    for r in range(3):
        for c in range(6):
            ax=fig.add_subplot(grid[r,c],projection='3d' if c>=2 else None)
            if c<2:ax.imshow(images[r][c]);ax.axis('off')
            else:
                p=displayed[r,c-2]
                color=['tab:blue','tab:orange','gray','teal'][c-2]
                for a,b in BONES:ax.plot(*p[[a,b]].T,c=color,lw=1.8)
                ax.scatter(*p.T,s=10,c=color)
                ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),xlabel='X (mm)',ylabel='Z (mm)',zlabel='Y (mm)')
                ax.set_box_aspect((1,1,1));ax.view_init(elev=15,azim=-72);ax.tick_params(labelsize=6)
                ax.set_xticks([-500,0,500]);ax.set_yticks([-500,0,500]);ax.set_zticks([-500,0,500])
            ax.set_title(titles[c],fontsize=10)
    fig.subplots_adjust(top=.92,bottom=.05,left=.03,right=.98)
    for r in range(3):
        fig.text(.03,grid[r,0].get_position(fig).y1+.035,f"{rows[r]['run']} | frame {rows[r]['midpoint']} | per-frame body coordinates",fontsize=12,weight='bold')
    fig.savefig(out/'realworld_per_frame_mm.pdf');fig.savefig(out/'realworld_per_frame_mm.png',dpi=160);plt.close(fig)
    finish(out,rows=rows,input_sha256=input_hashes,unit='mm',shared_limits={'center_xzy':center.tolist(),'radius':radius},
           interpretation='Y is body up, not gravity. Predictions are displayed without output recanonicalization or pose adjustment.')


def gate_figures(ctx):
    out=Path(ctx['root'])/'gate_figures';out.mkdir(exist_ok=False);old=Path(ctx['old_root'])
    from dual2pose.eval.evaluate_gate_interventions import CONDITIONS
    plt=pyplot();fig,axes=plt.subplots(2,2,figsize=(14,8));region=[];metric_rows=[]
    names=list(COMMON13_NAMES)
    for r,(dataset,directory) in enumerate([('unity',old/'evaluations/eval_gate_interventions'),('ski',Path(ctx['root'])/'ski_gate')]):
        for condition in CONDITIONS:
            d=read(directory/(condition+'.json'));diag=d['gate_diagnostics']
            means=np.asarray(diag['alpha_mean_by_time_joint']).mean(0)
            axes[r,1].plot(range(13),means,marker='.',label=condition)
            for method,m in d['metrics'].items():
                metric_rows.append(dict(dataset=dataset,condition=condition,method=method,**{k+'_mm':m['common13'][k]*1000 for k in ('mpjpe','pa_mpjpe','acceleration_error')}))
            if condition=='clean':
                edges=diag['histogram_edges'];axes[r,0].bar(edges[:-1],diag['histogram_counts'],width=.05,align='edge')
                for name,ix in [('upper_limbs',[0,1,2,3,10,11]),('lower_limbs',[4,5,6,7,8,9]),('common13',list(range(13)))]:
                    region.append(dict(dataset=dataset,region=name,joints=len(ix),alpha_mean=float(means[ix].mean()),alpha_std_across_joint_means=float(means[ix].std()),right_mean=float(1-means[ix].mean())))
        axes[r,0].set(title=dataset+' clean gate',xlabel='Left-candidate weight',ylabel='Joint-frame count')
        axes[r,1].set(title=dataset+' per-joint gate',ylim=(0,1),ylabel='Mean left-candidate weight')
        axes[r,1].set_xticks(range(13),names,rotation=45,ha='right');axes[r,1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'gate_diagnostics.pdf');fig.savefig(out/'gate_diagnostics.png',dpi=140);plt.close(fig)
    for filename,rows in [('regions.csv',region),('interventions_mm.csv',metric_rows)]:
        with (out/filename).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Fixed first two exemplars for both datasets, no accuracy-based selection.
    fig,axes=plt.subplots(4,5,figsize=(16,9))
    for r,(dataset,directory) in enumerate([('unity',old/'evaluations/eval_gate_interventions'),('ski',Path(ctx['root'])/'ski_gate')]):
        for c,condition in enumerate(CONDITIONS):
            ex=read(directory/(condition+'.json'))['fixed_first_two_exemplars'];a=np.asarray(ex['alpha'])[...,-13:,0]
            for s in range(2):
                ax=axes[2*r+s,c];im=ax.imshow(a[s],aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.axhline(14.5,color='white',ls='--',lw=.6)
                ax.set_title(f'{dataset} #{s} {condition}',fontsize=8);ax.set_xticks(range(13),names,rotation=90,fontsize=6)
    fig.tight_layout();fig.savefig(out/'gate_exemplars.pdf');plt.close(fig)
    finish(out,datasets=['unity','ski'],joint_subset='common13',region_std='population SD across per-joint means; not seed uncertainty')
