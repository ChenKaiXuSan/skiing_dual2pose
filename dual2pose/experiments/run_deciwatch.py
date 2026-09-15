"""Run DeciWatch with the official H36M-FCN-3D training protocol.

The only temporal adapter repeats the final archived frame once, runs the
unmodified official model on 31 frames, and crops predictions back to 30.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dual2pose.eval.main_baseline_data import EXPECTED, batch_slices, sha256, write_json
from dual2pose.eval.pa_mpjpe import PAMetricAccumulator, PA_PROTOCOL
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


def _official(repo):
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from lib.models.deciwatch import DeciWatch
    return DeciWatch


def _pad_for_official_window(sequence: torch.Tensor, interval: int):
    """Repeat the final frame until ``(length - 1) % interval == 0``."""
    if sequence.ndim != 3 or sequence.shape[1] < 2 or interval <= 0:
        raise ValueError('Expected a sequence (B,L,C), L >= 2, and a positive interval')
    original_length = sequence.shape[1]
    remainder = (original_length - 1) % interval
    if remainder == 0:
        return sequence, original_length
    pad_length = interval - remainder
    return torch.cat((sequence, sequence[:, -1:].expand(-1, pad_length, -1)), dim=1), original_length


def _masked_l1(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    keep = (~mask.bool()).unsqueeze(-1).expand_as(inputs)
    return torch.nn.functional.l1_loss(
        inputs * keep, targets * keep, reduction='sum'
    ) / keep.sum(dtype=torch.float32)


def _official_l1_loss(recovered, denoised, target, encoder_mask, decoder_mask):
    return _masked_l1(denoised, target, encoder_mask) + _masked_l1(
        recovered, target, decoder_mask
    )


def _make_optimizer(parameters, lr):
    return torch.optim.Adam(parameters, lr=lr, amsgrad=True)


def _should_save_checkpoint(*, has_validation, epoch, epochs, criterion, best):
    return criterion < best if has_validation else epoch == epochs




class DeciWatchWindow(nn.Module):
    """Unmodified official network behind a repeat-last/crop boundary adapter."""
    def __init__(self, repo, input_dim, interval=10, hidden=128, layers=5):
        super().__init__()
        if interval <= 0:
            raise ValueError('interval must be positive')
        DeciWatch = _official(repo)
        self.inner = DeciWatch(input_dim=input_dim, sample_interval=interval,
                               encoder_hidden_dim=hidden, decoder_hidden_dim=hidden,
                               dropout=.1, nheads=4, dim_feedforward=256,
                               enc_layers=layers, dec_layers=layers,
                               activation='leaky_relu', pre_norm=True,
                               recovernet_interp_method='linear', recovernet_mode='transformer')
        official_mask = self.inner.generate_unifrom_mask

        def bool_compatible_mask(length, sample_interval=interval):
            encoder, decoder = official_mask(length, sample_interval)
            return encoder.bool(), decoder.bool()
        self.inner.generate_unifrom_mask = bool_compatible_mask
        self.interval = interval
        self.last_observed_indices = torch.empty(0,dtype=torch.long)
        self.last_boundary_policy = 'repeat_last_to_31_then_crop'

    def forward_padded(self, sequence, device=None, target=None):
        if sequence.ndim != 3 or sequence.shape[1] < 2 or sequence.shape[2] != self.inner.deciwatch_par['input_dim']:
            raise ValueError(f'Expected (B,L,{self.inner.deciwatch_par["input_dim"]}), got {tuple(sequence.shape)}')
        device = device or sequence.device
        sequence = sequence.to(device)
        padded, original_length = _pad_for_official_window(sequence, self.interval)
        recovered, denoised = self.inner(padded, device)
        self.encoder_mask = self.inner.encoder_mask
        self.decoder_mask = self.inner.decoder_mask
        self.input_seq_interp = self.inner.input_seq_interp
        sample_positions = (~self.encoder_mask[0].bool()).nonzero(as_tuple=False).flatten()
        self.last_observed_indices = sample_positions.detach().cpu()
        self.last_original_length = original_length
        if target is None:
            return recovered, denoised, None
        if target.shape != sequence.shape:
            raise ValueError(f'Target shape {tuple(target.shape)} does not match input {tuple(sequence.shape)}')
        padded_target, _ = _pad_for_official_window(target.to(device), self.interval)
        return recovered, denoised, padded_target

    def forward(self, sequence, device=None):
        recovered, denoised, _ = self.forward_padded(sequence, device)
        return recovered[:, :self.last_original_length], denoised[:, :self.last_original_length]


class WindowDataset(Dataset):
    def __init__(self,input_path,target_path=None):
        self.x=np.load(input_path,mmap_mode='r',allow_pickle=False)
        self.y=np.load(target_path,mmap_mode='r',allow_pickle=False) if target_path else None
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        x=torch.from_numpy(np.array(self.x[i],copy=True)).reshape(30,-1)
        return (x,torch.from_numpy(np.array(self.y[i],copy=True)).reshape(30,-1)) if self.y is not None else x


def _canonical_average(cache, output, include_target):
    cache,output=Path(cache),Path(output)
    m=json.loads((cache/'manifest.json').read_text()); ds=m['dataset']; split=m['split']; n=m['sample_count']
    if ds not in EXPECTED or split not in EXPECTED[ds] or n != EXPECTED[ds][split]: raise ValueError('Unexpected split count')
    if m['representation']!='raw_pre_batch_canonicalization' or m['time_window']!=30: raise ValueError('Unexpected cache protocol')
    for name in ('left.npy','right.npy','frames.npy','samples.jsonl'):
        if sha256(cache/name)!=m['files'][name]: raise ValueError(f'cache checksum mismatch: {name}')
    left=np.load(cache/'left.npy',mmap_mode='r'); right=np.load(cache/'right.npy',mmap_mode='r')
    target=np.load(cache/'target.npy',mmap_mode='r') if include_target else None
    if target is not None and sha256(cache/'target.npy')!=m['files']['target.npy']: raise ValueError('target checksum mismatch')
    x=np.lib.format.open_memmap(output/'input.npy',mode='w+',dtype=np.float32,shape=(n,30,13,3))
    y=np.lib.format.open_memmap(output/'target.npy',mode='w+',dtype=np.float32,shape=(n,30,13,3)) if include_target else None
    bs=256 if ds=='unity' else 4; kw={} if ds=='unity' else dict(left_hip=4,right_hip=5,neck=12)
    for start,stop in batch_slices(n,bs):
        l,r=canonicalize_pose_torch(torch.tensor(np.array(left[start:stop])),**kw)[0],canonicalize_pose_torch(torch.tensor(np.array(right[start:stop])),**kw)[0]
        x[start:stop]=((l[:,:,2:15] if ds=='unity' else l)+(r[:,:,2:15] if ds=='unity' else r)).mul(.5).numpy()
        if y is not None:
            t=canonicalize_pose_torch(torch.tensor(np.array(target[start:stop])),**kw)[0]
            y[start:stop]=(t[:,:,2:15] if ds=='unity' else t).numpy()
    x.flush();
    if y is not None:y.flush()
    return dict(dataset=ds,split=split,sample_count=n,frames=n*30,input_sha256=sha256(output/'input.npy'),
                target_sha256=sha256(output/'target.npy') if y is not None else None,
                cache_manifest_sha256=sha256(cache/'manifest.json'),batch_size=bs,
                canonicalization='independent archived left/right/target batch transforms; average common13',
                target_loaded=include_target)


@torch.no_grad()
def _loss_epoch(model, loader, device):
    model.eval(); total=points=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        out,den,padded_target=model.forward_padded(x,device,target=y)
        value=_official_l1_loss(out,den,padded_target,model.encoder_mask,model.decoder_mask)
        total+=float(value)*len(x); points+=len(x)
    return total/points


def _run(args):
    out=Path(args.output_dir).resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    repo=Path(args.deciwatch_repo).resolve()
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit!='5afe2d67005441e2785a8d374d31a91f046a0da6': raise ValueError('Unexpected official DeciWatch commit')
    start=time.time(); config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}; config.update(method='DeciWatch official-protocol adaptation; common13 canonical average input; 4/30 observed source frames',source_commit=commit,seed=args.seed,official_reference_config='config_h36m_fcn_3D.yaml',boundary_policy='repeat last archived frame to length 31; official forward and loss; crop outputs to 30')
    write_json(out/'config.json',config)
    os.environ['PYTHONHASHSEED']=str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.enabled=True
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    prepared=out/'prepared'; prepared.mkdir()
    # Prepare subdirectories explicitly so no existing output is silently reused.
    metas={}
    jobs=[('train',args.train_cache,True),('test',args.test_cache,False)]
    if args.val_cache:
        jobs.insert(1,('val',args.val_cache,True))
    for role,cache,labels in jobs:
        path=prepared/role; path.mkdir()
        metas[role]=_canonical_average(cache,path,labels)
    write_json(out/'prepared_manifests.json',metas)
    model=DeciWatchWindow(repo,39,args.interval,args.hidden,args.layers).to(device)
    opt=_make_optimizer(model.parameters(),lr=args.lr)
    train_loader=DataLoader(WindowDataset(prepared/'train/input.npy',prepared/'train/target.npy'),batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=False)
    val_loader=DataLoader(WindowDataset(prepared/'val/input.npy',prepared/'val/target.npy'),batch_size=args.batch_size,shuffle=False,num_workers=0) if args.val_cache else None
    best=float('inf'); best_epoch=None
    epoch_log=out/'epochs.jsonl';
    with epoch_log.open('x') as log:
        for epoch in range(1,args.epochs+1):
            model.train(); total=points=0; epoch_start=time.time()
            for x,y in train_loader:
                x,y=x.to(device),y.to(device); opt.zero_grad()
                recovered,den,padded_target=model.forward_padded(x,device,target=y)
                loss=_official_l1_loss(recovered,den,padded_target,model.encoder_mask,model.decoder_mask)
                loss.backward(); opt.step()
                total+=float(loss.detach())*len(x); points+=len(x)
            val_loss=_loss_epoch(model,val_loader,device) if val_loader else None
            criterion=val_loss if val_loss is not None else total/points
            if _should_save_checkpoint(has_validation=val_loader is not None,epoch=epoch,epochs=args.epochs,criterion=criterion,best=best):
                best=criterion;best_epoch=epoch;torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'criterion':criterion,'config':config},out/'best.pth')
            rec=dict(epoch=epoch,train_loss=total/points,val_loss=val_loss,selected_epoch=best_epoch,seconds=time.time()-epoch_start)
            log.write(json.dumps(rec)+'\n');log.flush(); print(json.dumps(rec),flush=True)
            for group in opt.param_groups: group['lr']*=args.lr_decay
    checkpoint=torch.load(out/'best.pth',map_location=device,weights_only=False); model.load_state_dict(checkpoint['state_dict'],strict=True); model.eval()
    # Save predictions before opening any test target.
    test_x=np.load(prepared/'test/input.npy',mmap_mode='r'); n=len(test_x)
    pred=np.lib.format.open_memmap(out/'recovered.npy',mode='w+',dtype=np.float32,shape=(n,30,13,3)); den=np.lib.format.open_memmap(out/'denoised.npy',mode='w+',dtype=np.float32,shape=(n,30,13,3))
    loader=DataLoader(WindowDataset(prepared/'test/input.npy'),batch_size=args.batch_size,shuffle=False,num_workers=0)
    offset=0
    with torch.no_grad():
        for x in loader:
            x=x.to(device); a,b=model(x,device); size=len(x); pred[offset:offset+size]=a.cpu().numpy().reshape(size,30,13,3);den[offset:offset+size]=b.cpu().numpy().reshape(size,30,13,3);offset+=size
    pred.flush();den.flush()
    if not np.isfinite(pred).all() or not np.isfinite(den).all(): raise FloatingPointError('Nonfinite DeciWatch output')
    report=dict(status='predictions_fixed',dataset=metas['test']['dataset'],windows=n,frames=n*30,best_epoch=best_epoch,finite_coverage=1.,prediction_sha256=sha256(out/'recovered.npy'),denoised_sha256=sha256(out/'denoised.npy'),checkpoint_sha256=sha256(out/'best.pth'))
    write_json(out/'report.json',report)
    # Evaluation opens targets only after predictions/checksum are fixed.
    cache=Path(args.test_cache); target=np.load(cache/'target.npy',mmap_mode='r'); raw_left=np.load(cache/'left.npy',mmap_mode='r');raw_right=np.load(cache/'right.npy',mmap_mode='r')
    if sha256(cache/'target.npy')!=json.loads((cache/'manifest.json').read_text())['files']['target.npy']: raise ValueError('test target checksum mismatch')
    kwargs={} if report['dataset']=='unity' else dict(left_hip=4,right_hip=5,neck=12); acc={k:PAMetricAccumulator() for k in ('deciwatch','denoised','average_control')}; bs=256 if report['dataset']=='unity' else 4
    for s,e in batch_slices(n,bs):
        l=canonicalize_pose_torch(torch.tensor(np.array(raw_left[s:e])),**kwargs)[0];r=canonicalize_pose_torch(torch.tensor(np.array(raw_right[s:e])),**kwargs)[0];t=canonicalize_pose_torch(torch.tensor(np.array(target[s:e])),**kwargs)[0]
        if report['dataset']=='unity': l,r,t=l[:,:,2:15],r[:,:,2:15],t[:,:,2:15]
        truth=t; acc['deciwatch'].update(torch.tensor(np.array(pred[s:e])),truth);acc['denoised'].update(torch.tensor(np.array(den[s:e])),truth);acc['average_control'].update((l+r)/2,truth)
    metrics={k:v.result() for k,v in acc.items()}
    reference=json.loads(Path(args.control_metrics).read_text())['average_control']
    control_delta={k:metrics['average_control'][k]-reference[k] for k in ('mpjpe','pa_mpjpe','acceleration_error')}
    if any(abs(v)>1e-7 for v in control_delta.values()): raise ValueError(f'Average-control mismatch: {control_delta}')
    metrics.update(pa_protocol=PA_PROTOCOL,method_label=f'DeciWatch (official-protocol adaptation, common13, 4/30 observed source frames, seed {args.seed})',boundary_policy='repeat frame 29 at token 30; official 31-frame forward/loss samples tokens 0/10/20/30; crop outputs to 30',compatibility_shim='official integer attention masks converted to bool for PyTorch 2.11 without changing mask values',selection='minimum validation official loss when validation exists; otherwise fixed final epoch',control_delta=control_delta,control_metrics_sha256=sha256(args.control_metrics),evaluator_sha256=sha256(Path(__file__)))
    write_json(out/'metrics.json',metrics); report.update(status='complete',metrics=metrics,elapsed_seconds=time.time()-start);write_json(out/'report.json',report);print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('deciwatch-repo','train-cache','test-cache','output-dir'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--val-cache',type=Path);p.add_argument('--control-metrics',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--interval',type=int,default=10);p.add_argument('--hidden',type=int,default=128);p.add_argument('--layers',type=int,default=5);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--epochs',type=int,required=True);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--lr-decay',type=float,default=.95);p.add_argument('--seed',type=int,default=4321)
    a=p.parse_args();_run(a)
