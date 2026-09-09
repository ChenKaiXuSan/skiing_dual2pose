"""Recompute IVC main baselines without changing archived experiments."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from dual2pose.eval.main_baseline_data import ROOT, batch_slices, canonical_batch, load_cache, sha256, write_json

RUNNER_SHA256 = sha256(__file__)

CHECKPOINTS = {
 'unity': ROOT/'logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt',
 'ski': ROOT/'logs/train_ski_poseptz/crossview_fusion/2026-05-25/14-13-23/checkpoints/last.ckpt',
}

class MetricAccumulator:
    def __init__(self):
        self.distance=0.; self.accel=0.; self.points=0; self.accel_points=0; self.samples=0
    def update(self,pred,target):
        if pred.shape != target.shape or not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError('Invalid predictions or targets')
        error=torch.linalg.vector_norm(pred-target,dim=-1)
        self.distance+=error.double().sum().item(); self.points+=error.numel(); self.samples+=len(pred)
        pa=pred[:,2:]-2*pred[:,1:-1]+pred[:,:-2]
        ta=target[:,2:]-2*target[:,1:-1]+target[:,:-2]
        ae=torch.linalg.vector_norm(pa-ta,dim=-1)
        self.accel+=ae.double().sum().item(); self.accel_points+=ae.numel()
    def result(self):
        if not self.points or not self.accel_points: raise ValueError('Empty metric accumulator')
        return {'mpjpe':self.distance/self.points,'acceleration_error':self.accel/self.accel_points,
                'sample_count':self.samples,'point_count':self.points,'acceleration_point_count':self.accel_points}

class ReferenceModel(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        from dual2pose.models.crossview_fusion import CrossViewCanonicalFusion
        self.model=CrossViewCanonicalFusion(num_heads=4)
        checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state={k[len('models.'):]:v for k,v in checkpoint['state_dict'].items() if k.startswith('models.')}
        self.model.load_state_dict(state,strict=True)
    def forward(self,left,right): return self.model(left,right)[0]

class FunctionModel(nn.Module):
    def __init__(self,function): super().__init__(); self.function=function
    def forward(self,left,right): return self.function(left,right)

@torch.no_grad()
def evaluate(model,arrays,dataset,batch_size):
    model.eval()
    stats={key:MetricAccumulator() for key in (('all15','common13') if dataset=='unity' else ('common13',))}
    for start,stop in batch_slices(len(arrays[0]),batch_size):
        left,right,target=canonical_batch(*(x[start:stop] for x in arrays),dataset=dataset)
        prediction=model(left,right)
        for subset,metric in stats.items():
            idx=slice(2,None) if subset=='common13' and dataset=='unity' else slice(None)
            metric.update(prediction[:,:,idx],target[:,:,idx])
    return {name:metric.result() for name,metric in stats.items()}

def provenance(args,manifest):
    return {'dataset':args.dataset,'seed':args.seed,'test_batch_size':256 if args.dataset=='unity' else 4,
            'time_window':30,'drop_last':False,'input_group':'3D poses only',
            'units':'dataset_coordinate_units','test_cache':manifest,'command':sys.argv,
            'runner_sha256':RUNNER_SHA256,'torch_version':torch.__version__,
            'canonicalization':'independent left/right/target, first sample first frame of each batch'}

def run_evaluation(args):
    arrays,manifest=load_cache(args.root/'cache'/args.dataset/'test',args.device)
    output=args.root/'results'/args.dataset; output.mkdir(parents=True,exist_ok=True)
    batch_size=256 if args.dataset=='unity' else 4
    if args.mode=='reference':
        methods={'left_canonical':FunctionModel(lambda l,r:l),'right_canonical':FunctionModel(lambda l,r:r),
                 'canonical_avg':FunctionModel(lambda l,r:(l+r)/2),
                 'canonfuse3d':ReferenceModel(CHECKPOINTS[args.dataset]).to(args.device)}
    else:
        from dual2pose.models.main_baselines import aligned_average, quality_weighted_fusion
        methods={'aligned_average':FunctionModel(aligned_average),
                 'quality_weighted':FunctionModel(quality_weighted_fusion)}
    for name,model in methods.items():
        path=output/f'{name}.json'
        if path.exists(): raise FileExistsError(f'Preserving existing result: {path}')
        start=time.monotonic(); metrics=evaluate(model,arrays,args.dataset,batch_size)
        record={'method':name,'metrics':metrics,'provenance':provenance(args,manifest),
                'elapsed_seconds':time.monotonic()-start}
        if name=='canonfuse3d':
            record['checkpoint']=str(CHECKPOINTS[args.dataset]); record['checkpoint_sha256']=sha256(CHECKPOINTS[args.dataset])
        if args.mode=='heuristics': record['model_code_sha256']=sha256(ROOT/'dual2pose/models/main_baselines.py')
        write_json(path,record); print(json.dumps(record),flush=True)

def _save_checkpoint(path,model,optimizer,epoch,best,config):
    temporary=path.with_suffix('.tmp')
    torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                'best_validation_mpjpe':best,'config':config,'rng_state':torch.get_rng_state(),
                'cuda_rng_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},temporary)
    temporary.replace(path)

def validate_resume_config(saved, current):
    # Command spelling may change, but the data, code, objective, optimizer,
    # batches, precision, device and selection protocol must remain identical.
    for key in sorted((set(saved) | set(current)) - {"command"}):
        if saved.get(key) != current.get(key):
            raise RuntimeError(f'Resume configuration mismatch: {key}')

def train(args):
    from dual2pose.models.main_baselines import build_baseline
    output=args.root/'training'/args.dataset/args.method
    output.mkdir(parents=True,exist_ok=True)
    result_path=args.root/'results'/args.dataset/f'{args.method}.json'
    if result_path.exists(): raise FileExistsError(f'Preserving completed run: {result_path}')
    train_arrays,train_manifest=load_cache(args.root/'cache'/args.dataset/'train',args.device)
    val_arrays,val_manifest=load_cache(args.root/'cache'/args.dataset/'val',args.device) if args.dataset=='unity' else (None,None)
    batch_size=4096 if args.dataset=='unity' else 4
    model=build_baseline(args.method,num_joints=train_arrays[0].shape[2],time_window=30).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    model_source=ROOT/'dual2pose/models/main_baselines.py'
    config={'dataset':args.dataset,'method':args.method,'seed':args.seed,'epochs':args.epochs,
            'training_batch_size':batch_size,'microbatch_size':args.microbatch or batch_size,
            'train_drop_last':True,'learning_rate':.001,'weight_decay':.01,
            'validation_batch_size':4096 if args.dataset=='unity' else None,
            'selection':'minimum validation MPJPE' if args.dataset=='unity' else 'fixed final epoch',
            'loss':'L1 position + 0.01 predicted acceleration norm',
            'train_cache':train_manifest,'validation_cache':val_manifest,
            'model_code_sha256':sha256(model_source),'runner_sha256':RUNNER_SHA256,
            'parameters':sum(p.numel() for p in model.parameters()),'device':args.device,
            'precision':'float32; TF32 matmul disabled','command':sys.argv}
    if args.method=='smoothnet': config['loss']='L1 position + 0.1 L1 acceleration error (official SmoothNet objective)'
    first_epoch=0; best=float('inf'); config_path=output/'config.json'
    if (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location=args.device,weights_only=False)
        validate_resume_config(saved['config'],config)
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        first_epoch=saved['epoch']+1; best=saved['best_validation_mpjpe']
        torch.set_rng_state(saved['rng_state'].cpu())
        if saved['cuda_rng_state']: torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng_state']])
    elif config_path.exists():
        previous=json.loads(config_path.read_text())
        validate_resume_config(previous,config)
    if not config_path.exists(): write_json(config_path,config)
    started=time.monotonic()
    with (output/'history.jsonl').open('a') as history:
        for epoch in range(first_epoch,args.epochs):
            epoch_start=time.monotonic(); model.train()
            generator=torch.Generator().manual_seed(args.seed+epoch)
            order=torch.randperm(len(train_arrays[0]),generator=generator,device='cpu').to(args.device)
            loss_sum=0.; trained=0
            for start,stop in batch_slices(len(order),batch_size,drop_last=True):
                indices=order[start:stop]
                left,right,target=canonical_batch(*(x[indices] for x in train_arrays),dataset=args.dataset)
                optimizer.zero_grad(set_to_none=True)
                for a,b in batch_slices(len(left),args.microbatch or batch_size):
                    pred=model(left[a:b],right[a:b]); truth=target[a:b]
                    loss=torch.nn.functional.l1_loss(pred,truth)
                    acc=pred[:,2:]-2*pred[:,1:-1]+pred[:,:-2]
                    if args.method=='smoothnet':
                        gt_acc=truth[:,2:]-2*truth[:,1:-1]+truth[:,:-2]
                        loss=loss+.1*torch.nn.functional.l1_loss(acc,gt_acc)
                    else: loss=loss+.01*torch.linalg.vector_norm(acc,dim=-1).mean()
                    if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite training loss at epoch {epoch}')
                    (loss*((b-a)/len(left))).backward()
                    loss_sum+=loss.item()*(b-a)
                optimizer.step(); trained+=len(left)
            metrics=evaluate(model,val_arrays,args.dataset,4096) if val_arrays is not None else None
            value=metrics['all15']['mpjpe'] if metrics else None
            if value is not None and value<best:
                best=value; _save_checkpoint(output/'best.pt',model,optimizer,epoch,best,config)
            _save_checkpoint(output/'last.pt',model,optimizer,epoch,best,config)
            entry={'epoch':epoch,'training_samples':trained,'training_loss':loss_sum/trained,
                   'validation':metrics,'best_validation_mpjpe':None if value is None else best,
                   'epoch_seconds':time.monotonic()-epoch_start,'elapsed_seconds':time.monotonic()-started}
            history.write(json.dumps(entry,allow_nan=False)+'\n'); history.flush()
            write_json(output/'progress.json',{'pid':os.getpid(),'status':'training',**entry})
            print(json.dumps(entry,allow_nan=False),flush=True)
    selected=output/('best.pt' if args.dataset=='unity' else 'last.pt')
    checkpoint=torch.load(selected,map_location=args.device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    del train_arrays,val_arrays,optimizer
    arrays,manifest=load_cache(args.root/'cache'/args.dataset/'test',args.device)
    metrics=evaluate(model,arrays,args.dataset,256 if args.dataset=='unity' else 4)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    record={'method':args.method,'metrics':metrics,'provenance':provenance(args,manifest),
            'training':config,'selected_epoch':checkpoint['epoch'],'checkpoint':str(selected),
            'checkpoint_sha256':sha256(selected),'total_seconds':time.monotonic()-started}
    write_json(result_path,record)
    write_json(output/'progress.json',{'pid':os.getpid(),'status':'complete','result':str(result_path)})
    print(json.dumps(record),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',choices=['unity','ski'],required=True)
    p.add_argument('--mode',choices=['reference','heuristics','train'],required=True)
    p.add_argument('--method',choices=['mlp','tcn','smoothnet'])
    p.add_argument('--device',default='cuda:1'); p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--microbatch',type=int)
    args=p.parse_args(); args.root=args.root.resolve()
    if args.mode=='train' and args.method is None: p.error('--method is required for training')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.mode=='train': train(args)
    else: run_evaluation(args)

if __name__=='__main__': main()
