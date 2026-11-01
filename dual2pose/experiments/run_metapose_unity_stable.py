"""Unity MetaPose stability adaptation: train-only pilot, guarded fit, frozen export.

Keeps official modules and 2D MSE; depth health is an input-consistency rejection
rule, never a GT3D objective or a test-time correction. Existing runners unchanged.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from dual2pose.experiments.run_metapose import OfficialMetaPose, load_split, write_json, SOURCE_COMMIT
from dual2pose.experiments.run_metapose_stable import check_source
from dual2pose.eval.metapose_inputs import sha256, fit_point_uncertainty, point_mixtures, restore_left_gauge
from dual2pose.eval.metapose_unity_depth import DepthHealth, POLICY

TRAIN_FRAMES=5799600
VAL_FRAMES=3866400
TEST_FRAMES=1933200
PILOT_FRAMES=32768


def validate_request(*,pilot,epochs,max_frames,val_dir,pilot_run):
    if pilot:
        if val_dir is not None or pilot_run is not None or epochs!=3 or max_frames!=PILOT_FRAMES:
            raise ValueError('Pilot requires only train,32768 frames,3 epochs, no validation or prior fit')
    elif epochs!=10 or max_frames is not None or val_dir is None or pilot_run is None:
        raise ValueError('Full Unity requires10 epochs,full train,validation and passed pilot')


def validate_records(config,report,history,allow_pilot=False):
    pilot=config['pilot'];epochs=3 if pilot else 10;frames=PILOT_FRAMES if pilot else TRAIN_FRAMES
    if pilot and not allow_pilot:raise ValueError('Pilot cannot be exported')
    fixed=dict(epochs=epochs,stages=3,batch_size=1024,seed=42,learning_rate=1e-5,
               gradient_clip_norm=1.,identity_stage_initialization=True,deterministic_ops=True,tf32_disabled=True,
               policy=POLICY,max_frames=PILOT_FRAMES if pilot else None)
    if any(config.get(k)!=v for k,v in fixed.items()):raise ValueError('Changed stability recipe')
    if (report.get('status')!='complete_train_quality_passed' or report.get('pilot')!=pilot
        or report.get('dataset')!='unity' or report.get('train_frames')!=frames
        or report.get('val_frames')!=(0 if pilot else VAL_FRAMES)
        or report.get('test_data_opened') is not False or report.get('val_data_opened')!= (not pilot)):
        raise ValueError('Incomplete fit or invalid split boundary')
    if [(r['stage'],r['epoch']) for r in history]!=[(s,e) for s in range(1,4) for e in range(1,epochs+1)]:
        raise ValueError('Incomplete training history')
    selected=report['selected_stages']
    if [r['stage'] for r in selected]!=[1,2,3]:raise ValueError('Missing selected stages')
    for r in history:
        q=r['quality']
        if not q['passed'] or q['frames']!=frames or q['policy']!=POLICY:raise ValueError('Failed/incomplete training health')
        if not np.isfinite(r['full_train_2d_mse']) or r['full_train_2d_mse']<0:raise ValueError('Invalid train loss')
        if not np.isfinite(r['max_clipped_gradient_norm']) or r['max_clipped_gradient_norm']>1.000001:raise ValueError('Gradient clip failed')
        if pilot:
            if r['val_2d_mse'] is not None:raise ValueError('Pilot accessed validation')
        elif r['val_2d_mse'] is None or not np.isfinite(r['val_2d_mse']) or r['val_2d_mse']<0:raise ValueError('Incomplete validation')
    for s in selected:
        rows=[r for r in history if r['stage']==s['stage']]
        expected=epochs if pilot else min(rows,key=lambda r:r['val_2d_mse'])['epoch']
        if s['epoch']!=expected:raise ValueError('Wrong selected epoch')
    if pilot and not history[-1]['full_train_2d_mse']<report['initial_train_2d_mse']:
        raise ValueError('Pilot failed to improve training MSE')


def code_hashes():
    root=Path(__file__).resolve().parents[2]
    names=['dual2pose/experiments/run_metapose_unity_stable.py','dual2pose/experiments/run_metapose.py',
           'dual2pose/experiments/run_metapose_stable.py','dual2pose/eval/metapose_unity_depth.py',
           'dual2pose/eval/metapose_inputs.py']
    return {n:sha256(root/n) for n in names}


def verify_input_manifests(cfg):
    if sha256(Path(cfg['train_dir'])/'manifest.json')!=cfg['training_manifest_sha256']:
        raise ValueError('Changed training manifest')
    if not cfg['pilot'] and sha256(Path(cfg['val_dir'])/'manifest.json')!=cfg['validation_manifest_sha256']:
        raise ValueError('Changed validation manifest')


def verify_fit(path,allow_pilot=False):
    path=Path(path);report=json.loads((path/'training_report.json').read_text())
    for name,digest in report['artifacts'].items():
        if sha256(path/name)!=digest:raise ValueError('Changed artifact: '+name)
    cfg=json.loads((path/'config.json').read_text())
    hist=[json.loads(s) for s in (path/'epochs.jsonl').read_text().splitlines()]
    validate_records(cfg,report,hist,allow_pilot)
    if cfg['code_sha256']!=code_hashes():raise ValueError('Training source changed')
    if report['selected_stages']!=json.loads((path/'selected_stages.json').read_text()):raise ValueError('Selection differs')
    for row in report['selected_stages']:
        if sha256(path/f'stage_{row["stage"]}.weights.h5')!=row['checkpoint_sha256']:raise ValueError('Changed checkpoint')
    verify_input_manifests(cfg)
    return cfg,report


def model_for(cfg):
    m=OfficialMetaPose(Path(cfg['metapose_repo']),cfg['seed'],learning_rate=cfg['learning_rate'],
                       gradient_clip_norm=cfg['gradient_clip_norm'],identity_stages=True)
    m.tf.config.experimental.enable_op_determinism()
    m.tf.config.experimental.enable_tensor_float_32_execution(False)
    return m


def initialize(model,data,output,role,progress):
    x=np.lib.format.open_memmap(output/(role+'_x_init.npy'),mode='w+',dtype=np.float64,shape=(len(data['xyz']),59))
    for start in range(0,len(x),4096):
        x[start:start+4096]=model.initialize(data['xyz'][start:start+4096])
        if start%131072==0:progress('initializing',split=role,frames=min(start+4096,len(x)),total=len(x))
    x.flush();return x


def health(model,data,x,u,initial=False,batch=1024,progress=None):
    guard=DepthHealth();total=0.
    for start in range(0,len(x),batch):
        sl=slice(start,start+batch);ref=model.project(x[sl])
        if initial:state=x[sl];p=ref
        else:
            task=point_mixtures(data['uv'][sl],u).reshape(len(x[sl]),-1)
            state=model.predict(x[sl],task);p=model.project(state)
        guard.update(p,ref,np.logaddexp(0,state[:,51:53]))
        total+=float(np.square(p[...,:2]-data['labels2d'][sl]).sum())
        if progress and start%(batch*256)==0:progress('training_health',frames=min(start+batch,len(x)),total=len(x))
    return total/(len(x)*52),guard.result()


def validate_loss(model,data,x,u,batch=1024,progress=None):
    total=0.
    for start in range(0,len(x),batch):
        sl=slice(start,start+batch);task=point_mixtures(data['uv'][sl],u).reshape(len(x[sl]),-1)
        total+=model.eval_loss(x[sl],task,data['labels2d'][sl])*len(x[sl])
        if progress and start%(batch*256)==0:progress('validation',frames=min(start+batch,len(x)),total=len(x))
    return total/len(x)


def train(args):
    validate_request(pilot=args.pilot,epochs=args.epochs,max_frames=args.max_frames,val_dir=args.val_dir,pilot_run=args.pilot_run)
    output=args.output_dir.resolve()
    if output.exists():raise FileExistsError('Refusing to overwrite '+str(output))
    check_source(args.metapose_repo)
    pilot_report=None
    if not args.pilot:
        pilot_cfg,pilot_report=verify_fit(args.pilot_run,allow_pilot=True)
        if not pilot_cfg['pilot']:raise ValueError('Expected pilot prerequisite')
    data=load_split(args.train_dir,'train')
    if data['manifest']['dataset']!='unity' or len(data['xyz'])!=TRAIN_FRAMES:raise ValueError('Wrong full Unity train split')
    if pilot_report and pilot_cfg['training_manifest_sha256']!=data['manifest_hash']:raise ValueError('Pilot trained different data')
    output.mkdir(parents=True);started=time.monotonic();last=[0.]
    def progress(status,**extra):
        v=dict(status=status,elapsed_seconds=time.monotonic()-started,**extra);write_json(output/'progress.json',v)
        if time.monotonic()-last[0]>30:print(json.dumps(v),flush=True);last[0]=time.monotonic()
    try:
        cfg={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
        cfg.update(stages=3,batch_size=1024,seed=42,learning_rate=1e-5,gradient_clip_norm=1.,
                   identity_stage_initialization=True,deterministic_ops=True,tf32_disabled=True,policy=POLICY,
                   training_manifest_sha256=data['manifest_hash'],code_sha256=code_hashes(),source_commit=SOURCE_COMMIT,
                   method='MetaPose adapted; identity initialization, clipped deterministic training, S1 depth-consistency guard',
                   no_3d_supervision=True,no_supplied_calibration=True,
                   selection='fixed final epoch' if args.pilot else 'minimum full validation 2D MSE; all epochs require train-only health')
        if pilot_report:cfg['pilot_report_sha256']=sha256(args.pilot_run/'training_report.json')
        write_json(output/'config.json',cfg);model=model_for(cfg)
        if args.require_gpu and not model.tf.config.list_physical_devices('GPU'):raise RuntimeError('GPU required')
        write_json(output/'environment.json',dict(tensorflow=model.tf.__version__,deterministic_ops=True,tf32_enabled=model.tf.config.experimental.tensor_float_32_execution_enabled(),
                   devices=[str(x) for x in model.tf.config.list_physical_devices()],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        rng=np.random.default_rng(42);fit_ids=rng.choice(TRAIN_FRAMES,200000,replace=False)
        u=fit_point_uncertainty(data['uv'][fit_ids]-data['labels2d'][fit_ids]);u.update(training_manifest_sha256=data['manifest_hash'],seed=42,fit_indices_sha256=__import__('hashlib').sha256(fit_ids.tobytes()).hexdigest())
        write_json(output/'uncertainty.json',u)
        if args.pilot:
            ids=np.sort(np.random.default_rng(42).choice(TRAIN_FRAMES,PILOT_FRAMES,replace=False));np.save(output/'training_frame_indices.npy',ids)
            data={k:np.array(data[k][ids]) for k in ['xyz','uv','labels2d']}
        x=initialize(model,data,output,'train',progress)
        initial_loss,initial_quality=health(model,data,x,u,initial=True,progress=progress)
        write_json(output/'initial_health.json',dict(train_2d_mse=initial_loss,quality=initial_quality))
        if not initial_quality['passed']:raise ValueError('S1 reference cannot support health checks')
        val=vx=None
        if not args.pilot:
            val=load_split(args.val_dir,'val')
            if val['manifest']['dataset']!='unity' or len(val['xyz'])!=VAL_FRAMES:raise ValueError('Wrong Unity validation split')
            cfg['validation_manifest_sha256']=val['manifest_hash'];write_json(output/'config.json',cfg)
            vx=initialize(model,val,output,'val',progress)
        selected=[];history=[]
        with (output/'epochs.jsonl').open('x',buffering=1) as log:
            for stage in range(1,4):
                model.add_stage();best=float('inf');chosen=None;checkpoint=output/f'stage_{stage}.weights.h5'
                for epoch in range(1,args.epochs+1):
                    begin=time.monotonic();order=rng.permutation(len(x));loss_sum=0.;max_norm=max_clip=0.
                    for start in range(0,len(x),1024):
                        ix=order[start:start+1024];task=point_mixtures(data['uv'][ix],u).reshape(len(ix),-1)
                        loss_sum+=model.train_batch(x[ix],task,data['labels2d'][ix])*len(ix)
                        max_norm=max(max_norm,model.last_gradient_norm);max_clip=max(max_clip,model.last_clipped_gradient_norm)
                        if start%(1024*100)==0:progress('training',stage=stage,epoch=epoch,frames=start+len(ix),total=len(x))
                    train_loss,quality=health(model,data,x,u,progress=progress)
                    # Reject before opening validation data for this epoch or saving a candidate.
                    if not quality['passed']:
                        write_json(output/f'rejected_stage{stage}_epoch{epoch}.json',dict(quality=quality,train_2d_mse=train_loss))
                        raise ValueError(f'Train-only depth/geometry gate failed: stage{stage} epoch{epoch}: {quality["reasons"]}')
                    vl=validate_loss(model,val,vx,u,progress=progress) if val is not None else None
                    if vl is None or vl<best:
                        model.save_stage(checkpoint);best=vl if vl is not None else train_loss;chosen=epoch
                    row=dict(stage=stage,epoch=epoch,train_2d_mse=loss_sum/len(x),full_train_2d_mse=train_loss,val_2d_mse=vl,
                             quality=quality,max_gradient_norm=max_norm,max_clipped_gradient_norm=max_clip,selected_epoch=chosen,seconds=time.monotonic()-begin)
                    log.write(json.dumps(row,allow_nan=False)+'\n');history.append(row);progress('epoch_complete',**row)
                    print(json.dumps(row),flush=True)
                model.load_stage(checkpoint)
                sl=slice(0,min(128,len(x)));task=point_mixtures(data['uv'][sl],u).reshape(len(x[sl]),-1)
                before=model.predict(x[sl],task);model.load_stage(checkpoint)
                if not np.array_equal(before,model.predict(x[sl],task)):raise ValueError('Strict selected reload changed predictions')
                selected.append(dict(stage=stage,epoch=chosen,checkpoint_sha256=sha256(checkpoint)))
                write_json(output/'selected_stages.json',selected)
        # Reconstruct a fresh stack, using only fixed training inputs.
        sl=slice(0,min(128,len(x)));task=point_mixtures(data['uv'][sl],u).reshape(len(x[sl]),-1)
        expected=model.predict(x[sl],task);fresh=model_for(cfg)
        for stage in range(1,4):fresh.add_stage();fresh.load_stage(output/f'stage_{stage}.weights.h5')
        actual=fresh.predict(x[sl],task);reload_delta=float(np.max(np.abs(expected-actual)))
        if not np.array_equal(expected,actual):raise ValueError('Fresh selected stack differs from trained stack')
        write_json(output/'reload_check.json',dict(passed=True,split='train',frames=len(expected),max_abs_state_delta=reload_delta,fresh_stack=True))
        verify_input_manifests(cfg)
        report=dict(status='complete_train_quality_passed',dataset='unity',pilot=args.pilot,train_frames=len(x),
                    val_frames=0 if val is None else len(vx),test_data_opened=False,val_data_opened=val is not None,
                    initial_train_2d_mse=initial_loss,selected_stages=selected,total_seconds=time.monotonic()-started)
        validate_records(cfg,report,history,allow_pilot=args.pilot)
        names=['config.json','uncertainty.json','epochs.jsonl','selected_stages.json','initial_health.json','reload_check.json']+[f'stage_{s}.weights.h5' for s in (1,2,3)]
        if args.pilot:names.append('training_frame_indices.npy')
        report['artifacts']={n:sha256(output/n) for n in names}
        write_json(output/'training_report.json',report);progress('complete_train_quality_passed',pilot=args.pilot)
        return report
    except BaseException as error:
        write_json(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));progress('failed',error=repr(error));raise


def export(args):
    cfg,fit=verify_fit(args.train_run);check_source(cfg['metapose_repo'])
    output=args.output_dir.resolve()
    if output.exists():raise FileExistsError('Refusing to overwrite export')
    data=load_split(args.test_dir,'test')
    if data['manifest']['dataset']!='unity' or len(data['xyz'])!=TEST_FRAMES:raise ValueError('Wrong full test cohort')
    output.mkdir(parents=True);model=model_for(cfg);u=json.loads((args.train_run/'uncertainty.json').read_text())
    for stage in range(1,4):model.add_stage();model.load_stage(args.train_run/f'stage_{stage}.weights.h5')
    arrays={n:np.lib.format.open_memmap(output/(n+'_raw_left.npy'),mode='w+',dtype=np.float32,shape=(TEST_FRAMES//30,30,13,3)) for n in ['s1','metapose']};started=time.monotonic()
    for start in range(0,TEST_FRAMES,1024):
        sl=slice(start,min(start+1024,TEST_FRAMES));x=model.initialize(data['xyz'][sl]);task=point_mixtures(data['uv'][sl],u).reshape(len(x),-1)
        for name,state in [('s1',x),('metapose',model.predict(x,task))]:
            raw=restore_left_gauge(model.project(state)[:,0],data['q'][sl,0],data['t'][sl,0],data['native_left'][sl]);arrays[name].reshape(-1,13,3)[sl]=raw
        if start%(1024*100)==0:write_json(output/'progress.json',dict(status='exporting',frames=sl.stop,total=TEST_FRAMES,seconds=time.monotonic()-started))
    for value in arrays.values():value.flush()
    verify_fit(args.train_run)
    write_json(output/'report.json',dict(status='complete_predictions_not_yet_scored',dataset='unity',windows=TEST_FRAMES//30,
        frames=TEST_FRAMES,finite_coverage=1.,native_test_cache=data['manifest']['native_cache'],
        native_test_manifest_sha256=data['manifest']['native_manifest_sha256'],
        predictions={n+'_raw_left.npy':sha256(output/(n+'_raw_left.npy')) for n in arrays},
        train_run=str(args.train_run.resolve()),training_report_sha256=sha256(args.train_run/'training_report.json'),
        training_quality_passed=True,selected_stages=fit['selected_stages'],seed=42,inference_seconds=time.monotonic()-started,
        method=cfg['method'],protocol='common13_direct_per_frame_v1'))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for k in ['train-dir','output-dir','metapose-repo']:t.add_argument('--'+k,type=Path,required=True)
    for k in ['val-dir','pilot-run']:t.add_argument('--'+k,type=Path)
    t.add_argument('--pilot',action='store_true');t.add_argument('--epochs',type=int,default=10)
    t.add_argument('--max-frames',type=int);t.add_argument('--require-gpu',action='store_true')
    e=sub.add_parser('export')
    for k in ['train-run','test-dir','output-dir']:e.add_argument('--'+k,type=Path,required=True)
    return p


if __name__=='__main__':
    a=parser().parse_args()
    if a.command=='train':train(a)
    else:export(a)
