"""Quality-gated MetaPose: train-only optimization, then separately frozen export.

The train command has no test-data argument. It never loads 3D labels.
This is a stability adaptation of the official model, not its original trainer.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from dual2pose.experiments.run_metapose import OfficialMetaPose, SOURCE_COMMIT, load_split, write_json
from dual2pose.eval.metapose_inputs import sha256, fit_point_uncertainty, point_mixtures, restore_left_gauge
from dual2pose.eval.metapose_stability import POLICY, geometry_health, training_gate


def require_exportable(report):
    if (report.get('status') != 'complete_train_quality_passed' or report.get('pilot') is not False
            or report.get('frames') != 4350 or report.get('stages') != 3 or report.get('epochs') != 300
            or len(report.get('quality', [])) != 3 or not all(q['passed'] for q in report['quality'])):
        raise ValueError('Only a complete full-budget quality-passed Ski fit can be exported')


def check_source(source):
    rev = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain','--','metapose'],text=True)
    if rev != SOURCE_COMMIT or dirty:
        raise ValueError('Require unmodified pinned official model source')


def model_for(config):
    return OfficialMetaPose(config['metapose_repo'], config['seed'],
                            learning_rate=config['learning_rate'],
                            gradient_clip_norm=config['gradient_clip_norm'], identity_stages=True)


def initialized(model, xyz, batch_size):
    return np.concatenate([model.initialize(xyz[i:i+batch_size]) for i in range(0,len(xyz),batch_size)])


def full_health(model, x, task, data, initial=False, batch_size=128):
    projs, scales, losses = [], [], []
    for start in range(0,len(x),batch_size):
        stop = min(start+batch_size,len(x))
        state = x[start:stop] if initial else model.predict(x[start:stop],task[start:stop])
        projection = model.project(state)
        projs.append(projection)
        scales.append(np.logaddexp(0,state[:,51:53]))
        losses.append(np.square(projection[...,:2]-data['labels2d'][start:stop]).sum())
    geometry = geometry_health(np.concatenate(projs),data['xyz'],np.concatenate(scales))
    loss = float(sum(losses)/(len(x)*52))
    return loss, geometry


def train(args):
    output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    if not args.pilot and (args.epochs != 300 or args.max_frames is not None):
        raise ValueError('Full Ski fit requires 300 epochs and every training frame')
    if args.epochs < 1 or args.batch_size < 1 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError('Training budgets must be positive')
    check_source(args.metapose_repo)
    # load_split verifies role and every input hash before any optimization.
    data = load_split(args.train_dir, 'train')
    if data['manifest']['dataset'] != 'ski': raise ValueError('Stable runner currently validated for Ski only')
    if not args.pilot and len(data['xyz']) != 4350: raise ValueError('Unexpected full Ski training coverage')
    output.mkdir(parents=True)
    started = time.monotonic()
    config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(source_commit=SOURCE_COMMIT, no_3d_supervision=True, no_supplied_calibration=True,
                  identity_stage_initialization=True,
                  selection='fixed final epoch; abort on training-only quality failure',
                  policy=POLICY, training_manifest_sha256=data['manifest_hash'],
                  code_sha256={str(p):sha256(p) for p in (Path(__file__),
                      Path(__file__).with_name('run_metapose.py'),
                      Path(__file__).parents[1]/'eval/metapose_stability.py')})
    write_json(output/'config.json',config)
    try:
        model = model_for(config)
        if args.require_gpu and not model.tf.config.list_physical_devices('GPU'):
            raise RuntimeError('Requested GPU is unavailable')
        model.tf.config.experimental.enable_op_determinism()
        write_json(output/'environment.json', dict(tensorflow=model.tf.__version__,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            devices=[str(d) for d in model.tf.config.list_physical_devices()], deterministic_ops=True))
        rng = np.random.default_rng(args.seed)
        # Uncertainty is fit solely on the training split, before pilot sampling.
        uncertainty = fit_point_uncertainty(data['uv']-data['labels2d'])
        uncertainty['training_manifest_sha256'] = data['manifest_hash']
        write_json(output/'uncertainty.json',uncertainty)
        ids = np.arange(len(data['xyz']))
        if args.max_frames is not None and args.max_frames < len(ids):
            ids = np.sort(rng.choice(ids,args.max_frames,replace=False))
        arrays = {k:np.asarray(data[k][ids]) for k in ('xyz','uv','labels2d')}
        np.save(output/'training_frame_indices.npy',ids)
        x = initialized(model,arrays['xyz'],args.batch_size)
        task = point_mixtures(arrays['uv'],uncertainty).reshape(len(x),-1)
        initial_loss, initial_geometry = full_health(model,x,task,arrays,True,args.batch_size)
        write_json(output/'initial_health.json',dict(loss=initial_loss,geometry=initial_geometry))
        if not initial_geometry['passed']: raise ValueError('Degenerate training initialization')
        selected, quality = [], []
        with (output/'epochs.jsonl').open('x',buffering=1) as log:
            for stage in range(1,4):
                model.add_stage(); best = float('inf')
                for epoch in range(1,args.epochs+1):
                    epoch_started = time.monotonic(); total = 0.; max_norm = 0.; max_clipped = 0.
                    for start in range(0,len(x),args.batch_size):
                        # One reproducible full permutation per epoch.
                        if start == 0: order = rng.permutation(len(x))
                        idx = order[start:start+args.batch_size]
                        total += model.train_batch(x[idx],task[idx],arrays['labels2d'][idx])*len(idx)
                        max_norm = max(max_norm,model.last_gradient_norm)
                        max_clipped = max(max_clipped,model.last_clipped_gradient_norm)
                    loss, geometry = full_health(model,x,task,arrays,batch_size=args.batch_size)
                    best = min(best,loss)
                    gate = training_gate(geometry,loss,initial_loss,best)
                    record = dict(stage=stage,epoch=epoch,train_2d_mse=total/len(x),
                        full_train_2d_mse=loss,max_gradient_norm=max_norm,
                        max_clipped_gradient_norm=max_clipped,quality=gate,
                        seconds=time.monotonic()-epoch_started)
                    log.write(json.dumps(record,allow_nan=False)+'\n')
                    write_json(output/'progress.json',dict(status='training',elapsed_seconds=time.monotonic()-started,**record))
                    if epoch==1 or epoch%10==0 or epoch==args.epochs: print(json.dumps(record),flush=True)
                    # Ten warmup epochs permit the fresh residual stage to settle;
                    # geometry failure after warmup aborts, never silently retries.
                    if epoch >= 10 and not geometry['passed']:
                        raise ValueError(f'Degenerate training geometry: stage {stage}, epoch {epoch}')
                if not gate['passed']: raise ValueError(f'Training quality failed: stage {stage}: {gate["reasons"]}')
                checkpoint = output/f'stage_{stage}.weights.h5'
                model.save_stage(checkpoint)
                before = model.predict(x[:args.batch_size],task[:args.batch_size])
                model.load_stage(checkpoint)
                if not np.array_equal(before,model.predict(x[:args.batch_size],task[:args.batch_size])):
                    raise ValueError('Checkpoint reload changed predictions')
                selected.append(dict(stage=stage,epoch=args.epochs,checkpoint_sha256=sha256(checkpoint)))
                quality.append(gate)
                write_json(output/'selected_stages.json',selected)
        report = dict(status='complete_train_quality_passed',dataset='ski',pilot=args.pilot,
            frames=len(x),stages=3,epochs=args.epochs,selected_stages=selected,quality=quality,
            training_manifest_sha256=data['manifest_hash'],training_manifest_path=str(args.train_dir.resolve()/'manifest.json'),
            config_sha256=sha256(output/'config.json'),uncertainty_sha256=sha256(output/'uncertainty.json'),
            frame_indices_sha256=sha256(output/'training_frame_indices.npy'),
            history_sha256=sha256(output/'epochs.jsonl'),total_seconds=time.monotonic()-started,
            test_data_opened=False)
        write_json(output/'training_report.json',report)
        write_json(output/'progress.json',report)
        return report
    except BaseException as error:
        failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
        write_json(output/'failure.json',failure); write_json(output/'progress.json',failure)
        raise


def verified_fit(path):
    path=Path(path); report=json.loads((path/'training_report.json').read_text())
    require_exportable(report)
    for name,key in [('config.json','config_sha256'),('uncertainty.json','uncertainty_sha256'),
                     ('epochs.jsonl','history_sha256'),('training_frame_indices.npy','frame_indices_sha256')]:
        if sha256(path/name)!=report[key]: raise ValueError('Training artifact changed: '+name)
    config=json.loads((path/'config.json').read_text())
    if (config['epochs']!=300 or config['pilot'] or config['max_frames'] is not None
            or config['seed']!=42 or config['batch_size']!=128 or config['policy']!=POLICY
            or not config['identity_stage_initialization'] or report['test_data_opened'] is not False):
        raise ValueError('Training config is incompatible with the declared protocol')
    if not np.array_equal(np.load(path/'training_frame_indices.npy'),np.arange(4350)):
        raise ValueError('Training did not cover the complete fixed split')
    saved_code={Path(p).name:v for p,v in config['code_sha256'].items()}
    for p in (Path(__file__),Path(__file__).with_name('run_metapose.py'),
              Path(__file__).parents[1]/'eval/metapose_stability.py'):
        if saved_code.get(p.name)!=sha256(p):
            raise ValueError('Adapter code differs from the training implementation: '+p.name)
    if sha256(report['training_manifest_path']) != report['training_manifest_sha256']:
        raise ValueError('Training manifest changed')
    history=[json.loads(s) for s in (path/'epochs.jsonl').read_text().splitlines()]
    if [(r['stage'],r['epoch']) for r in history] != [(s,e) for s in range(1,4) for e in range(1,301)]:
        raise ValueError('Incomplete training history')
    for stage in range(1,4):
        rows=[r for r in history if r['stage']==stage]
        gate=rows[-1]['quality']; geometry=gate['geometry']
        if geometry['policy']!=POLICY or geometry['frames']!=4350:
            raise ValueError('Wrong health-check policy or coverage')
        geometry=dict(geometry,passed=geometry['bad_frame_fraction']<=POLICY['max_bad_frame_fraction']
                      and geometry['camera_scale_median']>=POLICY['min_median_camera_scale'])
        checked=training_gate(geometry,rows[-1]['full_train_2d_mse'],gate['initial_2d_mse'],
                              min(r['full_train_2d_mse'] for r in rows))
        if not checked['passed'] or checked!=gate or gate!=report['quality'][stage-1]:
            raise ValueError('Training quality report disagrees with verified history')
    if [s['stage'] for s in report['selected_stages']] != [1,2,3]: raise ValueError('Missing stages')
    for row in report['selected_stages']:
        if row['epoch']!=300 or sha256(path/f'stage_{row["stage"]}.weights.h5')!=row['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
    return report


def export(args):
    report=verified_fit(args.train_run)
    config=json.loads((args.train_run/'config.json').read_text())
    check_source(config['metapose_repo'])
    if args.output_dir.exists(): raise FileExistsError('Refusing to overwrite prediction export')
    test=load_split(args.test_dir,'test')
    if test['manifest']['dataset']!='ski' or len(test['xyz'])!=900:
        raise ValueError('Unexpected test coverage')
    model=model_for(config)
    for stage in range(1,4):
        model.add_stage(); model.load_stage(args.train_run/f'stage_{stage}.weights.h5')
    u=json.loads((args.train_run/'uncertainty.json').read_text())
    x=initialized(model,test['xyz'],config['batch_size'])
    preds={'s1':[],'metapose':[]}; started=time.monotonic()
    for start in range(0,len(x),config['batch_size']):
        stop=min(start+config['batch_size'],len(x))
        task=point_mixtures(test['uv'][start:stop],u).reshape(stop-start,-1)
        for name,state in [('s1',x[start:stop]),('metapose',model.predict(x[start:stop],task))]:
            raw=restore_left_gauge(model.project(state)[:,0],test['q'][start:stop,0],
                                   test['t'][start:stop,0],test['native_left'][start:stop])
            preds[name].append(raw)
    output=args.output_dir;output.mkdir(parents=True)
    for name,parts in preds.items():np.save(output/f'{name}_raw_left.npy',np.concatenate(parts).reshape(30,30,13,3))
    write_json(output/'report.json',dict(status='complete_predictions_not_yet_scored',dataset='ski',
        windows=30,frames=900,finite_coverage=1.,native_test_cache=test['manifest']['native_cache'],
        native_test_manifest_sha256=test['manifest']['native_manifest_sha256'],
        predictions={f'{name}_raw_left.npy':sha256(output/f'{name}_raw_left.npy') for name in preds},
        train_run=str(args.train_run.resolve()),training_report_sha256=sha256(args.train_run/'training_report.json'),
        training_quality_passed=True,selected_stages=report['selected_stages'],
        inference_seconds=time.monotonic()-started,seed=config['seed']))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for key in ('train-dir','output-dir','metapose-repo'):t.add_argument('--'+key,type=Path,required=True)
    t.add_argument('--epochs',type=int,default=300);t.add_argument('--batch-size',type=int,default=128)
    t.add_argument('--seed',type=int,default=42);t.add_argument('--learning-rate',type=float,default=1e-5)
    t.add_argument('--gradient-clip-norm',type=float,default=1.)
    t.add_argument('--pilot',action='store_true');t.add_argument('--max-frames',type=int)
    t.add_argument('--require-gpu',action='store_true')
    e=sub.add_parser('export')
    for key in ('train-run','test-dir','output-dir'):e.add_argument('--'+key,type=Path,required=True)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.command=='train':train(args)
    else:export(args)
