"""Domain-retrained common13 MetaPose using unmodified official TF modules.

Train/validation use 2D supervision only. Test exports contain predictions only;
3D accuracy is computed by the separate evaluate_metapose command.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from dual2pose.eval.metapose_inputs import (
    sha256, finite, fit_point_uncertainty, point_mixtures, restore_left_gauge,
)

SOURCE_COMMIT = '08a8d6736475776f42ffac23b2c13111a28e5795'
SPECS = ([(13,3),(2,2,3),(2,),(2,3)], [(2,13,4,4)])
MAIN_MLP = [512,512,'ccat',512,512,'ccat',512]


class OfficialMetaPose:
    def __init__(self, source, seed=42):
        os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '8')
        os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import tensorflow as tf
        for device in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                pass  # a second test instance may share an already initialized runtime
        sys.path.insert(0, str(Path(source).resolve()))
        from metapose import train_metapose as mp
        if not mp.flags.FLAGS.is_parsed():
            mp.flags.FLAGS(['ivc_metapose'])
        mp.flags.FLAGS['use_bone_len'].value = False
        mp.flags.FLAGS['activation_fn'].value = 'selu'
        mp.flags.FLAGS['debug_drop_mlp_inputs'].value = []
        tf.keras.backend.set_floatx('float64')
        tf.keras.utils.set_random_seed(seed)
        self.tf, self.mp, self.stages = tf, mp, []
        self.fwd_fn = mp.get_pose_forward_fn_batch(SPECS)
        self.logp_fn = mp.get_logp_re_evaluate_batch(SPECS)

        @tf.function(input_signature=[tf.TensorSpec((None,2,13,3), tf.float32)])
        def initialize_batch(xyz):
            def single(x):
                # Exact two-view unrolling of official initial_epi_estimate.
                # Its tf.range/list append loop is eager-only under TF 2.15.
                _, aligned, (mean0, mean1, r1, s1) = mp.inf_opt.align_aba(x[0], x[1])
                pose = aligned-mean0
                rot = tf.stack([tf.eye(3), r1])
                scale = tf.stack([tf.constant(1.), s1])
                shift = tf.stack([mean0, mean1])
                r, s = mp.inf_opt.reparam(rot, scale)
                return tf.cast(mp.pack_tensors_unbatched([pose,r,s,shift])[0], tf.float64)
            return tf.vectorized_map(single, xyz)
        self._initialize = initialize_batch

        @tf.function(input_signature=[tf.TensorSpec((None,59), tf.float64)])
        def project_batch(x):
            def single(v):
                pose, rot, _, scale, _, shift = mp.unpack_and_unparam(v, SPECS[0])
                return mp.inf_opt.project3d_weak(pose, rot, scale, shift)
            return tf.vectorized_map(single, x)
        self._project = project_batch

    def initialize(self, xyz):
        return finite(self._initialize(np.asarray(xyz, np.float32)).numpy(), 'official S1')

    def project(self, x):
        return finite(self._project(np.asarray(x, np.float64)).numpy(), 'weak projected 3D')

    def forward(self, x):
        return self.project(x)[..., :2].reshape(len(x), -1)

    def add_stage(self):
        tf, mp = self.tf, self.mp
        for previous in self.stages:
            previous.trainable = False
        stage = mp.InvariantStageModel(MAIN_MLP, 512, [256,128], SPECS)
        # Build before strict HDF5 save/load and optimizer variable creation.
        stage(tf.zeros((1,59+416+52+2), dtype=tf.float64))
        self.stages.append(stage)
        self.solver = mp.DeepInverseSolver(self.stages, self.fwd_fn, self.logp_fn, 'logp', {})
        self.solver.compile(custom_loss_weights={'logp': 0.0})
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
        optimizer._create_all_weights(stage.trainable_variables)

        @tf.function(reduce_retracing=True)
        def predict(x, task):
            result = self.solver({'x_init':x, 'task_vec':task})
            tf.debugging.assert_all_finite(result['x_opt'], 'nonfinite MetaPose state')
            return result

        @tf.function(reduce_retracing=True)
        def train(x, task, labels):
            with tf.GradientTape() as tape:
                result = self.solver({'x_init':x, 'task_vec':task})
                loss = tf.reduce_mean(tf.square(result['fwd']-tf.reshape(labels, (-1,52))))
                tf.debugging.assert_all_finite(loss, 'nonfinite MetaPose 2D loss')
            gradients = tape.gradient(loss, stage.trainable_variables)
            for g in gradients:
                tf.debugging.assert_all_finite(g, 'nonfinite MetaPose gradient')
            optimizer.apply_gradients(zip(gradients, stage.trainable_variables))
            return loss
        self._predict, self._train = predict, train

    def predict(self, x, task):
        return finite(self._predict(np.asarray(x,np.float64),np.asarray(task,np.float64))['x_opt'].numpy(), 'learned state')

    def train_batch(self, x, task, labels):
        return float(self._train(np.asarray(x,np.float64),np.asarray(task,np.float64),np.asarray(labels,np.float64)).numpy())

    def eval_loss(self, x, task, labels):
        out = self._predict(np.asarray(x,np.float64),np.asarray(task,np.float64))['fwd'].numpy()
        return float(np.mean((out-np.asarray(labels).reshape(-1,52))**2))

    def save_stage(self, path):
        self.stages[-1].save_weights(str(path))

    def load_stage(self, path):
        self.stages[-1].load_weights(str(path), skip_mismatch=False)
        for v in self.stages[-1].weights:
            finite(v.numpy(), v.name)


def load_split(directory, role):
    p = Path(directory)
    m = json.loads((p/'manifest.json').read_text())
    if m['split'] != role or m['ground_truth_3d_loaded'] or m['supplied_calibration']:
        raise ValueError('Wrong split role or input contamination')
    if m['labels2d_exported'] != (role in ('train','val')):
        raise ValueError('Incorrect supervision export boundary')
    fields = ['xyz','uv','q','t','native_left'] + (['labels2d'] if role != 'test' else [])
    out = {}
    for k in fields:
        if sha256(p/f'{k}.npy') != m['files'][f'{k}.npy']:
            raise ValueError(f'Prepared array hash mismatch: {p}/{k}')
        out[k] = np.load(p/f'{k}.npy', mmap_mode='r')
        if len(out[k]) != m['frames']:
            raise ValueError('Incomplete prepared frames')
    out['manifest'], out['manifest_hash'] = m, sha256(p/'manifest.json')
    return out


def write_json(path, content):
    tmp = Path(str(path)+'.partial')
    tmp.write_text(json.dumps(content, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def initialize_split(model, data, output, split, progress):
    n = len(data['xyz'])
    path = output/f'{split}_x_init.npy'
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(n,59))
    for start in range(0,n,4096):
        stop = min(start+4096,n)
        x[start:stop] = model.initialize(data['xyz'][start:stop])
        progress('initializing', split=split, completed_frames=stop, total_frames=n)
    x.flush()
    data['x'] = x


def run(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    revision = subprocess.check_output(['git','-C',str(args.metapose_repo),'rev-parse','HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(args.metapose_repo),'status','--porcelain','--','metapose'], text=True)
    if revision != SOURCE_COMMIT or dirty:
        raise ValueError('Expected unmodified, pinned official MetaPose source')
    output.mkdir(parents=True)
    started = time.monotonic()
    config = {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(source_commit=revision, main_mlp=MAIN_MLP, dtype='float64',
                  method='MetaPose common13 adapted, 2D-supervised, additional estimated 2D',
                  no_3d_supervision=True, no_supplied_calibration=True,
                  code_sha256={p:sha256(p) for p in [__file__,'dual2pose/eval/metapose_inputs.py']})
    write_json(output/'config.json',config)
    last_status = [0.]
    def progress(status, **extra):
        elapsed = time.monotonic()-started
        if elapsed > args.timeout_hours*3600:
            raise TimeoutError('Declared full-experiment hard timeout reached')
        info = dict(status=status, elapsed_seconds=elapsed, pid=os.getpid(), **extra)
        write_json(output/'progress.json',info)
        if elapsed-last_status[0] >= 30:
            print(json.dumps(info),flush=True); last_status[0]=elapsed
    try:
        train = load_split(args.train_dir,'train')
        val = load_split(args.val_dir,'val') if args.val_dir else None
        dataset = train['manifest']['dataset']
        if (dataset == 'unity') != (val is not None):
            raise ValueError('Unity requires validation; Ski requires fixed-budget no validation')
        if val and val['manifest']['dataset'] != dataset:
            raise ValueError('Validation dataset mismatch')
        model = OfficialMetaPose(args.metapose_repo,args.seed)
        tf = model.tf
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('Full run requires a GPU, refusing silent CPU fallback')
        write_json(output/'environment.json', dict(python=sys.version, tensorflow=tf.__version__,
                   devices=[str(d) for d in tf.config.list_physical_devices()],
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()))
        rng = np.random.default_rng(args.seed)
        selection = rng.choice(len(train['uv']), min(200000,len(train['uv'])), replace=False)
        uncertainty = fit_point_uncertainty(train['uv'][selection]-train['labels2d'][selection])
        uncertainty.update(training_manifest_sha256=train['manifest_hash'], seed=args.seed,
                           selected_frame_indices_sha256=__import__('hashlib').sha256(selection.tobytes()).hexdigest())
        write_json(output/'uncertainty.json',uncertainty)
        for name,data in [('train',train)]+([('val',val)] if val else []):
            initialize_split(model,data,output,name,progress)
        # Real-data bounded check precedes the full fit; weights are discarded.
        smoke_ids = rng.choice(len(train['uv']), min(128,len(train['uv'])), replace=False)
        x, uv, labels = (train[k][smoke_ids] for k in ('x','uv','labels2d'))
        task = point_mixtures(uv,uncertainty).reshape(len(x),-1)
        model.add_stage()
        initial_loss = model.eval_loss(x,task,labels)
        smoke_started = time.monotonic()
        losses = [model.train_batch(x,task,labels) for _ in range(40)]
        final_loss = model.eval_loss(x,task,labels)
        if not final_loss < initial_loss:
            raise ValueError('Real-data overfit gate failed')
        model.save_stage(output/'smoke.weights.h5')
        before = model.predict(x,task)
        model.load_stage(output/'smoke.weights.h5')
        if not np.array_equal(before,model.predict(x,task)):
            raise ValueError('Strict checkpoint reload gate failed')
        write_json(output/'smoke.json', dict(initial_2d_mse=initial_loss, final_2d_mse=final_loss,
                   train_steps=40, seconds=time.monotonic()-smoke_started, losses=losses,
                   strict_reload_identical=True, publication_eligible=False))
        # New model and seed ensure no diagnostic weight/optimizer carries into training.
        del model
        tf.keras.backend.clear_session()
        model = OfficialMetaPose(args.metapose_repo,args.seed)
        selected = []
        n = len(train['x'])
        with (output/'epochs.jsonl').open('x',buffering=1) as log:
            for stage in range(args.stages):
                model.add_stage()
                checkpoint = output/f'stage_{stage+1}.weights.h5'
                best = float('inf'); chosen = None
                for epoch in range(args.epochs):
                    epoch_started = time.monotonic()
                    order = rng.permutation(n)
                    total = 0.
                    for start in range(0,n,args.batch_size):
                        ids = order[start:start+args.batch_size]
                        task = point_mixtures(train['uv'][ids],uncertainty).reshape(len(ids),-1)
                        loss = model.train_batch(train['x'][ids],task,train['labels2d'][ids])
                        total += loss*len(ids)
                        if start % (args.batch_size*100) == 0:
                            progress('training',stage=stage+1,epoch=epoch+1,epochs=args.epochs,
                                     completed_frames=start+len(ids),total_frames=n,batch_2d_mse=loss)
                    val_loss = None
                    if val:
                        vn = len(val['x']); value = 0.
                        for start in range(0,vn,args.batch_size):
                            stop = min(start+args.batch_size,vn)
                            task = point_mixtures(val['uv'][start:stop],uncertainty).reshape(stop-start,-1)
                            value += model.eval_loss(val['x'][start:stop],task,val['labels2d'][start:stop])*(stop-start)
                            if start % (args.batch_size*100) == 0:
                                progress('validation',stage=stage+1,epoch=epoch+1,completed_frames=stop,total_frames=vn)
                        val_loss = value/vn
                    if val_loss is None or val_loss < best:
                        model.save_stage(checkpoint)
                        best = val_loss if val_loss is not None else total/n
                        chosen = epoch+1
                    record = dict(stage=stage+1,epoch=epoch+1,train_2d_mse=total/n,val_2d_mse=val_loss,
                                  selected_epoch=chosen,seconds=time.monotonic()-epoch_started)
                    log.write(json.dumps(record)+'\n')
                    if epoch == 0 or (epoch+1)%10 == 0 or epoch+1 == args.epochs:
                        print(json.dumps(record),flush=True)
                model.load_stage(checkpoint)
                selected.append(dict(stage=stage+1,epoch=chosen,checkpoint_sha256=sha256(checkpoint),
                                     selection='full validation 2D MSE' if val else 'fixed final epoch'))
                write_json(output/'selected_stages.json',selected)
        # Do not open test data until all stage weights and selection are frozen.
        test = load_split(args.test_dir,'test')
        if test['manifest']['dataset'] != dataset:
            raise ValueError('Test dataset mismatch')
        initialize_split(model,test,output,'test',progress)
        count = len(test['x'])
        predictions = {name:np.lib.format.open_memmap(output/f'{name}_raw_left.npy',mode='w+',dtype=np.float32,
                       shape=(count//30,30,13,3)) for name in ('s1','metapose')}
        infer_started = time.monotonic()
        for start in range(0,count,args.batch_size):
            stop = min(start+args.batch_size,count)
            x = test['x'][start:stop]
            task = point_mixtures(test['uv'][start:stop],uncertainty).reshape(stop-start,-1)
            for name,state in [('s1',x),('metapose',model.predict(x,task))]:
                projected = model.project(state)[:,0]
                raw = restore_left_gauge(projected,test['q'][start:stop,0],test['t'][start:stop,0],test['native_left'][start:stop])
                predictions[name].reshape(-1,13,3)[start:stop] = raw
            if start % (args.batch_size*100) == 0:
                progress('inference',completed_frames=stop,total_frames=count)
        for value in predictions.values():
            value.flush(); finite(value, 'complete predictions')
        report = dict(status='complete_predictions_not_yet_scored', dataset=dataset,
                      windows=count//30,frames=count,finite_coverage=1.0,
                      inference_seconds=time.monotonic()-infer_started,total_seconds=time.monotonic()-started,
                      selected_stages=selected,seed=args.seed,
                      split_manifests={k:v['manifest_hash'] for k,v in [('train',train),('test',test)]+([('val',val)] if val else [])},
                      native_test_cache=test['manifest']['native_cache'],
                      native_test_manifest_sha256=test['manifest']['native_manifest_sha256'],
                      uncertainty_sha256=sha256(output/'uncertainty.json'),
                      predictions={f'{k}_raw_left.npy':sha256(output/f'{k}_raw_left.npy') for k in predictions})
        write_json(output/'report.json',report)
        progress('complete_predictions_not_yet_scored',windows=count//30,frames=count)
        print(json.dumps(report,indent=2),flush=True)
        if args.evaluation_python:
            subprocess.run([str(args.evaluation_python),'-m','dual2pose.eval.evaluate_metapose',
                            '--run-dir',str(output),'--control-metrics',str(args.control_metrics)],check=True)
            progress('complete_evaluated',windows=count//30,frames=count)
    except BaseException as error:
        write_json(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                                              seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('metapose-repo','train-dir','test-dir','output-dir'):
        p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--val-dir',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--stages',type=int,default=3)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--timeout-hours',type=float,default=24.)
    p.add_argument('--evaluation-python',type=Path)
    p.add_argument('--control-metrics',type=Path)
    args = p.parse_args()
    if min(args.stages,args.epochs,args.batch_size,args.timeout_hours) <= 0:
        p.error('Training and timeout budgets must be positive')
    if bool(args.evaluation_python) != bool(args.control_metrics):
        p.error('Automatic evaluation needs both --evaluation-python and --control-metrics')
    run(args)
