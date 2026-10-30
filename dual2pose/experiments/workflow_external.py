"""External comparisons, immutable input reuse, and report-only DAG tasks.

The PA entrypoint and wrapper use the eleven approved local methods; three
external adaptations complete fourteen rows per dataset. No command writes into
the manuscript tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from dual2pose.runtime_paths import input_path
import shutil
from types import SimpleNamespace

LOCAL = ['left_canonical', 'right_canonical', 'canonical_avg', 'aligned_average',
         'quality_weighted', 'tcn', 'smoothnet', 'canonfuse3d', 'dlt', 'robust_dlt', 'dlt_mlp']
METHODS = LOCAL + ['deciwatch', 'stride', 'metapose']
COUNTS = {'unity': 64440, 'ski': 30}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def check_metrics(metrics, dataset, *, pa=True):
    n = COUNTS[dataset]
    counts = {'sample_count': n, 'point_count': n*30*13, 'acceleration_point_count': n*28*13}
    if pa:
        counts.update(pa_point_count=n*30*13, pa_frame_count=n*30)
    for key, expected in counts.items():
        if metrics.get(key) != expected:
            raise ValueError(f'{dataset}/{key}: expected {expected}, got {metrics.get(key)}')
    for key in ['mpjpe', 'acceleration_error'] + (['pa_mpjpe'] if pa else []):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{dataset}/{key}: nonfinite or invalid metric')


def current(record):
    from dual2pose.eval.main_baseline_data import require_current_protocol
    require_current_protocol(record)


def verify_raw_cache(root, datasets=None):
    """Hash raw arrays before reuse; inspect native dimensions and finite coverage."""
    import numpy as np
    from dual2pose.eval.main_baseline_data import EXPECTED
    datasets = datasets or {'unity':['train', 'val', 'test'], 'ski':['train', 'test']}
    records = {}
    for dataset, splits in datasets.items():
        for split in splits:
            directory = Path(root)/dataset/split; m = read(directory/'manifest.json')
            if (m.get('dataset') != dataset or m.get('split') != split or
                    m.get('representation') != 'raw_pre_batch_canonicalization' or
                    m.get('time_window') != 30 or m.get('sample_count') != EXPECTED[dataset][split]):
                raise ValueError(f'Raw cache identity/coverage mismatch: {directory}')
            for name, digest in m['files'].items():
                if sha(directory/name) != digest:
                    raise ValueError(f'Raw cache hash mismatch: {directory/name}')
            for name in ('left', 'right', 'target'):
                value = np.load(directory/f'{name}.npy', mmap_mode='r')
                if value.shape != (m['sample_count'], 30, 15 if dataset=='unity' else 13, 3):
                    raise ValueError(f'Raw cache shape mismatch: {directory}/{name}')
                for start in range(0, len(value), 1024):
                    if not np.isfinite(value[start:start+1024]).all():
                        raise ValueError(f'Nonfinite raw cache: {directory}/{name}')
            if sha(input_path(m['index'])) != m['index_sha256']:
                raise ValueError(f'Raw index hash mismatch: {m["index"]}')
            samples = (directory/'samples.jsonl').read_text().splitlines()
            if len(samples) != m['sample_count'] or [json.loads(s)['index'] for s in samples] != list(range(len(samples))):
                raise ValueError(f'Raw sample coverage/order mismatch: {directory}')
            records[f'{dataset}/{split}'] = {'manifest_sha256':sha(directory/'manifest.json'), 'files':m['files']}
    return records


def _link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    if not source.exists():
        raise FileNotFoundError(f'Required input missing: {source}')
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(f'Preserving conflicting destination: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def build_jobs(ctx):
    root, repo, core, raw = (Path(ctx[k]) for k in ('root', 'repo', 'core_root', 'raw_cache'))
    main = root/'main_baselines'; out = root/'external'; jobs = []
    native = repo/'logs/ivc_mmsports_extension/external_baselines/20260911'
    def command(module, *args):
        return [ctx['python'], '-m', module, *map(str, args)]
    def internal(action, *args):
        return command(__name__, '--context', root/'context.json', '--action', action, *args)
    def add(name, deps, commands, validation, resource='gpu', required=()):
        job = dict(id=name, deps=deps, commands=commands, env={'MPLBACKEND':'Agg'},
                   resource=resource, family='external', validation=validation)
        missing = [str(p) for p in required if not Path(p).exists()]
        if missing:
            job['blocked_reason'] = 'Required immutable input unavailable: ' + ', '.join(missing)
            job['missing_inputs'] = missing
        jobs.append(job)
    add('external_prepare_cache', [], [internal('prepare')],
        dict(kind='cache', path=str(out/'raw_cache_verified.json')), 'cpu', [raw])
    for dataset in COUNTS:
        checkpoint = '{'+dataset+'_checkpoint}'
        reference = ['left_canonical', 'right_canonical', 'canonical_avg', 'canonfuse3d']
        if dataset == 'ski':
            refcmd = internal('copy-ski-reference')
            refdeps = ['external_prepare_cache', 'core_ski_reference']
        else:
            refcmd = command('dual2pose.experiments.run_main_baselines', '--root', main,
                             '--dataset', dataset, '--mode', 'reference', '--checkpoint', checkpoint, '--device', 'cuda:0')
            refdeps = ['external_prepare_cache', 'main_unity']
        add('external_reference_'+dataset, refdeps, [refcmd],
            dict(kind='local', path=str(main/'results'/dataset), dataset=dataset, methods=reference, pa=False))
        add('external_heuristics_'+dataset, ['external_prepare_cache'],
            [command('dual2pose.experiments.run_main_baselines', '--root', main, '--dataset', dataset,
                     '--mode', 'heuristics', '--device', 'cuda:0')],
            dict(kind='local', path=str(main/'results'/dataset), dataset=dataset,
                 methods=['aligned_average','quality_weighted'], pa=False))
        add('external_collect_'+dataset,
            [f'core_{dataset}_tcn', f'core_{dataset}_smoothnet'],
            [internal('collect', '--dataset', dataset)],
            dict(kind='local', path=str(main/'results'/dataset), dataset=dataset, methods=['tcn','smoothnet'], pa=False), 'cpu')
        add('external_pa_'+dataset,
            ['external_reference_'+dataset, 'external_heuristics_'+dataset, 'external_collect_'+dataset,
             'geometry_'+dataset, 'dlt_residual_'+dataset],
            [internal('pa', '--dataset', dataset)],
            dict(kind='local', path=str(root/'main_baselines_pa/results'/dataset), dataset=dataset, methods=LOCAL, pa=True))
        source = native/f'metapose_{dataset}_full_seed42'
        add('external_metapose_'+dataset, ['external_prepare_cache'],
            [command('dual2pose.eval.evaluate_metapose', '--run-dir', source, '--output-dir', out/'metapose'/dataset)],
            dict(kind='metapose', path=str(out/'metapose'/dataset), dataset=dataset, native=str(source)),
            'cpu', [source/'report.json', source/'metapose_raw_left.npy', source/'s1_raw_left.npy'])
        stride_repo = repo/'.cache/external_baselines/STRIDE'
        weight = repo/'ckpt/external/stride_latest_epoch.bin'
        dataset_root = repo.parent/'data'/('skiing_unity_dataset' if dataset=='unity' else 'Ski-PosePTZ-CameraDataset-png')
        add('external_stride_'+dataset, ['external_prepare_cache'],
            [command('dual2pose.experiments.run_stride_external', '--cache-dir', raw/dataset/'test',
                     '--dataset-root', dataset_root, '--output-dir', out/'stride'/dataset,
                     '--stride-repo', stride_repo, '--checkpoint', weight, '--device', 'cuda:0', '--limit', '0',
                     '--max-seconds', str(60*60*24*30))],
            dict(kind='stride', path=str(out/'stride'/dataset), dataset=dataset),
            required=[stride_repo, weight, dataset_root])
        add('external_qualitative_'+dataset, ['external_prepare_cache', 'main_'+dataset],
            [internal('qualitative', '--dataset', dataset, '--checkpoint', checkpoint)],
            dict(kind='qualitative', path=str(out/'qualitative'/dataset), dataset=dataset))
    dependencies = [prefix+d for d in COUNTS for prefix in ('external_pa_', 'external_stride_', 'external_metapose_')]
    dependencies += ['core_'+d+'_deciwatch' for d in COUNTS]
    add('external_main_tables', dependencies, [internal('tables')],
        dict(kind='tables', path=str(out/'main_comparison.json'), methods=METHODS), 'cpu')
    add('external_profile_fusion', ['main_unity','external_collect_unity'],
        [command('dual2pose.eval.profile_fusion_efficiency', '--checkpoint', '{unity_checkpoint}',
                 '--baseline-results', main/'results/unity', '--device', 'cuda:0', '--output', out/'efficiency/fusion.json')],
        dict(kind='profile', path=str(out/'efficiency/fusion.json'), method='fusion'), 'exclusive_gpu')
    persistent_tf = repo/'.cache/external_baselines/metapose_env/bin/python'
    tf_python = persistent_tf if persistent_tf.is_file() else out/'metapose_env/bin/python'
    prepare_commands = [] if persistent_tf.is_file() else [
         [ctx['python'], '-m', 'venv', str(out/'metapose_env')],
         [str(tf_python), '-m', 'pip', 'install', 'tensorflow[and-cuda]==2.15.1',
          'tensorflow-probability==0.23.0', 'tensorflow-datasets==4.9.3',
          'tensorflow-metadata==1.14.0', 'protobuf==3.20.3', 'numpy==1.26.4']]
    add('external_prepare_metapose_env', [],
        prepare_commands + [
         [str(tf_python), '-m', __name__, '--context', str(root/'context.json'), '--action', 'environment']],
        dict(kind='environment', path=str(out/'metapose_environment.json')), 'cpu')
    jobs[-1]['env']['CUDA_VISIBLE_DEVICES'] = ''
    for method, run, dep in [('stride', out/'stride/unity', 'external_stride_unity'),
                             ('deciwatch', core/'deciwatch/unity', 'core_unity_deciwatch'),
                             ('metapose', native/'metapose_unity_full_seed42', 'external_metapose_unity')]:
        argv = command('dual2pose.eval.profile_external_efficiency', '--method', method, '--method-run-dir', run,
                       '--output', out/'efficiency'/f'{method}.json')
        if method == 'metapose':
            argv[0] = str(tf_python)
            argv += ['--prepared-data-dir', str(native/'metapose_data/unity/test'),
                     '--source-repo', str(repo/'.cache/external_baselines/google-research')]
        add('external_profile_'+method, [dep] + (['external_prepare_metapose_env'] if method=='metapose' else []), [argv],
            dict(kind='profile', path=str(out/'efficiency'/f'{method}.json'), method=method), 'exclusive_gpu')
        if method == 'metapose':
            packages = tf_python.parent.parent/'lib/python3.11/site-packages/nvidia'
            libraries = ['cublas','cuda_cupti','cuda_nvrtc','cuda_runtime','cudnn',
                         'cufft','curand','cusolver','cusparse','nccl','nvjitlink']
            jobs[-1]['env']['LD_LIBRARY_PATH'] = ':'.join(
                [str(packages/name/'lib') for name in libraries] + [os.environ.get('LD_LIBRARY_PATH','')])
            jobs[-1]['env']['PATH'] = str(packages/'cuda_nvcc/bin') + ':' + os.environ.get('PATH','')
    return jobs


def _local(path, dataset, methods, pa):
    artifacts = {}
    for method in methods:
        p = Path(path)/f'{method}.json'; r = read(p)
        current(r['provenance'])
        if r['method'] != method or r['provenance']['dataset'] != dataset:
            raise ValueError(f'Method/dataset mismatch: {p}')
        check_metrics(r['metrics']['common13'], dataset, pa=pa)
        if r.get('checkpoint') and sha(r['checkpoint']) != r['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {p}')
        artifacts[str(p)] = sha(p)
    return artifacts


def validate(job, ctx):
    v = job['validation']; kind = v['kind']; path = Path(v['path']); artifacts = {}
    if job.get('blocked_reason'):
        raise ValueError(job['blocked_reason'])
    if kind == 'local':
        artifacts = _local(path, v['dataset'], v['methods'], v['pa'])
    elif kind == 'cache':
        r = read(path)
        actual = verify_raw_cache(ctx['raw_cache'])
        if r['caches'] != actual:
            raise ValueError('Raw cache bytes changed since preparation')
        if (Path(ctx['root'])/'main_baselines/cache').resolve() != Path(ctx['raw_cache']).resolve():
            raise ValueError('Raw cache symlink mismatch')
        artifacts[str(path)] = sha(path)
    elif kind in ('stride', 'metapose'):
        import numpy as np
        dataset = v['dataset']; n = COUNTS[dataset]
        if kind == 'stride':
            r = read(path/'report.json')
            if r.get('status') != 'complete' or r.get('completed_count') != n or r.get('sample_count') != n or not r.get('publication_eligible'):
                raise ValueError('Incomplete STRIDE test coverage')
            pred = path/'predictions_common13.npy'
            if sha(pred) != r['prediction_sha256']:
                raise ValueError('STRIDE prediction hash mismatch')
            a = np.load(pred,mmap_mode='r')
            if a.shape != (n,30,13,3) or not np.isfinite(a).all():
                raise ValueError('STRIDE prediction shape/finite coverage mismatch')
            p = read(path/'input_provenance.json'); current(p)
            if p['sample_count'] != n or p['full_split_count'] != n:
                raise ValueError('STRIDE prepared coverage mismatch')
            if sha(Path(ctx['raw_cache'])/dataset/'test/manifest.json') != p['cache_manifest_sha256']:
                raise ValueError('STRIDE raw cache manifest mismatch')
            key = 'stride_adapted'
        else:
            r = read(Path(v['native'])/'report.json')
            if r.get('windows') != n or r.get('finite_coverage') != 1.0:
                raise ValueError('Incomplete MetaPose native coverage')
            key = 'metapose'
        m = read(path/'metrics.json'); current(m)
        if m.get('dataset') != dataset:
            raise ValueError('External dataset mismatch')
        if m.get('cache_manifest_sha256') != sha(Path(ctx['raw_cache'])/dataset/'test/manifest.json'):
            raise ValueError('External score raw cache manifest mismatch')
        check_metrics(m[key],dataset)
        check_metrics(m['average_control'],dataset)
        if kind == 'metapose':
            if m['report_sha256'] != sha(Path(v['native'])/'report.json') or m['native_prediction_sha256'] != r['predictions']:
                raise ValueError('MetaPose reused source provenance mismatch')
            for name, digest in r['predictions'].items():
                source = Path(v['native'])/name
                if sha(source) != digest:
                    raise ValueError('MetaPose native prediction hash mismatch')
                a = np.load(source,mmap_mode='r')
                if a.shape != (n,30,13,3) or not np.isfinite(a).all():
                    raise ValueError('MetaPose native finite coverage mismatch')
        artifacts[str(path/'metrics.json')] = sha(path/'metrics.json')
    elif kind == 'tables':
        r = read(path); current(r)
        keys = {(row['dataset'],row['method']) for row in r['rows']}
        if keys != {(d,m) for d in COUNTS for m in METHODS} or len(r['rows']) != 28:
            raise ValueError('Main comparison must contain exactly 14 methods per dataset')
        for row in r['rows']:
            check_metrics(row['metrics'],row['dataset'])
            if sha(row['source']) != row['source_sha256']:
                raise ValueError('Main comparison source hash mismatch')
        artifacts[str(path)] = sha(path)
        for suffix in ('csv','md'):
            report = path.with_suffix('.'+suffix)
            if not report.is_file() or not report.stat().st_size:
                raise ValueError(f'Missing report {report}')
            artifacts[str(report)] = sha(report)
    elif kind == 'profile':
        r = read(path)
        if r.get('status') not in ('complete_timing_audit','complete_local_models_partial_flops'):
            raise ValueError('Incomplete timing audit')
        groups = r['results'].values() if v['method']=='fusion' else [r]
        for group in groups:
            if group.get('total_parameters',0) <= 0:
                raise ValueError('Invalid parameter count')
        def finite_numbers(value):
            if isinstance(value,dict):
                for x in value.values(): finite_numbers(x)
            elif isinstance(value,list):
                for x in value: finite_numbers(x)
            elif isinstance(value,float) and not math.isfinite(value):
                raise ValueError('Nonfinite efficiency measurement')
        finite_numbers(r)
        if v['method']=='fusion':
            for group in r['results'].values():
                if len(group['latency_ms_samples']) != group['repeats'] or group['latency_ms_median'] <= 0:
                    raise ValueError('Incomplete latency measurements')
        else:
            fields = {'stride':['total_with_ttt','forward_only'], 'deciwatch':['forward_only'],
                      'metapose':['learned_solver','total_with_adapters']}[v['method']]
            for field in fields:
                measurement = r[field]
                if (len(measurement['samples_ms']) != measurement['repeats'] or
                        measurement['median_ms'] <= 0 or measurement['repeats'] < 2):
                    raise ValueError('Incomplete external latency measurements')
        artifacts[str(path)] = sha(path)
    elif kind == 'environment':
        r = read(path)
        for name, expected in [('tensorflow','2.15.1'),('tensorflow-probability','0.23.0'),('numpy','1.26.4')]:
            if r['packages'].get(name) != expected:
                raise ValueError(f'MetaPose environment version mismatch: {name}')
        if not Path(r['python']).is_file():
            raise ValueError('Missing isolated MetaPose Python')
        artifacts[str(path)] = sha(path)
    elif kind == 'qualitative':
        import numpy as np
        r = read(path/'manifest.json'); current(r)
        if r['sample_indices'] != [0] or r['frame_indices'] != [0,14,29]:
            raise ValueError('Qualitative sample selection mismatch')
        if sha(r['checkpoint']) != r['checkpoint_sha256']:
            raise ValueError('Qualitative checkpoint changed')
        if r['cache_manifest_sha256'] != sha(Path(ctx['raw_cache'])/v['dataset']/'test/manifest.json'):
            raise ValueError('Qualitative raw cache manifest changed')
        for name, digest in r['files'].items():
            if sha(path/name) != digest:
                raise ValueError('Qualitative artifact hash mismatch')
        arrays = np.load(path/'poses.npz')
        if any(a.shape != (1,30,13,3) or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Qualitative pose coverage invalid')
        artifacts[str(path/'manifest.json')] = sha(path/'manifest.json')
    else:
        raise ValueError(f'Unknown validation kind {kind}')
    return {'artifacts':artifacts, 'validated_kind':kind}


def copy_records(source, destination, methods, dataset):
    _local(source,dataset,methods,False)
    for method in methods:
        src, dst = Path(source)/f'{method}.json', Path(destination)/f'{method}.json'
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha(src) != sha(dst):
                raise FileExistsError(f'Conflicting result: {dst}')
        else:
            with src.open('rb') as inp, dst.open('xb') as out:
                shutil.copyfileobj(inp,out)


def tables(ctx):
    from dual2pose.eval.external_protocol import protocol_metadata
    root = Path(ctx['root']); rows = []
    for dataset in COUNTS:
        for method in METHODS:
            if method in LOCAL:
                source = root/'main_baselines_pa/results'/dataset/f'{method}.json'
                r = read(source); current(r['provenance']); metrics = r['metrics']['common13']
            else:
                source = (Path(ctx['core_root'])/'deciwatch'/dataset/'metrics.json' if method=='deciwatch'
                          else root/'external'/method/dataset/'metrics.json')
                r = read(source); current(r); metrics = r['stride_adapted' if method=='stride' else method]
            check_metrics(metrics,dataset)
            rows.append(dict(dataset=dataset,method=method,metrics=metrics,source=str(source),source_sha256=sha(source)))
    dest=root/'external/main_comparison.json'
    write_new(dest,dict(**protocol_metadata(),rows=rows,method_count_per_dataset=14,
                        limitation='Fixed partitions and single runs; external adaptations retain method-specific inputs and training objectives.'))
    with dest.with_suffix('.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','method','mpjpe','pa_mpjpe','acceleration_error','source_sha256']);w.writeheader()
        for row in rows:
            w.writerow({k:row[k] for k in ('dataset','method','source_sha256')} |
                       {k:row['metrics'][k] for k in ('mpjpe','pa_mpjpe','acceleration_error')})
    with dest.with_suffix('.md').open('x') as f:
        f.write('Per-frame common-13 direct scoring. Fixed partitions; single runs. External methods are adaptations.\n\n')
        f.write('| Dataset | Method | MPJPE | PA-MPJPE | Acceleration |\n|---|---|---:|---:|---:|\n')
        for row in rows:
            m=row['metrics']; f.write(f"| {row['dataset']} | {row['method']} | {m['mpjpe']:.6f} | {m['pa_mpjpe']:.6f} | {m['acceleration_error']:.6f} |\n")


def qualitative(ctx,dataset,checkpoint):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dual2pose.eval.external_protocol import protocol_metadata
    from dual2pose.eval.main_baseline_data import canonical_batch
    from dual2pose.experiments.run_main_baselines import ReferenceModel
    root = Path(ctx['root'])/'external/qualitative'/dataset
    root.mkdir(parents=True,exist_ok=False)
    cache=Path(ctx['raw_cache'])/dataset/'test'
    inputs=[torch.tensor(np.load(cache/f'{k}.npy',mmap_mode='r')[:1].copy(),device='cuda:0') for k in ('left','right','target')]
    left,right,target=canonical_batch(*inputs,dataset=dataset)
    model=ReferenceModel(checkpoint).to('cuda:0').eval()
    with torch.no_grad(): pred=model(left,right)
    poses={k:v[:,:,-13:].cpu().numpy() for k,v in dict(left=left,right=right,prediction=pred,target=target).items()}
    if any(not np.isfinite(a).all() for a in poses.values()):raise ValueError('Nonfinite qualitative output')
    np.savez(root/'poses.npz',**poses)
    # Joint scatter avoids introducing an unverified skeleton topology. Identical
    # axes and fixed predeclared frames make displacement directly visible.
    frames=[0,14,29]; fig=plt.figure(figsize=(12,4))
    points=np.concatenate([x[0,frames].reshape(-1,3) for x in poses.values()])
    lo,hi=points.min(axis=0),points.max(axis=0); center=(lo+hi)/2; radius=max((hi-lo).max()/2,1e-6)
    for i,frame in enumerate(frames,1):
        ax=fig.add_subplot(1,3,i,projection='3d')
        for name,color in [('target','black'),('prediction','tab:red'),('left','tab:blue'),('right','tab:green')]:
            p=poses[name][0,frame];ax.scatter(*p.T,label=name,color=color,s=12)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),title=f'{dataset}: window 0, frame {frame}')
        ax.set_box_aspect((1,1,1))
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/'poses.png',dpi=160);fig.savefig(root/'poses.pdf');plt.close(fig)
    write_new(root/'manifest.json',dict(**protocol_metadata(),dataset=dataset,sample_indices=[0],frame_indices=frames,
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha(checkpoint),cache_manifest_sha256=sha(cache/'manifest.json'),
        selection='First fixed test window; frames 0,14,29 predeclared without error-based selection',
        files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--action',required=True,choices=['prepare','copy-ski-reference','collect','pa','tables','qualitative','environment'])
    p.add_argument('--dataset',choices=list(COUNTS));p.add_argument('--checkpoint')
    args=p.parse_args();ctx=read(args.context);root=Path(ctx['root']);base=root/'main_baselines'
    if args.action=='prepare':
        verified=verify_raw_cache(ctx['raw_cache']);_link(ctx['raw_cache'],base/'cache')
        write_new(root/'external/raw_cache_verified.json',dict(caches=verified))
    elif args.action=='copy-ski-reference':
        copy_records(Path(ctx['core_root'])/'results/ski',base/'results/ski',
                     ['left_canonical','right_canonical','canonical_avg','canonfuse3d'],'ski')
    elif args.action=='collect':
        copy_records(Path(ctx['core_root'])/'results'/args.dataset,base/'results'/args.dataset,['tcn','smoothnet'],args.dataset)
    elif args.action=='pa':
        from dual2pose.eval import evaluate_main_baseline_pa as pa
        import torch
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        pa.run(SimpleNamespace(root=base,output=root/'main_baselines_pa',dataset=args.dataset,device='cuda:0'))
    elif args.action=='tables': tables(ctx)
    elif args.action=='qualitative': qualitative(ctx,args.dataset,args.checkpoint)
    elif args.action=='environment':
        import importlib.metadata
        import sys
        # Import the exact archived framework pair before publishing readiness.
        import tensorflow
        import tensorflow_probability
        write_new(root/'external/metapose_environment.json', dict(python=sys.executable,
            packages={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}))


if __name__=='__main__':main()
