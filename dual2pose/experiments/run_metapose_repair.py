"""Run a separate Ski repair after the existing GPU queue releases GPU 1."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from dual2pose.experiments.run_dependency_workflow import atomic_json, gpu_inventory, now, sha


def resource_ready(jobs, states, inventory):
    gpu=inventory.get(1,{})
    if gpu.get('processes') or gpu.get('free_mb',0)<40000: return False
    for job in jobs:
        if job['resource']=='gpu' and job['id']!='metapose_unity_train':
            if states[job['id']]['status']!='complete': return False
        if job['resource']=='exclusive_gpu' and states[job['id']]['status'] in ('running','verifying'):
            return False
    unity=states.get('metapose_unity_train',{})
    if unity.get('status')=='running' and unity.get('gpu')==0 and inventory.get(0,{}).get('processes'):
        return True
    # When Unity finishes, give the old queue its exclusive profile first.
    return all(states[j['id']]['status']=='complete' for j in jobs if j['resource'] in ('gpu','exclusive_gpu'))


def run(root):
    root=Path(root).resolve();context=json.loads((root/'repair_context.json').read_text())
    lock=(root/'repair.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=root/'repair_state.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('status') not in ('waiting_for_gpu',):
        raise ValueError('Existing repair state; inspect instead of overwriting/restarting')
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'source'),PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES='1',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    record=dict(started_at=now(),pid=os.getpid(),root=str(root),steps=[])
    def status(name,**extra):
        record.update(status=name,updated_at=now(),**extra);atomic_json(state_path,record)
    try:
        for relative,digest in json.loads((root/'source_hashes.json').read_text()).items():
            if sha(root/'source'/relative)!=digest:raise ValueError('Frozen source changed: '+relative)
        for p,digest in context['pilot_reports'].items():
            if sha(p)!=digest:raise ValueError('Pilot report changed')
            report=json.loads(Path(p).read_text())
            if report['status']!='complete_train_quality_passed' or not report['pilot'] or report['test_data_opened']:
                raise ValueError('Train-only pilot did not pass')
        if not context['code_checks_passed']:raise ValueError('Code checks did not pass')
        old=Path(context['existing_root']);jobs=json.loads((old/'manifest.json').read_text())['jobs']
        while True:
            states=json.loads((old/'state.json').read_text())['jobs'];inventory=gpu_inventory()
            if resource_ready(jobs,states,inventory):
                # Confirm this is sustained availability, not a teardown gap.
                time.sleep(5)
                states=json.loads((old/'state.json').read_text())['jobs'];inventory=gpu_inventory()
                if resource_ready(jobs,states,inventory):break
            status('waiting_for_gpu',waiting_reason='existing ordinary GPU jobs finish before Ski repair',
                   existing_gpu_jobs_remaining=[j['id'] for j in jobs if j['resource']=='gpu'
                     and j['id']!='metapose_unity_train' and states[j['id']]['status']!='complete'])
            time.sleep(20)
        tf=context['tensorflow_python'];python=context['python'];module='dual2pose.experiments.run_metapose_stable'
        train_dir=root/'training';pred_dir=root/'predictions'
        commands=[('gpu_validation',[tf,'-u','-m',module,'train','--train-dir',context['train_dir'],
                     '--output-dir',str(root/'gpu_validation'),'--metapose-repo',context['official_source'],
                     '--require-gpu','--pilot','--epochs','120','--max-frames','128']),
                  ('training',[tf,'-u','-m',module,'train','--train-dir',context['train_dir'],
                     '--output-dir',str(train_dir),'--metapose-repo',context['official_source'],'--require-gpu']),
                  ('export',[tf,'-u','-m',module,'export','--train-run',str(train_dir),
                     '--test-dir',context['test_dir'],'--output-dir',str(pred_dir)]),
                  ('evaluation',[python,'-u','-m','dual2pose.eval.evaluate_metapose',
                     '--run-dir',str(pred_dir),'--output-dir',str(pred_dir/'evaluation'),
                     '--prediction-origin','fresh_training'])]
        atomic_json(root/'commands.json',commands)
        for step,command in commands:
            entry=dict(step=step,command=command,started_at=now());record['steps'].append(entry)
            status(step,gpu=1)
            step_env=dict(env)
            if step=='evaluation':step_env['CUDA_VISIBLE_DEVICES']=''
            with (root/f'{step}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=root/'source',env=step_env,stdout=log,stderr=subprocess.STDOUT)
                entry['pid']=proc.pid;status(step)
                code=proc.wait()
            entry.update(return_code=code,finished_at=now())
            if code:raise RuntimeError(f'{step} exited with {code}; inspect {step}.log')
        metrics=json.loads((pred_dir/'evaluation/metrics.json').read_text())
        result=metrics['metapose']
        import math
        if (metrics['prediction_origin']!='fresh_training' or metrics['native_predictions_reused']
                or metrics['evaluation_protocol']['name']!='common13_direct_per_frame_v1'
                or result['point_count']!=11700 or result['sample_count']!=30
                or not all(math.isfinite(result[k]) and result[k]>=0 for k in ('mpjpe','pa_mpjpe'))):
            raise ValueError('Invalid final evaluation artifacts')
        summary=dict(method='MetaPose (adapted; stable training recipe)',dataset='ski',
                     mpjpe_mm=result['mpjpe']*1000,pa_mpjpe_mm=result['pa_mpjpe']*1000,
                     acceleration_mm_per_frame2=result['acceleration_error']*1000,
                     metrics_path=str(pred_dir/'evaluation/metrics.json'),
                     training_quality_passed=True,old_results_overwritten=False,paper_modified=False)
        atomic_json(root/'result_summary.json',summary)
        status('complete',finished_at=now(),result=summary)
    except BaseException as error:
        status('failed',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
