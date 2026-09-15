"""Score complete frozen MetaPose predictions under the archived IVC protocol."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dual2pose.eval.metapose_inputs import sha256, finite
from dual2pose.eval.pa_mpjpe import PAMetricAccumulator, PA_PROTOCOL
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


def canonical_prediction_and_control(prediction, left, right, dataset):
    kw = {} if dataset == 'unity' else dict(left_hip=4,right_hip=5,neck=12)
    lc, transform = canonicalize_pose_torch(torch.tensor(left),**kw)
    rc = canonicalize_pose_torch(torch.tensor(right),**kw)[0]
    prediction = (torch.tensor(np.array(prediction),dtype=torch.float32)-transform['pelvis']) @ transform['R']
    return prediction, (lc[:,:,-13:]+rc[:,:,-13:])/2


def evaluate(run_dir, control_metrics):
    run_dir = Path(run_dir)
    report = json.loads((run_dir/'report.json').read_text())
    if report['status'] != 'complete_predictions_not_yet_scored' or report['finite_coverage'] != 1.:
        raise ValueError('Require complete finite frozen predictions')
    if (run_dir/'metrics.json').exists():
        raise FileExistsError('Refusing to overwrite existing metrics')
    cache = Path(report['native_test_cache'])
    m = json.loads((cache/'manifest.json').read_text())
    if sha256(cache/'manifest.json') != report['native_test_manifest_sha256'] or m['split'] != 'test':
        raise ValueError('Native test protocol manifest mismatch')
    if m['sample_count'] != report['windows'] or m['dataset'] != report['dataset']:
        raise ValueError('Incomplete or mismatched test coverage')
    arrays = {}
    for k in ('left','right','target'):
        if sha256(cache/f'{k}.npy') != m['files'][f'{k}.npy']:
            raise ValueError(f'Native {k} hash mismatch')
        arrays[k] = np.load(cache/f'{k}.npy',mmap_mode='r')
    predictions = {}
    for name in ('s1','metapose'):
        path = run_dir/f'{name}_raw_left.npy'
        if sha256(path) != report['predictions'][path.name]:
            raise ValueError('Prediction checksum mismatch')
        value = np.load(path,mmap_mode='r')
        if value.shape != (m['sample_count'],30,13,3):
            raise ValueError('Wrong prediction shape')
        predictions[name] = finite(value,'frozen predictions')
    acc = {k:PAMetricAccumulator() for k in (*predictions,'average_control')}
    dataset = m['dataset']; bs = 256 if dataset == 'unity' else 4
    kw = {} if dataset == 'unity' else dict(left_hip=4,right_hip=5,neck=12)
    for start in range(0,m['sample_count'],bs):
        stop = min(start+bs,m['sample_count'])
        left,right,target = (np.array(arrays[k][start:stop]) for k in ('left','right','target'))
        truth = canonicalize_pose_torch(torch.tensor(target),**kw)[0][:,:,-13:]
        for name,value in predictions.items():
            pred,average = canonical_prediction_and_control(value[start:stop],left,right,dataset)
            acc[name].update(pred,truth)
        acc['average_control'].update(average,truth)
    scores = {k:v.result() for k,v in acc.items()}
    reference = json.loads(Path(control_metrics).read_text())['average_control']
    delta = {k:scores['average_control'][k]-reference[k] for k in ('mpjpe','pa_mpjpe','acceleration_error')}
    if any(abs(v)>1e-7 for v in delta.values()):
        raise ValueError(f'Archived average control mismatch: {delta}')
    scores.update(pa_protocol=PA_PROTOCOL, dataset=dataset,
                  method_label='MetaPose (adapted, 2D-supervised, additional estimated 2D)',
                  coordinate_gauge='updated left camera, native-input pelvis restored, archived native-left batch transform',
                  archived_average_control_delta=delta,
                  control_metrics_sha256=sha256(control_metrics), report_sha256=sha256(run_dir/'report.json'),
                  evaluator_sha256=sha256(__file__), canonicalizer_sha256=sha256('dual2pose/trainer/canonicalize.py'))
    (run_dir/'metrics.json').write_text(json.dumps(scores,indent=2,allow_nan=False)+'\n')
    print(json.dumps(scores,indent=2))
    return scores


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--control-metrics',type=Path,required=True)
    evaluate(**vars(p.parse_args()))
