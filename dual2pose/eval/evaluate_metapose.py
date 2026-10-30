"""Rescore frozen native MetaPose predictions in the current input-defined frame."""
import argparse
import json
from pathlib import Path
from dual2pose.runtime_paths import input_path
import numpy as np
import torch
from dual2pose.eval.metapose_inputs import sha256, finite
from dual2pose.eval.pa_mpjpe import PAMetricAccumulator, PA_PROTOCOL
from dual2pose.eval.external_protocol import protocol_metadata, check_current_control
from dual2pose.trainer.canonicalize import canonicalize_pose_torch


def canonical_prediction_and_control(prediction, left, right, dataset):
    if dataset not in ('unity', 'ski'):
        raise ValueError(dataset)
    kw = {} if dataset == 'unity' else dict(left_hip=4,right_hip=5,neck=12,enforce_face_z_positive=False)
    lc, transform = canonicalize_pose_torch(torch.tensor(left),**kw, mode="per_frame")
    rc = canonicalize_pose_torch(torch.tensor(right),**kw, mode="per_frame")[0]
    prediction = torch.tensor(np.array(prediction),dtype=torch.float32)
    if prediction.shape != lc[..., -13:, :].shape or not torch.isfinite(prediction).all():
        raise ValueError('Expected finite native-left common13 predictions matching inputs')
    # Each (B,T) input frame supplies its own origin and axes. Never derive a
    # transform from the prediction or target: output translation stays scored.
    prediction = (prediction-transform['pelvis'][..., None, :]) @ transform['R']
    return prediction, (lc[:,:,-13:]+rc[:,:,-13:])/2


def evaluate(run_dir, control_metrics=None, output_dir=None, prediction_origin="archived"):
    if prediction_origin not in ("archived", "fresh_training"):
        raise ValueError("Unknown prediction origin")
    run_dir = Path(run_dir)
    report = json.loads((run_dir/'report.json').read_text())
    if report['status'] != 'complete_predictions_not_yet_scored' or report['finite_coverage'] != 1.:
        raise ValueError('Require complete finite frozen predictions')
    output_dir = Path(output_dir) if output_dir is not None else run_dir/'per_frame_common13_v1'
    if output_dir.exists():
        raise FileExistsError('Refusing to overwrite existing evaluation directory')
    cache = input_path(report['native_test_cache'])
    m = json.loads((cache/'manifest.json').read_text())
    if sha256(cache/'manifest.json') != report['native_test_manifest_sha256'] or m['split'] != 'test':
        raise ValueError('Native test protocol manifest mismatch')
    if m['sample_count'] != report['windows'] or m['dataset'] != report['dataset']:
        raise ValueError('Incomplete or mismatched test coverage')
    if m.get('representation') != 'raw_pre_batch_canonicalization' or m.get('time_window') != 30:
        raise ValueError('Require verified raw native input cache for current rescoring')
    check_current_control(control_metrics,dataset=m['dataset'],
                          cache_manifest_sha256=report['native_test_manifest_sha256'])
    arrays = {}
    for k in ('left','right','target'):
        if sha256(cache/f'{k}.npy') != m['files'][f'{k}.npy']:
            raise ValueError(f'Native {k} hash mismatch')
        arrays[k] = np.load(cache/f'{k}.npy',mmap_mode='r')
        joints = 15 if m['dataset'] == 'unity' else 13
        if arrays[k].shape != (m['sample_count'],30,joints,3):
            raise ValueError(f'Native {k} coverage/shape mismatch')
        finite(arrays[k],f'native {k}')
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
    kw = {} if dataset == 'unity' else dict(left_hip=4,right_hip=5,neck=12,enforce_face_z_positive=False)
    for start in range(0,m['sample_count'],bs):
        stop = min(start+bs,m['sample_count'])
        left,right,target = (np.array(arrays[k][start:stop]) for k in ('left','right','target'))
        truth = canonicalize_pose_torch(torch.tensor(target),**kw, mode="per_frame")[0][:,:,-13:]
        for name,value in predictions.items():
            pred,average = canonical_prediction_and_control(value[start:stop],left,right,dataset)
            acc[name].update(pred,truth)
        acc['average_control'].update(average,truth)
    scores = {k:v.result() for k,v in acc.items()}
    control_info = check_current_control(control_metrics,dataset=dataset,
        cache_manifest_sha256=report['native_test_manifest_sha256'],measured=scores['average_control'])
    scores.update(**protocol_metadata(),**control_info,pa_protocol=PA_PROTOCOL, dataset=dataset,
                  method_label='MetaPose (adapted, 2D-supervised, additional estimated 2D)',
                  coordinate_gauge='native left-camera output mapped with input left-view per-frame pelvis and axes',
                  native_prediction_sha256=report['predictions'],
                  cache_manifest_sha256=report['native_test_manifest_sha256'],
                  native_predictions_reused=prediction_origin == 'archived',
                  prediction_origin=prediction_origin, report_sha256=sha256(run_dir/'report.json'),
                  evaluator_sha256=sha256(__file__))
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'metrics.json').write_text(json.dumps(scores,indent=2,allow_nan=False)+'\n')
    print(json.dumps(scores,indent=2))
    return scores


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--control-metrics',type=Path,help='Optional current control; archived scores are rejected')
    p.add_argument('--output-dir',type=Path,help='New destination; defaults to run-dir/per_frame_common13_v1')
    p.add_argument("--prediction-origin", choices=("archived", "fresh_training"), default="archived")
    evaluate(**vars(p.parse_args()))
