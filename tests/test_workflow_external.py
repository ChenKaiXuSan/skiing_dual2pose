import hashlib
import json
from pathlib import Path

import pytest

from dual2pose.experiments.workflow_external import build_jobs, validate, check_metrics, verify_raw_cache


def context(tmp_path):
    return {k: str(tmp_path/k) for k in ('repo','root','main_root','core_root','raw_cache')} | {'python':'python'}


def metrics(n):
    return dict(sample_count=n,point_count=n*30*13,acceleration_point_count=n*28*13,
                pa_point_count=n*30*13,pa_frame_count=n*30,mpjpe=0.1,pa_mpjpe=0.05,acceleration_error=0.2)


def test_dag_covers_fourteen_methods_and_exclusive_timing(tmp_path):
    jobs = {j['id']:j for j in build_jobs(context(tmp_path))}
    for dataset in ('unity','ski'):
        assert 'external_stride_'+dataset in jobs
        assert 'external_metapose_'+dataset in jobs
        pa = jobs['external_pa_'+dataset]
        assert 'dlt_residual_'+dataset in pa['deps']
        assert 'geometry_'+dataset in pa['deps']
        assert 'external_collect_'+dataset in pa['deps']
        assert len(pa['validation']['methods']) == 11
        assert 'mlp' not in pa['validation']['methods']
        assert any('--limit' in cmd and cmd[cmd.index('--limit')+1]=='0' for cmd in jobs['external_stride_'+dataset]['commands'])
    assert len(jobs['external_main_tables']['validation']['methods']) == 14
    for job in jobs.values():
        if job['id'].startswith('external_profile_'):
            assert job['resource'] == 'exclusive_gpu'
    assert jobs['external_reference_ski']['deps'] == ['external_prepare_cache','core_ski_reference']


def test_metrics_reject_partial_or_nonfinite():
    check_metrics(metrics(30),'ski')
    for field,value in [('point_count',1),('pa_frame_count',899),('mpjpe',float('nan')),('acceleration_error',-1)]:
        record=metrics(30);record[field]=value
        with pytest.raises(ValueError):check_metrics(record,'ski')


def test_raw_hash_mismatch_rejected(tmp_path):
    directory=tmp_path/'ski'/'test';directory.mkdir(parents=True)
    (directory/'left.npy').write_bytes(b'corrupt')
    (directory/'manifest.json').write_text(json.dumps(dict(dataset='ski',split='test',sample_count=30,
        time_window=30,num_joints=13,representation='raw_pre_batch_canonicalization',
        files={'left.npy':'0'*64})))
    with pytest.raises(ValueError,match='hash|Hash'):
        verify_raw_cache(tmp_path, datasets={'ski':['test']})


def test_validation_rejects_partial_stride_even_with_finite_metrics(tmp_path):
    ctx=context(tmp_path);job=next(j for j in build_jobs(ctx) if j['id']=='external_stride_ski')
    output=Path(job['validation']['path']);output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(dict(status='complete',completed_count=29,sample_count=30)))
    with pytest.raises(ValueError):validate(job,ctx)


def test_metapose_profile_has_isolated_environment_preparation(tmp_path):
    jobs={j['id']:j for j in build_jobs(context(tmp_path))}
    prep=jobs['external_prepare_metapose_env'];profile=jobs['external_profile_metapose']
    assert prep['resource']=='cpu'
    assert prep['id'] in profile['deps']
    assert 'metapose_env/bin/python' in profile['commands'][0][0]
    assert any('tensorflow[and-cuda]==2.15.1' in cmd for cmd in prep['commands'])


def test_local_artifact_requires_checkpoint_byte_identity(tmp_path):
    from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
    ctx=context(tmp_path)
    job=next(j for j in build_jobs(ctx) if j['id']=='external_reference_ski')
    directory=Path(job['validation']['path']);directory.mkdir(parents=True)
    checkpoint=tmp_path/'selected.ckpt';checkpoint.write_bytes(b'original selected weights')
    for method in job['validation']['methods']:
        record=dict(method=method,metrics={'common13':metrics(30)},provenance=dict(dataset='ski',canonical_mode='per_frame',evaluation_protocol=EVALUATION_PROTOCOL))
        if method=='canonfuse3d':record.update(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        (directory/f'{method}.json').write_text(json.dumps(record))
    assert len(validate(job,ctx)['artifacts'])==4
    checkpoint.write_bytes(b'different weights')
    with pytest.raises(ValueError,match='Checkpoint hash'):validate(job,ctx)


def test_persistent_metapose_environment_avoids_installation(tmp_path):
    ctx=context(tmp_path);python=Path(ctx['repo'])/'.cache/external_baselines/metapose_env/bin/python'
    python.parent.mkdir(parents=True);python.write_text('existing interpreter')
    jobs={j['id']:j for j in build_jobs(ctx)}
    prep=jobs['external_prepare_metapose_env']
    assert len(prep['commands'])==1
    assert prep['commands'][0][0]==str(python)
    assert prep['env']['CUDA_VISIBLE_DEVICES']==''
    assert jobs['external_profile_metapose']['commands'][0][0]==str(python)
