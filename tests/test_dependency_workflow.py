import pytest
from dual2pose.experiments.run_dependency_workflow import validate_graph, ready_assignments, dependency_blocks, resolve_tokens, verify_budget

def job(name,deps=(),resource='gpu'):
    return {'id':name,'deps':list(deps),'resource':resource,'commands':[['true']]}

def test_graph_rejects_missing_dependencies_and_cycles():
    with pytest.raises(ValueError,match='Missing'):
        validate_graph([job('b',['a'])])
    with pytest.raises(ValueError,match='cycle'):
        validate_graph([job('b',['a']),job('a',['b'])])

def test_failed_task_blocks_only_descendants():
    jobs=[job('a'),job('b',['a']),job('c'),job('d',['b'])]
    states={'a':{'status':'failed'},'b':{'status':'pending'},'c':{'status':'pending'},'d':{'status':'pending'}}
    assert set(dependency_blocks(jobs,states))=={'b','d'}
    assert ready_assignments(jobs,states,[0,1],True)==[('c',1)] or ready_assignments(jobs,states,[0,1],True)==[('c',0)]

def test_two_gpus_get_independent_jobs_but_never_unverified_dependency():
    jobs=[job('a'),job('b'),job('c',['a'])]
    states={j['id']:{'status':'pending'} for j in jobs}
    assignments=ready_assignments(jobs,states,[0,1],True)
    assert {x[0] for x in assignments}=={'a','b'}
    assert {x[1] for x in assignments}=={0,1}
    states['a']['status']='verifying'
    assert ('c',0) not in ready_assignments(jobs,states,[0],True)

def test_timing_requires_all_gpus_free_and_no_other_dispatched_work():
    jobs=[job('timing',resource='exclusive_gpu'),job('regular')]
    states={j['id']:{'status':'pending'} for j in jobs}
    assert ready_assignments(jobs,states,[1],True)==[('regular',1)]
    states['regular']['status']='complete'
    assert ready_assignments(jobs,states,[1],True)==[]
    assert ready_assignments(jobs,states,[0,1],True)==[('timing',0)]

def test_tokens_require_verified_checkpoint():
    assert resolve_tokens(['a','{checkpoint:fit}'],{'fit':{'checkpoint':'/tmp/fit.ckpt'}})==['a','/tmp/fit.ckpt']
    with pytest.raises(ValueError,match='checkpoint'):
        resolve_tokens('{unity_checkpoint}',{})

def test_final_epoch_field_alone_does_not_prove_budget():
    saved={'epoch':100,'global_step':3600,'loops':{'fit_loop':{'epoch_progress':{'total':{'processed':100}}}}}
    rows=[{'epoch':str(i),'train/loss_epoch':'0.1'} for i in range(100)]
    verify_budget(saved,rows,100,3600)
    with pytest.raises(ValueError):verify_budget(saved,rows[:-1],100,3600)
    saved['global_step']=3599
    with pytest.raises(ValueError):verify_budget(saved,rows,100,3600)


def test_worker_success_requires_validator_and_preserves_cuda_assignment(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from dual2pose.experiments import run_dependency_workflow as flow
    (tmp_path/'source').mkdir()
    ctx={'root':str(tmp_path),'repo':str(tmp_path),'python':sys.executable}
    j=job('tiny');j.update(family='test',env={},validation={})
    j['commands']=[[sys.executable,'-c','import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"']]
    flow.atomic_json(tmp_path/'context.json',ctx)
    flow.atomic_json(tmp_path/'manifest.json',{'jobs':[j]})
    flow.atomic_json(tmp_path/'state.json',{'metadata':{}})
    def validate(job,ctx):
        raise ValueError('Missing promised metric artifact')
    monkeypatch.setattr(flow,'family_module',lambda name:SimpleNamespace(validate=validate))
    with pytest.raises(ValueError,match='artifact'):flow.worker(tmp_path,'tiny',1)
    record=json.loads((tmp_path/'tasks/tiny/completion.json').read_text())
    assert record['status']=='failed'
    assert 'Missing promised metric artifact' in record['error']


def test_pending_adoption_can_dispatch_after_scheduler_restart():
    jobs=[dict(job('core_unity_tcn'),family='core')]
    assert ready_assignments(jobs,{'core_unity_tcn':{'status':'pending'}},[0],True)==[('core_unity_tcn',0)]


def test_worker_rejects_checkpoint_changed_after_upstream_verification(tmp_path):
    from dual2pose.experiments.run_dependency_workflow import verify_checkpoint_tokens
    p=tmp_path/'source.ckpt';p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        verify_checkpoint_tokens({'commands':[['eval','{unity_checkpoint}']]}, {'main_unity':{'checkpoint':str(p),'checkpoint_sha256':'old'}})


@pytest.fixture
def completed_unity_with_stale_last(tmp_path):
    import csv
    import json
    import torch
    from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
    root = tmp_path / 'unity'
    (root / 'checkpoints').mkdir(parents=True)
    best = root / 'checkpoints/best.ckpt'
    for path, epoch in [(best, 66), (root / 'checkpoints/last.ckpt', 72)]:
        torch.save({'epoch': epoch, 'global_step': (epoch + 1) * 47,
                    'loops': {'fit_loop': {'epoch_progress': {'total': {'processed': epoch + 1}}}},
                    'canonical_mode': 'per_frame', 'evaluation_protocol': EVALUATION_PROTOCOL,
                    'state_dict': {'weight': torch.ones(1)}}, path)
    (tmp_path / 'unity_status.json').write_text(json.dumps({'return_code': 0}))
    (tmp_path / 'unity.log').write_text('`Trainer.fit` stopped: `max_epochs=100` reached.\n')
    rows = [{'epoch': i, 'step': (i+1)*47-1, 'train/loss_epoch': .1,
             'val/mpjpe': .01 if i == 66 else .1} for i in range(100)]
    rows.append({'epoch': 100, 'step': 4700, 'test/mpjpe': .02})
    path = root / 'csv_logs/train/version_0/metrics.csv'
    path.parent.mkdir(parents=True)
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=['epoch','step','train/loss_epoch','val/mpjpe','test/mpjpe'])
        w.writeheader(); w.writerows(rows)
    (root / 'test_metrics_by_action.json').write_text(json.dumps({
        'evaluation_protocol': EVALUATION_PROTOCOL, 'canonical_mode': 'per_frame',
        'test_checkpoint_policy': 'best', 'best_model_path': str(best),
        'best_model_score': .01, 'sample_count': 64440, 'mpjpe': .02,
        'per_action': {'action': {'sample_count': 64440, 'mpjpe': .02}}}))
    return {'validation': {'dataset': 'unity'}}, {'main_root': str(tmp_path)}


def test_unity_budget_uses_full_history_and_best_selection_not_stale_last(completed_unity_with_stale_last):
    from dual2pose.experiments.run_dependency_workflow import verify_main
    result = verify_main(*completed_unity_with_stale_last)
    assert result['completed_epochs'] == 100
    assert result['checkpoint'].endswith('best.ckpt')
    assert result['terminal_checkpoint_epoch'] == 72


@pytest.mark.parametrize('damage', ['step', 'missing_epoch', 'best_selection', 'coverage'])
def test_unity_stale_last_does_not_bypass_completion_evidence(completed_unity_with_stale_last, damage):
    import json
    from pathlib import Path
    from dual2pose.experiments.run_dependency_workflow import verify_main
    job, ctx = completed_unity_with_stale_last
    root = Path(ctx['main_root']) / 'unity'
    csv = root / 'csv_logs/train/version_0/metrics.csv'
    if damage == 'step': csv.write_text(csv.read_text().replace('100,4700', '100,3431'))
    elif damage == 'missing_epoch': csv.write_text('\n'.join(x for x in csv.read_text().splitlines() if not x.startswith('99,')))
    else:
        p = root / 'test_metrics_by_action.json'; d = json.loads(p.read_text())
        d['best_model_score' if damage == 'best_selection' else 'sample_count'] = .1 if damage == 'best_selection' else 1
        p.write_text(json.dumps(d))
    with pytest.raises(ValueError): verify_main(job, ctx)
