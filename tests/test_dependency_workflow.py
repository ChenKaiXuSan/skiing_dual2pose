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
