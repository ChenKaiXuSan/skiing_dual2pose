from pathlib import Path
import pytest
from importlib.util import spec_from_file_location,module_from_spec


def module():
    spec=spec_from_file_location('cluster_chain',Path(__file__).parents[1]/'pegasus/runtime/chain.py')
    m=module_from_spec(spec);spec.loader.exec_module(m);return m


def test_recursive_remapping_handles_json_arguments_and_boundary():
    m=module();result=m.remap({'cmd':['{"output":"/old/run/out"}','root=/old/data/x','/older/file']},{'/old/run':'/new/run','/old/data':'/new/data'})
    assert result['cmd']==['{"output":"/new/run/out"}','root=/new/data/x','/older/file']


def test_preparation_refuses_existing_directory(tmp_path):
    m=module();(tmp_path/'keep').write_text('old')
    with pytest.raises(FileExistsError):m.prepare(Path.cwd(),tmp_path,Path('/new/runtime'))
    assert (tmp_path/'keep').read_text()=='old'


def test_unmet_dependencies_prevent_any_execution(tmp_path):
    m=module()
    with pytest.raises(RuntimeError,match='dependency'):m.dependencies({'deps':['upstream']},tmp_path)
    assert not list(tmp_path.iterdir())


def test_approved_training_matrix_and_dependency_graph(tmp_path):
    m=module();repo=Path(__file__).parents[1];out=tmp_path/'plan'
    plan=m.prepare(repo,out,tmp_path/'runtime')
    jobs={j['id']:j for j in plan['jobs']}
    assert len([j for j in jobs.values() if j.get('scientific_training')])==38
    assert 'metapose_ski_train' in jobs and 'metapose_ski_export' in jobs
    assert 'run_metapose_stable' in ' '.join(jobs['metapose_ski_train']['commands'][0])
    assert jobs['external_metapose_ski']['deps']==['metapose_ski_export']
    assert all(j['commands'] for j in jobs.values())
    assert plan['submitted_jobs']==[]
    assert all('CUDA_VISIBLE_DEVICES' not in j.get('env',{}) for j in jobs.values())
    assert all(str(out) in str(j) or j['id'].startswith('external_prepare') for j in jobs.values())


def test_plan_has_bounded_stride_shards_and_no_scheduler_wall_limit_violation(tmp_path):
    m=module();out=tmp_path/'plan';plan=m.prepare(Path(__file__).parents[1],out,tmp_path/'runtime')
    jobs={j['id']:j for j in plan['jobs']}
    shards=[j for j in jobs.values() if j['id'].startswith('stride_unity_shard_')]
    assert len(shards)==32
    from pegasus.runtime.stride_shards import check_ranges
    check_ranges(64440,[(j['validation']['start'],j['validation']['end']) for j in shards])
    assert set(jobs['external_stride_unity']['deps'])=={j['id'] for j in shards}
    assert all('elapstim_req=24:00:00' in p.read_text() for p in (out/'pbs').glob('*.pbs'))


def test_receipt_is_published_only_after_complete_serialization(tmp_path,monkeypatch):
    m=module();path=tmp_path/'receipts'/'job.json';original=m.json.dump
    def observe(value,handle,**kwargs):
        assert not path.exists(), 'A dependent node could read partial JSON'
        original(value,handle,**kwargs)
    monkeypatch.setattr(m.json,'dump',observe)
    m.write_new(path,{'status':'complete'})
    assert m.read(path)=={'status':'complete'}
    with pytest.raises(FileExistsError):m.write_new(path,{'status':'changed'})
    assert m.read(path)=={'status':'complete'}


def test_rejected_submission_is_retryable_and_evidence_preserved(tmp_path,monkeypatch):
    m=module()
    from types import SimpleNamespace
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='queue unavailable'))
    assert m.submit_job(tmp_path,'job').returncode==1
    assert not (tmp_path/'submissions/job').exists()
    failures=list((tmp_path/'submission_failures').glob('job.*/submission.json'))
    assert len(failures)==1 and m.read(failures[0])['returncode']==1
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='123.nqsv',stderr=''))
    assert m.submit_job(tmp_path,'job').returncode==0
    with pytest.raises(FileExistsError):m.submit_job(tmp_path,'job')


def test_final_coverage_revalidates_results_before_publication(tmp_path,monkeypatch):
    m=module()
    m.write_new(tmp_path/'plan.json',{'jobs':[{'id':'pocket_frozen','family':'training'},{'id':'final_coverage','deps':['pocket_frozen']}]})
    monkeypatch.setattr(m,'dependencies',lambda *a:{})
    def missing(job,plan,root):
        assert job['id']=='pocket_frozen'
        raise FileNotFoundError('deleted Pocket summary')
    monkeypatch.setattr(m,'validate_job',missing,raising=False)
    with pytest.raises(FileNotFoundError,match='deleted Pocket summary'):m.summarize(tmp_path)
    assert not (tmp_path/'final').exists()
