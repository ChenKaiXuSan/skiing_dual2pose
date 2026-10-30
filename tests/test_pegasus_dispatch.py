from pathlib import Path
import pytest
from pegasus.runtime import dispatch as d


def test_ready_dependency_failure_and_capacity():
    jobs=[dict(id='a',deps=[]),dict(id='b',deps=['a']),dict(id='c',deps=['b']),dict(id='d',deps=[])]
    assert d.select_ready(jobs,{},set(),set(),1)==['a']
    assert d.select_ready(jobs,{'a':'complete'},{'a'},set(),3)==['b','d']
    assert d.select_ready(jobs,{'a':'failed'},{'a'},set(),3)==['d']
    assert d.blocked_jobs(jobs,{'a':'failed'})=={'b','c'}
    assert d.select_ready(jobs,{},set(),{'a'},0)==[]
    assert d.select_ready(jobs,{},set(),{'a'},5)==['d']


def test_queue_parser_fails_closed_and_handles_empty_queue():
    assert d.parse_queue('')=={}
    text='RequestID ReqName UserName Queue Pri STT\n------ ------\n123.nqsv train user gen_S 0 RUN -\n124.nqsv train user gen_S 0 QUE -\n'
    assert d.parse_queue(text)=={'123.nqsv':'RUN','124.nqsv':'QUE'}
    with pytest.raises(ValueError):d.parse_queue('Server temporarily unavailable')
    assert d.submission_id('Request 123.nqsv submitted to queue: gen_S.')=='123.nqsv'
    with pytest.raises(ValueError):d.submission_id('connection lost')


def test_receipt_and_log_review_detects_changed_results_and_errors(tmp_path):
    from pegasus.runtime.chain import write_new,sha
    path=tmp_path/'result.json';path.write_text('{}')
    receipt={'status':'complete','artifacts':{str(path):sha(path)}}
    task=tmp_path/'tasks/a';task.mkdir(parents=True);(task/'run.log').write_text('training completed\n')
    assert d.check_receipt(tmp_path,'a',receipt)['status']=='complete'
    path.write_text('changed')
    assert d.check_receipt(tmp_path,'a',receipt)['status']=='failed'
    receipt['artifacts'][str(path)]=sha(path)
    (task/'run.log').write_text('Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n')
    assert d.check_receipt(tmp_path,'a',receipt)['status']=='failed'


def test_missing_receipt_requires_confirmed_absence_and_grace():
    assert not d.lost_job({'submitted_at':0,'absent_since':None},1000)
    assert not d.lost_job({'submitted_at':0,'absent_since':990},1000)
    assert d.lost_job({'submitted_at':0,'absent_since':800},1000)


def test_control_lock_prevents_two_dispatchers(tmp_path):
    with d.lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with d.lock(tmp_path):pass


def fake_plan(tmp_path):
    from pegasus.runtime.chain import write_new
    write_new(tmp_path/'plan.json',dict(account='HP260146',queue='gen_S',runtime='/runtime',jobs=[{'id':'a','deps':[]},{'id':'b','deps':['a']}]))
    write_new(tmp_path/'source_hashes.json',{})
    (tmp_path/'submissions').mkdir();(tmp_path/'receipts').mkdir()


def test_loop_advances_dependencies_without_duplicate_submissions(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from pegasus.runtime import chain
    fake_plan(tmp_path)
    submitted=[];ticks=[]
    def commands(cmd,**kwargs):
        if cmd[0]=='qstat':return SimpleNamespace(stdout='',returncode=0)
        job=cmd[-1];submitted.append(job)
        if job=='b':
            assert d.read(tmp_path/'dispatcher/state.json')['checks']['a']['status']=='complete'
        folder=tmp_path/'submissions'/job;folder.mkdir()
        (folder/'stdout.log').write_text('');(folder/'stderr.log').write_text('')
        chain.write_new(folder/'submission.json',dict(returncode=0,stdout=f'Request {len(submitted)}.nqsv submitted'))
        task=tmp_path/'tasks'/job;task.mkdir(parents=True);(task/'run.log').write_text('done')
        chain.write_new(tmp_path/'receipts'/f'{job}.json',dict(status='complete',artifacts={}))
        return SimpleNamespace(returncode=0,stdout='submitted',stderr='')
    def tick(_):
        ticks.append(1)
        if len(ticks)==5:(tmp_path/'dispatcher/STOP').touch()
    monkeypatch.setattr(d.subprocess,'run',commands);monkeypatch.setattr(d.time,'sleep',tick)
    monkeypatch.setattr(d.time,'time',lambda:1000+len(ticks)*31)
    assert d.run(tmp_path,2,30)==0
    assert submitted==['a','b']
    assert d.read(tmp_path/'dispatcher/state.json')['checks']['b']['status']=='complete'


def test_unavailable_scheduler_never_submits_or_infers_failure(tmp_path,monkeypatch):
    fake_plan(tmp_path)
    def unavailable(cmd,**kwargs):
        assert cmd[0]=='qstat'
        raise RuntimeError('unavailable')
    monkeypatch.setattr(d.subprocess,'run',unavailable)
    monkeypatch.setattr(d.time,'sleep',lambda _: (tmp_path/'dispatcher/STOP').touch())
    d.run(tmp_path)
    state=d.read(tmp_path/'dispatcher/state.json')
    assert state['checks']=={} and list((tmp_path/'submissions').iterdir())==[]
    assert 'unavailable' in state['last_error']


def test_running_parent_receipt_does_not_release_child(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from pegasus.runtime.chain import write_new
    fake_plan(tmp_path)
    folder=tmp_path/'submissions/a';folder.mkdir()
    write_new(folder/'submission.json',dict(returncode=0,stdout='Request 1.nqsv submitted'))
    task=tmp_path/'tasks/a';task.mkdir(parents=True);(task/'run.log').write_text('done')
    write_new(tmp_path/'receipts/a.json',dict(status='complete',artifacts={}))
    def commands(cmd,**kwargs):
        assert cmd[0]=='qstat', 'Child submitted before staged log review'
        return SimpleNamespace(stdout='1.nqsv train user gen_S 0 RUN -\n',returncode=0)
    monkeypatch.setattr(d.subprocess,'run',commands)
    monkeypatch.setattr(d.time,'sleep',lambda _: (tmp_path/'dispatcher/STOP').touch())
    d.run(tmp_path)
    assert d.read(tmp_path/'dispatcher/state.json')['checks']['a']['status']!='complete'


def test_unknown_reservation_consumes_capacity_and_can_reconcile(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from pegasus.runtime.chain import write_new
    fake_plan(tmp_path)
    definition=d.read(tmp_path/'plan.json');definition['jobs'].append(dict(id='c',deps=[]));d.atomic(tmp_path/'plan.json',definition)
    (tmp_path/'submissions/a').mkdir();ticks=[];submitted=[]
    def commands(cmd,**kwargs):
        if cmd[0]=='qstat':return SimpleNamespace(stdout='',returncode=0)
        assert ticks, 'Unresolved reservation must consume the only slot'
        submitted.append(cmd[-1]);raise RuntimeError('test stop submission')
    def tick(_):
        ticks.append(1)
        if len(ticks)==1:
            write_new(tmp_path/'submissions/a/submission.json',dict(returncode=0,stdout='Request 1.nqsv submitted'))
            task=tmp_path/'tasks/a';task.mkdir(parents=True);(task/'run.log').write_text('done')
            write_new(tmp_path/'receipts/a.json',dict(status='complete',artifacts={}))
            (tmp_path/'submissions/a/stdout.log').write_text('');(tmp_path/'submissions/a/stderr.log').write_text('')
        elif len(ticks)==3:(tmp_path/'dispatcher/STOP').touch()
    monkeypatch.setattr(d.subprocess,'run',commands);monkeypatch.setattr(d.time,'sleep',tick)
    monkeypatch.setattr(d.time,'time',lambda:1000+len(ticks)*31)
    d.run(tmp_path,1)
    assert d.read(tmp_path/'dispatcher/state.json')['checks']['a']['status']=='complete'
    assert submitted==['b']


def test_absent_request_waits_for_staged_logs_then_blocks_late_error(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from pegasus.runtime.chain import write_new
    fake_plan(tmp_path);folder=tmp_path/'submissions/a';folder.mkdir()
    write_new(folder/'submission.json',dict(returncode=0,stdout='Request 1.nqsv submitted'))
    task=tmp_path/'tasks/a';task.mkdir(parents=True);(task/'run.log').write_text('done')
    write_new(tmp_path/'receipts/a.json',dict(status='complete',artifacts={}))
    ticks=[]
    def commands(cmd,**kwargs):
        assert cmd[0]=='qstat','Do not submit before staged stderr'
        return SimpleNamespace(stdout='',returncode=0)
    def tick(_):
        assert d.read(tmp_path/'dispatcher/state.json')['checks']['a']['status']=='validating'
        ticks.append(1);(folder/'stdout.log').write_text('')
        (folder/'stderr.log').write_text('Traceback (most recent call last):\n')
    monkeypatch.setattr(d.subprocess,'run',commands);monkeypatch.setattr(d.time,'sleep',tick)
    monkeypatch.setattr(d.time,'time',lambda:1000+len(ticks)*31)
    assert d.run(tmp_path)==2
    assert d.read(tmp_path/'dispatcher/state.json')['checks']['a']['status']=='failed'
