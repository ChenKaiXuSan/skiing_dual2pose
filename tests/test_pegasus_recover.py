import json
from pathlib import Path

import pytest

from pegasus.runtime import chain


@pytest.fixture
def recovery_source(tmp_path):
    old = tmp_path / 'old'
    repo = tmp_path / 'repo'
    code = tmp_path / 'code'
    for name in ('dual2pose', 'configs', 'pegasus'):
        (code / name).mkdir(parents=True)
        (code / name / 'frozen.txt').write_text('repaired source')
    repo.mkdir()
    raw = repo / 'raw'
    raw.mkdir()
    ctx = dict(repo=str(repo), root=str(old / 'results'), main_root=str(old / 'main'),
               core_root=str(old / 'core'), raw_cache=str(raw), python='/runtime/python')
    jobs = []
    for name, deps in [('done', []), ('child', ['done']), ('metapose_unity_train', []),
                       ('failed', ['done']), ('main_unity', [])]:
        path = old / 'results' / name
        jobs.append(dict(id=name, deps=deps, family='cluster', commands=[['python', '--out', str(path)]],
                         validation={'kind': 'stable_export', 'path': str(path)}))
    plan = dict(root=str(old), repo=str(repo), runtime='/runtime', account='HP260146', queue='gen_S',
                protocol='common13_direct_per_frame_v1', context=ctx, jobs=jobs, submitted_jobs=['old'])
    chain.write_new(old / 'plan.json', plan)
    chain.write_new(old / 'results/context.json', ctx)
    chain.write_new(old / 'supplement/context.json', dict(ctx, root=str(old / 'supplement')))
    chain.write_new(old / 'results/state.json', {'must_not_import': True})
    for job in jobs:
        path = Path(job['validation']['path'])
        path.mkdir(parents=True)
        (path / 'pred.npy').write_bytes(b'prediction bytes')
        chain.write_new(path / 'report.json', dict(windows=30, finite_coverage=1,
                                                   predictions={'pred.npy': chain.sha(path / 'pred.npy')}))
        (old / 'tasks' / job['id']).mkdir(parents=True)
        (old / 'tasks' / job['id'] / 'run.log').write_text('done\n')
        submission = old / 'submissions' / job['id']
        submission.mkdir(parents=True)
        (submission / 'stdout.log').write_text('done\n')
        (submission / 'stderr.log').write_text('')
        if job['id'] != 'main_unity':
            md = chain.validate_job(job, plan, old)
            chain.write_new(old / 'receipts' / (job['id'] + '.json'),
                            dict(status='failed' if job['id'] == 'failed' else 'complete', metadata=md,
                                 artifacts=md['artifacts']))
    chain.write_new(old / 'dispatcher/state.json', dict(plan_sha256=chain.sha(old / 'plan.json'),
        checks={job['id']: {'status': 'failed' if job['id'] == 'failed' else 'complete'}
                for job in jobs if job['id'] != 'main_unity'},
        requests={job['id']: {'final_log_check': True} for job in jobs if job['id'] != 'main_unity'}))
    return old, tmp_path / 'new', code


def test_recovery_reuses_only_verified_complete_leaves_and_keeps_mutable_state_fresh(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    before = {str(p): p.read_bytes() for p in old.rglob('*') if p.is_file()}
    plan = prepare(old, new, code_root=code)
    assert plan['root'] == str(new)
    assert plan['submitted_jobs'] == []
    assert (new / 'results/done').is_dir() and not (new / 'results/done').is_symlink()
    assert (new / 'results/done/pred.npy').is_symlink()
    assert (new / 'results/done/pred.npy').resolve() == old / 'results/done/pred.npy'
    assert not (new / 'results/failed').exists()
    assert not (new / 'results/main_unity').exists()
    assert not (new / 'results/state.json').exists()
    assert not (new / 'dispatcher').exists()
    assert not list((new / 'submissions').iterdir())
    assert not list((new / 'tasks').iterdir())
    assert chain.read(new / 'results/context.json')['root'] == str(new / 'results')
    assert (new / 'receipts/done.json').read_bytes() == (old / 'receipts/done.json').read_bytes()
    imports = chain.read(new / 'recovery_imports.json')
    assert set(imports) == {'done', 'child', 'metapose_unity_train'}
    assert imports['done']['source_plan_sha256'] == chain.sha(old / 'plan.json')
    assert imports['done']['source_receipt_sha256'] == chain.sha(old / 'receipts/done.json')
    assert str(old / 'results/done/pred.npy') in imports['done']['artifacts']
    commands = {j['id']: j['commands'] for j in plan['jobs']}
    assert str(new / 'results/failed') in commands['failed'][0]
    assert any(s.startswith('+data.unity.raw_cache_root=') for s in commands['main_unity'][0])
    assert {str(p): p.read_bytes() for p in old.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('damage', ['artifact', 'receipt_status', 'log', 'final_log', 'plan'])
def test_recovery_refuses_changed_or_unreviewed_completed_evidence(recovery_source, damage):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    if damage == 'artifact':
        (old / 'results/done/pred.npy').write_bytes(b'changed')
    elif damage == 'receipt_status':
        receipt = chain.read(old / 'receipts/done.json'); receipt['status'] = 'failed'
        (old / 'receipts/done.json').write_text(json.dumps(receipt))
    elif damage == 'log':
        (old / 'submissions/done/stderr.log').write_text('Traceback (most recent call last):\n')
    elif damage == 'final_log':
        state = chain.read(old / 'dispatcher/state.json'); state['requests']['done']['final_log_check'] = False
        (old / 'dispatcher/state.json').write_text(json.dumps(state))
    else:
        (old / 'plan.json').write_text((old / 'plan.json').read_text() + '\n')
    with pytest.raises((ValueError, RuntimeError)):
        prepare(old, new, code_root=code)
    assert not new.exists()


def test_recovery_refuses_existing_output_and_excludes_descendants(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    new.mkdir(); (new / 'keep').write_text('safe')
    with pytest.raises(FileExistsError):
        prepare(old, new, code_root=code)
    assert (new / 'keep').read_text() == 'safe'
    other = new.parent / 'other'
    prepare(old, other, code_root=code, exclude={'done'})
    assert set(chain.read(other / 'recovery_imports.json')) == {'metapose_unity_train'}
    assert not (other / 'results/done').exists()
    assert not (other / 'results/child').exists()


def test_recovery_rejects_directory_alias_to_active_output(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    # A completed output may not smuggle another task's live tree into recovery.
    (old / 'results/done/live').symlink_to(old / 'results/main_unity', target_is_directory=True)
    with pytest.raises(ValueError, match='directory symlink'):
        prepare(old, new, code_root=code)
    assert not new.exists()


def test_recovery_validates_checkpoint_tokens_before_importing(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    plan = chain.read(old / 'plan.json')
    next(j for j in plan['jobs'] if j['id'] == 'child')['validation']['checkpoint'] = '{checkpoint:done}'
    (old / 'plan.json').write_text(json.dumps(plan))
    state = chain.read(old / 'dispatcher/state.json')
    state['plan_sha256'] = chain.sha(old / 'plan.json')
    (old / 'dispatcher/state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='checkpoint'):
        prepare(old, new, code_root=code)
    assert not new.exists()


def test_recovery_checks_metadata_hashes_when_legacy_receipt_has_no_artifacts(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    asset = old / 'results/done/assets.json'
    chain.write_new(asset, dict(verified_manifests=9, streams=3))
    plan = chain.read(old / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == 'done')
    job.update(family='training', validation={'kind': 'assets', 'output': str(asset)})
    (old / 'plan.json').write_text(json.dumps(plan))
    state = chain.read(old / 'dispatcher/state.json')
    state['plan_sha256'] = chain.sha(old / 'plan.json')
    (old / 'dispatcher/state.json').write_text(json.dumps(state))
    receipt = dict(status='complete', metadata=chain.validate_job(job, plan, old), artifacts={})
    (old / 'receipts/done.json').write_text(json.dumps(receipt))
    asset.write_text(json.dumps(dict(verified_manifests=9, streams=4)))
    with pytest.raises(ValueError, match='hash'):
        prepare(old, new, code_root=code)
    assert not new.exists()


def test_completed_geometry_input_alias_is_rebuilt_for_pending_residual(tmp_path):
    from pegasus.runtime.recover import _geometry_aliases
    old, new, source = tmp_path / 'old', tmp_path / 'new', tmp_path / 'immutable_geometry'
    source.mkdir()
    (source / 'manifest.json').write_text('original')
    old_geometry = old / 'results/main_baselines/geometry/unity'
    old_geometry.parent.mkdir(parents=True)
    old_geometry.symlink_to(source, target_is_directory=True)
    plan = {'jobs': [dict(id='geometry_unity', family='training', validation=dict(
        kind='geometry', root=str(old / 'results/main_baselines'), dataset='unity', source=str(source)))]}
    aliases = _geometry_aliases(plan, {'geometry_unity'}, old, new)
    assert aliases == {new / 'results/main_baselines/geometry/unity': source}
    assert _geometry_aliases(plan, set(), old, new) == {}
    old_geometry.unlink()
    wrong = tmp_path / 'wrong'; wrong.mkdir()
    old_geometry.symlink_to(wrong, target_is_directory=True)
    with pytest.raises(ValueError, match='Geometry input'):
        _geometry_aliases(plan, {'geometry_unity'}, old, new)


def test_recovery_preserves_leaf_alias_to_verified_completed_dependency(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    alias = old / 'results/child/inputs.npy'
    target = old / 'results/done/pred.npy'
    alias.symlink_to(target)
    prepare(old, new, code_root=code)
    imported = new / 'results/child/inputs.npy'
    assert imported.is_symlink() and imported.readlink() == alias
    assert imported.resolve() == target
    assert chain.read(new / 'recovery_imports.json')['child']['artifacts'][str(alias)] == chain.sha(target)


def test_recovery_keeps_all_reference_records_for_pending_collector(recovery_source):
    """Reference produces four records consumed by external_reference_ski."""
    from pegasus.runtime.recover import prepare

    old, new, code = recovery_source
    plan = chain.read(old / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == 'done')
    job.update(family='core', validation=dict(kind='reference', dataset='ski', method='canonfuse3d'))
    methods = ('left_canonical', 'right_canonical', 'canonical_avg', 'canonfuse3d')
    root = old / 'core/results/ski'
    for method in methods:
        chain.write_new(root / (method + '.json'), dict(
            method=method,
            metrics={'common13': dict(sample_count=30, point_count=11700,
                                      mpjpe=0.1, acceleration_error=0.01)},
            provenance={'evaluation_protocol': {'name': 'common13_direct_per_frame_v1'}}))
    (old / 'plan.json').write_text(json.dumps(plan))
    state = chain.read(old / 'dispatcher/state.json')
    state['plan_sha256'] = chain.sha(old / 'plan.json')
    (old / 'dispatcher/state.json').write_text(json.dumps(state))
    (old / 'receipts/done.json').write_text(json.dumps(dict(
        status='complete', metadata=chain.validate_job(job, plan, old), artifacts={})))

    prepare(old, new, code_root=code)
    imported = chain.read(new / 'recovery_imports.json')['done']['artifacts']
    for method in methods:
        source = root / (method + '.json')
        destination = new / 'core/results/ski' / source.name
        assert destination.is_file(), f'Missing downstream reference input: {method}'
        assert destination.read_bytes() == source.read_bytes()
        assert imported[str(source)] == chain.sha(source)


@pytest.mark.parametrize('target_kind', ['active', 'failed', 'external', 'unrelated'])
def test_recovery_rejects_leaf_alias_outside_completed_dependency_outputs(recovery_source, target_kind):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    targets = {'active': old / 'results/main_unity/pred.npy',
               'failed': old / 'results/failed/pred.npy',
               'unrelated': old / 'results/metapose_unity_train/pred.npy',
               'external': old.parent / 'outside.npy'}
    if target_kind == 'external':
        targets[target_kind].write_bytes(b'outside')
    (old / 'results/child/inputs.npy').symlink_to(targets[target_kind])
    with pytest.raises(ValueError, match='alias|escaped'):
        prepare(old, new, code_root=code)
    assert not new.exists()
