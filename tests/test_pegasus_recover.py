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


def _complete_unity(source):
    """Model a completed Unity job with its actual protected output location."""
    plan = chain.read(source / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == 'main_unity')
    previous = Path(job['validation']['path'])
    destination = source / 'main/unity'
    destination.parent.mkdir(exist_ok=True)
    previous.rename(destination)
    job['validation']['path'] = str(destination)
    job['commands'] = [['python', '--out', str(destination)]]
    (source / 'plan.json').write_text(json.dumps(plan))
    state = chain.read(source / 'dispatcher/state.json')
    state['plan_sha256'] = chain.sha(source / 'plan.json')
    state['checks']['main_unity'] = {'status': 'complete'}
    state['requests']['main_unity'] = {'final_log_check': True}
    (source / 'dispatcher/state.json').write_text(json.dumps(state))
    metadata = chain.validate_job(job, plan, source)
    chain.write_new(source / 'receipts/main_unity.json',
                    dict(status='complete', metadata=metadata, artifacts=metadata['artifacts']))


def _initialize_imported_dispatcher(plan):
    from pegasus.runtime import dispatch
    definition = chain.read(plan / 'plan.json')
    checks = dispatch.imported_checks(plan, {j['id'] for j in definition['jobs']})
    chain.write_new(plan / 'dispatcher/state.json', dict(
        plan_sha256=chain.sha(plan / 'plan.json'), checks=checks, requests={}))


def test_successor_keeps_complete_unity_and_revalidates_multiple_import_generations(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    _complete_unity(old)
    prepare(old, new, code_root=code, preserve_main_unity=True)
    _initialize_imported_dispatcher(new)
    third = old.parent / 'third'
    plan = prepare(new, third, code_root=code, preserve_main_unity=True,
                   exclude={'done'}, command_overrides={
                       'done': [['python', '--out', str(new / 'results/done'), '--cache', '/cache/v1']]})
    assert set(plan['recovery']['excluded_jobs']) == {'done', 'child', 'failed'}
    assert set(plan['recovery']['imported_jobs']) == {'main_unity', 'metapose_unity_train'}
    assert (third / 'main/unity/pred.npy').resolve() == old / 'main/unity/pred.npy'
    assert not (third / 'results/done').exists()
    assert not (third / 'submissions/main_unity').exists()
    command = next(j for j in plan['jobs'] if j['id'] == 'done')['commands'][0]
    assert command == ['python', '--out', str(third / 'results/done'), '--cache', '/cache/v1']
    _initialize_imported_dispatcher(third)
    fourth = old.parent / 'fourth'
    prepare(third, fourth, code_root=code, preserve_main_unity=True)
    assert (fourth / 'main/unity/pred.npy').resolve() == old / 'main/unity/pred.npy'


@pytest.mark.parametrize('damage', ['artifact', 'receipt', 'scheduler_log', 'final_review', 'plan'])
def test_successor_rejects_changed_ancestor_evidence(recovery_source, damage):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    prepare(old, new, code_root=code)
    _initialize_imported_dispatcher(new)
    if damage == 'artifact':
        (old / 'results/done/pred.npy').write_bytes(b'changed')
    elif damage == 'receipt':
        (old / 'receipts/done.json').write_text('{}')
    elif damage == 'scheduler_log':
        (old / 'submissions/done/stderr.log').write_text('Traceback (most recent call last):\n')
    elif damage == 'final_review':
        state = chain.read(old / 'dispatcher/state.json')
        state['requests']['done']['final_log_check'] = False
        (old / 'dispatcher/state.json').write_text(json.dumps(state))
    else:
        (old / 'plan.json').write_text((old / 'plan.json').read_text() + '\n')
    third = old.parent / 'third'
    with pytest.raises((ValueError, RuntimeError)):
        prepare(new, third, code_root=code, preserve_main_unity=True)
    assert not third.exists()


def test_successor_overrides_cannot_silently_change_imported_jobs(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    with pytest.raises(ValueError, match='override.*excluded|excluded.*override'):
        prepare(old, new, code_root=code, preserve_main_unity=True,
                command_overrides={'done': [['python', '--different']]})
    assert not new.exists()


def test_successor_rejects_unknown_exclusion(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    with pytest.raises(ValueError, match='Unknown.*exclu'):
        prepare(old, new, code_root=code, preserve_main_unity=True, exclude={'typo'})
    assert not new.exists()


def test_successor_preserves_c07_checkpoint_records_and_raw_input_alias(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    plan = chain.read(old / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == 'metapose_unity_train')
    job['id'] = 'eval_prepare_c07'
    root = old / 'results/evaluations/c07_baseline'
    asset = root / 'assets.json'
    manifests = {}
    for number in range(3):
        path = Path(plan['repo']) / f'manifest{number}.json'
        path.write_text('{}')
        manifests[str(path)] = chain.sha(path)
    chain.write_new(asset, dict(status='verified', manifests=manifests, files=manifests))
    for method in ('canonfuse3d', 'dlt_mlp'):
        chain.write_new(root / f'results/unity/{method}.json', {'checkpoint': '/verified/input.ckpt'})
    (root / 'cache').symlink_to(plan['context']['raw_cache'], target_is_directory=True)
    job.update(family='evaluations', validation=dict(kind='asset_audit', path=str(asset), manifests=3))
    (old / 'plan.json').write_text(json.dumps(plan))
    for directory in ('receipts', 'tasks', 'submissions'):
        source = old / directory / ('metapose_unity_train.json' if directory == 'receipts' else 'metapose_unity_train')
        source.rename(old / directory / ('eval_prepare_c07.json' if directory == 'receipts' else 'eval_prepare_c07'))
    metadata = chain.validate_job(job, plan, old)
    (old / 'receipts/eval_prepare_c07.json').write_text(json.dumps(
        dict(status='complete', metadata=metadata, artifacts=metadata['artifacts'])))
    state = chain.read(old / 'dispatcher/state.json')
    state['plan_sha256'] = chain.sha(old / 'plan.json')
    for key in ('checks', 'requests'):
        state[key]['eval_prepare_c07'] = state[key].pop('metapose_unity_train')
    (old / 'dispatcher/state.json').write_text(json.dumps(state))
    prepare(old, new, code_root=code, preserve_main_unity=True)
    imported_root = new / 'results/evaluations/c07_baseline'
    for method in ('canonfuse3d', 'dlt_mlp'):
        assert (imported_root / f'results/unity/{method}.json').read_bytes() == (
            root / f'results/unity/{method}.json').read_bytes()
    assert (imported_root / 'cache').is_symlink()
    assert (imported_root / 'cache').resolve() == Path(plan['context']['raw_cache'])


def test_successor_preserves_eval_sibling_provenance_and_figures(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    plan = chain.read(old / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == 'done')
    output = old / 'results/evaluations/sync'
    output.mkdir(parents=True)
    summary = output / 'temporal_alignment_summary.csv'
    rows = ['injected_offset,condition,sample_count,mpjpe_mean']
    for offset in (-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5):
        for condition in ('uncorrected', 'automatic', 'oracle'):
            rows.append(f'{offset},{condition},64440,0.1')
    summary.write_text('\n'.join(rows) + '\n')
    threshold = Path(plan['repo']) / 'threshold.json'
    threshold.write_text('{}')
    chain.write_new(output / 'temporal_alignment_provenance.json', dict(
        evaluation_protocol='common13_direct_per_frame_v1', joint_subset='common13',
        canonical_mode='per_frame', output_alignment='none', limited_smoke_test=False,
        evaluated_samples=64440, metric_frame_count=20,
        threshold_file=str(threshold), threshold_sha256=chain.sha(threshold)))
    (output / 'figure.png').write_bytes(b'figure')
    job.update(family='evaluations', env={'EVAL_OUTPUT_ROOT': str(output)},
               validation=dict(kind='sync', path=str(summary)))
    (old / 'plan.json').write_text(json.dumps(plan))
    state = chain.read(old / 'dispatcher/state.json')
    state['plan_sha256'] = chain.sha(old / 'plan.json')
    (old / 'dispatcher/state.json').write_text(json.dumps(state))
    metadata = chain.validate_job(job, plan, old)
    (old / 'receipts/done.json').write_text(json.dumps(
        dict(status='complete', metadata=metadata, artifacts=metadata['artifacts'])))
    prepare(old, new, code_root=code, preserve_main_unity=True)
    imported = new / 'results/evaluations/sync'
    assert (imported / 'temporal_alignment_provenance.json').read_bytes() == (
        output / 'temporal_alignment_provenance.json').read_bytes()
    assert (imported / 'figure.png').read_bytes() == b'figure'


def _legacy_reference_successor(recovery_source, *, omit_copy_hash=False):
    """Reproduce an older recovery that omitted three core reference aliases."""
    from pegasus.runtime.recover import prepare
    from dual2pose.eval.main_baseline_data import EVALUATION_PROTOCOL
    old, new, code = recovery_source
    plan = chain.read(old / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == 'done')
    job.update(family='core', validation=dict(kind='reference', dataset='ski', method='canonfuse3d'))
    methods = ['left_canonical', 'right_canonical', 'canonical_avg', 'canonfuse3d']
    root = old / 'core/results/ski'
    copies = old / 'results/main_baselines/results/ski'
    for method in methods:
        record = dict(method=method, metrics={'common13': dict(
            sample_count=30, point_count=11700, acceleration_point_count=10920,
            mpjpe=0.1, acceleration_error=0.01)}, provenance=dict(
                dataset='ski', canonical_mode='per_frame',
                evaluation_protocol=dict(EVALUATION_PROTOCOL)))
        chain.write_new(root / (method + '.json'), record)
        chain.write_new(copies / (method + '.json'), record)
    collector = dict(id='external_reference_ski', deps=['done'], commands=[['copy-reference']],
                     family='external', validation=dict(kind='local', dataset='ski',
                     path=str(copies), methods=methods, pa=False))
    plan['jobs'].append(collector)
    (old / 'plan.json').write_text(json.dumps(plan))
    for directory, name in [('tasks', 'run.log'), ('submissions', 'stdout.log'),
                            ('submissions', 'stderr.log')]:
        log = old / directory / collector['id'] / name
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('')
    for current in (job, collector):
        metadata = chain.validate_job(current, plan, old)
        artifacts = metadata.get('artifacts', {}).copy()
        if omit_copy_hash and current is collector:
            artifacts.pop(str(copies / 'left_canonical.json'))
        (old / 'receipts' / (current['id'] + '.json')).write_text(json.dumps(
            dict(status='complete', metadata=metadata, artifacts=artifacts)))
    state = chain.read(old / 'dispatcher/state.json')
    state['plan_sha256'] = chain.sha(old / 'plan.json')
    state['checks'][collector['id']] = {'status': 'complete'}
    state['requests'][collector['id']] = {'final_log_check': True}
    (old / 'dispatcher/state.json').write_text(json.dumps(state))
    prepare(old, new, code_root=code)
    imports = chain.read(new / 'recovery_imports.json')
    for method in methods[:3]:
        (new / 'core/results/ski' / (method + '.json')).unlink()
        imports['done']['artifacts'].pop(str(root / (method + '.json')))
    (new / 'recovery_imports.json').write_text(json.dumps(imports))
    _initialize_imported_dispatcher(new)
    return old, new, code


def test_successor_recovers_missing_reference_companions_from_receipt_pinned_ancestor_copies(recovery_source):
    from pegasus.runtime.recover import prepare
    old, new, code = _legacy_reference_successor(recovery_source)
    before = {str(p): p.read_bytes() for root in (old, new) for p in root.rglob('*') if p.is_file()}
    third = old.parent / 'third'
    prepare(new, third, code_root=code, preserve_main_unity=True)
    for method in ('left_canonical', 'right_canonical', 'canonical_avg', 'canonfuse3d'):
        assert (third / 'core/results/ski' / (method + '.json')).resolve() == (
            old / 'core/results/ski' / (method + '.json'))
    imports = chain.read(third / 'recovery_imports.json')
    assert imports['done']['source_plan'] == str(old)
    assert imports['external_reference_ski']['source_plan'] == str(new)
    _initialize_imported_dispatcher(third)
    assert {str(p): p.read_bytes() for root in (old, new) for p in root.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('damage', ['ancestor_companion', 'copy_receipt', 'missing_copy_hash', 'unrelated_collector'])
def test_missing_reference_fallback_requires_complete_hash_pinned_copy_evidence(recovery_source, damage):
    from pegasus.runtime.recover import prepare
    old, new, code = _legacy_reference_successor(recovery_source, omit_copy_hash=damage == 'missing_copy_hash')
    if damage == 'ancestor_companion':
        (old / 'core/results/ski/left_canonical.json').write_text('{}')
    elif damage == 'copy_receipt':
        (old / 'receipts/external_reference_ski.json').write_text('{}')
    elif damage == 'unrelated_collector':
        definition = chain.read(new / 'plan.json')
        next(j for j in definition['jobs'] if j['id'] == 'external_reference_ski')['deps'] = []
        (new / 'plan.json').write_text(json.dumps(definition))
        state = chain.read(new / 'dispatcher/state.json')
        state['plan_sha256'] = chain.sha(new / 'plan.json')
        (new / 'dispatcher/state.json').write_text(json.dumps(state))
    third = old.parent / 'third'
    with pytest.raises(ValueError):
        prepare(new, third, code_root=code, preserve_main_unity=True)
    assert not third.exists()


def test_recovery_keeps_main_comparison_companion_reports_without_shared_directory(recovery_source):
    from dual2pose.eval.main_baseline_data import EVALUATION_PROTOCOL
    from pegasus.runtime.recover import prepare
    old, new, code = recovery_source
    plan = chain.read(old / 'plan.json')
    job = next(j for j in plan['jobs'] if j['id'] == 'done')
    path = old / 'results/external/main_comparison.json'
    source = Path(plan['repo']) / 'verified_result.json'
    source.write_text('{}')
    rows = []
    for dataset, count in (('unity', 64440), ('ski', 30)):
        metrics = dict(sample_count=count, point_count=count * 30 * 13,
                       acceleration_point_count=count * 28 * 13, pa_point_count=count * 30 * 13,
                       pa_frame_count=count * 30, mpjpe=0.1, acceleration_error=0.01, pa_mpjpe=0.05)
        for method in ('left_canonical', 'right_canonical', 'canonical_avg', 'aligned_average',
                       'quality_weighted', 'tcn', 'smoothnet', 'canonfuse3d', 'dlt',
                       'robust_dlt', 'dlt_mlp', 'deciwatch', 'stride', 'metapose'):
            rows.append(dict(dataset=dataset, method=method, metrics=metrics,
                             source=str(source), source_sha256=chain.sha(source)))
    chain.write_new(path, dict(rows=rows, canonical_mode='per_frame',
                              evaluation_protocol=dict(EVALUATION_PROTOCOL)))
    path.with_suffix('.csv').write_text('dataset,method,mpjpe\nski,canonfuse3d,0.1\n')
    path.with_suffix('.md').write_text('Verified main comparison\n')
    (path.parent / 'unrelated_active_result.json').write_text('{"unfinished": true}')
    job.update(family='external', validation=dict(kind='tables', path=str(path)))
    (old / 'plan.json').write_text(json.dumps(plan))
    state = chain.read(old / 'dispatcher/state.json')
    state['plan_sha256'] = chain.sha(old / 'plan.json')
    (old / 'dispatcher/state.json').write_text(json.dumps(state))
    metadata = chain.validate_job(job, plan, old)
    (old / 'receipts/done.json').write_text(json.dumps(
        dict(status='complete', metadata=metadata, artifacts=metadata['artifacts'])))
    before = {str(p): p.read_bytes() for p in old.rglob('*') if p.is_file()}
    prepare(old, new, code_root=code)
    for suffix in ('.json', '.csv', '.md'):
        destination = new / 'results/external' / path.with_suffix(suffix).name
        assert destination.read_bytes() == path.with_suffix(suffix).read_bytes()
    assert not (new / 'results/external/unrelated_active_result.json').exists()
    assert {str(p): p.read_bytes() for p in old.rglob('*') if p.is_file()} == before
