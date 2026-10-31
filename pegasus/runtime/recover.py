"""Prepare an isolated successor DAG from verified completed predecessor jobs.

This module never submits jobs or writes to the predecessor. Completed outputs
are linked as individual files; all output directories and control state are new.
"""
import argparse
import copy
import json
from pathlib import Path
import shlex
import shutil
import time

from pegasus.runtime import chain, dispatch


def _validate_job(job, plan, root, metadata):
    from dual2pose.experiments import run_dependency_workflow as flow
    flow.verify_checkpoint_tokens(job, metadata)
    return chain.validate_job(flow.resolve_tokens(job, metadata), plan, root)


def _owned_paths(job, plan):
    """Select task-owned outputs, never a shared results directory wholesale."""
    spec = job['validation']
    family = job['family']
    ctx = plan['context']
    if family == 'main':
        root = Path(ctx['main_root'])
        ds = spec['dataset']
        return [root / ds, root / f'{ds}_status.json', root / f'{ds}.log']
    if family == 'core':
        root, ds = Path(ctx['core_root']), spec['dataset']
        if spec['kind'] == 'deciwatch':
            return [root / 'deciwatch' / ds]
        if spec['kind'] == 'reference':
            # The downstream reference collector consumes all four outputs.
            return [root / 'results' / ds / f'{method}.json' for method in
                    ('left_canonical', 'right_canonical', 'canonical_avg', 'canonfuse3d')]
        method = spec['method']
        paths = [root / 'results' / ds / f'{method}.json']
        if spec['kind'] == 'local':
            paths.append(root / 'training' / ds / method)
        return paths
    if family == 'external' and spec['kind'] == 'local':
        return [Path(spec['path']) / f'{method}.json' for method in spec['methods']]
    if family == 'training' and spec['kind'] in ('geometry', 'residual'):
        root, ds = Path(spec['root']), spec['dataset']
        methods = ('dlt', 'robust_dlt') if spec['kind'] == 'geometry' else ('dlt_mlp',)
        paths = [root / 'results' / ds / f'{method}.json' for method in methods]
        if spec['kind'] == 'residual':
            name = 'dlt_mlp' if ds == 'unity' else 'dlt_mlp_root_relative'
            paths.append(root / 'training' / ds / name)
        return paths
    value = spec.get('path', spec.get('output'))
    if not value:
        raise ValueError('No bounded output ownership for ' + job['id'])
    path = Path(value)
    if any(c in str(path) for c in '*?['):
        # Evaluation exports put figures beside their metric CSV. The dedicated
        # output root is explicit in the job, rather than inferred from a glob.
        output = job.get('env', {}).get('EVAL_OUTPUT_ROOT')
        if not output:
            raise ValueError('Unbounded glob output for ' + job['id'])
        return [Path(output)]
    return [path]


def _files(path, old):
    relative = path.relative_to(old)
    if relative.parts[0] not in ('main', 'core', 'results', 'supplement'):
        raise ValueError('Output outside permitted result trees: ' + str(path))
    if len(relative.parts) < 2 or relative.is_relative_to('main/unity'):
        raise ValueError('Shared or active output tree: ' + str(path))
    if path.is_symlink() and path.is_dir():
        raise ValueError('Refusing output directory symlink: ' + str(path))
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = []
        for candidate in path.rglob('*'):
            if candidate.is_symlink() and candidate.is_dir():
                raise ValueError('Refusing output directory symlink: ' + str(candidate))
            if candidate.is_file():
                candidates.append(candidate)
    else:
        raise FileNotFoundError(path)
    for candidate in candidates:
        if candidate.name in ('state.json', 'context.json'):
            continue
        if not candidate.resolve().is_relative_to(old):
            raise ValueError('Output escaped predecessor: ' + str(candidate))
        yield candidate


def _verify_source(old, plan, excluded):
    state = chain.read(old / 'dispatcher/state.json')
    digest = chain.sha(old / 'plan.json')
    if state['plan_sha256'] != digest or Path(plan['root']) != old:
        raise ValueError('Predecessor plan identity changed')
    jobs = {j['id']: j for j in plan['jobs']}
    if len(jobs) != len(plan['jobs']):
        raise ValueError('Duplicate job identity')
    excluded = set(excluded)
    while True:
        descendants = {j['id'] for j in jobs.values() if set(j['deps']) & excluded}
        if descendants <= excluded:
            break
        excluded |= descendants
    complete = {key for key, check in state['checks'].items() if check['status'] == 'complete'} - excluded
    receipts = {key: chain.read(old / 'receipts' / (key + '.json')) for key in complete}
    dependency_metadata = {key: receipt.get('metadata', {}) for key, receipt in receipts.items()}
    imports, all_files, owned_paths = {}, {}, {}
    for job_id in sorted(complete):
        if job_id not in jobs or not set(jobs[job_id]['deps']) <= complete:
            raise ValueError('Completed imports are not dependency closed: ' + job_id)
        if not state['requests'].get(job_id, {}).get('final_log_check'):
            raise ValueError('Missing final scheduler log review: ' + job_id)
        for name in ('stdout.log', 'stderr.log'):
            if not (old / 'submissions' / job_id / name).is_file():
                raise ValueError('Missing staged scheduler log: ' + job_id)
        receipt_path = old / 'receipts' / (job_id + '.json')
        receipt = receipts[job_id]
        checked = dispatch.check_receipt(old, job_id, receipt)
        if checked['status'] != 'complete':
            raise ValueError('Invalid completed receipt ' + job_id + ': ' + str(checked))
        metadata = _validate_job(jobs[job_id], plan, old, dependency_metadata)
        for key, value in receipt.get('metadata', {}).items():
            if key.endswith('_sha256') and metadata.get(key) != value:
                raise ValueError('Completed metadata hash changed: ' + job_id + '/' + key)
        owned_paths[job_id] = set()
        for root in _owned_paths(jobs[job_id], plan):
            owned_paths[job_id].update(_files(root, old))
        if not owned_paths[job_id]:
            raise ValueError('Completed task has no output files: ' + job_id)
        imports[job_id] = dict(source_plan=str(old), source_plan_sha256=digest,
            source_receipt=str(receipt_path), source_receipt_sha256=chain.sha(receipt_path))
    # STRIDE and collector outputs legitimately alias completed dependencies.
    # Check ownership only after every completed task's real files are known;
    # never follow an alias into an unfinished task or an external location.
    for job_id, paths in owned_paths.items():
        ancestors = {job_id}
        pending = list(jobs[job_id]['deps'])
        while pending:
            dependency = pending.pop()
            if dependency not in ancestors:
                ancestors.add(dependency)
                pending.extend(jobs[dependency]['deps'])
        allowed = {path for owner in ancestors for path in owned_paths[owner]
                   if path.resolve() == path}
        owned = {}
        for path in paths:
            if path.resolve() not in allowed:
                raise ValueError('Output alias is not owned by a completed dependency: ' + str(path))
            owned[str(path)] = chain.sha(path)
        imports[job_id]['artifacts'] = owned
        all_files.update(owned)
    return imports, all_files, excluded, dependency_metadata


def _freeze(code_root, repo, out):
    frozen = out / 'source'
    frozen.mkdir()
    for directory in ('dual2pose', 'configs', 'pegasus'):
        shutil.copytree(code_root / directory, frozen / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('logs', 'ckpt', '.cache', 'data', 'paper'):
        if (repo / name).exists():
            (frozen / name).symlink_to(repo / name, target_is_directory=True)
    chain.write_new(out / 'source_hashes.json', {
        str(p.relative_to(frozen)): chain.sha(p)
        for d in ('dual2pose', 'configs', 'pegasus') for p in (frozen / d).rglob('*') if p.is_file()})


def _geometry_aliases(plan, imported, old, out):
    aliases = {}
    for job in plan['jobs']:
        spec = job['validation']
        if job['id'] not in imported or job['family'] != 'training' or spec.get('kind') != 'geometry':
            continue
        source = Path(spec['source']).resolve()
        previous = Path(spec['root']) / 'geometry' / spec['dataset']
        if not source.is_dir() or previous.resolve() != source or source.is_relative_to(old):
            raise ValueError('Geometry input alias changed or is mutable: ' + job['id'])
        aliases[out / previous.relative_to(old)] = source
    return aliases


def _pbs(job, plan, out):
    frozen = out / 'source'
    lines = ['#!/bin/bash', '#PBS -A HP260146', '#PBS -q gen_S',
        '#PBS -N cf_' + job['id'][:10], '#PBS -l elapstim_req=24:00:00',
        'set -euo pipefail', 'export DUAL2POSE_REPO=' + shlex.quote(plan['repo']),
        'source ' + shlex.quote(str(frozen / 'pegasus/runtime/env.sh')),
        'export PYTHONPATH=' + shlex.quote(str(frozen)), 'cd ' + shlex.quote(str(frozen)),
        'exec "$DUAL2POSE_PYTHON" -m pegasus.runtime.chain run-one --plan ' +
        shlex.quote(str(out)) + ' --job ' + shlex.quote(job['id'])]
    (out / 'pbs' / (job['id'] + '.pbs')).write_text('\n'.join(lines) + '\n')


def prepare(old, out, *, code_root=None, exclude=()):
    old, out = Path(old).resolve(), Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out = out.resolve()
    if out.is_relative_to(old) or old.is_relative_to(out):
        raise ValueError('Successor and predecessor must have disjoint roots')
    original = chain.read(old / 'plan.json')
    if original['account'] != 'HP260146' or original['queue'] != 'gen_S':
        raise ValueError('Unexpected account or queue')
    if original['protocol'] != 'common13_direct_per_frame_v1':
        raise ValueError('Unexpected recovery protocol')
    imports, files, excluded, dependency_metadata = _verify_source(
        old, original, {'main_unity', *exclude})
    geometry_aliases = _geometry_aliases(original, imports, old, out)
    mapping = {str(old): str(out)}
    plan = chain.remap(copy.deepcopy(original), mapping)
    plan.update(root=str(out), submitted_jobs=[], created=time.time(), status='prepared_not_submitted',
                recovery=dict(source_plan=str(old), source_plan_sha256=chain.sha(old / 'plan.json'),
                              imported_jobs=sorted(imports), excluded_jobs=sorted(excluded)))
    for job in plan['jobs']:
        if job['id'] == 'main_unity':
            override = '+data.unity.raw_cache_root=' + str(Path(plan['context']['raw_cache']) / 'unity')
            job['commands'][0] = [arg for arg in job['commands'][0]
                                  if not arg.lstrip('+').startswith('data.unity.raw_cache_root=')]
            job['commands'][0].append(override)
    out.mkdir(parents=True, exist_ok=False)
    for directory in ('results', 'main', 'core', 'supplement', 'pbs', 'receipts', 'tasks', 'submissions'):
        (out / directory).mkdir()
    # Directory aliases are restricted to immutable raw and geometry inputs.
    # Every completed output directory itself remains a new, real directory.
    raw = Path(plan['context']['raw_cache'])
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    for relative in ('core/cache', 'results/main_baselines/cache', 'results/main_baselines_pa/cache'):
        dest = out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(raw, target_is_directory=True)
    for dest, source in geometry_aliases.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source, target_is_directory=True)
    for source, digest in files.items():
        path = Path(source)
        if chain.sha(path) != digest:
            raise ValueError('Completed file changed during preparation: ' + source)
        dest = out / path.relative_to(old)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(path)
    chain.write_new(out / 'results/context.json', plan['context'])
    supplement = chain.remap(chain.read(old / 'supplement/context.json'), mapping)
    chain.write_new(out / 'supplement/context.json', supplement)
    for job in plan['jobs']:
        if job['id'] in imports:
            _validate_job(job, plan, out, dependency_metadata)
            source = Path(imports[job['id']]['source_receipt'])
            # Keep original absolute checkpoint and artifact references intact.
            with (out / 'receipts' / source.name).open('xb') as dest:
                dest.write(source.read_bytes())
            if chain.sha(out / 'receipts' / source.name) != imports[job['id']]['source_receipt_sha256']:
                raise ValueError('Source receipt changed during preparation: ' + job['id'])
        _pbs(job, plan, out)
    code_root = Path(code_root).resolve() if code_root else Path(__file__).resolve().parents[2]
    _freeze(code_root, Path(plan['repo']), out)
    chain.write_new(out / 'recovery_imports.json', imports)
    # Publish the runnable plan last, after all imported results revalidate.
    chain.write_new(out / 'plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--exclude', action='append', default=[])
    args = parser.parse_args()
    plan = prepare(args.source_plan, args.output, code_root=args.code_root, exclude=args.exclude)
    print(json.dumps(dict(root=plan['root'], status=plan['status'], **plan['recovery']), indent=2))


if __name__ == '__main__':
    main()
