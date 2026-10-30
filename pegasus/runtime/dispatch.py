"""Persistent NQSV dependency dispatcher; never reruns or overwrites a task."""
import argparse
from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temp.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)  # Only this dispatcher's mutable control state.
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_queue(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] == 'RequestID' or set(fields[0]) == {'-'}:
            continue
        if len(fields) < 6 or not re.fullmatch(r'\d+\.[\w.-]+', fields[0]):
            raise ValueError('Unrecognized qstat output: ' + line[:200])
        result[fields[0]] = fields[5]
    return result


def submission_id(text):
    found = re.findall(r'Request (\d+\.[\w.-]+) submitted', text)
    if len(found) != 1:
        raise ValueError('Ambiguous scheduler acceptance: ' + text[:200])
    return found[0]


def blocked_jobs(jobs, states):
    blocked = {key for key, status in states.items() if status in ('failed', 'unknown')}
    failed = blocked.copy()
    while True:
        more = {j['id'] for j in jobs if any(d in blocked for d in j['deps'])}
        if more <= blocked:
            return blocked - failed
        blocked |= more


def select_ready(jobs, states, submitted, excluded, slots):
    return [j['id'] for j in jobs if j['id'] not in submitted | excluded
            and j['id'] not in states
            and all(states.get(d) == 'complete' for d in j['deps'])][:max(0, slots)]


def lost_job(record, now):
    absent = record.get('absent_since')
    return absent is not None and now - absent >= 120 and now - record['submitted_at'] >= 120


def log_errors(plan, job_id):
    errors = []
    for path in (plan / 'tasks' / job_id / 'run.log',
                 plan / 'submissions' / job_id / 'stderr.log'):
        if not path.is_file():
            continue
        with path.open(errors='replace') as f:
            for line in f:
                if re.search(r'Traceback \(most recent call last\):|CUDA out of memory|Segmentation fault|^Killed\s*$', line):
                    errors.append({'file': str(path), 'line': line.strip()[:400]})
                    if len(errors) >= 20:
                        return errors
    return errors


def check_receipt(plan, job_id, receipt):
    if receipt.get('status') != 'complete':
        return {'status': 'failed', 'error': receipt.get('error', 'Failed worker receipt')}
    try:
        for path, digest in receipt.get('artifacts', {}).items():
            if sha(path) != digest:
                raise ValueError('Changed artifact: ' + path)
        if not (plan / 'tasks' / job_id / 'run.log').is_file():
            raise FileNotFoundError('Missing task log')
        alerts = log_errors(plan, job_id)
        if alerts:
            return {'status': 'failed', 'error': 'Task log needs review', 'log_errors': alerts}
        return {'status': 'complete', 'artifact_count': len(receipt.get('artifacts', {}))}
    except Exception as exc:
        return {'status': 'failed', 'error': repr(exc)}


def run(plan, max_inflight=64, interval=30):
    plan = Path(plan).resolve()
    control = plan / 'dispatcher'
    control.mkdir(exist_ok=True)
    definition = read(plan / 'plan.json')
    jobs = definition['jobs']
    by_id = {j['id']: j for j in jobs}
    if definition['account'] != 'HP260146' or definition['queue'] != 'gen_S':
        raise ValueError('Unexpected account or queue')
    # Import exactly the immutable code used by allocated workers.
    sys.path.insert(0, str(plan / 'source'))
    from pegasus.runtime import chain
    for name, digest in read(plan / 'source_hashes.json').items():
        if sha(plan / 'source' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    with lock(control):
        state_path = control / 'state.json'
        state = read(state_path) if state_path.exists() else {
            'plan_sha256': sha(plan / 'plan.json'), 'checks': {}, 'requests': {},
            'attempts': {}, 'created': time.time(), 'controller_sha256': sha(__file__)}
        if state['plan_sha256'] != sha(plan / 'plan.json'):
            raise ValueError('Plan changed since controller start')

        def event(kind, **data):
            with (control / 'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(), kind=kind, **data)) + '\n')
                f.flush()
                os.fsync(f.fileno())
            print(json.dumps(dict(kind=kind, **data)), flush=True)

        event('controller_start', pid=os.getpid(), max_inflight=max_inflight, interval=interval)
        while True:
            now = time.time()
            if (control / 'STOP').exists():
                state['status'] = 'stopped_submission_only'
                state['updated'] = now
                atomic(state_path, state)
                event('controller_stopped')
                return 0
            try:
                query = subprocess.run(['qstat', '-u', getpass.getuser()], capture_output=True,
                                       text=True, timeout=30, check=True)
                active = parse_queue(query.stdout)
                submitted = {p.name for p in (plan / 'submissions').iterdir() if p.is_dir()}
                for job_id in submitted:
                    if job_id not in by_id:
                        raise ValueError('Unrecognized submission directory: ' + job_id)
                    folder = plan / 'submissions' / job_id
                    if job_id not in state['requests']:
                        try:
                            if (folder / 'submission.json').exists():
                                record = read(folder / 'submission.json')
                                if record['returncode'] != 0:
                                    raise ValueError('Rejected submission left reserved')
                                request = submission_id(record['stdout'])
                            else:
                                receipt = read(plan / 'receipts' / (job_id + '.json'))
                                request = receipt['pbs_job_id']
                                if not re.fullmatch(r'\d+\.[\w.-]+', request):
                                    raise ValueError('Invalid worker scheduler ID')
                            state['requests'][job_id] = {'id': request, 'submitted_at': folder.stat().st_mtime,
                                                         'absent_since': None}
                            if state['checks'].get(job_id, {}).get('status') == 'unknown':
                                del state['checks'][job_id]
                        except Exception as exc:
                            state['checks'][job_id] = {'status': 'unknown', 'error': repr(exc)}
                            continue
                    request = state['requests'][job_id]
                    if request['id'] in active:
                        request['absent_since'] = None
                        request['queue_state'] = active[request['id']]
                    else:
                        request['queue_state'] = 'absent'
                        if request['absent_since'] is None:
                            request['absent_since'] = now
                    receipt_path = plan / 'receipts' / (job_id + '.json')
                    if receipt_path.exists() and (job_id not in state['checks'] or state['checks'][job_id]['status'] == 'unknown'):
                        result = check_receipt(plan, job_id, read(receipt_path))
                        if result['status'] == 'complete':
                            result['status'] = 'validating'
                        state['checks'][job_id] = result
                        event('task_checked', job=job_id, **result)
                    elif job_id not in state['checks'] and lost_job(request, now):
                        result = {'status': 'failed', 'error': 'Request disappeared without a worker receipt',
                                  'log_errors': log_errors(plan, job_id)}
                        state['checks'][job_id] = result
                        event('task_checked', job=job_id, **result)
                    # Scheduler stderr may be staged after the worker receipt.
                    absent_for = now - request['absent_since'] if request['absent_since'] is not None else 0
                    logs_present = all((folder / name).is_file() for name in ('stdout.log', 'stderr.log'))
                    if request['queue_state'] == 'absent' and job_id in state['checks'] and not request.get('final_log_check') and absent_for >= 30 and (logs_present or absent_for >= 120):
                        alerts = log_errors(plan, job_id)
                        request['final_log_check'] = True
                        if not logs_present:
                            state['checks'][job_id] = {'status': 'failed', 'error': 'Scheduler logs missing after staging grace'}
                            event('missing_staged_logs', job=job_id)
                        elif alerts:
                            state['checks'][job_id] = {'status': 'failed', 'error': 'Staged log needs review', 'log_errors': alerts}
                            event('staged_log_alert', job=job_id, alerts=alerts)
                        elif state['checks'][job_id]['status'] == 'validating':
                            state['checks'][job_id]['status'] = 'complete'
                            event('task_complete', job=job_id)
                statuses = {k: v['status'] for k, v in state['checks'].items()}
                cooldown = {k for k, v in state['attempts'].items() if now < v['next_try']}
                # Count all this user's active requests, including other work;
                # keep reservations absent from qstat until they are classified.
                reservations = sum(
                    statuses.get(j) not in ('complete', 'failed') and
                    state['requests'].get(j, {}).get('id') not in active
                    for j in submitted)
                slots = max_inflight - len(active) - reservations
                ready = select_ready(jobs, statuses, submitted, cooldown, slots)
                atomic(state_path, state)
                for job_id in ready:
                    try:
                        chain.dependencies(by_id[job_id], plan)
                        command = [str(Path(definition['runtime']) / 'envs/torch/bin/python'),
                                   str(plan / 'source/pegasus/runtime/chain.py'), 'submit-one',
                                   '--plan', str(plan), '--job', job_id]
                        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                        if result.returncode:
                            attempt = state['attempts'].setdefault(job_id, {'count': 0, 'next_try': 0})
                            attempt['count'] += 1
                            attempt['next_try'] = time.time() + min(900, 120 * attempt['count'])
                            event('submission_rejected', job=job_id, returncode=result.returncode,
                                  stdout=result.stdout[-1000:], stderr=result.stderr[-1500:])
                        else:
                            event('submitted', job=job_id, response=result.stdout.strip())
                    except Exception as exc:
                        # A reservation prevents retries after ambiguous acceptance.
                        reserved = (plan / 'submissions' / job_id).exists()
                        state['checks'][job_id] = {'status': 'unknown' if reserved else 'failed', 'error': repr(exc)}
                        event('submission_error', job=job_id, reserved=reserved, error=repr(exc))
                    atomic(state_path, state)
                statuses = {k: v['status'] for k, v in state['checks'].items()}
                blocked = blocked_jobs(jobs, statuses)
                state.update(updated=time.time(), pid=os.getpid(), active_requests=active,
                             max_inflight=max_inflight, blocked=sorted(blocked),
                             counts={s: sum(v == s for v in statuses.values()) for s in ('complete', 'failed', 'unknown', 'validating')})
                state['counts'].update(total=len(jobs), blocked=len(blocked), submitted=len(list((plan / 'submissions').iterdir())))
                terminal = {k for k, v in statuses.items() if v in ('complete', 'failed')} | blocked
                own_active = any(r['id'] in active for r in state['requests'].values())
                settled = all(r.get('final_log_check') for r in state['requests'].values())
                state['status'] = 'running'
                if len(terminal) == len(jobs) and not own_active and settled:
                    state['status'] = 'complete' if all(s == 'complete' for s in statuses.values()) and not blocked else 'finished_with_failures'
                atomic(state_path, state)
                if state['status'] != 'running':
                    event('controller_finished', status=state['status'], counts=state['counts'])
                    return 0 if state['status'] == 'complete' else 2
            except Exception as exc:
                # No submission or disappearance inference on untrusted qstat.
                state.update(status='scheduler_or_controller_error', updated=time.time(), last_error=repr(exc))
                atomic(state_path, state)
                event('controller_error', error=repr(exc))
            time.sleep(interval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--max-inflight', type=int, default=64)
    p.add_argument('--interval', type=int, default=30)
    a = p.parse_args()
    if not 1 <= a.max_inflight <= 190 or a.interval < 10:
        p.error('max-inflight must be 1..190 and interval >=10 seconds')
    raise SystemExit(run(a.plan, a.max_inflight, a.interval))


if __name__ == '__main__':
    main()
