"""Check newly typeset numbers against compact experiment evidence."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
read = lambda p: json.loads(p.read_text())
checks = 0
def assert_rows(file, expected, digits):
    global checks
    lines = [line for line in file.read_text().splitlines() if '&' in line]
    rows = []
    for line in lines:
        nums = re.findall(r'(?<![\w.])\d+\.\d+', line)
        if len(nums) >= len(expected[0]):
            rows.append(nums[-len(expected[0]):])
    assert len(rows) == len(expected), (file, len(rows), len(expected))
    for actual, values in zip(rows, expected):
        assert actual == [f'{v:.{digits}f}' for v in values], (file, actual, values)
        checks += len(values)

conditions = ['clean','left_zero','right_zero','switch_left_to_right','switch_right_to_left']
variants = ['dynamic_residual','dynamic_no_residual','mean_residual','mean_no_residual',
            'left_residual','left_no_residual','right_residual','right_no_residual','canonical_average']
frozen = {}
expected = []
for dataset, count in [('unity',64440),('ski',30)]:
    frozen[dataset] = [read(ROOT/f'gate_{dataset}'/f'{c}.json') for c in conditions]
    for result in frozen[dataset]:
        assert result['default_reconstruction_max_abs_error'] == 0
        for scores in result['metrics'].values():
            for m in scores.values():
                assert m['sample_count'] == count
                assert all(np.isfinite(m[k]) for k in ['mpjpe','pa_mpjpe','acceleration_error'])
    expected.extend([[d['metrics'][v]['common13']['mpjpe'] for d in frozen[dataset]] for v in variants])
assert_rows(PAPER/'tables/gate_frozen_interventions.tex', expected, 4)

learned = []
for name in ['full','fixed_gate','no_output_residual']:
    d = read(ROOT/'gate_training'/f'{name}.json')
    assert d['status'] == 'complete_full_test'
    assert d['configuration']['seed'] == 42 and d['configuration']['epochs'] == 100
    assert d['configuration']['evaluation_batch_size'] == 256
    m = d['metrics']['common13']; assert m['sample_count'] == 64440
    learned.append(m)
learned.append(frozen['unity'][0]['metrics']['mean_no_residual']['common13'])
assert_rows(PAPER/'tables/gate_retraining.tex', [[m[k] for k in ['mpjpe','pa_mpjpe','acceleration_error']] for m in learned], 5)

mean_joint = np.asarray(frozen['unity'][0]['gate_diagnostics']['alpha_mean_by_time_joint']).mean(axis=0)
main = (PAPER/'main.tex').read_text()
for label, indices in [('Upper limbs',[2,3,4,5,12,13]),('Lower limbs',[6,7,8,9,10,11]),('All joints',list(range(15)))]:
    values=mean_joint[indices]
    row=next(line for line in main.splitlines() if line.startswith(label+' &'))
    assert re.findall(r'\d+\.\d+',row) == [f'{values.mean():.4f}',f'{values.std():.4f}',f'{1-values.mean():.4f}']
    checks += 3

table = (PAPER/'tables/fusion_efficiency_external.tex').read_text()
for method, first, second in [('stride','forward_only','total_with_ttt'),('deciwatch','forward_only','forward_only'),('metapose','learned_solver','total_with_adapters')]:
    d = read(ROOT/'external_efficiency'/f'{method}.json')
    assert d['status'] == 'complete_timing_audit'
    row = next(line for line in table.splitlines() if line.lower().startswith(method))
    for value in (d['total_parameters'],d['weight_bytes']):
        assert f'{value:,}' in row; checks += 1
    for key in (first,second):
        timing=d[key]
        assert len(timing['samples_ms']) == timing['repeats']
        assert np.isfinite(timing['samples_ms']).all()
        assert abs(np.median(timing['samples_ms'])-timing['median_ms']) < 1e-10
        assert f"{timing['median_ms']:.3f}" in row; checks += 1

assert hashlib.sha256((PAPER/'tables/native_reference.tex').read_bytes()).hexdigest() == 'b07555843db7c5a1f6aa68f6e2332591b6998011a0caf4724ea4b4c365034915'
# The approved layout revision removes empty Ski columns, not Unity values.
old = PAPER/'evidence/layout_revision_20260917/pre_revision/native_output_comparison.tex'
assert hashlib.sha256(old.read_bytes()).hexdigest() == '2f0604852dfe1a2d60c07e6e323a969b7ef9a923a5fb7822f6ce11a1bdf5c3d1'
old_rows = [line.split(' & ') for line in old.read_text().splitlines() if ' & No & ' in line]
new_text = (PAPER/'tables/native_output_comparison.tex').read_text()
new_rows = [line.split(' & ') for line in new_text.splitlines()
            if line.count(' & ') == 3 and not line.startswith('Method &')]
assert len(old_rows) == len(new_rows) == 16
assert 'Ski-PTZ-Pose' not in new_text
for before, after in zip(old_rows, new_rows):
    assert len(after) == 4
    assert [before[0], *before[2:4]] == after[:3]
    assert before[4].strip() == after[3].rstrip().removesuffix(r'\\').strip()
print(json.dumps(dict(status='passed', numeric_cells=checks,
                      original_benchmark_tables='canonical byte-identical; all 16 native Unity rows unchanged',
                      frozen_conditions=10, trained_configurations=3, external_timing_methods=3)))
