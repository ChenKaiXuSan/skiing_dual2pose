from pathlib import Path
import csv,json,re
from dual2pose.eval.main_baseline_data import sha256,write_json
from dual2pose.eval.render_main_baseline_tables import load_records,LABELS,METRICS

r=Path('logs/ivc_mmsports_extension/main_baselines/20260906/pa_mpjpe')
paper=Path('paper/ivc_draft_20260821')
records=load_records(r/'results')
packaged=json.loads((paper/'evidence/results/main_baselines__results.json').read_text())
assert records==packaged['records']
assert len(records)==24
checkpoints=set();sources=set();input_files=set();max_delta=0.;cells=0
snapshots=list((r/'source_at_evaluation').glob('*.py'))
for row in records:
 result=row['result'];audit=result['pa_evaluation']
 assert sha256(audit['original_result'])==audit['original_result_sha256']
 original=json.loads(Path(audit['original_result']).read_text())
 if result.get('checkpoint'):
  assert sha256(result['checkpoint'])==result['checkpoint_sha256']
  checkpoints.add(result['checkpoint'])
 for path,digest in audit['source_sha256'].items():
  assert sha256(path)==digest or any(sha256(p)==digest for p in snapshots),(path,digest)
  sources.add((path,digest))
 for subset,metrics in result['metrics'].items():
  assert metrics['pa_degenerate_prediction_frames']==0
  for key in ['mpjpe','acceleration_error']:
   delta=abs(original['metrics'][subset][key]-metrics[key])
   max_delta=max(max_delta,delta)
   assert delta<=5e-7
   assert f'{original["metrics"][subset][key]:.4f}'==f'{metrics[key]:.4f}'
 for name in ['cache','geometry']:
  p=r.parent/name/row['dataset']/'test/manifest.json'
  assert sha256(p)==audit[name+'_manifest_sha256']
for dataset in ['unity','ski']:
 for name,key in [('cache','files'),('geometry','output_sha256')]:
  p=r.parent/name/dataset/'test/manifest.json';m=json.loads(p.read_text())
  for filename,digest in m[key].items():
   assert sha256(p.parent/filename)==digest
   input_files.add(str(p.parent/filename))
for dataset,filename in [('unity','native_reference.tex'),('ski','ski_poseptz_reference.tex')]:
 text=(paper/'tables'/filename).read_text()
 subsets=['all15','common13'] if dataset=='unity' else ['common13']
 assert text.count('PA-MPJPE')==len(subsets)
 for row in [x for x in records if x['dataset']==dataset]:
  line=next(x for x in text.splitlines() if x.startswith(LABELS[row['method']]+' & '))
  values=line.removesuffix(r' \\').split(' & ')[1:]
  actual=[float(re.sub(r'\\textbf\{([^}]+)\}',r'\1',v)) for v in values]
  expected=[float(f"{row['result']['metrics'][subset][key]:.4f}") for subset in subsets for key in METRICS]
  assert actual==expected,(row['method'],actual,expected)
  cells+=len(actual)
assert cells==108
csv_rows=list(csv.DictReader((paper/'evidence/results/main_baselines__summary.csv').open()))
assert len(csv_rows)==36 and all(float(x['pa_mpjpe'])>=0 for x in csv_rows)
aux=(paper/'main.aux').read_text();pages={}
pdftext=Path('/tmp/ivc-pa-main.txt').read_text()
for label in ['native_reference','ski_poseptz_reference']:
 match=re.search(r'\\newlabel\{tab:'+label+r'\}\{\{(\d+)\}\{(\d+)\}',aux)
 assert match
 pages[label]={'table':int(match.group(1)),'page':int(match.group(2))}
 page=pdftext.split('\f')[int(match.group(2))-1]
 assert f'Table {match.group(1)}:' in page
 assert page.count('PA-MPJPE') >= (2 if label=='native_reference' else 1)
 for name in ['Cross-view MLP','CanonFuse3D','DLT-residual MLP','SmoothNet']:
  assert name in page
log=(paper/'main.log').read_text()
assert not re.search(r'Overfull|undefined references|Citation .+ undefined|LaTeX Error',log)
from paper.ivc_draft_20260821.scripts import generate_extension_artifacts as legacy
before={f:sha256(paper/'tables'/f) for f in ['native_reference.tex','ski_poseptz_reference.tex']}
legacy.native_reference_table([])
assert before=={f:sha256(paper/'tables'/f) for f in before}
tests=(r/'final_tests.log').read_text()
assert 'Ran 46 tests in' in tests and tests.rstrip().endswith('OK')
parity=json.loads((r/'official_parity.json').read_text())
assert parity['status']=='verified'
assert all(x['absolute_difference']<1e-12 for x in parity['nondegenerate_formula_parity'].values())
result={'status':'verified','method_records':len(records),'joint_subset_records':len(csv_rows),
 'displayed_metric_cells_verified':cells,'new_pa_mpjpe_cells':36,'existing_rounded_metric_cells_unchanged':72,
 'max_existing_metric_absolute_difference':max_delta,'checkpoint_files_rehashed':len(checkpoints),
 'evaluation_source_versions_verified':len(sources),'test_cache_and_geometry_files_rehashed':len(input_files),
 'focused_tests_passed':46,'official_formula_parity':parity,'no_dropped_or_degenerate_test_frames':True,
 'table_pages':pages,'pdf_sha256':sha256(paper/'main.pdf'),'latex_no_undefined_or_overfull':True,
 'visual_inspection':['page25_table2','page26_table3'],
 'legacy_regeneration_preserves_pa_columns':True,'run_root':str(r.resolve()),
 'previous_verification':str((r/'manuscript_before/evidence/main_baselines_verification.json').resolve())}
write_json(r/'verification.json',result)
write_json(paper/'evidence/main_baselines_verification.json',result)
write_json(paper/'evidence/main_baselines_pa_verification.json',result)
print(json.dumps(result,indent=2))
