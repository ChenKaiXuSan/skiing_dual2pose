from pathlib import Path
import json,re,sys
import numpy as np
from dual2pose.eval.main_baseline_data import sha256,write_json
from dual2pose.eval.render_main_baseline_tables import load_records,DIRECT,GEOMETRY,LABELS
r=Path('logs/ivc_mmsports_extension/main_baselines/20260906')
paper=Path('paper/ivc_draft_20260821')
checks=[]
for p in sorted((r/'cache').glob('*/*/manifest.json')):
 m=json.loads(p.read_text())
 for name,digest in m['files'].items():
  assert sha256(p.parent/name)==digest,(p,name)
  checks.append(str((p.parent/name).relative_to(r)))
for p in sorted((r/'geometry').glob('*/*/manifest.json')):
 m=json.loads(p.read_text())
 for name,digest in m['output_sha256'].items():
  assert sha256(p.parent/name)==digest,(p,name)
  checks.append(str((p.parent/name).relative_to(r)))
 assert m['coverage']['dlt_valid']==1. and m['coverage']['robust_valid']==1.
records=load_records(r/'results')
packaged=json.loads((paper/'evidence/results/main_baselines__results.json').read_text())
assert records==packaged['records']
metric_cells=0
for dataset,filename in [('unity','native_reference.tex'),('ski','ski_poseptz_reference.tex')]:
 text=(paper/'tables'/filename).read_text()
 subsets=['all15','common13'] if dataset=='unity' else ['common13']
 for record in [x for x in records if x['dataset']==dataset]:
  label=LABELS[record['method']]
  lines=[line for line in text.splitlines() if line.startswith(label+' & ')]
  assert len(lines)==1,(dataset,label)
  values=lines[0].removesuffix(r' \\').split(' & ')[1:]
  actual=[float(re.sub(r'\\textbf\{([^}]+)\}',r'\1',v)) for v in values]
  expected=[float(f"{record['result']['metrics'][subset][key]:.4f}") for subset in subsets for key in ['mpjpe','acceleration_error']]
  assert actual==expected,(dataset,label,actual,expected)
  metric_cells+=len(actual)
  if record['result'].get('checkpoint'):
   assert sha256(record['result']['checkpoint'])==record['result']['checkpoint_sha256']
for dataset in ['unity','ski']:
 for method in ['aligned_average','quality_weighted']:
  old=json.loads((r/'results_before_degenerate_guard'/dataset/f'{method}.json').read_text())
  new=json.loads((r/'results'/dataset/f'{method}.json').read_text())
  assert old['metrics']==new['metrics'],(dataset,method,'clean input metrics changed after degenerate guard')
aux=(paper/'main.aux').read_text();pages={}
for label in ['native_reference','ski_poseptz_reference']:
 match=re.search(r'\\newlabel\{tab:'+label+r'\}\{\{(\d+)\}\{(\d+)\}',aux)
 assert match
 pages[label]={'table':int(match.group(1)),'page':int(match.group(2))}
log=(paper/'main.log').read_text()
assert not re.search(r'Overfull|undefined references|Citation .+ undefined|LaTeX Error',log)
pdftext=Path('/tmp/ivc-main-baselines-main.txt').read_text()
for label,info in pages.items():
 page=pdftext.split('\f')[info['page']-1]
 assert f"Table {info['table']}:" in page
 for needle in ['Cross-view MLP','SmoothNet','Reprojection-gated DLT','DLT-residual MLP','CanonFuse3D']:
  assert needle in page,(label,needle)
# Re-running the legacy native-table entry point must preserve the expanded tables.
from paper.ivc_draft_20260821.scripts import generate_extension_artifacts as legacy
before={name:sha256(paper/'tables'/name) for name in ['native_reference.tex','ski_poseptz_reference.tex']}
legacy.native_reference_table([])
assert before=={name:sha256(paper/'tables'/name) for name in before}
tests_log=(r/'final_tests.log').read_text()
assert re.search(r'Ran 37 tests in',tests_log) and tests_log.rstrip().endswith('OK')
result={'status':'verified','focused_tests_passed':37,'method_records':len(records),'metric_cells_checked':metric_cells,
        'raw_and_geometry_files_rehashed':len(checks),'table_pages':pages,'pdf_sha256':sha256(paper/'main.pdf'),
        'latex_no_undefined_or_overfull':True,'visual_inspection':['page25_table2','page26_table3'],
        'legacy_regeneration_preserves_expanded_tables':True,'clean_heuristic_values_unchanged_after_guard':True,
        'tests_log':str(r/'final_tests.log'),'independent_review':str(r/'final_review.md')}
write_json(r/'verification.json',result)
write_json(paper/'evidence/main_baselines_verification.json',result)
print(json.dumps(result,indent=2))
