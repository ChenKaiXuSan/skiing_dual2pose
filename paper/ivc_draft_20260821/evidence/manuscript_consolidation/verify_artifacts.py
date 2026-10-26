from pathlib import Path
import csv,json,re,tempfile
import subprocess
from dual2pose.eval.main_baseline_data import sha256,write_json
from paper.ivc_draft_20260821.scripts import generate_consolidated_artifacts as new
from paper.ivc_draft_20260821.scripts import generate_extension_artifacts as legacy

r=Path('logs/ivc_mmsports_extension/manuscript_consolidation/20260906')
p=Path('paper/ivc_draft_20260821')
before=json.loads((r/'before_manifest.json').read_text())
unchanged=[]
for name,digest in before.items():
 if name.startswith('tables/'):
  assert sha256(p/name)==digest,name
  unchanged.append(name)
assert len(unchanged)==12
proof=json.loads((p/'evidence/consolidated_artifacts.json').read_text())
for name,digest in proof['sources'].items(): assert sha256(name)==digest
for name,digest in proof['outputs'].items(): assert sha256(p/name)==digest
assert proof['generator_sha256']==sha256(new.__file__)
plot=json.loads((p/'figures/extension/temporal_sampling.json').read_text())
plot_points=0
for study,filename,xkey in [('offset','temporal_offset/temporal_offset_summary_last.csv','offset_frames'),
                            ('rate','sampling_rate/sampling_rate_summary_last.csv','rate_error_percent')]:
 with (new.LOGS/filename).open(newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==7
 for metric,series in plot['series'][study].items():
  expected=new.relative_series(rows,xkey,metric)
  assert expected==series
  plot_points+=len(series['x'])
assert plot_points==21
combined=(p/'tables/view_angle_combined.tex').read_text()
for row in proof['merged_angle_rows']:
 line=next(x for x in combined.splitlines() if x.startswith(row['angle_bin'].replace('-','--')+' & '))
 expected=(f"{row['angle_bin'].replace('-','--')} & {row['camera_pairs']:,} & "
           f"{row['canonical_avg_mpjpe']:.4f} & {row['fused_mpjpe']:.4f} & {row['mean_gain']:.4f} & "
           f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"+new.END)
 assert line==expected
# The older generator must retain the new relative-change plot behavior.
with tempfile.TemporaryDirectory() as tmp:
 old_paper=legacy.PAPER;legacy.PAPER=Path(tmp)
 try:
  temporal=legacy.read_csv(new.LOGS/'temporal_offset/temporal_offset_summary_last.csv')
  sampling=legacy.read_csv(new.LOGS/'sampling_rate/sampling_rate_summary_last.csv')
  view=legacy.read_csv(new.LOGS/'view_angle/view_angle_summary_last.csv')
  masking=[]
  for mode in ['left','right','both']:
   rows=legacy.read_csv(new.LOGS/f'masking/{mode}/occlusion_summary_last.csv')
   for row in rows:row['view_mode']=mode
   masking.extend(rows)
  legacy.figures(temporal,sampling,view,masking)
  assert json.loads((Path(tmp)/'figures/extension/temporal_sampling.json').read_text())==plot
 finally:legacy.PAPER=old_paper

def labels(path):
 return {name:{'number':number,'page':int(page)} for name,number,page in
    re.findall(r'\\newlabel\{((?:tab|fig):[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',path.read_text())}
main=labels(p/'main.aux');sup=labels(p/'supplementary.aux')
assert len([x for x in main if x.startswith('tab:')])==9
assert len([x for x in main if x.startswith('fig:')])==9
assert len([x for x in sup if x.startswith('tab:')])==5
assert len([x for x in sup if x.startswith('fig:')])==2
assert main['fig:temporal_alignment']['number']=='8'
assert main['tab:frontend_adaptation']['number']=='8'
assert sup['tab:temporal_alignment']['number']=='S5'
assert sup['fig:frontend_adaptation']['number']=='S2'
assert main['tab:native_reference']['number']=='2' and main['tab:ski_poseptz_reference']['number']=='3'
for label in ['tab:view_angle_significance','tab:temporal_sampling','tab:masking_extension','tab:nview_scaling','tab:temporal_alignment','fig:view_angle_extension','fig:frontend_adaptation']:
 assert label not in main and label in sup,label
body=(p/'main.tex').read_text()+(p/'sections/p1_results.tex').read_text()+combined
for label in re.findall(r'\\ref\{S-([^}]+)\}',body): assert label in sup,label
for term in ['3.3\\% worse','28.4\\%','negative integer lead','41.5\\%','13.2\\%']:
 assert term in body,term
page_counts={};pdf_hashes={};external_links=0
for name in ['main','supplementary']:
 log=(p/f'{name}.log').read_text()
 assert not re.search(r'Overfull|undefined references|Citation .+ undefined|LaTeX Error|multiply defined',log)
 info=subprocess.check_output(['pdfinfo',str(p/f'{name}.pdf')],text=True)
 page_counts[name]=int(re.search(r'Pages:\s+(\d+)',info).group(1))
 pdf_hashes[name]=sha256(p/f'{name}.pdf')
 if name=='main':
  xml=subprocess.check_output(['pdftohtml','-xml','-stdout','-i',str(p/'main.pdf')],text=True)
  external_links=len([x for x in re.findall(r'<a href="([^"]+)"',xml) if 'supplementary' in x])
 text=Path(f'/tmp/ivc-consolidated-{name}.txt').read_text()
 assert '??' not in text
assert external_links>=5
# Main-table results and provenance package are unchanged.
pack=json.loads((p/'evidence/results/main_baselines__results.json').read_text())
assert len(pack['records'])==24
pa_columns=sum((p/'tables'/f).read_text().count('PA-MPJPE') for f in ['native_reference.tex','ski_poseptz_reference.tex'])
assert pa_columns==3
for record in pack['records']: assert sha256(record['source_path'])==record['source_sha256']
tests=(r/'final_tests.log').read_text()
assert 'Ran 50 tests in' in tests and tests.rstrip().endswith('OK')
result={'status':'verified','main_page_count':page_counts['main'],'supplementary_page_count':page_counts['supplementary'],
 'previous_main_page_count':52,'main_tables':9,'main_figures':9,'supplementary_tables':5,'supplementary_figures':2,
 'main_labels':main,'supplementary_labels':sup,'all_12_original_table_files_unchanged':True,
 'main_tables_2_3_with_pa_unchanged':True,'source_result_records_unchanged':24,
 'combined_angle_rows_checked':6,'relative_curve_points_checked':plot_points,
 'legacy_temporal_generator_matches':True,'cross_pdf_links_checked':external_links,'focused_tests_passed':50,
 'latex_no_undefined_or_overfull':True,'negative_findings_retained':True,
 'visual_inspection':{'main':[29,30,34,37],'supplementary':[1,2,5,7]},
 'pdf_sha256':pdf_hashes,'source_manifest_sha256':sha256(p/'evidence/consolidated_artifacts.json'),
 'backup_root':str((r/'before').resolve())}
write_json(r/'verification.json',result)
write_json(p/'evidence/manuscript_consolidation_verification.json',result)
print(json.dumps(result,indent=2))
