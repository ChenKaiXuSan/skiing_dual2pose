"""Render expanded IVC Tables 2/3 only from complete measured results."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from dual2pose.eval.main_baseline_data import sha256, write_json, require_current_protocol

DIRECT = ['left_canonical','right_canonical','canonical_avg','aligned_average',
          'quality_weighted','mlp','tcn','smoothnet','canonfuse3d']
GEOMETRY = ['dlt','robust_dlt','dlt_mlp']
LABELS = {'left_canonical':'Left canonical','right_canonical':'Right canonical',
          'canonical_avg':'Canonical average','aligned_average':r'Aligned average~\cite{umeyama1991least}',
          'quality_weighted':'3D quality-weighted fusion','mlp':'Cross-view MLP',
          'tcn':r'Temporal fusion (local TCN)~\cite{bai2018empirical}','smoothnet':r'Avg. + SmoothNet~\cite{zeng2022smoothnet}',
          'canonfuse3d':'CanonFuse3D','dlt':r'Calibrated DLT~\cite{hartley2004multiple}',
          'robust_dlt':r'Reprojection-gated DLT~\cite{hartley2004multiple}',
          'dlt_mlp':r'DLT-residual MLP~\cite{hartley2004multiple}'}
METRICS = ['mpjpe', 'pa_mpjpe', 'acceleration_error']
END=r' \\'

def load_records(results):
    records=[]
    for dataset in ['unity','ski']:
        for method in DIRECT+GEOMETRY:
            path=Path(results)/dataset/f'{method}.json'
            record=json.loads(path.read_text())
            require_current_protocol(record.get("provenance", {}))
            if record['method']!=method: raise ValueError(f'Method mismatch: {path}')
            if dataset == 'ski' and method in GEOMETRY and not record['provenance'].get('ski_prediction_root_centered', False):
                raise ValueError(f'Ski geometric prediction must use the documented pelvis-relative convention: {path}')
            subsets=['common13']
            for subset in subsets:
                metrics=record['metrics'][subset]
                if metrics['sample_count']!=(64440 if dataset=='unity' else 30):
                    raise ValueError(f'Incomplete evaluation: {path}/{subset}')
                joints = 15 if subset == 'all15' else 13
                expected_points = metrics['sample_count'] * 30 * joints
                expected_accel = metrics['sample_count'] * 28 * joints
                if metrics['point_count'] != expected_points or metrics['acceleration_point_count'] != expected_accel:
                    raise ValueError(f'Incorrect frame/joint denominator: {path}/{subset}')
                if metrics.get('pa_point_count') != expected_points or metrics.get('pa_frame_count') != metrics['sample_count'] * 30:
                    raise ValueError(f'Incorrect PA-MPJPE denominator: {path}/{subset}')
                if any(not math.isfinite(metrics.get(k, float('nan'))) or metrics[k]<0 for k in METRICS):
                    raise ValueError(f'Invalid metrics: {path}/{subset}')
            records.append({'dataset':dataset,'method':method,'result':record,
                            'source_path':str(path.resolve()),'source_sha256':sha256(path)})
    return records

def render_table(dataset,records):
    lookup={r['method']:r['result'] for r in records if r['dataset']==dataset}
    subsets=['common13']
    ncols=1+3*len(subsets)
    caption=(r'Expanded Unity comparison on all 64,440 held-out camera-pair sequences. '
             if dataset=='unity' else r'Expanded Ski-PTZ-Pose comparison on all 30 fixed test camera-pair sequences. ')
    caption+=r'Lower values are better for all errors. Bold identifies the lowest value within each input group.'
    caption+=(r' Citations for aligned averaging, DLT variants, and the local TCN identify technical components or background; '
              r'these study-specific combinations are not reproductions of published fusion systems.')
    label='native_reference' if dataset=='unity' else 'ski_poseptz_reference'
    lines=[r'\begin{table}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',
           (r'\footnotesize' if dataset=='unity' else r'\small'),r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}l'+'rrr'*len(subsets)+'@{}}',r'\toprule']
    lines.append('Method' + r' & MPJPE $\downarrow$ & PA-MPJPE $\downarrow$ & Accel. $\downarrow$'*len(subsets)+END)
    for title,methods in [(r'3D pose inputs only',DIRECT),(r'Additional estimated 2D and camera calibration',GEOMETRY)]:
        lines.extend([r'\midrule',r'\multicolumn{'+str(ncols)+r'}{l}{\textit{'+title+'}}'+END,r'\midrule'])
        minima={(s,k):min(lookup[m]['metrics'][s][k] for m in methods) for s in subsets for k in METRICS}
        for method in methods:
            values=[]
            for subset in subsets:
                for metric in METRICS:
                    value=lookup[method]['metrics'][subset][metric]
                    text=f'{value:.4f}'
                    if value==minima[subset,metric]: text=r'\textbf{'+text+'}'
                    values.append(text)
            lines.append(LABELS[method]+' & '+' & '.join(values)+END)
    note=(r'Errors use dataset coordinate units in independent per-frame orthonormal body frames on the common 13 joints; '
          r'30 frames per sequence, with the final batch retained. Primary MPJPE compares output directly with GT without output alignment. '
          r'The two input groups have different information requirements. '
          r'PA uses a separate per-frame similarity fit for each joint subset, with no reflection; MPJPE and acceleration use predictions before this evaluation-only fit. ')
    if dataset=='ski':
        note+=(r'The target is a constructed annotation reference, not independent motion capture. '
               r'Geometric predictions are first centered at their own per-frame pelvis, then independently body-canonicalized; this preprocessing uses no test-fitted similarity map. ')
    note+=r'Raw-space single-view metrics remain in the archived evidence.'
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\vspace{0.4em}',r'\begin{minipage}{0.98\linewidth}',
                  r'\footnotesize\textit{Note.} '+note,r'\end{minipage}',r'\end{table}'])
    return '\n'.join(lines)+'\n'

def render_package(records,paper):
    paper=Path(paper)
    (paper/'tables').mkdir(parents=True,exist_ok=True)
    for dataset,filename in [('unity','native_reference.tex'),('ski','ski_poseptz_reference.tex')]:
        (paper/'tables'/filename).write_text(render_table(dataset,records))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--paper',type=Path,required=True)
    args=parser.parse_args(); records=load_records(args.results)
    evidence=args.paper/'evidence'; (evidence/'results').mkdir(parents=True,exist_ok=True)
    json_path=evidence/'results/main_baselines__results.json'
    write_json(json_path,{'records':records,'renderer_sha256':sha256(__file__)})
    rows=[]
    for item in records:
        result=item['result']
        for subset,metrics in result['metrics'].items():
            rows.append({'dataset':item['dataset'],'method':item['method'],
                         'input_group':'3D_only' if item['method'] in DIRECT else '2D_and_calibration',
                         'joint_subset':subset,**metrics,'source_sha256':item['source_sha256']})
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (evidence/'results/main_baselines__summary.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    render_package(records,args.paper)
    write_json(evidence/'main_baselines_manifest.json',{
        'run_root':str(args.results.parent.resolve()),'method_count_per_dataset':12,'direct_group_rows':9,
        'additional_input_rows':3,'record_count':len(records),'metric_rows':len(rows),
        'source_results':[{k:v for k,v in item.items() if k!='result'} for item in records],
        'packaged_result_sha256':sha256(json_path),'renderer_sha256':sha256(__file__),
        'tables':{f:sha256(args.paper/'tables'/f) for f in ['native_reference.tex','ski_poseptz_reference.tex']}})
    print(f'Generated 24 measured method rows / {len(rows)} joint-subset metric rows.')

if __name__=='__main__': main()
