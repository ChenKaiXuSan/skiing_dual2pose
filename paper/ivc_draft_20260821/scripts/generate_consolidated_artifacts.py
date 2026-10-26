"""Generate the consolidated angle table and relative temporal-error figure."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
try:
    from .journal_figure_style import JOURNAL_COLORS, apply_journal_style, style_axes
except ImportError:
    from journal_figure_style import JOURNAL_COLORS, apply_journal_style, style_axes

PAPER=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[3]
LOGS=ROOT/'logs/ivc_mmsports_extension'
CHECKPOINT='869a2217f8676c0ada75ed3c9a3c82a9b8efbb105749f6ffb8bef71e9172f50f'
END=r' \\'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def relative_series(rows, x_key, metric):
    ordered=sorted(rows,key=lambda r:float(r[x_key]))
    zero=[r for r in ordered if float(r[x_key])==0.]
    if len(zero)!=1: raise ValueError('Exactly one unperturbed row is required')
    baseline=float(zero[0][metric])
    if not math.isfinite(baseline) or baseline<=0: raise ValueError('Invalid unperturbed MPJPE')
    x=[float(r[x_key]) for r in ordered]
    values=[float(r[metric]) for r in ordered]
    if len(set(x))!=len(x) or not all(math.isfinite(v) for v in x+values):
        raise ValueError('Repeated or nonfinite plot data')
    return {'x':x,'mpjpe':values,'unperturbed_mpjpe':baseline,
            'relative_percent':[100.*(v-baseline)/baseline for v in values]}


def merged_angle_rows(view, statistics):
    lookup={r['angle_bin']:r for r in statistics}
    bins=[r['angle_bin'] for r in view]
    if len(set(bins))!=len(bins) or len(lookup)!=len(statistics) or set(bins)!=set(lookup):
        raise ValueError('Angle summary and inference bins must match')
    merged=[]
    for row in view:
        stat=lookup[row['angle_bin']]
        if int(row['sample_count'])!=4*int(stat['cluster_count']):
            raise ValueError('Expected four action observations per camera pair')
        for key in ['canonical_avg_mpjpe','fused_mpjpe']:
            if abs(float(row[key])-float(stat[key+'_mean']))>1e-6:
                raise ValueError('Angle summary and inference means differ')
        result={'angle_bin':row['angle_bin'],'camera_pairs':int(stat['cluster_count']),
            'sample_count':int(row['sample_count']),'canonical_avg_mpjpe':float(row['canonical_avg_mpjpe']),
            'fused_mpjpe':float(row['fused_mpjpe']),'mean_gain':float(stat['mean_gain_mpjpe']),
            'ci_low':float(stat['mean_gain_ci95_low']),'ci_high':float(stat['mean_gain_ci95_high']),
            'p_holm':float(stat['p_holm']),'rank_biserial':float(stat['rank_biserial'])}
        if not all(math.isfinite(v) for k,v in result.items() if k!='angle_bin'):
            raise ValueError('Nonfinite angle statistic')
        if not result['ci_low']<=result['mean_gain']<=result['ci_high']:
            raise ValueError('Invalid interval for reported mean gain')
        merged.append(result)
    return merged


def angle_table(rows):
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Accuracy and camera-pair bootstrap intervals by azimuth separation. Positive gain favors CanonFuse3D.}',
        r'\label{tab:view_angle}',r'\small',r'\setlength{\tabcolsep}{4pt}',
        r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{lrrrrl}',r'\toprule',
        r'Angle (deg) & Pairs & Canon. avg. & Fused & Mean gain & 95\% CI'+END,r'\midrule']
    for row in rows:
        angle=row['angle_bin'].replace('-','--')
        lines.append(f"{angle} & {row['camera_pairs']:,} & {row['canonical_avg_mpjpe']:.4f} & "
            f"{row['fused_mpjpe']:.4f} & {row['mean_gain']:.4f} & [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"+END)
    lines += [r'\bottomrule',r'\end{tabular}}',r'\vspace{0.3em}',r'\begin{minipage}{0.97\linewidth}',
        r'\footnotesize\textit{Note.} Each camera pair contributes four action-specific sequences; the bins cover all 64,440 test sequences. MPJPE and absolute gain use dataset coordinate units. Intervals resample camera pairs. Detailed paired tests and effect sizes appear in Supplementary Table~\ref{S-tab:view_angle_significance}.',
        r'\end{minipage}',r'\end{table}']
    return '\n'.join(lines)+'\n'


def temporal_sampling_figure(temporal, sampling, output):
    apply_journal_style()
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.7))
    data={}
    for ax, study, rows, x_key, xlabel, methods in [
        (axes[0],'offset',temporal,'offset_frames','Right-stream offset (frames)',
         [('fused_mpjpe','CanonFuse3D'),('canonical_avg_mpjpe','Canonical average')]),
        (axes[1],'rate',sampling,'rate_error_percent','Right-stream rate error (%)',
         [('fused_mpjpe','CanonFuse3D')])]:
        data[study]={}
        for metric,label in methods:
            series=relative_series(rows,x_key,metric);data[study][metric]=series
            fused=metric=='fused_mpjpe'
            ax.plot(series['x'],series['relative_percent'],label=label,
                color=JOURNAL_COLORS['blue' if fused else 'grey'],marker='o' if fused else 's',
                linestyle='-' if fused else '--',markerfacecolor=JOURNAL_COLORS['blue'] if fused else 'white')
        ax.axhline(0,color=JOURNAL_COLORS['black'],linestyle=':',linewidth=1,label='Unperturbed')
        ax.set(xlabel=xlabel,ylabel='Relative MPJPE change (%)')
        ax.legend(loc='upper left',fontsize=7)
        ax.set_title('(a) Temporal offset' if study=='offset' else '(b) Sampling-rate drift')
        style_axes(ax)
    fig.tight_layout()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output);plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'definition':'100*(MPJPE - method-specific zero)/method-specific zero','series':data},indent=2)+'\n')
    return data


def main():
    paths={'view':LOGS/'view_angle/view_angle_summary_last.csv',
        'statistics':LOGS/'view_angle/view_angle_significance_last.csv',
        'temporal':LOGS/'temporal_offset/temporal_offset_summary_last.csv',
        'sampling':LOGS/'sampling_rate/sampling_rate_summary_last.csv'}
    source={key:read_csv(path) for key,path in paths.items()}
    for name in ['view','temporal','sampling']:
        for row in source[name]:
            if row['checkpoint_sha256']!=CHECKPOINT: raise ValueError('Wrong checkpoint')
            count_key='total_sample_count' if name=='view' else 'sample_count'
            if int(row[count_key])!=64440: raise ValueError('Incomplete evaluation')
    merged=merged_angle_rows(source['view'],source['statistics'])
    if len(merged)!=6 or sum(r['sample_count'] for r in merged)!=64440: raise ValueError('Incomplete angle bins')
    if len(source['temporal'])!=7 or len(source['sampling'])!=7: raise ValueError('Incomplete temporal grid')
    table=PAPER/'tables/view_angle_combined.tex';table.write_text(angle_table(merged))
    figure=PAPER/'figures/extension/temporal_sampling.pdf'
    temporal_sampling_figure(source['temporal'],source['sampling'],figure)
    output=PAPER/'evidence/consolidated_artifacts.json'
    output.write_text(json.dumps({'sources':{str(path):sha256(path) for path in paths.values()},
        'generator_sha256':sha256(__file__),'merged_angle_rows':merged,
        'outputs':{str(path.relative_to(PAPER)):sha256(path) for path in [table,figure,figure.with_suffix('.json')]}},indent=2)+'\n')
    print('Generated six-row combined angle table and complete relative-change temporal panels.')


if __name__=='__main__': main()
