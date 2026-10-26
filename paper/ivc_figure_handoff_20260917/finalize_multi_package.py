"""Verify exported samples, add selection guidance, and refresh checksums/ZIP.

Run after build_multi_samples.py and document rendering.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image
import torch

P=Path(__file__).resolve().parent
ROOT=P.parents[1]
sys.path.insert(0,str(ROOT))
from dual2pose.experiments.run_main_baselines import ReferenceModel, CHECKPOINTS
torch.set_num_threads(2)
manifest=json.loads((P/'multi_samples/manifest.json').read_text())
assert manifest['sample_count']==12
assert manifest['by_dataset']=={'unity':4,'ski':4,'author':4}
models={k:ReferenceModel(v).model.eval() for k,v in CHECKPOINTS.items()}
stats=[];vectors=0;attention_parity={};output_parity={}
for m in manifest['samples']:
    folder=P/'multi_samples'/m['dataset']/m['id']
    with np.load(folder/'arrays.npz') as z:a={k:z[k] for k in z.files}
    J=m['num_joints'];T=30
    assert a['fused'].shape==(T,J,3)
    assert a['alpha'].shape==(T,J,1)
    assert all(np.isfinite(v).all() for v in a.values())
    assert np.array_equal(np.loadtxt(folder/'gate_alpha.csv',delimiter=',',skiprows=1)[:,2:],a['alpha'][:,:,0])
    assert np.allclose(a['fused'],a['alpha']*a['base_left']+(1-a['alpha'])*a['base_right']+a['output_residual'],atol=1e-6,rtol=0)
    for key in ['attention_left_to_right','attention_right_to_left']:
        assert a[key].shape==(T,J,J)
        assert np.allclose(a[key].sum(-1),1,atol=2e-6)
    # Independent single-sample inference from exported canonical inputs verifies
    # that flattened attention did not select a different sample or frame.
    l,r=[torch.from_numpy(a[k])[None] for k in ['left','right']]
    with torch.no_grad():f,aux=models['ski' if m['dataset']=='ski' else 'unity'](l,r,return_components=True)
    attention_parity[m['id']]=float(np.max(np.abs(aux['attn_l'].numpy().reshape(T,J,J)-a['attention_left_to_right'])))
    output_parity[m['id']]=float(np.max(np.abs(f[0].numpy()-a['fused'])))
    assert attention_parity[m['id']]<2e-5
    assert output_parity[m['id']]<2e-5
    for n in m['assets']:
        Image.open(folder/f'{n}.png').verify();ET.parse(folder/f'{n}.svg')
        assert (folder/f'{n}.pdf').read_bytes().startswith(b'%PDF')
    for v in ['left','right']:Image.open(folder/f'rgb_{v}.png').verify()
    if m['dataset']=='author':
        assert 'reference' not in a and not (folder/'local_errors.csv').exists()
        status='无 GT，仅定性展示';avg=fused=''
    else:
        assert a['reference'].shape==(T,J,3)
        t=m['window_index_0based'];avg=float(np.linalg.norm(a['canonical_average'][t]-a['reference'][t],axis=-1).mean());fused=float(np.linalg.norm(a['fused'][t]-a['reference'][t],axis=-1).mean())
        status='本帧误差降低，可作局部对照' if fused<avg else '本帧误差增大，不宜展示改善'
    stats.append([m['id'],m['dataset'],m['source_frame'],m['reference_kind'],avg,fused,status])
    vectors+=len(m['assets'])

with (P/'multi_samples/sample_selection.csv').open('w') as h:
    w=csv.writer(h);w.writerow(['id','dataset','source_frame','reference','average_error','fused_error','use_note']);w.writerows(stats)
guidance=['## 本帧对照与选图提示','', '以下数值是预选主帧的平均关节距离，单位为各数据集 canonical coordinate units；不能跨数据集比较，不能作为总体性能结论。保留所有预选样本，包括效果不佳者。','', '|编号|原始帧|Canonical average|CanonFuse3D|选图提示|','|---|---|---|---|---|']
for sid,ds,fr,ref,avg,fused,status in stats:
    guidance.append(f'|{sid}|{fr}|{avg:.5f}|{fused:.5f}|{status}|' if avg!='' else f'|{sid}|{fr}|—|—|{status}|')
guidance+=['','U01 适合说明同一 Unity 示例的输入与输出改善；U02 保留为不利示例，不应用来说明普遍改善。S01–S04 可以展示相对于构造参考的本帧变化；A01–A04 适合展示室内外双视角输入和定性输出。选择其中某组用于论文时，仍应保留其样本范围说明。']
idx=P/'MULTI_SAMPLE_INDEX.md';s=idx.read_text().split('\n## 本帧对照与选图提示')[0];idx.write_text(s+'\n'+'\n'.join(guidance)+'\n')
gallery=P/'ASSET_GALLERY.html';s=gallery.read_text()
for sid,ds,fr,ref,avg,fused,status in stats:
    needle=f'<h3>{sid} · frame {fr} · {ref}</h3>'
    note=f'<p class="use-note"><strong>{status}</strong>'+(f'；canonical average {avg:.5f} → fused {fused:.5f}（本帧）' if avg!='' else '')+'</p>'
    if needle+note not in s:s=s.replace(needle,needle+note)
gallery.write_text(s)

# Legacy scientific assets must be unchanged; document/gallery updates are intentional.
old=[]
for line in (P/'SHA256SUMS.txt').read_text().splitlines():
    digest,name=line.split('  ',1)
    if name.startswith(('assets/','data/')):
        assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
        old.append(name)
report={'status':'passed','new_samples':12,'by_dataset':manifest['by_dataset'],'new_vector_asset_sets':vectors,'new_rgb_images':24,
        'legacy_scientific_files_unchanged':len(old),'independent_attention_max_abs_error':attention_parity,'independent_output_max_abs_error':output_parity,
        'checks':['Finite arrays and expected joint counts','Gate CSV exact equality','Fusion reconstruction','Attention shape and row normalization',
                  'Independent sample/frame attention and output parity','All PNG decode and SVG XML parse','PDF file signatures','No GT or error file for author samples'],
        'unfavorable_examples':[x[0] for x in stats if x[4]!='' and x[5]>x[4]],
        'scope':'Fixed illustrative samples. New CPU inference from frozen checkpoints; not retraining or full-split reevaluation.'}
(P/'multi_samples/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if '--pack' in sys.argv:
    files=sorted(x for x in P.rglob('*') if x.is_file() and x.name!='SHA256SUMS.txt')
    (P/'SHA256SUMS.txt').write_text(''.join(f'{hashlib.sha256(x.read_bytes()).hexdigest()}  {x.relative_to(P)}\n' for x in files))
    zp=P.with_suffix('.zip')
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for x in sorted(P.rglob('*')):
            if x.is_file():z.write(x,x.relative_to(P.parent))
    with zipfile.ZipFile(zp) as z:assert z.testzip() is None
    print('ZIP',zp.stat().st_size,'bytes',len(files)+1,'files')
print(json.dumps(report,ensure_ascii=False,indent=2))
