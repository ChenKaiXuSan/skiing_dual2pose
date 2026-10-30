"""Independent-window STRIDE shards; only complete coverage may be scored."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


def ranges(total,size):
    if total<=0 or size<=0:raise ValueError('Positive total and shard size required')
    return [(i,min(i+size,total)) for i in range(0,total,size)]


def check_ranges(total,spans):
    offset=0
    for start,end in sorted(spans):
        if start!=offset or end<=start or end>total:raise ValueError('Gap, overlap or invalid shard range')
        offset=end
    if offset!=total:raise ValueError('Incomplete shard coverage')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','shard','merge']);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepared',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--dataset-root',type=Path);p.add_argument('--stride-repo',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--shards-root',type=Path);a=p.parse_args()
    import numpy as np
    import torch
    from dual2pose.experiments.run_stride_external import prepare_inputs,evaluate_predictions
    from dual2pose.eval.stride_external import load_official_stride,COMMON
    from dual2pose.eval.main_baseline_data import sha256,write_json
    from dual2pose.eval.external_protocol import protocol_metadata
    torch.set_num_threads(2)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.action=='prepare':
        inputs,provenance=prepare_inputs(a.cache,a.dataset_root,limit=0,workers=4)
        np.save(out/'inputs17.npy',inputs);write_json(out/'input_provenance.json',provenance)
        write_json(out/'report.json',dict(status='prepared',input_sha256=sha256(out/'inputs17.npy'),provenance_sha256=sha256(out/'input_provenance.json'),sample_count=len(inputs)))
        return
    prepared=a.prepared.resolve();prep=json.loads((prepared/'report.json').read_text())
    if sha256(prepared/'inputs17.npy')!=prep['input_sha256'] or sha256(prepared/'input_provenance.json')!=prep['provenance_sha256']:raise ValueError('Prepared inputs changed')
    inputs=np.load(prepared/'inputs17.npy',mmap_mode='r');provenance=json.loads((prepared/'input_provenance.json').read_text());n=len(inputs)
    if a.action=='shard':
        if a.start is None or a.end is None or not 0<=a.start<a.end<=n:raise ValueError('Invalid shard range')
        adapter,info=load_official_stride(a.stride_repo,a.checkpoint,'cuda:0')
        info['source_commit']=subprocess.check_output(['git','-C',str(a.stride_repo),'rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','-C',str(a.stride_repo),'status','--porcelain'],text=True).strip():raise ValueError('Official source is modified')
        pred=np.lib.format.open_memmap(out/'predictions_common13.npy',mode='w+',dtype=np.float32,shape=(a.end-a.start,30,13,3));pred[:]=np.nan;began=time.time()
        with (out/'per_window.jsonl').open('x') as log:
            for i in range(a.start,a.end):
                if time.time()-began>22*3600:raise TimeoutError('Shard exceeded 22-hour compute budget')
                value,timing=adapter.refine(np.array(inputs[i:i+1]));pred[i-a.start]=value.numpy()[0][:,COMMON,:]
                log.write(json.dumps({'index':i,**timing})+'\n');log.flush()
        pred.flush()
        if not np.isfinite(pred).all():raise ValueError('Nonfinite shard')
        write_json(out/'report.json',dict(status='complete_shard',publication_eligible=False,start=a.start,end=a.end,full_count=n,input_sha256=prep['input_sha256'],provenance_sha256=prep['provenance_sha256'],model=info,prediction_sha256=sha256(out/'predictions_common13.npy'),seconds=time.time()-began))
        return
    records=[(f.parent,json.loads(f.read_text())) for f in a.shards_root.glob('*/report.json')]
    check_ranges(n,[(r['start'],r['end']) for _,r in records]);records.sort(key=lambda x:x[1]['start'])
    pred=np.lib.format.open_memmap(out/'predictions_common13.npy',mode='w+',dtype=np.float32,shape=(n,30,13,3));model=records[0][1]['model']
    for directory,r in records:
        if r['status']!='complete_shard' or r['input_sha256']!=prep['input_sha256'] or r['provenance_sha256']!=prep['provenance_sha256'] or r['model']!=model or r['full_count']!=n:raise ValueError('Shard provenance/model mismatch')
        f=directory/'predictions_common13.npy'
        if sha256(f)!=r['prediction_sha256']:raise ValueError('Shard prediction hash mismatch')
        array=np.load(f,mmap_mode='r')
        if array.shape!=(r['end']-r['start'],30,13,3) or not np.isfinite(array).all():raise ValueError('Bad shard shape or finite coverage')
        pred[r['start']:r['end']]=array
    pred.flush();(out/'inputs17.npy').symlink_to(prepared/'inputs17.npy');shutil.copy2(prepared/'input_provenance.json',out/'input_provenance.json')
    # No partial test scoring; global expected split size is enforced here.
    from dual2pose.eval.main_baseline_data import EXPECTED
    if provenance['split']!='test' or n!=EXPECTED[provenance['dataset']]['test']:raise ValueError('Only complete official test coverage can be scored')
    metrics=evaluate_predictions(a.cache,pred,inputs,provenance['dataset']);write_json(out/'metrics.json',metrics)
    write_json(out/'report.json',dict(**protocol_metadata(),status='complete',publication_eligible=True,sample_count=n,completed_count=n,model=model,input_sha256=prep['input_sha256'],prediction_sha256=sha256(out/'predictions_common13.npy'),shards=len(records),metrics=metrics))


if __name__=='__main__':main()
