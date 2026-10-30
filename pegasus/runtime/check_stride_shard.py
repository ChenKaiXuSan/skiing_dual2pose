"""Compare a single training-window shard to the validated serial STRIDE run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
from dual2pose.eval.main_baseline_data import sha256
ref=a.reference;prepared=a.output/'prepared';prepared.mkdir()
for name in ('inputs17.npy','input_provenance.json'):(prepared/name).symlink_to((ref/name).resolve())
(prepared/'report.json').write_text(json.dumps(dict(input_sha256=sha256(prepared/'inputs17.npy'),provenance_sha256=sha256(prepared/'input_provenance.json'))))
repo=Path(os.environ['DUAL2POSE_REPO'])
subprocess.run([os.environ['DUAL2POSE_PYTHON'],'-m','pegasus.runtime.stride_shards','shard','--prepared',str(prepared),'--start','0','--end','1','--stride-repo',str(repo/'.cache/external_baselines/STRIDE'),'--checkpoint',str(repo/'ckpt/external/stride_latest_epoch.bin'),'--output',str(a.output/'shard')],check=True)
import numpy as np
x=np.load(ref/'predictions_common13.npy');y=np.load(a.output/'shard/predictions_common13.npy')
np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6)
(a.output/'result.json').write_text(json.dumps(dict(status='passed',publication_eligible=False,scope='one training window; serial versus sharded STRIDE',max_abs_difference=float(np.abs(x-y).max()))))
