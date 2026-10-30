"""Runtime safety checks independent of a cluster or GPU."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def test_runtime_paths_follow_project_base(tmp_path):
    env=os.environ.copy();env['DUAL2POSE_PROJECT_BASE']=str(tmp_path)
    for key in ['DUAL2POSE_REPO','DUAL2POSE_DATA_ROOT','DUAL2POSE_RUNTIME']:
        env.pop(key,None)
    result=subprocess.check_output(['bash','-c','source "$1"; printf "%s\\n" "$DUAL2POSE_REPO" "$DUAL2POSE_POCKET3_ROOT" "$DUAL2POSE_PYTHON"','bash',str(ROOT/'pegasus/runtime/env.sh')],env=env,text=True)
    paths=result.splitlines()
    assert paths[0]==str(tmp_path/'Skiing_Canonical_DualView_3D_Pose_PyTorch')
    assert paths[1]==str(tmp_path/'Skiing_Canonical_DualView_3D_Pose_PyTorch_data/skiing_pocket3_20261012')
    assert paths[2]==str(tmp_path/'Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime/envs/torch/bin/python')

def test_smoke_refuses_existing_output(tmp_path):
    (tmp_path/'sentinel').write_text('keep')
    r=subprocess.run([sys.executable,str(ROOT/'pegasus/runtime/smoke.py'),'--output',str(tmp_path)],capture_output=True,text=True)
    assert r.returncode!=0 and 'FileExistsError' in r.stderr
    assert (tmp_path/'sentinel').read_text()=='keep'
