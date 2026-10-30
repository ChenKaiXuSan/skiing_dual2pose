"""Bounded GPU checks of real training/evaluation entrypoints; fresh outputs only."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
repo=Path(os.environ['DUAL2POSE_REPO']);py=sys.executable;state={'status':'running','publication_eligible':False,'steps':[],'started_unix':time.time()}
def save():
 tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'status.json')
def step(name,args,env=None):
 item={'name':name,'command':args,'status':'running','started_unix':time.time()};state['steps'].append(item);save()
 with (out/(name+'.log')).open('w') as log:
  try:r=subprocess.run(args,cwd=repo,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=900);item.update(status='passed' if r.returncode==0 else 'failed',returncode=r.returncode)
  except subprocess.TimeoutExpired:item.update(status='failed',error='900 second limit exceeded')
 item['finished_unix']=time.time();save()
step('gpu',[py,'-c','import torch; print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(); x=torch.randn(32,32,device="cuda",requires_grad=True); x.square().mean().backward(); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0))'])
if state['steps'][-1]['status']!='passed':state['status']='failed';save();raise SystemExit(1)
step('unity',[py,'-m','dual2pose.train_unity','data.batch_size=2','data.num_workers=0','train.max_epochs=1','+train.limit_train_batches=1','+train.limit_val_batches=1','+train.limit_test_batches=1','log_path='+str(out/'unity')])
step('ski',[py,'-m','dual2pose.train_ski_poseptz','--batch-size','2','--num-workers','0','--max-epochs','1','--limit-train-batches','1','--limit-test-batches','1','--output-dir',str(out/'ski')])
ckpt=repo/'logs/per_frame_common13_v1/main_20261028T053836Z/unity/checkpoints/66-0.0287-0.0639.ckpt'
step('pocket3',[py,'-m','dual2pose.pocket3.train','--dataset-root',os.environ['DUAL2POSE_POCKET3_ROOT'],'--ckpt',str(ckpt),'--epochs','1','--patience','1','--train-stride','30','--output-dir',str(out/'pocket3'),'--save-weights'])
tfpy=Path(os.environ['DUAL2POSE_TF_PYTHON']);tfenv=os.environ.copy();site=tfpy.parent.parent/'lib/python3.11/site-packages';libs=sorted(str(x) for x in (site/'nvidia').glob('*/lib'));tfenv['LD_LIBRARY_PATH']=':'.join(libs+[tfenv.get('LD_LIBRARY_PATH','')]);tfenv['XLA_FLAGS']='--xla_gpu_cuda_data_dir='+str(site/'nvidia/cuda_nvcc')
step('metapose',[str(tfpy),str(repo/'pegasus/runtime/metapose_smoke.py'),str(out/'metapose')],tfenv)
state['status']='passed' if all(s['status']=='passed' for s in state['steps']) else 'failed';state['finished_unix']=time.time();save();raise SystemExit(0 if state['status']=='passed' else 1)
