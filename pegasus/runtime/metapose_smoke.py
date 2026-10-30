"""GPU training/checkpoint/inference diagnostic on training frames only."""
import json,os,sys
from pathlib import Path
import numpy as np
from dual2pose.experiments.run_metapose import OfficialMetaPose,load_split
from dual2pose.eval.metapose_inputs import fit_point_uncertainty,point_mixtures
repo=Path(os.environ['DUAL2POSE_REPO']);out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
source=repo/'.cache/external_baselines/google-research'
data=load_split(repo/'logs/ivc_mmsports_extension/external_baselines/20260911/metapose_data/ski/train','train')
m=OfficialMetaPose(source,42,learning_rate=1e-5,gradient_clip_norm=1.,identity_stages=True)
assert m.tf.config.list_physical_devices('GPU'), 'GPU unavailable'
xyz=np.asarray(data['xyz'][:4]);uv=np.asarray(data['uv'][:4]);labels=np.asarray(data['labels2d'][:4])
uncertainty=fit_point_uncertainty(data['uv'][:64]-data['labels2d'][:64]);task=point_mixtures(uv,uncertainty).reshape(4,-1)
x=m.initialize(xyz);m.add_stage();before=[v.numpy().copy() for v in m.stages[-1].trainable_variables]
losses=[m.train_batch(x,task,labels) for _ in range(3)]
assert np.isfinite(losses).all()
assert any(not np.array_equal(v.numpy(),b) for v,b in zip(m.stages[-1].trainable_variables,before))
weights=out/'stage_1.weights.h5';m.stages[-1].save_weights(str(weights));pred=m.predict(x,task)
fresh=OfficialMetaPose(source,42,learning_rate=1e-5,gradient_clip_norm=1.,identity_stages=True);fresh.add_stage();fresh.load_stage(weights)
np.testing.assert_allclose(fresh.predict(x,task),pred,rtol=1e-9,atol=1e-9)
(out/'result.json').write_text(json.dumps(dict(status='passed',scope='4 training frames; 3 updates; checkpoint reload and inference',tensorflow=m.tf.__version__,gpu_devices=[d.name for d in m.tf.config.list_physical_devices('GPU')],losses=losses,publication_eligible=False),indent=2)+'\n')
