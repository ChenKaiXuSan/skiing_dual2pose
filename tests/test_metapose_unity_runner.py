import copy
import pytest
from dual2pose.experiments.run_metapose_unity_stable import validate_records, validate_request
from dual2pose.eval.metapose_unity_depth import POLICY


def fixture(pilot=False):
    cfg=dict(pilot=pilot,epochs=3 if pilot else 10,stages=3,batch_size=1024,seed=42,
             learning_rate=1e-5,gradient_clip_norm=1.,identity_stage_initialization=True,
             deterministic_ops=True,tf32_disabled=True,policy=POLICY,max_frames=32768 if pilot else None)
    frames=32768 if pilot else 5799600
    hist=[dict(stage=s,epoch=e,full_train_2d_mse=.01/e,val_2d_mse=None if pilot else .01/e,
               quality=dict(passed=True,frames=frames,policy=POLICY,reasons=[]),max_clipped_gradient_norm=.8)
          for s in range(1,4) for e in range(1,cfg['epochs']+1)]
    selected=[dict(stage=s,epoch=cfg['epochs'],checkpoint_sha256='a'*64) for s in range(1,4)]
    rep=dict(status='complete_train_quality_passed',pilot=pilot,dataset='unity',train_frames=frames,
             val_frames=0 if pilot else 3866400,test_data_opened=False,val_data_opened=not pilot,
             initial_train_2d_mse=.1,selected_stages=selected)
    return cfg,rep,hist


def test_complete_full_fit_allowed_and_pilot_not_exportable():
    validate_records(*fixture())
    with pytest.raises(ValueError):validate_records(*fixture(True))
    validate_records(*fixture(True),allow_pilot=True)


def test_lowest_complete_validation_loss_selected():
    c,r,h=fixture();r['selected_stages'][0]['epoch']=1
    with pytest.raises(ValueError):validate_records(c,r,h)


def test_failed_training_health_blocks_even_if_final_epoch_good():
    c,r,h=fixture();h[2]['quality']['passed']=False
    with pytest.raises(ValueError):validate_records(c,r,h)


def test_missing_epoch_or_wrong_coverage_blocks():
    c,r,h=fixture()
    with pytest.raises(ValueError):validate_records(c,r,h[:-1])
    h[0]['quality']['frames']-=1
    with pytest.raises(ValueError):validate_records(c,r,h)


def test_pilot_cannot_read_validation_or_test():
    c,r,h=fixture(True);r['val_data_opened']=True
    with pytest.raises(ValueError):validate_records(c,r,h,allow_pilot=True)
    with pytest.raises(ValueError):validate_request(pilot=True,epochs=3,max_frames=32768,val_dir='val',pilot_run=None)
    with pytest.raises(ValueError):validate_request(pilot=False,epochs=10,max_frames=None,val_dir='val',pilot_run=None)


def test_changed_recipe_or_policy_blocked():
    c,r,h=fixture();c['policy']=dict(POLICY,min_depth_cosine=-1)
    with pytest.raises(ValueError):validate_records(c,r,h)
    c,r,h=fixture();c['learning_rate']=1e-4
    with pytest.raises(ValueError):validate_records(c,r,h)


def test_no_train_loss_improvement_pilot_rejected():
    c,r,h=fixture(True);r['initial_train_2d_mse']=1e-7
    with pytest.raises(ValueError):validate_records(c,r,h,allow_pilot=True)


def test_validation_manifest_is_sealed(tmp_path):
    from dual2pose.experiments.run_metapose_unity_stable import verify_input_manifests
    from dual2pose.eval.metapose_inputs import sha256
    tr=tmp_path/'train';va=tmp_path/'val';tr.mkdir();va.mkdir()
    for d in [tr,va]:(d/'manifest.json').write_text('{}')
    cfg=dict(train_dir=str(tr),val_dir=str(va),pilot=False,training_manifest_sha256=sha256(tr/'manifest.json'),validation_manifest_sha256=sha256(va/'manifest.json'))
    verify_input_manifests(cfg)
    (va/'manifest.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):verify_input_manifests(cfg)
