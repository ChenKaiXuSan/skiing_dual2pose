from omegaconf import OmegaConf
import pytest
from dual2pose import train_frontend_adaptation as entry


def config(tmp_path,frontend='videopose3d',cached=False):
    path=tmp_path/'manifest.json';path.write_text('{}')
    return OmegaConf.create({'data':{'unity':{'raw_cache_root':'/raw' if cached else None}},'adaptation':{'frontend':frontend,'cache_root':'/packed' if cached else None,'manifests':{name:{'train':str(path),'val':str(path)} for name in ('videopose3d','poseformer','motionbert')}}})


def test_default_frontend_loader_preserves_original_selection(tmp_path,monkeypatch):
    base=object();seen={}
    monkeypatch.setattr(entry,'UnityDataModule',lambda cfg:base)
    def wrapper(dm,**kwargs):seen.update(kwargs);assert dm is base;return 'original'
    monkeypatch.setattr(entry,'FrontEndDataModule',wrapper)
    assert entry.build_data_module(config(tmp_path))=='original'
    assert seen['test_manifest'] is None
    assert seen['train_manifest']==tmp_path/'manifest.json'


def test_cache_options_must_be_supplied_together(tmp_path):
    cfg=config(tmp_path,cached=True);cfg.data.unity.raw_cache_root=None
    with pytest.raises(ValueError,match='together'):entry.build_data_module(cfg)


def test_verified_mixed_sources_keep_original_block_order(tmp_path,monkeypatch):
    from dual2pose.dataloader import frontend_cached_data,unity_cached_data
    base=object();seen={}
    monkeypatch.setattr(unity_cached_data,'VerifiedCacheUnityDataModule',lambda cfg:base)
    def wrapper(dm,cache_root,**kwargs):assert dm is base;assert str(cache_root)=='/packed';seen.update(kwargs);return 'cached'
    monkeypatch.setattr(frontend_cached_data,'VerifiedFrontendDataModule',wrapper)
    assert entry.build_data_module(config(tmp_path,'mixed',True))=='cached'
    assert seen['train_sources']==[None]+[tmp_path/'manifest.json']*3
    assert seen['val_sources']==[None]+[tmp_path/'manifest.json']*3


def test_training_main_reaches_fit_and_records_frontend(tmp_path,monkeypatch):
    from types import SimpleNamespace
    cfg=OmegaConf.create({'train':{'seed':42,'fold':0,'gpu':0},'data':{'unity':{'root_path':str(tmp_path),'index_mapping_path':''}},'model':{'optim':{'lr':.01}},'log_path':str(tmp_path/'output'),'adaptation':{'source_checkpoint':str(tmp_path/'source.ckpt'),'learning_rate':.0001,'scope':'full','frontend':'videopose3d','epochs':20}})
    monkeypatch.setattr(entry,'seed_everything',lambda *a,**k:None)
    monkeypatch.setattr(entry,'resolve_fold_index_path',lambda *a:tmp_path/'fold.json')
    monkeypatch.setattr(entry,'validate_fold_metadata',lambda *a:None)
    monkeypatch.setattr(entry,'load_fusion_checkpoint',lambda *a:SimpleNamespace(left_hip_idx=6,right_hip_idx=7,neck_idx=14,face_sign=True))
    monkeypatch.setattr(entry,'FrontendAdaptationTrainer',lambda cfg:object())
    monkeypatch.setattr(entry.CrossViewFusionTrainer,'check_checkpoint_protocol',lambda *a,**k:None)
    monkeypatch.setattr(entry.torch,'load',lambda *a,**k:{})
    monkeypatch.setattr(entry,'load_model_weights_only',lambda *a:{})
    monkeypatch.setattr(entry,'asdict',lambda x:x)
    monkeypatch.setattr(entry,'configure_trainable_scope',lambda *a:{})
    monkeypatch.setattr(entry,'build_data_module',lambda *a:object())
    checkpoint=SimpleNamespace(best_model_path='best.ckpt',best_model_score=None,last_model_path='last.ckpt')
    monkeypatch.setattr(entry,'ModelCheckpoint',lambda **k:checkpoint)
    for name in ('CSVLogger','TensorBoardLogger','LearningRateMonitor'):
        monkeypatch.setattr(entry,name,lambda **k:None)
    calls=[]
    trainer=SimpleNamespace(current_epoch=20,global_step=940,fit=lambda *a,**k:calls.append('fit'),save_checkpoint=lambda *a:calls.append('save'))
    monkeypatch.setattr(entry,'Trainer',lambda **k:trainer)
    entry.main.__wrapped__(cfg)
    assert calls==['fit','save']
    import json
    assert json.loads((tmp_path/'output/adaptation_init.json').read_text())['frontend']=='videopose3d'
