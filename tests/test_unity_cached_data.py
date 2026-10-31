import hashlib
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dual2pose.dataloader.unity_cached_data import RawUnityCacheDataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cache(tmp_path):
    index = tmp_path / 'index.json'
    items = [dict(person_id='female', action_id=f'action{i}', cam1_id='a', cam2_id='b') for i in range(5)]
    index.write_text(json.dumps({'train': items}))
    root = tmp_path / 'cache'
    root.mkdir()
    for i, name in enumerate(('left', 'right', 'target')):
        np.save(root / f'{name}.npy', np.arange(5*30*15*3, dtype=np.float32).reshape(5,30,15,3)+i)
    np.save(root / 'frames.npy', np.tile(np.arange(30, dtype=np.int64), (5,1)))
    (root / 'samples.jsonl').write_text(''.join(json.dumps(dict(index=i, **item, num_aligned_frames=30))+'\n' for i,item in enumerate(items)))
    manifest = dict(dataset='unity', split='train', sample_count=5, time_window=30,
                    num_joints=15, representation='raw_pre_batch_canonicalization',
                    index_sha256=digest(index), files={p.name:digest(p) for p in root.iterdir()})
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root, index, items


def build(cache):
    root, index, items = cache
    return RawUnityCacheDataset(root, split='train', index_path=index, index_mapping=items, target_t=30)


def change_manifest(cache, **changes):
    path=cache[0]/'manifest.json'
    data=json.loads(path.read_text()); data.update(changes);path.write_text(json.dumps(data))


def test_native_values_and_no_cache_mutation(cache):
    ds=build(cache)
    assert len(ds)==5
    assert torch.equal(ds[2]['kpt3d_sam']['cam1'], torch.from_numpy(np.load(cache[0]/'left.npy')[2]))
    assert ds[2]['meta']['action_id']=='action2'
    ds[2]['kpt3d_sam']['cam1'].zero_()
    assert ds[2]['kpt3d_sam']['cam1'].sum()>0
    assert ds[0]['kpt3d_gt_canonical']=={}


@pytest.mark.parametrize('change', [dict(representation='per_frame'), dict(index_sha256='wrong'),
    dict(split='val'), dict(time_window=29), dict(sample_count=4)])
def test_rejects_incompatible_provenance(cache, change):
    change_manifest(cache, **change)
    with pytest.raises(ValueError):build(cache)


def test_rejects_modified_array(cache):
    p=cache[0]/'left.npy'; a=np.load(p);a[0,0,0,0]+=1;np.save(p,a)
    with pytest.raises(ValueError, match='checksum'):build(cache)


def test_rejects_reordered_rows_even_with_valid_checksum(cache):
    p=cache[0]/'samples.jsonl'; rows=p.read_text().splitlines();rows[0],rows[1]=rows[1],rows[0];p.write_text('\n'.join(rows)+'\n')
    manifest=json.loads((cache[0]/'manifest.json').read_text());manifest['files'][p.name]=digest(p)
    change_manifest(cache, files=manifest['files'])
    with pytest.raises(ValueError, match='order|metadata'):build(cache)


def test_preserves_rng_and_shuffle_drop_last(cache):
    torch.manual_seed(123)
    before=torch.random.get_rng_state().clone()
    ds=build(cache)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.manual_seed(123)
    reference=[b.tolist() for b in DataLoader(list(range(5)), batch_size=2, shuffle=True,drop_last=True)]
    torch.manual_seed(123)
    actual=[[int(v[6:]) for v in b['meta']['action_id']] for b in DataLoader(ds,batch_size=2,shuffle=True,drop_last=True)]
    assert actual==reference


@pytest.mark.parametrize("kind", ["nan", "dtype", "shape"])
def test_rejects_invalid_array_even_with_valid_checksum(cache, kind):
    p=cache[0]/'target.npy'; a=np.load(p)
    if kind=="nan":a[0,0,0,0]=np.nan
    elif kind=="dtype":a=a.astype(np.float64)
    else:a=a[:4]
    np.save(p,a)
    manifest=json.loads((cache[0]/'manifest.json').read_text());manifest['files'][p.name]=digest(p)
    change_manifest(cache, files=manifest['files'])
    with pytest.raises(ValueError, match='nonfinite|shape/dtype'):build(cache)


def test_training_entrypoint_default_loader_unchanged(monkeypatch):
    from omegaconf import OmegaConf
    from dual2pose import train_unity
    marker=object()
    monkeypatch.setattr(train_unity, 'UnityDataModule', lambda cfg: marker)
    assert train_unity.build_data_module(OmegaConf.create({'data':{'unity':{}}})) is marker


def test_training_entrypoint_explicit_cache_selection(monkeypatch):
    from omegaconf import OmegaConf
    from dual2pose import train_unity
    from dual2pose.dataloader import unity_cached_data
    marker=object()
    monkeypatch.setattr(unity_cached_data, 'VerifiedCacheUnityDataModule', lambda cfg, **kwargs: marker)
    assert train_unity.build_data_module(OmegaConf.create({'data':{'unity':{'raw_cache_root':'/cache/unity'}}})) is marker
