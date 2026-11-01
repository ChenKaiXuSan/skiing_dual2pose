"""Storage acceleration must preserve raw frontend values and sample ordering."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tests.test_unity_cached_data import cache, build, digest
from dual2pose.eval.frontend_manifest import FrontEndManifest, FrontEndPoseDataset
from dual2pose.dataloader.frontend_pose_data import MixedFrontEndDataset


@pytest.fixture
def sources(cache, tmp_path):
    manifests = []
    for name, suffix in [('indexed', '.npz'), ('interpolated', '.npy')]:
        entries = []
        # Deliberately share paths across rows; the builder must decode each once.
        for cam, offset in [('a', 100), ('b', 200)]:
            path = tmp_path / f'{name}_{cam}{suffix}'
            length = 30 if suffix == '.npz' else 7
            pose = np.arange(length * 17 * 3, dtype=np.float32).reshape(length, 17, 3) + offset
            if suffix == '.npz':
                np.savez_compressed(path, pose=pose, frame_indices=np.arange(29, -1, -1))
            else:
                np.save(path, pose)
            entries.extend(dict(person_id=item['person_id'], action_id=item['action_id'],
                                camera_id=cam, pose_path=path.name) for item in cache[2])
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(dict(frontend_name=name, entries=entries,
                                       joint_indices=list(range(16, 1, -1)))))
        manifests.append(FrontEndManifest.load(path))
    return manifests


def api():
    from dual2pose.dataloader import frontend_cached_data
    return frontend_cached_data


def assert_same(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        if isinstance(expected[key], dict):
            assert_same(actual[key], expected[key])
        elif isinstance(expected[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key]


def create(cache, source, root):
    base = build(cache)
    api().build_frontend_cache(root, base, source, split='train')
    return api().VerifiedFrontEndPoseDataset(base, source, root, split='train')


@pytest.mark.parametrize('source_index', [0, 1])
def test_exact_original_values_frame_alignment_and_joint_selection(cache, sources, tmp_path, source_index):
    source = sources[source_index]
    ds = create(cache, source, tmp_path / 'packed')
    reference = FrontEndPoseDataset(build(cache), source)
    for i in range(len(ds)):
        assert_same(ds[i], reference[i])
    if source_index == 0:
        # Target frame 0 selects the last source row, then source joint 16.
        assert ds[0]['kpt3d_sam']['cam1'][0, 0, 0].item() == 1627
    ds[0]['kpt3d_sam']['cam1'].zero_()
    assert ds[0]['kpt3d_sam']['cam1'].sum() > 0


def test_builder_decodes_each_unique_source_once_and_runtime_opens_none(cache, sources, tmp_path, monkeypatch):
    source = sources[0]
    original = np.load
    opened = []
    def tracked(path, *args, **kwargs):
        if Path(path).suffix == '.npz':
            opened.append(Path(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', tracked)
    ds = create(cache, source, tmp_path / 'packed')
    assert sorted(opened) == sorted(set(source.entries.values()))
    opened.clear()
    for i in range(len(ds)):
        ds[i]
    assert opened == []


def test_builder_reuses_identical_alignment_within_each_source(cache, sources, tmp_path, monkeypatch):
    original = FrontEndManifest.load_pose
    conversions = []
    def counted(self, *args, **kwargs):
        conversions.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(FrontEndManifest, 'load_pose', counted)
    ds = create(cache, sources[0], tmp_path / 'packed')
    # Five rows share the same two files and target frame grid.
    assert len(conversions) == 2
    assert ds[4]['kpt3d_sam']['cam1'][0, 0, 0].item() == 1627


def test_mixed_preserves_source_blocks_rng_shuffle_and_drop_last(cache, sources, tmp_path):
    base = build(cache)
    # Existing mixed wrapper requires the original mapping; raw cache only has samples.
    base._index_mapping = cache[2]
    for source in sources:
        api().build_frontend_cache(tmp_path / source.frontend_name / 'train', base, source, split='train')
    torch.manual_seed(342)
    before = torch.random.get_rng_state().clone()
    actual = api().VerifiedMixedFrontEndDataset(base, [None, *sources], tmp_path, split='train')
    assert torch.equal(before, torch.random.get_rng_state())
    reference = MixedFrontEndDataset(base, [None, *sources])
    assert len(actual) == 15
    for i in range(len(actual)):
        assert_same(actual[i], reference[i])
    assert_same(actual[-1], reference[-1])
    with pytest.raises(IndexError):
        actual[len(actual)]
    def batches(ds):
        torch.manual_seed(17)
        return list(DataLoader(ds, batch_size=4, shuffle=True, drop_last=True))
    for a, b in zip(batches(actual), batches(reference)):
        assert_same(a, b)


@pytest.mark.parametrize('changed', ['array', 'raw_manifest', 'frontend_manifest', 'source_pose', 'implementation'])
def test_rejects_drift(cache, sources, tmp_path, changed, monkeypatch):
    root = tmp_path / 'packed'
    ds = create(cache, sources[0], root)
    if changed == 'array':
        with (root / 'left.npy').open('ab') as handle:
            handle.write(b'changed')
    elif changed == 'raw_manifest':
        ds.base_dataset.provenance['manifest_sha256'] = 'changed'
    elif changed == 'frontend_manifest':
        sources[0].source_path.write_text(sources[0].source_path.read_text() + ' ')
    elif changed == 'source_pose':
        with next(iter(sources[0].entries.values())).open('ab') as handle:
            handle.write(b'changed')
    else:
        monkeypatch.setattr(api(), 'implementation_digest', lambda: 'changed')
    with pytest.raises(ValueError, match='mismatch'):
        api().VerifiedFrontEndPoseDataset(ds.base_dataset, sources[0], root, split='train')


def test_builder_refuses_overwrite_and_does_not_publish_failed_cache(cache, sources, tmp_path):
    root = tmp_path / 'packed'
    ds = create(cache, sources[0], root)
    with pytest.raises(FileExistsError):
        api().build_frontend_cache(root, ds.base_dataset, sources[0], split='train')
    missing = tmp_path / 'missing.npz'
    np.savez(missing, pose=np.zeros((1, 17, 3), dtype=np.float32), frame_indices=np.array([999]))
    bad = FrontEndManifest('bad', {key: missing for key in sources[0].entries}, sources[0].source_path,
                           sources[0].joint_indices)
    failed = tmp_path / 'failed'
    with pytest.raises(KeyError, match='missing target frames'):
        api().build_frontend_cache(failed, ds.base_dataset, bad, split='train')
    assert not (failed / 'manifest.json').exists()


def test_repeated_target_frames_are_preserved(cache, sources, tmp_path):
    frame_path = cache[0] / 'frames.npy'
    frames = np.load(frame_path)
    frames[:, 2] = 1
    np.save(frame_path, frames)
    manifest_path = cache[0] / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['frames.npy'] = digest(frame_path)
    manifest_path.write_text(json.dumps(manifest))
    ds = create(cache, sources[0], tmp_path / 'packed')
    assert_same(ds[0], FrontEndPoseDataset(build(cache), sources[0])[0])
    assert torch.equal(ds[0]['kpt3d_sam']['cam1'][1], ds[0]['kpt3d_sam']['cam1'][2])


def test_datamodule_keeps_loader_policy_and_avoids_test_data(cache, sources, tmp_path):
    from omegaconf import OmegaConf
    from dual2pose.dataloader.unity_cached_data import VerifiedCacheUnityDataModule
    raw_root = tmp_path / 'raw'
    packed_root = tmp_path / 'packed'
    for split in ('train', 'val'):
        target = raw_root / split
        shutil.copytree(cache[0], target)
        path = target / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['split'] = split
        path.write_text(json.dumps(manifest))
        from dual2pose.dataloader.unity_cached_data import RawUnityCacheDataset
        raw = RawUnityCacheDataset(target, split=split, index_path=cache[1], index_mapping=cache[2])
        api().build_frontend_cache(packed_root / sources[0].frontend_name / split,
                                   raw, sources[0], split=split)
    config = OmegaConf.create(dict(data=dict(batch_size=2, num_workers=0, img_size=256,
        load_frames=False, load_2d_kpt=False, load_3d_kpt=True, time_window=30,
        unity=dict(index_mapping_path=str(cache[1]), root_path=str(tmp_path)))))
    base_dm = VerifiedCacheUnityDataModule(config, raw_root)
    # No test files exist: constructing the adaptation data must not require them.
    base_dm._dataset_idx = dict(train=cache[2], val=cache[2])
    dm = api().VerifiedFrontendDataModule(base_dm, packed_root,
        train_sources=[None, sources[0]], val_sources=[None, sources[0]])
    dm.setup('fit')
    first_dataset = base_dm.train_gait_dataset
    dm.setup('fit')
    assert base_dm.train_gait_dataset is first_dataset
    assert not hasattr(base_dm, 'test_gait_dataset')
    assert len(dm.train_dataloader()) == 5
    assert dm.train_dataloader().drop_last
    assert not dm.val_dataloader().drop_last
    assert set(dm.cache_provenance) == {'train', 'val'}


def test_cli_build_validates_fold_and_exact_rows(cache, sources, tmp_path, capsys):
    index_payload = json.loads(cache[1].read_text())
    index_payload['_metadata'] = dict(fold_idx=0)
    cache[1].write_text(json.dumps(index_payload))
    raw_manifest = cache[0] / 'manifest.json'
    payload = json.loads(raw_manifest.read_text())
    payload['index_sha256'] = digest(cache[1])
    raw_manifest.write_text(json.dumps(payload))
    raw_root = tmp_path / 'raw'
    shutil.copytree(cache[0], raw_root / 'train')
    args = ['--raw-cache-root', str(raw_root), '--index-path', str(cache[1]),
            '--frontend-manifest', str(sources[0].source_path), '--cache-root', str(tmp_path / 'packed'),
            '--split', 'train', '--fold', '0', '--verify-samples', '5']
    api().main(args)
    report = json.loads(capsys.readouterr().out)
    assert report['verified_samples'] == 5
    assert report['sample_count'] == 5
    assert report['max_abs_difference'] == 0
    with pytest.raises(ValueError, match='metadata fold'):
        api().main([*args, '--fold', '1'])
