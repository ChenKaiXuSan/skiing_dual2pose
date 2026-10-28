"""Portable camera-pair split tests with a synthetic on-disk dataset."""
from itertools import combinations

import pytest

from cross_validation.cross_validation_camera_pairs import CameraPairCrossValidation
from dual2pose.map_config import UnityDataConfig


@pytest.fixture
def dataset_root(tmp_path):
    for person in ("male", "female"):
        for action in range(6):
            for angle in (0, 90, 180, 270):
                (tmp_path / "data_pole_ski" / person / f"action_{action}" /
                 "frames" / f"capture_L0_A{angle:03d}").mkdir(parents=True)
    return tmp_path


def make_cv(root, strategy="by_action"):
    return CameraPairCrossValidation(
        data_root=root, num_persons=2, num_actions=6, num_cameras=4,
        split_strategy=strategy, n_splits=2,
        index_save_path=root / f"index_{strategy}.json",
    )


def test_basic_functionality(dataset_root):
    folds = make_cv(dataset_root)(force_recreate=True)
    assert len(folds) == 2
    for fold in folds.values():
        groups = [{s.action_id for s in fold[split]} for split in ("train", "val", "test")]
        assert all(groups)
        assert all(not (a & b) for a, b in combinations(groups, 2))
        assert sum(len(fold[s]) for s in ("train", "val", "test")) == 72


def test_different_strategies(dataset_root):
    for strategy in ("by_action", "by_camera_pair"):
        folds = make_cv(dataset_root, strategy)(force_recreate=True)
        assert len(folds) == 2
        for fold in folds.values():
            key = (lambda s: s.action_id) if strategy == "by_action" else (lambda s: (s.cam1_id, s.cam2_id))
            groups = [{key(s) for s in fold[split]} for split in ("train", "val", "test")]
            assert all(groups)
            assert all(not (a & b) for a, b in combinations(groups, 2))
            assert sum(len(fold[s]) for s in ("train", "val", "test")) == 72


def test_sample_access(dataset_root):
    sample = make_cv(dataset_root)(force_recreate=True)[0]["train"][0]
    recovered = UnityDataConfig.from_dict(sample.to_dict())
    assert recovered.to_dict() == sample.to_dict()
    assert sample.cam1_id != sample.cam2_id


def test_load_existing(dataset_root):
    first = make_cv(dataset_root)(force_recreate=True)
    second = make_cv(dataset_root)(force_recreate=False)
    assert first.keys() == second.keys()
    for index, fold in first.items():
        for split in ("train", "val", "test"):
            assert [s.to_dict() for s in fold[split]] == [s.to_dict() for s in second[index][split]]
