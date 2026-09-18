"""C07: paired SAM3D 2D/3D predictions from identical clean/occluded RGB.

2D observations are SAM3D's projected keypoints in original-image pixels,
not GT joints or an independent 2D detector. GT is used only to place masks.
All outputs are separate from the original E5 archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np

from dual2pose.eval.image_occlusion import (
    ImageOcclusionSetting, OcclusionFrameKey, apply_image_occlusion,
    build_required_frames_manifest, mask_protocol_payload,
)
from dual2pose.eval.run_unity_image_occlusion_frontend import (
    SAM3DPredictor, DEFAULT_CONFIG, DEFAULT_CHECKPOINT, DEFAULT_MHR,
    DEFAULT_DATA_ROOT, DEFAULT_REWRITE_FROM, REPO_ROOT, stream_output_path,
    _default_image_loader, _default_joint_loader, _filtered_prediction,
    _sha256, _atomic_json,
)
from dual2pose.map_config import filter_sam3d_body_kpts, filter_unity_kpts

DEFAULT_INDEX = DEFAULT_DATA_ROOT / 'index_mapping/use_layer_camera_filter_disabled/camera_pairs_by_action_folds/fold_00.json'
CONDITIONS = {'clean': None, 'random_0p5': ImageOcclusionSetting('random', .5),
              'random_1': ImageOcclusionSetting('random', 1.)}


def signature(setting):
    payload = {'schema': 1, 'mask': None if setting is None else mask_protocol_payload(setting),
               'keypoints_2d': 'SAM3D projected original image pixels; same selected person as 3D'}
    return json.dumps(payload, sort_keys=True)


def validate_matched_stream(path, expected_frames, setting, run_signature=''):
    expected = np.asarray(sorted(set(map(int, expected_frames))), dtype=np.int64)
    try:
        with np.load(path, allow_pickle=False) as d:
            pose, uv, valid, failed = (d[k] for k in ('pose', 'keypoints_2d', 'valid_2d', 'detection_failed'))
            n = len(expected)
            return bool(
                d['protocol'].item() == signature(setting) and d['run_signature'].item() == run_signature
                and np.array_equal(d['frame_indices'], expected)
                and pose.shape == (n, 15, 3) and np.isfinite(pose).all()
                and uv.shape == (n, 15, 2) and valid.shape == (n, 15) and valid.dtype == np.bool_
                and failed.shape == (n,) and failed.dtype == np.bool_
                and np.array_equal(valid, np.isfinite(uv).all(axis=-1))
                and valid[~failed].all() and not valid[failed].any()
                and np.isnan(uv[failed]).all() and not pose[failed].any()
                and d['image_size_hw'].shape == (n, 2) and (d['image_size_hw'] > 0).all()
                and d['selected_joint_count'].shape == (n,)
                and d['masked_rgb_sha256'].shape == (n,))
    except (OSError, ValueError, KeyError, EOFError):
        return False


def infer_matched_stream(stream, *, predictor, setting, output_root,
                         image_loader=_default_image_loader, joint_loader=_default_joint_loader,
                         run_signature=''):
    frames = sorted(set(map(int, stream['frame_indices'])))
    if not frames:
        raise ValueError('Empty stream')
    path = stream_output_path(Path(output_root), stream)
    if path.exists():
        if validate_matched_stream(path, frames, setting, run_signature):
            return path
        raise FileExistsError(f'Preserving incompatible or invalid cache: {path}')
    values = {k: [] for k in ('pose', 'keypoints_2d', 'valid_2d', 'detection_failed',
                              'image_size_hw', 'selected_joint_count', 'masked_rgb_sha256')}
    for frame in frames:
        image = image_loader(Path(stream['rgb_dir']) / f'frame_{frame:06d}.png')
        masked, selected = image, 0
        if setting is not None:
            raw = joint_loader(Path(stream['kpt2d_dir']) / f'kpt2d_{frame:06d}.npy')
            joints = filter_unity_kpts(raw, flag='2d', gender=stream['person_id'])
            masked, record = apply_image_occlusion(image, joints, setting,
                OcclusionFrameKey(stream['person_id'], stream['action_id'], stream['camera_id'], frame),
                source_frame_ids=frames)
            selected = int(record['selected_joint_count'])
        result = predictor(masked)
        failed = result is None
        pose = np.zeros((15, 3), np.float32) if failed else _filtered_prediction(result['pred_keypoints_3d'])
        uv = np.full((15, 2), np.nan, np.float32) if failed else np.asarray(
            filter_sam3d_body_kpts(np.asarray(result['pred_keypoints_2d'], dtype=np.float32)), dtype=np.float32)
        if uv.shape != (15, 2) or (not failed and not np.isfinite(uv).all()):
            raise ValueError('SAM3D must return finite original-image 15x2 observations for a detected person')
        row = dict(pose=pose, keypoints_2d=uv, valid_2d=np.isfinite(uv).all(-1),
                   detection_failed=failed, image_size_hw=image.shape[:2], selected_joint_count=selected,
                   masked_rgb_sha256=hashlib.sha256(masked.tobytes()).hexdigest())
        for key in values:
            values[key].append(row[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, **{k: np.asarray(v) for k, v in values.items()},
                        frame_indices=np.asarray(frames, np.int64), protocol=signature(setting),
                        run_signature=run_signature)
    if not validate_matched_stream(temporary, frames, setting, run_signature):
        raise RuntimeError(f'Invalid new paired archive: {temporary}')
    # A hard link publishes atomically without clobbering a concurrently created file.
    os.link(temporary, path)
    temporary.unlink()
    return path


class MatchedSAM3DPredictor(SAM3DPredictor):
    def __call__(self, image):
        from SAM3Dbody.infer import select_best_person
        outputs = self.estimator.process_one_image(img=image, bboxes=None)
        best, _ = select_best_person(outputs, verbose=False)
        return best


def run(args):
    if args.max_pairs is not None and (args.split != 'val' or args.max_pairs < 1):
        raise ValueError('Partial runs are permitted only on validation data')
    rows = json.loads(args.index.read_text())[args.split]
    if args.max_pairs is not None:
        rows = rows[:args.max_pairs]
    required = build_required_frames_manifest({'test': rows}, target_length=30,
        data_root=args.data_root, rewrite_from=DEFAULT_REWRITE_FROM)
    # Pin the same locally cached auxiliary assets used by the original E5 run.
    # No download/latest-revision resolution is permitted for this experiment.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    detector_dir = Path(cfg.model.get('detector_path', '') or os.environ.get('SAM3D_DETECTOR_PATH', '') or
        '/home/kaixu_chen/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692')
    fov_file = Path(cfg.model.get('fov_path', '') or os.environ.get('SAM3D_FOV_PATH', '') or
        '/home/kaixu_chen/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/cb0e8bbd6b1e243589717c78e750b1ba4c093acf/model.pt')
    auxiliary = [detector_dir/'model_final_f05665.pkl', fov_file, args.checkpoint.parent/'model_config.yaml']
    if cfg.model.get('detector_name', 'vitdet') != 'vitdet' or cfg.model.get('fov_name', 'moge2') != 'moge2' or cfg.model.get('segmentor_name', ''):
        raise ValueError('C07 requires the archived ViTDet + MoGe2 frontend configuration')
    os.environ['SAM3D_DETECTOR_PATH'] = str(detector_dir)
    os.environ['SAM3D_FOV_PATH'] = str(fov_file)
    contract = {'split': args.split, 'max_pairs': args.max_pairs, 'mask_seed': 42,
        'inference_seed_policy': 'unchanged E5 inference; mask seed does not imply retraining or seeded inference',
        'auxiliary_files_sha256': {str(p.resolve()): _sha256(p) for p in auxiliary},
        'index': str(args.index.resolve()), 'index_sha256': _sha256(args.index),
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': _sha256(args.checkpoint),
        'mhr_sha256': _sha256(args.mhr), 'config_sha256': _sha256(args.config),
        'source_sha256': {str(p.relative_to(REPO_ROOT)): _sha256(p) for p in (
            Path(__file__), REPO_ROOT/'dual2pose/eval/image_occlusion.py', REPO_ROOT/'dual2pose/map_config.py',
            REPO_ROOT/'SAM3Dbody/sam_3d_body/sam_3d_body_estimator.py',
            REPO_ROOT/'SAM3Dbody/sam_3d_body/models/meta_arch/sam3d_body.py',
            REPO_ROOT/'SAM3Dbody/infer.py', REPO_ROOT/'SAM3Dbody/tools/build_detector.py',
            REPO_ROOT/'SAM3Dbody/tools/build_fov_estimator.py',
            REPO_ROOT/'SAM3Dbody/tools/cascade_mask_rcnn_vitdet_h_75ep.py',
            REPO_ROOT/'dual2pose/eval/run_unity_image_occlusion_frontend.py')},
        'required_sha256': hashlib.sha256(json.dumps(required, sort_keys=True).encode()).hexdigest(),
        'conditions': {name: signature(setting) for name, setting in CONDITIONS.items()}}
    run_signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract['run_signature'] = run_signature
    args.output.mkdir(parents=True, exist_ok=True)
    contract_path = args.output/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise FileExistsError('Existing output contract differs; choose a new directory')
    _atomic_json(contract_path, contract)
    _atomic_json(args.output/'required_frames_manifest.json', required)
    predictor = MatchedSAM3DPredictor(config=args.config, checkpoint=args.checkpoint, mhr=args.mhr, gpu=args.gpu)
    started = time.monotonic()
    for name, setting in CONDITIONS.items():
        entries, failures = [], 0
        for index, stream in enumerate(required['streams']):
            path = infer_matched_stream(stream, predictor=predictor, setting=setting,
                output_root=args.output/name, run_signature=run_signature)
            with np.load(path) as data:
                failed = int(data['detection_failed'].sum())
            failures += failed
            entries.append({'path': str(path.resolve()), 'sha256': _sha256(path),
                            'frames': len(stream['frame_indices']), 'detection_failed': failed})
            progress = {'status': 'running', 'condition': name, 'streams_done': index+1,
                'streams_total': len(required['streams']), 'detection_failed': failures,
                'elapsed_seconds': time.monotonic()-started, 'pid': os.getpid()}
            _atomic_json(args.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        _atomic_json(args.output/name/'frontend_manifest.json', {
            'status': 'complete', 'run_signature': run_signature, 'condition': name,
            'stream_count': len(entries), 'unique_image_count': sum(e['frames'] for e in entries),
            'detection_failed': failures, 'entries': entries})
    _atomic_json(args.output/'progress.json', {'status': 'complete', 'conditions': list(CONDITIONS),
        'elapsed_seconds': time.monotonic()-started, 'pid': os.getpid()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--max-pairs', type=int)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--mhr', type=Path, default=DEFAULT_MHR)
    parser.add_argument('--gpu', type=int, default=0)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
