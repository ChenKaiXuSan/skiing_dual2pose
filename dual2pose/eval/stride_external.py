"""STRIDE adaptation with native MHR joints and independent-window TTT.

This is not the official mesh-regressor demo. The protocol differences are
predeclared in docs/superpowers/plans/2026-09-10-ivc-external-baselines.md.
No ground truth or calibrated camera parameters are accepted here.
"""
from __future__ import annotations

from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from dual2pose.eval.frontend_lifters import (
    H36M17_TO_CANONFUSE_COMMON13_INDICES, _load_python_file, _temporary_import_root,
)
from dual2pose.eval.main_baseline_data import sha256
from dual2pose.trainer.canonicalize import canonicalize_pose_torch

COMMON = list(H36M17_TO_CANONFUSE_COMMON13_INDICES)
EXTRA_SLOTS = [0, 7, 8, 10]
MHR_BONES = [1, 35, 37, 126]


def assemble_h36m17(common, bones):
    common, bones = np.asarray(common), np.asarray(bones)
    if (common.shape[-2:] != (13, 3) or bones.shape[-2:] != (127, 3)
            or common.shape[:-2] != bones.shape[:-2]
            or not np.isfinite(common).all() or not np.isfinite(bones).all()):
        raise ValueError('Expected matching finite common13 and actual MHR127 arrays')
    result = np.empty((*common.shape[:-2], 17, 3), np.float32)
    result[..., COMMON, :] = common
    result[..., EXTRA_SLOTS, :] = bones[..., MHR_BONES, :]
    return result


def canonical_average17(left, right, bones_left, bones_right, dataset):
    if dataset not in ('unity', 'ski'):
        raise ValueError(dataset)
    kwargs = {} if dataset == 'unity' else dict(left_hip=4, right_hip=5, neck=12)
    joint_slice = slice(2, 15) if dataset == 'unity' else slice(None)
    expected_joints = 15 if dataset == 'unity' else 13
    outputs = []
    for native, bones in ((left,bones_left), (right,bones_right)):
        native = np.asarray(native, np.float32)
        if native.ndim != 4 or native.shape[-2:] != (expected_joints,3):
            raise ValueError('Expected native protocol batch, not preselected points')
        _, transform = canonicalize_pose_torch(torch.from_numpy(native.copy()), **kwargs)
        full = torch.from_numpy(assemble_h36m17(native[..., joint_slice, :], bones))
        output = (full-transform['pelvis']) @ transform['R']
        outputs.append(output.numpy())
    result = (outputs[0]+outputs[1])*.5
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite canonical input')
    return result


class IndependentStride:
    def __init__(self, model, loss_module, flip_fn, config, seed=42):
        self.model, self.loss, self.flip_fn = model, loss_module, flip_fn
        self.config, self.seed = config, seed
        self.prior = {k:v.detach().clone() for k,v in model.state_dict().items()}
        if any(not torch.isfinite(v).all() for v in self.prior.values()):
            raise ValueError('Nonfinite prior parameters')
        if config.epochs < 1 or config.learning_rate <= 0:
            raise ValueError('Positive fixed adaptation budget required')

    def refine(self, sequence):
        cfg, loss = self.config, self.loss
        device = next(self.model.parameters()).device
        x = torch.as_tensor(sequence, dtype=torch.float32, device=device).detach().clone()
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[1:] != (30,17,3):
            raise ValueError('Exactly one independent (1,30,17,3) window is required')
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite input')
        def synchronize():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        synchronize(); start = time.perf_counter()
        self.model.load_state_dict(self.prior, strict=True)
        torch.manual_seed(self.seed)
        rng = random.Random(self.seed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                                      weight_decay=cfg.weight_decay)
        synchronize(); initialized = time.perf_counter()
        self.model.train()
        losses = []
        for _ in range(cfg.epochs):
            inp = self.flip_fn(x) if cfg.flip and rng.random() > .5 else x.clone()
            pseudo = inp - inp[:,:,0:1,:]
            pred = self.model(inp)
            total = (cfg.lambda_3d_pos*loss.loss_mpjpe(pred,pseudo)
                     +cfg.lambda_scale*loss.n_mpjpe(pred,pseudo)
                     +cfg.lambda_3d_velocity*loss.loss_velocity(pred,pseudo)
                     +cfg.lambda_lv*loss.loss_limb_var(pred))
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite adaptation loss; no fallback prediction')
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'),
                                           error_if_nonfinite=True)
            optimizer.step()
            for group in optimizer.param_groups:
                group['lr'] *= cfg.lr_decay
            losses.append(float(total.detach().cpu()))
        synchronize(); adapted = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)
            if cfg.flip:
                prediction = (prediction+self.flip_fn(self.model(self.flip_fn(x))))*.5
            prediction[:,:,0,:] = 0
            # Restore the observed root, never a test-GT translation or scale.
            prediction = prediction+x[:,:,0:1,:]
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite final prediction; no fallback prediction')
        synchronize(); finished = time.perf_counter()
        return prediction.detach().cpu(), dict(
            steps=cfg.epochs, losses=losses, reset_seconds=initialized-start,
            adaptation_seconds=adapted-initialized, inference_seconds=finished-adapted,
            total_seconds=finished-start, seed=self.seed)


def load_official_stride(repo: Path, checkpoint: Path, device='cuda:1', seed=42):
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    config_path = repo/'stride/configs/pose3d/MB_ft_h36m.yaml'
    cfg = SimpleNamespace(**yaml.safe_load(config_path.read_text()))
    if (cfg.num_joints != 17 or not cfg.rootrel or cfg.synthetic or cfg.gt_2d
            or cfg.no_conf or cfg.noise or cfg.mask_ratio or cfg.mask_T_ratio
            or any(getattr(cfg,key) != 0 for key in ('lambda_lg','lambda_a','lambda_av'))):
        raise ValueError('Official configuration differs from supported frozen contract')
    with _temporary_import_root(repo/'stride', 'lib'):
        learning = _load_python_file('_ivc_stride_learning', repo/'stride/lib/utils/learning.py')
        losses = _load_python_file('_ivc_stride_losses', repo/'stride/lib/model/loss.py')
        utils = _load_python_file('_ivc_stride_utils', repo/'stride/lib/utils/utils_data.py')
        model = learning.load_backbone(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_pos']
    state = {key.removeprefix('module.'):value for key,value in state.items()}
    model.load_state_dict(state, strict=True)
    adapter = IndependentStride(model.to(device), losses, utils.flip_data, cfg, seed)
    return adapter, dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                         config_sha256=sha256(config_path), config=vars(cfg),
                         parameter_count=sum(p.numel() for p in model.parameters()),
                         strict_model_load=True, source_repo=str(repo))
