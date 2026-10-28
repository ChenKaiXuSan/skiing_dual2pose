#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from pytorch_lightning import LightningModule

from dual2pose.models.crossview_fusion import CrossViewCanonicalFusion
from dual2pose.eval.pose_metrics import COMMON13_NAMES, EVALUATION_PROTOCOL, common13_point_errors, common13_pose
from .canonicalize import canonicalize_pose_torch

logger = logging.getLogger(__name__)

# The training protocol is fixed.  Both input streams and the supervision
# target are expressed in the per-frame body frame (every frame centred at its
# own pelvis and rotated into its own hip/neck frame), and the model is trained
# with an L1 loss against the ground-truth pose in that frame.  Checkpoints
# record the mode so that weights from the manuscript's batch-referenced
# first-frame protocol are rejected instead of being fed a representation they
# were never trained on.
CANONICAL_MODE = "per_frame"


class CrossViewFusionTrainer(LightningModule):
    """CanonFuse3D trainer: per-frame body-frame canonicalization, supervised L1.

    Every batch must carry the two SAM 3D Body streams (``kpt3d_sam``) and, for
    training and validation, the ground-truth pose (``kpt3d_gt``).  Test-time
    inference on unlabeled recordings carries no target.
    """

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        data_cfg = getattr(hparams, "data", None)
        self.left_hip_idx = int(getattr(data_cfg, "left_hip_idx", 6))
        self.right_hip_idx = int(getattr(data_cfg, "right_hip_idx", 7))
        self.neck_idx = int(getattr(data_cfg, "neck_idx", 14))
        # The eye pair fixes the sign of the forward axis.  Skeletons without
        # eyes (the 13-joint Ski-PosePTZ subset) must set data.face_sign=false:
        # the frame then takes z = x cross y, which is consistent as long as the
        # inputs and the reference share the same handedness.
        self.left_eye_idx = int(getattr(data_cfg, "left_eye_idx", 0))
        self.right_eye_idx = int(getattr(data_cfg, "right_eye_idx", 1))
        self.face_sign = bool(getattr(data_cfg, "face_sign", True))

        # Hyperparameters come from the `model` config group
        # (configs/model/cross_view_fusion.yaml). The fallbacks keep partial
        # configs, such as the ones the evaluation scripts build, working.
        model_cfg = getattr(hparams, "model", None)
        arch_cfg = getattr(model_cfg, "arch", None)
        optim_cfg = getattr(model_cfg, "optim", None)
        loss_cfg = getattr(model_cfg, "loss", None)

        # The body-frame protocol is not a hyperparameter; refuse stale configs
        # that still try to select another mode.
        configured_mode = getattr(arch_cfg, "canonical_mode", None)
        if configured_mode is not None and str(configured_mode) != CANONICAL_MODE:
            raise ValueError(
                f"The training protocol is fixed to canonical_mode={CANONICAL_MODE!r}; "
                f"model.arch.canonical_mode={configured_mode!r} is not supported"
            )
        self.canonical_mode = CANONICAL_MODE

        self.lr = float(getattr(optim_cfg, "lr", 0.001))
        self.weight_decay = float(getattr(optim_cfg, "weight_decay", 0.01))
        self.lambda_view_recon = float(getattr(loss_cfg, "lambda_view_recon", 0.05))
        self.lambda_alpha_balance = float(getattr(loss_cfg, "lambda_alpha_balance", 0.02))
        self.lambda_alpha_entropy = float(getattr(loss_cfg, "lambda_alpha_entropy", 0.005))
        self.lambda_temporal_smooth = float(
            getattr(loss_cfg, "lambda_temporal_smooth", 0.01)
        )

        self.models = CrossViewCanonicalFusion(
            hidden_dim=int(getattr(arch_cfg, "hidden_dim", 128)),
            num_heads=int(getattr(arch_cfg, "num_heads", 4)),
            dropout=float(getattr(arch_cfg, "dropout", 0.1)),
            temporal_kernel_size=int(getattr(arch_cfg, "temporal_kernel_size", 5)),
        )

        self.save_root = str(getattr(hparams, "log_path", "./logs"))

        self.test_save_dir: Path = Path(self.save_root) / "summary"

    @staticmethod
    def _get_character_data(
        batch: Dict[str, Any],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Extract character SAM, GT data, and frames.

        Returns:
            Tuple of (p_left, p_right, p_gt) where:
                - p_left, p_right: SAM 3D predictions
                - p_gt: ground truth (or None if not available)
        """

        kpt3d_sam = batch["kpt3d_sam"]
        if "cam1" in kpt3d_sam and "cam2" in kpt3d_sam:
            left_sam_kpt3d = kpt3d_sam["cam1"].float()
            right_sam_kpt3d = kpt3d_sam["cam2"].float()
        else:
            left_sam_kpt3d = kpt3d_sam["character_cam1"].float()
            right_sam_kpt3d = kpt3d_sam["character_cam2"].float()

        # Validate shape
        for p in [left_sam_kpt3d, right_sam_kpt3d]:
            if p.ndim != 4 or p.shape[-1] != 3:
                raise ValueError(
                    f"Expected pose tensor shape (B,T,J,3) for character, got {tuple(p.shape)}"
                )

        # GT data
        unity_gt_kpt3d = None
        if "kpt3d_gt" in batch:
            if isinstance(batch["kpt3d_gt"], dict):
                unity_gt_kpt3d = batch["kpt3d_gt"].get("character")
                if unity_gt_kpt3d is not None:
                    unity_gt_kpt3d = unity_gt_kpt3d.float()
            else:
                unity_gt_kpt3d = batch["kpt3d_gt"].float()

        return (
            left_sam_kpt3d,
            right_sam_kpt3d,
            unity_gt_kpt3d,
        )

    def canonicalize(
        self, pose: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Express ``pose`` (B, T, J, 3) in the per-frame body frame of the protocol.

        Every consumer that feeds the fusion network must go through this
        method so that inputs match the representation the weights were
        trained on.
        """
        return canonicalize_pose_torch(
            pose,
            left_hip=self.left_hip_idx,
            right_hip=self.right_hip_idx,
            neck=self.neck_idx,
            mode=self.canonical_mode,
            enforce_face_z_positive=self.face_sign,
            left_eye=self.left_eye_idx,
            right_eye=self.right_eye_idx,
        )

    @staticmethod
    def check_checkpoint_protocol(
        checkpoint: Mapping[str, Any], source: str = "checkpoint"
    ) -> None:
        """Raise unless ``checkpoint`` was written by this per-frame training protocol.

        Checkpoints of the manuscript's batch-referenced first-frame protocol
        carry no ``canonical_mode`` entry; their weights expect a different
        input representation and can be neither evaluated nor fine-tuned here.
        """
        recorded = checkpoint.get("canonical_mode")
        if recorded != CANONICAL_MODE:
            raise ValueError(
                f"{source} was not trained with the per-frame protocol "
                f"(recorded canonical_mode={recorded!r}, expected {CANONICAL_MODE!r}). "
                "Train a checkpoint with dual2pose/train_unity.py on this branch."
            )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["canonical_mode"] = CANONICAL_MODE
        checkpoint["evaluation_protocol"] = dict(EVALUATION_PROTOCOL)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.check_checkpoint_protocol(checkpoint)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        """Process character variant with dual-view SAM 3D poses.

        Returns:
            Total loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        variant_results: Dict[str, Any] = {}

        p_left, p_right, p_gt = self._get_character_data(batch)
        if p_gt is None and stage != "test":
            raise RuntimeError(
                "The training protocol is supervised: train/val batches must carry kpt3d_gt"
            )

        # Per-frame body frame for both streams and the target (fixed protocol).
        left_canonical, left_transform = self.canonicalize(p_left)
        right_canonical, right_transform = self.canonicalize(p_right)
        gt_reliable = batch.get("kpt3d_gt_reliable")
        if gt_reliable is not None:
            if p_gt is None or gt_reliable.shape != p_gt.shape[:-1]:
                raise ValueError("kpt3d_gt_reliable must match the native GT joint dimensions")
            gt_reliable = gt_reliable.to(device=p_gt.device, dtype=torch.bool)
            anchors = [self.left_hip_idx, self.right_hip_idx, self.neck_idx]
            if self.face_sign:
                anchors += [self.left_eye_idx, self.right_eye_idx]
            if not gt_reliable[..., anchors].all():
                raise ValueError("Every supervised frame requires reliable GT anchors")
            if not torch.isfinite(p_gt[gt_reliable]).all():
                raise ValueError("Reliable GT coordinates must be finite")
            # Sanitize BEFORE geometry and subtraction; NaN * 0 still leaks NaNs.
            p_gt = torch.where(gt_reliable[..., None], p_gt, torch.zeros_like(p_gt))
        gt_canonical, gt_transform = self.canonicalize(p_gt) if p_gt is not None else (None, None)
        if gt_reliable is not None and gt_transform["degenerate"].any():
            raise ValueError("Reliable GT anchors must define a nondegenerate body frame")
        # Input frames whose landmarks could not define a body frame (failed
        # detections) reuse a neighbouring rotation; keep their share visible.
        degenerate_inputs = torch.cat(
            [left_transform["degenerate"].flatten(), right_transform["degenerate"].flatten()]
        )

        # Forward pass
        fused, aux = self.models(
            left_canonical,
            right_canonical,
        )
        alpha = aux["alpha"]

        # Reconstruct left/right poses from fused pose + reliability weight.
        # With delta = (p_left - p_right):
        #   p_hat = alpha * p_left_recon + (1-alpha) * p_right_recon
        #   p_left_recon - p_right_recon = delta
        delta_lr = left_canonical - right_canonical
        p_left_recon = fused + (1.0 - alpha) * delta_lr
        p_right_recon = fused - alpha * delta_lr

        loss_recon_left = torch.nn.functional.l1_loss(p_left_recon, left_canonical)
        loss_recon_right = torch.nn.functional.l1_loss(p_right_recon, right_canonical)
        loss_view_recon = 0.5 * (loss_recon_left + loss_recon_right)

        # Reliability diagnostics: cross-view discrepancy and temporal stability.
        cv_gap = torch.norm(left_canonical - right_canonical, dim=-1).mean()

        # Supervised term: L1 against the ground truth in the per-frame body
        # frame.  Test-time inference on unlabeled data has no target and
        # contributes no supervised term.
        mpjpe = fused.new_tensor(float("nan"))
        loss_sup = fused.new_zeros(())
        if gt_canonical is not None:
            if gt_reliable is None:
                loss_sup = torch.nn.functional.l1_loss(fused, gt_canonical)
                mpjpe = common13_point_errors(fused, gt_canonical).mean()
                accel_err = self._safe_accel_err(fused, gt_canonical)[1]
            else:
                loss_sup = (fused - gt_canonical).abs()[gt_reliable].mean()
                mask13 = common13_pose(gt_reliable[..., None].expand(*gt_reliable.shape, 3))[..., 0]
                errors = common13_point_errors(fused, gt_canonical)
                if not mask13.any() or not torch.isfinite(errors[mask13]).all():
                    raise ValueError("Supervised common13 scoring requires finite valid points")
                mpjpe = errors[mask13].mean()
                delta = common13_pose(fused - gt_canonical)
                accel = torch.linalg.vector_norm(delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2], dim=-1)
                triples = mask13[:, 2:] & mask13[:, 1:-1] & mask13[:, :-2]
                accel_err = accel[triples].mean() if triples.any() else fused.new_tensor(float("nan"))
            self.log(
                f"{stage}/mpjpe",
                mpjpe,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=fused.shape[0],
            )
            self.log(
                f"{stage}/accel_err",
                accel_err,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=fused.shape[0],
            )

        # Prevent gating collapse to one side.
        eps = 1e-6
        alpha_mean = alpha.mean()
        loss_alpha_balance = (alpha_mean - 0.5).abs()
        alpha_entropy = -(
            alpha * torch.log(alpha.clamp(min=eps))
            + (1.0 - alpha) * torch.log((1.0 - alpha).clamp(min=eps))
        ).mean()
        # Minimize negative entropy == maximize entropy.
        loss_alpha_entropy = -alpha_entropy

        # temporla smooth loss
        acc = fused[:, 2:] - 2.0 * fused[:, 1:-1] + fused[:, :-2]
        temporal_smooth_loss = torch.norm(acc, dim=-1).mean() if acc.numel() else fused.new_zeros(())
        loss = (
            loss_sup
            + self.lambda_view_recon * loss_view_recon
            + self.lambda_alpha_balance * loss_alpha_balance
            + self.lambda_alpha_entropy * loss_alpha_entropy
            + self.lambda_temporal_smooth * temporal_smooth_loss
        )
        total_loss = total_loss + loss

        # Log loss components
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/temporal_smooth_loss",
            temporal_smooth_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=fused.shape[0],
        )

        # Log alpha statistics
        self.log(
            f"{stage}/alpha_mean",
            alpha.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/alpha_std",
            alpha.std(),
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )

        self.log(
            f"{stage}/cross_view_gap",
            cv_gap,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/degenerate_input_frames",
            degenerate_inputs.float().mean(),
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )

        self.log(
            f"{stage}/recon_left",
            loss_recon_left,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/recon_right",
            loss_recon_right,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/recon_view",
            loss_view_recon,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/loss_alpha_balance",
            loss_alpha_balance,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/alpha_entropy",
            alpha_entropy,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )

        # Store results
        variant_results = {
            "fused": fused,
            "alpha": alpha,
            "p_left": p_left,
            "p_right": p_right,
            "left_canonical": left_canonical,
            "right_canonical": right_canonical,
            "p_left_recon": p_left_recon,
            "p_right_recon": p_right_recon,
            "ground_truth": p_gt if p_gt is not None else None,
            "ground_truth_canonical": (
                gt_canonical if gt_canonical is not None else None
            ),
        }

        # Include input masks (if provided by the dataloader/collate)
        kpt3d_mask = batch.get("kpt3d_mask", {})
        if isinstance(kpt3d_mask, dict):
            variant_results["mask_left"] = kpt3d_mask.get("cam1")
            variant_results["mask_right"] = kpt3d_mask.get("cam2")

        # Store variant results in batch for later use
        batch["_variant_results"] = variant_results

        return total_loss

    def training_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_test_start(self) -> None:
        self.test_outputs = []
        self.test_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cross-view Fusion test start")

    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        """Run inference on all variants and collect outputs."""
        self._shared_step(batch, stage="test")

        _results = dict(batch.get("_variant_results", {}))
        if isinstance(batch.get("meta"), dict):
            _results["meta"] = batch["meta"]

        self.test_outputs.append(_results)

        return torch.tensor(0.0)

    @staticmethod
    def _safe_mpjpe(
        pred: torch.Tensor, label: torch.Tensor | None
    ) -> tuple[torch.Tensor, float]:

        if label is None:
            return pred.new_zeros(*common13_pose(pred).shape[:-1]), float("nan")
        mpjpe_per_point = common13_point_errors(pred, label)
        _mean = mpjpe_per_point.mean().item()
        return mpjpe_per_point, _mean

    @staticmethod
    def _safe_accel_err(
        pred: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Acceleration error against GT via second-order temporal differences."""
        pred = common13_pose(pred)
        label = common13_pose(label) if label is not None else None
        if pred.shape[1] < 3 or label is None or label.shape[1] < 3:
            empty = pred.new_zeros((*pred.shape[:1], 0, pred.shape[2]))
            return empty, float("nan")

        pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
        label_acc = label[:, 2:] - 2.0 * label[:, 1:-1] + label[:, :-2]
        accel_err_per_point = torch.norm(pred_acc - label_acc, dim=-1)
        _mean = accel_err_per_point.mean().item()
        return accel_err_per_point, _mean

    def on_test_epoch_end(self) -> None:

        save_dir = self.test_save_dir

        # per-joint collectors: each element shape (B, T, J)
        all_fused_per_joint = []
        all_left_raw_per_joint = []
        all_right_raw_per_joint = []
        all_left_canonical_per_joint = []
        all_right_canonical_per_joint = []
        all_left_recon_per_joint = []
        all_right_recon_per_joint = []
        all_pesudo_fuse_per_joint = []
        all_pesudo_canonical_fuse_per_joint = []

        # acceleration-error collectors: each element shape (B, T-2, J)
        all_fused_accel_err = []
        all_left_raw_accel_err = []
        all_right_raw_accel_err = []
        all_left_canonical_accel_err = []
        all_right_canonical_accel_err = []
        all_pesudo_fuse_accel_err = []
        all_pesudo_canonical_fuse_accel_err = []

        for output in self.test_outputs:
            fused = output.get("fused")
            p_left = output.get("p_left")
            p_right = output.get("p_right")
            left_canonical = output.get("left_canonical")
            right_canonical = output.get("right_canonical")
            p_left_recon = output.get("p_left_recon")
            p_right_recon = output.get("p_right_recon")
            ground_truth = output.get("ground_truth")
            ground_truth_canonical = output.get("ground_truth_canonical")

            # calc mpjpe
            fused_mpjpe_per_point, _ = self._safe_mpjpe(fused, ground_truth_canonical)

            left_raw_mpjpe_per_point, _ = self._safe_mpjpe(p_left, ground_truth)
            right_raw_mpjpe_per_point, _ = self._safe_mpjpe(p_right, ground_truth)

            left_canonical_mpjpe_per_point, _ = self._safe_mpjpe(
                left_canonical, ground_truth_canonical
            )
            right_canonical_mpjpe_per_point, _ = self._safe_mpjpe(
                right_canonical, ground_truth_canonical
            )

            # Reconstruction error should be measured against canonical inputs.
            left_recon_mpjpe_per_point, _ = self._safe_mpjpe(
                p_left_recon, left_canonical
            )
            right_recon_mpjpe_per_point, _ = self._safe_mpjpe(
                p_right_recon, right_canonical
            )

            pesudo_fuse = 0.5 * (p_left + p_right)
            pesudo_fuse_mpjpe_per_point, _ = self._safe_mpjpe(pesudo_fuse, ground_truth)

            pesudo_canonical_fuse = 0.5 * (left_canonical + right_canonical)
            pesudo_canonical_fuse_mpjpe_per_point, _ = self._safe_mpjpe(
                pesudo_canonical_fuse, ground_truth_canonical
            )

            fused_accel_per_point, _ = self._safe_accel_err(
                fused, ground_truth_canonical
            )
            left_raw_accel_per_point, _ = self._safe_accel_err(p_left, ground_truth)
            right_raw_accel_per_point, _ = self._safe_accel_err(p_right, ground_truth)
            left_canonical_accel_per_point, _ = self._safe_accel_err(
                left_canonical, ground_truth_canonical
            )
            right_canonical_accel_per_point, _ = self._safe_accel_err(
                right_canonical, ground_truth_canonical
            )
            pesudo_fuse_accel_per_point, _ = self._safe_accel_err(
                pesudo_fuse, ground_truth
            )
            pesudo_canonical_fuse_accel_per_point, _ = self._safe_accel_err(
                pesudo_canonical_fuse, ground_truth_canonical
            )

            # collect per-joint tensors (B, T, J) -> flatten to (N, J)
            all_fused_per_joint.append(
                fused_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_left_raw_per_joint.append(
                left_raw_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_right_raw_per_joint.append(
                right_raw_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_left_canonical_per_joint.append(
                left_canonical_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_right_canonical_per_joint.append(
                right_canonical_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_left_recon_per_joint.append(
                left_recon_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_right_recon_per_joint.append(
                right_recon_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_pesudo_fuse_per_joint.append(
                pesudo_fuse_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_pesudo_canonical_fuse_per_joint.append(
                pesudo_canonical_fuse_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )

            if fused_accel_per_point.numel() > 0:
                all_fused_accel_err.append(
                    fused_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_left_raw_accel_err.append(
                    left_raw_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_right_raw_accel_err.append(
                    right_raw_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_left_canonical_accel_err.append(
                    left_canonical_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_right_canonical_accel_err.append(
                    right_canonical_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_pesudo_fuse_accel_err.append(
                    pesudo_fuse_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_pesudo_canonical_fuse_accel_err.append(
                    pesudo_canonical_fuse_accel_per_point.detach().cpu().flatten(0, -2)
                )

        # aggregate metrics across all samples: concatenate (N, J) then reduce.
        def _per_joint_mean(lst):
            if not lst:
                return []
            return torch.cat(lst, dim=0).mean(dim=0).tolist()  # list of length J

        def _global_mean(lst):
            if not lst:
                return float("nan")
            return float(torch.cat(lst, dim=0).mean().item())

        per_joint_stats = {
            "fused": _per_joint_mean(all_fused_per_joint),
            "left_raw": _per_joint_mean(all_left_raw_per_joint),
            "right_raw": _per_joint_mean(all_right_raw_per_joint),
            "left_canonical": _per_joint_mean(all_left_canonical_per_joint),
            "right_canonical": _per_joint_mean(all_right_canonical_per_joint),
            "left_recon": _per_joint_mean(all_left_recon_per_joint),
            "right_recon": _per_joint_mean(all_right_recon_per_joint),
            "pesudo_fuse": _per_joint_mean(all_pesudo_fuse_per_joint),
            "pesudo_canonical_fuse": _per_joint_mean(
                all_pesudo_canonical_fuse_per_joint
            ),
        }

        gt_summary = {
            "fused_mpjpe_mean": _global_mean(all_fused_per_joint),
            "sam3d_left_mpjpe_mean": _global_mean(all_left_raw_per_joint),
            "sam3d_right_mpjpe_mean": _global_mean(all_right_raw_per_joint),
            "left_canonical_mpjpe_mean": _global_mean(all_left_canonical_per_joint),
            "right_canonical_mpjpe_mean": _global_mean(all_right_canonical_per_joint),
            "pesudo_fuse_mpjpe_mean": _global_mean(all_pesudo_fuse_per_joint),
            "pesudo_canonical_fuse_mpjpe_mean": _global_mean(
                all_pesudo_canonical_fuse_per_joint
            ),
        }

        recon_summary = {
            "left_recon_mpjpe_mean": _global_mean(all_left_recon_per_joint),
            "right_recon_mpjpe_mean": _global_mean(all_right_recon_per_joint),
        }

        accel_summary = {
            "fused_accel_err_mean": _global_mean(all_fused_accel_err),
            "sam3d_left_accel_err_mean": _global_mean(all_left_raw_accel_err),
            "sam3d_right_accel_err_mean": _global_mean(all_right_raw_accel_err),
            "left_canonical_accel_err_mean": _global_mean(all_left_canonical_accel_err),
            "right_canonical_accel_err_mean": _global_mean(
                all_right_canonical_accel_err
            ),
            "pesudo_fuse_accel_err_mean": _global_mean(all_pesudo_fuse_accel_err),
            "pesudo_canonical_fuse_accel_err_mean": _global_mean(
                all_pesudo_canonical_fuse_accel_err
            ),
        }

        # save results (disabled: eval scripts get tensors from memory via _flatten_test_outputs)
        # save_file = save_dir / f"outputs.pt"
        # torch.save(self.test_outputs, save_file)

        # report summary to txt
        txt_file = save_dir / f"report.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("Cross-View Fusion Test Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"evaluation_protocol: {EVALUATION_PROTOCOL['name']}\n")
            f.write("joint_subset: common13\noutput_alignment: none\n")
            f.write(f"joint_order: {', '.join(COMMON13_NAMES)}\n")
            f.write(
                f"canonical_mode: {self.canonical_mode} (hips={self.left_hip_idx},{self.right_hip_idx}; "
                f"neck={self.neck_idx}; face_sign={self.face_sign}; eyes={self.left_eye_idx},{self.right_eye_idx})\n"
            )
            f.write("[Canonical MPJPE vs Ground Truth]\n")
            f.write("Primary MPJPE compares common-13 outputs directly to per-frame canonical GT; no output alignment.\n")
            for k, v in gt_summary.items():
                f.write(f"{k}: {v:.4f}\n")

            f.write("\n[Raw SAM3D vs Ground Truth]\n")
            f.write(
                "These values compare the original pseudo GT poses before canonicalization.\n"
            )
            f.write(
                f"sam3d_cam1_mpjpe_mean: {gt_summary['sam3d_left_mpjpe_mean']:.4f}\n"
            )
            f.write(
                f"sam3d_cam2_mpjpe_mean: {gt_summary['sam3d_right_mpjpe_mean']:.4f}\n"
            )

            f.write("\n[Reconstruction Error (vs canonical input)]\n")
            f.write("This section is against canonicalized camera inputs.\n")
            for k, v in recon_summary.items():
                f.write(f"{k}: {v:.4f}\n")

            f.write("\n[Acceleration Error vs Ground Truth]\n")
            f.write(
                "All values compare second-order temporal differences against GT.\n"
            )
            for k, v in accel_summary.items():
                f.write(f"{k}: {v:.4f}\n")

            f.write(
                "\n[Per-Joint Canonical MPJPE vs Ground Truth (joint index: mean error)]\n"
            )
            for variant_name in [
                "fused",
                "sam3d_cam1",
                "sam3d_cam2",
                "left_canonical",
                "right_canonical",
                "pesudo_fuse",
                "pesudo_canonical_fuse",
            ]:
                if variant_name == "sam3d_cam1":
                    joint_values = per_joint_stats["left_raw"]
                elif variant_name == "sam3d_cam2":
                    joint_values = per_joint_stats["right_raw"]
                else:
                    joint_values = per_joint_stats[variant_name]
                f.write(f"  {variant_name}:\n")
                for j, val in enumerate(joint_values):
                    f.write(f"    joint_{j:02d}: {val:.4f}\n")

            f.write("\n[Per-Joint Reconstruction Error (vs canonical input)]\n")
            for variant_name in ["left_recon", "right_recon"]:
                joint_values = per_joint_stats[variant_name]
                f.write(f"  {variant_name}:\n")
                for j, val in enumerate(joint_values):
                    f.write(f"    joint_{j:02d}: {val:.4f}\n")

        logger.info("Saved test report to %s", txt_file)

    def configure_optimizers(self):
        """Configure optimizer for the character model."""
        optimizer = torch.optim.AdamW(
            self.models.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # T_max counts optimizer updates, including accumulation.
                "interval": "step",
                "frequency": 1,
            },
        }
