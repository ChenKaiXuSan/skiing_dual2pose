"""Framewise PA-MPJPE with a proper (reflection-free) similarity alignment.

The alignment convention follows VideoPose3D's Protocol #2 definition:
https://github.com/facebookresearch/VideoPose3D/blob/main/common/loss.py
This implementation uses row-vector covariance X.T @ Y, hence R = U D V.T.
Each reported joint subset is aligned independently in float64. Degenerate
predictions are mapped to the target centroid and retained in the denominator.
"""
from __future__ import annotations

import numpy as np
from dual2pose.experiments.run_main_baselines import MetricAccumulator

PA_PROTOCOL = {
    'alignment': 'per frame; proper rotation, isotropic scale, translation',
    'reflection_allowed': False,
    'joint_subset_fit': 'independent fit on each reported subset',
    'aggregation': 'mean Euclidean distance over all joint-frame points',
    'precision': 'numpy float64 SVD',
    'prediction_centered_squared_norm_threshold': 1e-12,
    'degenerate_prediction': 'map to target centroid; retain all points',
    'evaluation_only': True,
    'affects_mpjpe_or_acceleration': False,
}


def pa_joint_errors(prediction, target):
    """Return (..., J) aligned distances and (...) degenerate-frame flags."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim < 3 or pred.shape[-1] != 3 or pred.shape[-2] < 3:
        raise ValueError('Expected matching (..., J, 3) arrays, with at least three joints')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('PA-MPJPE requires finite predictions and targets')
    x = pred - pred.mean(axis=-2, keepdims=True)
    y = truth - truth.mean(axis=-2, keepdims=True)
    variance = np.sum(x * x, axis=(-2, -1))
    degenerate = variance <= PA_PROTOCOL['prediction_centered_squared_norm_threshold']
    u, singular, vt = np.linalg.svd(np.swapaxes(x, -2, -1) @ y)
    signs = np.ones_like(singular)
    signs[..., -1] = np.where(np.linalg.det(u @ vt) < 0, -1., 1.)
    rotation = (u * signs[..., None, :]) @ vt
    numerator = np.sum(singular * signs, axis=-1)
    scale = np.divide(np.maximum(numerator, 0.), variance,
                      out=np.zeros_like(variance), where=~degenerate)
    aligned_centered = scale[..., None, None] * (x @ rotation)
    distances = np.linalg.norm(aligned_centered - y, axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite PA-MPJPE after alignment')
    return distances, degenerate


class PAMetricAccumulator(MetricAccumulator):
    """Add PA-MPJPE without applying alignment to the original two metrics."""
    def __init__(self):
        super().__init__()
        self.pa_distance = 0.
        self.pa_points = 0
        self.pa_frames = 0
        self.pa_degenerate_frames = 0

    def update(self, pred, target):
        super().update(pred, target)
        error, degenerate = pa_joint_errors(pred.detach().cpu().numpy(), target.detach().cpu().numpy())
        self.pa_distance += error.sum(dtype=np.float64).item()
        self.pa_points += error.size
        self.pa_frames += degenerate.size
        self.pa_degenerate_frames += int(degenerate.sum())

    def result(self):
        result = super().result()
        if self.pa_points != self.points:
            raise ValueError('PA-MPJPE coverage differs from MPJPE coverage')
        result.update(pa_mpjpe=self.pa_distance / self.pa_points,
                      pa_point_count=self.pa_points, pa_frame_count=self.pa_frames,
                      pa_degenerate_prediction_frames=self.pa_degenerate_frames)
        return result
