"""Training-only guards for finite but degenerate MetaPose solutions."""
import numpy as np


POLICY = dict(radius_ratio_min=0.1, radius_ratio_max=10.0,
              max_bad_frame_fraction=0.01, min_median_camera_scale=1e-4,
              max_loss_rebound=5.0)


def geometry_health(projected, reference, scales):
    projected, reference, scales = (np.asarray(v, dtype=np.float64)
                                    for v in (projected, reference, scales))
    if (projected.shape != reference.shape or reference.ndim != 4
            or reference.shape[1:] != (2, 13, 3) or not len(reference)
            or scales.shape != reference.shape[:2]):
        raise ValueError('Expected nonempty matching (N,2,13,3) poses and (N,2) scales')
    if any(not np.isfinite(v).all() for v in (projected, reference, scales)):
        raise ValueError('Nonfinite geometry')
    def radius(v):
        return np.linalg.norm(v-v.mean(-2, keepdims=True), axis=-1).mean(-1)
    base = radius(reference)
    if np.any(base <= 1e-8) or np.any(scales <= 0):
        raise ValueError('Degenerate reference or nonpositive camera scale')
    ratio = radius(projected)/base
    bad = ((ratio < POLICY['radius_ratio_min']) |
           (ratio > POLICY['radius_ratio_max'])).any(-1)
    fraction, median_scale = float(bad.mean()), float(np.median(scales))
    return dict(passed=fraction <= POLICY['max_bad_frame_fraction'] and
                median_scale >= POLICY['min_median_camera_scale'],
                frames=len(reference), bad_frame_fraction=fraction,
                radius_ratio_min=float(ratio.min()), radius_ratio_median=float(np.median(ratio)),
                radius_ratio_max=float(ratio.max()), camera_scale_min=float(scales.min()),
                camera_scale_median=median_scale, camera_scale_max=float(scales.max()),
                policy=POLICY)


def training_gate(geometry, loss, initial_loss, best_loss):
    values = np.asarray([loss, initial_loss, best_loss], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid training loss')
    reasons = []
    if not geometry['passed']:
        reasons.append('degenerate_geometry')
    if loss > initial_loss + 1e-10:
        reasons.append('worse_than_initialization')
    if loss > POLICY['max_loss_rebound']*max(best_loss, 1e-10):
        reasons.append('training_loss_rebound')
    return dict(passed=not reasons, reasons=reasons, geometry=geometry,
                full_train_2d_mse=float(loss), initial_2d_mse=float(initial_loss),
                best_full_train_2d_mse=float(best_loss))
