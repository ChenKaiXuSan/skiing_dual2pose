"""Primary metrics for active per-frame fusion pipelines.

Inputs and targets have already been canonicalized per frame. Predictions are
scored directly: no output recentering, rotation, scale fit or recanonicalization.
The model's native 15-joint or common-13 skeleton is selected only for scoring.
"""
import torch

from dual2pose.map_config import FILTERED_15_TO_COMMON_13_INDICES, SKI_PTZ_COMMON_KPTS_MAPPING


COMMON13_NAMES = tuple(SKI_PTZ_COMMON_KPTS_MAPPING.values())
EVALUATION_PROTOCOL = {
    "name": "common13_direct_per_frame_v1",
    "joint_subset": "common13",
    "canonical_mode": "per_frame",
    "output_alignment": "none",
    "reduction": "mean_joint_frame_distance",
}


def common13_pose(pose):
    """Select common-13 from an ordered (..., J, 3) NumPy array or tensor."""
    if pose.ndim < 2 or pose.shape[-1] != 3:
        raise ValueError(f"Expected (..., J, 3) pose, got {pose.shape}")
    if pose.shape[-2] == 15:
        return pose[..., FILTERED_15_TO_COMMON_13_INDICES, :]
    if pose.shape[-2] == 13:
        return pose
    raise ValueError(f"Expected the 15-joint or common-13 skeleton, got {pose.shape[-2]} joints")


def common13_point_errors(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Euclidean errors (..., 13) with no additional output alignment."""
    prediction, target = common13_pose(prediction), common13_pose(target)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shapes differ: {prediction.shape} vs {target.shape}")
    return torch.linalg.vector_norm(prediction - target, dim=-1)
