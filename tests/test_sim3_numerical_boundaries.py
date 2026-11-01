"""Independent rotation/point recovery checks for Sim(3) numerical boundaries."""
import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from dual2pose.models.crossview_fusion import rotation_matrix_to_rotvec
from dual2pose.models.sim3 import apply_sim3, invert_sim3


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_half_turns_reconstruct_instead_of_becoming_identity(dtype):
    # Includes mixed-sign, non-axis-aligned axes; pi's axis sign is ambiguous.
    axes = np.array([[1,0,0], [0,1,0], [0,0,1], [1,-2,3]], dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    matrices = Rotation.from_rotvec(axes * math.pi).as_matrix()
    matrices[:3] = np.array([np.diag([1,-1,-1]), np.diag([-1,1,-1]), np.diag([-1,-1,1])])
    actual = rotation_matrix_to_rotvec(torch.tensor(matrices, dtype=dtype))
    np.testing.assert_allclose(actual.norm(dim=-1).numpy(), math.pi, atol=1e-6)
    np.testing.assert_allclose(Rotation.from_rotvec(actual.numpy()).as_matrix(), matrices, atol=1e-6)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_small_angles_and_both_sides_of_pi_reconstruct(dtype):
    axes = np.array([[1,-2,3]], dtype=float) / math.sqrt(14)
    angles = np.array([0, 1e-7, -1e-5, 1e-4, .4, math.pi-1e-5, math.pi+1e-5, -math.pi+1e-5])
    matrices = Rotation.from_rotvec(angles[:,None] * axes).as_matrix()
    actual = rotation_matrix_to_rotvec(torch.tensor(matrices, dtype=dtype))
    np.testing.assert_allclose(actual.numpy()[:5], angles[:5,None] * axes, atol=1e-8, rtol=1e-5)
    np.testing.assert_allclose(Rotation.from_rotvec(actual.numpy()).as_matrix(), matrices, atol=5e-7)
    assert actual.dtype == dtype
    assert torch.all(actual.norm(dim=-1) <= math.pi + 5e-7)


def test_identity_rotation_has_finite_correct_local_derivative():
    matrix = torch.eye(3, dtype=torch.float64).unsqueeze(0).requires_grad_()
    assert torch.autograd.gradcheck(rotation_matrix_to_rotvec, (matrix,), atol=1e-5)
    rotation_matrix_to_rotvec(matrix).sum().backward()
    assert torch.isfinite(matrix.grad).all()


@pytest.mark.parametrize('batch', [1, 3])
def test_inverse_keeps_batch_and_recovers_independent_points(batch):
    # Directly construct y by the declared formula, not apply_sim3 under test.
    points = torch.tensor(np.random.default_rng(42).normal(size=(batch,7,3)), dtype=torch.float64)
    rotations = torch.tensor(Rotation.from_rotvec([[.3,-.5,.2],[-.7,.1,.4],[.2,.5,-.9]][:batch]).as_matrix())
    scales = torch.tensor([.7,2.,1.4][:batch], dtype=torch.float64)
    shifts = torch.tensor([[1.,2.,-3.],[4.,-2.,.3],[-1.,.6,2.]][:batch], dtype=torch.float64)
    transformed = torch.stack([scales[i] * (points[i] @ rotations[i].T) + shifts[i] for i in range(batch)])
    inv_s, inv_r, inv_t = invert_sim3(scales, rotations, shifts)
    assert inv_s.shape == (batch,)
    assert inv_r.shape == (batch,3,3)
    assert inv_t.shape == (batch,3)
    torch.testing.assert_close(apply_sim3(transformed, inv_s, inv_r, inv_t), points, atol=1e-12, rtol=1e-12)


def test_unbatched_inverse_preserves_scalar_and_vector_shapes():
    scale = torch.tensor(2., dtype=torch.float64)
    rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]], dtype=torch.float64)
    shift = torch.tensor([2.,4.,6.], dtype=torch.float64)
    inv_s, inv_r, inv_t = invert_sim3(scale, rotation, shift)
    assert inv_s.shape == () and inv_r.shape == (3,3) and inv_t.shape == (3,)
    torch.testing.assert_close(inv_t, torch.tensor([-2.,1.,-3.], dtype=torch.float64))
