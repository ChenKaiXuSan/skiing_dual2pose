"""Catch horizontal-aspect inflation and double correction of Unity intrinsics."""
import json
from pathlib import Path
import numpy as np
import pytest
from dual2pose.eval.main_baseline_geometry import unity_projection_matrix, project_points
from dual2pose.eval.evaluate_occluded_geometry import camera_matrices


def camera(tmp_path, **changes):
    folder = tmp_path / 'data_pole_ski/male/cameras/L0_A000'
    folder.mkdir(parents=True)
    intr = dict(width=200, height=100, fx=100., fy=50., cx=99.5, cy=49.5)
    intr.update(changes)
    (folder/'intrinsics.json').write_text(json.dumps(intr))
    (folder/'extrinsics.json').write_text(json.dumps({'t_world_cam_4x4': np.eye(4).ravel().tolist()}))
    return folder


def test_legacy_camera_matches_independent_viewport_annotation(tmp_path):
    camera(tmp_path)
    p = unity_projection_matrix(tmp_path, 'male', 'capture_L0_A000')
    # Vertical FOV 90, aspect 2: viewport=(.55,.6), top-left pixels=(109.45,39.6).
    np.testing.assert_allclose(project_points(p, np.array([[1., 1., -5.]])), [[109.45, 39.6]], atol=1e-10)


def test_versioned_camera_is_not_corrected_twice(tmp_path):
    camera(tmp_path, fx=49.75, fy=49.5, projection_convention='unity_viewport_w_minus_1_v2')
    p = unity_projection_matrix(tmp_path, 'male', 'capture_L0_A000')
    np.testing.assert_allclose(project_points(p, np.array([[1., 1., -5.]])), [[109.45, 39.6]], atol=1e-10)


def test_occlusion_perturbations_use_same_corrected_intrinsics(tmp_path):
    camera(tmp_path)
    ps, ks, _ = camera_matrices(tmp_path, [dict(person_id='male',cam1_id='capture_L0_A000',cam2_id='capture_L0_A000')])
    k = ks[('male','capture_L0_A000')]
    np.testing.assert_allclose(k, [[49.75,0,99.5],[0,49.5,49.5],[0,0,1]])
    np.testing.assert_allclose(np.linalg.inv(k) @ ps[('male','capture_L0_A000')], np.diag([1,-1,-1,1])[:3])


@pytest.mark.parametrize('changes', [dict(projection_convention='unknown'),dict(width=0),dict(fx=77.)])
def test_unknown_or_inconsistent_exports_are_rejected(tmp_path, changes):
    camera(tmp_path, **changes)
    with pytest.raises(ValueError): unity_projection_matrix(tmp_path,'male','capture_L0_A000')


@pytest.mark.parametrize('subject', ['male', 'female'])
def test_capture_export_before_camera_setup_uses_render_dimensions(subject):
    """Check actual C# focal expressions when shared metadata precedes RT setup.

    This source-level numerical contract is not a Unity Editor integration test.
    The simulated initial camera aspect deliberately differs from the render RT.
    """
    import math
    import re

    source = (Path(__file__).resolve().parents[1] / 'unity' / subject /
              'Assets/Scripts/OneCameraCaptureFrame.cs').read_text()
    body = source.split('void WriteCameraIntrinsics(string path)', 1)[1].split('var intr', 1)[0]
    env = dict(captureWidth=200, captureHeight=100, camera_aspect=1.,
               camera_fov=90., radians=math.pi / 180., tan=math.tan)
    for name, expression in re.findall(r'float\s+(\w+)\s*=\s*([^;]+);', body):
        expression = expression.replace('cam.fieldOfView', 'camera_fov').replace('cam.aspect', 'camera_aspect')
        expression = expression.replace('Mathf.Deg2Rad', 'radians').replace('Mathf.Tan', 'tan')
        expression = expression.replace('(float)', '')
        expression = re.sub(r'(?<=\d)f\b', '', expression)
        env[name] = eval(expression, {'__builtins__': {}}, env)
    # Vertical FOV 90 / capture aspect 2 yields viewport (.55, .6).
    pixel = [env['fx'] * 1. / 5. + 99.5, -env['fy'] * 1. / 5. + 49.5]
    np.testing.assert_allclose(pixel, [109.45, 39.6], atol=1e-10)
