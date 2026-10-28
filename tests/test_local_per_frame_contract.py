"""Synthetic train/save/load/direct-score closures for current local baselines."""
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dual2pose.eval.main_baseline_data import canonical_batch, sha256
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL
from dual2pose.eval.evaluate_main_baseline_pa import selected_model
from dual2pose.experiments.run_main_baselines import _save_checkpoint, evaluate
from dual2pose.models.main_baselines import build_baseline
from dual2pose.models.dlt_residual_baseline import DLTResidualMLP, dlt_residual_loss


@pytest.mark.parametrize('dataset,joints', [('unity', 15), ('ski', 13)])
@pytest.mark.parametrize('method', ['tcn', 'smoothnet', 'dlt_mlp'])
def test_train_save_load_score_native_skeleton(dataset, joints, method):
    torch.manual_seed(321)
    raw = tuple(torch.randn(2, 30, joints, 3) for _ in range(3))
    left, right, target = canonical_batch(*raw, dataset=dataset)
    model = DLTResidualMLP(joints) if method == 'dlt_mlp' else build_baseline(method, num_joints=joints, time_window=30)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    reprojection = torch.zeros(2, 30, joints)
    validity = torch.ones_like(reprojection, dtype=torch.bool)
    def predict(network):
        return network(left, reprojection, validity) if method == 'dlt_mlp' else network(left, right)
    loss = dlt_residual_loss(predict(model), target)
    loss.backward()
    optimizer.step()
    model.eval()
    expected = predict(model).detach()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'new.pt'
        _save_checkpoint(path, model, optimizer, 0, 1., {'dataset':dataset})
        record = {'checkpoint':str(path), 'checkpoint_sha256':sha256(path), 'selected_epoch':0,
                  'provenance':{'canonical_mode':'per_frame', 'evaluation_protocol':dict(EVALUATION_PROTOCOL)}}
        restored = selected_model(method, record, joints, 'cpu')
        torch.testing.assert_close(predict(restored), expected, rtol=0, atol=0)
        if method != 'dlt_mlp':
            score = evaluate(restored, raw, dataset, 1)['common13']
            assert score['point_count'] == 2*30*13
            assert np.isfinite(score['mpjpe'])


@pytest.mark.parametrize('dataset,joints', [('unity', 15), ('ski', 13)])
def test_geometry_direct_output_keeps_learned_translation(dataset, joints):
    from dual2pose.experiments.run_geometry_baselines import evaluate as evaluate_geometry
    raw = np.random.default_rng(5).normal(size=(3, 6, joints, 3)).astype(np.float32)
    inputs = SimpleNamespace(target=raw, dlt=raw, robust_dlt=raw,
        reprojection_error_px=np.zeros(raw.shape[:-1], np.float32),
        dlt_valid=np.ones(raw.shape[:-1], bool))
    class Translation(torch.nn.Module):
        def forward(self, dlt, reprojection, valid):
            return dlt + dlt.new_tensor([2., 0., 0.])
    scores = evaluate_geometry(inputs, dataset=dataset, batch_size=2, device='cpu', method='dlt_mlp', model=Translation())
    assert scores['common13']['mpjpe'] == pytest.approx(2., abs=1e-6)
    assert scores['common13']['point_count'] == 3*6*13


def test_selected_model_rejects_historical_result_even_for_geometry():
    with pytest.raises(ValueError, match='protocol'):
        selected_model('dlt', {'provenance':{}}, 15, 'cpu')


@pytest.mark.parametrize('variant', ['full', 'fixed_gate', 'no_output_residual'])
def test_gate_one_step_save_load_score_smoke(tmp_path, variant):
    from dual2pose.experiments.run_gate_ablation import run_variant, build_model, evaluate as evaluate_gate
    from dual2pose.eval.main_baseline_data import require_current_protocol
    arrays = tuple(torch.randn(2, 6, 15, 3) for _ in range(3))
    args = SimpleNamespace(seed=42, output=tmp_path, device='cpu', epochs=1,
                           batch_size=2, smoke_steps=1)
    run_variant(args, variant, arrays, arrays, {'synthetic':True}, {'synthetic':True})
    saved = torch.load(tmp_path/variant/'best.pt', weights_only=False)
    require_current_protocol(saved)
    model = build_model(variant)
    model.load_state_dict(saved['state_dict'], strict=True)
    scores = evaluate_gate(model, arrays, batch_size=1)
    assert scores['common13']['point_count'] == 2*6*13
    assert np.isfinite(scores['common13']['mpjpe'])
    assert 'common13' in saved['config']['selection']
    assert 'no scheduler' in saved['config']['optimizer']
