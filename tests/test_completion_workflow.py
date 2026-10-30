"""Supplement tasks must preserve cohort, provenance and prior artifacts."""
import json
from pathlib import Path
import numpy as np
import pytest
from dual2pose.experiments import workflow_completion as c


def test_shift_mask_uses_same_joint_frames_for_every_offset():
    rel = np.ones((12, 15), bool)
    rel[6, 2] = False
    rel[8, 6] = False
    ids = np.array([[0, 3, 4, 5, 6, 7, 11]])
    shifts = [-1, 0, 1]
    mask = c.shared_shift_mask(ids, rel, [6, 7, 14, 0, 1], shifts)
    assert mask.shape == (1, 7, 13)
    assert not mask[0, 0].any() and not mask[0, -1].any()
    assert mask[0, 1].all()
    assert not mask[0, 3, 0]  # +1 hits unreliable shoulder
    assert not mask[0, 5].any()  # +1 hits unreliable hip anchor
    assert mask[0, 3, 1:].all()


def test_choose_qualitative_window_is_held_out_and_not_error_selected():
    record = dict(blocks={'test':[100,300]}, window_starts={'test':[110,150,200]},
                  metrics={'model':{'mpjpe_mm':1}})
    assert c.qualitative_window(record)==150
    record['window_starts']['test']=[90,150,200]
    with pytest.raises(ValueError,match='test block'):c.qualitative_window(record)


def test_preservation_detects_changed_or_removed_files(tmp_path):
    p=tmp_path/'old.json';p.write_text('old');snapshot=c.protection_snapshot([p])
    assert c.verify_protection(snapshot)==[]
    p.write_text('new');assert c.verify_protection(snapshot)==[str(p)]
    p.unlink();assert c.verify_protection(snapshot)==[str(p)]


def test_metapose_validation_rejects_incomplete_training(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):
        c.validate_metapose_training(tmp_path,'ski')


def test_completion_graph_retrains_metapose_and_keeps_old_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "__name__", "__main__")
    old=tmp_path/'old';old.mkdir()
    job={'id':'eval_mask_left_random_0','family':'evaluations','deps':['main_unity'],
         'resource':'gpu','commands':[['python','-m','eval']],
         'env':{'EVAL_OUTPUT_ROOT':str(old/'evaluations/eval_mask_left_random_0')},
         'validation':{'path':str(old/'evaluations/eval_mask_left_random_0/*.csv')}}
    (old/'manifest.json').write_text(json.dumps({'jobs':[job]}))
    ctx={'root':str(tmp_path/'new'),'repo':str(tmp_path/'repo'),'old_root':str(old),
         'python':'python','raw_cache':'cache'}
    jobs=c.build_jobs(ctx)
    assert all(command[2] != '__main__' for j in jobs for command in j['commands'] if len(command)>2 and command[1]=='-m')
    by={j['id']:j for j in jobs}
    assert by['metapose_unity_train']['resource']=='gpu'
    assert by['metapose_ski_train']['resource']=='gpu'
    assert by['metapose_unity_score']['deps']==['metapose_unity_train']
    assert all('evaluate_metapose' not in str(by[f'metapose_{ds}_train']['commands']) for ds in ['unity','ski'])
    assert str(old/'evaluations') not in json.dumps(by['eval_mask_left_random_0']['env'])
    assert set(by['final_coverage']['deps'])==set(by)-{'final_coverage'}


def test_retrained_row_updates_metrics_and_provenance(tmp_path):
    p=tmp_path/'metrics.json';p.write_text(json.dumps({'metapose':{'mpjpe':.1}}))
    row={'method':'metapose','metrics':{'mpjpe':9},'source':'old','source_sha256':'old'}
    new=c.retrained_row(row,p)
    assert row['source']=='old'
    assert new['metrics']['mpjpe']==.1
    assert new['source']==str(p.resolve())
    assert new['source_sha256']==c.flow.sha(p)
    assert new['prediction_stage']=='fresh_training_and_full_inference_this_completion_run'
