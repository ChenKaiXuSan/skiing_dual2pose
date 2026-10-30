import json
from pathlib import Path

import pytest

from dual2pose.experiments.resume_metapose import verify_bundle, sha, reserve_output
from dual2pose.experiments.run_metapose_completion_recovery import replace_metapose_rows, execute_steps


def bundle(tmp_path):
    cfg = dict(stages=3, epochs=10, seed=42, no_3d_supervision=True,
               no_supplied_calibration=True, val_dir='val')
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    history = [dict(stage=s, epoch=e, train_2d_mse=0.1,
                    val_2d_mse=abs(e-5)*0.01, selected_epoch=min(e,5))
               for s in range(1,4) for e in range(1,11)]
    (tmp_path/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    selected=[]
    for s in range(1,4):
        p=tmp_path/f'stage_{s}.weights.h5';p.write_bytes(bytes([s]))
        selected.append(dict(stage=s,epoch=5,checkpoint_sha256=sha(p)))
    (tmp_path/'selected_stages.json').write_text(json.dumps(selected))
    return selected


def test_resume_requires_all_training_and_selected_best_validation(tmp_path):
    bundle(tmp_path)
    assert len(verify_bundle(tmp_path)[1])==3
    p=tmp_path/'epochs.jsonl'; p.write_text('\n'.join(p.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match='history'):verify_bundle(tmp_path)


def test_resume_rejects_changed_weights(tmp_path):
    bundle(tmp_path);(tmp_path/'stage_2.weights.h5').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):verify_bundle(tmp_path)


def test_resume_rejects_nonbest_selection(tmp_path):
    selected=bundle(tmp_path);selected[2]['epoch']=10
    (tmp_path/'selected_stages.json').write_text(json.dumps(selected))
    with pytest.raises(ValueError, match='selection'):verify_bundle(tmp_path)


def test_resume_never_reuses_existing_output(tmp_path):
    out=tmp_path/'output';reserve_output(out);(out/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):reserve_output(out)
    assert (out/'sentinel').read_text()=='keep'


def test_summary_requires_both_replacements_and_preserves_other_rows():
    rows=[dict(dataset=d,method=m,metrics={'mpjpe':9})
          for d in ('unity','ski') for m in ('metapose','canonfuse3d')]
    repl={d:dict(metrics={'mpjpe':v},source=d,recipe=d) for d,v in [('unity',.1),('ski',.2)]}
    updated=replace_metapose_rows(rows,repl)
    assert [r['metrics']['mpjpe'] for r in updated]==[.1,9,.2,9]
    assert rows[0]['metrics']['mpjpe']==9
    with pytest.raises(ValueError):replace_metapose_rows(rows,{'unity':repl['unity']})
    with pytest.raises(ValueError):replace_metapose_rows(rows+rows[:1],repl)


def test_failed_step_stops_dependents_and_preserves_error(tmp_path):
    import sys
    steps=[('failure',[sys.executable,'-c','raise SystemExit(7)']),
           ('dependent',[sys.executable,'-c','raise SystemExit(0)'])]
    with pytest.raises(RuntimeError):execute_steps(tmp_path,steps,cwd=tmp_path,env=None)
    state=json.loads((tmp_path/'state.json').read_text())
    assert state['status']=='failed' and state['steps'][0]['return_code']==7
    assert state['steps'][1]['status']=='blocked'
    assert not (tmp_path/'dependent.log').exists()
