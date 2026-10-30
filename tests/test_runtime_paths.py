import hashlib
import json
from pathlib import Path
import pytest


def test_relocated_inputs_preserve_bytes_and_original_metadata(tmp_path, monkeypatch):
    from dual2pose.runtime_paths import input_path
    source=tmp_path/'old';target=tmp_path/'new';target.mkdir()
    payload=b'original index';(target/'index.json').write_bytes(payload)
    record={'index':str(source/'index.json'),'sha256':hashlib.sha256(payload).hexdigest()}
    before=json.dumps(record)
    monkeypatch.setenv('DUAL2POSE_INPUT_PATH_MAP',json.dumps({str(source):str(target)}))
    assert input_path(record['index'])==target/'index.json'
    assert hashlib.sha256(input_path(record['index']).read_bytes()).hexdigest()==record['sha256']
    assert json.dumps(record)==before
    assert input_path(str(source)+'_other/file')==Path(str(source)+'_other/file')


def test_relocation_longest_prefix_and_no_missing_fallback(tmp_path,monkeypatch):
    from dual2pose.runtime_paths import input_path
    monkeypatch.setenv('DUAL2POSE_INPUT_PATH_MAP',json.dumps({'/old':'/new','/old/data':'/dataset'}))
    assert input_path('/old/data/x')==Path('/dataset/x')
    assert not input_path('/old/data/missing').exists()
    with pytest.raises(ValueError): input_path('/old/data/../../escape')


def test_relocation_is_opt_in(monkeypatch):
    from dual2pose.runtime_paths import input_path
    monkeypatch.delenv('DUAL2POSE_INPUT_PATH_MAP',raising=False)
    assert input_path('/old/path')==Path('/old/path')
