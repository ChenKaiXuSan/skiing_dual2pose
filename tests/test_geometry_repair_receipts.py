import hashlib,json
import pytest
from pegasus.runtime.chain import validate_cluster


def test_artifact_bundle_receipt_tracks_outputs_and_rejects_changed_bytes(tmp_path):
    a=tmp_path/'array.npy';a.write_bytes(b'known-output')
    report=tmp_path/'verified.json';report.write_text(json.dumps({'all_required_results_available':True,'artifacts':{str(a):hashlib.sha256(a.read_bytes()).hexdigest()}}))
    job={'validation':{'kind':'verified_artifacts','path':str(report)}}
    result=validate_cluster(job,{})
    assert str(a) in result['artifacts']
    a.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact'):validate_cluster(job,{})
