import numpy as np
import pytest
from dual2pose.eval.metapose_unity_depth import DepthHealth


def poses(n=100):
    return np.random.default_rng(42).normal(size=(n,2,13,3))


def health(pred,ref,chunk=None):
    h=DepthHealth()
    for s in range(0,len(ref),chunk or len(ref)):
        h.update(pred[s:s+(chunk or len(ref))],ref[s:s+(chunk or len(ref))],np.ones((len(ref[s:s+(chunk or len(ref))]),2)))
    return h.result()


def test_identity_and_translation_pass():
    p=poses();r=health(p+[4,8,-100],p)
    assert r['passed'] and r['depth_bad_view_frames']==0 and r['frames']==100


def test_depth_reflection_fails_despite_same_xy_and_radius():
    p=poses();bad=p*[1,1,-1];assert np.array_equal(bad[...,:2],p[...,:2])
    r=health(bad,p);assert not r['passed'] and r['depth_bad_fraction']==1
    assert r['geometry_bad_fraction']==0


def test_partial_reflection_is_not_hidden_by_aggregate_correlation():
    p=poses();q=p.copy();q[:3,:, :,2]*=-1
    r=health(q,p);assert not r['passed'] and r['depth_bad_fraction']==.03


def test_depth_collapse_fails():
    p=poses();q=p.copy();q[...,2]=0
    assert not health(q,p)['passed']


def test_planar_reference_does_not_create_false_direction_signal():
    p=poses();p[:50,...,2]=0
    r=health(p,p);assert r['passed'] and r['informative_view_frames']==100
    p[...,2]=0;assert not health(p,p)['passed']


def test_streaming_decision_and_counts_match():
    p=poses(101);q=p.copy();q[:5,...,2]*=-1
    a=health(q,p);b=health(q,p,17)
    for k in ['passed','frames','informative_view_frames','depth_bad_view_frames','geometry_bad_frames','depth_bad_fraction']:
        assert a[k]==b[k]


def test_nonfinite_and_nonpositive_scale_rejected():
    p=poses();q=p.copy();q[0,0,0,0]=np.nan
    with pytest.raises(ValueError):health(q,p)
    with pytest.raises(ValueError):DepthHealth().update(p,p,np.zeros((100,2)))


def test_radius_collapse_and_scale_floor_block():
    p=poses();assert not health(p*.01,p)['passed']
    h=DepthHealth();h.update(p,p,np.full((100,2),1e-8));assert not h.result()['passed']
