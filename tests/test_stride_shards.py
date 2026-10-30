import pytest


def test_ranges_cover_each_window_once():
    from pegasus.runtime.stride_shards import ranges,check_ranges
    spans=ranges(64440,2048)
    assert len(spans)==32
    check_ranges(64440,spans)
    assert spans[0][0]==0 and spans[-1][1]==64440
    with pytest.raises(ValueError):check_ranges(64440,spans[:-1])
    with pytest.raises(ValueError):check_ranges(64440,spans+[spans[0]])
    with pytest.raises(ValueError):check_ranges(64440,[(0,2),(1,64440)])


def test_invalid_shard_budget_rejected():
    from pegasus.runtime.stride_shards import ranges
    with pytest.raises(ValueError):ranges(0,2048)
    with pytest.raises(ValueError):ranges(5,0)
