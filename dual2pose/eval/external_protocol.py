"""Current external-method provenance and optional independently scored control."""
import json
from pathlib import Path

import numpy as np

from dual2pose.eval.main_baseline_data import require_current_protocol, sha256
from dual2pose.eval.pose_metrics import EVALUATION_PROTOCOL


def protocol_metadata():
    return dict(canonical_mode='per_frame', evaluation_protocol=dict(EVALUATION_PROTOCOL),
                protocol_version='per_frame_common13_v1',
                canonicalizer_sha256=sha256(Path(__file__).parents[1]/'trainer/canonicalize.py'),
                acceleration_coordinate_frame='per-frame body coordinates')


def check_current_control(path, *, dataset, cache_manifest_sha256, measured=None):
    """Old archived numbers are never a reference for current scores.

    With no reference file the runner still reports its newly computed common13
    control. An optional reference must be from the same raw cache and current
    implementation; protocol labels alone are insufficient provenance.
    """
    if path is None:
        return dict(control_reference=None)
    record = json.loads(Path(path).read_text())
    require_current_protocol(record)
    if (record.get('dataset') != dataset or
            record.get('cache_manifest_sha256') != cache_manifest_sha256 or
            record.get('canonicalizer_sha256') != protocol_metadata()['canonicalizer_sha256']):
        raise ValueError('Current control provenance mismatch; regenerate from the same raw cache')
    reference = record['average_control']
    delta = None
    if measured is not None:
        for key in ('sample_count', 'point_count', 'pa_frame_count', 'acceleration_point_count'):
            if measured[key] != reference[key]:
                raise ValueError(f'Current control coverage mismatch: {key}')
        delta = {k: measured[k]-reference[k] for k in ('mpjpe', 'pa_mpjpe', 'acceleration_error')}
        if any(not np.isfinite(v) or abs(v) > 1e-6 for v in delta.values()):
            raise ValueError(f'Current common13 average-control mismatch: {delta}')
    return dict(control_reference=str(Path(path).resolve()),
                control_metrics_sha256=sha256(path), current_average_control_delta=delta)
