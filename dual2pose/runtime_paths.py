"""Opt-in relocation of archived input references without rewriting metadata.

Use only for reads. Output paths must be explicitly selected by the caller.
"""
import json
import os
from pathlib import Path


def input_path(value):
    path = Path(value)
    mapping = json.loads(os.environ.get('DUAL2POSE_INPUT_PATH_MAP', '{}'))
    if not isinstance(mapping, dict):
        raise ValueError('DUAL2POSE_INPUT_PATH_MAP must be a JSON object')
    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        source, target = Path(old), Path(new)
        if not source.is_absolute() or not target.is_absolute():
            raise ValueError('Input relocation roots must be absolute')
        try:
            suffix = path.relative_to(source)
        except ValueError:
            continue
        if '..' in suffix.parts:
            raise ValueError('Input relocation may not escape its mapped root')
        return target / suffix
    return path
