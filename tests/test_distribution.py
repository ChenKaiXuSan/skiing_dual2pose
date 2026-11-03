"""Exercise installed distributions outside the source checkout."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def test_wheel_composes_default_and_model_configs_outside_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / 'src'
    src.mkdir()
    for name in ('dual2pose', 'configs', 'cross_validation', 'pegasus'):
        shutil.copytree(root / name, src / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('setup.py', 'setup.cfg', 'MANIFEST.in', 'README.md', 'LICENSE', 'requirements.txt'):
        shutil.copy2(root / name, src / name)
    subprocess.run([sys.executable, 'setup.py', 'bdist_wheel'], cwd=src, check=True, capture_output=True)
    installed = tmp_path / 'installed'
    with zipfile.ZipFile(next((src / 'dist').glob('*.whl'))) as archive:
        archive.extractall(installed)
    probe = '''import sys
sys.path.insert(0, sys.argv[1])
from dual2pose.config import load_config
cfg = load_config()
assert cfg.model.arch.hidden_dim > 0
assert cfg.train.fold == 0
'''
    result = subprocess.run([sys.executable, '-I', '-c', probe, str(installed)], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
