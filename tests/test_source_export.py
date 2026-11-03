from pathlib import Path
import subprocess
import sys
import tarfile
import importlib.util
import json
import pytest


def test_export_excludes_untracked_files(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'tools/export_source.py'
    spec = importlib.util.spec_from_file_location('export_source', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'repo'
    (root / 'dual2pose').mkdir(parents=True)
    (root / 'dual2pose/config.py').write_text('# core\n')
    subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(root), 'add', 'dual2pose/config.py'], check=True)
    (root / 'dual2pose/private_notes.json').write_text('{"private": true}')
    result = module.export(root, tmp_path / 'release.tar.gz')
    assert set(result['files']) == {'dual2pose/config.py'}


def test_export_manifest_isolated_from_parent_git_and_rejects_traversal(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'tools/export_source.py'
    spec = importlib.util.spec_from_file_location('export_source', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    root = tmp_path / 'exported'
    (root / 'dual2pose').mkdir(parents=True)
    (root / 'dual2pose/config.py').write_text('# core\n')
    inventory = root / 'SOURCE_MANIFEST.json'
    inventory.write_text(json.dumps({'files': {'dual2pose/config.py': ''}}))
    assert set(module.export(root, tmp_path / 'ok.tar.gz')['files']) == {'dual2pose/config.py'}
    inventory.write_text(json.dumps({'files': {'../private.json': ''}}))
    with pytest.raises(ValueError, match='Invalid source inventory path'):
        module.export(root, tmp_path / 'bad.tar.gz')


def test_export_has_core_and_no_private_or_vendor_material(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'release.tar.gz'
    result = subprocess.run([sys.executable, str(root / 'tools/export_source.py'), '--output', str(output)], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with tarfile.open(output) as archive:
        names = {str(Path(n).relative_to('canonfuse3d')) for n in archive.getnames()}
    assert {'dual2pose/config.py', 'configs/dual2pose.yaml', 'configs/model/cross_view_fusion.yaml', 'SOURCE_MANIFEST.json'} <= names
    assert not any(Path(n).parts[0] in {'paper', 'reports', 'logs', '.git', 'SAM3Dbody', 'ckpt', '.cache', '.vscode'} for n in names)
    assert {n for n in names if n.startswith("unity/")} == {f"unity/{subject}/Assets/Scripts/OneCameraCaptureFrame.cs" for subject in ("male", "female")}
    before = output.read_bytes()
    result = subprocess.run([sys.executable, str(root / 'tools/export_source.py'), '--output', str(output)], cwd=root, capture_output=True)
    assert result.returncode != 0 and output.read_bytes() == before
