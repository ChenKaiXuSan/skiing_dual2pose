"""Create a history-free, allowlisted core source archive without overwriting outputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOTS = {'dual2pose', 'configs', 'cross_validation', 'pegasus', 'tests', 'tools'}
FILES = {f'unity/{subject}/Assets/Scripts/OneCameraCaptureFrame.cs' for subject in ('male', 'female')} | {'README.md', 'LICENSE', 'THIRD_PARTY.md', 'setup.py', 'setup.cfg', 'pyproject.toml', 'MANIFEST.in', 'requirements.txt', 'environment.yaml', '.gitignore', '.github/workflows/ci-testing.yml', '.github/workflows/install-pkg.yml', 'docs/reproducibility/clean_install.md', 'docs/reproducibility/data_and_runs.md', 'docs/reproducibility/release_readiness.md'}


def export(root: Path, output: Path):
    root = root.resolve()
    if output.exists():
        raise FileExistsError(output)
    # Never collect stray local files, or discover an enclosing repository.
    commit = None
    try:
        top = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        top = None
    if top and Path(top).resolve() == root:
        names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'], text=True).split('\0')
        try:
            commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            pass  # A newly initialized source tree may have only an index.
    elif (root / 'SOURCE_MANIFEST.json').is_file():
        previous = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
        names = list(previous['files'])
        commit = previous.get('base_commit')
    else:
        raise ValueError('Source export requires a Git root or SOURCE_MANIFEST.json')
    candidates = []
    for name in names:
        if not name:
            continue
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Invalid source inventory path: {name}')
        p = root / rel
        if any((root / Path(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts) + 1)):
            continue
        if p.is_file() and not any(part in {'__pycache__', '.git', '.pytest_cache', 'build', 'dist'} for part in rel.parts):
            candidates.append(p)
    files = []
    for p in candidates:
        rel = p.relative_to(root)
        if (rel.parts[0] in ROOTS or rel.as_posix() in FILES or rel.as_posix().startswith('docs/reproducibility/final_20261103/')) and (p.suffix in {'.py', '.yaml', '.yml', '.json', '.csv', '.md', '.txt', '.sh', '.pbs', '.cs', '.toml', '.cfg', '.in'} or rel.as_posix() in FILES):
            files.append(p)
    files.sort()
    if not (root / 'dual2pose/config.py').is_file():
        raise ValueError('Not a CanonFuse3D source tree')
    manifest = {'base_commit': commit, 'scope': 'Core source only; data, weights, third-party frontends and immutable run archives are supplied separately.',
                'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle, tarfile.open(fileobj=handle, mode='w:gz') as archive:
        for p in files:
            archive.add(p, arcname='canonfuse3d/' + p.relative_to(root).as_posix(), recursive=False)
        info = tarfile.TarInfo('canonfuse3d/SOURCE_MANIFEST.json'); info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = export(Path(__file__).resolve().parents[1], args.output.resolve())
    print(json.dumps({'output': str(args.output), 'files': len(result['files'])}))
