# Clean installation

Use Python 3.11 or newer; CI checks Python 3.11. The existing run-archive integrity code uses Python 3.11 hashlib.file_digest. Create a new environment; do not replace a running experiment environment.

The commands below use a Git checkout or the complete `canonfuse3d-source.tar.gz` export. The wheel and PyPI-style sdist are installation artifacts; they omit the regression suite and source-export tooling.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# CPU validation; select a matching CUDA torch/torchvision pair for GPU training.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r tests/requirements.txt
python -m pytest -q
python -m build
```

The wheel includes Hydra YAML and model groups. Validate from outside the checkout:

```bash
python -m pip install dist/*.whl
cd /tmp
python -c 'from dual2pose.config import load_config; print(load_config().model.arch)'
```

The core dependency list supports pose training/evaluation. Raw-image frontends require separately installed upstream code and weights. MetaPose runs in a separate TensorFlow environment; do not install it into the PyTorch environment. `pegasus/runtime/requirements-main.txt` and `requirements-metapose.txt` record the experiment environments, rather than promising binary portability across CUDA systems.

Set `DUAL2POSE_UNITY_ROOT`, `DUAL2POSE_SKI_ROOT`, `DUAL2POSE_POCKET3_ROOT`, and `DUAL2POSE_RAW_ROOT` before training. The fallback paths in archived configuration/index metadata record provenance; they are not data downloads. See [data and run contract](data_and_runs.md).

CPU regression tests and clean package installation do not replace GPU validation or replicate all final experiments.
