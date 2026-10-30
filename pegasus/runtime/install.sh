#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
kind=${1:?Usage: install.sh torch|metapose}
case "$kind" in torch|metapose) ;; *) exit 2;; esac
bootstrap=${DUAL2POSE_BOOTSTRAP_PYTHON:-/home/SKIING/chenkaixu/miniconda3/envs/direction/bin/python3.11}
prefix="$DUAL2POSE_RUNTIME/envs/$kind"
mkdir -p "$DUAL2POSE_RUNTIME"/{envs,runs,setup,cache}
if [[ ! -x "$prefix/bin/python" ]]; then "$bootstrap" -m venv "$prefix"; fi
py="$prefix/bin/python"
"$py" -m pip install --upgrade pip
if [[ "$kind" == torch ]]; then
    "$py" -m pip install torch==2.11.0+cu126 torchvision==0.26.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    "$py" -m pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements-main.txt"
else
    "$py" -m pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements-metapose.txt"
fi
"$py" -m pip check
"$py" -m pip freeze > "$DUAL2POSE_RUNTIME/setup/$kind.freeze.txt"
printf 'INSTALL_OK %s\n' "$kind"
