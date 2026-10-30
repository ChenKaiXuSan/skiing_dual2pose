#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$DUAL2POSE_RUNTIME/setup"
submission_dir=$(mktemp -d "$DUAL2POSE_RUNTIME/setup/smoke_submit_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
qsub -A HP260146 -q gen_S \
  -o "$submission_dir/stdout.log" -e "$submission_dir/stderr.log" \
  "$DUAL2POSE_REPO/pegasus/runtime/smoke.pbs" | tee "$submission_dir/submission.txt"
printf 'Submission log directory: %s\n' "$submission_dir"
