# SAM3Dbody provenance

`SAM3Dbody/` is **not** a pristine copy of an upstream release, so it cannot be
replaced by a git submodule as it stands. This file records what it actually
contains, verified on 2026-09-25.

## Contents

| Part | Files | Origin |
|---|---|---|
| `sam_3d_body/`, `tools/` | 52 | [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body), vendored and then patched |
| `__init__.py`, `infer.py`, `load.py`, `save.py`, `vis.py`, `main_unity.py`, `main_ski_pose_ptz.py` | 7 | this project (University of Tsukuba) |
| `patches/` | 7 | the local modifications below, as unified diffs |

Model weights are not vendored; they live in `ckpt/sam-3d-body-dinov3/`
(`facebook/sam-3d-body-dinov3`, SAM License, gated download).

## Upstream base

The vendored files match upstream as of **2025-11-19 … 2025-11-25**, i.e. the
range `4e33ae4` (Initial commit) .. `3e6d20f` (“fixed bugs to create
webdatasets”); every commit in that range gives the same result. The copy
predates the December 2025 upstream changes (sam3 detector/segmentor options,
webdataset fixes, MHR params in output).

Measured against `3e6d20f`: 45 of 52 files are byte-identical, and 50 of 52 are
identical once the vendoring import rewrite is ignored.

## Local modifications

Reproduce with `diff -u` against `3e6d20f`; the diffs are stored in `patches/`.

**Vendoring import rewrite** (6 files, no behaviour change). Upstream imports the
package as `sam_3d_body.*`; vendored under `SAM3Dbody/` it is imported as
`SAM3Dbody.sam_3d_body.*` or relatively:

- `sam_3d_body/data/transforms/common.py`
- `sam_3d_body/models/decoders/prompt_encoder.py`
- `sam_3d_body/models/heads/camera_head.py`
- `sam_3d_body/models/meta_arch/sam3d_body.py`
- `sam_3d_body/models/modules/camera_embed.py`
- `sam_3d_body/sam_3d_body_estimator.py` (also functional, below)

**Functional changes**

- `sam_3d_body/sam_3d_body_estimator.py`
  - `recursive_to(batch, "cuda")` → `recursive_to(batch, self.device)`. Upstream
    pins the batch to the default CUDA device; the IVC experiments run on two
    GPUs and need the estimator's own device.
  - adds `"mhr_model_params"` to the per-instance output. Upstream added the same
    field later, in `c259bfc` (2025-12-19).
- `tools/vis_utils.py` — visualization substantially rewritten (95 changed lines
  beyond the import rewrite).

## Why this is not a submodule

A submodule tracking upstream would revert all of the above: the imports would no
longer resolve, the estimator would go back to a hardcoded `"cuda"` device and
lose the `mhr_model_params` output, and `tools/vis_utils.py` would be replaced.

It would also invalidate the manuscript's evidence package, which pins SHA-256
hashes of these exact files by path — `SAM3Dbody/sam_3d_body/sam_3d_body_estimator.py`
and `SAM3Dbody/sam_3d_body/models/meta_arch/sam3d_body.py` among them (see
`paper/ivc_draft_20260821/evidence/c07_matched_occlusion_20260917/frontend/contract.json`).
Both are files this project modified.

Converting later would mean publishing the patched tree as a fork and pointing the
submodule at that fork, not at `facebookresearch/sam-3d-body`, and re-pinning the
evidence hashes afterwards.
