# Third-party material boundaries

The root Apache-2.0 license covers this project's code, not every resource in the historical repository.

- SAM 3D Body: `SAM3Dbody/UPSTREAM.md` records the patched upstream revision and local modifications. Upstream source, gated model weights, MHR assets and detector dependencies have their own terms. The core wheel and `tools/export_source.py` deliberately exclude the vendored frontend. Obtain approved upstream resources separately; do not infer redistribution permission from the root license.
- Unity projects: character meshes, textures, animations and other assets remain in the private development history. They are excluded from the core distribution and source export. Do not make the historical repository public until asset permissions and history have been reviewed.
- STRIDE, DeciWatch and MetaPose: externally downloaded repositories and weights are not included in the core distribution. Preserve their licenses and pinned revisions when obtaining them; see `docs/reproducibility/data_and_runs.md`.
- Ski-PTZ-Pose and author-collected data: datasets are not redistributed with this code. Obtain Ski-PTZ-Pose from its provider and request the permitted derived research data and checkpoints from the corresponding author. Raw author-collected videos are not shared.

A source export is not a claim that all historical files are cleared for publication. No existing history has been removed or relicensed by merging the per-frame branch.

The 2026-11-03 tracked-file audit found no license/notice/copyright-named files under `SAM3Dbody/` or `unity/`. Provenance alone does not settle redistribution terms. See [public-readiness audit](docs/reproducibility/release_readiness.md) for the exact release boundary and outstanding work.
