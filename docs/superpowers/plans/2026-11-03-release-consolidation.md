# Per-frame consolidation and release preparation

> Historical snapshot: branch names and preservation instructions below apply to the recorded operation. Since 2026-11-03, only `master` is retained; see `docs/branch_management.md` for the current policy.

Approved scope: repair the public-readiness findings, validate integration, and merge the current per-frame method into master. Keep the per-frame branch, immutable experiment code/results and manuscript files. Do not make the repository public or rewrite existing history.

1. Reproduce distribution-config failure and clean-checkout DAG preparation failure; package configs and version the non-data plan seeds with provenance.
2. Replace template CI/build settings and supply a portable CPU development environment, separate optional frontends and TensorFlow baselines.
3. Update current result/checkpoint documentation from the approved final collection; record legacy plans separately. Add a data/weight acquisition contract and an allowlisted source-export command that excludes manuscript/history, binary assets and third-party frontend code.
4. Run regression tests, build/install the wheel and sdist in isolation, and verify exported-source tests. Review the complete diff.
5. Preserve old master/per-frame refs, fast-forward both branches only when checks pass, push without force, then synchronize only the Pegasus development checkout. Verify manuscript and frozen-run protections.

Review focus: composed model configuration survives wheel and sdist installation; missing resources fail clearly; archived plan seeds keep precedence; bundled seeds prepare without private logs or submission; exported source contains no paper/results/Unity assets; output paths cannot overwrite existing files; clean-checkout test collection does not require private manuscripts.

Public visibility remains a separate user action. History/material redistribution review is not solved by merging branches; the clean source export is the reviewable alternative.
