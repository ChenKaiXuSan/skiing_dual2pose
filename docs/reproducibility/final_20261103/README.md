# Final adopted results, 2026-11-03

These CSVs and checkpoint hashes mirror the approved unified collection used by the IEEE Access manuscript. Position errors are mm; acceleration is mm per sampled-frame squared. `${RUNTIME_ROOT}` denotes the independently supplied immutable run archive, not a path required on the reader's computer. Source paths are provenance records, not executable download URLs. Numerical values and checkpoint hashes are unchanged.

Primary CanonFuse3D MPJPE: Unity 50.9071429738854 mm; Ski 124.32837168213216 mm. Main Unity and C07 share the same fusion and DLT+MLP weights. Different clean observation cohorts yield different C07 and main errors. Independent ablations and adapted models retain separate weights. Timing is on H100 with each method's scope specified in the accompanying protocol.

Base-chain recipes alone do not reproduce all final corrections: preserve/recreate the stable Unity MetaPose recipe, corrected Unity geometry/calibration and final unified reevaluation with the pinned models. See ../data_and_runs.md. No new GPU result is asserted by a source release.
