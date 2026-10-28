# Common-13 evaluation and alignment

The user requested unified evaluation joints and primary metric alignment.

## Contract

- All active Unity, Ski-PosePTZ and Pocket3 primary metrics use common-13 in the existing shared anatomical order.
- Inputs and GT are canonicalized per frame. Score model outputs directly against that GT, without recentering, rotating or rescaling the output.
- Validation checkpoint selection uses the same primary MPJPE. Keep PA and output-recanonicalized errors explicitly auxiliary.
- Retain native model input/output skeletons and the existing training loss. Native units and GT availability remain dataset-specific.
- Preserve prior edits, checkpoints and research results. Archived first-frame analyses retain their explicit historical protocol.

## Work

- [x] Add regression tests for eye exclusion, 15/13-joint equivalence, output translation sensitivity, selection and report consistency.
- [x] Add shared common-13 scoring helpers and protocol metadata; apply to trainer validation/test, per-action summaries and active N-view/temporal metrics.
- [x] Use direct common-13 MPJPE for Pocket3 selection, summaries, figures and CLI; retain separately named auxiliary aligned scores.
- [x] Add protocol metadata to reports and new checkpoints; document the evaluation contract.
- [x] Run focused tests and the full suite, request read-only review and resolve material findings.

## Verification

- Full suite: 430 passed, 3 skipped, 24 subtests passed; four expected smoke-test/upstream warnings. Log: `/tmp/common13-full.log`.
- Independent review confirmed the primary numerical and checkpoint-selection paths; its 13 focused regression tests passed.
- Resolved review findings by explicitly tagging current extension artifacts and preserving/checking protocols in multiseed aggregation. Image/pose occlusion comparisons also reject untagged or mismatched protocols. Legacy-only aggregates retain a null protocol.
- Regression evidence: eye-only errors do not change the primary metric; a 100 mm output translation remains approximately 100 mm in primary MPJPE and is only removed in the separately reported recanonicalized auxiliary metric.
- No actual dataset retraining/rescoring was performed. Historical outputs are unchanged. Ski validation splitting and Pocket3 training reliability masking remain separate, previously disclosed issues.
