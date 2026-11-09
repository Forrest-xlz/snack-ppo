# Name small/base models and align the B/C study

User requests B `transformer_v1_small` (input->32->16, two encoders), C
`transformer_v1_base` (input->64->32, three encoders), all other configs matched,
and removal of D/E active experiments.

- Rename serialized V7 to small and V6 to base. User explicitly rejected
  compatibility aliases; retired v6/v7 names must fail parsing. Shapes,
  parameter names and features stay unchanged.
- Both B/C use EntityRegions66 plus Snake21/Food24/Global8 and CUDA. Copy B's
  full training/evaluation config for C, changing only model version/identity.
- Existing checkpoints now live in exp-B (old exp-B-v7 folder is absent). Align
  B output identity/path and infer path to exp-B. Preserve user's selected
  samples-300072960.ot snapshot and inference fps60.
- C's exp-C folder is absent, so use its standard per-letter identity/path.
- Delete only active D.yaml/E.yaml; retain checkpoints and generic optional
  kickstarting support. Remove obsolete D/E config tests and README recipes.
- Test new selectors before implementation, retired-name rejection,
  B/C protocol and shape equality/differences, and actual sweep discovery.
- Full tests with CUDA required; build runtime and headless-check model loading
  against the existing B checkpoint. Do not start formal training or W&B.

## Completed

- Named-model and active-sweep tests first failed, then passed. A separate
  rejection test first failed with aliases enabled, then passed after removal.
- B/C serialized configs match except model version, run name and checkpoint
  identity. Both use all66 cell features and CUDA, with2/3 encoder layers.
- Active feature experiment folder contains only A.yaml/B.yaml/C.yaml.
- Inference retains the user's samples-300072960.ot and fps60, under actual
  exp-B storage. No checkpoint was moved or deleted by these edits.
- Full suite passed109 tests with CUDA required; release build, formatting
  and whitespace checks passed. Read-only reviewer found no concrete defects.
- English E004 README now describes current A/B/C, shared features, small/base
  architecture and launch commands. Deleted D/E recipe links were removed.
