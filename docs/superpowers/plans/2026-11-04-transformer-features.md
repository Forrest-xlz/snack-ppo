# Experiment C: Geometry and Body-Order Features

**Goal:** Add the user's requested x/y, empty-cell, and chain-distance-to-tail features to the existing Transformer as a separate comparable experiment.

**Design:** One model version `transformer_v2` selects seven channel-major planes in this order: head, body excluding head, food, x/(width-1), y/(height-1), empty, tail distance/(board_area-1). Empty means neither snake nor food. Distance counts links along the ordered snake: tail=0, preceding segment=1, head=length-1; non-snake cells=0 and occupancy planes disambiguate them. A dimension of one uses coordinate zero. This distance is not a promise about future release time when the snake grows.

**Architecture:** Same as B, with only token embedding input width changed from 3 to 7. Keep learned positions, encoder width32/depth1/heads2/FFN64, selected-query optimization, hidden_size256 readouts, and all PPO/evaluation settings. All prior feature/model versions remain unchanged. C trains from scratch and cannot load B's differently shaped embedding weights.

## Tasks

- [x] Write failing tests for feature values, alias disambiguation, dimension edge cases, rectangular-board CLI train/eval/checkpoint reload, and C/B protocol consistency.
- [x] Add seven-plane encoder and model-version dispatch; parameterize Transformer input channels without changing B checkpoint names/shapes.
- [x] Add C.yaml, document exact semantics and C launch command in the English README; existing alphabetical sweep discovers C.
- [x] Run targeted/full tests, release build and C-only launch preview; do not launch full training or online logging.

## Verification

All five new tests first failed because transformer_v2/C.yaml were unavailable, then passed after implementation. The full suite passed 72 tests, including old MLP/CNN/Transformer checkpoint tests, PPO training and evaluation using seven-plane observations on a rectangular board, exact feature values, snake-order alias disambiguation, full single-row board finite values, and C-versus-B protocol equality. Release compilation, format/diff checks and the C-only sweep preview passed. Independent read-only review found no critical, important or minor issues. No formal training, online W&B run, commit or push was performed.
