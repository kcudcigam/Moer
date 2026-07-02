# Two-Level MC Branch

This branch extends `feature/nrc-framework` with the residual-correction hooks needed for a Neural Two-Level Monte Carlo implementation.

## Implemented

- `ResidualSample` stores cache prediction, path-traced target, and residual weight.
- `ResidualBuffer` stores bounded residual samples.
- `ResidualCorrector` estimates a global residual correction from stored samples.
- `TwoLevelContinuationEstimator` returns `cache_prediction + residual_correction`.

## Remaining implementation

The full paper method should add:

- residual sample collection at selected continuation queries,
- unbiased or low-bias path-traced residual targets,
- sampling probability control via `renderer.nrc.residual_probability`,
- per-region or feature-aware residual correction instead of the current global average.

The current branch is intentionally a compileable hook layer, not a final quality implementation.
