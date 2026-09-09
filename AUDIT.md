# v12.1 Hardened – Audit Summary

This release hardens the original project after a code-level review.

## Fixed
- Model path is now relative to `app.py`, not the process working directory.
- XGBoost predictions are treated as continuous auxiliary regression output instead of assumed winner labels.
- Non-finite and extreme ML predictions are contained before entering the search.
- Classical evaluation remains the dominant evaluation component.
- Men can capture in all diagonal directions while normal movement remains directional.
- Capture generation no longer silently filters to the longest chain by default.
- Added `status(white_turn)` for side-to-move terminal detection.
- Search timeout is checked on every visited node instead of every 4096 nodes.
- `analyze_all()` uses a global deadline and remaining-time allocation.
- Added `score_loss` and `mistake_level` to move analysis.
- Added a syntax smoke test.

## Important compatibility note
The exact rules of draughts/checkers variants differ. The move generator now follows the common English/American convention of mandatory capture without a longest-capture rule. If the intended variant requires maximum capture or different king movement, that should be exposed as an explicit rule setting rather than being hard-coded.
