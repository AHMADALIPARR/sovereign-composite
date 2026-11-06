# Stage 3 — Big-Machine Runbook (TimesFM-3 → Meridian distillation)

The local rung on the Muse VM (4 throttled cores) proves the pipeline at
slice scale. This runbook repeats it at full scale on the owner's iron.
Nothing in the harness changes except the slice constants — same frozen
data file, same stated bound, same evaluator.

## What to bring over

- `sovereign-composite/frozen_series.npz` — SHA-256
  b7151be60bd244bfa9e4a93f4ede4dc98ca2c9f31ef9e8dbdc511a67057a34d2
  (256 train + 32 eval series, context 512, horizon 128, five families)
- `sovereign-composite/stage3_distill.py` — the harness
- The teacher stack: `Google_Black_Box/src/timesfm3/` (from the owner's fork)
- Checkpoint `google/timesfm-3.0-pytorch` (ungated; model.safetensors
  1,322,898,824 bytes + config.json) → local dir, e.g. `models/timesfm-3.0/`

## Environment

- Python 3.12, torch (CPU or CUDA — the forecaster auto-selects),
  numpy, huggingface_hub, safetensors.
- Point `CKPT_DIR` in the harness at the local checkpoint directory.

## Full-scale changes (the only edits)

In `stage3_distill.py`:

- `CTX, HOR = 512, 128` (full frozen spec)
- `N_TEACH_TRAIN = 256` (the whole training pool)
- Student patch stays 16 (32 patches at CTX 512); epochs may rise to 800
  at the operator's discretion — record the value used in the results file.

## Run

```
python3 stage3_distill.py
```

Outputs: `teacher_outputs.npz` (the teacher's forecasts — the distillation
corpus), `stage3-results.json`, and the printed gate verdict.

## The bound (unchanged, stated before measurement)

On the 32 held-out frozen series: **student truth-MAE ≤ 1.25 × teacher
truth-MAE**. The evaluator also reports student↔teacher MAE — the pure
imitation gap, independent of either model's accuracy against reality.

## Send back

`stage3-results.json` + `teacher_outputs.npz`. Both get hashed into the
composite manifest next to the local rung's numbers, so the two runs are
comparable line by line: same data, same bound, different iron.
