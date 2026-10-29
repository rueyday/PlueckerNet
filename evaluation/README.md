# PlueckerNet in the SCALAR comparison (Semantic3D)

This fork (rueyday/PlueckerNet, of Liumouliu/PlueckerNet) is checked out in SCALAR at `third_party/PlueckerNet`.
Changes to the authors' code are torch-2 / any-device fixes only (`next(it)`, integer `//` in their validation,
`torch.load(..., weights_only=False)`, the graph op's hardcoded `torch.device('cuda')` -> the input's device);
model, loss, optimizer and schedule are untouched.  This folder holds what reproduces the two PlueckerNet arms;
the evaluation (and the arm that trains OUR matcher on their data) lives in SCALAR's `tools/pluckernet/`.

| arm | weights | training data |
|---|---|---|
| 1 | their pretrained Semantic3D | — |
| 2 | `train_pn.py` (run `pn_sem3d_s1to10r90`, best val epoch 236, 42.3% inlier ratio) | their Semantic3D; cloud 2 under log-uniform scale [1, 10] and an extra rotation of 0-90 deg about a random axis, then the hemisphere canon on both clouds; their translations and recipe (Adam 1e-3, exp 0.99/epoch, batch 12) |

## Reproduce
```
evaluation/fetch_data.sh     # their dataset -> /mnt/crucial/datasets/pluckernet, Semantic3D linked into dataset/,
                             # their pretrained Semantic3D weights -> output/semantic3D/preTrained/
evaluation/launch_pn.sh      # arm 2 in a memory-capped systemd unit (resumable), 400 epochs; the
                             # shipped checkpoint is best_val (epoch 236; the exp schedule does not depend on the total)
```
Then, from the SCALAR root: `tools/pluckernet/sweep_sem3d.sh` (their protocol, scale 1x-1000x, rotation 30-180 deg,
three noise levels; arms 1, 2 and our v102 on identical inputs).

## Files
- `fetch_data.sh` — data and pretrained weights (both gitignored).
- `pn_model.py` — build / load their PluckerNetKnn on any device, plus the hemisphere `canon` and random sign `flip`.
- `train_pn.py` — arm 2 (their PluckerTrainer on scaled / rotated / canon'd pairs); `--cpu --limit N` smoke test.
- `launch_pn.sh` — arm 2 launcher.

## Two facts the comparison depends on
- **Input order is `[m; d]`, unchanged.** Their variable names say `[d; m]`, but their data and weights are `[m; d]`:
  their pretrained models give 83% (Structured3D) / 71% (Semantic3D) val precision as-is and 6% / 15% channel-swapped.
- **PlueckerNet has no sign handling.** Its pairs agree in sign by construction (cloud 2 = transformed cloud 1).
  Pretrained Semantic3D, 40 valid pairs, precision: stored signs 75.4% | half of cloud 2 flipped 23.0% | all flipped
  0.0% | hemisphere canon 41.0% (identical after any flips).  Real maps' signs are arbitrary, so every PlueckerNet
  input is flip -> canon.  The canon only agrees across two maps at small rotation (0/30/60/90/120/180 deg extra:
  41.0/30.9/15.6/2.9/1.0/2.6% through the canon), which is why arm 2 trains with at most 90 deg extra rotation --
  with full SO(3) its validation stayed at 3-9%.
