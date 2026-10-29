# PlueckerNet comparison (Semantic3D only, three cases; set up 2026-10-29, not yet run)

This fork (rueyday/PlueckerNet, of Liumouliu/PlueckerNet) is checked out in SCALAR at
`third_party/PlueckerNet`.  Changes to the authors' code are torch-2 / any-device fixes only (`next(it)`,
integer `//` in their validation, `torch.load(..., weights_only=False)`, the graph op's hardcoded
`torch.device('cuda')` -> the input's device); model, loss, optimizer and schedule are untouched.
Everything SCALAR-specific is in `evaluation/`.  Ruey 2026-10-29: Semantic3D (outdoor) only -- indoor
is argued elsewhere.  (An earlier case C, their architecture on SCALAR's generator, was dropped: SCALAR's
generator has no scale or translation to give it.)  Case C is now OUR matcher on THEIR data.

| arm | weights | training data | train |
|---|---|---|---|
| A | their pretrained Semantic3D | — | — |
| B | trained here | their Semantic3D, cloud 2 under log-uniform scale [1, 10] + an extra rotation of 0-90 deg about a random axis, then the hemisphere canon on both clouds; their translations and recipe | `launch_pn.sh` |
| C | OUR matcher (ScalePluckerNet, v106 architecture + optimizer), trained here | their Semantic3D as-is, NO scale signal (our encoder is scale-invariant) | `make_sem3d_spn.py`, then `launch_spn_sem3d.sh` |

Recipe: B = their optimizer exactly (Adam 1e-3, exp 0.99 per epoch, batch 12) for 600 epochs (LR 2.4e-6 by then), one
freshly augmented visit per pair per epoch (~1M samples).  Run `pn_sem3d_s1to10r90`.  Superseded: `pn_sem3d_scale` (scale
only), `pn_sem3d_sim3` (full SO(3): val stuck at 3-9% -- the canon only agrees across two maps at small rotation and
PlueckerNet has no sign handling, so ~90 deg is the canon's limit, not a choice of theirs).
C = v106's optimizer settings (lr 5e-4 cosine, 32 pairs per step, match_w 0.2) for the same 400 passes over the
1683 pairs, on their data unscaled (ScalePluckerNet reads static pickles, so make_sem3d_spn.py converts it).  Our matcher
is sign-invariant by construction, so C gets their stored signs with no canon, at training and test time.

## Signs and rotation (measured 2026-10-29, their pretrained Semantic3D weights, 40 valid pairs, precision@K)
PlueckerNet has NO sign handling, and its pairs agree in sign by construction (cloud 2 = transformed cloud 1):
stored 75.4% | 50% of cloud 2 flipped 23.0% | 50% of both 14.8% | all of cloud 2 flipped 0.0% |
hemisphere canon (ours) 41.0%, identical after any flips.  Real maps' signs are arbitrary (47% of real matches
are sign-flipped), so every PlueckerNet input at TEST time is flip -> canon.
Rotation: their training pairs stop at 82 deg (median 45); extra rotation of cloud 2 gives
0/30/60/90/120/180 deg -> 75.4/68.1/49.1/13.7/2.3/0.1% (stored signs), 41.0/30.9/15.6/2.9/1.0/2.6% (canon).
Our ScanNet++ cell has median native rotation 139 deg; B trains with up to 90 deg extra (16% of pairs end past 90) and the canon applied after
the rotation -- the same input A and B get at test time.

## Evaluation

1. **Semantic3D test pairs with added noise** -- `eval_sem3d.py` (their valid split = their test set, 298 pairs,
   those with <= 3000 lines as in their main_test.py).  Noise on both clouds: direction jitter `--dir_deg`, foot-point
   jitter `--pos_frac` x cloud radius, `--outlier` fraction of random extra lines; cloud 2 scaled log-uniform in
   `[--smin, --smax]`, `--rot_deg` extra rotation of cloud 2 (the rotation sweep), `--flip` per-line sign flips
   (default 0.5; PlueckerNet then gets the canon), or a fixed magnitude `--scale_level L` (s = 10^L) for the
   scale sweep L = 0, 0.5, 1, 1.5, 1.7, 2, 2.5, 3 (1x to 1000x).  Training covers s in [1, 10] (|log10 s| <= 1, Ruey:
   a proof of concept), so L <= 1 is in-distribution and above is not.  The sweep scales UP (s = 10^L, 1x..1000x, as training);
   `--scale_both` draws up or down at random per pair.  Seeded per pair, so every model sees identical inputs.  Our checkpoints run in the same
   table with `--ckpt tag=spn:<path>`.  Metrics: their top-200 precision and the frozen solve_sim3 (med e_R,
   rot<5, rot<10, strict).
   ```
   python third_party/PlueckerNet/evaluation/eval_sem3d.py --ckpt pn_outdoor=outdoor \
       --ckpt pn_sem3d=third_party/PlueckerNet/output/semantic3D/pn_sem3d_s1to10r90/best_val_checkpoint.pth \
       --ckpt v106=spn:third_party/ScalePluckerNet/output/synthetic_fsp5q8k/v106_od2/best_val_checkpoint.pth \
       --ckpt spn_sem3d=spn:$HOME/spn-cleanup/output/semantic3D/spn_sem3d/best_val_checkpoint.pth \
       --dir_deg 2 --pos_frac 0.01 --outlier 0.3 --smin 0.1 --smax 10 --out evaluation/pluckernet/sem3d_noise.json
   ```
2. **Our real cells** -- `eval_pn.py` writes a `solver_sweep.py` cache (`--add-to` a v106 cache to score every
   model on identical pairs), then `solver_sweep.py sweep` with the frozen solver.

**Input order: `[m; d]`, unchanged** (`pn_model.py`). Their variable names say `[d; m]`, but their data and
weights are `[m; d]`: their pretrained models give 83% (Structured3D) / 71% (Semantic3D) val precision as-is and
6% / 15% channel-swapped.  `tools/cache_baseline_corr.py` and `tools/kitti_rotated_table.py` swapped until
2026-10-29 (now fixed), so every earlier "PlueckerNet-SE3" baseline row was computed on scrambled input.
PlueckerNet gets raw metric lines (as it was trained); `eval_pn.py --norm qstd` feeds what our matcher sees.

## Files
- `fetch_data.sh` — their dataset archive -> /mnt/crucial/datasets/pluckernet, Semantic3D splits linked into
  `dataset/`, and their pretrained Semantic3D weights into `output/semantic3D/preTrained/` (both gitignored).
  The archive also holds Structured3D, unused.
- `pn_model.py` — load/build their PluckerNetKnn on any device.
- `train_pn.py` — case B (their PluckerTrainer); `--cpu --limit N` smoke test.
- `make_sem3d_spn.py` + `launch_spn_sem3d.sh` — case C: their data in ScalePluckerNet's format
  (/mnt/crucial/datasets/pluckernet/spn, built and GT-convention-checked), trained with ~/spn-cleanup/train.py.
- `launch_pn.sh` — case B in a memory-capped systemd unit, GPU share 0.70, resumable.
- `eval_sem3d.py`, `eval_pn.py` — the two evaluations above.

Smoke-tested on CPU 2026-10-29: B trains and validates; eval_sem3d and eval_pn run (their weights, v102).
