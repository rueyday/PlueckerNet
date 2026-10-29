#!/usr/bin/env bash
# Case 3: OUR matcher (ScalePluckerNet, v106's architecture and optimizer) trained on their Semantic3D pairs
# as-is, no scale signal (our encoder is scale-invariant).  Build the data first: evaluation/make_sem3d_spn.py.
# Same number of passes over the data as case 2 (400 epochs over the 1683 pairs); v106's optimizer settings.
#   evaluation/launch_spn_sem3d.sh    -> ~/spn-cleanup/output/semantic3D/spn_sem3d, log SCALAR/evaluation/pluckernet/spn_sem3d.log
# Resumable.  Do not run while another training/eval holds the GPU.
set -euo pipefail
PN=$(cd "$(dirname "$0")/.." && pwd)
R=$(cd "$PN/../.." && pwd)                       # SCALAR root
SPN=${SPN:-$HOME/spn-cleanup}                    # the ScalePluckerNet tree v106 trains from
DATA=${DATA:-/mnt/crucial/datasets/pluckernet/spn}
NAME=${NAME:-spn_sem3d}
[ -f "$DATA/semantic3D_valid/s_gt.pkl" ] || { echo "run evaluation/make_sem3d_spn.py first"; exit 1; }
CK=$SPN/output/semantic3D/$NAME/checkpoint.pth
RESUME=""; [ -f "$CK" ] && RESUME="--resume $CK"
mkdir -p "$R/evaluation/pluckernet"
systemd-run --user --collect --unit="$NAME" -p MemoryMax="${TRAIN_MEM:-20G}" -p OOMPolicy=continue \
  -E NUMPY_MADVISE_HUGEPAGE=0 -E OMP_NUM_THREADS=2 -E PYTHONUNBUFFERED=1 -E WANDB=1 -E WANDB_PROJECT=scalepluckernet \
  -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -E MAX_LINES=8000 \
  bash -c "cd $SPN && $HOME/miniconda3/envs/torch5090/bin/python train.py --dataset semantic3D --data_dir $DATA \
           --name $NAME $RESUME --lr 5e-4 --cosine_lr --epochs 400 --batch 1 --iter_size 32 --workers 2 --bucket_batch \
           --match_w 0.2 --batch_budget 16384 --max_bucket 32 --mem_budget 8400000 --nchannel 128 \
           --gpu_mem_frac ${GPU_FRAC:-0.70} >> $R/evaluation/pluckernet/$NAME.log 2>&1"
echo "started unit $NAME -> $R/evaluation/pluckernet/$NAME.log"
