#!/usr/bin/env bash
# Train case 2 (their Semantic3D data + random scale (1..10) and rotation (<= 90 deg extra), their recipe) in a memory-capped unit.
#   evaluation/launch_pn.sh            -> unit pn_sem3d_s1to10r90, log SCALAR/evaluation/pluckernet/pn_sem3d_s1to10r90.log
# Re-running resumes from its checkpoint.pth.  Do not run while another training/eval holds the GPU.
set -euo pipefail
PN=$(cd "$(dirname "$0")/.." && pwd)
R=$(cd "$PN/../.." && pwd)                       # SCALAR root
NAME=${NAME:-pn_sem3d_s1to10r90}
mkdir -p "$R/evaluation/pluckernet"
systemd-run --user --collect --unit="$NAME" -p MemoryMax="${TRAIN_MEM:-24G}" -p OOMPolicy=continue \
  -E NUMPY_MADVISE_HUGEPAGE=0 -E OMP_NUM_THREADS=2 -E PYTHONUNBUFFERED=1 \
  bash -c "cd $PN && $HOME/miniconda3/envs/torch5090/bin/python evaluation/train_pn.py --name $NAME --resume --epochs ${EPOCHS:-400} \
           --gpu_mem_frac ${GPU_FRAC:-0.70} >> $R/evaluation/pluckernet/$NAME.log 2>&1"
echo "started unit $NAME -> $R/evaluation/pluckernet/$NAME.log"
