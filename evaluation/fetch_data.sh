#!/usr/bin/env bash
# Download the authors' line datasets (one archive: Structured3D + Semantic3D, the Google Drive link in
# the top-level README) to /mnt/crucial and link the Semantic3D splits into dataset/, where lib/dataloader.py
# looks, plus their pre-trained Semantic3D weights into output/semantic3D/preTrained/.  Only Semantic3D is used.
set -euo pipefail
PN=$(cd "$(dirname "$0")/.." && pwd)
D=${PN_DATA:-/mnt/crucial/datasets/pluckernet}
mkdir -p "$D"; cd "$D"
[ -s s3d.zip ] || ${GDOWN:-$HOME/miniconda3/bin/gdown} 1bVI0Ny4Ly1M4cBxbgRIjgHr8DtIXZLbb -O s3d.zip
[ -f .extracted ] || { python -c "import zipfile; zipfile.ZipFile('s3d.zip').extractall('.')"; touch .extracted; }
# their pre-trained weights (README "download" link), unless already in place
W="$PN/output/semantic3D/preTrained/best_val_checkpoint.pth"
if [ ! -s "$W" ]; then
  [ -s weights.zip ] || ${GDOWN:-$HOME/miniconda3/bin/gdown} 1XhjGkwlSK9jV4ZGLd4Ucg327mHpr_0r9 -O weights.zip
  python -c "import zipfile; zipfile.ZipFile('weights.zip').extractall('weights')"
  mkdir -p "$(dirname "$W")"; cp "$(find weights -path '*semantic3D/preTrained/best_val_checkpoint.pth' | head -1)" "$W"
fi
find "$D" -path "*semantic3D_*" -name matches.pkl -printf '%h\n' | sort -u | while read -r d; do
  ln -sfn "$d" "$PN/dataset/$(basename "$d")"; echo "linked $(basename "$d")"
done
