#!/usr/bin/env python3
"""Train PlueckerNet (their code and recipe) on their Semantic3D data with a random similarity scale and rotation.

    python evaluation/train_pn.py --name pn_sem3d_s1to10r90

Changes from their training (Ruey 2026-10-29), all on cloud 2 of each pair, drawn afresh every time a pair is used:
  * scale    s ~ log-uniform [smin, smax] about the origin (moments x s);
  * rotation Ra by an angle uniform in [0, max_rot_deg] (90) about a random axis, composed with the stored one
             (theirs stop at 82 deg) -- the GT similarity cloud1 -> cloud2 becomes (s, Ra R_gt, s Ra t_gt).
             Full SO(3) did not train (val 3-9% after 31 epochs vs 43% for scale alone);
  * signs    both clouds through the hemisphere canon (pn_model.canon) AFTER the rotation -- what PlueckerNet
             gets at test time, since real maps' signs are arbitrary and it has no sign handling of its own;
  * more data from the same pairs: an epoch visits each of the 1683 pairs --epoch_mult times, each visit with its
    own scale and rotation.
Their translations, model, loss, optimizer, LR schedule (x0.99 per epoch) and batch of 12 are
unchanged; their trainer (PluckerTrainer) runs as-is apart from this fork's torch-2 fixes.  Validation pairs get a
seeded scale and rotation, so the best checkpoint is chosen on augmented pairs.
Their validation runs an SE(3) RANSAC whose errors are meaningless once a scale is present, so validation
here reports only their selection metric, `avg_inlier_ratio` (GT precision of the top-min(100) scores),
computed as theirs.

Line order: [m; d] -- their data, their network and SCALAR agree (see pn_model.py).
Checkpoints: output/semantic3D/<name>/{checkpoint,best_val_checkpoint}.pth, loadable with pn_model.load(path).
"""
import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pn_model  # noqa: E402  (puts this repo's root on sys.path)

pn_model._import_pn()
from lib.dataloader import load_data_plucker_pairs  # noqa: E402  (theirs)
from trainer_plucker import PluckerTrainer  # noqa: E402  (theirs)


def rand_rot(rng, max_deg):
    """Rotation by an angle uniform in [0, max_deg] about a uniformly random axis (Rodrigues)."""
    k = (lambda v: v / np.linalg.norm(v))(rng.randn(3))
    a = np.radians(rng.uniform(0.0, max_deg))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class ScaledPairs(Dataset):
    """Their Semantic3D pkl split; cloud 2 under a random scale and SO(3) rotation (seeded per pair in valid)."""

    def __init__(self, a, phase):
        self.a, self.phase = a, phase
        self.d = load_data_plucker_pairs(argparse.Namespace(data_dir=str(pn_model.PN / "dataset"),
                                                            dataset="semantic3D"), phase)
        self.n = min(len(self.d["t_gt"]), a.limit or len(self.d["t_gt"]))
        self.mult = a.epoch_mult if phase == "train" else 1

    def __len__(self):
        return self.n * self.mult

    def __getitem__(self, j):
        d, a, i = self.d, self.a, j % self.n
        rng = np.random.RandomState(i) if self.phase == "valid" else np.random
        s = math.exp(rng.uniform(math.log(a.smin), math.log(a.smax)))
        Ra = rand_rot(rng, a.max_rot_deg) if a.max_rot_deg > 0 else np.eye(3)
        p1, p2 = d["plucker1"][i], d["plucker2"][i].astype(np.float64)
        p2 = np.c_[s * p2[:, :3] @ Ra.T, p2[:, 3:] @ Ra.T]           # x -> s Ra x (moments scale, directions do not)
        if a.canon:                                                  # hemisphere canon AFTER the rotation, both clouds:
            p1, p2 = pn_model.canon(p1), pn_model.canon(p2)          # the input it gets at test time (real signs are arbitrary)
        M = np.zeros((len(p1), len(p2)), np.float32)
        M[d["matches"][i][0], d["matches"][i][1]] = 1.0
        return (M, p1.astype(np.float32), p2.astype(np.float32), (Ra @ d["R_gt"][i]).astype(np.float32),
                (s * Ra @ d["t_gt"][i].reshape(3)).reshape(3, 1).astype(np.float32))


class Trainer(PluckerTrainer):
    def _valid_epoch(self):
        """Their avg_inlier_ratio (top-min(100) GT precision); the SE(3) RANSAC part is dropped."""
        self.model.eval()
        ratios = []
        for b, (matches, p1, p2, _, _) in enumerate(self.val_data_loader):
            if 0 < self.val_max_iter <= b:
                break
            if p1.size(1) > 3000:                                    # their own skip ("my GPU is short of memory")
                continue
            prob = self.model(p1.to(self.device), p2.to(self.device))[0]
            k = min(100, prob.size(-1) * prob.size(-2))
            top = torch.topk(prob.flatten(start_dim=-2), k=k, dim=-1).indices[0].cpu()
            i1, i2 = top // prob.size(-1), top % prob.size(-1)
            ratios.append(float(matches[0, i1, i2].sum()) / k * 100.0)
            torch.cuda.empty_cache()
        r = float(np.mean(ratios)) if ratios else 0.0
        print(f"validation: avg_inlier_ratio {r:.2f} over {len(ratios)} pairs", flush=True)
        return {"avg_inlier_ratio": r}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    # Ruey 2026-10-29 (a proof of concept): train up to 50x either way, evaluate up to 1000x
    # Ruey 2026-10-29, a proof of concept: scale 1..10, an extra rotation of at most 90 deg.
    # (Scale 1/50..50 alone trained fine -- val 43% by ep 85; adding full SO(3) + canon stuck at 3-9%.)
    ap.add_argument("--smin", type=float, default=1.0, help="similarity scale, log-uniform in [smin, smax]")
    ap.add_argument("--smax", type=float, default=10.0)
    ap.add_argument("--max_rot_deg", type=float, default=90.0,
                    help="extra rotation of cloud 2: angle uniform in [0, max] about a random axis (0 = theirs only)")
    ap.add_argument("--canon", type=int, default=1, help="hemisphere sign canon on both clouds after the rotation")
    ap.add_argument("--epoch_mult", type=int, default=1, help="visits of each pair per epoch, each freshly augmented")
    # their recipe (main_train.py / config.py); 600 epochs: their x0.99/epoch LR is 2.4e-6 by then
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu_mem_frac", type=float, default=0.0, help="cap this process's share of the GPU")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pth")
    ap.add_argument("--cpu", action="store_true", help="smoke test on CPU")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: at most this many train/val pairs")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m/%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])               # as their main_train.py
    cfg = pn_model.config(
        out_dir=str(pn_model.PN / "output"), dataset="semantic3D", model_nb=a.name, weights="", resume=None,
        train_lr=a.lr, train_weight_decay=1e-3, train_momentum=0.8, exp_gamma=0.99, train_epoches=a.epochs,
        train_start_epoch=0, train_save_freq_epoch=1, iter_size=1, val_max_iter=-1, val_epoch_freq=1,
        optimizer="Adam", train_seed=0, use_gpu=not a.cpu, gpu_inds=-1 if a.cpu else 0, print_freq=10,
        best_val_metric="avg_inlier_ratio", data_dir=str(pn_model.PN / "dataset"),
        scale_aug=dict(smin=a.smin, smax=a.smax, max_rot_deg=a.max_rot_deg, canon=a.canon, epoch_mult=a.epoch_mult))
    ck = pn_model.PN / "output" / "semantic3D" / a.name / "checkpoint.pth"
    if a.resume and ck.exists():
        cfg.resume = str(ck)
    if a.gpu_mem_frac > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)
    torch.manual_seed(0)
    np.random.seed(0)
    train = DataLoader(ScaledPairs(a, "train"), batch_size=a.batch, shuffle=True, drop_last=True,
                       num_workers=a.workers, persistent_workers=a.workers > 0,
                       worker_init_fn=lambda w: np.random.seed((torch.initial_seed() + w) % 2 ** 32))
    val = DataLoader(ScaledPairs(a, "valid"), batch_size=1, shuffle=False, num_workers=1)
    Trainer(cfg, train, val).train()


if __name__ == "__main__":
    main()
