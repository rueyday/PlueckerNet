#!/usr/bin/env python3
"""Matchers on PlueckerNet's Semantic3D test pairs (their valid split, 298 pairs), with added noise and scale.

    python third_party/PlueckerNet/evaluation/eval_sem3d.py --ckpt pn_outdoor=outdoor \
        --ckpt pn_sem3d_scale=third_party/PlueckerNet/output/semantic3D/pn_sem3d_scale/best_val_checkpoint.pth \
        --ckpt spn_sem3d=spn:$HOME/spn-cleanup/output/semantic3D/spn_sem3d/best_val_checkpoint.pth \
        --ckpt v106=spn:third_party/ScalePluckerNet/output/synthetic_fsp5q8k/v106_od2/best_val_checkpoint.pth \
        --dir_deg 2 --pos_frac 0.01 --outlier 0.3 --smin 0.1 --smax 10 --out evaluation/pluckernet/sem3d_noise.json

Per pair (seeded by pair index, so every model sees identical inputs):
  1. noise on BOTH clouds: each line's direction rotated by N(0, dir_deg) degrees, its foot point (closest
     point to the origin) moved by N(0, pos_frac x cloud radius) per axis -- then m = f x d again;
  2. outliers: round(outlier x n) random lines appended to each cloud (foot uniform in the cloud's box,
     direction uniform), unmatched;
  3. scale: cloud 2 scaled about the origin by s (their R_gt, t_gt kept, so the GT similarity
     cloud1 -> cloud2 is (s, R_gt, s t_gt)).  Either s ~ log-uniform [smin, smax], or with --scale_level L
     a FIXED magnitude s = 10^L, scaled up as case B trains (--scale_both: up or down at random) -- the scale sweep:
         for L in 0 0.5 1 1.5 1.7 2 2.5 3; do eval_sem3d.py ... --scale_level $L --out .../scale_L$L.json; done
     i.e. 1x to 1000x; case B trains on s in [1, 10].  Real cells reach 21x (|log10 s| <= 1.33).
  4. rotation: --rot_deg rotates cloud 2 by that angle about a random axis (GT updated) -- the rotation sweep;
  5. signs: each line flipped with probability --flip (default 0.5: real maps' signs are arbitrary);
     PlueckerNet then gets both clouds through the hemisphere canon (pn_model.canon), ours gets them as-is.
Noise defaults are zero / s = 1; --flip 0 with them is their own test protocol.  Lines are [m; d]; PlueckerNet gets them raw (as it
was trained), a `spn:` checkpoint gets our matcher_input normalization (as our pipeline does).

Metrics: their top-200 precision against the GT matches (main_test.py, k = min(200, n1 n2)), and the frozen
solve_sim3 (shipped defaults) on the scores: median e_R, rot<5, rot<10, strict (rot<5 and scale err<10%),
median-rotation seed of --seeds.  Pairs with > 3000 lines in cloud 1 are skipped, as in their test.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R = HERE.parents[2]                                # SCALAR root (this fork lives at SCALAR/third_party/PlueckerNet)
sys.path.insert(0, str(R / "tools"))
import eval_matchers_real as E  # noqa: E402  (our lib first: solve_sim3, matcher_input)

sys.path.insert(0, str(HERE))
import pn_model  # noqa: E402

SEM = pn_model.PN / "dataset" / "semantic3D_valid"


def load_split():
    d = {k: pickle.load(open(SEM / f"{k}.pkl", "rb"), encoding="latin1")
         for k in ("matches", "plucker1", "plucker2", "R_gt", "t_gt")}
    return [dict(i=i, m=d["matches"][i].astype(np.int64), p1=d["plucker1"][i].astype(np.float64),
                 p2=d["plucker2"][i].astype(np.float64), R=d["R_gt"][i].astype(np.float64),
                 t=d["t_gt"][i].reshape(3).astype(np.float64)) for i in range(len(d["t_gt"]))]


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def perturb(p, rng, dir_deg, pos_frac, outlier):
    m, d = p[:, :3], _unit(p[:, 3:])
    f = np.cross(d, m)                                             # foot point: d x (x x d)
    rad = np.median(np.linalg.norm(f - np.median(f, 0), axis=1)) + 1e-9
    if dir_deg > 0:                                                # rotate d by a small random angle
        ax = _unit(np.cross(d, rng.randn(*d.shape)))
        a = np.radians(rng.randn(len(d), 1) * dir_deg)
        d = _unit(d * np.cos(a) + np.cross(ax, d) * np.sin(a))
    if pos_frac > 0:
        f = f + rng.randn(*f.shape) * pos_frac * rad
    n_out = int(round(outlier * len(p)))
    if n_out:
        lo, hi = f.min(0), f.max(0)
        f = np.r_[f, lo + rng.rand(n_out, 3) * (hi - lo)]
        d = np.r_[d, _unit(rng.randn(n_out, 3))]
    return np.concatenate([np.cross(f, d), d], 1)                  # m = f x d (f is a point on the line)


def make_inputs(pair, a):
    rng = np.random.RandomState(1000 + pair["i"])
    p1 = perturb(pair["p1"], rng, a.dir_deg, a.pos_frac, a.outlier)
    p2 = perturb(pair["p2"], rng, a.dir_deg, a.pos_frac, a.outlier)
    if a.scale_level is not None:
        s = 10.0 ** (a.scale_level * (rng.choice([-1.0, 1.0]) if a.scale_both else 1.0))
    else:
        s = math.exp(rng.uniform(math.log(a.smin), math.log(a.smax)))
    p2[:, :3] *= s                                                 # x -> s x  scales moments only
    if a.rot_deg:                                                  # extra rotation of cloud 2, random axis
        k = _unit(rng.randn(3))
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rx = np.eye(3) + np.sin(np.radians(a.rot_deg)) * K + (1 - np.cos(np.radians(a.rot_deg))) * K @ K
        p2 = np.c_[p2[:, :3] @ Rx.T, p2[:, 3:] @ Rx.T]
        pair = dict(pair, R=Rx @ pair["R"])
    if a.flip > 0:                                                 # real maps: per-line signs are arbitrary
        p1, p2 = pn_model.flip(p1, rng, a.flip), pn_model.flip(p2, rng, a.flip)
    return p1, p2, s, pair


def load_matcher(spec, dev):
    if spec.startswith("spn:"):
        from eval_registration_gt import load_network
        net = load_network(spec[4:], dev)
        return lambda p1, p2: _fw(net, *E.matcher_input(p1, p2)[::-1], dev).T          # ours: model(ref, query)
    net = pn_model.load(spec, dev)
    return lambda p1, p2: _fw(net, pn_model.canon(p1), pn_model.canon(p2), dev)      # theirs: canon, model(p1, p2)


def _fw(net, a, b, dev):
    a, b = (torch.from_numpy(np.asarray(x, np.float32))[None].to(dev) for x in (a, b))
    with torch.no_grad():
        o = net(a, b)
    return (o[0] if isinstance(o, (tuple, list)) else o)[0].float().cpu().numpy()


def score(P, pair, p1, p2, s, seeds):
    """P: (n1, n2) scores.  Their top-200 precision, then solve_sim3 with cloud 2 as the reference."""
    gt = np.zeros(P.shape, bool)
    gt[pair["m"][0], pair["m"][1]] = True                          # outliers were appended, GT indices unchanged
    k = min(200, P.size)
    top = np.argpartition(P.ravel(), -k)[-k:]
    prec = 100.0 * gt.ravel()[top].mean()
    es = []
    for sd in range(seeds):
        try:
            Rm, t, sm = E.solve_sim3(p1, p2, P.T, seed=sd)        # prob rows = reference (cloud 2)
            es.append((E.rot_err(np.asarray(Rm), pair["R"]), abs(float(sm) / s - 1) * 100))
        except Exception:
            es.append((180.0, 1e9))
    A = np.array(es)
    eR, es_ = A[np.argsort(A[:, 0], kind="stable")[len(A) // 2]]
    return prec, eR, es_


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", required=True, help="tag=outdoor|<PN ckpt>|spn:<our ckpt>")
    ap.add_argument("--dir_deg", type=float, default=0.0)
    ap.add_argument("--pos_frac", type=float, default=0.0)
    ap.add_argument("--outlier", type=float, default=0.0)
    ap.add_argument("--smin", type=float, default=1.0)
    ap.add_argument("--smax", type=float, default=1.0)
    ap.add_argument("--scale_level", type=float, default=None, help="fixed log10 s: cloud 2 scaled UP by 10^L (as case B trains)")
    ap.add_argument("--scale_both", action="store_true", help="--scale_level: up or down at random per pair")
    ap.add_argument("--rot_deg", type=float, default=0.0, help="extra rotation of cloud 2 about a random axis")
    ap.add_argument("--flip", type=float, default=0.5, help="per-line sign-flip probability (0 = their stored signs)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_lines", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(R / "evaluation/pluckernet/sem3d.json"))
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    pairs = [p for p in load_split() if 2 <= len(p["p1"]) <= a.max_lines][:a.limit or None]
    scale_desc = (f"10^{'+-' if a.scale_both else ''}{a.scale_level}" if a.scale_level is not None
                  else f"[{a.smin}, {a.smax}]")
    inputs = [make_inputs(p, a) for p in pairs]
    pairs = [x[3] for x in inputs]                                 # GT rotation includes any --rot_deg
    print(f"[{time.strftime('%T')}] {len(pairs)} Semantic3D pairs, noise dir {a.dir_deg} deg / pos {a.pos_frac} / "
          f"outliers {a.outlier} / flip {a.flip} / rot +{a.rot_deg} deg / scale "
          f"{scale_desc}, device {dev}", flush=True)
    res = dict(when=time.strftime("%F %T"), args=vars(a), pairs=[p["i"] for p in pairs], models={})
    for spec in a.ckpt:
        tag, path = spec.split("=", 1)
        f = load_matcher(path, dev)
        rows = [score(f(p1, p2), pr, p1, p2, s, a.seeds) for pr, (p1, p2, s, _) in zip(pairs, inputs)]
        A = np.array(rows)
        summ = dict(prec200=float(A[:, 0].mean()), med_eR=float(np.median(A[:, 1])), rot5=int((A[:, 1] < 5).sum()),
                    rot10=int((A[:, 1] < 10).sum()), strict=int(((A[:, 1] < 5) & (A[:, 2] < 10)).sum()), n=len(A))
        res["models"][tag] = dict(summary=summ, per_pair=A.tolist())
        print(f"[{time.strftime('%T')}] {tag:16s} P@200 {summ['prec200']:5.1f}%  med eR {summ['med_eR']:6.2f}  "
              f"rot<5 {summ['rot5']}/{len(A)}  rot<10 {summ['rot10']}  strict {summ['strict']}", flush=True)
        del f
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
