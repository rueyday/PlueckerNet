#!/usr/bin/env python3
"""Case 3 data: their Semantic3D pairs, unscaled, in ScalePluckerNet's dataset format.

    python evaluation/make_sem3d_spn.py        # -> /mnt/crucial/datasets/pluckernet/spn/semantic3D_{train,valid}

Our encoder is scale-invariant, so case 3 trains on their data as-is: no scale signal (Ruey 2026-10-29).
--smin/--smax (default 1, 1) would add case 2's log-uniform scale on cloud 2 (validation seeded as case 2's).

Format (ScalePluckerNet lib/dataloader.py): plucker1 = REFERENCE, plucker2 = QUERY, matches (2, n) =
(reference index, query index), and the GT maps the query into the reference, m_r = s R m_q + t x R d_q.
Their cloud 1 is the reference; their cloud 2 (= T cloud 1, then x s) is the query, so with their (R, t) and
our scale s (query = s (R ref + t)):  R_gt = R^T,  t_gt = -R^T t,  s_gt = 1 / s.
"""
import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np

PN = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/mnt/crucial/datasets/pluckernet/spn")
    ap.add_argument("--smin", type=float, default=1.0)
    ap.add_argument("--smax", type=float, default=1.0)
    a = ap.parse_args()
    for split in ("train", "valid"):
        src = PN / "dataset" / f"semantic3D_{split}"
        d = {k: pickle.load(open(src / f"{k}.pkl", "rb"), encoding="latin1")
             for k in ("matches", "plucker1", "plucker2", "R_gt", "t_gt")}
        out = {k: [] for k in ("matches", "plucker1", "plucker2", "R_gt", "t_gt", "s_gt")}
        for i in range(len(d["t_gt"])):
            rng = np.random.RandomState(i if split == "valid" else 7_000_000 + i)   # valid = case 2's seeds
            s = math.exp(rng.uniform(math.log(a.smin), math.log(a.smax)))
            p2 = d["plucker2"][i].astype(np.float64)
            R, t = d["R_gt"][i].astype(np.float64), d["t_gt"][i].reshape(3).astype(np.float64)   # theirs, unscaled
            out["plucker1"].append(d["plucker1"][i].astype(np.float32))
            out["plucker2"].append(np.c_[s * p2[:, :3], p2[:, 3:]].astype(np.float32))
            out["matches"].append(d["matches"][i].astype(np.int32))
            out["R_gt"].append(R.T.astype(np.float32))
            out["t_gt"].append((-R.T @ t).reshape(3, 1).astype(np.float32))
            out["s_gt"].append(np.float32(1.0 / s))
        dst = Path(a.out) / (f"semantic3D_{split}" if a.smin == a.smax == 1 else f"semantic3D_scaled_{split}")
        os.makedirs(dst, exist_ok=True)
        for k, v in out.items():
            with open(dst / f"{k}.pkl", "wb") as f:
                pickle.dump(v, f, protocol=4)
        print(f"{split}: {len(v)} pairs -> {dst}", flush=True)


if __name__ == "__main__":
    main()
