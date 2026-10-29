#!/usr/bin/env python3
"""PlueckerNet matcher forward on the real cells, written as a solver_sweep cache.

    # all three PlueckerNet arms, next to v106 on the same pairs (CELLS / MERGED_CAP / PAIRS_TAG as in
    # tools/fullrange_sp/solver_sweep.py forward; default = every cell at the benchmark protocol)
    python third_party/PlueckerNet/evaluation/eval_pn.py --cache evaluation/pluckernet/cache_pn.pkl \
        --ckpt pn_outdoor=outdoor \
        --ckpt pn_sem3d_scale=third_party/PlueckerNet/output/semantic3D/pn_sem3d_scale/best_val_checkpoint.pth \
    # then score with the frozen solver, exactly like our own checkpoints
    python tools/fullrange_sp/solver_sweep.py sweep --cache evaluation/pluckernet/cache_pn.pkl --seeds 11 \
        --procs 12 --out evaluation/pluckernet/pn_s11.json --configs topk=300,nsamp=3000,proj_thresh=1.984

`--add-to <solver_sweep cache>` appends the PlueckerNet tags to an existing cache (e.g. v106's), so one
sweep scores every model on identical pairs; the pair list must match.  Input: the prepared [m; d] lines
as-is -- the order their network was trained on (pn_model.py) and no normalization, as in their training.
`--norm qstd` instead feeds what our matcher sees (both clouds' moments / the query moment std).
Both clouds go through the hemisphere canon (pn_model.canon): real per-line signs are arbitrary and
PlueckerNet has no sign handling of its own.
"""
import argparse
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
import eval_matchers_real as E  # noqa: E402  (pairs, protocol, matcher_input -- imports our lib first)

sys.path.insert(0, str(HERE))
import pn_model  # noqa: E402


def pairs_like_solver_sweep():
    pairs = E._load_pairs_cap4000()
    if os.environ.get("CELLS"):
        pairs = [p for p in pairs if any(k in p["cell"] for k in os.environ["CELLS"].split(","))]
    pairs = E._attach(pairs, E.PROTOCOL)
    keep = ("cell", "pid", "Pq", "Pr", "s", "Rg", "t", "t500")
    return [{k: p[k] for k in keep} for p in pairs]


def forward(net, pairs, dev, norm, k=1000):
    out = []
    for p in pairs:
        pq, pr = E.lines_of(p)
        if norm == "qstd":
            pq, pr = E.matcher_input(pq, pr)
        pq, pr = pn_model.canon(pq), pn_model.canon(pr)          # real signs are arbitrary: their input via the canon
        pq, pr = (torch.from_numpy(np.asarray(x, np.float32))[None].to(dev) for x in (pq, pr))
        with torch.no_grad():
            prob = net(pr, pq)[0][0].float().cpu().numpy()        # rows = reference, cols = query, as ours
        kk = min(k, prob.size)
        flat = np.argpartition(prob.ravel(), -kk)[-kk:]
        flat = flat[np.argsort(-prob.ravel()[flat])]
        r, q = np.unravel_index(flat, prob.shape)
        out.append((q, r, prob.ravel()[flat].astype(np.float32), prob.shape))
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", required=True, help="tag=outdoor|<checkpoint path>")
    ap.add_argument("--cache", help="write a new solver_sweep cache here")
    ap.add_argument("--add-to", dest="add_to", help="append the tags to this existing solver_sweep cache")
    ap.add_argument("--norm", choices=["none", "qstd"], default="none")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    assert bool(a.cache) != bool(a.add_to), "give exactly one of --cache / --add-to"
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    pairs = pairs_like_solver_sweep()
    if a.add_to:
        d = pickle.load(open(a.add_to, "rb"))
        assert [(p["cell"], p["pid"]) for p in d["pairs"]] == [(p["cell"], p["pid"]) for p in pairs], \
            "pair list differs from the cache's (check CELLS / MERGED_CAP / PAIRS_TAG)"
    else:
        d = dict(pairs=pairs, corr={})
    print(f"[{time.strftime('%T')}] {len(pairs)} pairs, device {dev}, input {a.norm}", flush=True)
    for spec in a.ckpt:
        tag, path = spec.split("=", 1)
        net = pn_model.load(path, dev)
        d["corr"][tag] = forward(net, d["pairs"], dev, a.norm)
        print(f"[{time.strftime('%T')}] {tag} forward done", flush=True)
        del net
    out = a.add_to or a.cache
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pickle.dump(d, open(out, "wb"), protocol=4)
    print(f"-> {out}: tags {list(d['corr'])}", flush=True)


if __name__ == "__main__":
    main()
