"""PlueckerNet (Liu et al., CVPR'21) for the SCALAR comparison: build / load PluckerNetKnn on any device.

Their stored lines and their network input are both [m; d] (moment first, unit direction last) --
the SAME order as SCALAR's.  Their model calls the first-three-channel branch "direction", but their
data puts the moment there and their RANSAC (lib/ransac_l2l.py) reads the direction from [3:], so
the name is the only thing that says otherwise.  Feed lines as-is, never channel-swapped.
"""
import sys
from pathlib import Path

import numpy as np
import torch

PN = Path(__file__).resolve().parents[1]          # this fork's root
SCALAR = PN.parents[1]                             # checked out at SCALAR/third_party/PlueckerNet
WEIGHTS = {"outdoor": PN / "output/semantic3D/preTrained/best_val_checkpoint.pth"}   # their pretrained Semantic3D


def canon(p):
    """SCALAR's hemisphere sign canon for [m; d] lines: flip each line so its direction's largest-|component| is
    positive.  PlueckerNet has no sign handling of its own and its training pairs agree in sign by construction
    (cloud 2 is a transformed cloud 1), so real maps -- whose signs come from each map's own endpoint order --
    reach it through this canon, in training (after random flips) and at test time alike."""
    p = np.asarray(p)
    a = np.abs(p[:, 3:]).argmax(1)
    s = np.sign(p[np.arange(len(p)), 3 + a])
    s[s == 0] = 1
    return p * s[:, None]


def flip(p, rng, frac=0.5):
    """Flip the sign of each line ([m; d] -> [-m; -d], the same line) with probability frac."""
    p = np.asarray(p)
    return p * np.where(rng.rand(len(p)) < frac, -1.0, 1.0)[:, None]


def _import_pn():
    if str(PN) not in sys.path:
        sys.path.insert(0, str(PN))
    for m in [m for m in sys.modules if m == "model" or m.startswith("model.")]:
        if not str(getattr(sys.modules[m], "__file__", "")).startswith(str(PN)):
            del sys.modules[m]
    import model.model_plucker as M
    return M


def config(**kw):
    """Their default network config (config.py), as an EasyDict."""
    from easydict import EasyDict
    c = EasyDict(net_lambda=0.1, net_maxiter=30, net_nchannel=128, net_depth=12, GNN_layers=["self", "cross"] * 6,
                 net_gcnorm=True, net_batchnorm=True, net_knn=10, net_act_pos="post", net_topK=2000)
    c.update(kw)
    return c


def build(device="cpu", **kw):
    return _import_pn().PluckerNetKnn(config(**kw)).to(device)


def load(path_or_name, device="cpu"):
    """'outdoor' (their pre-trained Semantic3D weights) or a checkpoint path."""
    path = WEIGHTS.get(path_or_name, path_or_name)
    net = build(device)
    sd = torch.load(path, map_location=device, weights_only=False)["state_dict"]
    net.load_state_dict(sd, strict=True)                    # raises on any missing/unexpected key
    return net.eval()
