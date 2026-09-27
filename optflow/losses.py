"""Objective terms: soft-correspondence fit (Eqs. 3-6) and rigidity (Eqs. 7-8)."""
import torch
from .geometry import knn


def soft_correspondence_loss(src, tgt, k_local, eps, thresh, use_correlation=True):
    """Mean squared distance between each src point and its correspondence in tgt.

    use_correlation=True : q_avg = weighted average of k_local neighbours (Eqs. 3-5)
    use_correlation=False: q     = single nearest neighbour (ablation (c))
    Neighbours further than `thresh` get weight 0 (Sec. 3.3); points with no neighbour
    inside the threshold are ignored.
    """
    k = k_local if use_correlation else 1
    _, idx = knn(src.detach(), tgt, k)                     # neighbour search, no grad
    q = tgt[idx]                                           # (N,k,3)
    d2 = ((src.unsqueeze(1) - q) ** 2).sum(-1)             # (N,k), differentiable
    valid = d2.detach() < thresh ** 2
    point_valid = valid.any(1)
    if point_valid.sum() == 0:
        return src.sum() * 0.0

    if use_correlation:
        sim = torch.exp(-d2)                               # Eq. 3
        logw = (sim - 1.0) / eps                           # log of Eq. 4
        # -1e9 instead of -inf: never produces NaN. Rows with no valid neighbour get
        # meaningless weights, but those points are dropped below via point_valid.
        logw = logw.masked_fill(~valid, -1e9)
        w = torch.softmax(logw, dim=1)                     # Eq. 5 normalisation, stable
        q_avg = (w.unsqueeze(-1) * q).sum(1)               # (N,3)
    else:
        q_avg = q[:, 0]

    res = ((src - q_avg) ** 2).sum(-1)                     # Eq. 6 per point
    return res[point_valid].mean()


def build_rigidity_graph(pc, k_rigid):
    """kNN graph on the source cloud (fixed during optimisation) + weights of Eq. 8."""
    d, idx = knn(pc, pc, k_rigid + 1)
    d, idx = d[:, 1:], idx[:, 1:]                          # drop self
    w = torch.exp(-d ** 2)
    return idx, w


def rigidity_loss(flow, graph_idx, graph_w):
    """Eq. 7: sum_ij W_ij ||f_i - f_j||^2  (averaged instead of summed for scale)."""
    diff = flow.unsqueeze(1) - flow[graph_idx]             # (N,K,3)
    return (graph_w * (diff ** 2).sum(-1)).mean()
