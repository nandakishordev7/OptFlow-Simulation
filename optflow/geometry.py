"""Geometry helpers: kNN, rotation parametrisation, ICP (for initialising T)."""
import torch


@torch.no_grad()
def knn(query, ref, k, chunk=4096):
    """k nearest neighbours of each query point in ref.
    query (N,3), ref (M,3) -> dist (N,k), idx (N,k). Chunked to keep memory bounded."""
    k = min(k, ref.shape[0])
    dists, idxs = [], []
    for s in range(0, query.shape[0], chunk):
        d = torch.cdist(query[s:s + chunk], ref)          # (c, M)
        dd, ii = d.topk(k, dim=1, largest=False)
        dists.append(dd)
        idxs.append(ii)
    return torch.cat(dists), torch.cat(idxs)


def rotvec_to_matrix(r):
    """Rodrigues formula, differentiable. r (3,) axis-angle -> R (3,3)."""
    theta = torch.sqrt((r * r).sum() + 1e-12)
    k = r / theta
    K = torch.zeros(3, 3, dtype=r.dtype, device=r.device)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    I = torch.eye(3, dtype=r.dtype, device=r.device)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def matrix_to_rotvec(R):
    cos = ((R.trace() - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos)
    if theta.abs() < 1e-6:
        return torch.zeros(3, dtype=R.dtype, device=R.device)
    w = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * theta / (2 * torch.sin(theta))


def best_fit_transform(A, B):
    """Least-squares rigid transform A->B (Arun et al. 1987, the paper's ref [2])."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.det(R) < 0:                  # reflection fix
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


@torch.no_grad()
def icp(src, tgt, iters=30, max_dist=1.0):
    """Point-to-point ICP src->tgt. Returns R (3,3), t (3,)."""
    R = torch.eye(3, dtype=src.dtype, device=src.device)
    t = torch.zeros(3, dtype=src.dtype, device=src.device)
    for _ in range(iters):
        moved = src @ R.T + t
        d, i = knn(moved, tgt, 1)
        d, i = d[:, 0], i[:, 0]
        keep = d < max_dist
        if keep.sum() < 10:
            break
        dR, dt = best_fit_transform(moved[keep], tgt[i[keep]])
        R, t = dR @ R, dR @ t + dt
        if (dR - torch.eye(3, device=src.device)).abs().max() < 1e-6 and dt.norm() < 1e-6:
            break
    return R, t
