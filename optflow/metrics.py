"""Standard scene-flow metrics (same definitions as FlowNet3D / NSFP / the paper)."""
import numpy as np


def compute_metrics(pred, gt, mask=None):
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    err = np.linalg.norm(pred - gt, axis=1)
    gt_norm = np.linalg.norm(gt, axis=1)
    rel = err / (gt_norm + 1e-6)

    epe = err.mean()
    acc_strict = ((err < 0.05) | (rel < 0.05)).mean() * 100
    acc_relax = ((err < 0.10) | (rel < 0.10)).mean() * 100
    # the paper's Table 1 caption: outlier = EPE >= 0.3 m. Report this one against the paper.
    outliers = (err >= 0.30).mean() * 100
    # the common FlowNet3D/NSFP convention also counts relative error > 10%. Blows up on
    # scenes where the car barely moves (tiny GT flow), so it is reported separately.
    outliers_rel = ((err > 0.30) | (rel > 0.10)).mean() * 100

    u_pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    u_gt = gt / (gt_norm[:, None] + 1e-8)
    angle = np.arccos(np.clip((u_pred * u_gt).sum(1), -1, 1)).mean()

    return {'EPE': float(epe), 'Acc5': float(acc_strict), 'Acc10': float(acc_relax),
            'Outliers': float(outliers), 'Outliers_rel': float(outliers_rel),
            'AngleErr': float(angle)}


def average_metrics(list_of_dicts):
    keys = [k for k, v in list_of_dicts[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}
