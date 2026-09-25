"""Dataset loaders.

KITTI / nuScenes / Argoverse: the preprocessed .npz files released with Graph Prior
[Pontes et al., 3DV 2020] and reused by NSFP (keys usually pc1, pc2, flow).
FlyingThings3D: FlowNet3D preprocessed .npz (keys points1, points2, flow, valid_mask1).
Key names differ between releases, so we try several. Check one file with
    python -c "import numpy as np; print(np.load('file.npz').files)"
"""
import glob
import os
import numpy as np

KEYS = {
    'pc1': ['pc1', 'points1', 'pos1'],
    'pc2': ['pc2', 'points2', 'pos2'],
    'flow': ['flow', 'gt', 'sf'],
    'mask': ['mask', 'valid_mask1', 'mask1'],
}


def _get(d, name):
    for k in KEYS[name]:
        if k in d.files:
            return d[k]
    return None


class SceneFlowNPZ:
    def __init__(self, root, n_points=2048, max_range=None, seed=0):
        self.files = sorted(glob.glob(os.path.join(root, '**', '*.npz'), recursive=True))
        if not self.files:
            raise FileNotFoundError(f'no .npz files under {root}')
        self.n_points = n_points          # int, or None/'full' for the full cloud
        self.max_range = max_range        # e.g. 35 (Table 1); Euclidean distance from sensor
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def _sample(self, n_total):
        if self.n_points in (None, 'full') or n_total <= self.n_points:
            return np.arange(n_total)
        return self.rng.choice(n_total, self.n_points, replace=False)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        pc1 = _get(d, 'pc1').astype(np.float32)
        pc2 = _get(d, 'pc2').astype(np.float32)
        flow = _get(d, 'flow').astype(np.float32)
        mask = _get(d, 'mask')
        mask = np.ones(len(pc1), bool) if mask is None else mask.astype(bool).reshape(-1)

        if self.max_range is not None:
            k1 = np.linalg.norm(pc1, axis=1) < self.max_range
            k2 = np.linalg.norm(pc2, axis=1) < self.max_range
            pc1, flow, mask, pc2 = pc1[k1], flow[k1], mask[k1], pc2[k2]

        i1, i2 = self._sample(len(pc1)), self._sample(len(pc2))
        return {'pc1': pc1[i1], 'pc2': pc2[i2], 'flow': flow[i1], 'mask': mask[i1],
                'name': os.path.basename(self.files[i])}
