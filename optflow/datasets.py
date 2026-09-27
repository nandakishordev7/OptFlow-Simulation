"""Dataset loaders.

KITTI / nuScenes / Argoverse: the preprocessed .npz files released with Graph Prior
[Pontes et al., 3DV 2020] and reused by NSFP. FlyingThings3D: FlowNet3D preprocessed .npz.
Key names differ between releases (KITTI here uses pos1, pos2, gt), so we try several.
Check a dataset with   python scripts/check_data.py data/<name>

Choosing which files to use (train/tuning vs test):
    file_list='splits/kitti_test.txt'   one file name per line (preferred, once known)
    start=100, end=150                   index range into the sorted file list
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


def select_files(root, file_list=None, start=None, end=None):
    files = sorted(glob.glob(os.path.join(root, '**', '*.npz'), recursive=True))
    if not files:
        raise FileNotFoundError(f'no .npz files under {root}')
    if file_list:
        with open(file_list) as f:
            wanted = {line.strip() for line in f if line.strip() and not line.startswith('#')}
        wanted = {w if w.endswith('.npz') else w + '.npz' for w in wanted}
        files = [p for p in files if os.path.basename(p) in wanted]
        if not files:
            raise FileNotFoundError(f'none of the files in {file_list} exist under {root}')
    return files[start:end]


class SceneFlowNPZ:
    def __init__(self, root, n_points=2048, max_range=None, seed=0,
                 file_list=None, start=None, end=None):
        self.files = select_files(root, file_list, start, end)
        self.n_points = n_points          # int, or None/'full' for the full cloud
        # max_range: Euclidean distance from the sensor. NOTE: SCOOP/Table 1 cut on depth (z)
        # on a different KITTI preprocessing, so this is NOT an exact Table 1 reproduction.
        self.max_range = max_range
        self.seed = seed

    def __len__(self):
        return len(self.files)

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

        # per-sample RNG -> the same points are sampled no matter the loading order
        rng = np.random.default_rng(self.seed + i)
        i1 = self._sample(len(pc1), rng)
        i2 = self._sample(len(pc2), rng)
        return {'pc1': pc1[i1], 'pc2': pc2[i2], 'flow': flow[i1], 'mask': mask[i1],
                'name': os.path.basename(self.files[i])}

    def _sample(self, n_total, rng):
        if self.n_points in (None, 'full') or n_total <= self.n_points:
            return np.arange(n_total)
        return rng.choice(n_total, self.n_points, replace=False)
