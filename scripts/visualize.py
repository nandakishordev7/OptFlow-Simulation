"""Visualise pc1 (red), pc2 (green), pc1+predicted_flow (blue), like Fig. 2 of the paper.

python scripts/visualize.py --file data/kitti/<name>.npz --config configs/kitti.yaml
python scripts/visualize.py --synthetic
python scripts/visualize.py --file data/kitti/<name>.npz --gt     # also show ground-truth flow in yellow
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d
from optflow import OptFlow, OptFlowConfig
from optflow.synthetic import make_scene
from optflow.datasets import _get   # same key-name lookup used by datasets.py (pc1/points1/pos1, ...)

p = argparse.ArgumentParser()
p.add_argument('--file'); p.add_argument('--synthetic', action='store_true')
p.add_argument('--config', default='configs/kitti.yaml')
p.add_argument('--gt', action='store_true', help='also draw pc1 + ground-truth flow, in yellow')
p.add_argument('--n_points', default='2048',
                help="subsample each cloud to this many points before solving ('full' = no "
                     "subsampling; on CPU 'full' can take 20+ min per iteration on real LiDAR "
                     "data, so only use it on a GPU)")
args = p.parse_args()

if args.synthetic:
    s = make_scene(); cfg = OptFlowConfig()
else:
    if not args.file:
        sys.exit('pass --file data/<dataset>/<name>.npz (see scripts/check_data.py to list files) or --synthetic')
    d = np.load(args.file)
    pc1, pc2, gt = _get(d, 'pc1'), _get(d, 'pc2'), _get(d, 'flow')
    if pc1 is None or pc2 is None:
        sys.exit(f"couldn't find point-cloud arrays in {args.file}; keys present: {d.files}")
    pc1, pc2, gt = pc1.astype(np.float32), pc2.astype(np.float32), None if gt is None else gt.astype(np.float32)

    if args.n_points != 'full':
        n = int(args.n_points)
        rng = np.random.default_rng(0)
        if len(pc1) > n:
            i1 = rng.choice(len(pc1), n, replace=False)
            pc1, gt = pc1[i1], (None if gt is None else gt[i1])
        if len(pc2) > n:
            pc2 = pc2[rng.choice(len(pc2), n, replace=False)]

    s = {'pc1': pc1, 'pc2': pc2, 'flow': gt}
    cfg = OptFlowConfig.from_yaml(args.config)
    print(f"solving on {len(pc1)} / {len(pc2)} points "
          f"(pass --n_points full to use every point, --n_points 8192 etc. to change this)")

out = OptFlow(cfg)(s['pc1'], s['pc2'])
print(f"solved in {out['iters']} iterations, {out['time']:.2f}s")

def cloud(x, c):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x.astype(np.float64)))
    pc.paint_uniform_color(c); return pc

geoms = [cloud(s['pc1'], [1, 0, 0]),                     # red   = pc1 (time t-1)
         cloud(s['pc2'], [0, 0.8, 0]),                   # green = pc2 (time t)
         cloud(s['pc1'] + out['flow'], [0, 0, 1])]        # blue  = pc1 + predicted flow
if args.gt and s.get('flow') is not None:
    geoms.append(cloud(s['pc1'] + s['flow'], [1, 0.85, 0]))  # yellow = pc1 + ground-truth flow
o3d.visualization.draw_geometries(geoms)