"""Visualise pc1 (red), pc2 (green), pc1+flow (blue), like Fig. 2 of the paper.
python scripts/visualize.py --file data/kitti/000000.npz      (or --synthetic)"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import open3d as o3d
from optflow import OptFlow, OptFlowConfig
from optflow.synthetic import make_scene

p = argparse.ArgumentParser()
p.add_argument('--file'); p.add_argument('--synthetic', action='store_true')
p.add_argument('--config', default='configs/kitti.yaml')
args = p.parse_args()

if args.synthetic:
    s = make_scene(); cfg = OptFlowConfig()
else:
    d = np.load(args.file); s = {'pc1': d['pc1'], 'pc2': d['pc2']}
    cfg = OptFlowConfig.from_yaml(args.config)
out = OptFlow(cfg)(s['pc1'], s['pc2'])

def cloud(x, c):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x.astype(np.float64)))
    pc.paint_uniform_color(c); return pc
o3d.visualization.draw_geometries([cloud(s['pc1'], [1, 0, 0]), cloud(s['pc2'], [0, 0.8, 0]),
                                   cloud(s['pc1'] + out['flow'], [0, 0, 1])])
