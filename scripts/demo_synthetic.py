"""Sanity check on a synthetic scene: python scripts/demo_synthetic.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from optflow import OptFlow, OptFlowConfig, compute_metrics
from optflow.synthetic import make_scene

scene = make_scene(n_points=2048, seed=1)
cfg = OptFlowConfig(verbose=True)
out = OptFlow(cfg)(scene['pc1'], scene['pc2'])

print(f"\nstopped after {out['iters']} iterations, {out['time']:.2f}s")
print("zero-flow baseline:", {k: round(v, 4) for k, v in compute_metrics(np.zeros_like(scene['flow']), scene['flow']).items()})
print("OptFlow           :", {k: round(v, 4) for k, v in compute_metrics(out['flow'], scene['flow']).items()})
