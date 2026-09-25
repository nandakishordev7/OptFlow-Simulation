"""Evaluate OptFlow on a folder of .npz samples.

python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points 2048
python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points full --max_range 35
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
from optflow import OptFlow, OptFlowConfig, compute_metrics
from optflow.metrics import average_metrics
from optflow.datasets import SceneFlowNPZ

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--config', default='configs/kitti.yaml')
p.add_argument('--n_points', default='2048')
p.add_argument('--max_range', type=float, default=None)
p.add_argument('--limit', type=int, default=None, help='evaluate only the first N samples')
p.add_argument('--out', default='results.json')
args = p.parse_args()

n = None if args.n_points == 'full' else int(args.n_points)
ds = SceneFlowNPZ(args.data, n_points=n, max_range=args.max_range)
solver = OptFlow(OptFlowConfig.from_yaml(args.config))

results, times = [], []
for i in tqdm(range(len(ds) if args.limit is None else min(args.limit, len(ds)))):
    s = ds[i]
    out = solver(s['pc1'], s['pc2'])
    m = compute_metrics(out['flow'], s['flow'], s['mask'])
    m['time'] = out['time']
    results.append(m)

avg = average_metrics(results)
print(json.dumps(avg, indent=2))
with open(args.out, 'w') as f:
    json.dump({'config': solver.cfg.to_dict(), 'average': avg, 'per_sample': results}, f, indent=2)
