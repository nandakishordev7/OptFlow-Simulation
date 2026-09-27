"""Find out WHY accuracy is below the paper. Runs on a few tuning files and prints one table.

python scripts/diagnose.py --data data/kitti --config configs/kitti.yaml --start 0 --end 10

Steps
 1. one verbose run: compare the size of `fit` vs `alpha*rigid` (is rigidity too weak?)
 2. baselines: zero flow, ICP only (single 1 m stage, like the old code), ICP only (coarse-to-fine)
 3. alpha_rigid sweep
 4. separate learning rates for rotation / translation
 5. per-sample ICP translation (KITTI at 10 Hz: expect ~0.5-2 m forward)
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from optflow import OptFlow, OptFlowConfig, compute_metrics
from optflow.metrics import average_metrics
from optflow.datasets import SceneFlowNPZ
from optflow.geometry import icp

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--config', default='configs/kitti.yaml')
p.add_argument('--n_points', type=int, default=2048)
p.add_argument('--file_list'); p.add_argument('--start', type=int, default=0); p.add_argument('--end', type=int, default=10)
p.add_argument('--alphas', default='1,10,100,1000')
args = p.parse_args()

ds = SceneFlowNPZ(args.data, n_points=args.n_points, file_list=args.file_list, start=args.start, end=args.end)
samples = [ds[i] for i in range(len(ds))]
base = OptFlowConfig.from_yaml(args.config)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{len(samples)} samples on {dev}: {[s['name'] for s in samples]}\n")

rows = []
def row(name, preds, extra=None):
    ms = [compute_metrics(pr, s['flow'], s['mask']) for pr, s in zip(preds, samples)]
    a = average_metrics(ms)
    if extra: a.update(extra)
    rows.append((name, a))
    print(f"  done: {name}")

# ---- 1. verbose single run
print("== 1. loss balance on", samples[0]['name'], "(fit vs alpha*rigid) ==")
OptFlow(OptFlowConfig(**{**base.to_dict(), 'verbose': True}))(samples[0]['pc1'], samples[0]['pc2'])
print("If alpha*rigid is orders of magnitude below fit, rigidity does almost nothing.\n")

# ---- 2. baselines
print("== 2-4. running variants ==")
row('zero flow', [np.zeros_like(s['flow']) for s in samples])
icp_rows = []
for label, dists in [('ICP only, 1 m (old)', (1.0,)), ('ICP only, 3/1/0.5 m', (3.0, 1.0, 0.5))]:
    preds = []
    for s in samples:
        a = torch.as_tensor(s['pc1'], device=dev); b = torch.as_tensor(s['pc2'], device=dev)
        R, t = icp(a, b, base.icp_iters, dists)
        preds.append((a @ R.T + t - a).cpu().numpy())
        if len(dists) > 1:
            icp_rows.append((s['name'], t.cpu().numpy()))
    row(label, preds)

def run(name, **ov):
    solver = OptFlow(OptFlowConfig(**{**base.to_dict(), **ov}))
    outs = [solver(s['pc1'], s['pc2']) for s in samples]
    row(name, [o['flow'] for o in outs], {'iters': float(np.mean([o['iters'] for o in outs])),
                                         'time': float(np.mean([o['time'] for o in outs]))})

run('OptFlow (config as is)')
for a in [float(x) for x in args.alphas.split(',')]:
    run(f'alpha_rigid = {a:g}', alpha_rigid=a)
run('lr_rot 1e-4, lr_trans 1e-3', lr_rot=1e-4, lr_trans=1e-3)

# ---- table
print(f"\n{'variant':30s} {'EPE':>7s} {'Acc5':>7s} {'Acc10':>7s} {'Out':>7s} {'iters':>6s} {'time':>6s}")
for name, a in rows:
    it = f"{a['iters']:6.0f}" if 'iters' in a else f"{'-':>6s}"
    tm = f"{a['time']:6.2f}" if 'time' in a else f"{'-':>6s}"
    print(f"{name:30s} {a['EPE']:7.3f} {a['Acc5']:7.2f} {a['Acc10']:7.2f} {a['Outliers']:7.2f} {it} {tm}")

print("\n== 5. ICP translation per sample (coarse-to-fine) ==")
for name, t in icp_rows:
    print(f"  {name:14s} t = [{t[0]:6.2f} {t[1]:6.2f} {t[2]:6.2f}]  |t| = {np.linalg.norm(t):.2f} m")
