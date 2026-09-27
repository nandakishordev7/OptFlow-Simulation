"""Reproduce Tables 3/4: switch off one component at a time.

python scripts/ablation.py --data data/kitti --n_points 2048 --file_list splits/kitti_test.txt
python scripts/ablation.py --synthetic                             (no data needed)

Timing note: variant (b) has a fixed threshold, so it may early-stop from the start and
its time is not directly comparable to the adaptive variants.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optflow import OptFlow, OptFlowConfig, compute_metrics
from optflow.metrics import average_metrics
from optflow.datasets import SceneFlowNPZ
from optflow.synthetic import make_scene

p = argparse.ArgumentParser()
p.add_argument('--data')
p.add_argument('--synthetic', action='store_true')
p.add_argument('--config', default='configs/kitti.yaml')
p.add_argument('--n_points', type=int, default=2048)
p.add_argument('--file_list'); p.add_argument('--start', type=int); p.add_argument('--end', type=int)
p.add_argument('--limit', type=int, default=10)
args = p.parse_args()

if args.synthetic:
    samples = [make_scene(args.n_points, seed=s) for s in range(args.limit)]
    base = OptFlowConfig()
else:
    ds = SceneFlowNPZ(args.data, n_points=args.n_points, file_list=args.file_list,
                      start=args.start, end=args.end)
    samples = [ds[i] for i in range(min(args.limit, len(ds)))]
    base = OptFlowConfig.from_yaml(args.config)

variants = {
    '(a) w/o ego-motion T': dict(use_ego_motion=False),
    '(b) w/o adaptive thresh (2m)': dict(use_adaptive_thresh=False),
    '(c) w/o correlation matrix': dict(use_correlation=False),
    'full method': dict(),
}
print(f"{len(samples)} samples")
print(f"{'experiment':32s} {'EPE':>8s} {'Acc5':>8s} {'Acc10':>8s} {'Out':>7s} {'iters':>6s} {'time':>7s}")
for name, ov in variants.items():
    cfg = OptFlowConfig(**{**base.to_dict(), **ov})
    solver = OptFlow(cfg)
    ms = []
    for s in samples:
        out = solver(s['pc1'], s['pc2'])
        m = compute_metrics(out['flow'], s['flow'], s['mask']); m['time'] = out['time']; m['iters'] = out['iters']
        ms.append(m)
    a = average_metrics(ms)
    print(f"{name:32s} {a['EPE']:8.4f} {a['Acc5']:8.2f} {a['Acc10']:8.2f} {a['Outliers']:7.2f} "
          f"{a['iters']:6.0f} {a['time']:6.2f}s")
