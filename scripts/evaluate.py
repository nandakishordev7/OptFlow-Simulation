"""Evaluate OptFlow on a folder of .npz samples.

python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points 2048 --file_list splits/kitti_test.txt
python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points 2048 --start 0 --end 10
python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --alpha_rigid 100   (override one value)
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from optflow import OptFlow, OptFlowConfig, compute_metrics
from optflow.metrics import average_metrics
from optflow.datasets import SceneFlowNPZ

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--config', default='configs/kitti.yaml')
p.add_argument('--n_points', default='2048')
p.add_argument('--max_range', type=float, default=None)
p.add_argument('--file_list', default=None, help='txt file with one .npz name per line (e.g. the test split)')
p.add_argument('--start', type=int, default=None, help='first index into the sorted file list')
p.add_argument('--end', type=int, default=None, help='last index (exclusive)')
p.add_argument('--limit', type=int, default=None, help='evaluate only the first N selected samples')
p.add_argument('--out', default='results.json')
# quick overrides without editing the YAML
p.add_argument('--alpha_rigid', type=float); p.add_argument('--k_local', type=int)
p.add_argument('--lr_rot', type=float); p.add_argument('--lr_trans', type=float)
args = p.parse_args()

n = None if args.n_points == 'full' else int(args.n_points)
ds = SceneFlowNPZ(args.data, n_points=n, max_range=args.max_range,
                  file_list=args.file_list, start=args.start, end=args.end)
cfg = OptFlowConfig.from_yaml(args.config, alpha_rigid=args.alpha_rigid, k_local=args.k_local,
                              lr_rot=args.lr_rot, lr_trans=args.lr_trans)
solver = OptFlow(cfg)
print(f"evaluating {len(ds) if args.limit is None else min(args.limit, len(ds))} samples "
      f"on {solver.device}, first file {os.path.basename(ds.files[0])}")

results = []
for i in tqdm(range(len(ds) if args.limit is None else min(args.limit, len(ds)))):
    s = ds[i]
    out = solver(s['pc1'], s['pc2'])
    m = compute_metrics(out['flow'], s['flow'], s['mask'])
    m.update(name=s['name'], time=out['time'], iters=out['iters'], stopped_early=out['stopped_early'],
             t_icp=out['t_icp'].tolist(), t_final=out['t'].tolist(), R_final=out['R'].tolist())
    results.append(m)

avg = average_metrics(results)
avg['stopped_early_frac'] = sum(r['stopped_early'] for r in results) / len(results)
print(json.dumps(avg, indent=2))
worst = sorted(results, key=lambda r: -r['EPE'])[:5]
print("worst samples:", [(r['name'], round(r['EPE'], 3)) for r in worst])
with open(args.out, 'w') as f:
    json.dump({'config': cfg.to_dict(), 'files': [os.path.basename(x) for x in ds.files],
               'average': avg, 'per_sample': results}, f, indent=2)
print(f"saved {args.out}")
