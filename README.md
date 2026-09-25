# OptFlow — re-implementation

Unofficial PyTorch re-implementation of *OptFlow: Fast Optimization-based Scene Flow
Estimation without Supervision* (Ahuja, Baker, Schwarting, WACV 2024, arXiv:2401.02550).
No training: every pair of point clouds is solved by optimising a flow field `F` and an
ego-motion transform `T` with AdamW.

## Structure

```
optflow/
├── configs/                 hyperparameters per dataset (YAML)
│   ├── kitti.yaml
│   ├── flyingthings.yaml
│   └── nuscenes.yaml        (also use for Argoverse)
├── optflow/                 the method
│   ├── config.py            every hyperparameter + ablation switches
│   ├── geometry.py          kNN, Rodrigues rotation, ICP (Arun et al.)
│   ├── losses.py            soft correspondence (Eq. 3-6), rigidity (Eq. 7-8)
│   ├── solver.py            OptFlow optimisation loop (Eq. 9/10)
│   ├── metrics.py           EPE, Acc5, Acc10, outliers, angle error
│   ├── datasets.py          loader for preprocessed .npz scene-flow data
│   └── synthetic.py         fake driving scene for testing without data
├── scripts/
│   ├── demo_synthetic.py    smoke test, no data needed
│   ├── evaluate.py          full benchmark evaluation -> results.json
│   ├── ablation.py          Tables 3 & 4
│   └── visualize.py         Fig. 2 style view (open3d)
├── data/                    put datasets here (not included)
└── requirements.txt
```

## Paper → code map

| Paper | Code |
|---|---|
| Eq. 2, ego-motion `T` (Sec 3.1), ICP init | `solver.py` (`rot`, `trans`), `geometry.icp` |
| Eq. 3–5, local correlation weights (Sec 3.2) | `losses.soft_correspondence_loss` |
| Eq. 6, fit term + bidirectional use | `solver.py` (`bidirectional`) |
| Adaptive threshold (Sec 3.3) | `OptFlow.threshold_at` |
| Eq. 7–8, rigidity (Sec 3.4) | `losses.build_rigidity_graph`, `losses.rigidity_loss` |
| Eq. 9/10, final objective | `solver.py` loop |
| Metrics (Sec 4) | `metrics.compute_metrics` |

## Setup

```bash
pip install -r requirements.txt
python scripts/demo_synthetic.py                 # should print Acc5 well above the zero-flow baseline
python scripts/ablation.py --synthetic --limit 3
```

## Data

Use the preprocessed data from the Graph Prior / Neural Scene Flow Prior authors
(KITTI, nuScenes, Argoverse) and FlowNet3D (FlyingThings3D); see the NSFP GitHub repo
for download links. Put each dataset in `data/<name>/` as `.npz` files. Check the keys:

```bash
python -c "import numpy as np; print(np.load('data/kitti/<file>.npz').files)"
```
If they differ from `pc1/pc2/flow` (or `points1/points2/flow`), add them to `KEYS` in `datasets.py`.

## Running the benchmarks

```bash
# Table 2 setting: 2048 points, no range limit
python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points 2048
# Table 1 setting: full cloud, 35 m range
python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points full --max_range 35
# Tables 3/4
python scripts/ablation.py --data data/kitti --n_points 2048 --limit 50
```
Run on a GPU (Colab T4 matches the paper). On CPU a 2048-point sample takes ~15-40 s.

## Things the paper does not specify (our choices)

- `k_local`, `alpha_rigid`: not given → tune on KITTI's 100 train samples.
- Flow initialised to zeros ("empty tensors" in the paper).
- Early stopping: patience 50, only after the threshold reaches its 0.2 m floor.
- ICP is run once to initialise `T`; `T` is then refined by gradient descent.
- Loss terms are averaged rather than summed (only changes the scale of `alpha_rigid`).
- `--max_range` uses Euclidean distance from the sensor.
- Reported flow = ego flow (`Tp - p`) + residual flow `f`, because GT flow includes ego-motion.

## Known behaviour

With AdamW at lr 4e-3 each step moves a point by roughly 4 mm, while the correspondence
threshold halves every 100 steps. Very large residual motions (> ~0.5-0.8 m after ego
compensation) may not be reached before the threshold shrinks. If you see this, try a larger
`lr` or `thresh_interval`, and report it as an observation.
