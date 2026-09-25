"""Synthetic driving-like scene for testing without downloading data:
static walls/poles seen from a moving car + a few independently moving boxes."""
import numpy as np


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def _box(center, size, n, rng):
    pts = (rng.random((n, 3)) - 0.5) * size
    face = rng.integers(0, 3, n)                  # push points onto the box surface
    side = rng.choice([-0.5, 0.5], n)
    pts[np.arange(n), face] = side * np.asarray(size)[face]
    return (pts + center).astype(np.float32)


def make_scene(n_points=2048, n_objects=3, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n_static = int(n_points * 0.6)
    n_obj = (n_points - n_static) // n_objects

    # static world: two walls along the road + poles
    walls = np.concatenate([
        np.stack([rng.uniform(0, 40, n_static // 2), np.full(n_static // 2, s),
                  rng.uniform(0, 3, n_static // 2)], 1) + rng.normal(0, 0.05, (n_static // 2, 3))
        for s in (-6.0, 6.0)]).astype(np.float32)
    # poles/buildings break the along-road ambiguity of the walls
    n_poles = n_static // 3
    pole_xy = np.stack([rng.uniform(2, 40, 12), rng.choice([-5.0, 5.0], 12)], 1)
    pid = rng.integers(0, 12, n_poles)
    poles = np.concatenate([pole_xy[pid] + rng.normal(0, 0.15, (n_poles, 2)),
                            rng.uniform(0, 4, (n_poles, 1))], 1).astype(np.float32)
    static = np.concatenate([walls[:n_static - n_poles], poles])

    # ego-motion of the car between the two frames
    R_ego, t_ego = _rot_z(rng.uniform(-0.05, 0.05)), np.array([rng.uniform(0.5, 1.5), 0, 0], np.float32)

    objs1, objs2 = [], []
    for k in range(n_objects):
        c = np.array([rng.uniform(8, 35), rng.uniform(-4, 4), 0.8], np.float32)
        box = _box(c, (4.0, 1.8, 1.5), n_obj, rng)
        R_o, t_o = _rot_z(rng.uniform(-0.1, 0.1)), np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.3, 0.3), 0], np.float32)
        moved = (box - c) @ R_o.T + c + t_o      # object's own motion (world frame)
        objs1.append(box)
        objs2.append(moved)

    pc1_world = np.concatenate([static] + objs1)
    pc2_world = np.concatenate([static] + objs2)

    # express frame-2 points in the frame-2 sensor coordinates (inverse ego-motion)
    to_f2 = lambda x: (x - t_ego) @ R_ego          # R^T (x - t)
    pc2_same = to_f2(pc2_world)                     # 1:1 correspondences of pc1
    flow_gt = pc2_same - pc1_world

    # frame 2 is a *different* sampling of the scene (no 1:1 correspondence)
    pc2 = to_f2(np.concatenate(
        [np.concatenate([static[rng.permutation(len(static))]])] + objs2))
    pc2 = pc2 + rng.normal(0, 0.03, pc2.shape)      # resampling-like jitter
    pc1 = pc1_world + rng.normal(0, noise, pc1_world.shape)
    pc2 = pc2 + rng.normal(0, noise, pc2.shape)
    return {'pc1': pc1.astype(np.float32), 'pc2': pc2.astype(np.float32),
            'flow': flow_gt.astype(np.float32), 'mask': np.ones(len(pc1), bool)}
