"""All hyperparameters in one place. Values marked (paper) come from the OptFlow paper;
values marked (tune) are NOT given in the paper and must be tuned per dataset."""
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple
import yaml


@dataclass
class OptFlowConfig:
    # optimisation (paper, Sec. 4 "Implementation Details")
    lr: float = 4e-3                   # (paper) learning rate of the flow field F
    lr_rot: Optional[float] = None     # (ours) separate lr for the rotation of T; None = use lr (paper)
    lr_trans: Optional[float] = None   # (ours) separate lr for the translation of T; None = use lr (paper)
    iters: int = 600
    early_stop_patience: int = 50      # (tune) paper says "early stopping on loss", no value given
    early_stop_tol: float = 1e-5       # (tune)

    # local correlation weight matrix (Sec. 3.2)
    k_local: int = 8                   # (tune) larger for sparse data, smaller for dense
    eps: float = 0.03                  # (paper)

    # adaptive distance threshold (Sec. 3.3)
    thresh_init: float = 2.0           # (paper) start value, metres
    thresh_min: float = 0.2            # (paper) lower limit
    thresh_interval: int = 100         # (paper) halve every N iterations

    # rigidity constraint (Sec. 3.4)
    k_rigid: int = 50                  # (paper)
    alpha_rigid: float = 1.0           # (tune) sweep this first, see scripts/diagnose.py

    # ego-motion (Sec. 3.1)
    use_ego_motion: bool = True
    icp_iters: int = 20                # iterations per ICP stage
    icp_max_dists: Tuple[float, ...] = (3.0, 1.0, 0.5)   # (ours) coarse-to-fine rejection distances

    # ablation switches (Tables 3 & 4)
    use_correlation: bool = True       # False -> plain 1-nearest-neighbour correspondence
    use_adaptive_thresh: bool = True   # False -> fixed threshold = thresh_init
    bidirectional: bool = True

    verbose: bool = False

    def __post_init__(self):
        # YAML gives lists; also accept a single number (old configs used icp_max_dist: 1.0)
        if isinstance(self.icp_max_dists, (int, float)):
            self.icp_max_dists = (float(self.icp_max_dists),)
        self.icp_max_dists = tuple(float(x) for x in self.icp_max_dists)

    @classmethod
    def from_yaml(cls, path, **overrides):
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        if 'icp_max_dist' in d:                       # backwards compatibility
            d['icp_max_dists'] = d.pop('icp_max_dist')
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**d)

    def to_dict(self):
        d = asdict(self)
        d['icp_max_dists'] = list(self.icp_max_dists)
        return d
