"""All hyperparameters in one place. Values marked (paper) come from the OptFlow paper;
values marked (tune) are NOT given in the paper and must be tuned per dataset."""
from dataclasses import dataclass, asdict
import yaml


@dataclass
class OptFlowConfig:
    # optimisation (paper, Sec. 4 "Implementation Details")
    lr: float = 4e-3
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
    alpha_rigid: float = 1.0           # (tune)

    # ego-motion (Sec. 3.1)
    use_ego_motion: bool = True
    icp_iters: int = 30
    icp_max_dist: float = 1.0          # (tune) correspondence rejection for the ICP initialisation

    # ablation switches (Tables 3 & 4)
    use_correlation: bool = True       # False -> plain 1-nearest-neighbour correspondence
    use_adaptive_thresh: bool = True   # False -> fixed threshold = thresh_init
    bidirectional: bool = True

    verbose: bool = False

    @classmethod
    def from_yaml(cls, path, **overrides):
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**d)

    def to_dict(self):
        return asdict(self)
