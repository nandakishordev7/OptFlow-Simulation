"""OptFlow: per-sample runtime optimisation of the flow field F and ego-motion T."""
import time
import torch
from .config import OptFlowConfig
from .geometry import icp, rotvec_to_matrix, matrix_to_rotvec
from .losses import soft_correspondence_loss, build_rigidity_graph, rigidity_loss


class OptFlow:
    def __init__(self, cfg: OptFlowConfig = None, device=None):
        self.cfg = cfg or OptFlowConfig()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    def threshold_at(self, it):
        c = self.cfg
        if not c.use_adaptive_thresh:
            return c.thresh_init
        return max(c.thresh_init * 0.5 ** (it // c.thresh_interval), c.thresh_min)

    def is_final_stage(self, thresh):
        """Early stopping is only allowed once the threshold can no longer change."""
        return (not self.cfg.use_adaptive_thresh) or thresh <= self.cfg.thresh_min + 1e-9

    def __call__(self, pc1, pc2):
        """pc1 (N1,3), pc2 (N2,3) numpy or torch. Returns dict with total flow (N1,3)."""
        c = self.cfg
        t_start = time.time()
        pc1 = torch.as_tensor(pc1, dtype=torch.float32, device=self.device)
        pc2 = torch.as_tensor(pc2, dtype=torch.float32, device=self.device)

        # --- ego-motion T initialised from coarse-to-fine ICP (Sec. 3.1 + implementation details)
        if c.use_ego_motion:
            R_icp, t_icp = icp(pc1, pc2, c.icp_iters, c.icp_max_dists)
            rot = matrix_to_rotvec(R_icp).clone().requires_grad_(True)
            trans = t_icp.clone().requires_grad_(True)
        else:
            R_icp = torch.eye(3, device=self.device)
            t_icp = torch.zeros(3, device=self.device)
            rot = torch.zeros(3, device=self.device)
            trans = torch.zeros(3, device=self.device)

        # --- flow field F, initialised to zero
        flow = torch.zeros_like(pc1, requires_grad=True)

        # parameter groups: T can get its own learning rates. Adam's step is ~lr per
        # parameter, so rot at lr 4e-3 rad/step moves a point 40 m away by ~16 cm/step.
        groups = [{'params': [flow], 'lr': c.lr}]
        if c.use_ego_motion:
            groups.append({'params': [rot], 'lr': c.lr_rot if c.lr_rot is not None else c.lr})
            groups.append({'params': [trans], 'lr': c.lr_trans if c.lr_trans is not None else c.lr})
        opt = torch.optim.AdamW(groups, lr=c.lr, weight_decay=0.0)   # no decay: it would pull F to 0

        graph_idx, graph_w = build_rigidity_graph(pc1, c.k_rigid)

        best_loss, best_state, bad, stage_thresh = float('inf'), None, 0, None
        stopped_early = False
        history = []
        for it in range(c.iters):
            thresh = self.threshold_at(it)
            # losses from different threshold stages are not comparable (different
            # valid-point sets), so restart the best-state tracking at every stage.
            if thresh != stage_thresh:
                stage_thresh, best_loss, bad = thresh, float('inf'), 0

            R = rotvec_to_matrix(rot) if c.use_ego_motion else torch.eye(3, device=self.device)
            warped = pc1 @ R.T + trans + flow                     # T p_i + f_i

            loss_fit = soft_correspondence_loss(warped, pc2, c.k_local, c.eps, thresh, c.use_correlation)
            if c.bidirectional:                                  # Chamfer-style backward term
                loss_fit = loss_fit + soft_correspondence_loss(pc2, warped, c.k_local, c.eps, thresh,
                                                               c.use_correlation)
            loss_rigid = rigidity_loss(flow, graph_idx, graph_w)
            loss = loss_fit + c.alpha_rigid * loss_rigid         # Eq. 9/10

            opt.zero_grad()
            loss.backward()
            opt.step()

            lv = loss.item()
            history.append(lv)
            if lv < best_loss - c.early_stop_tol:
                best_loss, bad = lv, 0
                best_state = (flow.detach().clone(), rot.detach().clone(), trans.detach().clone())
            else:
                bad += 1
                if bad >= c.early_stop_patience and self.is_final_stage(thresh):
                    stopped_early = True
                    break
            if c.verbose and it % 50 == 0:
                print(f"it {it:4d}  loss {lv:.6f}  fit {loss_fit.item():.6f}  "
                      f"alpha*rigid {c.alpha_rigid * loss_rigid.item():.6f}  thresh {thresh:.2f}")

        f, r, t = best_state
        with torch.no_grad():
            R = rotvec_to_matrix(r) if c.use_ego_motion else torch.eye(3, device=self.device)
            ego_flow = pc1 @ R.T + t - pc1                        # flow caused by the vehicle
            total_flow = ego_flow + f                             # what the ground truth measures
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return {
            'flow': total_flow.cpu().numpy(),        # compare this with GT scene flow
            'residual_flow': f.cpu().numpy(),        # non-rigid part -> dynamic objects
            'R': R.cpu().numpy(), 't': t.cpu().numpy(),
            'R_icp': R_icp.cpu().numpy(), 't_icp': t_icp.cpu().numpy(),
            'iters': it + 1, 'stopped_early': stopped_early,
            'time': time.time() - t_start, 'loss_history': history,
        }
