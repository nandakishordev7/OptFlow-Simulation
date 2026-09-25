"""OptFlow: per-sample runtime optimisation of the flow field F and ego-motion T."""
import time
import torch
from .config import OptFlowConfig
from .geometry import icp, rotvec_to_matrix, matrix_to_rotvec
from .losses import soft_correspondence_loss, build_rigidity_graph, rigidity_loss


class OptFlow:
    def __init__(self, cfg: OptFlowConfig = None, device=None):
        self.cfg = cfg or OptFlowConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def threshold_at(self, it):
        c = self.cfg
        if not c.use_adaptive_thresh:
            return c.thresh_init
        return max(c.thresh_init * 0.5 ** (it // c.thresh_interval), c.thresh_min)

    def __call__(self, pc1, pc2):
        """pc1 (N1,3), pc2 (N2,3) numpy or torch. Returns dict with total flow (N1,3)."""
        c = self.cfg
        t0 = time.time()
        pc1 = torch.as_tensor(pc1, dtype=torch.float32, device=self.device)
        pc2 = torch.as_tensor(pc2, dtype=torch.float32, device=self.device)

        # --- ego-motion T initialised from ICP (Sec. 3.1 + implementation details)
        if c.use_ego_motion:
            R0, t0_ = icp(pc1, pc2, c.icp_iters, c.icp_max_dist)
            rot = matrix_to_rotvec(R0).clone().requires_grad_(True)
            trans = t0_.clone().requires_grad_(True)
            params_T = [rot, trans]
        else:
            rot = torch.zeros(3, device=self.device)
            trans = torch.zeros(3, device=self.device)
            params_T = []

        # --- flow field F, initialised to zero
        flow = torch.zeros_like(pc1, requires_grad=True)
        opt = torch.optim.AdamW([flow] + params_T, lr=c.lr, weight_decay=0.0)

        graph_idx, graph_w = build_rigidity_graph(pc1, c.k_rigid)

        best_loss, best_state, bad = float('inf'), None, 0
        history = []
        for it in range(c.iters):
            thresh = self.threshold_at(it)
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
                # only stop once the threshold has reached its floor
                if bad >= c.early_stop_patience and thresh <= c.thresh_min + 1e-9:
                    break
            if c.verbose and it % 50 == 0:
                print(f"it {it:4d}  loss {lv:.5f}  fit {loss_fit.item():.5f}  "
                      f"rigid {loss_rigid.item():.5f}  thresh {thresh:.2f}")

        f, r, t = best_state
        with torch.no_grad():
            R = rotvec_to_matrix(r) if c.use_ego_motion else torch.eye(3, device=self.device)
            ego_flow = pc1 @ R.T + t - pc1                        # flow caused by the vehicle
            total_flow = ego_flow + f                             # what the ground truth measures
        if self.device == 'cuda':
            torch.cuda.synchronize()
        return {
            'flow': total_flow.cpu().numpy(),        # compare this with GT scene flow
            'residual_flow': f.cpu().numpy(),        # non-rigid part -> dynamic objects
            'R': R.cpu().numpy(), 't': t.cpu().numpy(),
            'iters': it + 1, 'time': time.time() - t0, 'loss_history': history,
        }
