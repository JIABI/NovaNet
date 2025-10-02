
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDP(nn.Module):
    """
    Differentiable 'soft' dynamic programming for next-step choice.
    If given E with shape [B, K] -> returns q_next = softmax(-E / tau).
    If given E with shape [B, H, K] and switch_cost kappa -> performs
    a soft value-iteration for H steps with log-sum-exp (temperature tau).
    """
    def __init__(self, horizon: int = 1, switch_cost: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.horizon = int(horizon)
        self.kappa = float(switch_cost)
        self.tau = float(max(1e-6, temperature))

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        E: [B,K] (single step) or [B,H,K] (multi-step energies, lower is better).
        Returns q_next: [B,K] as a probability distribution for next-step choice.
        """
        if E.dim() == 2:
            return F.softmax(-E / self.tau, dim=1)
        elif E.dim() == 3:
            B, H, K = E.shape
            V = torch.zeros(B, K, device=E.device, dtype=E.dtype)
            for t in reversed(range(H)):
                V_next = V  # [B,K]
                # stay vs switch utilities
                stay = V_next                         # [B,K]
                switch = V_next - self.kappa          # [B,K]
                # logsumexp over j: log( exp(stay_i) + sum_{j!=i} exp(switch_j) )
                # approximate using logaddexp of stay vs sum of switch
                lse_switch = torch.logsumexp(switch, dim=1, keepdim=True)   # [B,1]
                # broadcast to per-i
                lse = torch.logsumexp(torch.stack([lse_switch.expand(-1, K), stay], dim=2), dim=2)  # [B,K]
                V = -E[:, t, :] + lse
            return F.softmax(V / self.tau, dim=1)
        else:
            raise ValueError("E must be [B,K] or [B,H,K]")
