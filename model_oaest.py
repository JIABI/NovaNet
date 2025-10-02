import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

CFG = importlib.import_module('config')


def _get(name, default):
    return getattr(CFG, name, default)


E_ALPHA_SNR = _get('E_ALPHA_SNR', 1.0)
E_BETA_TTL = _get('E_BETA_TTL', 1.0)
E_GAMMA_SWITCH = _get('E_GAMMA_SWITCH', 2.5)
E_KAPPA_UNCERT = _get('E_KAPPA_UNCERT', 1.0)
ENERGY_USE_LCB = _get('ENERGY_USE_LCB', True)

DP_TEMPERATURE = _get('DP_TEMPERATURE', 1.0)
DP_SWITCH_COST = _get('DP_SWITCH_COST', 2.5)

from soft_dp import SoftDP


class STMessageLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.msg = nn.Linear(dim, dim)
        self.upd = nn.GRUCell(dim, dim)

    def forward(self, Hprev, A_sp, A_tm):
        # Hprev: [B,K,D], A_*: [B,K,K]
        Msp = torch.bmm(A_sp, self.msg(Hprev))
        Mtm = torch.bmm(A_tm, self.msg(Hprev))
        M = Msp + Mtm
        B, K, D = M.shape
        H = self.upd(M.reshape(B * K, D), Hprev.reshape(B * K, D))
        return H.view(B, K, D)


class EnergyHead(nn.Module):
    def __init__(self, use_lcb=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(_get('E_ALPHA_SNR', 1.0))))
        self.beta = nn.Parameter(torch.tensor(float(_get('E_BETA_TTL', 1.5))))
        self.beta2 = nn.Parameter(torch.tensor(float(_get('E_BETA2_TTL', 0.5))))
        self.gamma = nn.Parameter(torch.tensor(float(_get('E_GAMMA_SWITCH', 5.0))))
        self.k_u = nn.Parameter(torch.tensor(float(_get('E_KAPPA_UNCERT', 1.0))))
        self.g_w = nn.Parameter(torch.tensor(1.0))
        self.ttl_scale = nn.Parameter(torch.tensor(float(_get('TTL_SCALE', 8.0))))
        self.use_lcb = use_lcb
    def forward(self, snr_mu, snr_logvar, ttl, cur_idx, delev):
        cur_idx = cur_idx.long()
        sigma = torch.sqrt(torch.clamp(snr_logvar.exp(), min=1e-6))
        snr_eff = snr_mu - self.k_u * sigma if self.use_lcb else snr_mu
        term_snr = - self.alpha * snr_eff  # [B,K]
        # 留存惩罚：只在切换时生效（i != cur），当前TTL越大越不想切
        ttl_cur = ttl.gather(1, cur_idx.view(-1, 1)).clamp_min(1.0)  # [B,1]
        stay_bonus = torch.tanh(ttl_cur / self.ttl_scale)  # [B,1]
        not_cur = (torch.arange(snr_mu.size(1), device=snr_mu.device)[None, :] != cur_idx[:, None]).float()
        term_ttl_sw = self.beta * not_cur * stay_bonus  # [B,K]

        # 候选TTL偏好：候选TTL越大越“便宜”（可选，先开小）
        term_ttl_cand = self.beta2 * (1.0 / (ttl + 1.0))  # [B,K]

        # 切换代价：几何迟滞（delev差越大，越容易切；用softplus避免负）

        delta_de = torch.abs(delev - delev.gather(1, cur_idx.view(-1, 1)))
        term_sw = self.gamma * not_cur * F.softplus(self.g_w * delta_de)
        E = term_snr + term_ttl_sw + term_ttl_cand + term_sw
        return E, {'snr': term_snr, 'ttl_sw': term_ttl_sw, 'ttl_cand': term_ttl_cand, 'switch': term_sw}


class PCGNN_OAEST(nn.Module):
    def __init__(self, f_ue, f_sat, f_edge, hidden=128, gnn_layers=2):
        super().__init__()
        self.f_ue = f_ue;
        self.f_sat = f_sat;
        self.f_edge = f_edge
        self.hidden = hidden

        # 节点编码
        self.enc = nn.Sequential(
            nn.Linear(f_sat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        # 时空消息传递（图上的 GRU）
        self.layers = nn.ModuleList([STMessageLayer(hidden) for _ in range(gnn_layers)])

        # ---- heads：每个节点输出一个标量（修复关键点）----
        self.psnr_mu = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.psnr_lv = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.ttl_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
                                      nn.Softplus())

        self.energy_head = EnergyHead(use_lcb=ENERGY_USE_LCB)
        self.softdp = SoftDP(horizon=1, switch_cost=DP_SWITCH_COST, temperature=DP_TEMPERATURE)

        # 可学习的邻接混合门
        self.sp_gate = nn.Parameter(torch.tensor(1.0))
        self.tm_gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, ue, sat_t, sat_prev, A_sp_init, A_tm_init, cur_idx):
        """
        ue: [B,1,Fu] (保留接口)
        sat_t, sat_prev: [B,K,Fs]
        A_sp_init, A_tm_init: [B,K,K]，已按行归一化更稳
        cur_idx: [B] 当前连接的候选局部索引
        """
        B, K, Fs = sat_t.shape

        # 编码
        H = self.enc(sat_t)  # [B,K,D]

        # 归一化邻接 + 学习门
        A_sp = torch.clamp(A_sp_init, min=0.0)
        A_sp = A_sp / (A_sp.sum(-1, keepdim=True) + 1e-6)
        A_tm = torch.clamp(A_tm_init, min=0.0)
        A_tm = A_tm / (A_tm.sum(-1, keepdim=True) + 1e-6)
        A_sp = self.sp_gate * A_sp
        A_tm = self.tm_gate * A_tm

        # 时空消息传递
        for layer in self.layers:
            H = layer(H, A_sp, A_tm)

        # ---- heads 输出 [B,K] ----
        mu = self.psnr_mu(H).squeeze(-1)  # [B,K]
        lv = self.psnr_lv(H).squeeze(-1)  # [B,K]
        ttl = self.ttl_head(H).squeeze(-1)  # [B,K]

        # 特征里第1列是 d_elev（或按你的列定义替换）
        delev = sat_t[..., 1] if sat_t.size(-1) > 1 else torch.zeros_like(mu)

        # 能量 + soft 策略
        E, comp = self.energy_head(mu, lv, ttl, cur_idx, delev)  # [B,K]
        q_next = self.softdp(E)  # [B,K]

        return {'psnr_mu': mu, 'psnr_lv': lv, 'ttl': ttl, 'energy': E, 'q_next': q_next, 'energy_comp': comp}