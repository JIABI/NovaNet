
# train_oaest.py — Train OA-EST-PCGNN end-to-end
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import importlib
CFG = importlib.import_module('config')

def _get(name, default):
    return getattr(CFG, name, default)


# ==== Training & model configs (all overridable in config.py) ====

RNG_SEED = _get('RNG_SEED', 42)

EPOCHS = _get('EPOCHS', 20)

BATCH_SIZE = _get('BATCH_SIZE', 64)

NUM_WORKERS = _get('NUM_WORKERS', 0)

LR = _get('LR', 1e-3)

WEIGHT_DECAY = _get('WEIGHT_DECAY', 1e-4)

GRAD_CLIP = _get('GRAD_CLIP', 1.0)

USE_AMP = _get('USE_AMP', True)

F_UE = _get('F_UE', 1)

F_SAT = _get('F_SAT', 6)

F_EDGE = _get('F_EDGE', 0)

HIDDEN = _get('HIDDEN', 128)

GNN_LAYERS = _get('GNN_LAYERS', 2)

TOP_K = _get('TOP_K', 8)

# ==== Loss weights ====

L_UNCERT_W = _get('L_UNCERT_W', 0.2)  # Gaussian NLL for psnr (mu, logvar)

L_PATH_W = _get('L_PATH_W', 0.5)  # KL(planned || teacher_next)

L_TTL_W = _get('L_TTL_W', 1.0)  # TTL regression weight

L_ENERGY_SPARSE_W = _get('L_ENERGY_SPARSE_W', 0.0)

# ==== Adaptive lambda for switching constraint ====

TARGET_SWITCH_RATE = _get('TARGET_SWITCH_RATE', 0.12)

LAMBDA_RHO = _get('LAMBDA_RHO', 0.01)

LAMBDA_INIT = _get('LAMBDA_INIT', 1.0)

LAMBDA_MAX = _get('LAMBDA_MAX', 10.0)

HOM_DB = _get('HOM_DB', 0.0)  # stay margin baseline (dB)

EXTRA_MARGIN_DB = _get('EXTRA_MARGIN_DB', 0.0)  # extra margin (dB)

# ==== Soft-DP temperature (used to compute log_q from energy) ====

DP_TEMPERATURE = _get('DP_TEMPERATURE', 1.0)

# ==== Dataset generation params ====

NUM_SAMPLES = _get('NUM_SAMPLES', 2000)

LIMIT_SATS = _get('LIMIT_SATS', 32)

from sim_dataset_oaest import generate_dataset_oaest
from model_oaest import PCGNN_OAEST

class OADataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        d = self.samples[i]
        def t(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype)
        return (
            t(d['ue_feat']), t(d['sat_feats_t']), t(d['sat_feats_prev']),
            t(d['orbit_prior']), t(d['A_sp']), t(d['A_tm']),
            torch.tensor(d['cur_idx'], dtype=torch.long),
            t(d['label_snr']), t(d['ttl']), t(d['teacher_next_dist'])
        )

def collate(batch):
    ue = torch.stack([b[0] for b in batch])
    st = torch.stack([b[1] for b in batch])
    sp = torch.stack([b[2] for b in batch])
    op = torch.stack([b[3] for b in batch])
    Asp= torch.stack([b[4] for b in batch])
    Atm= torch.stack([b[5] for b in batch])
    cur= torch.stack([b[6] for b in batch])
    ysnr=torch.stack([b[7] for b in batch])
    yttl=torch.stack([b[8] for b in batch])
    qte =torch.stack([b[9] for b in batch])
    return ue, st, sp, op, Asp, Atm, cur, ysnr, yttl, qte

def gaussian_nll(mu, logvar, target):
    #var = torch.clamp(logvar.exp(), min=1e-6)
    var = (logvar.exp()).clamp(1e-3,1e3)
    return 0.5 * ( (target - mu)**2 / var + torch.log(var) )


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def has_bad(t: torch.Tensor) -> bool:
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()

def main():
    set_seed(RNG_SEED)

    print(">> Generating OA-EST dataset ...", flush=True)

    samples = generate_dataset_oaest(num_samples=NUM_SAMPLES, limit_sats=LIMIT_SATS, seed=RNG_SEED)

    n_train = int(0.8 * len(samples))

    train_data, val_data = samples[:n_train], samples[n_train:]

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tl = DataLoader(

        OADataset(train_data),

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(dev.type == 'cuda'),

        collate_fn=collate,

        drop_last=False

    )

    vl = DataLoader(

        OADataset(val_data),

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(dev.type == 'cuda'),

        collate_fn=collate,

        drop_last=False

    )

    model = PCGNN_OAEST(f_ue=F_UE, f_sat=F_SAT, f_edge=F_EDGE, hidden=HIDDEN, gnn_layers=GNN_LAYERS).to(dev)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and dev.type == 'cuda'))

    best = float('inf')

    lambda_sw = float(LAMBDA_INIT)

    for ep in range(EPOCHS):

        # ---------------- Train ----------------

        model.train()

        tloss_sum = tsum = 0

        # metrics

        acc_sum = sw_sum = cnt_sum = 0.0

        epoch_switch_sum = 0.0

        epoch_sample_sum = 0.0

        for ue, st, sp, op, Asp, Atm, cur, ysnr, yttl, qte in tl:

            ue, st, sp, op = ue.to(dev), st.to(dev), sp.to(dev), op.to(dev)

            Asp, Atm = Asp.to(dev), Atm.to(dev)

            cur, ysnr, yttl, qte = cur.to(dev), ysnr.to(dev), yttl.to(dev), qte.to(dev)

            # ensure teacher is a proper distribution

            qte = torch.clamp(qte, min=0.0)

            qte = qte / qte.sum(dim=1, keepdim=True).clamp_min(1e-12)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and dev.type == 'cuda')):

                outs = model(ue, st, sp, Asp, Atm, cur)

                mu, lv, ttl, E = outs['psnr_mu'], outs['psnr_lv'], outs['ttl'], outs['energy']

                # NaN/Inf guard on outputs

                if any(has_bad(t) for t in (mu, lv, ttl, E)):
                    print("[warn] NaN/Inf in outputs — skipping batch")

                    continue

                # losses

                loss_nll = gaussian_nll(mu, lv, ysnr).mean()

                # TTL 用 SmoothL1 更稳

                loss_ttl = F.smooth_l1_loss(ttl, yttl, beta=1.0)

                # 使用 energy 的 log_softmax 计算策略（与单步 SoftDP 等价），数值稳定

                log_q = F.log_softmax(-E / max(1e-6, DP_TEMPERATURE), dim=1)

                loss_path = F.kl_div(log_q, qte, reduction='batchmean')

                # 简单能量正则（可选）

                loss_e_reg = E.abs().mean()

                # switch penalty via margin true (teacher-based; compare pred vs current using next-step ysnr)

                pred_idx = torch.argmin(E, dim=1)  # [B]

                snr_future_pred = ysnr.gather(1, pred_idx.view(-1, 1)).squeeze(1)

                snr_future_cur = ysnr.gather(1, cur.view(-1, 1)).squeeze(1)

                margin_true = snr_future_pred - snr_future_cur

                switch_flag = (pred_idx != cur).float()

                loss_sw = lambda_sw * torch.relu(
                    torch.tensor(HOM_DB + EXTRA_MARGIN_DB, device=dev) - margin_true) * switch_flag

                loss_sw = loss_sw.mean()

                loss = (L_UNCERT_W * loss_nll

                        + L_TTL_W * loss_ttl

                        + L_PATH_W * loss_path

                        + L_ENERGY_SPARSE_W * loss_e_reg

                        + loss_sw)

            # more guards

            if not torch.isfinite(loss):
                print("[warn] Non-finite loss — skipping batch")

                continue

            # backward

            if scaler.is_enabled():

                scaler.scale(loss).backward()

                if GRAD_CLIP and GRAD_CLIP > 0:
                    scaler.unscale_(opt)

                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                scaler.step(opt)

                scaler.update()

            else:

                loss.backward()

                if GRAD_CLIP and GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                opt.step()

            B = ue.size(0)

            tloss_sum += float(loss.item()) * B

            tsum += B

            # metrics

            teacher_idx = torch.argmax(qte, dim=1)

            acc_batch = (pred_idx == teacher_idx).float().mean()

            switch_rate = switch_flag.mean()

            acc_sum += float(acc_batch.item()) * B

            sw_sum += float(switch_rate.item()) * B

            cnt_sum += B

            epoch_switch_sum += float((pred_idx != cur).float().sum().item())

            epoch_sample_sum += float(pred_idx.numel())

        tloss = tloss_sum / max(1, tsum)

        train_acc = acc_sum / max(1, cnt_sum)

        train_sw = sw_sum / max(1, cnt_sum)

        # ---------------- Val ----------------

        model.eval()

        vloss_sum = vsum = 0

        v_acc_sum = v_sw_sum = v_cnt_sum = 0.0

        with torch.no_grad():

            for ue, st, sp, op, Asp, Atm, cur, ysnr, yttl, qte in vl:

                ue, st, sp, op = ue.to(dev), st.to(dev), sp.to(dev), op.to(dev)

                Asp, Atm = Asp.to(dev), Atm.to(dev)

                cur, ysnr, yttl, qte = cur.to(dev), ysnr.to(dev), yttl.to(dev), qte.to(dev)

                qte = torch.clamp(qte, min=0.0)

                qte = qte / qte.sum(dim=1, keepdim=True).clamp_min(1e-12)

                outs = model(ue, st, sp, Asp, Atm, cur)

                mu, lv, ttl, E = outs['psnr_mu'], outs['psnr_lv'], outs['ttl'], outs['energy']

                # NaN guard

                if any(has_bad(t) for t in (mu, lv, ttl, E)):
                    print("[warn] NaN/Inf in VAL outputs — skipping batch")

                    continue

                loss_nll = gaussian_nll(mu, lv, ysnr).mean()

                loss_ttl = F.smooth_l1_loss(ttl, yttl, beta=1.0)

                log_q = F.log_softmax(-E / max(1e-6, DP_TEMPERATURE), dim=1)

                loss_path = F.kl_div(log_q, qte, reduction='batchmean')

                loss_e_reg = E.abs().mean()

                pred_idx = torch.argmin(E, dim=1)

                snr_future_pred = ysnr.gather(1, pred_idx.view(-1, 1)).squeeze(1)

                snr_future_cur = ysnr.gather(1, cur.view(-1, 1)).squeeze(1)

                margin_true = snr_future_pred - snr_future_cur

                switch_flag = (pred_idx != cur).float()

                loss_sw = lambda_sw * torch.relu(
                    torch.tensor(HOM_DB + EXTRA_MARGIN_DB, device=dev) - margin_true) * switch_flag

                loss_sw = loss_sw.mean()

                loss = (L_UNCERT_W * loss_nll

                        + L_TTL_W * loss_ttl

                        + L_PATH_W * loss_path

                        + L_ENERGY_SPARSE_W * loss_e_reg

                        + loss_sw)

                B = ue.size(0)

                vloss_sum += float(loss.item()) * B

                vsum += B

                teacher_idx = torch.argmax(qte, dim=1)

                acc_batch = (pred_idx == teacher_idx).float().mean()

                switch_rate = switch_flag.mean()

                v_acc_sum += float(acc_batch.item()) * B

                v_sw_sum += float(switch_rate.item()) * B

                v_cnt_sum += B

        vloss = vloss_sum / max(1, vsum)

        val_acc = v_acc_sum / max(1, v_cnt_sum)

        val_sw = v_sw_sum / max(1, v_cnt_sum)

        # print epoch summary

        print(f"Epoch {ep:03d} | "

              f"TrainLoss {tloss:.4f} | ValLoss {vloss:.4f} | "

              f"TrainAcc {train_acc:.3f} | ValAcc {val_acc:.3f} | "

              f"TrainSwitchRate {train_sw:.3f} | ValSwitchRate {val_sw:.3f}")

        # update lambda_sw by observed train switch rate

        if epoch_sample_sum > 0:

            sw_rate_true = epoch_switch_sum / epoch_sample_sum

            lambda_sw = float(min(LAMBDA_MAX, max(0.0, lambda_sw + LAMBDA_RHO * (sw_rate_true - TARGET_SWITCH_RATE))))

            print(f"    [λ] observed_sw={sw_rate_true:.3f} -> lambda_sw={lambda_sw:.3f}")

        else:

            print(f"    [λ] no switch stats; keep lambda_sw={lambda_sw:.3f}")

        # save checkpoints

        os.makedirs("checkpoints", exist_ok=True)

        torch.save(model.state_dict(), "checkpoints/oaest_last.pt")

        if vloss < best and math.isfinite(vloss):
            best = vloss

            torch.save(model.state_dict(), "checkpoints/oaest_best.pt")

            print(f">> Saved best model (Val Loss={vloss:.4f})", flush=True)


if __name__ == "__main__":
    main()
