# train.py —— PCGNN + TTL/Stay 多头联合训练 + True-Margin 切换惩罚（完整可跑，健壮版）

# train_plus_clean.py — PCGNN with anti-ping-pong training (temporal consistency + adaptive lambda)

import os, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import PCGNN
from sim_dataset import generate_dataset_from_tle
from config import (
    # data/sim
    TLE_PATH, NUM_SAMPLES, LIMIT_SATS, RNG_SEED,
    # training
    EPOCHS, BATCH_SIZE, NUM_WORKERS, LR, WEIGHT_DECAY, GRAD_CLIP,
    # model
    F_UE, F_SAT, F_EDGE, HIDDEN, GNN_LAYERS, GRAPH_TOPK, ADJ_TAU,
    # multi-task loss
    MSE_W, CE_W, BCE_W,
    TTL_W, STAY_W, EXTRA_MARGIN_DB, TAU_TTL_STEPS,
    # switch penalty & link constants
    SWITCH_PENALTY_W, HOM_DB,
    SAT_TX_POWER_DBM, SAT_ANT_GAIN_DBI, BANDWIDTH_HZ, NOISE_PSD_DBM_HZ,
    # new: training stabilizers
    TARGET_SWITCH_RATE, LAMBDA_RHO, LAMBDA_INIT, LAMBDA_MAX, CONS_W
)

# ---------------------------
# Dataset & Collate
# ---------------------------
class HODataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        d = self.samples[i]
        ue   = torch.tensor(d['ue_feat'],   dtype=torch.float32)  # [1, F_UE]
        sat  = torch.tensor(d['sat_feats'], dtype=torch.float32)  # [K, F_SAT]
        edge = torch.tensor(d['edge_attr'], dtype=torch.float32)  # [K, F_EDGE]
        ycl  = torch.tensor(d['label_cls'], dtype=torch.long)     # ()
        ysnr = torch.tensor(d['label_snr'], dtype=torch.float32)  # [K]
        yho  = torch.tensor(d['ho_event'],  dtype=torch.float32)  # ()
        yttl = torch.tensor(d['ttl'],        dtype=torch.float32) if 'ttl'        in d else None  # [K]
        ystay= torch.tensor(d['y_stay'],     dtype=torch.float32) if 'y_stay'     in d else None  # ()
        ycur = torch.tensor(d['label_curr'], dtype=torch.long)    if 'label_curr' in d else None  # ()
        return ue, sat, edge, ycl, ysnr, yho, yttl, ystay, ycur

def collate(batch):
    B = len(batch)
    ue   = torch.stack([b[0] for b in batch])  # [B,1,Fu]
    sat  = torch.stack([b[1] for b in batch])  # [B,K,Fs]
    edge = torch.stack([b[2] for b in batch])  # [B,K,Fe]
    ycl  = torch.stack([b[3] for b in batch])  # [B]
    ysnr = torch.stack([b[4] for b in batch])  # [B,K]
    yho  = torch.stack([b[5] for b in batch])  # [B]
    any_yttl_missing  = any(b[6] is None for b in batch)
    any_ystay_missing = any(b[7] is None for b in batch)
    any_ycur_missing  = any(b[8] is None for b in batch)
    if any_yttl_missing:
        yttl = torch.zeros((B, ysnr.shape[1]), dtype=torch.float32)
    else:
        yttl = torch.stack([b[6] for b in batch])
    if any_ystay_missing:
        ystay = torch.zeros((B,), dtype=torch.float32)
    else:
        ystay = torch.stack([b[7] for b in batch])
    if any_ycur_missing:
        ycur = torch.zeros((B,), dtype=torch.long)
    else:
        ycur = torch.stack([b[8] for b in batch])
    return (ue, sat, edge, ycl, ysnr, yho, yttl, ystay, ycur,
            any_yttl_missing, any_ystay_missing, any_ycur_missing)

# ---------------------------
# Train
# ---------------------------
def main():
    # seeds
    torch.manual_seed(RNG_SEED); np.random.seed(RNG_SEED); random.seed(RNG_SEED)
    lambda_sw = float(LAMBDA_INIT)

    # data
    print(">> Generating dataset ...", flush=True)
    try:
        samples = generate_dataset_from_tle(TLE_PATH, num_samples=NUM_SAMPLES, seed=RNG_SEED, limit_sats=LIMIT_SATS)
    except TypeError:
        samples = generate_dataset_from_tle(TLE_PATH, num_samples=NUM_SAMPLES, seed=RNG_SEED)
    n_train = int(0.8 * len(samples))
    train_data, val_data = samples[:n_train], samples[n_train:]

    tl = DataLoader(HODataset(train_data), batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True)
    vl = DataLoader(HODataset(val_data),   batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True)

    # model/optim
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCGNN(f_ue=F_UE, f_sat=F_SAT, f_edge=F_EDGE,
                  hidden=HIDDEN, gnn_layers=GNN_LAYERS,
                  graph_topk=GRAPH_TOPK, adj_tau=ADJ_TAU).to(dev)
    ce  = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse = nn.MSELoss(); bce = nn.BCELoss(); l1 = nn.L1Loss()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # link constants
    EIRP  = (SAT_TX_POWER_DBM + SAT_ANT_GAIN_DBI)                       # dBm
    NOISE = (NOISE_PSD_DBM_HZ + 10.0 * math.log10(BANDWIDTH_HZ))        # dBm
    PL_COL = 4 if F_SAT >= 6 else 2

    best_vacc = 0.0
    last_plog_detached = None  # for temporal consistency

    for ep in range(EPOCHS):
        model.train()
        tloss_sum = tacc_sum = tsum = 0.0
        epoch_switch_sum = 0.0
        epoch_sample_sum = 0.0

        for batch in tl:
            (ue, sat, edge, ycl, ysnr, yho, yttl, ystay, ycur,
             yttl_missing, ystay_missing, ycur_missing) = batch

            ue, sat, edge = ue.to(dev), sat.to(dev), edge.to(dev)
            ycl, ysnr, yho = ycl.to(dev), ysnr.to(dev), yho.to(dev)
            yttl = yttl.to(dev); ystay = ystay.to(dev); ycur = ycur.to(dev)
            use_ttl_loss = not yttl_missing

            opt.zero_grad()
            psnr, plog, pho, p_stay, p_ttl = model(ue, sat, edge)

            # multi-task losses
            loss_reg = MSE_W * mse(psnr / 10.0, ysnr / 10.0)
            loss_ce  = CE_W  * ce(plog, ycl)
            loss_ho  = BCE_W * bce(pho, yho)
            loss_ttl = TTL_W * l1(p_ttl, yttl) if use_ttl_loss else torch.zeros((), device=dev)
            loss_stay= STAY_W * bce(p_stay, ystay)

            # temporal consistency (KL to previous-batch avg), only when have TTL & current label
            if (last_plog_detached is not None) and (not yttl_missing) and (not ycur_missing):
                p_now  = torch.softmax(plog, dim=1)                      # [B,K]
                p_prev = torch.softmax(last_plog_detached.to(dev), dim=1)# [B0,K]
                p_ref  = p_prev.mean(0, keepdim=True)                    # [1,K]
                ttl_cur = yttl.gather(1, ycur.view(-1,1)).squeeze(1)     # [B]
                mask = (ttl_cur > TAU_TTL_STEPS).float()                 # [B]
                kl = (p_now * (torch.log(p_now + 1e-8) - torch.log(p_ref + 1e-8))).sum(1)  # [B]
                loss_cons = (CONS_W * kl * mask).mean()
            else:
                loss_cons = torch.zeros((), device=dev)
            last_plog_detached = plog.detach().cpu()

            # true-margin switch penalty (depends on predicted target)
            with torch.no_grad():
                pred_idx = plog.argmax(1)                                # [B]
            pl_now      = sat[:, :, PL_COL]                               # [B,K]
            snr_now_cur = EIRP - pl_now.gather(1, ycur.view(-1,1)).squeeze(1) - NOISE  # [B]
            snr_future_pred = ysnr.gather(1, pred_idx.view(-1,1)).squeeze(1)          # [B]
            margin_true = snr_future_pred - snr_now_cur                                  # [B]
            switch_flag = (pred_idx != ycur).float()
            epoch_switch_sum += float(switch_flag.sum().item())
            epoch_sample_sum += float(switch_flag.numel())
            # λ from config (adaptive)
            loss_sw = lambda_sw * torch.relu(torch.tensor(HOM_DB + EXTRA_MARGIN_DB, device=dev) - margin_true) * switch_flag
            loss_sw = loss_sw.mean()

            loss = loss_reg + loss_ce + loss_ho + loss_ttl + loss_stay + loss_sw + loss_cons
            loss.backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            tloss_sum += float(loss.item()) * ycl.numel()
            tacc_sum  += float((plog.argmax(1) == ycl).float().sum().item())
            tsum      += float(ycl.numel())

        tloss = tloss_sum / max(1.0, tsum)
        tacc  = tacc_sum  / max(1.0, tsum)

        # validation
        model.eval()
        vloss_sum = vacc_sum = vsum = 0.0
        with torch.no_grad():
            for batch in vl:
                (ue, sat, edge, ycl, ysnr, yho, yttl, ystay, ycur,
                 yttl_missing, ystay_missing, ycur_missing) = batch
                ue, sat, edge = ue.to(dev), sat.to(dev), edge.to(dev)
                ycl, ysnr, yho = ycl.to(dev), ysnr.to(dev), yho.to(dev)
                yttl = yttl.to(dev); ystay = ystay.to(dev); ycur = ycur.to(dev)
                use_ttl_loss = not yttl_missing

                psnr, plog, pho, p_stay, p_ttl = model(ue, sat, edge)
                loss_reg = MSE_W * mse(psnr / 10.0, ysnr / 10.0)
                loss_ce  = CE_W  * ce(plog, ycl)
                loss_ho  = BCE_W * bce(pho, yho)
                loss_ttl = TTL_W * l1(p_ttl, yttl) if use_ttl_loss else torch.zeros((), device=dev)
                loss_stay= STAY_W * bce(p_stay, ystay)
                # no temporal consistency on val
                with torch.no_grad():
                    pred_idx = plog.argmax(1)
                pl_now      = sat[:, :, PL_COL]
                snr_now_cur = EIRP - pl_now.gather(1, ycur.view(-1,1)).squeeze(1) - NOISE
                snr_future_pred = ysnr.gather(1, pred_idx.view(-1,1)).squeeze(1)
                margin_true = snr_future_pred - snr_now_cur
                switch_flag = (pred_idx != ycur).float()
                loss_sw = lambda_sw * torch.relu(torch.tensor(HOM_DB + EXTRA_MARGIN_DB, device=dev) - margin_true) * switch_flag
                loss_sw = loss_sw.mean()

                loss = loss_reg + loss_ce + loss_ho + loss_ttl + loss_stay + loss_sw
                vloss_sum += float(loss.item()) * ycl.numel()
                vacc_sum  += float((pred_idx == ycl).float().sum().item())
                vsum      += float(ycl.numel())

        vloss = vloss_sum / max(1.0, vsum)
        vacc  = vacc_sum  / max(1.0, vsum)
        print(f"Epoch {ep:03d} | TrainLoss {tloss:.4f} Acc {tacc:.3f} | ValLoss {vloss:.4f} Acc {vacc:.3f}")

        # adaptive lambda update (Lagrangian)
        if epoch_sample_sum > 0:
            sw_rate = epoch_switch_sum / epoch_sample_sum
            lambda_sw = float(min(LAMBDA_MAX, max(0.0, lambda_sw + LAMBDA_RHO * (sw_rate - TARGET_SWITCH_RATE))))
            print(f"    [lambda] sw_rate={sw_rate:.3f} -> lambda_sw={lambda_sw:.3f}")
        else:
            print(f"    [lambda] no switch stats; lambda_sw={lambda_sw:.3f}")

        # save
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/last.pt")
        if vacc > best_vacc:
            best_vacc = vacc
            torch.save(model.state_dict(), "checkpoints/best.pt")
            print(f">> Saved best model (Val Acc={vacc:.3f})", flush=True)

if __name__ == "__main__":
    main()

