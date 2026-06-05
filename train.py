"""
EgoHAnG — Training Script
==========================
Trains the EgoHAnG model on EPIC-Kitchens or EGTEA Gaze+.

Usage:
    python train.py --dataset epic_kitchens \
        --fused_csv EPIC-Kitchens/Features/P01_04_fused_features_PCA.csv \
        --label_csv EPIC-Kitchens/Labels/P01_04.csv \
        --save_path checkpoints/P01_04_model.pth

    python train.py --dataset egtea \
        --fused_csv EGTEA/Features/OP01-R01_fused_features_PCA.csv \
        --label_csv EGTEA/Labels/OP01-R01.csv \
        --save_path checkpoints/OP01-R01_model.pth
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from dataset import SingleVideoAnticipationDataset, IGNORE_INDEX
from model   import AnticipationModel

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASET_CFG = {
    "epic_kitchens": {
        "fps":         60.0,
        "num_classes": {"verb": 97, "noun": 300, "action": 2513},
    },
    "egtea": {
        "fps":         24.0,
        "num_classes": {"verb": 19, "noun": 51, "action": 106},
    },
}

HORIZONS_S = [2.0, 1.75, 1.50, 1.25, 1.0, 0.75, 0.50, 0.25]

# ── Hyperparameters ───────────────────────────────────────────────────────────
T_OBS      = 90
FEAT_DIM   = 512
K_FUT      = 8
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR         = 1e-4
WD         = 1e-4
VAL_SPLIT  = 0.2


# ── Loss helpers ──────────────────────────────────────────────────────────────
def masked_ce(logits, labels, ignore=-1):
    B, K, C    = logits.shape
    lf, ll     = logits.view(B*K, C), labels.view(B*K)
    loss_flat  = F.cross_entropy(lf, ll, reduction="none",
                                 ignore_index=ignore)
    mask       = (ll != ignore).float()
    valid      = mask.sum()
    return (loss_flat * mask).sum() / valid if valid > 0 \
        else (lf * 0.0).sum()


def focal_loss(logits, labels, gamma=2.0, ignore=-1):
    B, K, C = logits.shape
    lf, ll  = logits.view(B*K, C), labels.view(B*K)
    mask    = (ll != ignore)
    if mask.sum() == 0:
        return (lf * 0.0).sum()
    log_p   = F.log_softmax(lf, dim=-1)
    p       = log_p.exp()
    lf_m    = lf[mask]; ll_m = ll[mask]
    log_p_m = log_p[mask]; p_m = p[mask]
    pt      = p_m.gather(1, ll_m.unsqueeze(1)).squeeze(1)
    loss    = -((1 - pt) ** gamma) * log_p_m.gather(
        1, ll_m.unsqueeze(1)).squeeze(1)
    return loss.mean()


def temporal_smoothness(logits_list):
    """logits_list: list of (B, K, C) tensors for each task."""
    loss = 0.0
    for lg in logits_list:
        # lg: (B, K, C) -> probs (B, K, C)
        p = F.softmax(lg, dim=-1)
        loss += ((p[:, 1:, :] - p[:, :-1, :]) ** 2).mean()
    return loss


def topk_counts(logits, labels, k, ignore=-1):
    B, K, C = logits.shape
    preds   = logits.topk(k, dim=-1)[1]    # (B, K, k)
    hits, total = 0, 0
    for b in range(B):
        for h in range(K):
            lab = int(labels[b, h].item())
            if lab == ignore:
                continue
            total += 1
            if lab in preds[b, h].tolist():
                hits += 1
    return hits, total


# ── Training / Validation ─────────────────────────────────────────────────────
def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum, n_samples = 0.0, 0
    counts = {t: {"top1": [0,0], "top5": [0,0]}
              for t in ["verb", "noun", "action"]}
    logits_store = {t: [] for t in ["verb", "noun", "action"]}
    labels_store = {t: [] for t in ["verb", "noun", "action"]}

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for F_batch, y_multi, _ in tqdm(loader, leave=False):
            F_batch = F_batch.to(device)
            y_v = y_multi["verb"].to(device)
            y_n = y_multi["noun"].to(device)
            y_a = y_multi["action"].to(device)

            logits = model(F_batch)

            lce  = (masked_ce(logits["verb"],   y_v) +
                    masked_ce(logits["noun"],   y_n) +
                    masked_ce(logits["action"], y_a))
            lf   = (focal_loss(logits["verb"],   y_v) +
                    focal_loss(logits["noun"],   y_n) +
                    focal_loss(logits["action"], y_a))
            lts  = temporal_smoothness(
                [logits["verb"], logits["noun"], logits["action"]])
            loss = lce + lf + 0.1 * lts

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            b = F_batch.size(0)
            loss_sum  += float(loss.item()) * b
            n_samples += b

            for task, y in [("verb", y_v), ("noun", y_n),
                             ("action", y_a)]:
                lg = logits[task].detach().cpu()
                lb = y.detach().cpu()
                h1, t1 = topk_counts(lg, lb, k=1)
                h5, t5 = topk_counts(lg, lb, k=5)
                counts[task]["top1"][0] += h1
                counts[task]["top1"][1] += t1
                counts[task]["top5"][0] += h5
                counts[task]["top5"][1] += t5
                logits_store[task].append(lg)
                labels_store[task].append(lb)

    avg_loss = loss_sum / max(1, n_samples)
    metrics  = {}
    for task in ["verb", "noun", "action"]:
        h1, t1 = counts[task]["top1"]
        h5, t5 = counts[task]["top5"]
        metrics[f"{task}_top1"] = h1/t1 if t1 > 0 else None
        metrics[f"{task}_top5"] = h5/t5 if t5 > 0 else None

    return avg_loss, metrics, logits_store, labels_store


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EgoHAnG Training")
    parser.add_argument("--dataset",    required=True,
                        choices=["epic_kitchens", "egtea"])
    parser.add_argument("--fused_csv",  required=True,
                        help="Path to fused features CSV")
    parser.add_argument("--label_csv",  required=True,
                        help="Path to action label CSV")
    parser.add_argument("--save_path",  required=True,
                        help="Path to save best model .pth")
    parser.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--wd",         type=float, default=WD)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    cfg = DATASET_CFG[args.dataset]
    print(f"[INFO] Dataset: {args.dataset}  FPS={cfg['fps']}")

    # ── Dataset ──
    full_ds = SingleVideoAnticipationDataset(
        fused_csv_path = args.fused_csv,
        label_csv_path = args.label_csv,
        t_obs          = T_OBS,
        k_fut          = K_FUT,
        feat_dim       = FEAT_DIM,
        fps            = cfg["fps"],
        horizons_s     = HORIZONS_S,
    )
    n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    print(f"[INFO] Train: {n_train}  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model ──
    model = AnticipationModel(
        feat_dim    = FEAT_DIM,
        num_classes = cfg["num_classes"],
        k_fut       = K_FUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {total_params/1e6:.2f} M")

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_m, _, _ = run_epoch(
            model, train_loader, device, optimizer)
        val_loss,   val_m,   _, _ = run_epoch(
            model, val_loader,   device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs}  [{elapsed:.1f}s]")
        print(f"  Loss  Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        for task in ["verb", "noun", "action"]:
            print(f"  {task.upper():6s}  "
                  f"Train Top1={train_m[f'{task}_top1']}  "
                  f"Top5={train_m[f'{task}_top5']}  |  "
                  f"Val Top1={val_m[f'{task}_top1']}  "
                  f"Top5={val_m[f'{task}_top5']}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "val_loss":    val_loss,
                "dataset":     args.dataset,
                "num_classes": cfg["num_classes"],
                "feat_dim":    FEAT_DIM,
                "k_fut":       K_FUT,
                "horizons_s":  HORIZONS_S,
            }, args.save_path)
            print(f"  [SAVED BEST] -> {args.save_path}")

    print("\n[DONE] Training finished.")


if __name__ == "__main__":
    main()
