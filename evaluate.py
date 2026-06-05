"""
EgoHAnG — Evaluation Script
=============================
Loads a pretrained model and evaluates on a fused features CSV.
Reports Top-1, Top-5 accuracy and Top-5 recall per horizon.

Usage:
    python evaluate.py --dataset epic_kitchens \
        --fused_csv EPIC-Kitchens/Features/P01_04_fused_features_PCA.csv \
        --label_csv EPIC-Kitchens/Labels/P01_04.csv \
        --model_path checkpoints/P01_04_Fused_model.pth

    python evaluate.py --dataset egtea \
        --fused_csv EGTEA/Features/OP01-R01_fused_features_PCA.csv \
        --label_csv EGTEA/Labels/OP01-R01.csv \
        --model_path checkpoints/OP01-R01_Fused_model.pth
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import SingleVideoAnticipationDataset, IGNORE_INDEX
from model   import AnticipationModel

HORIZONS_S = [2.0, 1.75, 1.50, 1.25, 1.0, 0.75, 0.50, 0.25]
T_OBS      = 90
FEAT_DIM   = 512
K_FUT      = 8
BATCH_SIZE = 8

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


# ── Metrics ───────────────────────────────────────────────────────────────────
def topk_hits(logits, labels, k, ignore=IGNORE_INDEX):
    """Returns (hits, total) for top-k accuracy."""
    preds = logits.topk(k, dim=-1)[1]    # (B, K, k)
    B, K, _ = preds.shape
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


def per_horizon_topk(logits_all, labels_all, k, ignore=IGNORE_INDEX):
    """
    Returns list of top-k accuracy per horizon.
    logits_all: (N, K_fut, C)
    labels_all: (N, K_fut)
    """
    K = logits_all.size(1)
    preds = logits_all.topk(k, dim=-1)[1]   # (N, K, k)
    accs = []
    for h in range(K):
        lab  = labels_all[:, h]
        mask = (lab != ignore)
        if mask.sum() == 0:
            accs.append(None)
            continue
        predk  = preds[:, h, :]               # (N, k)
        lab_e  = lab.unsqueeze(1).expand(-1, k)
        hits   = (predk == lab_e)[mask].any(dim=1).float().sum().item()
        accs.append(hits / mask.sum().item())
    return accs


def top5_recall(logits_all, labels_all, ignore=IGNORE_INDEX):
    """
    Mean Top-5 recall across all horizons and samples.
    Equivalent to the metric used in EPIC-Kitchens benchmark.
    """
    N, K, C = logits_all.shape
    preds   = logits_all.topk(5, dim=-1)[1]  # (N, K, 5)
    hits, total = 0, 0
    for h in range(K):
        lab  = labels_all[:, h]
        mask = (lab != ignore)
        if mask.sum() == 0:
            continue
        predk = preds[:, h, :][mask]
        lab_m = lab[mask].unsqueeze(1).expand(-1, 5)
        hits  += (predk == lab_m).any(dim=1).float().sum().item()
        total += mask.sum().item()
    return hits / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EgoHAnG Evaluation")
    parser.add_argument("--dataset",    required=True,
                        choices=["epic_kitchens", "egtea"])
    parser.add_argument("--fused_csv",  required=True)
    parser.add_argument("--label_csv",  required=True)
    parser.add_argument("--model_path", required=True,
                        help="Path to pretrained .pth file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    cfg = DATASET_CFG[args.dataset]

    # ── Load model ──
    print(f"[INFO] Loading model from {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)

    # Support both raw state_dict and full checkpoint dict
    if "model_state" in ckpt:
        state_dict  = ckpt["model_state"]
        num_classes = ckpt.get("num_classes", cfg["num_classes"])
        feat_dim    = ckpt.get("feat_dim",    FEAT_DIM)
        k_fut       = ckpt.get("k_fut",       K_FUT)
        horizons_s  = ckpt.get("horizons_s",  HORIZONS_S)
    else:
        state_dict  = ckpt
        num_classes = cfg["num_classes"]
        feat_dim    = FEAT_DIM
        k_fut       = K_FUT
        horizons_s  = HORIZONS_S

    model = AnticipationModel(
        feat_dim    = feat_dim,
        num_classes = num_classes,
        k_fut       = k_fut,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Model loaded successfully.")

    # ── Dataset ──
    ds = SingleVideoAnticipationDataset(
        fused_csv_path = args.fused_csv,
        label_csv_path = args.label_csv,
        t_obs          = T_OBS,
        k_fut          = k_fut,
        feat_dim       = feat_dim,
        fps            = cfg["fps"],
        horizons_s     = horizons_s,
    )
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    print(f"[INFO] Evaluating on {len(ds)} samples...")

    # ── Collect predictions ──
    all_logits = {t: [] for t in ["verb", "noun", "action"]}
    all_labels = {t: [] for t in ["verb", "noun", "action"]}

    with torch.no_grad():
        for F_batch, y_multi, _ in loader:
            F_batch = F_batch.to(device)
            logits  = model(F_batch)
            for t in ["verb", "noun", "action"]:
                all_logits[t].append(logits[t].detach().cpu())
                all_labels[t].append(y_multi[t].detach().cpu())

    # ── Compute metrics ──
    print("\n" + "="*60)
    print("  EgoHAnG Evaluation Results")
    print("="*60)

    for task in ["verb", "noun", "action"]:
        lg_all = torch.cat(all_logits[task], dim=0)   # (N, K, C)
        lb_all = torch.cat(all_labels[task], dim=0)   # (N, K)

        h1, t1 = topk_hits(lg_all, lb_all, k=1)
        h5, t5 = topk_hits(lg_all, lb_all, k=5)
        t5r    = top5_recall(lg_all, lb_all)

        print(f"\n  {task.upper()}")
        print(f"    Top-1 Accuracy : {100*h1/t1:.2f}%  ({h1}/{t1})")
        print(f"    Top-5 Accuracy : {100*h5/t5:.2f}%  ({h5}/{t5})")
        print(f"    Top-5 Recall   : {100*t5r:.2f}%")

        # Per-horizon breakdown
        h_accs = per_horizon_topk(lg_all, lb_all, k=5)
        print(f"    Per-Horizon Top-5 Accuracy:")
        for i, (h_sec, acc) in enumerate(
                zip(horizons_s, h_accs)):
            val = f"{100*acc:.2f}%" if acc is not None else "N/A"
            print(f"      T={h_sec:.2f}s  ->  {val}")

    print("\n" + "="*60)
    print("[DONE]")


if __name__ == "__main__":
    main()
