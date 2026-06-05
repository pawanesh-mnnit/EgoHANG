"""
EgoHAnG — Step 1: Feature Extraction
======================================
Extracts RGB + Optical Flow features using ResNet-50,
applies PCA compression (2048 -> 512), and saves
fused feature CSVs for each video participant.

Usage:
    python extract_features.py --dataset epic_kitchens \
        --rgb_root /path/to/RGB \
        --flow_root /path/to/OpticalFlow \
        --labels_root EPIC-Kitchens/Labels \
        --output_root EPIC-Kitchens/Features \
        --pca_path tools/pca_2048_to_512.pkl

    python extract_features.py --dataset egtea \
        --rgb_root /path/to/EGTEA/RGB \
        --flow_root /path/to/EGTEA/Flow \
        --labels_root EGTEA/Labels \
        --output_root EGTEA/Features \
        --pca_path tools/pca_2048_to_512.pkl
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
FEAT_DIM    = 512
SAMPLE_RATE = 1
W_RGB       = 0.6
W_FLOW      = 0.4
IMG_EXTS    = {".jpg", ".jpeg", ".png"}

_frame_re = re.compile(r"(\d+)(?=\.[^.]+$)")

def parse_frame_index(fname: str) -> int:
    m = _frame_re.search(fname)
    if m:
        return int(m.group(1))
    digits = re.findall(r"\d+", fname)
    return int(digits[-1]) if digits else 0


# ── Model ─────────────────────────────────────────────────────────────────────
def build_resnet(device):
    model = resnet50(weights="IMAGENET1K_V1")
    model = nn.Sequential(*list(model.children())[:-1])  # remove fc layer
    return model.to(device).eval()


_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


@torch.no_grad()
def extract_feat(pil_img, model, pca, device):
    x    = _transform(pil_img).unsqueeze(0).to(device)
    feat = model(x).view(-1).cpu().numpy()   # (2048,)
    feat = pca.transform(feat[None, :])[0]   # (512,)
    return feat.astype(np.float32)


# ── Main extraction ───────────────────────────────────────────────────────────
def extract_and_save(rgb_folder, flow_folder, label_csv, out_csv,
                     model, pca, device, sample_rate=1,
                     w_rgb=0.6, w_flow=0.4, feat_dim=512):

    labels_df = pd.read_csv(label_csv)

    rgb_files = sorted(
        [p for p in Path(rgb_folder).iterdir()
         if p.suffix.lower() in IMG_EXTS],
        key=lambda p: parse_frame_index(p.name)
    )
    sampled = rgb_files[::sample_rate]

    if len(sampled) == 0:
        raise RuntimeError(f"No frames found in {rgb_folder}")

    rows = []
    for fp in tqdm(sampled, desc=f"Extracting {Path(rgb_folder).parent.name}"):
        fname     = fp.name
        frame_idx = parse_frame_index(fname)

        # ── RGB ──
        try:
            rgb_feat = extract_feat(Image.open(fp).convert("RGB"),
                                    model, pca, device)
        except Exception as e:
            print(f"[WARN] RGB skip {fname}: {e}")
            continue

        # ── Flow ──
        if flow_folder is not None:
            ffp = Path(flow_folder) / fname
            if not ffp.exists():
                ffp = fp
            try:
                flow_feat = extract_feat(Image.open(ffp).convert("RGB"),
                                         model, pca, device)
            except Exception as e:
                print(f"[WARN] Flow skip {fname}: {e}")
                flow_feat = np.zeros(feat_dim, dtype=np.float32)
        else:
            flow_feat = np.zeros(feat_dim, dtype=np.float32)

        # ── Fuse ──
        fused = w_rgb * rgb_feat + w_flow * flow_feat

        # ── Label ──
        lr = labels_df[
            (labels_df["StartFrame"] <= frame_idx) &
            (labels_df["EndFrame"]   >= frame_idx)
        ]
        if not lr.empty:
            action_label = int(lr.iloc[0].get("ActionLabel", -1))
            action_name  = str(lr.iloc[0].get("ActionName",  "Unknown"))
        else:
            action_label, action_name = -1, "Unknown"

        row = {"frame_idx": int(frame_idx), "frame_name": fname,
               "ActionLabel": action_label, "ActionName": action_name}
        for i, v in enumerate(fused):
            row[f"feat_{i}"] = float(v)
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows extracted.")

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}  ({len(df)} frames)")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EgoHAnG Feature Extraction")
    parser.add_argument("--dataset",      required=True,
                        choices=["epic_kitchens", "egtea"],
                        help="Dataset name")
    parser.add_argument("--rgb_root",     required=True,
                        help="Root folder containing per-participant RGB folders")
    parser.add_argument("--flow_root",    default=None,
                        help="Root folder for optical flow (optional)")
    parser.add_argument("--labels_root",  required=True,
                        help="Folder containing label CSV files")
    parser.add_argument("--output_root",  required=True,
                        help="Output folder for fused feature CSVs")
    parser.add_argument("--pca_path",     required=True,
                        help="Path to pretrained PCA .pkl file")
    parser.add_argument("--sample_rate",  type=int, default=1,
                        help="Frame sampling rate (default: 1)")
    parser.add_argument("--w_rgb",        type=float, default=0.6,
                        help="RGB fusion weight (default: 0.6)")
    parser.add_argument("--w_flow",       type=float, default=0.4,
                        help="Flow fusion weight (default: 0.4)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading ResNet-50...")
    model = build_resnet(device)

    print(f"[INFO] Loading PCA from {args.pca_path}...")
    pca = joblib.load(args.pca_path)
    assert pca.components_.shape == (FEAT_DIM, 2048), \
        f"PCA shape mismatch: expected (512, 2048), got {pca.components_.shape}"

    rgb_root    = Path(args.rgb_root)
    flow_root   = Path(args.flow_root) if args.flow_root else None
    labels_root = Path(args.labels_root)
    output_root = Path(args.output_root)

    # Find all participant folders
    participants = sorted([p for p in rgb_root.iterdir() if p.is_dir()])
    print(f"[INFO] Found {len(participants)} participant folders")

    for rgb_folder in participants:
        pid        = rgb_folder.name
        label_csv  = labels_root / f"{pid}.csv"
        out_csv    = output_root / f"{pid}_fused_features_PCA.csv"

        if not label_csv.exists():
            print(f"[SKIP] No label CSV for {pid}")
            continue

        flow_folder = (flow_root / pid) if flow_root else None

        print(f"\n[PROCESSING] {pid}")
        try:
            extract_and_save(
                rgb_folder   = rgb_folder,
                flow_folder  = flow_folder,
                label_csv    = label_csv,
                out_csv      = out_csv,
                model        = model,
                pca          = pca,
                device       = device,
                sample_rate  = args.sample_rate,
                w_rgb        = args.w_rgb,
                w_flow       = args.w_flow,
                feat_dim     = FEAT_DIM
            )
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")

    print("\n[DONE] Feature extraction complete.")


if __name__ == "__main__":
    main()
