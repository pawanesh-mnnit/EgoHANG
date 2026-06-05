"""
EgoHAnG — Dataset Loader
=========================
SingleVideoAnticipationDataset:
    Loads pre-extracted fused feature CSVs and
    assigns time-based future labels at multiple
    anticipation horizons.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

IGNORE_INDEX = -1


class SingleVideoAnticipationDataset(Dataset):
    """
    Dataset for egocentric action anticipation.

    Args:
        fused_csv_path  : path to fused features CSV
                          (columns: frame_idx, feat_0 … feat_N)
        label_csv_path  : path to action label CSV
                          (columns: StartFrame, EndFrame,
                           Verb_class, Noun_class, ActionLabel)
        t_obs           : observation window length in frames (default 90)
        k_fut           : number of anticipation horizons (default 8)
        feat_dim        : feature dimension (default 512)
        fps             : video frame rate
                            EPIC-Kitchens = 60.0
                            EGTEA Gaze+   = 24.0
        horizons_s      : list of anticipation times in seconds
                          e.g. [2.0, 1.75, 1.50, 1.25, 1.0, 0.75, 0.50, 0.25]
    """

    def __init__(self, fused_csv_path, label_csv_path,
                 t_obs: int = 90, k_fut: int = 8,
                 feat_dim: int = 512, fps: float = 60.0,
                 horizons_s: list = None):

        if horizons_s is None:
            horizons_s = [2.0, 1.75, 1.50, 1.25, 1.0, 0.75, 0.50, 0.25]

        assert len(horizons_s) == k_fut, \
            f"len(horizons_s)={len(horizons_s)} must equal k_fut={k_fut}"

        # ── Load CSVs ──
        fused_df = pd.read_csv(fused_csv_path)
        if "frame_idx" not in fused_df.columns:
            raise KeyError("fused CSV must contain 'frame_idx' column")
        fused_df["frame_idx"] = fused_df["frame_idx"].astype(int)
        self.fused_df = (fused_df
                         .set_index("frame_idx", drop=False)
                         .sort_index())

        labels_df = pd.read_csv(label_csv_path)
        for col in ["StartFrame", "EndFrame"]:
            if col not in labels_df.columns:
                raise KeyError(f"label CSV must contain '{col}'")
        self.labels_df = labels_df.reset_index(drop=True)

        self.t_obs      = int(t_obs)
        self.k_fut      = int(k_fut)
        self.feat_dim   = int(feat_dim)
        self.feat_cols  = [f"feat_{i}" for i in range(self.feat_dim)]
        self.fps        = float(fps)
        self.horizons_s = list(horizons_s)

        # ── Build samples (one per label row) ──
        self.samples = []
        for ridx, row in self.labels_df.iterrows():
            try:
                obs_end = int(row["EndFrame"])
            except Exception:
                continue
            self.samples.append({"label_row_idx": int(ridx),
                                  "obs_end": obs_end})

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found in label CSV.")

    def __len__(self):
        return len(self.samples)

    # ── Label helpers ─────────────────────────────────────────────────────────
    def _col(self, *candidates):
        for c in candidates:
            if c in self.labels_df.columns:
                return c
        return None

    def _future_labels(self, obs_end: int):
        vcol = self._col("Verb_class",   "verb",   "Verb",   "verb_class")
        ncol = self._col("Noun_class",   "noun",   "Noun",   "noun_class")
        acol = self._col("Action_class", "action", "Action", "ActionLabel")

        verb_t, noun_t, action_t = [], [], []

        for h_sec in self.horizons_s:
            future_frame = obs_end + int(round(h_sec * self.fps))
            seg = self.labels_df[
                (self.labels_df["StartFrame"] <= future_frame) &
                (self.labels_df["EndFrame"]   >= future_frame)
            ]
            if seg.empty:
                verb_t.append(IGNORE_INDEX)
                noun_t.append(IGNORE_INDEX)
                action_t.append(IGNORE_INDEX)
            else:
                r = seg.iloc[0]
                verb_t.append(
                    int(r[vcol]) if vcol and not pd.isna(r[vcol])
                    else IGNORE_INDEX)
                noun_t.append(
                    int(r[ncol]) if ncol and not pd.isna(r[ncol])
                    else IGNORE_INDEX)
                action_t.append(
                    int(r[acol]) if acol and not pd.isna(r[acol])
                    else IGNORE_INDEX)

        return {
            "verb":   torch.LongTensor(verb_t),
            "noun":   torch.LongTensor(noun_t),
            "action": torch.LongTensor(action_t),
        }

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        rec     = self.samples[idx]
        obs_end = rec["obs_end"]

        fmin = int(self.fused_df.index.min())
        fmax = int(self.fused_df.index.max())
        obs_end   = min(obs_end, fmax)
        obs_start = max(obs_end - (self.t_obs - 1), fmin)

        desired = list(range(obs_start, obs_end + 1))
        sel = (self.fused_df
       .reindex(desired)
       .ffill()
       .bfill()
       .fillna(0.0))

        # Pad if shorter than t_obs
        if sel.shape[0] < self.t_obs:
            if sel.shape[0] == 0:
                sel = pd.DataFrame(
                    [{c: 0.0 for c in self.feat_cols}] * self.t_obs)
            else:
                first = sel.iloc[[0]]
                pads  = pd.concat(
                    [first] * (self.t_obs - sel.shape[0]),
                    ignore_index=True)
                sel = pd.concat(
                    [pads, sel.reset_index(drop=True)],
                    ignore_index=True)

        # Ensure all feature columns exist
        for c in self.feat_cols:
            if c not in sel.columns:
                sel[c] = 0.0

        F_window = torch.from_numpy(
            sel[self.feat_cols].values).float()   # (T_obs, feat_dim)

        y_multi = self._future_labels(obs_end)

        meta = {
            "obs_start":     int(obs_start),
            "obs_end":       int(obs_end),
            "label_row_idx": int(rec["label_row_idx"]),
        }
        return F_window, y_multi, meta
