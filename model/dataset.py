import random

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfilt
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

FS = 200
WIN_SAMPLES = 10_000
MICRO = 10                          # micro-steps per macro column (5 ms each)
N_MACRO = WIN_SAMPLES // MICRO      # 1000 macro columns
CROP_LENGTHS = [2000, 5000, 10_000]  # 10 s / 25 s / 50 s → RGB channels
VOTE_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
CLASS_NAMES = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
LABEL_SMOOTHING = 0.005

# Double banana montage — 16 bipolar pairs (no midline)
DOUBLE_BANANA = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),   # LL
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),   # RL
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),   # LP
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),   # RP
]
N_CH = len(DOUBLE_BANANA)  # 16
EEG_COLS = list(dict.fromkeys(c for pair in DOUBLE_BANANA for c in pair))

# Left-right hemisphere flip: swap LL↔RL (0-3 ↔ 4-7) and LP↔RP (8-11 ↔ 12-15)
LR_FLIP = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]


def _bandpass(sig: np.ndarray, lo: float = 0.5, hi: float = 20.0) -> np.ndarray:
    sos = butter(5, [lo, hi], btype='bandpass', fs=FS, output='sos')
    return sosfilt(sos, sig, axis=-1).astype(np.float32)


def _signals_to_image(sig: np.ndarray, center: int | None = None) -> np.ndarray:
    """
    3 temporal crops → (3, 160, 1000), z-score normalised.
    Each crop uses the same micro-step reshape (N_CH × MICRO → height),
    then is linearly stretched to N_MACRO columns.
    Channel 0 = 10 s (fine), 1 = 25 s (mid), 2 = 50 s (full).
    """
    IMG_H = N_CH * MICRO  # 160
    T = sig.shape[1]
    if center is None:
        center = T // 2
    channels = []
    for crop_len in CROP_LENGTHS:
        if crop_len >= T:
            crop = sig
        else:
            s = int(np.clip(center - crop_len // 2, 0, T - crop_len))
            crop = sig[:, s: s + crop_len]
        n = (crop.shape[1] // MICRO) * MICRO
        frame = (crop[:, :n]
                 .reshape(N_CH, n // MICRO, MICRO)
                 .transpose(0, 2, 1)
                 .reshape(IMG_H, n // MICRO)
                 .astype(np.float32))
        if frame.shape[1] != N_MACRO:
            frame = cv2.resize(frame, (N_MACRO, IMG_H), interpolation=cv2.INTER_LINEAR)
        channels.append(frame)
    img = np.stack(channels)  # (3, 160, 1000)
    return (img - img.mean()) / (img.std() + 1e-6)


def build_df(csv_path: str | Path) -> pd.DataFrame:
    """
    Load train.csv and add:
      n_votes          = total raw vote count per annotation window
      expert_consensus = dominant class (for fold stratification)
    Vote columns are kept as raw counts (not normalised) for loss weighting.
    """
    df = pd.read_csv(csv_path)
    df['n_votes'] = df[VOTE_COLS].sum(axis=1)
    df['expert_consensus'] = df[VOTE_COLS].values.argmax(axis=1)
    df['expert_consensus'] = df['expert_consensus'].map(dict(enumerate(CLASS_NAMES)))
    return df


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Assign fold column. All rows of the same eeg_id share a fold.
    Stratify by per-eeg expert_consensus; group by patient_id (no leakage).
    """
    df = df.copy()
    eeg_meta = (
        df.sort_values('eeg_sub_id').groupby('eeg_id', sort=False)
        .first().reset_index()[['eeg_id', 'patient_id', 'expert_consensus']]
    )
    eeg_meta['fold'] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
        sgkf.split(eeg_meta, y=eeg_meta['expert_consensus'], groups=eeg_meta['patient_id'])
    ):
        eeg_meta.loc[val_idx, 'fold'] = fold
    df['fold'] = df['eeg_id'].map(eeg_meta.set_index('eeg_id')['fold'])
    return df


class EEGDataset(Dataset):
    """
    One item = one unique eeg_id.

    Signal:  random sub-window (train) / first sub-window by eeg_sub_id (val)
    Label:   aggregated mean votes across all sub-windows for this eeg_id —
             stable regardless of which sub-window is loaded
    Weight:  min(n_votes_of_chosen_row / 20, 1.0) — for stage-1 loss weighting

    min_votes: skip eeg_ids whose first sub-window has n_votes < min_votes
               (used to restrict val to high-quality annotations)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: str | Path,
        augment: bool = False,
        min_votes: int = 0,
    ):
        self.eeg_dir = Path(eeg_dir)
        self.augment = augment

        self.groups: dict[int, list[dict]] = {}
        self.agg_labels: dict[int, np.ndarray] = {}

        for eid, grp in df.sort_values('eeg_sub_id').groupby('eeg_id', sort=False):
            rows = grp.to_dict('records')
            if min_votes > 0 and rows[0]['n_votes'] < min_votes:
                continue
            self.groups[eid] = rows
            mean_votes = grp[VOTE_COLS].mean().values.astype(np.float32)
            mean_votes += LABEL_SMOOTHING
            self.agg_labels[eid] = mean_votes / mean_votes.sum()

        self.eeg_ids = list(self.groups.keys())

    def __len__(self) -> int:
        return len(self.eeg_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg_id = self.eeg_ids[idx]
        rows = self.groups[eeg_id]
        row = random.choice(rows) if self.augment else rows[0]

        sig = self._load(eeg_id, int(row['eeg_label_offset_seconds']))

        if self.augment and random.random() < 0.5:
            sig = sig[LR_FLIP]

        if self.augment:
            half = CROP_LENGTHS[1] // 2  # 2500 — keeps all crops in-bounds
            center = random.randint(half, WIN_SAMPLES - half)
        else:
            center = None

        img = _signals_to_image(sig, center=center)

        label = torch.from_numpy(self.agg_labels[eeg_id])
        weight = torch.tensor(min(row['n_votes'] / 20.0, 1.0), dtype=torch.float32)
        return torch.from_numpy(img), label, weight

    def _load(self, eeg_id: int, offset_sec: int) -> np.ndarray:
        raw = pq.read_table(self.eeg_dir / f'{eeg_id}.parquet', columns=EEG_COLS).to_pandas()
        start = offset_sec * FS
        chunk = raw.iloc[start: start + WIN_SAMPLES]
        sig = np.stack(
            [chunk[a].values - chunk[b].values for a, b in DOUBLE_BANANA], axis=0
        ).astype(np.float32)
        if sig.shape[1] < WIN_SAMPLES:
            sig = np.pad(sig, ((0, 0), (0, WIN_SAMPLES - sig.shape[1])))
        sig = np.nan_to_num(sig, nan=0.0, posinf=1024.0, neginf=-1024.0)
        sig = np.clip(sig, -1024.0, 1024.0) / 32.0
        return _bandpass(sig)
