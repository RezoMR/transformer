from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Feature engineering
# -----------------------------
def add_time_features_15min(df: pd.DataFrame, tz: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex for time features.")

    idx = out.index
    if tz is not None:
        if idx.tz is None:
            idx = idx.tz_localize(
                tz,
                nonexistent="shift_forward",
                ambiguous="infer"
            )
        else:
            idx = idx.tz_convert(tz)

    slot = (idx.hour * 4 + idx.minute // 15).astype(np.float32)
    dow  = idx.dayofweek.astype(np.float32)
    doy  = idx.dayofyear.astype(np.float32)

    out["slot_sin"] = np.sin(2 * np.pi * slot / 96.0)
    out["slot_cos"] = np.cos(2 * np.pi * slot / 96.0)
    out["dow_sin"]  = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"]  = np.cos(2 * np.pi * dow / 7.0)
    out["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    hour = idx.hour.astype(np.float32)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    return out



def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Sequence[int] = (1, 4, 96, 672),          # 15m, 1h, 1d, 1w
    rolling_windows: Sequence[int] = (4, 96, 672),  # 1h, 1d, 1w
) -> pd.DataFrame:
    """
    Adds target lags and rolling statistics computed from the target.
    """
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)

    for w in rolling_windows:
        out[f"{target_col}_rollmean{w}"] = out[target_col].shift(1).rolling(w).mean()
        out[f"{target_col}_rollstd{w}"]  = out[target_col].shift(1).rolling(w).std()

    return out


# -----------------------------
# Scaling (fit on train only)
# -----------------------------
@dataclass
class StandardScalerNP:
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    eps: float = 1e-8

    def fit(self, x: np.ndarray) -> "StandardScalerNP":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fit yet.")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fit yet.")
        return x * self.std_ + self.mean_


# -----------------------------
# Windowing helpers
# -----------------------------
def _to_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def make_supervised_arrays(
    df: pd.DataFrame,
    target_col: str,
    enc_cols: List[str],
    dec_cols: Optional[List[str]],
    L: int,
    H: int,
    stride: int = 1,
    dropna: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:

    if dropna:
        df = df.dropna()

    values_enc = df[enc_cols].to_numpy()
    y = df[target_col].to_numpy()

    values_dec = None
    if dec_cols is not None and len(dec_cols) > 0:
        values_dec = df[dec_cols].to_numpy()

    n = len(df)
    starts = np.arange(L, n - H + 1, stride, dtype=np.int64)

    X_hist = np.empty((len(starts), L, len(enc_cols)), dtype=np.float32)
    Y_fut  = np.empty((len(starts), H), dtype=np.float32)
    X_fut  = None if values_dec is None else np.empty((len(starts), H, len(dec_cols)), dtype=np.float32)

    for k, i in enumerate(starts):
        X_hist[k] = values_enc[i - L : i]
        Y_fut[k]  = y[i : i + H]
        if X_fut is not None:
            X_fut[k] = values_dec[i : i + H]

    return X_hist, X_fut, Y_fut, starts


# -----------------------------
# Torch Dataset
# -----------------------------
class WindowedForecastDataset(Dataset):
    def __init__(self, X_hist: np.ndarray, Y_fut: np.ndarray, X_fut: Optional[np.ndarray] = None):
        self.X_hist = torch.from_numpy(_to_float32(X_hist))
        self.Y_fut  = torch.from_numpy(_to_float32(Y_fut))
        self.X_fut  = None if X_fut is None else torch.from_numpy(_to_float32(X_fut))

    def __len__(self) -> int:
        return self.X_hist.shape[0]

    def __getitem__(self, idx: int):
        if self.X_fut is None:
            return self.X_hist[idx], self.Y_fut[idx]
        return self.X_hist[idx], self.X_fut[idx], self.Y_fut[idx]


# -----------------------------
# Train/val/test split
# -----------------------------
def time_split_indices(n_rows: int, train_frac: float = 0.7, val_frac: float = 0.15):
    n_train = int(n_rows * train_frac)
    n_val   = int(n_rows * val_frac)
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, n_rows)
    return train_idx, val_idx, test_idx


# -----------------------------
# End-to-end builder
# -----------------------------
@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scalers: Dict[str, StandardScalerNP]
    feature_cols: Dict[str, List[str]]
    arrays: Dict[str, Dict[str, np.ndarray]]


def prepare_for_quantile_transformer(
    df: pd.DataFrame,
    target_col: str,
    L: int,
    H: int = 96,                       # 24h dopredu pri 15min kroku
    period_col: str = "period_end",
    state_col: str = "Stav sustavy",
    encoder_feature_cols: Optional[List[str]] = None,
    decoder_feature_cols: Optional[List[str]] = None,
    add_time_feats: bool = True,
    add_target_lags: bool = True,
    tz: Optional[str] = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    stride: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
    scale_target: bool = True,
    scale_features: bool = True,
) -> PreparedData:

    work = df.copy()

    # 1) ensure period_end is datetime and set as index
    if period_col in work.columns:
        work[period_col] = pd.to_datetime(work[period_col])
        work = work.sort_values(period_col).set_index(period_col)
    else:
        if not isinstance(work.index, pd.DatetimeIndex):
            raise ValueError(f"Missing '{period_col}' column and index is not DatetimeIndex.")

    work = work.sort_index()

    # 2) encode Stav sustavy (categorical) -> one-hot
    if state_col in work.columns and work[state_col].dtype == object:
        dummies = pd.get_dummies(work[state_col], prefix="state", dtype=np.float32)
        work = work.drop(columns=[state_col]).join(dummies)

    # 3) time features for 15-min data
    if add_time_feats:
        work = add_time_features_15min(work, tz=tz)

    # 4) lag features on target
    if add_target_lags:
        work = add_lag_features(work, target_col=target_col)

    # 5) auto-pick feature columns if not provided
    if encoder_feature_cols is None:
        candidates = work.select_dtypes(include=[np.number]).columns.tolist()
        encoder_feature_cols = [c for c in candidates if c != target_col]

    if decoder_feature_cols is None:
        # known-future safe defaults: only calendar features
        decoder_feature_cols = [c for c in [
            "slot_sin","slot_cos","dow_sin","dow_cos","doy_sin","doy_cos","hour_sin","hour_cos"
        ] if c in work.columns]

    # validate
    missing_enc = [c for c in encoder_feature_cols if c not in work.columns]
    missing_dec = [c for c in decoder_feature_cols if c not in work.columns]
    if missing_enc:
        raise ValueError(f"Missing encoder columns: {missing_enc}")
    if decoder_feature_cols and missing_dec:
        raise ValueError(f"Missing decoder columns: {missing_dec}")
    if target_col not in work.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # 6) build windows
    X_hist, X_fut, Y_fut, starts = make_supervised_arrays(
        work,
        target_col=target_col,
        enc_cols=encoder_feature_cols,
        dec_cols=decoder_feature_cols if len(decoder_feature_cols) > 0 else None,
        L=L,
        H=H,
        stride=stride,
        dropna=True,
    )

    # 7) split sequentially (no leakage)
    n_samples = X_hist.shape[0]
    train_idx, val_idx, test_idx = time_split_indices(n_samples, train_frac=train_frac, val_frac=val_frac)

    scalers: Dict[str, StandardScalerNP] = {}

    # 8) scaling fit only on train
    if scale_features:
        enc_scaler = StandardScalerNP().fit(X_hist[train_idx].reshape(-1, X_hist.shape[-1]))
        X_hist = enc_scaler.transform(X_hist.reshape(-1, X_hist.shape[-1])).reshape(X_hist.shape)
        scalers["enc"] = enc_scaler

        if X_fut is not None:
            dec_scaler = StandardScalerNP().fit(X_fut[train_idx].reshape(-1, X_fut.shape[-1]))
            X_fut = dec_scaler.transform(X_fut.reshape(-1, X_fut.shape[-1])).reshape(X_fut.shape)
            scalers["dec"] = dec_scaler

    if scale_target:
        y_scaler = StandardScalerNP().fit(Y_fut[train_idx].reshape(-1, 1))
        Y_fut = y_scaler.transform(Y_fut.reshape(-1, 1)).reshape(Y_fut.shape)
        scalers["y"] = y_scaler

    # 9) datasets + loaders (for time series: shuffle=False!)
    train_ds = WindowedForecastDataset(X_hist[train_idx], Y_fut[train_idx], None if X_fut is None else X_fut[train_idx])
    val_ds   = WindowedForecastDataset(X_hist[val_idx],   Y_fut[val_idx],   None if X_fut is None else X_fut[val_idx])
    test_ds  = WindowedForecastDataset(X_hist[test_idx],  Y_fut[test_idx],  None if X_fut is None else X_fut[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    arrays = {
        "train": {"X_hist": X_hist[train_idx], "X_fut": None if X_fut is None else X_fut[train_idx], "Y_fut": Y_fut[train_idx]},
        "val":   {"X_hist": X_hist[val_idx],   "X_fut": None if X_fut is None else X_fut[val_idx],   "Y_fut": Y_fut[val_idx]},
        "test":  {"X_hist": X_hist[test_idx],  "X_fut": None if X_fut is None else X_fut[test_idx],  "Y_fut": Y_fut[test_idx]},
    }

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scalers=scalers,
        feature_cols={"enc": encoder_feature_cols, "dec": decoder_feature_cols, "target": [target_col]},
        arrays=arrays,
    )


def cut_last_full_days(df: pd.DataFrame, period_col="period_end", days=2, steps_per_day=96) -> pd.DataFrame:
    work = df.copy()
    if period_col in work.columns:
        work[period_col] = pd.to_datetime(work[period_col])
        work = work.sort_values(period_col)
    else:
        raise ValueError(f"Missing '{period_col}' column.")

    n_cut = days * steps_per_day
    if len(work) <= n_cut:
        raise ValueError("Not enough rows to cut last days.")

    return work.iloc[:-n_cut].reset_index(drop=True)


df = pd.read_csv("raw_data/raw_data_merged.csv")
df_cut = cut_last_full_days(df)
prepared = prepare_for_quantile_transformer(
    df=df_cut,
    target_col="zuctovacia cena za odchylku",
    L=96*7,     # napr. 7 dní histórie (7*96 = 672)
    H=96,       # 1 deň dopredu
)