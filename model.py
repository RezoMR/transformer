# Quantile Encoder–Decoder Transformer for Direct Multi-Horizon Forecasting
# horizon = 96, quantiles = 0.1..0.9, loss = pinball (quantile regression)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any

import torch
import torch.nn as nn


# -----------------------------
# Utilities
# -----------------------------
def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Causal mask for decoder self-attention (True = masked). Shape (T, T)."""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def pinball_loss(
    y: torch.Tensor,
    yhat_q: torch.Tensor,
    quantiles: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Pinball (quantile) loss.

    y:      (B, H) or (B, H, 1)
    yhat_q: (B, H, Q)
    quantiles: (Q,) in (0,1)
    """
    if y.dim() == 2:
        y = y.unsqueeze(-1)  # (B,H,1)
    if yhat_q.dim() != 3:
        raise ValueError(f"yhat_q must be (B,H,Q), got {tuple(yhat_q.shape)}")
    if quantiles.dim() != 1 or quantiles.numel() != yhat_q.size(-1):
        raise ValueError("quantiles must be (Q,) matching last dim of yhat_q")

    q = quantiles.view(1, 1, -1).to(yhat_q.device)  # (1,1,Q)
    err = y - yhat_q                                 # (B,H,Q)
    loss = torch.maximum(q * err, (1.0 - q) * (-err))

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be one of: 'mean','sum','none'")


def quantile_crossing_penalty(yhat_q: torch.Tensor) -> torch.Tensor:
    """
    Penalize quantile crossings: q_{k} should be <= q_{k+1}.
    yhat_q: (B,H,Q)
    Returns scalar penalty (mean of positive crossings).
    """
    # diff along quantile dimension
    diffs = yhat_q[..., 1:] - yhat_q[..., :-1]  # should be >= 0
    return torch.relu(-diffs).mean()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for (B, T, D) with batch_first=True."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1,max_len,D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


# -----------------------------
# Model
# -----------------------------
@dataclass
class QuantileTransformerConfig:
    horizon: int = 96
    quantiles: Sequence[float] = tuple([0.1 * i for i in range(1, 10)])  # 0.1..0.9

    # Feature dimensions (MUSIA sedieť s tvojimi DataLoadermi)
    d_enc_in: int = 16
    d_dec_in: int = 8  # can be 0

    # Transformer sizes
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Positional encoding
    max_len: int = 4096

    # Whether decoder uses known-future covariates (x_future)
    use_known_future: bool = True

    # Optional quantile crossing penalty (0 = off)
    crossing_penalty_weight: float = 0.0


class QuantileEncoderDecoderTransformer(nn.Module):
    """
    Direct multi-horizon quantile forecaster.

    Inputs:
      x_hist   : (B, L, d_enc_in)
      x_future : (B, H, d_dec_in)  [optional]
    Output:
      yhat_q   : (B, H, Q)
    """
    def __init__(self, cfg: QuantileTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.horizon
        self.Q = len(cfg.quantiles)

        # store quantiles as buffer so it moves with .to(device)
        self.register_buffer("quantiles", torch.tensor(list(cfg.quantiles), dtype=torch.float32))

        # Projections
        self.enc_in_proj = nn.Linear(cfg.d_enc_in, cfg.d_model)

        self.horizon_queries = nn.Parameter(torch.randn(1, cfg.horizon, cfg.d_model) * 0.02)

        if cfg.d_dec_in > 0:
            self.dec_in_proj = nn.Linear(cfg.d_dec_in, cfg.d_model)
        else:
            self.dec_in_proj = None

        # Positional encodings
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        self.pos_dec = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)

        # Encoder / Decoder stacks
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)

        # Quantile head
        self.out_head = nn.Linear(cfg.d_model, self.Q)

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: Optional[torch.Tensor] = None,
        hist_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_hist: (B, L, d_enc_in)
        x_future: (B, H, d_dec_in) or None
        hist_padding_mask: (B, L) bool, True for padded positions (optional)
        """
        B, L, d = x_hist.shape
        if d != self.cfg.d_enc_in:
            raise ValueError(f"x_hist last dim {d} != cfg.d_enc_in {self.cfg.d_enc_in}")

        # Encoder
        enc = self.pos_enc(self.enc_in_proj(x_hist))  # (B,L,D)
        memory = self.encoder(enc, src_key_padding_mask=hist_padding_mask)  # (B,L,D)

        # Decoder tokens (one per horizon step)
        tgt = self.horizon_queries.repeat(B, 1, 1)  # (B,H,D)

        # Add known-future covariates if configured
        if self.cfg.use_known_future and self.dec_in_proj is not None:
            if x_future is None:
                raise ValueError("use_known_future=True but x_future is None.")
            if x_future.shape != (B, self.H, self.cfg.d_dec_in):
                raise ValueError(
                    f"x_future shape {tuple(x_future.shape)} expected {(B, self.H, self.cfg.d_dec_in)}"
                )
            tgt = tgt + self.dec_in_proj(x_future)

        tgt = self.pos_dec(tgt)

        # causal mask for decoder self-attention across horizon
        tgt_mask = make_causal_mask(self.H, device=x_hist.device)  # (H,H) bool

        dec = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)  # (B,H,D)
        yhat_q = self.out_head(dec)  # (B,H,Q)
        return yhat_q

    def loss(self, y_true: torch.Tensor, yhat_q: torch.Tensor) -> torch.Tensor:
        base = pinball_loss(y_true, yhat_q, self.quantiles, reduction="mean")
        if self.cfg.crossing_penalty_weight > 0:
            base = base + self.cfg.crossing_penalty_weight * quantile_crossing_penalty(yhat_q)
        return base


# -----------------------------
# Helpers to build cfg from your PreparedData
# -----------------------------
def build_cfg_from_prepared(
    prepared: Any,  # your PreparedData
    horizon: int = 96,
    quantiles: Sequence[float] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    d_model: int = 128,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    use_known_future: bool = True,
    crossing_penalty_weight: float = 0.0,
) -> QuantileTransformerConfig:
    d_enc_in = len(prepared.feature_cols["enc"])
    d_dec_in = len(prepared.feature_cols["dec"]) if use_known_future else 0
    return QuantileTransformerConfig(
        horizon=horizon,
        quantiles=quantiles,
        d_enc_in=d_enc_in,
        d_dec_in=d_dec_in,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_known_future=use_known_future,
        crossing_penalty_weight=crossing_penalty_weight,
    )


# -----------------------------
# Training step compatible with your loaders
# -----------------------------
def train_one_epoch(
    model: QuantileEncoderDecoderTransformer,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        if len(batch) == 2:
            x_hist, y_true = batch
            x_fut = None
        else:
            x_hist, x_fut, y_true = batch

        x_hist = x_hist.to(device)
        y_true = y_true.to(device)
        if x_fut is not None:
            x_fut = x_fut.to(device)

        yhat_q = model(x_hist, x_fut)
        loss = model.loss(y_true, yhat_q)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += float(loss.detach().cpu())
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: QuantileEncoderDecoderTransformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0

    for batch in loader:
        if len(batch) == 2:
            x_hist, y_true = batch
            x_fut = None
        else:
            x_hist, x_fut, y_true = batch

        x_hist = x_hist.to(device)
        y_true = y_true.to(device)
        if x_fut is not None:
            x_fut = x_fut.to(device)

        yhat_q = model(x_hist, x_fut)
        loss = model.loss(y_true, yhat_q)

        total += float(loss.detach().cpu())
        n += 1

    return total / max(n, 1)
