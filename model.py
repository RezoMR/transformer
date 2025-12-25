# Quantile Encoder–Decoder Transformer for Direct Multi-Horizon Forecasting
# horizon = 96, quantiles = 0.1..0.9, loss = pinball (quantile regression)

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence

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

    Returns scalar loss (default) or unreduced tensor if reduction="none".
    """
    if y.dim() == 2:
        y = y.unsqueeze(-1)  # (B,H,1)
    if yhat_q.dim() != 3:
        raise ValueError(f"yhat_q must be (B,H,Q), got {tuple(yhat_q.shape)}")
    if quantiles.dim() != 1 or quantiles.numel() != yhat_q.size(-1):
        raise ValueError("quantiles must be (Q,) matching last dim of yhat_q")

    q = quantiles.view(1, 1, -1).to(yhat_q.device)  # (1,1,Q)
    err = y - yhat_q  # (B,H,Q)
    loss = torch.maximum(q * err, (1.0 - q) * (-err))  # (B,H,Q)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be one of: 'mean','sum','none'")


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Works with batch_first=True tensors: (B, T, D).
    """
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
        # x: (B,T,D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


# -----------------------------
# Model
# -----------------------------
@dataclass
class QuantileTransformerConfig:
    horizon: int = 96
    quantiles: Sequence[float] = tuple([0.1 * i for i in range(1, 10)])  # 0.1..0.9

    # Feature dimensions
    d_enc_in: int = 16         # number of encoder input features per timestep
    d_dec_in: int = 8          # number of decoder known-future features per timestep (can be 0)

    # Transformer sizes
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Positional encoding
    max_len: int = 4096

    # If you don't have known-future features, set d_dec_in=0
    # and the decoder will use only learned horizon queries.
    use_known_future: bool = True  # if False, ignores x_future even if provided


class QuantileEncoderDecoderTransformer(nn.Module):
    """
    Direct multi-horizon quantile forecaster.

    Inputs:
      x_hist   : (B, L, d_enc_in)   - historical sequence
      x_future : (B, H, d_dec_in)   - known future covariates (optional)

    Output:
      yhat_q   : (B, H, Q)          - quantile forecasts for each horizon step
    """
    def __init__(self, cfg: QuantileTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.horizon
        self.quantiles = torch.tensor(list(cfg.quantiles), dtype=torch.float32)
        self.Q = len(cfg.quantiles)

        # Projections
        self.enc_in_proj = nn.Linear(cfg.d_enc_in, cfg.d_model)

        # Decoder input: learned horizon queries + optional known-future covariates projection
        self.horizon_queries = nn.Parameter(torch.randn(1, cfg.horizon, cfg.d_model) * 0.02)

        if cfg.d_dec_in > 0:
            self.dec_in_proj = nn.Linear(cfg.d_dec_in, cfg.d_model)
        else:
            self.dec_in_proj = None

        # Positional encodings
        self.pos_enc = Posit_
