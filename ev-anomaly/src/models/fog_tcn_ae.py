from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Remove the last 'chomp' timesteps to keep causality after padding."""
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp == 0:
            return x
        return x[:, :, :-self.chomp].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

        # init
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    def __init__(self, n_inputs: int, channels: list[int], kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        n_in = n_inputs
        for i, n_out in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(n_in, n_out, kernel_size, dilation, dropout))
            n_in = n_out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TCNAutoEncoder(nn.Module):
    """
    Input:  x [B, L, C]  (batch, seq_len, channels/features)
    Output: recon [B, L, C]
    """
    def __init__(self, n_features: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_features = int(n_features)
        self.encoder = TCNBackbone(n_features, channels, kernel_size, dropout)
        self.proj = nn.Conv1d(channels[-1], n_features, kernel_size=1)

        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,L,C] -> [B,C,L]
        x_c = x.transpose(1, 2).contiguous()
        z = self.encoder(x_c)                 # [B, hidden, L]
        recon_c = self.proj(z)                # [B, C, L]
        recon = recon_c.transpose(1, 2).contiguous()
        return recon

    @staticmethod
    def recon_error(x: torch.Tensor, recon: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
        # per-sample scalar score
        # x/recon: [B,L,C]
        err = (x - recon) ** 2
        if reduce == "mean":
            return err.mean(dim=(1, 2))
        if reduce == "sum":
            return err.sum(dim=(1, 2))
        raise ValueError(f"Unknown reduce: {reduce}")
