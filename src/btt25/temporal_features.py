"""Aligned temporal views of the released 20 ms features."""
import torch
from torch import nn

from .model import smooth_valid


class TemporalFeatures(nn.Module):
    """All branches retain the official nine-bin window and center."""

    def __init__(self, baseline_kernel, channels, method):
        super().__init__()
        self.method = method
        if baseline_kernel.numel() != 9:
            raise ValueError("Temporal comparisons require the official nine-bin kernel")
        coords = torch.arange(-4, 5, dtype=baseline_kernel.dtype)
        light = torch.exp(-.5 * coords.square())
        box = (coords.abs() <= 2).to(coords.dtype)
        self.register_buffer("baseline", baseline_kernel)
        self.register_buffer("light", light / light.sum())
        self.register_buffer("box", box / box.sum())
        if method == "channel-residual":
            self.residual = nn.Parameter(torch.zeros(2, channels))
        elif method == "dense-residual":
            self.residual = nn.Linear(2 * channels, channels, bias=False)
            nn.init.zeros_(self.residual.weight)
        elif method not in {"baseline", "raw", "gaussian-1", "box-5", "coarse-40", "coarse-80"}:
            raise ValueError(f"Unknown temporal feature method: {method}")

    def forward(self, x):
        if self.method.startswith("coarse-"):
            # Aggregate complete blocks of the released standardized features.
            # Repeat the block means on the existing grid to keep GRU patch weights
            # compatible. The held values carry no within-block information.
            width = 2 if self.method == "coarse-40" else 4
            nblocks = x.shape[1] // width
            x = x[:, :nblocks * width].reshape(x.shape[0], nblocks, width, x.shape[2])
            x = x.mean(dim=2).repeat_interleave(width, dim=1)
        if x.shape[1] < 9:
            raise ValueError("Trial shorter than the common alignment window")
        if self.method == "raw":
            return x[:, 4:-4]
        if self.method == "gaussian-1":
            return smooth_valid(x, self.light)
        if self.method == "box-5":
            return smooth_valid(x, self.box)
        slow = smooth_valid(x, self.baseline)
        if self.method not in {"channel-residual", "dense-residual"}:
            return slow
        residuals = (x[:, 4:-4] - slow, smooth_valid(x, self.light) - slow)
        if self.method == "channel-residual":
            return slow + residuals[0] * self.residual[0] + residuals[1] * self.residual[1]
        return slow + self.residual(torch.cat(residuals, dim=-1))
