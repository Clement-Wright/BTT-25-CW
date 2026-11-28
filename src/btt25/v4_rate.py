"""T15 encoders with explicit acquisition, patch, recurrence and emission clocks."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import smooth_valid


@dataclass(frozen=True)
class RateSpec:
    grid_ms: int = 20
    patch: int = 14
    stride: int = 2
    family: str = "gru"
    front: str = "official"


SPECS = {
    "gru80": RateSpec(stride=4),
    "gru40": RateSpec(),
    "two_rate": RateSpec(family="two_rate"),
    "patch10": RateSpec(patch=10),
    "patch20": RateSpec(patch=20),
    "bin40": RateSpec(grid_ms=40, patch=7, stride=1),
    "bin10_hold": RateSpec(grid_ms=10, patch=28, stride=4),
    "adaptive_filter": RateSpec(front="adaptive_filter"),
    "activation_only": RateSpec(front="activation_only"),
    "transform_only": RateSpec(front="transform_only"),
    "input_transform": RateSpec(front="input_transform"),
    "causal_conv": RateSpec(family="causal_conv"),
}


def geometry(name, frames):
    spec = SPECS[name]
    bins = frames * 20 // spec.grid_ms
    taps = {10: 17, 20: 9, 40: 5}[spec.grid_ms]
    return max(0, (bins - taps + 1 - spec.patch) // spec.stride + 1)


class HistoryBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.pad = dilation * 2
        self.norm = nn.LayerNorm(width)
        self.history = nn.Conv1d(width, width, 3, groups=width, dilation=dilation)
        self.mix = nn.Linear(width, width * 2)

    def forward(self, features):
        x = self.norm(features).transpose(1, 2)
        x = self.history(F.pad(x, (self.pad, 0))).transpose(1, 2)
        return features + F.glu(self.mix(x))


class RateEncoder(nn.Module):
    """The supplied official module is owned by this encoder. Input is one unpadded trial."""

    def __init__(self, official_cpu, condition, kernel_cpu):
        super().__init__()
        self.condition = condition
        self.spec = spec = SPECS[condition]
        self.contract = {"condition": condition, **asdict(spec), "acquisition_ms": 20,
                         "synthetic_held_grid": spec.grid_ms == 10, "classes": official_cpu.n_classes,
                         "blank": 0, "parent_contract": getattr(official_cpu, "contract_hash", None)}
        taps = {10: 17, 20: 9, 40: 5}[spec.grid_ms]
        if taps == 9:
            kernel = kernel_cpu
        else:
            t = (torch.arange(taps, dtype=kernel_cpu.dtype) - taps // 2) * spec.grid_ms
            kernel = torch.exp(-.5 * (t / 40).square())
            kernel /= kernel.sum()
        self.register_buffer("kernel", kernel.detach().clone())
        self.base_patch = official_cpu.patch_size
        self.days_w = official_cpu.day_weights.requires_grad_(False)
        self.days_b = official_cpu.day_biases.requires_grad_(False)
        channels, hidden = official_cpu.neural_dim, official_cpu.n_units
        classes = official_cpu.n_classes
        if spec.family == "causal_conv":
            width = 192
            self.projection = nn.Linear(channels * 2, width)
            self.history = nn.Sequential(*(HistoryBlock(width, d) for d in (1, 2, 4, 8, 16, 32)))
            self.out = nn.Linear(width, classes)
        else:
            self.h0 = official_cpu.h0.requires_grad_(False)
            self.gru = official_cpu.gru.requires_grad_(False)
            self.gru.dropout = 0.
            # Train two recurrent layers for every GRU condition at its declared clock.
            for name, parameter in self.gru.named_parameters():
                layer = int(name.rsplit("_l", 1)[1])
                parameter.requires_grad_(layer >= self.gru.num_layers - 2)
            self.out = official_cpu.out.requires_grad_(True)
        if spec.family == "two_rate":
            self.mid = nn.Sequential(nn.Linear(hidden + channels, 96), nn.SiLU(), nn.Linear(96, classes))
            nn.init.zeros_(self.mid[-1].weight)
            nn.init.zeros_(self.mid[-1].bias)
        if spec.front == "adaptive_filter":
            self.filter_offsets = nn.Parameter(torch.zeros(channels, taps))
            self.sharpness = nn.Parameter(torch.zeros(channels))
        if spec.front in {"input_transform", "transform_only"}:
            self.project_in = nn.Linear(channels, 16, bias=False)
            self.project_out = nn.Linear(16, channels, bias=False)
            nn.init.zeros_(self.project_out.weight)
        if spec.front in {"input_transform", "activation_only"}:
            self.activation_mix = nn.Parameter(torch.zeros(channels))

    def output_length(self, frames):
        return geometry(self.condition, frames)

    def trainable_state(self):
        return {name: p.detach() for name, p in self.named_parameters() if p.requires_grad}

    def load_delta(self, state):
        target = self.trainable_state()
        if target.keys() != state.keys():
            raise ValueError("Checkpoint trainable keys differ from the encoder")
        with torch.no_grad():
            for name, param in target.items():
                value = state[name]
                if value.shape != param.shape or value.dtype != param.dtype:
                    raise ValueError(f"Checkpoint parameter contract mismatch: {name}")
                param.copy_(value)

    def smooth(self, x):
        if self.spec.grid_ms == 40:
            n = x.shape[1] // 2
            x = x[:, :2*n].reshape(x.shape[0], n, 2, x.shape[-1]).mean(dim=2)
        elif self.spec.grid_ms == 10:
            x = x.repeat_interleave(2, dim=1)
        if self.spec.front != "adaptive_filter":
            return smooth_valid(x, self.kernel)
        # Transients may prefer narrow support. Start with the exact official FIR.
        views = x.unfold(1, len(self.kernel), 1)
        activity = (views[..., -1] - views[..., 0]).abs().clamp(max=6)
        radius = x.new_tensor([4, 3, 2, 1, 0, 1, 2, 3, 4]) / 4
        logits = self.kernel.log() + self.filter_offsets - (
            activity * self.sharpness.tanh()).unsqueeze(-1) * radius
        return (views * logits.softmax(-1)).sum(-1)

    def forward(self, x, day):
        if x.ndim != 3 or x.shape[0] != 1 or not 0 <= day < len(self.days_w):
            raise ValueError("Require one unpadded trial and its checkpoint day")
        if self.output_length(x.shape[1]) < 1:
            raise ValueError("Trial too short for the chosen smoothing and patch")
        x = self.smooth(x) @ self.days_w[day] + self.days_b[day]
        if self.spec.front in {"input_transform", "transform_only"}:
            x = x + self.project_out(self.project_in(x))
        if self.spec.front in {"input_transform", "activation_only"}:
            y = F.softsign(x)
            x = y + self.activation_mix.tanh() * (x.tanh() - y)
        else:
            x = F.softsign(x)
        patches = x.unfold(1, self.spec.patch, self.spec.stride).transpose(-1, -2)
        if self.spec.family == "causal_conv":
            # Level across the patch plus its most recent change, then 127 output
            # positions of causal context through dilated depthwise convolutions.
            recent = patches[..., -2:, :].mean(-2) - patches[..., -4:-2, :].mean(-2)
            features = torch.cat((patches.mean(-2), recent), dim=-1)
            return self.out(self.history(self.projection(features)))
        recurrent = patches[:, ::2] if self.spec.family == "two_rate" else patches
        if self.spec.patch != self.base_patch:
            b, t, p, c = recurrent.shape
            mapped = F.interpolate(recurrent.reshape(b*t, p, c).transpose(1, 2),
                                   size=self.base_patch, mode="linear", align_corners=True)
            recurrent = mapped.transpose(1, 2).reshape(b, t, self.base_patch, c)
        initial = self.h0.expand(self.gru.num_layers, 1, self.gru.hidden_size).contiguous()
        hidden = self.gru(recurrent.flatten(-2), initial)[0]
        anchor = self.out(hidden)
        if self.spec.family != "two_rate":
            return anchor
        odd = patches[:, 1::2]
        change = odd[..., -2:, :].mean(-2) - odd[..., -4:-2, :].mean(-2)
        n = odd.shape[1]
        between = anchor[:, :n] + self.mid(torch.cat((hidden[:, :n], change), dim=-1))
        pairs = torch.stack((anchor[:, :n], between), dim=2).flatten(1, 2)
        return torch.cat((pairs, anchor[:, n:]), dim=1)
