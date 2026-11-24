"""State-compatible implementation of the published T15 GRU method.

The upstream source is acquired separately for parity checks; it is not vendored.
"""
from pathlib import Path
import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from .assets import sha256
from .data import TOKENS, object_hash

OFFICIAL_CHECKPOINT_SHA256 = "47969ef1de6ba1510e685a02fadca27b7f40aac0b9fc033c8f2e9e0cec8933d1"
OFFICIAL_ARGS_SHA256 = "405695a9a928a78ddd6f493e10ae1f376b81d2d52cc2a38de225f15fe9011785"


def gaussian_kernel(std=2, nominal_size=100):
    impulse = np.zeros(nominal_size, dtype=np.float32)
    impulse[nominal_size // 2] = 1
    kernel = gaussian_filter1d(impulse, std)
    kernel = kernel[kernel > 0.01]
    if len(kernel) == 0:
        raise ValueError("Empty Gaussian kernel")
    return torch.from_numpy(kernel / kernel.sum())


def smooth_valid(x, kernel):
    if x.shape[1] < kernel.numel():
        raise ValueError("Trial shorter than effective smoothing kernel")
    weights = kernel.to(device=x.device, dtype=x.dtype).view(1, 1, -1).expand(x.shape[2], 1, -1).contiguous()
    return F.conv1d(x.transpose(1, 2), weights, groups=x.shape[2]).transpose(1, 2)


def output_length(raw_length, effective_kernel=9, patch_size=14, patch_stride=4):
    smoothed = raw_length - effective_kernel + 1
    if smoothed < patch_size:
        raise ValueError("Trial too short after valid smoothing and patching")
    return (smoothed - patch_size) // patch_stride + 1


class OfficialGRU(nn.Module):
    def __init__(self, neural_dim, n_units, n_days, n_classes, n_layers=5,
                 patch_size=14, patch_stride=4, rnn_dropout=0.4, input_dropout=0.2):
        super().__init__()
        self.neural_dim, self.n_units, self.n_days = neural_dim, n_units, n_days
        self.n_classes, self.n_layers = n_classes, n_layers
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.day_weights = nn.ParameterList([nn.Parameter(torch.eye(neural_dim)) for _ in range(n_days)])
        self.day_biases = nn.ParameterList([nn.Parameter(torch.zeros(1, neural_dim)) for _ in range(n_days)])
        self.day_layer_dropout = nn.Dropout(input_dropout)
        self.gru = nn.GRU(neural_dim * patch_size, n_units, n_layers, batch_first=True, dropout=rnn_dropout)
        self.out = nn.Linear(n_units, n_classes)
        self.h0 = nn.Parameter(torch.zeros(1, 1, n_units))

    def encode(self, features, day_indices):
        if features.ndim != 3 or features.shape[-1] != self.neural_dim:
            raise ValueError("Invalid input feature contract")
        days = [int(d) for d in day_indices]
        if len(days) != features.shape[0] or any(d < 0 or d >= self.n_days for d in days):
            raise ValueError("Unknown or inconsistent day index")
        weights = torch.stack([self.day_weights[d] for d in days])
        biases = torch.stack([self.day_biases[d] for d in days])
        transformed = self.day_layer_dropout(F.softsign(features.bmm(weights) + biases))
        patches = transformed.unfold(1, self.patch_size, self.patch_stride).permute(0, 1, 3, 2)
        patches = patches.reshape(features.shape[0], patches.shape[1], -1)
        initial = self.h0.expand(self.n_layers, features.shape[0], self.n_units).contiguous()
        return self.gru(patches, initial)[0]

    def forward(self, features, day_indices):
        return self.out(self.encode(features, day_indices))


def config_contract(args):
    model, dataset = args["model"], args["dataset"]
    transforms = dataset["data_transforms"]
    if model["n_input_features"] != 512 or dataset["n_classes"] != len(TOKENS):
        raise ValueError("Unsupported feature/vocabulary dimensions")
    if model.get("bidirectional", False) or not transforms.get("smooth_data"):
        raise ValueError("Checkpoint does not specify the official unidirectional smoothed path")
    kernel = gaussian_kernel(transforms["smooth_kernel_std"], transforms["smooth_kernel_size"])
    return {"model_family": "official_t15_gru", "model": {k: model[k] for k in ("n_input_features", "n_units", "n_layers", "patch_size", "patch_stride", "rnn_dropout")}, "input_dropout": model["input_network"]["input_layer_dropout"], "tokens": TOKENS, "blank_id": 0, "boundary_id": 40, "sessions": dataset["sessions"], "execution": {"dtype": "float32", "allow_tf32": False}, "features": {"dimensions": 512, "bin_ms": 20, "normalization": "released preceding-20-trial z-score; no additional normalization", "smoothing_padding": "valid", "kernel": kernel.tolist(), "kernel_sha256": hashlib.sha256(kernel.numpy().tobytes()).hexdigest()}, "prior_exposure": "All 45 known sessions; best checkpoint selected on official validation. Local report split holds out new choices only."}


def safe_load(path):
    # Official SHA-verified archive contains NumPy scalar metrics. Allow only
    # these scalar constructors; arbitrary pickle execution stays disabled.
    from numpy._core.multiarray import scalar
    allowed = [(scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def unwrap_compiled_state(state):
    compiled = [key.startswith("_orig_mod.") for key in state]
    if any(compiled) and not all(compiled):
        raise ValueError("Mixed compiled and uncompiled checkpoint keys")
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()} if all(compiled) else state


def load_encoder(checkpoint_path, args_path, device="cpu", expected_sha256=None):
    checkpoint_path, args_path = Path(checkpoint_path), Path(args_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint required: {checkpoint_path}")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    contract = config_contract(args)
    digest = sha256(checkpoint_path)
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("Checkpoint SHA256 mismatch")
    checkpoint = safe_load(checkpoint_path)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no documented model_state_dict")
    if "contract" in checkpoint and checkpoint["contract"] != contract:
        raise ValueError("Checkpoint token/day/preprocessing contract mismatch")
    if "contract" not in checkpoint and (digest != OFFICIAL_CHECKPOINT_SHA256 or sha256(args_path) != OFFICIAL_ARGS_SHA256):
        raise ValueError("Unproven checkpoint/args pairing: require exact official artifacts or saved adaptation contract")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = args["model"]
    model = OfficialGRU(512, m["n_units"], len(contract["sessions"]), 41, m["n_layers"], m["patch_size"], m["patch_stride"], m["rnn_dropout"], contract["input_dropout"])
    state = unwrap_compiled_state(checkpoint["model_state_dict"])
    model.load_state_dict(state, strict=True)
    # No optimizer state is retained for inference.
    del checkpoint
    model.eval().to(device)
    model.contract = contract
    model.contract_hash = object_hash(contract)
    model.checkpoint_hash = digest
    model.load_transform = "Strip uniform torch.compile _orig_mod. prefix, then strict state loading"
    return model, args
