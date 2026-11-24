"""Separate T15 encoder contracts for the monophone controls and alternatives."""
from math import log

import torch
from torch import nn
from torch.nn import functional as F

from .ctc import ctc_minimum_frames


PHONE_COUNT = 40
SILENCE = 40


def diphone_id(previous, current):
    if not (1 <= previous <= PHONE_COUNT and 1 <= current <= PHONE_COUNT):
        raise ValueError("Diphone contexts require two nonblank T15 symbols")
    return (previous - 1) * PHONE_COUNT + current


def diphone_targets(phones):
    """Precede each change with its transition; retain every labeled self-pair.

    T15 SIL symbols in the original target remain phones. The first context is
    SIL. Leading, repeated and trailing labels are preserved without adding
    fictitious terminal phones.
    """
    previous = SILENCE
    labels = []
    for current in phones:
        transition = diphone_id(previous, current)
        if previous != current:
            labels.append(transition)
        labels.append(diphone_id(current, current))
        previous = current
    return labels


def marginal_log_probs(joint_log_probs):
    if joint_log_probs.shape[-1] != 1 + PHONE_COUNT**2:
        raise ValueError("Expected one blank and the complete 40x40 joint distribution")
    phones = joint_log_probs[..., 1:].unflatten(-1, (PHONE_COUNT, PHONE_COUNT))
    return torch.cat((joint_log_probs[..., :1], phones.logsumexp(dim=-2)), dim=-1)


class DiphoneHead(nn.Linear):
    @classmethod
    def warm_start(cls, phone_head):
        if phone_head.out_features != PHONE_COUNT + 1 or phone_head.bias is None:
            raise ValueError("Expected the official affine 41-class head")
        head = cls(phone_head.in_features, 1 + PHONE_COUNT**2,
                   device=phone_head.weight.device, dtype=phone_head.weight.dtype)
        with torch.no_grad():
            head.weight.copy_(torch.cat((phone_head.weight[:1], phone_head.weight[1:].repeat(PHONE_COUNT, 1))))
            head.bias.copy_(torch.cat((phone_head.bias[:1], phone_head.bias[1:].repeat(PHONE_COUNT) - log(PHONE_COUNT))))
        return head

    def emissions(self, hidden):
        return marginal_log_probs(self(hidden).float().log_softmax(dim=-1))


def ctc_for_trial(log_probs, phones):
    if log_probs.ndim != 2 or not phones or len(log_probs) < ctc_minimum_frames(phones):
        raise ValueError("CTC is infeasible at the true emission length, including repeated labels")
    if any(not 1 <= phone < log_probs.shape[-1] for phone in phones):
        raise ValueError("CTC targets require nonblank valid classes")
    return F.ctc_loss(log_probs[:, None], torch.tensor(phones, device=log_probs.device),
                      (len(log_probs),), (len(phones),), reduction="mean", blank=0, zero_infinity=False)


def diphone_objective(joint_logits, phones, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Curriculum weight must be in [0, 1]")
    joint = joint_logits.float().log_softmax(dim=-1)
    transition = ctc_for_trial(joint, diphone_targets(phones))
    phone = ctc_for_trial(marginal_log_probs(joint), phones)
    loss = alpha * transition + (1 - alpha) * phone
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite joint/monophone objective")
    return loss, transition, phone


class RelativeCausalBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                                          nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        encoded = self.attention_norm(x)
        x = x + self.dropout(self.attention(encoded, encoded, encoded, attn_mask=mask, need_weights=False)[0])
        return x + self.dropout(self.feed_forward(x))


class CausalT15Transformer(nn.Module):
    """Time-masked 80 ms patches, learned relative bias and local causal attention.

    Operates on released signed 20 ms feature bins with no additional smoothing
    or logarithm. Outputs become eligible after the last bin of each patch.
    Batch inference is offline and exposes no streaming interface.
    """
    def __init__(self, session_count, width=384, layers=5, heads=6, window=128, dropout=.1):
        super().__init__()
        if session_count < 1 or width % heads or window < 1:
            raise ValueError("Invalid T15 Transformer architecture")
        self.heads, self.window = heads, window
        self.day_scale = nn.Parameter(torch.ones(session_count, 512))
        self.day_bias = nn.Parameter(torch.zeros(session_count, 512))
        self.relative = nn.Parameter(torch.zeros(layers, heads, window))
        self.projection = nn.Linear(4 * 512, width)
        self.layers = nn.ModuleList(RelativeCausalBlock(width, heads, dropout) for _ in range(layers))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 41))

    def forward(self, features, days, raw_lengths):
        if (features.ndim != 3 or features.shape[-1] != 512 or len(days) != len(features)
                or len(raw_lengths) != len(features) or min(raw_lengths) < 4 or max(raw_lengths) > features.shape[1]
                or min(days) < 0 or max(days) >= len(self.day_scale)):
            raise ValueError("Invalid T15 feature, length or day identity")
        frame_count = max(raw_lengths) // 4
        day_indices = torch.as_tensor(days, dtype=torch.long, device=features.device)
        x = features[:, :4 * frame_count] * self.day_scale[day_indices, None] + self.day_bias[day_indices, None]
        x = self.projection(x.reshape(len(features), frame_count, -1))
        indices = torch.arange(frame_count, device=features.device)
        distance = indices[:, None] - indices[None, :]
        valid = (distance >= 0) & (distance < self.window)
        offsets = distance.clamp(0, self.window - 1)
        for layer, bias in zip(self.layers, self.relative, strict=True):
            mask = bias[:, offsets].masked_fill(~valid, float("-inf"))
            mask = mask[None].expand(len(features), -1, -1, -1).flatten(0, 1)
            x = layer(x, mask)
        return self.output(x), tuple(length // 4 for length in raw_lengths)
