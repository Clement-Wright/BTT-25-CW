"""Right-edge causal replay of one official T15 trial, with explicit trial resets."""

import torch
from torch.nn import functional as F

from .model import smooth_valid


class OnlineGRU:
    def __init__(self, model, kernel, day, modality="T15-released-intracortical-20ms"):
        self._start(model, kernel, day, modality, model.day_weights, model.day_biases,
                    model.patch_size, model.patch_stride)

    def _start(self, model, kernel, day, modality, weights, biases, size, stride):
        if modality != "T15-released-intracortical-20ms":
            raise ValueError("This checkpoint requires released T15 intracortical features")
        if not 0 <= day < len(weights) or model.training:
            raise ValueError("Require a known session and an evaluation-mode encoder")
        self.model, self.kernel = model, kernel
        self.weight, self.bias = weights[day], biases[day]
        self.size, self.stride = size, stride
        self.raw = self.weight.new_empty((1, 0, self.weight.shape[0]))
        self.patch = self.raw.clone()
        self.hidden = model.h0.expand(model.gru.num_layers, 1, model.gru.hidden_size).contiguous()

    @torch.inference_mode()
    def push(self, bins):
        if bins.ndim != 2 or bins.shape[1] != self.weight.shape[0] or bins.dtype != torch.float32:
            raise ValueError("Expected [bins, T15 features] in FP32")
        if not len(bins) or not torch.isfinite(bins).all():
            raise ValueError("Require finite, nonempty bins")
        self.raw = torch.cat((self.raw, bins[None]), dim=1)
        if self.raw.shape[1] < len(self.kernel):
            return None
        x = smooth_valid(self.raw, self.kernel)
        self.raw = self.raw[:, x.shape[1] :].clone()
        self.patch = torch.cat((self.patch, F.softsign(x @ self.weight + self.bias)), dim=1)
        size, stride = self.size, self.stride
        if self.patch.shape[1] < size:
            return None
        patches = self.patch.unfold(1, size, stride).permute(0, 1, 3, 2)
        self.patch = self.patch[:, patches.shape[1] * stride :].clone()
        return self._advance(patches)

    def _advance(self, patches):
        encoded, self.hidden = self.model.gru(patches.flatten(-2), self.hidden)
        return self.model.out(encoded)[0]


class OnlineTwoRate(OnlineGRU):
    def __init__(self, model, day, modality="T15-released-intracortical-20ms"):
        if model.condition != "two_rate":
            raise ValueError("Expected the trained two-rate model")
        self._start(model, model.kernel, day, modality, model.days_w, model.days_b,
                    model.spec.patch, model.spec.stride)
        self.next_is_anchor = True
        self.anchor_hidden = self.anchor_logits = None

    def _advance(self, patches):
        outputs = []
        for patch in patches.unbind(dim=1):
            if self.next_is_anchor:
                encoded, self.hidden = self.model.gru(patch.flatten(1)[:, None], self.hidden)
                self.anchor_hidden = encoded[:, -1]
                self.anchor_logits = self.model.out(self.anchor_hidden)
                output = self.anchor_logits
            else:
                change = patch[:, -2:].mean(1) - patch[:, -4:-2].mean(1)
                output = self.anchor_logits + self.model.mid(torch.cat((self.anchor_hidden, change), dim=-1))
            self.next_is_anchor = not self.next_is_anchor
            outputs.append(output)
        return torch.cat(outputs, dim=0)
