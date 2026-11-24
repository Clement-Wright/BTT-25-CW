"""Fixed, training-free likelihood scoring of existing word candidates.

For nonempty text, prepend one ASCII space to the exact candidate. Empty text
has no text tokens. Score [EOS-as-BOS, text tokens, EOS] by next-token natural
log likelihood, excluding BOS and right padding and including the final EOS.
No generation, text normalization, fitting, or reference access occurs here.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any

import torch


MODEL_ID = "distilbert/distilgpt2"
MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
_MODEL_FILES = {
    "model.safetensors", "config.json", "vocab.json", "merges.txt",
    "tokenizer.json", "tokenizer_config.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_assets(model_dir: Path) -> dict:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("fixed scorer requires a local asset manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("scorer asset model/revision does not match the fixed model")
    files = manifest.get("files", [])
    names = [item.get("file") for item in files]
    if len(set(names)) != len(names) or not _MODEL_FILES.issubset(names):
        raise ValueError("scorer asset manifest is incomplete or contains duplicate files")
    for item in files:
        name = item.get("file")
        if name not in _MODEL_FILES | {"README.md"}:
            raise ValueError("unexpected file in scorer asset manifest")
        path = model_dir / name
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise ValueError(f"scorer asset checksum mismatch: {name}")
    return {"revision": MODEL_REVISION, "manifest_sha256": _sha256(manifest_path)}


class DistilGPT2Scorer:
    """Load the pinned local model or accept explicitly supplied components.

    Loading from disk verifies the acquisition manifest, uses safetensors and
    local_files_only, and never downloads implicitly. Component injection is
    available for small deterministic fixtures and records unverified provenance.
    CUDA fallback requires allow_cpu_fallback=True and is recorded in metadata.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        device: str = "cpu",
        allow_cpu_fallback: bool = False,
        dtype: torch.dtype = torch.float32,
        model: Any = None,
        tokenizer: Any = None,
    ):
        if (model is None) != (tokenizer is None):
            raise ValueError("supply both model and tokenizer components")
        if model is not None and model_dir is not None:
            raise ValueError("choose a local model directory or supplied components")
        requested = torch.device(device)
        if requested.type not in {"cpu", "cuda"}:
            raise ValueError("scorer supports explicit CPU or CUDA devices")
        fallback = None
        if requested.type == "cuda" and not torch.cuda.is_available():
            if not allow_cpu_fallback:
                raise RuntimeError("CUDA is unavailable; enable explicit CPU fallback")
            requested_actual = torch.device("cpu")
            dtype = torch.float32
            fallback = "CUDA unavailable"
        else:
            requested_actual = requested

        verified = None
        if model is None:
            if model_dir is None:
                raise ValueError("a local model directory is required")
            model_dir = Path(model_dir)
            verified = _verify_assets(model_dir)
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            tokenizer = GPT2TokenizerFast.from_pretrained(
                str(model_dir), local_files_only=True,
            )
            model = GPT2LMHeadModel.from_pretrained(
                str(model_dir), local_files_only=True, use_safetensors=True,
                dtype=dtype, attn_implementation="eager",
            )

        self.allow_cpu_fallback = allow_cpu_fallback
        self.device = requested_actual
        self.tokenizer = tokenizer
        self.model = model
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None or model.config.eos_token_id != self.eos_token_id:
            raise ValueError("model and tokenizer EOS identities must agree")
        self.context_length = int(model.config.n_positions)
        if self.context_length < 2:
            raise ValueError("model context must contain BOS and EOS")
        try:
            self.model.to(device=self.device, dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            if self.device.type != "cuda" or not allow_cpu_fallback:
                raise
            self.device = torch.device("cpu")
            self.model.to(device=self.device, dtype=torch.float32)
            fallback = "CUDA out of memory during model placement"
        self.model.eval()
        self.model.requires_grad_(False)
        parameter = next(self.model.parameters())
        self.metadata = {
            "model_id": MODEL_ID if verified else "injected-components",
            "revision": MODEL_REVISION if verified else None,
            "asset_manifest_sha256": verified["manifest_sha256"] if verified else None,
            "asset_hashes_verified": verified is not None,
            "requested_device": str(requested),
            "device": str(self.device),
            "cpu_fallback_allowed": allow_cpu_fallback,
            "cpu_fallback_reason": fallback,
            "dtype": str(parameter.dtype).removeprefix("torch."),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "context_length": self.context_length,
            "license": "Apache-2.0" if verified else None,
            "pretraining_corpus": "OpenWebText" if verified else None,
            "pretraining_method": "knowledge distillation from GPT-2" if verified else None,
            "benchmark_text_overlap": "unknown",
            "leading_space": "one ASCII space prepended to each nonempty exact text",
            "bos_token_id": self.eos_token_id,
            "eos_token_id": self.eos_token_id,
            "include_eos_target": True,
            "score_log_base": "e",
            "context_policy": "reject if BOS + text + EOS exceeds context; no truncation",
            "default_batch_size": 1,
            "batched_timing": "batch elapsed seconds divided equally among candidates",
        }

    def _tokens(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("candidate text must be a string")
        tokens = self.tokenizer.encode(" " + text, add_special_tokens=False) if text else []
        sequence = [self.eos_token_id, *tokens, self.eos_token_id]
        if len(sequence) > self.context_length:
            raise ValueError(
                f"candidate needs {len(sequence)} tokens including BOS/EOS; "
                f"model context is {self.context_length}; refusing truncation"
            )
        return sequence

    def score(self, text: str) -> dict:
        return self.score_many([text])[0]

    def score_many(self, texts: list[str], *, batch_size: int = 1) -> list[dict]:
        """Score complete texts, with right-padding targets masked in batches."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        results = []
        for start in range(0, len(texts), batch_size):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            began = perf_counter()
            sequences = [self._tokens(text) for text in texts[start:start + batch_size]]
            width = max(map(len, sequences))
            ids = torch.full((len(sequences), width), self.eos_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, sequence in enumerate(sequences):
                ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                mask[row, :len(sequence)] = 1
            try:
                scores, counts = self._score_batch(ids, mask)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or not self.allow_cpu_fallback:
                    raise
                self.device = torch.device("cpu")
                self.model.to(device=self.device, dtype=torch.float32)
                self.metadata.update(
                    device="cpu", dtype="float32",
                    cpu_fallback_reason="CUDA out of memory during candidate scoring",
                )
                scores, counts = self._score_batch(ids, mask)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = (perf_counter() - began) / len(sequences)
            results.extend({
                "log_likelihood": score,
                "token_count": count,
                "elapsed_seconds": elapsed,
            } for score, count in zip(scores, counts, strict=True))
        return results

    @torch.inference_mode()
    def _score_batch(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple[list, list]:
        ids, mask = ids.to(self.device), mask.to(self.device)
        logits = self.model(
            ids, attention_mask=mask, use_cache=False, return_dict=True,
        ).logits
        log_probs = logits[:, :-1].float().log_softmax(dim=-1)
        token_scores = log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        valid = mask[:, 1:].bool()
        scores = token_scores.masked_fill(~valid, 0).sum(dim=-1).cpu().tolist()
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("candidate scorer returned a nonfinite likelihood")
        return scores, valid.sum(dim=-1).cpu().tolist()

    def score_candidates(self, candidates: list[dict], *, batch_size: int = 1) -> list[dict]:
        scores = self.score_many([row["text"] for row in candidates], batch_size=batch_size)
        return [dict(
            row, lm_log_likelihood=value["log_likelihood"],
            lm_token_count=value["token_count"], lm_elapsed_seconds=value["elapsed_seconds"],
        ) for row, value in zip(candidates, scores, strict=True)]


def cheap_features(candidates: list[dict], input_length: int) -> dict:
    """Reference-free compact-decoder confidence, normalized by best-text words.

    Missing a runner-up is an unmeasured margin, represented by None. 'score'
    is the primary candidate field; 'compact_score' is an explicit alias only
    when 'score' is absent. Higher compact scores must mean better candidates.
    """
    if isinstance(input_length, bool) or not isinstance(input_length, Integral) or input_length < 0:
        raise ValueError("input_length must be a nonnegative integer")
    ordered = []
    for candidate in candidates:
        raw = candidate.get("score", candidate.get("compact_score"))
        if raw is None or not math.isfinite(float(raw)):
            raise ValueError("every candidate needs a finite compact score")
        text = candidate["text"]
        if not isinstance(text, str):
            raise TypeError("candidate text must be a string")
        ordered.append((float(raw), text))
    ordered.sort(key=lambda item: (-item[0], item[1]))
    words = len(ordered[0][1].split()) if ordered else 0
    denominator = max(1, words)
    return {
        "candidate_count": len(ordered),
        "input_length": int(input_length),
        "word_count": words,
        "top_score_per_word": ordered[0][0] / denominator if ordered else None,
        "top_two_margin_per_word": (ordered[0][0] - ordered[1][0]) / denominator
        if len(ordered) >= 2 else None,
    }


def select_low_confidence(rows: list[dict], fraction: float, *, seed: int | str = 0) -> list[str]:
    """Select floor(fraction * N) IDs by ascending margin and stable hash ties.

    None margins come first because confidence cannot be measured without a
    runner-up. This helper performs no fitting and does not inspect references.
    """
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("gate fraction must be between zero and one")
    ranked, seen = [], set()
    for row in rows:
        trial_id = row["trial_id"]
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("gate trial IDs must be nonempty strings")
        if trial_id in seen:
            raise ValueError("duplicate trial ID in gate selection")
        seen.add(trial_id)
        margin = row.get("top_two_margin_per_word")
        if margin is not None and (not math.isfinite(margin) or margin < 0):
            raise ValueError("gate margin must be nonnegative and finite or None")
        tie = hashlib.sha256(f"btt25-gate-v1\0{seed}\0{trial_id}".encode()).hexdigest()
        ranked.append((float("-inf") if margin is None else margin, tie, trial_id))
    ranked.sort()
    return [item[2] for item in ranked[:math.floor(fraction * len(rows))]]
