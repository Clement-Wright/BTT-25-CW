"""Pinned Qwen Base scoring for explicitly allocated Gilbreth GPUs."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .assets import sha256
from .data import write_json
from .resources import ResourceMonitor


REPO = "Qwen/Qwen3-1.7B-Base"
REV = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
FILES = {
    "config.json": ("git", "78f3deadbfd84408806f8f8c46abbc9647f673d2"),
    "generation_config.json": ("git", "cbbb3133034e192527e5321b4c679154e4819ab8"),
    "tokenizer.json": ("git", "443909a61d429dff23010e5bddd28ff530edda00"),
    "tokenizer_config.json": ("git", "6a3829ee9491f36113e64df37573be81df0366f5"),
    "merges.txt": ("git", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
    "vocab.json": ("git", "4783fe10ac3adce15ac8f358ef5462739852c569"),
    "model.safetensors": ("sha256", "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841"),
}
START_END = 151643
REPO_4B = "Qwen/Qwen3-4B-Base"
REV_4B = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
FILES_4B = {name: value for name, value in FILES.items() if name != "model.safetensors"}
FILES_4B.update({
    "config.json": ("git", "df6b6d9d0d1ce7d1456641adb64a5aefe1c4c9bc"),
    "model.safetensors.index.json": ("git", "4747b0297d3109f14db49886972e3369c9a00b2a"),
    "model-00001-of-00003.safetensors": ("sha256", "4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241"),
    "model-00002-of-00003.safetensors": ("sha256", "f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826"),
    "model-00003-of-00003.safetensors": ("sha256", "c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965"),
})
SPECS = {"1.7b": (REPO, REV, FILES), "4b": (REPO_4B, REV_4B, FILES_4B)}


def cluster_guard():
    if not (os.environ.get("SLURM_JOB_ID") and os.environ.get("RCAC_SCRATCH")):
        raise RuntimeError("Use an allocated Gilbreth Slurm job for the Qwen workflow")


def verify(model_dir, files=FILES):
    fingerprints = {}
    for name, (kind, expected) in files.items():
        path = model_dir / name
        if kind == "git":
            body = path.read_bytes()
            actual = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        else:
            actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Qwen revision content mismatch for {name}")
        fingerprints[name] = {kind: actual}
    return fingerprints


def acquire(model_dir, output, size="1.7b"):
    cluster_guard()
    from huggingface_hub import snapshot_download

    output.mkdir(parents=True, exist_ok=False)
    repo, revision, files = SPECS[size]
    started = time.monotonic()
    with ResourceMonitor(output / "resources.json", max_seconds=1100) as monitor:
        snapshot_download(repo_id=repo, revision=revision, allow_patterns=list(files),
                          local_dir=model_dir, max_workers=2, token=False)
        hashes = verify(model_dir, files)
        monitor.check()
    write_json(output / "acquisition.json", {"model_id": repo, "revision": revision, "files": hashes,
                                             "elapsed_s": time.monotonic() - started})


@torch.inference_mode()
def batched_scores(model, tokenizer, texts, start_id, end_id, device, *, batch_size=8,
                   max_tokens=192, add_eos=True):
    """Teacher-force exact candidate strings, truncating none and masking padding."""
    results = []
    for offset in range(0, len(texts), batch_size):
        chunk = texts[offset:offset + batch_size]
        tokenized = tokenizer(chunk, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        labels = [ids + ([end_id] if add_eos else []) for ids in tokenized]
        lengths = [len(ids) for ids in labels]
        if max(lengths, default=0) > max_tokens:
            raise ValueError("Some exact candidates exceed the configured token cap")
        width = max(lengths, default=0)
        if not width:
            results.extend((0.0, 0) for _ in chunk)
            continue
        inputs = [[start_id, *ids[:-1], *([end_id] * (width - len(ids)))] if ids
                  else [start_id] + [end_id] * (width - 1) for ids in labels]
        targets = [ids + [end_id] * (width - len(ids)) for ids in labels]
        x = torch.tensor(inputs, dtype=torch.long, device=device)
        target = torch.tensor(targets, dtype=torch.long, device=device)
        keep = (torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None])
        # An otherwise-empty prefix occasionally shares a padded batch.
        attention = keep.clone()
        attention[:, 0] = True
        logits = model(input_ids=x, attention_mask=attention.long(), use_cache=False).logits
        valid_logits = logits[keep].float()
        chosen = valid_logits.gather(1, target[keep].unsqueeze(1)).squeeze(1)
        token_scores = chosen - torch.logsumexp(valid_logits, dim=-1)
        if not torch.isfinite(token_scores).all():
            raise ValueError("Nonfinite candidate likelihood")
        scores = torch.zeros_like(target, dtype=torch.float32)
        scores[keep] = token_scores
        results.extend(zip(scores.sum(1).tolist(), lengths, strict=True))
    return results


class QwenScorer:
    def __init__(self, model_dir, size="1.7b"):
        cluster_guard()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Qwen Base requires the allocated BF16 CUDA device")
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

        started = time.monotonic()
        repo, revision, files = SPECS[size]
        fingerprints = verify(Path(model_dir), files)
        settings = {"local_files_only": True, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, **settings)
        self.device = torch.device("cuda:0")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, **settings, use_safetensors=True, dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map={"": 0}, attn_implementation="sdpa",
        ).eval()
        configuration = self.model.config
        if (configuration.model_type != "qwen3" or configuration.bos_token_id != START_END
                or configuration.eos_token_id != START_END
                or self.tokenizer.eos_token_id != START_END
                or self.tokenizer.pad_token_id != START_END):
            raise ValueError("Qwen model/tokenizer identity mismatch")
        if ({p.device for p in self.model.parameters()} != {self.device}
                or {p.dtype for p in self.model.parameters()} != {torch.bfloat16}):
            raise RuntimeError("Unexpected model placement or precision")
        torch.cuda.synchronize(self.device)
        self.metadata = {"model_id": repo, "revision": revision, "files": fingerprints,
                         "device": str(self.device), "precision": "bfloat16 weights; float32 scoring",
                         "transformers": __version__, "load_seconds": time.monotonic() - started,
                         "bos_id": START_END, "eos_id": START_END, "prompt": None,
                         "text_policy": "encode exact candidate; no prepended space or chat template",
                         "pretraining_overlap": "General text pretraining overlap with trial text is unknown"}

    def score(self, texts, *, add_eos=True, batch_size=8):
        return batched_scores(self.model, self.tokenizer, texts, START_END, START_END,
                              self.device, add_eos=add_eos, batch_size=batch_size)
