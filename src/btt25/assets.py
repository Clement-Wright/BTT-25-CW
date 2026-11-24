"""Streaming, checksum-verified acquisition. Assets never enter source control."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MODEL_SHA256 = "04ab2275eeb48abad9b4011dde6f7c9e5dd9f643b8a677c1ba5e06dc53bb4111"
MODEL_BYTES = 484944889
MODEL_URL = "https://huggingface.co/datasets/yw784/brain-to-text-25-competition/resolve/2f2fabacbc9af637da3bac483bf14e1231fb4e8a/t15_pretrained_rnn_baseline.zip"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def acquire_model(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.stat().st_size != MODEL_BYTES or sha256(output) != MODEL_SHA256:
            raise ValueError("Existing model archive checksum mismatch; preserve and inspect it")
        return {"status": "reused", "sha256": MODEL_SHA256, "bytes": MODEL_BYTES}
    start = time.monotonic()
    temporary = output.with_suffix(output.suffix + ".partial")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "BTT25-CW-reproducibility/0.1"})
    digest, count = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=90) as source, temporary.open("wb") as target:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            target.write(chunk)
            digest.update(chunk)
            count += len(chunk)
    if count != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256:
        raise ValueError(f"Model download failed integrity check: {count} bytes, {digest.hexdigest()}")
    temporary.replace(output)
    evidence = {"status": "downloaded", "url": MODEL_URL, "authoritative_metadata": "https://datadryad.org/api/v2/files/4150349", "sha256": digest.hexdigest(), "bytes": count, "setup_elapsed_seconds": time.monotonic() - start, "mirror": "Independent mirror; exact hash matches official Dryad asset"}
    output.with_suffix(".acquisition.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    print(json.dumps(acquire_model(parser.parse_args().output)))
