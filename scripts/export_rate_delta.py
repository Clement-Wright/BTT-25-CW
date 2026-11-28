"""Export trainable weights from a verified, completed rate-study endpoint."""
import argparse
import json
from pathlib import Path

import torch

from btt25.assets import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    metrics = json.loads((args.study/"metrics.json").read_text())
    source = args.study/"checkpoints/last.pt"
    digest = sha256(source)
    if metrics["status"] != "complete" or metrics["checkpoint_sha256"] != digest:
        raise ValueError("Incomplete or unverified training checkpoint")
    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if not (state["condition"] == metrics["contract"]["condition"] == "two_rate"
            and state["step"] == metrics["updates"] == 2048):
        raise ValueError("Expected the registered two-rate endpoint")
    payload = {k:state[k] for k in ("source_sha256","condition","step")}
    # GRU parameters can share storage with the frozen lower layers. Clone
    # each selected tensor so serialization writes only trainable values.
    payload["trainable"] = {k:v.detach().clone(memory_format=torch.contiguous_format)
                            for k,v in state["trainable"].items()}
    payload.update(contract=metrics["contract"], training_checkpoint_sha256=digest)
    args.output.mkdir(parents=True,exist_ok=False)
    out = args.output/"two_rate.pt"
    torch.save(payload, out)
    tensor_bytes = sum(v.numel() * v.element_size() for v in payload["trainable"].values())
    if out.stat().st_size > tensor_bytes + 1024**2:
        raise ValueError("Serialized delta contains more than the selected tensors")
    receipt = {"delta_sha256":sha256(out), "delta_bytes":out.stat().st_size,
               "source_checkpoint_sha256":digest, "parent_sha256":payload["source_sha256"],
               "updates":state["step"], "trainable_parameters":metrics["trainable_parameters"]}
    (args.output/"delta.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
