"""Time one warm pass from a released feature trial through final lexical selection."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .assets import sha256
from .benchmark import native_decoder
from .data import load_split_manifest, read_trial, write_json
from .evaluation import _by_id, _selection
from .inference import load_config, load_json, read_jsonl
from .model import gaussian_kernel, load_encoder, smooth_valid
from .resources import ResourceMonitor
from .v4_rate import RateEncoder, SPECS


@torch.inference_mode()
def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an allocated GPU")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    records = load_json(cfg["data_manifest"])
    split = load_split_manifest(cfg["split_manifest"], records)
    tune = _selection(_by_id(records, "trial"), split, "dev_tune")
    ordered = sorted(tune, key=lambda r: (r["input_length"], r["trial_id"]))
    chosen = [ordered[i] for i in np.linspace(0, len(ordered)-1, 24, dtype=int)]
    # Vary the request order between passes identically for every model.
    np.random.default_rng(40 + args.pass_index).shuffle(chosen)
    torch.set_num_threads(4)
    state = {"status": "started", "condition": args.condition, "pass_index": args.pass_index,
             "changed_replays": 0}
    write_json(out / "config.resolved.yaml", {**state, "official": cfg["checkpoint_hash"],
               "beam": 50, "nbest": args.nbest, "lm_weight": args.lm_weight,
               "trial_ids": [r["trial_id"] for r in chosen], "split_hash": split["manifest_hash"]})
    try:
        with ResourceMonitor(out / "resources.json", max_seconds=420) as monitor:
            start = perf_counter()
            official, up = load_encoder(cfg["checkpoint"], cfg["model_args"], "cpu", cfg["checkpoint_hash"])
            if official.contract_hash != cfg["contract_hash"]:
                raise ValueError("Official contract mismatch")
            tr = up["dataset"]["data_transforms"]
            kernel = gaussian_kernel(tr["smooth_kernel_std"], tr["smooth_kernel_size"])
            provenance = {"benchmark": sha256(__file__), "model": cfg["checkpoint_hash"],
                          "lexicon": sha256(args.lexicon), "lm": sha256(args.kenlm_model)}
            if args.condition == "official":
                model = official.eval().cuda()
                kernel = kernel.cuda()
                previous = args.study_dir / "predictions.jsonl"
                def forward(x, day):
                    return model(smooth_valid(x, kernel), [day])
            else:
                root = args.study_dir / args.condition
                previous = root / "predictions.jsonl"
                metrics = load_json(root / "metrics.json")
                path = root / "checkpoints/last.pt"
                if metrics["status"] != "complete" or sha256(path) != metrics["checkpoint_sha256"]:
                    raise ValueError("A verified completed checkpoint is required")
                model = RateEncoder(official, args.condition, kernel)
                del official
                state_dict = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if (state_dict["condition"] != args.condition or model.contract != metrics["contract"]
                    or state_dict["step"] != metrics["updates"] or state_dict["source_sha256"] != cfg["checkpoint_hash"]):
                    raise ValueError("Trained model identity mismatch")
                model.load_delta(state_dict["trainable"])
                del state_dict
                model.eval().cuda()
                forward = model
                provenance["delta"] = metrics["checkpoint_sha256"]
            expected = {i: r["text"] for i,r in _by_id(read_jsonl(previous), "saved prediction").items()}
            if set(expected) != {r["trial_id"] for r in tune}:
                raise ValueError("Expected complete saved tuning predictions for timing verification")
            write_json(out / "provenance.json", provenance)
            torch.cuda.synchronize()
            state["cold_model_verify_and_load_s"] = perf_counter() - start
            with native_decoder(args, 50, out / "decoder.log") as (decode, setup):
                state["decoder_startup"] = setup
                with (out / "predictions.jsonl").open("x", buffering=1) as stream:
                    for i, row in enumerate([chosen[0], *chosen]):
                        monitor.check()
                        state["active_request"] = {"trial_id": row["trial_id"], "warmup": i == 0}
                        torch.cuda.synchronize()
                        t0 = perf_counter()
                        x = torch.from_numpy(read_trial(row)["features"]).cuda().unsqueeze(0)
                        logits = forward(x, row["day_index"])[0].float().cpu().numpy()
                        t1 = perf_counter()
                        cands = decode(row["trial_id"], logits)["candidates"]
                        text = cands[0]["text"] if cands else ""
                        torch.cuda.synchronize()
                        dt = perf_counter() - t0
                        matched = text == expected[row["trial_id"]]
                        if i and not matched:
                            state["changed_replays"] += 1
                        stream.write(json.dumps({"trial_id": row["trial_id"], "warmup": i == 0,
                            "text": text, "status": "ok" if text else "empty", "raw_frames": row["input_length"],
                            "matches_training_run_replay": matched,
                            "emission_rows": len(logits), "feature_read_through_encoder_s": t1-t0,
                            "elapsed_s": dt, "rtf": dt / (.020 * row["input_length"])}) + "\n")
                        del state["active_request"]
                # Keep the worker alive while sampling the complete resident inference tree.
                monitor.sample()
                monitor.check()
        if load_json(out / "resources.json")["outcome"] != "complete":
            raise RuntimeError("Budget reached")
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="incomplete", error=repr(exc))
        raise
    finally:
        write_json(out / "metrics.json", state)
        (out / "report.md").write_text(f"# {args.condition}, timing pass {args.pass_index}\n\n"
            f"Status {state['status']}. Each measured request reads one trial and returns final text.\n")


if __name__ == "__main__":
    from .v4_dense_clock import register
    register()
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("output", "config", "decoder-python", "decoder-module-path", "lexicon", "kenlm-model"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--condition", required=True, choices=["official", *SPECS])
    p.add_argument("--pass-index", required=True, type=int, choices=(0, 1, 2))
    p.add_argument("--study-dir", type=Path)
    p.set_defaults(nbest=10, lm_weight=2., beam_threshold=25., start_timeout=120., request_timeout=30.)
    run(p.parse_args())
