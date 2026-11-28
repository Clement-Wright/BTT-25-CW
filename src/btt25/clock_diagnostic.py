"""Change the recurrent and emission clocks independently on the same T15 trials."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from .assets import sha256
from .benchmark import native_decoder
from .ctc import greedy_ctc
from .data import load_split_manifest, read_trial, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .inference import load_config, load_json
from .model import gaussian_kernel, load_encoder, smooth_valid
from .resources import ResourceMonitor


def clock_views(slow, fast):
    if len(fast[::2]) != len(slow):
        raise ValueError("Native anchor counts differ")
    repeated = np.repeat(slow, 2, axis=0)[:len(fast)]
    return {"native80": slow, "repeat80_on40": repeated,
            "fast40_on80": fast[::2], "native40": fast}


@torch.inference_mode()
def run(args):
    cfg = load_config(args.config)
    records = load_json(cfg["data_manifest"])
    split = load_split_manifest(cfg["split_manifest"], records)
    trials = _selection(_by_id(records, "trial"), split, "dev_tune")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    totals = {}
    status = {"status": "started"}
    start = time.monotonic()
    write_json(out / "config.resolved.yaml", {**{k: str(v) for k, v in vars(args).items()},
               "split_hash": split["manifest_hash"], "checkpoint": cfg["checkpoint_hash"]})
    write_json(out / "provenance.json", {str(k): sha256(k) for k in
               (Path(__file__), args.config, args.lexicon, args.kenlm_model)})
    try:
        with ResourceMonitor(out / "resources.json", max_seconds=1800) as monitor:
            torch.set_num_threads(4)
            model, up = load_encoder(cfg["checkpoint"], cfg["model_args"], "cuda", cfg["checkpoint_hash"])
            if model.contract_hash != cfg["contract_hash"]:
                raise ValueError("Official contract mismatch")
            trans = up["dataset"]["data_transforms"]
            kernel = gaussian_kernel(trans["smooth_kernel_std"], trans["smooth_kernel_size"])
            with native_decoder(args, 50, out / "decoder.log") as (decode, setup), (
                    out / "predictions.jsonl").open("x", buffering=1) as stream:
                status["decoder"] = setup
                for row in trials:
                    monitor.check()
                    trial = row["trial_id"]
                    x = torch.from_numpy(read_trial(row)["features"]).cuda()[None]
                    x = smooth_valid(x, kernel)
                    model.patch_stride = 4
                    slow = model(x, [row["day_index"]])[0].cpu().numpy()
                    model.patch_stride = 2
                    fast = model(x, [row["day_index"]])[0].cpu().numpy()
                    predictions = {}
                    for condition, logits in clock_views(slow, fast).items():
                        frames = logits.argmax(-1)
                        phones = greedy_ctc(frames.tolist())
                        cands = decode(trial, logits)["candidates"]
                        predictions[condition] = {"text": cands[0]["text"] if cands else "",
                            "phones": phones, "frames": len(frames), "blank_frames": int((frames == 0).sum())}
                    ref = reference_only(row)
                    for condition, pred in predictions.items():
                        counts = totals.setdefault(condition, Counter())
                        counts.update(trials=1, frames=pred["frames"], blank_frames=pred["blank_frames"])
                        for unit, target, hypothesis in (
                            ("word", normalize_text(ref["reference_text"]).split(), normalize_text(pred["text"]).split()),
                            ("phone", ref["phoneme_ids"], pred["phones"])):
                            edits = edit_counts(target, hypothesis)
                            pred[unit + "_edits"] = edits
                            counts.update({unit + "_" + k: v for k, v in edits.items()})
                    stream.write(json.dumps({"trial_id": trial, "predictions": predictions}) + "\n")
                monitor.check()
        if load_json(out / "resources.json")["outcome"] != "complete":
            raise RuntimeError("Resource monitor reported an incomplete run")
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="incomplete", error=repr(exc))
        raise
    finally:
        status.update(elapsed_seconds=time.monotonic() - start, counts=totals)
        write_json(out / "metrics.json", status)
        (out / "report.md").write_text("# Recurrent clock diagnostic\n\n" +
            f"Status {status['status']}. All four views used the same beam-50 decoder.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "output", "lexicon", "kenlm-model", "decoder-python", "decoder-module-path"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.set_defaults(nbest=10, beam_threshold=25., lm_weight=2., start_timeout=120., request_timeout=30.)
    run(parser.parse_args())
