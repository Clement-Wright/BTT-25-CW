"""Screen T15 temporal representations or pilot matched GRU adaptation on CUDA."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from .alternative_encoders import ctc_for_trial
from .assets import sha256
from .benchmark import native_decoder
from .ctc import ctc_minimum_frames, greedy_ctc
from .data import attach_corpus_metadata, load_split_manifest, read_trial, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .inference import load_config, load_json
from .model import gaussian_kernel, load_encoder, output_length
from .resources import BudgetExceeded, ResourceMonitor
from .temporal_features import TemporalFeatures
from .train_v2 import trial_order
from .training import atomic_checkpoint


def run(args):
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA GPU is required")
    if args.pilot_seconds and not 900 <= args.pilot_seconds <= 1800:
        raise ValueError("Use a sustained pilot of 900–1800 seconds")
    if not 0 < args.max_seconds <= 28200:
        raise ValueError("Maximum application time is 28200 seconds")
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(20261127)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = load_json(config["data_manifest"])
    if args.corpus_metadata:
        records = attach_corpus_metadata(records, args.corpus_metadata)
    split = load_split_manifest(config["split_manifest"], records)
    tuning = _selection(_by_id(records, "trial"), split, "dev_tune")
    longest = sorted(tuning, key=lambda r: r["input_length"])
    subset = [longest[i] for i in np.linspace(0, len(longest) - 1, 12, dtype=int)]
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {"arguments": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()}, "split_hash": split["manifest_hash"],
                "checkpoint": config["checkpoint_hash"], "official_contract": config["contract_hash"],
                "seed": 20261127, "trainable": "phone head and optional fusion" if args.pilot_seconds else "none",
                "matched_updates": 256 if args.pilot_seconds else None,
                "gru_training_dropout": 0. if args.pilot_seconds else None,
                "hashes": {str(p): sha256(p) for p in (Path(__file__),
                    Path(__file__).with_name("temporal_features.py"), args.config, args.lexicon,
                    args.kenlm_model, Path(config["data_manifest"]))}}
    write_json(args.output / "config.resolved.yaml", identity)
    write_json(args.output / "provenance.json", identity["hashes"])
    outcome = {"status": "started", "updates": 0}
    started = time.monotonic()
    try:
        with ResourceMonitor(args.output / "resources.json", max_seconds=args.max_seconds) as monitor:
            model, upstream = load_encoder(config["checkpoint"], config["model_args"], "cuda", config["checkpoint_hash"])
            if model.contract_hash != config["contract_hash"]:
                raise ValueError("Official input contract mismatch")
            transforms = upstream["dataset"]["data_transforms"]
            kernel = gaussian_kernel(transforms["smooth_kernel_std"], transforms["smooth_kernel_size"])
            model.patch_stride = args.stride
            model.eval().requires_grad_(False)
            model.out.requires_grad_(bool(args.pilot_seconds))
            if args.pilot_seconds:
                # cuDNN needs a training-mode GRU for gradients to the adapter.
                model.gru.dropout = 0.
                model.gru.train()
            view = TemporalFeatures(kernel, model.neural_dim, args.method).cuda()

            def emissions(record, features):
                x = torch.from_numpy(features).cuda().unsqueeze(0)
                return model(view(x), [record["day_index"]])[0]

            @torch.inference_mode()
            def predict(allocation, decode, filename):
                totals, groups = Counter(), {}
                with (args.output / filename).open("w", encoding="utf-8", buffering=1) as stream:
                    for row in allocation:
                        monitor.check()
                        trial = row["trial_id"]
                        features = read_trial(row)["features"]
                        logits = emissions(row, features).cpu().numpy()
                        expected_raw = (len(features) // 4 * 4 if args.method == "coarse-80" else
                                        len(features) // 2 * 2 if args.method == "coarse-40" else len(features))
                        if len(logits) != output_length(expected_raw, len(kernel), model.patch_size, args.stride):
                            raise ValueError(f"Unexpected emission length on {trial}")
                        phones = greedy_ctc(logits.argmax(-1).tolist())
                        candidates = decode(trial, logits)["candidates"]
                        text = candidates[0]["text"] if candidates else ""
                        label = reference_only(row)
                        word = edit_counts(normalize_text(label["reference_text"]).split(), normalize_text(text).split())
                        phone = edit_counts(label["phoneme_ids"], phones)
                        counts = Counter(trials=1, word_errors=word["errors"], words=word["reference_length"],
                                         phone_errors=phone["errors"], phones=phone["reference_length"],
                                         empty=int(not text), frames=len(logits))
                        totals.update(counts)
                        groups.setdefault(row.get("corpus_id") or "unclassified", Counter()).update(counts)
                        stream.write(json.dumps({"trial_id": trial, "status": "ok" if text else "empty",
                                                 "text": text, "phones": phones, "counts": counts}) + "\n")
                return {"counts": totals, "by_corpus": groups, "per": totals["phone_errors"] / totals["phones"],
                        "wer": totals["word_errors"] / totals["words"]}

            with native_decoder(args, args.beam, args.output / "decoder.log") as (decode, setup):
                outcome["decoder"] = setup
                if args.pilot_seconds:
                    # Select using the coarsest condition and common stride-4 floor.
                    # Keep the same IDs for every matched pilot.
                    eligible = []
                    for row in records:
                        monitor.check()
                        if row["split"] != "train" or row["input_length"] < 24:
                            continue
                        shortest = (row["input_length"] // 4) * 4
                        if output_length(shortest) >= ctc_minimum_frames(reference_only(row)["phoneme_ids"]):
                            eligible.append(row)
                    if not eligible:
                        raise ValueError("No eligible training trials")
                    write_json(args.output / "train_membership.json", [r["trial_id"] for r in eligible])
                    parameters = [*model.out.parameters(), *view.parameters()]
                    outcome["trainable_parameters"] = sum(p.numel() for p in parameters)
                    optimizer = torch.optim.AdamW(parameters, lr=0.0003, weight_decay=0.001)
                    begin = saved = time.monotonic()

                    def save(name="last"):
                        atomic_checkpoint(args.output / f"checkpoints/{name}.pt", {
                            "model_state_dict": model.out.state_dict(), "fusion_state_dict": view.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(), "identity": identity,
                            "step": outcome["updates"], "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all()})

                    with (args.output / "training.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
                        try:
                            for step, row in enumerate(trial_order(eligible), 1):
                                monitor.check()
                                item = read_trial(row, include_labels=True)
                                optimizer.zero_grad(set_to_none=True)
                                logits = emissions(row, item["features"])
                                loss = ctc_for_trial(logits.float().log_softmax(-1), item["phoneme_ids"])
                                if not torch.isfinite(loss):
                                    raise ValueError("Nonfinite CTC loss")
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                                optimizer.step()
                                outcome["updates"] = step
                                stream.write(json.dumps({"step": step, "loss": float(loss.detach())}) + "\n")
                                if step == 256:
                                    save("matched-256")
                                if time.monotonic() - saved >= 120:
                                    save()
                                    outcome["pilot_validation"] = predict(subset, decode, "pilot_predictions.jsonl")
                                    write_json(args.output / "metrics.json", outcome)
                                    saved = time.monotonic()
                                if time.monotonic() - begin >= args.pilot_seconds:
                                    break
                        finally:
                            save()
                    outcome["sustained_seconds"] = time.monotonic() - begin
                    outcome["checkpoint_sha256"] = sha256(args.output / "checkpoints/last.pt")
                outcome.update(predict(tuning, decode, "predictions.jsonl"))
                if args.pilot_seconds:
                    matched = args.output / "checkpoints/matched-256.pt"
                    if matched.is_file():
                        state = torch.load(matched, map_location="cpu", weights_only=True)
                        model.out.load_state_dict(state["model_state_dict"], strict=True)
                        view.load_state_dict(state["fusion_state_dict"], strict=True)
                        outcome["matched_256"] = predict(tuning, decode, "matched-256-predictions.jsonl")
                        outcome["matched_256"]["checkpoint_sha256"] = sha256(matched)
                monitor.check()
                outcome["status"] = "pilot_complete" if args.pilot_seconds else "complete"
        if load_json(args.output / "resources.json")["outcome"] != "complete":
            raise BudgetExceeded("Resource limit reached at the final sample")
    except BaseException as error:
        outcome.update(status="budget_stopped" if isinstance(error, (BudgetExceeded, KeyboardInterrupt)) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        outcome["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "metrics.json", outcome)
        (args.output / "report.md").write_text(f"# Temporal feature study\n\n"
            f"{args.method}, stride {args.stride}. Status `{outcome['status']}`.\n\n"
            f"Updates {outcome['updates']}. Tuning PER {outcome.get('per')}, WER {outcome.get('wer')}.\n",
            encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "lexicon", "kenlm-model", "decoder-python", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--decoder-module-path", type=Path)
    parser.add_argument("--corpus-metadata", type=Path)
    parser.add_argument("--method", choices=("baseline", "raw", "gaussian-1", "box-5",
                                           "coarse-40", "coarse-80", "channel-residual", "dense-residual"),
                        required=True)
    parser.add_argument("--stride", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--pilot-seconds", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--beam", type=int, default=50)
    parser.add_argument("--lm-weight", type=float, default=1.)
    parser.set_defaults(nbest=10, beam_threshold=25., start_timeout=120., request_timeout=30.)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
