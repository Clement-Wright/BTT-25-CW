"""Pilot and train the separate T15 encoder conditions."""
import argparse
from contextlib import ExitStack
from itertools import islice
import json
from pathlib import Path
import random
from time import monotonic

import numpy as np
import torch

from .alternative_encoders import CausalT15Transformer, DiphoneHead, ctc_for_trial, diphone_objective, diphone_targets
from .assets import sha256
from .benchmark import native_decoder
from .ctc import ctc_minimum_frames, greedy_ctc
from .data import TOKENS, load_split_manifest, object_hash, read_trial, reference_only, write_json
from .evaluation import edit_counts, normalize_text
from .inference import load_config, load_json, recover_cached_predictions
from .model import gaussian_kernel, load_encoder, output_length, smooth_valid
from .resources import BudgetExceeded, ResourceMonitor
from .training import atomic_checkpoint


class Encoder:
    def __init__(self, config, mode, device):
        self.mode, self.device = mode, device
        self.source = None
        if mode == "transformer":
            self.model = CausalT15Transformer(45).to(device)
            self.parameters = list(self.model.parameters())
        else:
            self.source, _ = load_encoder(config["checkpoint"], config["model_args"], device, config["checkpoint_hash"])
            self.source.requires_grad_(False)
            self.source.eval()
            self.kernel = gaussian_kernel()
            if self.source.contract_hash != config["contract_hash"]:
                raise ValueError("Official encoder preprocessing contract mismatch")
            self.model = (DiphoneHead.warm_start(self.source.out) if mode == "diphone"
                          else torch.nn.Linear(self.source.out.in_features, 41, device=device))
            if mode == "monophone":
                self.model.load_state_dict(self.source.out.state_dict())
            self.parameters = list(self.model.parameters())
        self.contract = {"version": 1, "family": mode, "tokens": TOKENS, "blank": 0, "silence": 40,
                         "feature_contract": config["contract_hash"], "day_transforms": 45,
                         "upstream_checkpoint_hash": config["checkpoint_hash"] if self.source else None,
                         "length_rule": "floor(raw_frames/4)" if mode == "transformer" else "official GRU 9/14/4",
                         "architecture": "384x5x6 128-frame local causal attention" if mode == "transformer"
                         else "frozen official GRU with 1601-class joint or 41-class affine head"}

    def forward(self, features, day, training):
        x = torch.from_numpy(features).to(self.device).unsqueeze(0)
        if training and self.mode == "transformer":
            for _ in range(2):
                width = random.randint(1, min(8, x.shape[1]))
                start = random.randrange(x.shape[1] - width + 1)
                x[:, start:start + width] = 0
        if self.mode == "transformer":
            logits, lengths = self.model(x, [day], [x.shape[1]])
            return logits[0, :lengths[0]]
        with torch.no_grad():
            hidden = self.source.encode(smooth_valid(x, self.kernel), [day])
        return self.model(hidden)[0]

    @torch.inference_mode()
    def emissions(self, record):
        self.model.eval()
        item = read_trial(record, include_labels=False)
        logits = self.forward(item["features"], record["day_index"], training=False)
        if self.mode == "diphone":
            from .alternative_encoders import marginal_log_probs
            logits = marginal_log_probs(logits.float().log_softmax(-1))
        array = logits.cpu().numpy()
        length = record["input_length"] // 4 if self.mode == "transformer" else output_length(record["input_length"])
        if array.shape != (length, 41) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid v2 emissions for {record['trial_id']}")
        return array

    def step(self, record, optimizer, step):
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        item = read_trial(record, include_labels=True)
        phones = item["phoneme_ids"]
        logits = self.forward(item["features"], record["day_index"], training=True)
        if self.mode == "diphone":
            alpha = max(.2, .8 - .6 * step / 1024)
            loss, transition, mono = diphone_objective(logits, phones, alpha)
            tracked = {"transition_ctc": float(transition.detach()), "phone_ctc": float(mono.detach()), "alpha": alpha}
        else:
            loss = ctc_for_trial(logits.float().log_softmax(-1), phones)
            tracked = {}
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        return {"loss": float(loss.detach()), **tracked}


def eligible_train(records, mode, monitor):
    kept = []
    skipped = []
    for record in records:
        if record["split"] != "train":
            continue
        monitor.check()
        target = reference_only(record)["phoneme_ids"]
        # Keep exactly the same feasible examples for both GRU head conditions.
        length = record["input_length"] // 4 if mode == "transformer" else output_length(record["input_length"])
        labels = target if mode == "transformer" else diphone_targets(target)
        group = kept if length >= ctc_minimum_frames(labels) else skipped
        group.append(record)
    if not kept:
        raise ValueError("No feasible official train examples")
    kept.sort(key=lambda r: object_hash([20261124, r["trial_id"]]))
    return kept, skipped


def trial_order(records):
    epoch = 0
    while True:
        order = list(records)
        random.Random(20261124 + epoch).shuffle(order)
        longest = max(range(len(order)), key=lambda i: order[i]["input_length"])
        order[0], order[longest] = order[longest], order[0]
        yield from order
        epoch += 1


def validate(encoder, selected, decode, output):
    word_errors = phone_errors = words = phones = 0
    predictions = []
    for record in selected:
        trial = record["trial_id"]
        emissions = encoder.emissions(record)
        predicted = greedy_ctc(emissions.argmax(-1).tolist())
        reply = decode(trial, emissions)
        choices = reply["candidates"]
        text = choices[0]["text"] if choices else ""
        label = reference_only(record)
        word = edit_counts(normalize_text(label["reference_text"]).split(), normalize_text(text).split())
        phone = edit_counts(label["phoneme_ids"], predicted)
        word_errors += word["errors"]
        phone_errors += phone["errors"]
        words += word["reference_length"]
        phones += phone["reference_length"]
        predictions.append({"trial_id": trial, "status": "ok" if choices else "decoder_error",
                            "text": text, "phoneme_ids": predicted, "emission_frames": len(emissions)})
    with output.open("w", encoding="utf-8") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in predictions)
    return {"evaluation_scope": "fixed tuning diagnostic subset", "trials": len(predictions),
            "word_errors": word_errors, "reference_words": words, "wer": word_errors / words,
            "phone_errors": phone_errors, "reference_phones": phones, "per": phone_errors / phones}


def cache_validation(config, records, split, checkpoint, output, monitor):
    digest = load_json(checkpoint.with_suffix(checkpoint.suffix + ".sha256.json"))["sha256"]
    if sha256(checkpoint) != digest:
        raise ValueError("V2 inference checkpoint hash mismatch")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    identity = saved["identity"]
    if (identity["data_manifest_sha256"] != sha256(config["data_manifest"])
            or identity["split"] != split["manifest_hash"]
            or identity["training_code_sha256"] != sha256(__file__)
            or identity["encoder_code_sha256"] != sha256(Path(__file__).with_name("alternative_encoders.py"))):
        raise ValueError("V2 inference code, data or split mismatch")
    encoder = Encoder(config, identity["mode"], config["device"])
    if encoder.contract != saved["contract"]:
        raise ValueError("V2 inference checkpoint contract mismatch")
    encoder.model.load_state_dict(saved["model_state_dict"], strict=True)
    val = sorted((r for r in records if r["split"] == "val"), key=lambda r: r["official_order_index"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "logits").mkdir(exist_ok=True)
    expected = {"checkpoint_hash": digest, "contract_hash": object_hash(encoder.contract),
                "split_hash": split["manifest_hash"], "manifest_hash": object_hash(records),
                "dtype": "float32", "allow_tf32": False, "partition": "val", "references_loaded": False}
    existing = recover_cached_predictions(output, expected, {r["trial_id"] for r in val})
    if [r["trial_id"] for r in existing] != [r["trial_id"] for r in val[:len(existing)]]:
        raise ValueError("V2 inference resume requires an official-order prefix")
    write_json(output / "cache_manifest.json", {**expected, "expected_trials": len(val), "contract": encoder.contract})
    write_json(output / "config.resolved.yaml", {"feature_source": config, "training": identity, **expected})
    status = {"state": "inferencing", "saved_trials": len(existing), "expected_trials": len(val)}
    start = monotonic()
    try:
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            for record in val[len(existing):]:
                monitor.check()
                array = encoder.emissions(record)
                path = f"logits/{record['official_order_index']:06d}.npy"
                np.save(output / path, array, allow_pickle=False)
                phones = greedy_ctc(array.argmax(-1).tolist())
                row = {key: record[key] for key in ("trial_id", "official_order_index", "session_id", "split", "input_length")}
                row.update(status="ok", output_length=len(array), phoneme_ids=phones, phonemes=[TOKENS[i] for i in phones],
                           logit_path=path, logit_sha256=sha256(output / path), text="", text_available=False)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                status["saved_trials"] += 1
        status["state"] = "complete"
    except BaseException as error:
        status.update(state="budget_stopped" if isinstance(error, (BudgetExceeded, KeyboardInterrupt)) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status.update(elapsed_seconds=monotonic() - start, references_loaded=False)
        write_json(output / "run_manifest.json", status)
        write_json(output / "metrics.json", status)
        (output / "report.md").write_text(f"# V2 encoder validation cache\n\nState `{status['state']}`. "
            f"Saved {status['saved_trials']} of {len(val)} emissions without references.\n", encoding="utf-8")


def train(args):
    if args.pilot_seconds and not 900 <= args.pilot_seconds <= 1800:
        raise ValueError("Training pilots require 900 to 1800 sustained seconds")
    if not args.pilot_seconds and not args.admission:
        raise ValueError("A non-pilot launch requires a matching admission receipt")
    if args.resume and args.pilot_seconds:
        raise ValueError("Resume requires a non-pilot launch")
    if args.updates < 1 or not 0 < args.max_seconds <= 28200:
        raise ValueError("Expected positive updates and a watchdog of at most 28200 seconds")
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"], records)
    ids = set(split["dev_tune"])
    tuning = sorted((r for r in records if r["trial_id"] in ids),
                    key=lambda r: (r["input_length"], r["official_order_index"]))
    tuning = [tuning[i] for i in np.linspace(0, len(tuning) - 1, 8, dtype=int)]
    identity = {"mode": args.mode, "seed": 20261124, "lr": .0003, "device": config["device"],
                "data_manifest_sha256": sha256(config["data_manifest"]), "split": split["manifest_hash"],
                "decoder": {"beam": 50, "nbest": 50, "lexicon_sha256": sha256(args.lexicon),
                            "lm_sha256": sha256(args.kenlm_model), "weight": args.lm_weight},
                "checkpoint_input": config["checkpoint_hash"], "feature_contract": config["contract_hash"],
                "training_code_sha256": sha256(__file__),
                "encoder_code_sha256": sha256(Path(__file__).with_name("alternative_encoders.py"))}
    if args.admission:
        admission = load_json(args.admission)
        if (admission["identity"] != identity or args.updates > admission["admitted_updates"]
                or args.max_seconds < admission["forecast_seconds"]):
            raise ValueError("Run differs from the measured admission")
    write_json(args.output / "config.resolved.yaml", {"identity": identity, "pilot_seconds": args.pilot_seconds,
               "requested_updates": args.updates, "max_seconds": args.max_seconds,
               "resume_sha256": sha256(args.resume) if args.resume else None,
               "tuning_ids": [r["trial_id"] for r in tuning]})
    outcome = {"status": "running", "updates": 0}
    try:
        torch.manual_seed(20261124)
        random.seed(20261124)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(config.get("cpu_threads", 4))
        with ResourceMonitor(args.output / "resources.json", max_seconds=args.max_seconds) as monitor, ExitStack() as stack:
            encoder = Encoder(config, args.mode, config["device"])
            initial_validation = stack.enter_context(native_decoder(args, 50, args.output / "decoder.log"))
            decode, decoder_setup = initial_validation
            kept, skipped = eligible_train(records, args.mode, monitor)
            write_json(args.output / "training_membership.json", {"included": [r["trial_id"] for r in kept],
                       "excluded_infeasible": [r["trial_id"] for r in skipped]})
            optimizer = torch.optim.AdamW(encoder.parameters, lr=.0003, weight_decay=.001)
            if args.resume:
                digest = load_json(args.resume.with_suffix(args.resume.suffix + ".sha256.json"))
                if digest["sha256"] != sha256(args.resume):
                    raise ValueError("Resume checkpoint hash mismatch")
                saved = torch.load(args.resume, weights_only=True, map_location="cpu")
                if (saved["identity"] != identity or saved["contract"] != encoder.contract
                        or saved["step"] != digest["step"] or saved["step"] >= args.updates):
                    raise ValueError("Resume checkpoint contract or update count mismatch")
                encoder.model.load_state_dict(saved["model_state_dict"], strict=True)
                optimizer.load_state_dict(saved["optimizer_state_dict"])
                torch.set_rng_state(saved["torch_rng"])
                if saved["cuda_rng"]:
                    torch.cuda.set_rng_state_all(saved["cuda_rng"])
                random.setstate(saved["python_rng"])
                outcome["updates"] = saved["step"]
            begin = last_checkpoint = monotonic()
            context = {"decoder_setup": decoder_setup, "contract": encoder.contract, "excluded_infeasible": len(skipped),
                       "trainable_parameters": sum(p.numel() for p in encoder.parameters),
                       "total_parameters": sum(p.numel() for p in encoder.model.parameters()) +
                       (sum(p.numel() for p in encoder.source.parameters()) if encoder.source else 0)}
            outcome.update(context)

            def save():
                atomic_checkpoint(args.output / "checkpoints/last.pt", {"model_state_dict": encoder.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "step": outcome["updates"],
                    "contract": encoder.contract, "identity": identity, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "python_rng": random.getstate()})

            def checkpoint_and_validate():
                save()
                summary = validate(encoder, tuning, decode, args.output / "predictions.jsonl")
                outcome.update(summary, sustained_seconds=monotonic() - begin)
                write_json(args.output / "metrics.json", outcome)

            completed_updates = outcome["updates"]
            order = islice(trial_order(kept), completed_updates, None)
            with (args.output / "training.jsonl").open("w", encoding="utf-8") as log:
                try:
                    for step, record in enumerate(order, completed_updates + 1):
                        monitor.check()
                        row = encoder.step(record, optimizer, step)
                        outcome["updates"] = step
                        log.write(json.dumps({"update": step, **row}) + "\n")
                        if step % 32 == 0:
                            log.flush()
                        if monotonic() - last_checkpoint >= 120:
                            checkpoint_and_validate()
                            last_checkpoint = monotonic()
                        if (args.pilot_seconds and monotonic() - begin >= args.pilot_seconds) or (
                            not args.pilot_seconds and step >= args.updates):
                            break
                    checkpoint_and_validate()
                except (BudgetExceeded, KeyboardInterrupt):
                    save()
                    raise
            monitor.check()
            outcome["status"] = "pilot_complete" if args.pilot_seconds else "training_complete"
            outcome["checkpoint_sha256"] = sha256(args.output / "checkpoints/last.pt")
            if not args.pilot_seconds:
                del encoder, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cache_validation(config, records, split, args.output / "checkpoints/last.pt",
                                 args.output / "inference", monitor)
                outcome["status"] = "complete"
            monitor.check()
        if load_json(args.output / "resources.json")["outcome"] != "complete":
            raise BudgetExceeded("Encoder work exceeded the resource envelope at the final sample")
    except BaseException as error:
        outcome.update(status="budget_stopped" if isinstance(error, (BudgetExceeded, KeyboardInterrupt)) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "run_manifest.json", outcome)
        write_json(args.output / "metrics.json", outcome)
        (args.output / "report.md").write_text(f"# Encoder study\n\nState `{outcome['status']}` after "
            f"{outcome['updates']} updates. See metrics.json for the fixed tuning subset and actual training cost.\n",
            encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "lexicon", "kenlm-model", "decoder-python", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mode", choices=("monophone", "diphone", "transformer"), required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pilot-seconds", type=int, default=900)
    parser.add_argument("--max-seconds", type=float, default=1350)
    parser.add_argument("--updates", type=int, default=1024)
    parser.add_argument("--lm-weight", type=float, default=2.0)
    parser.set_defaults(nbest=50, beam_threshold=25, start_timeout=120, request_timeout=30)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
