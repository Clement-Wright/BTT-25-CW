"""Train rate/multiscale encoders on matched official-train trials and score dev_tune."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import torch.nn.functional as F

from .alternative_encoders import ctc_for_trial
from .assets import sha256
from .benchmark import native_decoder
from .cache_advice import run as advise_training_cache
from .ctc import ctc_minimum_frames, greedy_ctc
from .data import attach_corpus_metadata, load_split_manifest, object_hash, read_trial, reference_only, write_json
from .evaluation import _by_id, _selection, edit_counts, normalize_text
from .inference import load_config, load_json
from .model import gaussian_kernel, load_encoder, smooth_valid
from .resources import BudgetExceeded, ResourceMonitor
from .train_v2 import trial_order
from .training import atomic_checkpoint
from .v4_rate import RateEncoder, SPECS, geometry


def common_order(records, count, monitor):
    # Shuffle once before feasibility filtering. Check only the requested prefix,
    # sharing the most restrictive supported geometry across all conditions.
    train = [r for r in records if r["split"] == "train"]
    if not train:
        raise ValueError("No official training records")
    result, rejected = [], []
    for row in trial_order(train):
        monitor.check()
        labels = reference_only(row)["phoneme_ids"]
        shortest = min(geometry(name, row["input_length"]) for name in SPECS)
        if labels and shortest >= ctc_minimum_frames(labels):
            result.append(row)
            if len(result) == count:
                return result, rejected
        else:
            rejected.append(row["trial_id"])
            if len(rejected) == len(train) and not result:
                raise ValueError("No feasible training records")
    raise AssertionError("Empty traversal")


def optimizer_for(model):
    groups = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fast = model.spec.family == "causal_conv" or not name.startswith(("gru.", "out."))
            groups.setdefault(.0003 if fast else .00003, []).append(param)
    return torch.optim.AdamW([{"params": params, "lr": lr} for lr, params in groups.items()],
                             weight_decay=.001)


def step_model(model, teacher, opt, row):
    model.train()
    item = read_trial(row, include_labels=True)
    x = torch.from_numpy(item["features"]).cuda().unsqueeze(0)
    opt.zero_grad(set_to_none=True)
    logits = model(x, row["day_index"])[0].float()
    loss = ctc_for_trial(logits.log_softmax(-1), item["phoneme_ids"])
    if teacher is not None:
        with torch.no_grad():
            anchor = teacher(smooth_valid(x, model.kernel), [row["day_index"]])[0]
            prior = (anchor / 2).softmax(-1).repeat_interleave(2, 0)[:len(logits)]
        loss = loss + .35 * 4 * F.kl_div((logits / 2).log_softmax(-1), prior, reduction="batchmean")
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training objective")
    loss.backward()
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                  5., error_if_nonfinite=True)
    opt.step()
    return float(loss.detach())


@torch.inference_mode()
def evaluate(model, records, decode, directory, monitor, final):
    model.eval()
    total, by_session, by_corpus = Counter(), {}, {}
    seconds = []
    directory.mkdir(parents=True, exist_ok=True)
    if final:
        (directory / "logits").mkdir(exist_ok=True)
    with (directory / "predictions.jsonl").open("w", encoding="utf-8", buffering=1) as stream:
        for row in records:
            monitor.check()
            start = monotonic()
            x = torch.from_numpy(read_trial(row)["features"]).cuda().unsqueeze(0)
            logits = model(x, row["day_index"])[0].float().cpu().numpy()
            if logits.shape != (model.output_length(row["input_length"]), 41) or not np.isfinite(logits).all():
                raise ValueError(f"Invalid emissions for {row['trial_id']}")
            phones = greedy_ctc(logits.argmax(-1).tolist())
            candidates = decode(row["trial_id"], logits)["candidates"]
            text = candidates[0]["text"] if candidates else ""
            elapsed = monotonic() - start
            seconds.append(elapsed)
            relative = f"logits/{row['official_order_index']:05d}.npy" if final else None
            if final:
                np.save(directory / relative, logits, allow_pickle=False)
            label = reference_only(row)
            w = edit_counts(normalize_text(label["reference_text"]).split(), normalize_text(text).split())
            p = edit_counts(label["phoneme_ids"], phones)
            counts = Counter(trials=1, words=w["reference_length"], word_errors=w["errors"],
                             phones=p["reference_length"], phone_errors=p["errors"], empty=int(not text))
            for key in ("insertions", "deletions", "substitutions"):
                counts["word_" + key] = w[key]
                counts["phone_" + key] = p[key]
            total.update(counts)
            by_corpus.setdefault(row.get("corpus_id") or "unclassified", Counter()).update(counts)
            by_session.setdefault(row["session_id"], Counter()).update(counts)
            stream.write(json.dumps({"trial_id": row["trial_id"], "status": "ok" if text else "empty",
                                     "text": text, "phones": phones, "counts": counts,
                                     "source_frames": row["input_length"], "emission_frames": len(logits),
                                     "elapsed_seconds": elapsed, "logits": relative}) + "\n")
    return {"counts": total, "by_session": by_session, "by_corpus": by_corpus,
            "per": total["phone_errors"] / total["phones"], "wer": total["word_errors"] / total["words"],
            "mean_pipeline_seconds": float(np.mean(seconds)), "trials": len(records)}


def study(args):
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use this real-data study inside an allocated CUDA Slurm job")
    if len(set(args.conditions)) != len(args.conditions) or args.updates < 1:
        raise ValueError("Require unique conditions and positive updates")
    if not 900 <= args.pilot_seconds <= 1800 or not args.pilot_seconds < args.max_seconds <= 28200:
        raise ValueError("Require a 15–30 minute group pilot and a bounded wall budget")
    started = monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = load_json(cfg["data_manifest"])
    if args.corpus_metadata:
        records = attach_corpus_metadata(records, args.corpus_metadata)
    split = load_split_manifest(cfg["split_manifest"], records)
    tune = _selection(_by_id(records, "trial"), split, "dev_tune")
    ordered = sorted(tune, key=lambda r: r["input_length"])
    pilot = [ordered[i] for i in np.unique(np.linspace(0, len(ordered)-1, min(12, len(ordered)), dtype=int))]
    source_hashes = {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("v4_rate.py"),
                     args.lexicon, args.kenlm_model, Path(cfg["data_manifest"]), args.config]}
    setup = args.output / "setup"
    setup.mkdir()
    with ResourceMonitor(setup / "resources.json", max_seconds=args.max_seconds) as monitor:
        order, rejected = common_order(records, args.updates, monitor)
        base, upstream = load_encoder(cfg["checkpoint"], cfg["model_args"], "cpu", cfg["checkpoint_hash"])
        if base.contract_hash != cfg["contract_hash"]:
            raise ValueError("Checkpoint contract mismatch")
        tr = upstream["dataset"]["data_transforms"]
        kernel = gaussian_kernel(tr["smooth_kernel_std"], tr["smooth_kernel_size"])
        if len(kernel) != 9:
            raise ValueError("Rate contracts require the official nine-tap kernel")
        membership = {"train": [r["trial_id"] for r in order], "infeasible": rejected,
                      "pilot_dev_tune": [r["trial_id"] for r in pilot], "dev_tune": [r["trial_id"] for r in tune]}
        write_json(args.output / "membership.json", membership)
        monitor.check()
    if load_json(setup / "resources.json")["outcome"] != "complete":
        raise BudgetExceeded("Resource budget during setup")
    identity = {"source_sha256": cfg["checkpoint_hash"], "contract_hash": cfg["contract_hash"],
                "split_hash": split["manifest_hash"], "membership_hash": object_hash(membership),
                "seed": 20261127, "updates": args.updates, "hashes": source_hashes, "decode": {
                    "beam": args.beam, "lm_weight": args.lm_weight, "nbest": args.nbest,
                    "beam_threshold": args.beam_threshold}}
    write_json(args.output / "config.resolved.yaml", {**identity, "arguments": {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    progress = {}
    try:
        for phase in ("pilot", "final"):
            if phase == "final":
                estimate = 1.25 * sum(p["mean_step_seconds"] * (args.updates - p["updates"])
                              + p["mean_evaluation_seconds"] * len(tune) + p["save_seconds"] * 2
                              for p in progress.values()) + 45
                left = args.max_seconds - (monotonic() - started)
                write_json(args.output / "admission.json", {"estimated_seconds": estimate,
                           "remaining_seconds": left, "admitted": estimate <= left, "safety_factor": 1.25})
                if estimate > left:
                    raise BudgetExceeded("Measured remaining training, validation and saving exceed budget")
            for name in args.conditions:
                root = args.output / name
                out = root / "pilot" if phase == "pilot" else root
                out.mkdir(parents=True, exist_ok=True)
                checkpoint = root / "checkpoints/last.pt"
                state = {"status": "started", "phase": phase, "condition": name, "updates": 0}
                write_json(root / "provenance.json", identity)
                write_json(root / "config.resolved.yaml", {**identity, "condition": name,
                           "loss": "CTC + 0.35 * T^2 KL(T=2) from held GRU teacher" if name == "causal_conv" else "CTC"})
                with ResourceMonitor(out / "resources.json", max_seconds=args.max_seconds - (monotonic() - started)) as monitor:
                    try:
                        torch.manual_seed(identity["seed"])
                        model = RateEncoder(deepcopy(base), name, kernel).cuda()
                        teacher = deepcopy(base).cuda().eval().requires_grad_(False) if name == "causal_conv" else None
                        optimizer = optimizer_for(model)
                        state.update(contract=model.contract, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
                        resume = checkpoint if phase == "final" else (args.resume_from / name / "checkpoints/last.pt"
                                                                    if args.resume_from else None)
                        if resume is not None:
                            if not resume.is_file():
                                raise FileNotFoundError(f"Requested resume checkpoint is unavailable: {resume}")
                            if sha256(resume) != load_json(resume.with_suffix(".pt.sha256.json"))["sha256"]:
                                raise ValueError("Resume checkpoint digest mismatch")
                            saved = torch.load(resume, map_location="cpu", weights_only=True)
                            if saved["identity"] != identity or saved["condition"] != name:
                                raise ValueError("Resume training identity mismatch")
                            model.load_delta(saved["trainable"])
                            optimizer.load_state_dict(saved["optimizer"])
                            state["updates"] = saved["step"]
                            del saved

                        def save():
                            atomic_checkpoint(checkpoint, {"trainable": model.trainable_state(), "optimizer": optimizer.state_dict(),
                                              "identity": identity, "condition": name, "step": state["updates"],
                                              "source_sha256": identity["source_sha256"]})

                        with native_decoder(args, args.beam, out / "decoder.log") as (decode, setup_info):
                            state["decoder"] = setup_info
                            begin = last = monotonic()
                            steps, train_time = 0, 0.
                            with (out / "training.jsonl").open("w", encoding="utf-8", buffering=1) as log:
                                try:
                                    limit = min(args.updates, 512) if phase == "pilot" else args.updates
                                    while state["updates"] < limit:
                                        monitor.check()
                                        row = order[state["updates"]]
                                        tick = monotonic()
                                        loss = step_model(model, teacher, optimizer, row)
                                        train_time += monotonic() - tick
                                        steps += 1
                                        state["updates"] += 1
                                        log.write(json.dumps({"step": state["updates"], "trial_id": row["trial_id"], "loss": loss}) + "\n")
                                        if state["updates"] % 32 == 0 and monitor.host.cgroup is not None:
                                            current = int((monitor.host.cgroup / "memory.current").read_text())
                                            if current > .78 * monitor.soft_ram:
                                                advise_training_cache(Path(cfg["data_manifest"]), Path(os.environ["RCAC_SCRATCH"]),
                                                                      out / f"cache-advice-{state['updates']}.json")
                                        if monotonic() - last >= 120:
                                            save()
                                            last = monotonic()
                                        if phase == "pilot" and monotonic()-begin >= args.pilot_seconds / len(args.conditions) * .80:
                                            break
                                finally:
                                    tick = monotonic()
                                    save()
                                    state["save_seconds"] = monotonic() - tick
                            if phase == "pilot":
                                tick = monotonic()
                                state["pilot_validation"] = evaluate(model, pilot, decode, out, monitor, False)
                                state["mean_evaluation_seconds"] = (monotonic() - tick) / len(pilot)
                                # Continue genuine workload until the aggregate sustained pilot lasts 15–30 min.
                                while monotonic()-begin < args.pilot_seconds / len(args.conditions):
                                    evaluate(model, pilot, decode, out, monitor, False)
                                if not steps and state["updates"] < args.updates:
                                    receipt = args.resume_from / name / "pilot/metrics.json" if args.resume_from else None
                                    if not receipt or not receipt.is_file() or not load_json(receipt).get("mean_step_seconds"):
                                        # A preempted pilot may have a checkpoint but no timing receipt.
                                        # Measure one disposable optimizer update, then restore the durable endpoint.
                                        tick = monotonic()
                                        step_model(model, teacher, optimizer, order[state["updates"]])
                                        train_time, steps = monotonic() - tick, 1
                                    else:
                                        train_time, steps = load_json(receipt)["mean_step_seconds"], 1
                                state["mean_step_seconds"] = train_time / max(steps, 1)
                                state["sustained_seconds"] = monotonic() - begin
                                progress[name] = state.copy()
                            else:
                                state.update(evaluate(model, tune, decode, out, monitor, True))
                            monitor.check()
                        state.update(status="pilot_complete" if phase == "pilot" else "complete",
                                     checkpoint_sha256=sha256(checkpoint))
                    except BaseException as exc:
                        state.update(status="budget_stopped" if isinstance(exc, (BudgetExceeded, KeyboardInterrupt)) else "failed",
                                     error=f"{type(exc).__name__}: {exc}")
                        raise
                    finally:
                        write_json(out / "metrics.json", state)
                        (out / "report.md").write_text(f"# {name}\n\n{phase}, {state['status']}, "
                            f"{state['updates']} optimizer steps. Tuning WER {state.get('wer')}, PER {state.get('per')}.\n")
                if load_json(out / "resources.json")["outcome"] != "complete":
                    state.update(status="budget_stopped", error="Final telemetry sample exceeded budget")
                    write_json(out / "metrics.json", state)
                    raise BudgetExceeded(state["error"])
                del model, teacher, optimizer
                torch.cuda.empty_cache()
        write_json(args.output / "metrics.json", {"status": "complete", "conditions": args.conditions})
    except BaseException as exc:
        write_json(args.output / "metrics.json", {"status": "budget_stopped" if isinstance(exc, BudgetExceeded)
                   else "failed", "error": f"{type(exc).__name__}: {exc}", "completed_pilots": list(progress)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "lexicon", "kenlm-model", "decoder-python", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--decoder-module-path", type=Path)
    parser.add_argument("--corpus-metadata", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--conditions", choices=list(SPECS), required=True, nargs="+")
    parser.add_argument("--updates", type=int, default=2048)
    parser.add_argument("--pilot-seconds", type=int, default=900)
    parser.add_argument("--max-seconds", type=float, default=5400)
    parser.add_argument("--beam", type=int, default=50)
    parser.add_argument("--lm-weight", type=float, default=2.)
    parser.set_defaults(nbest=10, beam_threshold=25., start_timeout=120., request_timeout=30.)
    study(parser.parse_args())


if __name__ == "__main__":
    main()
