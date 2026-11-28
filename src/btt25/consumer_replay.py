"""Replay arrived T15 bins at 50 Hz through a stateful CPU GRU and Flashlight."""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from .assets import sha256
from .benchmark import native_decoder
from .consumer_v4 import decoder_args, evidence_inputs, load_cpu, sample_lengths, score
from .ctc import greedy_ctc
from .data import object_hash, read_trial, write_json
from .model import output_length, smooth_valid
from .online_encoder import OnlineGRU, OnlineTwoRate
from .v4_rate import RateEncoder
from .resources import ResourceMonitor


def common_prefix(left, right):
    n = 0
    while n < min(len(left), len(right)) and left[n] == right[n]:
        n += 1
    return left[:n]


def stable_latencies(drafts, final, duration):
    """Earliest observation after which each final word's full prefix never changes."""
    common, times = list(final), [duration] * len(final)
    for elapsed, words in reversed(drafts):
        common = common_prefix(common, words)
        for i in range(len(common)):
            times[i] = elapsed
    return times


def early_commit_audit(drafts, final):
    """Simulate irreversible confirmation after five agreeing views and one later word."""
    committed, first_time, triggered_conflicts = [], None, 0
    for i in range(4, len(drafts)):
        stable = drafts[i][1]
        for j in range(i - 4, i):
            stable = common_prefix(stable, drafts[j][1])
        eligible = stable[:-1]
        if len(eligible) > len(committed) and eligible[: len(committed)] == committed:
            if first_time is None:
                first_time = drafts[i][0]
            committed = list(eligible)
        if drafts[i][1][: len(committed)] != committed:
            triggered_conflicts += 1
    return {
        "words_confirmed_early": len(committed),
        "first_confirmation_s": first_time,
        "later_drafts_conflicting": triggered_conflicts,
        "final_prefix_agreement": final[: len(committed)] == committed,
    }


@torch.inference_mode()
def replay(record, model, kernel, decode, monitor, trace):
    clock = time.perf_counter
    trial, day, length = record["trial_id"], record["day_index"], record["input_length"]
    two_rate = isinstance(model, RateEncoder)
    network = OnlineTwoRate(model, day) if two_rate else OnlineGRU(model, kernel, day)
    patch_size, stride = network.size, network.stride
    def batch(x):
        return model(x, day) if two_rate else model(smooth_valid(x, kernel), [day])
    empty = np.empty((0, 41), np.float32)
    decode(trial, empty, "begin")
    emissions, drafts, delay, step_cpu_wall = [], [], [], []
    with h5py.File(record["source_file"]) as file:
        dataset = file[record["source_group"]]["input_features"]
        t0 = clock()
        for i in range(length):
            monitor.check()
            due = t0 + (i + 1) * 0.020
            time.sleep(max(0.0, due - clock()))
            step_start = clock()
            observed = clock() - t0
            bins = np.asarray(dataset[i : i + 1], np.float32)
            result = network.push(torch.from_numpy(bins))
            if result is not None:
                logits = result.numpy()
                emissions.append(logits.copy())
                reply = decode(trial, logits, "step")
                words = reply["candidates"][0]["text"].split() if reply["candidates"] else []
                elapsed = clock() - t0
                drafts.append((elapsed, words))
                trace.write(
                    json.dumps(
                        {"trial_id": trial, "input_bins_seen": i + 1, "elapsed_s": elapsed, "draft": words}
                    )
                    + "\n"
                )
            delay.append(observed - (i + 1) * 0.020)
            step_cpu_wall.append(clock() - step_start)
        reply = decode(trial, empty, "end")
        final_time = clock() - t0
    final = reply["candidates"][0]["text"] if reply["candidates"] else ""
    trace.flush()
    array = np.concatenate(emissions)
    if array.shape != (output_length(length, len(kernel), patch_size, stride), 41):
        raise RuntimeError("Lost stream emissions")
    # Separate offline checks AFTER finalization. No future bins enter the live path.
    x = torch.from_numpy(read_trial(record)["features"])[None]
    offline = batch(x)[0].numpy()
    prefix_checks = []
    for quarter in (1, 2, 3):
        frames = max(1, quarter * len(offline) // 4)
        seen_bins = len(kernel) + patch_size - 1 + stride * (frames - 1)
        prefix = batch(x[:, :seen_bins])[0].numpy()
        if prefix.shape != offline[:frames].shape:
            raise ValueError("Mismatch in full-pass prefix boundaries")
        prefix_checks.append({
            "frames":frames, "seen_bins":seen_bins,
            "max_abs_logit_delta":float(np.abs(offline[:frames] - prefix).max()),
            "changed_argmax":int(np.count_nonzero(offline[:frames].argmax(-1) != prefix.argmax(-1))),
        })
    from_stream_output = decode(trial, array)["candidates"]
    batch_text = from_stream_output[0]["text"] if from_stream_output else ""
    batch_score = from_stream_output[0]["component_scores"]["decoder_total"] if from_stream_output else None
    stream_score = reply["candidates"][0]["score"] if reply["candidates"] else None
    repeated = decode(trial, array)["candidates"]
    from_offline = decode(trial, offline)["candidates"]
    stable = stable_latencies(drafts, final.split(), final_time)
    retracts, words_retracted = 0, 0
    revisions = [w for _, w in drafts] + [final.split()]
    for left, right in zip(revisions, revisions[1:]):
        removed = len(left) - len(common_prefix(left, right))
        retracts += int(removed > 0)
        words_retracted += removed
    return {
        "trial_id": trial,
        "status": "ok" if reply["candidates"] else "empty_search",
        "phoneme_status": "ok",
        "text": final,
        "phoneme_ids": greedy_ctc(array.argmax(-1)),
        "input_seconds": length * 0.020,
        "raw_frames": length,
        "emissions": len(array),
        "first_phone_update_s": drafts[0][0],
        "first_partial_s": next((t for t, w in drafts if w), None),
        "first_stable_word_s": stable[0] if stable else None,
        "draft_revision_events": retracts,
        "words_retracted": words_retracted,
        "finalization_after_last_scheduled_bin_s": final_time - length * 0.020,
        "finalization_after_last_observed_bin_s": final_time - observed,
        "max_arrival_lag_s": max(delay),
        "arrival_lag_p95_s": float(np.percentile(delay, 95)),
        "input_bins_at_least_one_period_late": sum(lag >= 0.020 for lag in delay),
        "replay_processing_work_s": sum(step_cpu_wall),
        "max_abs_full_pass_logit_delta": float(np.abs(offline - array).max()),
        "offline_frame_argmax_differences": int(np.count_nonzero(offline.argmax(-1) != array.argmax(-1))),
        "full_pass_prefix_checks": prefix_checks,
        "offline_text_agreement": final == (from_offline[0]["text"] if from_offline else ""),
        "batch_text_same_emissions": batch_text,
        "stream_search_score": stream_score,
        "batch_score_same_emissions": batch_score,
        "stream_minus_batch_search_score": (
            stream_score - batch_score if None not in (stream_score, batch_score) else None
        ),
        "batch_repeated_same_emissions_agrees": batch_text == (repeated[0]["text"] if repeated else ""),
        "stream_vs_batch_search_same_emissions": final == batch_text,
        "early_commit_audit": early_commit_audit(drafts, final.split()),
    }


def run(args):
    if os.getenv("CUDA_VISIBLE_DEVICES") not in ("", "-1") or torch.cuda.is_available():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=-1 before running CPU replay")
    args.out.mkdir(parents=True, exist_ok=False)
    config, split, tune = evidence_inputs(args)
    records = sample_lengths(tune, 8)
    delta_hash = None
    if args.two_rate_delta:
        if args.precision != "fp32" or not args.delta_receipt:
            raise ValueError("The two-rate deployment check requires FP32 and the export receipt")
        receipt = json.loads(args.delta_receipt.read_text())
        delta_hash = sha256(args.two_rate_delta)
        if delta_hash != receipt["delta_sha256"] or receipt["parent_sha256"] != config["checkpoint_hash"]:
            raise ValueError("Two-rate transfer or parent checkpoint hash mismatch")
    resolved = {
        "encoder":"two_rate" if delta_hash else "official",
        "delta_sha256":delta_hash,
        "phone_step_ms":40 if delta_hash else 80,
        "precision": args.precision,
        "score_split": "dev_tune",
        "split_sha256": split["manifest_hash"],
        "ids": [r["trial_id"] for r in records],
        "id_hash": object_hash([r["trial_id"] for r in records]),
        "checkpoint": config["checkpoint_hash"],
        "lexicon": sha256(args.lexicon),
        "contract": config["contract_hash"],
        "cpu_threads": 4,
        "beam": 50,
        "lm": "KenLM" if args.kenlm_model else "ZeroLM",
        "lm_sha256": sha256(args.kenlm_model) if args.kenlm_model else None,
        "lm_weight_natural_log": args.lm_weight if args.kenlm_model else 0.0,
        "word_commitment": "Provisional words may revise on every phone update. Commit only on supplied trial end.",
        "early_commit_audit": "Separately audit irreversible confirmation after five agreeing successive draft prefixes and one stable later word.",
        "input": "Released standardized intracortical 20 ms bins replayed off disk on a wall clock, known session and trial boundary",
        "torch": str(torch.__version__),
    }
    (args.out / "config.resolved.yaml").write_text(yaml.safe_dump(resolved), encoding="utf8")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with ResourceMonitor(args.out / "resources.json", args.watchdog) as monitor:
        t0 = time.perf_counter()
        model, kernel = load_cpu(config, args.precision)
        if delta_hash:
            delta = torch.load(args.two_rate_delta, map_location="cpu", weights_only=True, mmap=True)
            adapted = RateEncoder(model, "two_rate", kernel)
            if (delta["source_sha256"] != config["checkpoint_hash"] or delta["step"] != 2048
                    or delta["condition"] != "two_rate" or delta["contract"] != adapted.contract
                    or delta["training_checkpoint_sha256"] != receipt["source_checkpoint_sha256"]):
                raise ValueError("Invalid two-rate training identity")
            adapted.load_delta(delta["trainable"])
            model = adapted.eval()
            del delta
        cold = time.perf_counter() - t0
        rows = []
        with native_decoder(decoder_args(args), 50, args.out / "decoder.stderr.log") as (decode, setup):
            write_json(args.out / "setup.json", {"load_convert_s": cold, "decoder": setup})
            with (args.out / "drafts.jsonl").open("x", encoding="utf8") as trace, (
                args.out / "predictions.jsonl"
            ).open("x", encoding="utf8") as pred:
                for record in records:
                    row = replay(record, model, kernel, decode, monitor, trace)
                    rows.append(row)
                    pred.write(json.dumps(row) + "\n")
                    pred.flush()
        measures = {}
        for key in (
            "first_phone_update_s",
            "first_partial_s",
            "first_stable_word_s",
            "finalization_after_last_scheduled_bin_s",
            "finalization_after_last_observed_bin_s",
            "max_arrival_lag_s",
            "max_abs_full_pass_logit_delta",
            "draft_revision_events",
        ):
            values = [r[key] for r in rows if r[key] is not None]
            measures[key] = (
                {"count": len(values), "median": float(np.median(values)), "max": max(values)}
                if values
                else None
            )
        confirmations = [r["early_commit_audit"] for r in rows]
        first_confirmations = [
            c["first_confirmation_s"] for c in confirmations if c["first_confirmation_s"] is not None
        ]
        write_json(
            args.out / "metrics.json",
            {
                "quality": score(rows, records)["overall"],
                "latency": measures,
                "trials": len(rows),
                "offline_text_agreements": sum(r["offline_text_agreement"] for r in rows),
                "offline_frame_argmax_differences": sum(
                    r["offline_frame_argmax_differences"] for r in rows
                ),
                "batch_vs_stream_identical_emission_text_agreements": sum(
                    r["stream_vs_batch_search_same_emissions"] for r in rows
                ),
                "batch_repeat_identical_emission_text_agreements": sum(
                    r["batch_repeated_same_emissions_agrees"] for r in rows
                ),
                "max_abs_batch_vs_stream_score_difference": max(
                    (
                        abs(r["stream_minus_batch_search_score"])
                        for r in rows
                        if r["stream_minus_batch_search_score"] is not None
                    ),
                    default=None,
                ),
                "emission_frames": sum(r["emissions"] for r in rows),
                "full_pass_prefix_checks": {
                    "comparisons":sum(len(r["full_pass_prefix_checks"]) for r in rows),
                    "worst_max_abs_logit_delta":max(c["max_abs_logit_delta"] for r in rows for c in r["full_pass_prefix_checks"]),
                    "changed_argmax_over_overlapping_prefixes":sum(c["changed_argmax"] for r in rows for c in r["full_pass_prefix_checks"]),
                    "frames_over_overlapping_prefixes":sum(c["frames"] for r in rows for c in r["full_pass_prefix_checks"]),
                },
                "retracted_words": sum(r["words_retracted"] for r in rows),
                "raw_frames": sum(r["raw_frames"] for r in rows),
                "bins_one_period_late": sum(r["input_bins_at_least_one_period_late"] for r in rows),
                "input_seconds": sum(r["input_seconds"] for r in rows),
                "early_commit_audit": {
                    "trials_confirming_before_end": len(first_confirmations),
                    "words_confirmed": sum(c["words_confirmed_early"] for c in confirmations),
                    "trials_disagreeing_with_final": sum(
                        not c["final_prefix_agreement"] for c in confirmations
                    ),
                    "median_first_confirmation_s": (
                        float(np.median(first_confirmations)) if first_confirmations else None
                    ),
                },
                "active_per_bin_work_rtf": sum(r["replay_processing_work_s"] for r in rows)
                / sum(r["input_seconds"] for r in rows),
            },
        )
    partial = measures["first_partial_s"]
    first = (
        f"The first draft appeared at a median {partial['median']:.2f} s after trial start. "
        if partial
        else "The decoder returned no nonempty drafts. "
    )
    (args.out / "report.md").write_text(
        f"# {args.precision.upper()} T15 sensor-clock replay\n\nEight tuning trials were replayed at 50 Hz. "
        f"{first}"
        f"Final text arrived {1000*measures['finalization_after_last_observed_bin_s']['median']:.1f} ms "
        "after the last observed bin (median). Words remained provisional until supplied trial end.\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--precision", choices=("fp32", "int8"), required=True)
    p.add_argument("--two-rate-delta",type=Path)
    p.add_argument("--delta-receipt",type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    p.add_argument("--lexicon", type=Path, default=Path("assets/local/lexicon.txt"))
    p.add_argument("--kenlm-model", type=Path)
    p.add_argument("--lm-weight", type=float, default=2.0)
    p.add_argument("--decoder-python", type=Path, default=Path(".decoder-venv/Scripts/python.exe"))
    p.add_argument("--watchdog", type=int, default=28200)
    run(p.parse_args())
