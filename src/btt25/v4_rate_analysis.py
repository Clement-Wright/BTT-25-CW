"""Explain emission-clock changes on complete, matched development predictions."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from btt25.assets import sha256
from btt25.model import gaussian_kernel

from btt25.ctc import ctc_minimum_frames, greedy_ctc
from btt25.data import load_split_manifest, reference_only, write_json
from btt25.evaluation import _by_id, _selection
from btt25.inference import load_config, load_json, read_jsonl
from btt25.temporal_analysis import align, summarize


def emission_stats(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 41 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite T15 logits")
    peak = x.max(axis=1, keepdims=True)
    logp = x - peak - np.log(np.exp(x - peak).sum(axis=1, keepdims=True))
    p = np.exp(logp)
    best = x.argmax(-1)
    return best, Counter(frames=len(best), blank_frames=int(np.sum(best == 0)),
                         sum_blank_probability=float(p[:, 0].sum()),
                         sum_entropy_nats=-float((p * logp).sum()))


def repeated_phone_counts(targets, hypothesis):
    counts = Counter()
    labels = list(reversed(list(align(targets, hypothesis))))
    i = 0
    for target, actual in labels:
        if target is None:
            continue
        # Mark both members of adjacent identical reference phones.
        adjacent = (i > 0 and targets[i-1] == target) or (i+1 < len(targets) and targets[i+1] == target)
        key = "adjacent_repeat" if adjacent else "other"
        counts[key + "_reference"] += 1
        if target != actual:
            counts[key + ("_deleted" if actual is None else "_substituted")] += 1
        i += 1
    return counts



def distribution(x):
    x = np.asarray(x, dtype=np.float64)
    lo, p10, mid, p90, hi = np.quantile(x, [0, .1, .5, .9, 1])
    return dict(min=float(lo), p10=float(p10), median=float(mid), p90=float(p90),
                max=float(hi), mean=float(x.mean()))


def kernel_width(weights):
    # Interpolate half-height crossings around the highest tap. The FIR ends
    # outside taps -4..4. Zero-valued guard taps define both edge crossings.
    padded = np.pad(weights, (1, 1))
    k = int(np.argmax(padded))
    half = padded[k] / 2
    l = k
    while padded[l-1] >= half:
        l -= 1
    r = k
    while padded[r+1] >= half:
        r += 1
    left = l - (padded[l]-half)/(padded[l]-padded[l-1])
    right = r + (padded[r]-half)/(padded[r]-padded[r+1])
    return (right-left)*20


def describe_kernel(weights):
    t = np.arange(-4, 5, dtype=float)*20
    center = (weights*t).sum(-1)
    std = np.sqrt((weights*(t-center[:, None])**2).sum(-1))
    return {"center_ms": distribution(center), "std_ms": distribution(std),
            "half_height_width_ms": distribution([kernel_width(w) for w in weights]),
            "channel_mean_taps": weights.mean(0).tolist()}


def frontend_parameters(state):
    result = {}
    if "filter_offsets" in state:
        kernel = gaussian_kernel().double().numpy()
        result["original_gaussian"] = describe_kernel(kernel[None,:])
        offsets = state["filter_offsets"].double().numpy()
        sharp = state["sharpness"].double().tanh().numpy()[:,None]
        radius = np.abs(np.arange(-4,5))/4
        result["learned_at_endpoint_difference"] = {}
        for magnitude in (0,1,3):
            logits = np.log(kernel) + offsets - magnitude*sharp*radius
            weights = np.exp(logits-logits.max(axis=1,keepdims=True))
            weights /= weights.sum(axis=1,keepdims=True)
            result["learned_at_endpoint_difference"][str(magnitude)] = describe_kernel(weights)
        result["tanh_sharpness"] = distribution(sharp)
    if "activation_mix" in state:
        coef = state["activation_mix"].double().tanh().numpy()
        result["softsign_plus_coefficient_times_tanh_minus_softsign"] = distribution(coef)
        result["negative_coefficient_channels"] = int((coef<0).sum())
    if "project_in.weight" in state:
        a,b = (state[k].double().numpy() for k in ("project_in.weight","project_out.weight"))
        residual = b @ a
        # Thin QR factors keep the exact nonzero singular values while
        # computing the final SVD in the small bottleneck space.
        _,ra = np.linalg.qr(a.T,mode="reduced")
        _,rb = np.linalg.qr(b,mode="reduced")
        result["residual_relative_frobenius_to_identity"] = float(np.linalg.norm(residual)/np.sqrt(a.shape[1]))
        result["residual_spectral_norm"] = float(np.linalg.svd(rb @ ra.T,compute_uv=False)[0])
    return result


def rate_diagnostics(config, paths, directory, metadata=None):
    directory.mkdir(parents=True, exist_ok=False)
    # Recount every phone and word against frozen development references, then
    # compute matched session/block bootstrap intervals before stratification.
    summarize(config, paths, directory / "errors.json", metadata)
    cfg = load_config(config)
    records = load_json(cfg["data_manifest"])
    split = load_split_manifest(cfg["split_manifest"], records)
    tuning = _selection(_by_id(records, "trial"), split, "dev_tune")
    refs = {r["trial_id"]: reference_only(r)["phoneme_ids"] for r in tuning}
    seconds = np.array([r["input_length"] * .020 for r in tuning])
    density = np.array([len(refs[r["trial_id"]]) for r in tuning]) / seconds
    cuts = {key: np.quantile(v, [.25, .5, .75]).tolist() for key, v in
            (("duration_seconds", seconds), ("target_phones_per_trial_second", density))}
    output = {"score_split": "dev_tune", "split_hash": split["manifest_hash"],
              "quantile_cuts": cuts, "quantile_ties": "assign equality to upper group",
              "counts": {}, "note": "Target density includes SIL and divides by the whole trial duration. "
                                    "These rate associations do not measure individual phone durations."}
    for path in paths:
        name = path.parent.name
        rows = _by_id(read_jsonl(path), name)
        counts, repeat, groups = Counter(), Counter(), defaultdict(Counter)
        digest = load_json(path.parent / "metrics.json")
        if digest["status"] != "complete":
            raise ValueError(f"{name} does not have a complete run receipt")
        if name in {"adaptive_filter", "activation_only", "transform_only", "input_transform"}:
            ckpt = path.parent / "checkpoints/last.pt"
            if sha256(ckpt) != digest["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for {name}")
            state = torch.load(ckpt,map_location="cpu",weights_only=True,mmap=True)
            if state["condition"] != name or state["step"] != digest["updates"]:
                raise ValueError("Checkpoint condition or step mismatch")
            output.setdefault("frontends", {})[name] = frontend_parameters(state["trainable"])
            del state
        for i, row in enumerate(tuning):
            trial = row["trial_id"]
            pred = rows[trial]
            expected = f"logits/{row['official_order_index']:05d}.npy"
            if pred.get("logits") != expected:
                raise ValueError(f"Unexpected logit identity for {trial}")
            logits = np.load(path.parent / expected, allow_pickle=False)
            best, acc = emission_stats(logits)
            if greedy_ctc(best.tolist()) != pred["phones"] or len(logits) != pred["emission_frames"]:
                raise ValueError(f"Argmax collapse/length disagrees with predictions on {trial}")
            target = refs[trial]
            acc.update(**pred["counts"])
            acc["ctc_minimum_frames"] += ctc_minimum_frames(target)
            acc["source_bins"] += row["input_length"]
            counts.update(acc)
            repeat.update(repeated_phone_counts(target, pred["phones"]))
            for key, val in (("duration_seconds", seconds[i]), ("target_phones_per_trial_second", density[i])):
                q = int(np.searchsorted(cuts[key], val, side="right")) + 1
                groups[f"{key}_Q{q}"].update(acc)
        def rates(a):
            return {"counts": a, "wer": a["word_errors"]/a["words"], "per": a["phone_errors"]/a["phones"],
                    "argmax_blank_fraction": a["blank_frames"]/a["frames"],
                    "mean_blank_probability": a["sum_blank_probability"]/a["frames"],
                    "mean_entropy_nats": a["sum_entropy_nats"]/a["frames"],
                    "frames_per_minimum_ctc_frame": a["frames"]/a["ctc_minimum_frames"]}
        output["counts"][name] = {**rates(counts), "repeat_neighbors": repeat,
                                  "strata": {key: rates(acc) for key, acc in groups.items()}}
    output["analysis_code_sha256"] = sha256(Path(__file__))
    write_json(directory / "cadence.json", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-metadata", type=Path)
    parser.add_argument("predictions", nargs="+", type=Path, help="Control first, then matched conditions")
    args = parser.parse_args()
    rate_diagnostics(args.config, args.predictions, args.output, args.corpus_metadata)
