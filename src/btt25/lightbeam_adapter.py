"""Persistent T15-emission adapter for a private, pinned BrainAudio checkout."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from math import isfinite, log
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from .assets import sha256
from .benchmark import cached_encoder, timing_summary
from .ctc import greedy_ctc
from .data import TOKENS, load_split_manifest, reference_only, write_json
from .evaluation import edit_counts, normalize_text
from .inference import load_config, load_json
from .qwen_scorer import QwenScorer, cluster_guard
from .resources import BudgetExceeded, ResourceMonitor


REV = "90e0ac086d4dee643be154f446a6c374600575d7"


class ExactLikelihood:
    """Route the two pinned upstream hooks through the same exact-token scorer."""
    def __init__(self, scorer):
        self.scorer = scorer
        self.eos = False
        self.calls = Counter()
        self.scores = {}

    def reset(self):
        self.calls.clear()
        self.scores.clear()

    def __call__(self, model, tokenizer, texts, device, chunk_size=8, length_normalize=False):
        if (model is not self.scorer.model or tokenizer is not self.scorer.tokenizer
                or torch.device(device) != self.scorer.device or length_normalize):
            raise ValueError("LightBeam requested an unexpected scoring contract")
        values = self.scorer.score(texts, add_eos=self.eos, batch_size=chunk_size)
        key = "eos" if self.eos else "prefix"
        self.calls[key + "_calls"] += 1
        self.calls[key + "_candidates"] += len(values)
        self.calls[key + "_target_tokens"] += sum(count for _, count in values)
        self.scores.update(((self.eos, text), value) for text, (value, _) in zip(texts, values, strict=True))
        return [value for value, _ in values]


def inputs(args, config, output):
    records = load_json(config["data_manifest"])
    split = load_split_manifest(config["split_manifest"], records)
    allowed = set(split["dev_tune"])
    ordered = sorted((r for r in records if r["trial_id"] in allowed),
                     key=lambda r: (r["input_length"], r["official_order_index"]))
    selected = [ordered[i] for i in np.linspace(0, len(ordered) - 1, 24, dtype=int)]
    tokens = output / "ordered_tokens.txt"
    tokens.write_text("\n".join([*TOKENS[:-1], "|"]) + "\n", encoding="utf-8")
    lexicon = output / "lexicon.txt"
    entries = args.lexicon.read_text(encoding="utf-8").splitlines()
    phones = set(TOKENS[1:-1])
    if not entries or any(len(row.split()) < 2 or set(row.split()[1:]) - phones for row in entries):
        raise ValueError("LightBeam lexicon contains invalid pronunciations")
    lexicon.write_text("".join(line + " |\n" for line in entries), encoding="utf-8")
    return selected, tokens, lexicon, cached_encoder(args.logits, config, records, split)


def decode_to_candidates(result, history, ngram, scorer):
    beam_scores = result.scores[0].tolist()
    by_text = {}
    for i in sorted(range(len(beam_scores)), key=beam_scores.__getitem__, reverse=True):
        contexts = result.context_texts[0][i]
        if not contexts or not isfinite(beam_scores[i]):
            continue
        context, _, text_id = contexts[0]
        text = history.get_text(text_id)
        if not text or text in by_text:
            continue
        raw_ngram = ngram.model.score(normalize_text(text), bos=True, eos=False) * log(10)
        eos_text = text[0].upper() + text[1:]
        neural = scorer.scores.get((True, eos_text)) if scorer else None
        by_text[text] = {"text": text, "score": float(beam_scores[i]),
                         "component_scores": {"upstream_context": float(context),
                                              "acoustic_residual_after_context": float(beam_scores[i] - context),
                                              "completed_text_ngram_natural_log": float(raw_ngram),
                                              "eos_neural_natural_log": neural,
                                              "insertion_bonus": 0.0}}
    return list(by_text.values())


def compare(args):
    cluster_guard()
    if not torch.cuda.is_available():
        raise RuntimeError("LightBeam comparison requires its requested allocated CUDA device")
    registration = load_json(args.registration)
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != REV or revision != registration["upstream_revision"]:
        raise ValueError("LightBeam upstream source revision mismatch")
    if registration["reference_access"] != "dev_tune":
        raise ValueError("The adapter reports only the declared tuning subset")
    from brainaudio.inference.decoder import (
        BatchedBeamCTCComputer, LLMRescorer, VectorizedLexiconConstraint, neural_lm_rescoring,
    )
    from brainaudio.inference.decoder import ctc_batched_beam_decoding as decoder_module
    from brainaudio.inference.decoder.word_ngram_lm_optimized import FastNGramLM, WordHistory

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    selected, tokens, lexicon_path, read_emissions = inputs(args, cfg, out)
    arms = registration["conditions"]
    provenance = {"upstream_revision": revision, "registration_sha256": sha256(args.registration),
                  "lexicon_sha256": sha256(lexicon_path), "tokens_sha256": sha256(tokens),
                  "kenlm_sha256": sha256(args.kenlm), "args_sha256": sha256(args.config),
                  "adapter_sha256": sha256(__file__), "references_loaded_during_decode": False}
    write_json(out / "config.resolved.yaml", {**registration, "input_ids": [r["trial_id"] for r in selected],
               "data_config": cfg, "arms": arms, "provenance": provenance})
    traces = {name: [] for name in arms}
    predictions = {name: {} for name in arms}
    result = {"status": "running", "admitted_as": "sustained diagnostic", "passes": 0}
    write_json(out / "run_manifest.json", result)
    try:
        with ResourceMonitor(out / "resources.json", max_seconds=args.max_seconds) as monitor, ExitStack() as stack:
            setup = time.perf_counter()
            device = torch.device("cuda:0")
            lexicon = VectorizedLexiconConstraint.from_file_paths(tokens_file=str(tokens), lexicon_file=str(lexicon_path),
                                                                  device=device)
            if lexicon.blank_index != 0 or lexicon.word_boundary_token != 40:
                raise ValueError("LightBeam token boundary contract mismatch")
            ngram = FastNGramLM(str(args.kenlm), alpha=registration["ngram_log_weight"], unk_score_offset=-10.0)
            qwen = QwenScorer(args.model_dir)
            exact = ExactLikelihood(qwen)
            eos_function = decoder_module.apply_llm_eos_scoring

            def record_eos(*call_args, **kwargs):
                exact.eos = True
                try:
                    return eos_function(*call_args, **kwargs)
                finally:
                    exact.eos = False

            stack.enter_context(patch.object(neural_lm_rescoring, "score_texts_batch", exact))
            stack.enter_context(patch.object(decoder_module, "apply_llm_eos_scoring", record_eos))
            fusion = LLMRescorer(model=qwen.model, tokenizer=qwen.tokenizer, device=device,
                                llm_weight=registration["qwen_log_weight"], ngram_weight=0.0,
                                scoring_chunk_size=8, length_normalize=False)
            result.update(model=qwen.metadata, cold_setup_s=time.perf_counter() - setup)
            warm_start = time.perf_counter()
            with torch.inference_mode(), (out / "timing.jsonl").open("w", encoding="utf-8") as timing, (
                out / "predictions.jsonl").open("w", encoding="utf-8") as saved:
                while result["passes"] < 1 or time.perf_counter() - warm_start < registration["pilot_seconds"]:
                    for record in selected:
                        for name in list(arms)[result["passes"] % 3:] + list(arms)[:result["passes"] % 3]:
                            monitor.check()
                            began = time.perf_counter()
                            interval = arms[name]
                            array, _ = read_emissions(record)
                            exact.reset()
                            ngram.reset_cache()
                            history = WordHistory()
                            decoder = BatchedBeamCTCComputer(
                                blank_index=0, beam_size=registration["initial_beam"], lexicon=lexicon,
                                lm_fusion=fusion if interval is not None else None,
                                allow_cuda_graphs=False, num_homophone_beams=registration["initial_contexts_per_beam"],
                                beam_threshold=25.0, homophone_prune_threshold=25.0, beam_beta=0.0,
                                word_boundary_bonus=0.0, word_ngram_lm=ngram, word_history=history,
                                lm_rescore_interval=interval or 0, score_combination="logsumexp",
                            )
                            x = torch.from_numpy(array).to(device)
                            x = F.pad(x, (0, 0, 0, 2)).log_softmax(-1).unsqueeze(0)
                            decoded = decoder(x, torch.tensor([x.shape[1]], device=device))
                            choices = decode_to_candidates(decoded, history, ngram, exact if interval is not None else None)
                            torch.cuda.synchronize(device)
                            elapsed = time.perf_counter() - began
                            status = "ok" if choices else "decoder_error"
                            trial = record["trial_id"]
                            row = {"condition": name, "trial_id": trial, "status": status,
                                   "pass": result["passes"] + 1, "latency_s": elapsed,
                                   "input_seconds": record["input_length"] * .020, "stages": {},
                                   "lm_work": dict(exact.calls)}
                            traces[name].append(row)
                            timing.write(json.dumps(row) + "\n")
                            timing.flush()
                            if result["passes"] == 0:
                                prediction = {**row, "text": choices[0]["text"] if choices else "",
                                              "candidates": choices, "phoneme_ids": greedy_ctc(array.argmax(-1).tolist()),
                                              "input_length": record["input_length"], "output_length": len(array)}
                                predictions[name][trial] = prediction
                                saved.write(json.dumps(prediction, allow_nan=False) + "\n")
                                saved.flush()
                    result["passes"] += 1
                    write_json(out / "run_manifest.json", result)
                result["warm_seconds"] = time.perf_counter() - warm_start
                monitor.check()
        # Keep scorer and decoder execution isolated from those labels. Each missing
        # beam output is an empty hypothesis, still scored against its reference.
        if load_json(out / "resources.json")["outcome"] != "complete":
            raise BudgetExceeded("LightBeam exceeded its resource envelope at the final sample")
        references = {r["trial_id"]: normalize_text(reference_only(r)["reference_text"]).split() for r in selected}
        result["rows"] = {}
        for name, prediction_map in predictions.items():
            edits = Counter()
            oracle = 0
            for trial, ref in references.items():
                row = prediction_map[trial]
                edits.update(edit_counts(ref, normalize_text(row["text"]).split()))
                oracle += min((edit_counts(ref, normalize_text(c["text"]).split())["errors"]
                               for c in row["candidates"]), default=len(ref))
            work = Counter()
            for sample in traces[name]:
                work.update(sample["lm_work"])
            result["rows"][name] = {**edits, "wer": edits["errors"] / edits["reference_length"],
                                    "oracle_errors": oracle, "oracle_wer": oracle / edits["reference_length"],
                                    "timing": timing_summary(traces[name]), "lm_work": work}
        result.update(status="complete", scope="predeclared 24-trial tuning subset", provenance=provenance)
    except BaseException as error:
        result.update(status="budget_stopped" if isinstance(error, (BudgetExceeded, KeyboardInterrupt)) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out / "metrics.json", result)
        write_json(out / "run_manifest.json", result)
        (out / "report.md").write_text(f"# LightBeam feasibility comparison\n\nState `{result['status']}`. "
            "See metrics.json for trial scope, errors, timing, and candidate/token work. "
            "Only the declared tuning subset was eligible for scoring.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "config", "upstream", "logits", "kenlm", "lexicon", "model-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1500)
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
