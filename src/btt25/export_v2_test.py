"""Export ordered hidden-test text using the original v2 freeze's selected system."""
import argparse
from argparse import Namespace
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

from .assets import sha256
from .benchmark import feature_encoder, native_decoder, neural_ranker, timing_summary
from .ctc import greedy_ctc
from .data import object_hash, write_json
from .evaluation import export_submission
from .inference import load_config, load_json
from .report_v2 import STATUS, REQUIRED, hash_bundle
from .resources import ResourceMonitor, BudgetExceeded


def verify_native_module(python, directory):
    probe = """
from flashlight.lib.text import flashlight_lib_text_decoder as native
trie = native.Trie(41, 40)
for value in range(32):
    node = trie.insert([1, 1], value, 0.)
if list(node.labels) != list(range(32)):
    raise ValueError('Unexpected native terminal-label cap')
print(native.__file__)
"""
    env = dict(os.environ, PYTHONPATH=str(directory) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([str(python), "-c", probe], env=env, text=True, check=True,
                            capture_output=True, timeout=30)
    binary = Path(result.stdout.strip()).resolve()
    if not binary.is_relative_to(directory.resolve()):
        raise ValueError("The decoder loaded outside the repaired Flashlight directory")
    digest = sha256(binary)
    receipt = (directory.parent / "build_receipt.txt").read_text(encoding="utf-8")
    if f"{digest}  {binary}" not in receipt.splitlines():
        raise ValueError("The repaired native binding differs from its build receipt")
    return digest


def run(args):
    config = load_config(args.config)
    freeze = load_json(args.freeze)
    freeze_id = freeze.pop("manifest_hash")
    if (freeze_id != object_hash(freeze) or freeze["evaluation_status"] != STATUS
            or sha256(config["data_manifest"]) != freeze["data_manifest_sha256"]):
        raise ValueError("Could not verify the original v2 selection and release manifest")
    condition = freeze["selected_condition"]
    if condition != "v2-f1-distilgpt2":
        raise ValueError("This export uses the original v2 DistilGPT2 selection")
    binding = freeze["conditions"][condition]
    source = Path(binding["directory"]) / "decode"
    if hash_bundle(source, REQUIRED) != binding["decode"]:
        raise ValueError("Selected decoder artifacts changed after sealing")
    cache = load_json(source / "cache_manifest.json")
    native = cache["decoder"]
    if (cache["source_cache_manifest"]["checkpoint_hash"] != config["checkpoint_hash"]
            or native["variant"] != "unlimited-label-patch"
            or not args.decoder_module_path.is_dir()
            or sha256(args.lexicon) != cache["lexicon_sha256"]
            or sha256(cache["kenlm_model_path"]) != cache["kenlm_model_sha256"]):
        raise ValueError("Export encoder, runtime, or decoder assets differ from the selected system")
    settings = Namespace(decoder_python=args.decoder_python, lexicon=args.lexicon,
        decoder_module_path=args.decoder_module_path, nbest=native["nbest"], beam_threshold=native["beam_threshold"],
        kenlm_model=cache["kenlm_model_path"], lm_weight=native["lm_weight"], start_timeout=120, request_timeout=30)
    scorer = {"model": "distilgpt2", "path": str(args.distil), "revision": cache["fixed_neural_model"]["revision"],
              "batch_size": 8, "candidate_limit": cache["hypothesis_limit"], "weight": cache["neural_weight"]}
    records = load_json(config["data_manifest"])
    test = sorted((r for r in records if r["split"] == "test"), key=lambda row: row["official_order_index"])
    if not test or [r["official_order_index"] for r in test] != list(range(len(test))):
        raise ValueError("Expected complete and sequential hidden-test traversal")
    benchmark = load_json(args.freeze.parent / "feature-benchmark/metrics.json")
    measured = benchmark["by_condition"][condition]
    forecast = 2 * len(test) * measured["latency_seconds"]["p95"] + 180
    if benchmark["state"] != "complete" or measured["failures"] or forecast > args.max_seconds:
        raise ValueError("The measured feature-to-text benchmark does not admit this export window")
    native_hash = verify_native_module(args.decoder_python, args.decoder_module_path)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.resolved.yaml", {"freeze": freeze_id, "condition": condition,
        "source_decoder_cache": cache["manifest_hash"], "decoder": native, "native_binding_sha256": native_hash,
        "scorer": scorer,
        "data_manifest_sha256": freeze["data_manifest_sha256"], "checkpoint_sha256": config["checkpoint_hash"],
        "max_seconds": args.max_seconds, "forecast_seconds": forecast, "references_loaded": False})
    predictions, timing = [], []
    metrics = {"state": "failed", "expected_trials": len(test), "references_loaded": False, "hidden_test_score": None}
    try:
        with ResourceMonitor(out / "resources.json", max_seconds=args.max_seconds) as monitor, ExitStack() as stack:
            began = perf_counter()
            encode = feature_encoder(config)
            decode, decoder_setup = stack.enter_context(native_decoder(settings, native["beam"], out / "decoder.log"))
            rank, model = neural_ranker(scorer)
            if model["asset_manifest_sha256"] != cache["fixed_neural_model"]["asset_manifest_sha256"]:
                raise ValueError("DistilGPT2 asset receipt differs from the frozen scorer")
            metrics["cold_setup_s"] = perf_counter() - began
            metrics["native_setup_s"] = decoder_setup["native_setup_s"]
            saved = stack.enter_context((out / "predictions.jsonl").open("w", encoding="utf-8", buffering=1))
            trace = stack.enter_context((out / "timing.jsonl").open("w", encoding="utf-8", buffering=1))
            for record in test:
                monitor.check()
                start = perf_counter()
                trial = record["trial_id"]
                failure = None
                stages, phones, output_length = {}, [], None
                try:
                    logits, stages = encode(record)
                    phones, output_length = greedy_ctc(logits.argmax(-1)), len(logits)
                    choices = decode(trial, logits)["candidates"]
                    text, status = (rank(choices), "ok") if choices else ("", "decoder_error")
                except Exception as error:
                    failure = error
                    text, status = "", "timeout" if isinstance(error, TimeoutError) else "failed"
                prediction = {"trial_id": trial, "text": text, "status": status,
                              "phoneme_ids": phones, "output_length": output_length}
                row = {"latency_s": perf_counter() - start, "status": status,
                       "input_seconds": record["input_length"] * .020, "stages": stages}
                predictions.append(prediction)
                timing.append(row)
                saved.write(json.dumps(prediction) + "\n")
                trace.write(json.dumps({"trial_id": trial, **row}) + "\n")
                if failure:
                    raise failure
            monitor.check()
        if load_json(out / "resources.json")["outcome"] != "complete":
            raise BudgetExceeded("Export exceeded its resource envelope")
        submission = export_submission(predictions, records, out / "submission.csv")
        metrics.update(state="complete", submission_sha256=sha256(submission))
    except BudgetExceeded:
        metrics["state"] = "budget_stopped"
        raise
    finally:
        metrics.update(saved_trials=len(predictions), successful_trials=sum(r["status"] == "ok" for r in predictions),
                       timing=timing_summary(timing))
        write_json(out / "metrics.json", metrics)
        (out / "report.md").write_text(f"# V2 hidden-test export\n\nState `{metrics['state']}`. Saved {len(predictions)} "
                f"of {len(test)} trial predictions. Only coverage and official order can be verified.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "freeze", "lexicon", "decoder-python", "decoder-module-path", "distil", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
