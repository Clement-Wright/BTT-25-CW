"""Command entry points with explicit reference-access boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import yaml

from .data import build_manifest, freeze_split, object_hash, reference_only, write_json, attach_corpus_metadata
from .inference import inference, load_json, load_config, read_jsonl


def checked_json(path):
    value = load_json(path)
    if isinstance(value, dict) and "manifest_hash" in value:
        expected = object_hash({k: v for k, v in value.items() if k != "manifest_hash"})
        if expected != value["manifest_hash"]:
            raise ValueError(f"Manifest content hash mismatch: {path}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog="btt25")
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("acquire-model")
    acquire.add_argument("--output", default="assets/local/t15_pretrained_rnn_baseline.zip")
    inv = commands.add_parser("inventory")
    inv.add_argument("--roots", nargs="+", required=True)
    inv.add_argument("--output-dir", required=True)
    manifest = commands.add_parser("validate-data")
    manifest.add_argument("--data-root", required=True)
    manifest.add_argument("--model-args", required=True)
    manifest.add_argument("--output-dir", required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--config", required=True)
    infer.add_argument("--split", choices=["train", "val", "test"], default="val")
    infer.add_argument("--run-dir", required=True)
    infer.add_argument("--limit", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--score-split", choices=["dev_tune", "dev_report", "official_val"], default="dev_tune")
    evaluate.add_argument("--freeze-manifest")
    evaluate.add_argument("--artifact-registry", "--expected-hashes", dest="artifact_registry", help="Live artifact paths for both studies; bare hash claims are rejected")
    evaluate.add_argument("--phoneme-only", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--predictions", required=True)
    export.add_argument("--manifest", required=True)
    export.add_argument("--output", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--artifact-registry",required=True)
    freeze.add_argument("--output",required=True)
    freeze.add_argument("--selection-rules",required=True)
    pilot = commands.add_parser("profile")
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--minutes", type=int, default=15)
    pilot.add_argument("--mode", choices=["head", "session_bias"], default="session_bias")
    pilot.add_argument("--run-dir", required=True)
    for name in ("train","resume"):
        training=commands.add_parser(name)
        training.add_argument("--config",required=True)
        training.add_argument("--run-dir",required=True)
        training.add_argument("--mode",choices=["head","session_bias"],required=True)
        training.add_argument("--updates",type=int,default=720)
        training.add_argument("--profile-dir",required=True)
        training.add_argument("--max-hours",type=float,default=7.5)
        training.add_argument("--stop-after-updates",type=int)
    args = parser.parse_args(argv)
    if args.command == "acquire-model":
        from .assets import acquire_model
        print(json.dumps(acquire_model(args.output)))
    elif args.command == "inventory":
        from .inventory import inventory
        result=inventory(args.roots,args.output_dir)
        print(json.dumps({"repositories":len(result["repositories"]),"assets":len(result["assets"]),"output":str(Path(args.output_dir)/"inventory.json")}))
    elif args.command == "validate-data":
        model_args = yaml.safe_load(Path(args.model_args).read_text(encoding="utf-8"))
        records = build_manifest(args.data_root, model_args["dataset"]["sessions"])
        split = freeze_split(records)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Rebuilding a frozen membership is an error unless byte-identical.
        if (out / "splits.json").exists() and checked_json(out / "splits.json") != split:
            raise ValueError("Existing frozen validation membership differs; preserve it")
        write_json(out / "data_manifest.json", records)
        write_json(out / "splits.json", split)
        print(json.dumps({"counts": dict(Counter(r["split"] for r in records)), "dev_tune": len(split["dev_tune"]), "dev_report": len(split["dev_report"]), "split_hash": split["manifest_hash"]}))
    elif args.command == "infer":
        print(json.dumps(inference(load_config(args.config), args.split, args.run_dir, args.limit)))
    elif args.command == "evaluate":
        from .assets import sha256
        from .evaluation import evaluate_predictions
        from .reporting import write_evaluation
        from .resources import ResourceMonitor
        config = load_config(args.config)
        metadata = {"checkpoint_hash": config["checkpoint_hash"], "contract_hash": config["contract_hash"],
                    "predictions_sha256": sha256(args.predictions), "text_available": not args.phoneme_only}
        records=load_json(config["data_manifest"])
        if config.get("corpus_metadata"):
            records=attach_corpus_metadata(records,config["corpus_metadata"])
        frozen=checked_json(args.freeze_manifest) if args.freeze_manifest else None
        expected=None
        if args.score_split!="dev_tune":
            if not args.artifact_registry or not frozen:
                raise ValueError("Reporting requires the joint freeze and live artifact registry")
            from .freeze import validate_score_inputs
            expected=validate_score_inputs(args.artifact_registry,frozen,args.predictions,args.config)
        output=Path(args.output_dir)
        output.mkdir(parents=True,exist_ok=True)
        resolved={**config,"score_split":args.score_split,"source_predictions":args.predictions,**metadata}
        (output/"config.resolved.yaml").write_text(yaml.safe_dump(resolved),encoding="utf-8")
        with ResourceMonitor(output/"resources.json",max_seconds=config.get("max_seconds",28200)) as monitor:
            def selected_reference(record):
                monitor.check()
                return reference_only(record)
            result = evaluate_predictions(read_jsonl(args.predictions), records, checked_json(config["split_manifest"]), selected_reference, args.score_split, frozen, expected)
            if args.phoneme_only:
                result["metrics"]["wer"] = None
                result["metrics"]["word_metric_status"] = "not measured; no word decoder"
            write_evaluation(result, output, metadata)
        print(json.dumps(result["metrics"]))
    elif args.command == "export":
        from .evaluation import export_submission
        print(str(export_submission(read_jsonl(args.predictions), load_json(args.manifest), args.output)))
    elif args.command == "freeze":
        from .freeze import seal
        print(json.dumps(seal(args.artifact_registry,args.output,args.selection_rules)))
    elif args.command == "profile":
        from .training import profile
        print(json.dumps(profile(load_config(args.config), args.minutes, args.run_dir, args.mode)))
    elif args.command in ("train","resume"):
        from .training import train
        print(json.dumps(train(load_config(args.config),args.run_dir,args.mode,args.updates,args.profile_dir,args.max_hours,args.command=="resume",args.stop_after_updates)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
