"""Create local manifests/config from authorized assets; never reads references."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import zipfile

import yaml

from btt25.assets import MODEL_SHA256, sha256
from btt25.data import build_manifest,freeze_split,write_json,object_hash
from btt25.model import config_contract,OFFICIAL_CHECKPOINT_SHA256,OFFICIAL_ARGS_SHA256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--model-zip",default="assets/local/t15_pretrained_rnn_baseline.zip")
    parser.add_argument("--asset-dir",default="assets/local")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=parser.parse_args()
    archive=Path(args.model_zip)
    if sha256(archive)!=MODEL_SHA256: raise ValueError("Official model ZIP hash mismatch")
    asset=Path(args.asset_dir)
    dest=(asset/"official_model").resolve()
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as file:
        for info in file.infolist():
            if info.filename.startswith("__MACOSX/") or info.is_dir(): continue
            target=(dest/info.filename).resolve()
            if not target.is_relative_to(dest): raise ValueError("Unsafe archive path")
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                partial=target.with_suffix(target.suffix+".partial")
                with file.open(info) as src,partial.open("wb") as out:
                    shutil.copyfileobj(src,out)
                partial.replace(target)
    checkpoint=dest/"t15_pretrained_rnn_baseline/checkpoint/best_checkpoint"
    model_args=checkpoint.parent/"args.yaml"
    if sha256(checkpoint)!=OFFICIAL_CHECKPOINT_SHA256 or sha256(model_args)!=OFFICIAL_ARGS_SHA256:
        raise ValueError("Expanded official checkpoint/args mismatch")
    settings=yaml.safe_load(model_args.read_text(encoding="utf-8"))
    contract=config_contract(settings)
    records=build_manifest(args.data_root,settings["dataset"]["sessions"])
    split=freeze_split(records)
    split_path=asset/"splits.json"
    if split_path.exists() and __import__('json').loads(split_path.read_text())!=split:
        raise ValueError("Frozen split differs; preserve it")
    write_json(asset/"data_manifest.json",records)
    write_json(split_path,split)
    write_json(asset/"official_contract.json",{"contract":contract,"contract_hash":object_hash(contract),"checkpoint_hash":OFFICIAL_CHECKPOINT_SHA256})
    config={"checkpoint":str(checkpoint),"model_args":str(model_args),"checkpoint_hash":OFFICIAL_CHECKPOINT_SHA256,"contract_hash":object_hash(contract),"data_manifest":str(asset/"data_manifest.json"),"split_manifest":str(split_path),"device":args.device,"cpu_threads":4,"max_seconds":28200}
    if (asset/"corpus_metadata.csv").exists(): config["corpus_metadata"]=str(asset/"corpus_metadata.csv")
    Path("configs/local.yaml").write_text(yaml.safe_dump(config),encoding="utf-8")
    print({"counts":dict(Counter(r["split"] for r in records)),"dev_tune":len(split["dev_tune"]),"dev_report":len(split["dev_report"]),"references_loaded":False})


if __name__=="__main__": main()
