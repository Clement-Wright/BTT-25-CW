"""Optional real-data parity audit against separately acquired upstream code."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from btt25.data import read_trial, write_json
from btt25.inference import load_config, load_json
from btt25.model import load_encoder, gaussian_kernel, smooth_valid


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    torch.set_num_threads(4)
    config = load_config(options.config)
    device = config.get("device", "cuda")
    model, _ = load_encoder(config["checkpoint"], config["model_args"], device, config["checkpoint_hash"])
    source = import_file("upstream_rnn", Path(options.upstream)/"rnn_model.py")
    smoothing = import_file("upstream_smooth", Path(options.upstream)/"data_augmentations.py")
    with torch.device("meta"):
        upstream = source.GRUDecoder(512, model.n_units, model.n_days, 41, n_layers=model.n_layers, patch_size=model.patch_size, patch_stride=model.patch_stride)
    upstream.load_state_dict(model.state_dict(), strict=True, assign=True)
    upstream.eval()
    records = [r for r in load_json(config["data_manifest"]) if r["split"] == "val"]
    ordered = sorted(records, key=lambda r:r["input_length"])
    selected = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
    results = []
    with torch.inference_mode():
        for record in selected:
            x = torch.from_numpy(read_trial(record)["features"]).unsqueeze(0).to(device)
            ours = smooth_valid(x, gaussian_kernel())
            theirs = smoothing.gauss_smooth(x, device, padding="valid")
            days = torch.tensor([record["day_index"]],device=device)
            a, b = model(ours, days), upstream(theirs, days)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
            results.append({"trial_id":record["trial_id"],"input_length":len(x[0]),"output_length":a.shape[1],"max_smoothing_deviation":float((ours-theirs).abs().max()),"max_logit_deviation":float((a-b).abs().max())})
        short, long = selected[0], selected[-1]
        x = torch.from_numpy(read_trial(short)["features"]).unsqueeze(0).to(device)
        y = torch.from_numpy(read_trial(long)["features"]).unsqueeze(0).to(device)
        padded = torch.zeros(2,y.shape[1],512,device=device)
        padded[0,:x.shape[1]],padded[1] = x[0],y[0]
        alone=model(smooth_valid(x,gaussian_kernel()),torch.tensor([short["day_index"]],device=device))
        together=model(smooth_valid(padded,gaussian_kernel()),torch.tensor([short["day_index"],long["day_index"]],device=device))
        torch.testing.assert_close(alone[0],together[0,:alone.shape[1]],rtol=1e-4,atol=1e-4)
        result={"status":"passed","references_loaded":False,"fixtures":results,"padding_max_logit_deviation":float((alone[0]-together[0,:alone.shape[1]]).abs().max()),"checkpoint_hash":model.checkpoint_hash,"contract_hash":model.contract_hash}
    write_json(options.output,result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
