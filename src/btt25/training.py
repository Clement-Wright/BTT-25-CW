"""Small head/session-bias calibration and sustained real-data profiling."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .ctc import ctc_minimum_frames, greedy_ctc
from .data import read_trial, object_hash, write_json,load_split_manifest
from .evaluation import edit_counts
from .inference import load_json
from .model import load_encoder, gaussian_kernel, smooth_valid, output_length
from .resources import ResourceMonitor, BudgetExceeded
from .assets import sha256
from functools import wraps
import inspect


def record_failures(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        bound=inspect.signature(function).bind(*args,**kwargs)
        directory=Path(bound.arguments["run_dir"])
        path=directory/"run_manifest.json"
        before=path.read_bytes() if path.exists() else None
        try:
            return function(*args,**kwargs)
        except BaseException as error:
            # Do not overwrite a previous completed run when preflight rejects reuse.
            if path.exists() and path.read_bytes()!=before:
                state=load_json(path)
                state.update(state="budget_stopped" if isinstance(error,(BudgetExceeded,KeyboardInterrupt)) else "failed",error=f"{type(error).__name__}: {error}")
                write_json(path,state)
            raise
    return wrapped


def configure_trainable(model, mode):
    if mode not in ("head", "session_bias"):
        raise ValueError("Adaptation must be head or session_bias")
    names = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith("out.") or (mode == "session_bias" and name.startswith("day_biases."))
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    # Frozen backbone dropout stays OFF in both conditions.
    model.eval()
    if mode == "session_bias":
        # cuDNN needs training-mode reserve buffers for dL/d(input). Frozen
        # weights do not remove that derivative. Disable stochastic dropout
        # explicitly rather than relying on eval-mode RNN backward support.
        model.gru.train()
        model.gru.dropout = 0.0
    return names


def train_step(model, batch, optimizer, mode, anchor=None, regularization=0):
    optimizer.zero_grad(set_to_none=True)
    features = batch["features"]
    kernel = gaussian_kernel().to(features.device)
    x = smooth_valid(features, kernel)
    days = torch.tensor([batch["day_index"]], device=features.device)
    target = batch["phoneme_ids"]
    n_output = output_length(features.shape[1], len(kernel), model.patch_size, model.patch_stride)
    if n_output < ctc_minimum_frames(target):
        raise ValueError("Infeasible CTC target after smoothing/patching, including repeats")
    if mode == "head":
        with torch.no_grad():
            hidden = model.encode(x, days)
    else:
        hidden = model.encode(x, days)
    logits = model.out(hidden)
    labels = torch.tensor(target, device=features.device, dtype=torch.long)
    loss = F.ctc_loss(logits.log_softmax(-1).transpose(0,1), labels, torch.tensor([n_output]), torch.tensor([len(target)]), blank=0, reduction="mean", zero_infinity=False)
    penalty = loss.new_zeros(())
    if mode == "session_bias" and anchor is not None and regularization:
        bias = model.day_biases[batch["day_index"]]
        penalty = regularization * (bias-anchor[batch["day_index"]]).square().mean()
    objective = loss + penalty
    if not torch.isfinite(objective):
        raise ValueError("Nonfinite CTC objective")
    objective.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        raise ValueError("Nonfinite gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, 5)
    optimizer.step()
    return {"loss": float(loss.detach()), "penalty": float(penalty.detach()), "gradient_norm": float(norm), "output_length": n_output}


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256(temporary)
    if path.is_file():
        previous = path.with_name(f"{path.stem}.previous{path.suffix}")
        link = previous.with_suffix(previous.suffix+".tmp")
        if link.exists():
            link.unlink()
        try:
            os.link(path, link)
        except OSError:
            shutil.copy2(path, link)
        link.replace(previous)
    temporary.replace(path)
    write_json(path.with_suffix(path.suffix+".sha256.json"), {"sha256":digest,"step":state.get("step")})


def restore_optimizer(model, state, lr):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return optimizer


def calibration_manifest(records, per_session=16, seed=20261116):
    sessions = sorted({r["session_id"] for r in records if r["split"] == "train"})
    selected = []
    for session in sessions:
        choices = [r for r in records if r["split"] == "train" and r["session_id"] == session]
        choices.sort(key=lambda r: object_hash([seed,r["trial_id"]]))
        if len(choices) < per_session:
            raise ValueError(f"Not enough allowed calibration trials for {session}")
        selected.extend(r["trial_id"] for r in choices[:per_session])
    result = {"seed":seed,"per_session":per_session,"trial_ids":selected,"source_split":"official train","claim":"Further calibration of known sessions already seen by the source checkpoint"}
    result["manifest_hash"] = object_hash(result)
    return result


def validation_diagnostic(model, records, device):
    errors, denominator = 0, 0
    predictions = []
    with torch.no_grad():
        for record in records:
            item = read_trial(record, include_labels=True)
            x = torch.from_numpy(item["features"]).unsqueeze(0).to(device)
            logits = model(smooth_valid(x,gaussian_kernel()),torch.tensor([record["day_index"]],device=device))[0]
            phones = greedy_ctc(logits.argmax(-1).tolist())
            counts = edit_counts(item["phoneme_ids"],phones)
            errors += counts["errors"]
            denominator += counts["reference_length"]
            predictions.append({"trial_id":record["trial_id"],"phoneme_ids":phones,"reference_phoneme_ids":item["phoneme_ids"],"reference_text":item["reference_text"],"text":"","status":"ok","diagnostic":True})
    return {"phoneme_errors":errors,"reference_phonemes":denominator,"per":errors/denominator,"trials":len(records),"predictions":predictions}


def profile_inputs(config):
    return {key: {"sha256":sha256(config[key])} for key in ("data_manifest","split_manifest","model_args")}


@record_failures
def profile(config, minutes, run_dir, mode="session_bias"):
    if not 15 <= minutes <= 30:
        raise ValueError("Sustained training pilot must last 15–30 minutes")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    if (run_dir/"run_manifest.json").exists():
        raise ValueError("Pilot directory already exists; preserve its evidence")
    torch.set_num_threads(config.get("cpu_threads",4))
    device = config.get("device","cuda")
    records = load_json(config["data_manifest"])
    splits = load_split_manifest(config["split_manifest"],records)
    write_json(run_dir/"input_snapshot.json",{"recording_scope":"Pilot inputs before model updates","files":profile_inputs(config)})
    ordered = sorted([r for r in records if r["split"]=="train"],key=lambda r:r["input_length"])
    # Fixed length quantiles plus the longest utterance, all real train data.
    indices = np.linspace(0,len(ordered)-1,24,dtype=int)
    train_records = [ordered[int(i)] for i in indices]
    tune_ids = set(splits["dev_tune"])
    vals = sorted([r for r in records if r["trial_id"] in tune_ids],key=lambda r:r["input_length"])
    validation = [vals[int(i)] for i in np.linspace(0,len(vals)-1,8,dtype=int)]
    executed = {**config,"pilot_minutes":minutes,"mode":mode,"train_ids":[r["trial_id"] for r in train_records],"validation_ids":[r["trial_id"] for r in validation],"diagnostic":True,"backbone_dropout":"eval/off","lr":1e-4,"regularization":1.0}
    (run_dir/"config.resolved.yaml").write_text(yaml.safe_dump(executed),encoding="utf-8")
    state = {"state":"profiling","mode":mode,"pid":__import__('os').getpid(),"started_at":datetime.now(timezone.utc).isoformat(),"diagnostic":True}
    write_json(run_dir/"run_manifest.json",state)
    durations, saves, validations = [], [], []
    with ResourceMonitor(run_dir/"resources.json",max_seconds=minutes*60+180) as monitor:
        model,_ = load_encoder(config["checkpoint"],config["model_args"],device,config["checkpoint_hash"])
        names = configure_trainable(model,mode)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        anchor = [p.detach().clone() for p in model.day_biases]
        start, step, next_event = time.monotonic(),0,0
        with (run_dir/"learning.jsonl").open("w",encoding="utf-8") as log:
            while time.monotonic()-start < minutes*60:
                monitor.check()
                record = train_records[step%len(train_records)]
                item = read_trial(record,include_labels=True)
                x = torch.from_numpy(item["features"]).unsqueeze(0).to(device)
                if device.startswith("cuda"): torch.cuda.synchronize()
                t=time.monotonic()
                result=train_step(model,{"features":x,"day_index":record["day_index"],"phoneme_ids":item["phoneme_ids"]},optimizer,mode,anchor,1.0)
                if device.startswith("cuda"): torch.cuda.synchronize()
                elapsed=time.monotonic()-t
                durations.append(elapsed)
                step+=1
                log.write(json.dumps({"step":step,"trial_id":record["trial_id"],"seconds":elapsed,**result})+"\n")
                if time.monotonic()-start >= next_event:
                    t=time.monotonic()
                    diagnostic=validation_diagnostic(model,validation,device)
                    validations.append(time.monotonic()-t)
                    t=time.monotonic()
                    atomic_checkpoint(run_dir/"checkpoints/last.pt",{"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"contract":model.contract,"step":step,"mode":mode,"config":executed,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"python_rng":random.getstate()})
                    saves.append(time.monotonic()-t)
                    write_json(run_dir/"diagnostic_metrics.json",diagnostic)
                    log.flush()
                    print(json.dumps({"stage":"pilot","elapsed_seconds":time.monotonic()-start,"steps":step,"last_loss":result["loss"],"diagnostic_per":diagnostic["per"]}),flush=True)
                    next_event+=120
        final=validation_diagnostic(model,validation,device)
        (run_dir/"predictions.jsonl").write_text("".join(json.dumps(p)+"\n" for p in final.pop("predictions")),encoding="utf-8")
        atomic_checkpoint(run_dir/"checkpoints/last.pt",{"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"contract":model.contract,"step":step,"mode":mode,"config":executed,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"python_rng":random.getstate()})
        write_json(run_dir/"metrics.json",{**final,"scope":"8 dev_tune trials; learning diagnostic only; not a completed research condition"})
        summary={"steps":step,"sustained_seconds":time.monotonic()-start,"step_median_seconds":float(np.median(durations)),"step_p95_seconds":float(np.quantile(durations,.95)),"early_mean_seconds":float(np.mean(durations[:max(1,len(durations)//5)])),"late_mean_seconds":float(np.mean(durations[-max(1,len(durations)//5):])),"checkpoint_median_seconds":float(np.median(saves)),"validation_8_median_seconds":float(np.median(validations)),"longest_train_input":train_records[-1]["input_length"],"trainable_names":names,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"backpropagation_through_backbone":mode=="session_bias"}
        write_json(run_dir/"profile.json",summary)
    state.update(state="diagnostic_complete",**summary)
    write_json(run_dir/"run_manifest.json",state)
    (run_dir/"report.md").write_text(f"# Sustained real-data pilot\n\n{minutes} minutes of {mode} updates; {step} steps. Includes all length quantiles, longest training trial, eight dev_tune validation trials and checkpoint I/O. This is a diagnostic, not a research result. See profile.json and resources.json.\n",encoding="utf-8")
    return summary


def admit(profile_dir, updates, final_seconds, max_hours=7.5, config=None, mode=None):
    evidence=load_json(Path(profile_dir)/"profile.json")
    resource=load_json(Path(profile_dir)/"resources.json")
    if evidence["sustained_seconds"]<900 or resource["stop_reason"]:
        raise ValueError("A successful sustained >=15-minute pilot is required")
    host_peak = max(resource.get("peak_accounted_host_bytes", resource["peak_tree_rss_bytes"]),
                    resource.get("peak_cgroup_bytes") or 0)
    if (host_peak >= resource.get("host_target_bytes", 12_000_000_000)
            or (not resource.get("slurm_requested_bytes")
                and resource["minimum_system_available_bytes"] < 2_000_000_000)):
        raise ValueError("Pilot did not leave the required host memory headroom")
    for gpu in resource["hardware"].get("gpu",[]):
        if resource["peak_device_used_bytes"]>0.8*gpu["total_vram_bytes"]:
            raise ValueError("Pilot does not leave 20% VRAM headroom")
    if config is not None:
        executed=yaml.safe_load((Path(profile_dir)/"config.resolved.yaml").read_text(encoding="utf-8"))
        for key in ("checkpoint_hash","contract_hash","data_manifest","split_manifest","device"):
            if executed.get(key)!=config.get(key): raise ValueError(f"Pilot does not match current {key}")
        snapshot=load_json(Path(profile_dir)/"input_snapshot.json")["files"]
        for key, current in profile_inputs(config).items():
            if snapshot.get(key,{}).get("sha256")!=current["sha256"]:
                raise ValueError(f"Pilot input content changed: {key}")
        load_split_manifest(config["split_manifest"],load_json(config["data_manifest"]))
        if executed["mode"]!=mode: raise ValueError("Pilot did not measure this adaptation mode")
    # Include the sustained wall time, which covers data reads and validation
    # as well as updates. Keep the measured p95 as the lower bound.
    step_seconds=evidence["step_p95_seconds"]
    if evidence.get("steps"):
        step_seconds=max(step_seconds,evidence["sustained_seconds"]/evidence["steps"])
    forecast=(updates*step_seconds+final_seconds+3*evidence["checkpoint_median_seconds"])*1.3
    if forecast>max_hours*3600:
        raise ValueError(f"Projected whole job {forecast:.1f}s exceeds requested budget")
    return {"admitted":True,"pilot":str(profile_dir),"updates":updates,"estimated_update_seconds":step_seconds,"forecast_seconds":forecast,"final_stage_seconds":final_seconds,"safety_multiplier":1.3,"target_seconds":max_hours*3600,"absolute_watchdog_seconds":28200}


def _archive_attempt(run_dir, parent_checkpoint_hash):
    """Copy attempt evidence before a continuation replaces the current files."""
    history=run_dir/"attempts"
    history.mkdir(exist_ok=True)
    indices=[int(p.name) for p in history.iterdir() if p.is_dir() and p.name.isdigit()]
    archive=history/f"{max(indices,default=0)+1:04d}"
    archive.mkdir()
    for name in ("run_manifest.json","admission.json","config.resolved.yaml",
                 "calibration_manifest.json","trainable_parameters.json","learning.jsonl",
                 "resources.json","resources.samples.jsonl",
                 "resources-resume.json","resources-resume.samples.jsonl","inference.yaml"):
        source=run_dir/name
        if source.is_file():
            shutil.copy2(source,archive/name)
    write_json(archive/"checkpoint_provenance.json",{"checkpoint":"checkpoints/last.pt","sha256":parent_checkpoint_hash,
               "note":"Hash of the checkpoint loaded by the continuation; weights remain in the current checkpoint path."})
    return archive.relative_to(run_dir).as_posix()


@record_failures
def train(config, run_dir, mode, updates, profile_dir, max_hours=7.5, resume=False, stop_after_updates=None):
    run_dir=Path(run_dir)
    target=run_dir/"checkpoints/last.pt"
    prior_state=None
    if resume:
        if not (run_dir/"run_manifest.json").is_file() or not target.is_file():
            raise ValueError("Resume requires a previous run manifest and checkpoint")
        prior_state=load_json(run_dir/"run_manifest.json")
        if prior_state.get("state") not in {"training","budget_stopped","failed"}:
            raise ValueError("Resume requires an interrupted training attempt")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Existing training directory contains evidence; use resume or a new directory")
    run_dir.mkdir(parents=True,exist_ok=True)
    executed={**config,"mode":mode,"updates":updates,"lr":config.get("lr",1e-5),"per_session":config.get("calibration_per_session",16),"seed":config.get("seed",20261116),"regularization":1.0,"backbone_dropout":0,"selection":"last checkpoint at fixed matched update count; no report selection"}
    job_admission=admit(profile_dir,updates,config.get("reserved_final_seconds",1800),max_hours,config,mode)
    config_path=run_dir/"config.resolved.yaml"
    if config_path.exists() and yaml.safe_load(config_path.read_text(encoding="utf-8"))!=executed:
        raise ValueError("Resume configuration changed")
    records=load_json(config["data_manifest"])
    load_split_manifest(config["split_manifest"],records)
    calibration=calibration_manifest(records,executed["per_session"],executed["seed"])
    parent_hash=sha256(target) if resume else config["checkpoint_hash"]
    previous_attempt=_archive_attempt(run_dir,parent_hash) if resume else None
    write_json(run_dir/"admission.json",job_admission)
    config_path.write_text(yaml.safe_dump(executed),encoding="utf-8")
    write_json(run_dir/"calibration_manifest.json",calibration)
    by_id={r["trial_id"]:r for r in records}
    ids=calibration["trial_ids"][:]
    random.Random(executed["seed"]).shuffle(ids)
    state={"state":"training","pid":__import__('os').getpid(),"mode":mode,"diagnostic":False,
           "source_checkpoint_hash":config["checkpoint_hash"],"parent_checkpoint_hash":parent_hash,
           "started_at":datetime.now(timezone.utc).isoformat(),"resume":resume,
           "attempt":prior_state.get("attempt",1)+1 if resume else 1,"previous_attempt":previous_attempt}
    write_json(run_dir/"run_manifest.json",state)
    torch.set_num_threads(config.get("cpu_threads",4))
    torch.manual_seed(executed["seed"])
    random.seed(executed["seed"])
    device=config.get("device","cuda")
    start_step=0
    previous_seconds=0
    with ResourceMonitor(run_dir/("resources-resume.json" if resume else "resources.json"),max_seconds=min(max_hours*3600,28200)) as monitor:
        source,_=load_encoder(config["checkpoint"],config["model_args"],device,config["checkpoint_hash"])
        anchor=[p.detach().clone() for p in source.day_biases]
        model=source
        if resume:
            if not target.is_file(): raise FileNotFoundError("No valid checkpoint to resume")
            # Model first, optimizer second: never bind an optimizer to discarded weights.
            model,_=load_encoder(target,config["model_args"],device)
            del source
        names=configure_trainable(model,mode)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=executed["lr"])
        if resume:
            from .model import safe_load
            previous=safe_load(target)
            if previous.get("calibration_hash")!=calibration["manifest_hash"] or previous.get("config_hash")!=object_hash(executed):
                raise ValueError("Resume calibration/configuration mismatch")
            optimizer=restore_optimizer(model,previous,executed["lr"])
            start_step=previous["step"]
            previous_seconds=previous.get("cumulative_training_seconds",0)
            torch.set_rng_state(previous["torch_rng"])
            if torch.cuda.is_available(): torch.cuda.set_rng_state_all(previous["cuda_rng"])
            random.setstate(previous["python_rng"])
            del previous
        write_json(run_dir/"trainable_parameters.json",{"names":names,"count":sum(p.numel() for p in model.parameters() if p.requires_grad),"backprop_through_frozen_gru":mode=="session_bias","gru_training_mode":model.gru.training,"gru_dropout":model.gru.dropout})
        start=time.monotonic()
        step=start_step
        def save():
            checkpoint={"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"contract":model.contract,"step":step,"mode":mode,"config_hash":object_hash(executed),"calibration_hash":calibration["manifest_hash"],"source_checkpoint_hash":config["checkpoint_hash"],"cumulative_training_seconds":previous_seconds+time.monotonic()-start,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],"python_rng":random.getstate()}
            atomic_checkpoint(target,checkpoint)
        try:
            with (run_dir/"learning.jsonl").open("a",encoding="utf-8") as log:
                while step<updates:
                    monitor.check()
                    if stop_after_updates is not None and step-start_step>=stop_after_updates:
                        raise BudgetExceeded("Requested interruption/resume diagnostic")
                    # Reserve measured final-stage and save time in the whole-job target.
                    if time.monotonic()-monitor.start>max_hours*3600-config.get("reserved_final_seconds",1800):
                        raise BudgetExceeded("Training cutoff reserves inference/decoding/reporting")
                    record=by_id[ids[step%len(ids)]]
                    item=read_trial(record,include_labels=True)
                    x=torch.from_numpy(item["features"]).unsqueeze(0).to(device)
                    result=train_step(model,{"features":x,"day_index":record["day_index"],"phoneme_ids":item["phoneme_ids"]},optimizer,mode,anchor,1.0)
                    step+=1
                    log.write(json.dumps({"step":step,"trial_id":record["trial_id"],**result})+"\n")
                    if step%100==0:
                        save()
                        log.flush()
                        print(json.dumps({"stage":"calibration","mode":mode,"step":step,"updates":updates,"last_loss":result["loss"]}),flush=True)
                save()
        except (BudgetExceeded,KeyboardInterrupt) as error:
            try: save()
            except BaseException as save_error:
                state.update(state="failed",checkpoint_save_error=f"{type(save_error).__name__}: {save_error}")
                write_json(run_dir/"run_manifest.json",state)
                raise
            state.update(state="budget_stopped",step=step,reason=str(error),checkpoint_hash=sha256(target),resume_command=f"btt25 resume --config <same config> --run-dir {run_dir.as_posix()} --mode {mode} --updates {updates} --profile-dir {profile_dir}")
            write_json(run_dir/"run_manifest.json",state)
            return state
        except BaseException as error:
            try: save()
            except BaseException as save_error:
                state["checkpoint_save_error"]=f"{type(save_error).__name__}: {save_error}"
            state.update(state="failed",step=step,error=f"{type(error).__name__}: {error}")
            write_json(run_dir/"run_manifest.json",state)
            raise
        state.update(state="training_complete_inference_pending",step=step,checkpoint_hash=sha256(target),cumulative_training_seconds=previous_seconds+time.monotonic()-start)
    write_json(run_dir/"run_manifest.json",state)
    # Last fixed-budget state is the selected state; no favorable-number search.
    shutil.copy2(target,run_dir/"checkpoints/best.pt")
    result_config={**config,"checkpoint":str(target),"checkpoint_hash":state["checkpoint_hash"]}
    (run_dir/"inference.yaml").write_text(yaml.safe_dump(result_config),encoding="utf-8")
    return state
