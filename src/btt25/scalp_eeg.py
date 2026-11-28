"""A fixed four-direction, held-session classifier for public scalp EEG."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.signal import welch
from scipy.stats import binomtest

from btt25.assets import sha256
from btt25.data import write_json
from btt25.resources import ResourceMonitor


WORDS = ("arriba","abajo","derecha","izquierda")
BANDS = ((4,8),(8,13),(13,20),(20,40))


def decode_i24(octets):
    val = octets[...,0].astype(np.int32) + (octets[...,1].astype(np.int32)<<8) + (octets[...,2].astype(np.int32)<<16)
    return (val ^ (1<<23)) - (1<<23)


class Recording:
    """Read only complete, equal-rate BioSemi records and requested scalp windows."""
    def __init__(self, path):
        with path.open("rb") as f:
            head = f.read(256)
            nchan = int(head[252:256])
            header = int(head[184:192])
            if head[:8]!=b"\xffBIOSEMI" or header!=256*(nchan+1):
                raise ValueError("Invalid BioSemi BDF header")
            duration = float(head[244:252])
            def fields(width):
                return [f.read(width).decode("latin-1").strip() for _ in range(nchan)]
            self.labels = fields(16)
            fields(80)
            units = fields(8)
            pmin,pmax,dmin,dmax = [np.asarray(fields(8),dtype=float) for _ in range(4)]
            fields(80)
            samples = np.asarray(fields(8),dtype=int)
        if not np.all(samples==samples[0]) or duration<=0 or samples[0]/duration!=1024:
            raise ValueError("Expected native equal-rate 1024 Hz channel sampling")
        if self.labels[-1].lower()!="status":
            raise ValueError("Missing BioSemi status channel")
        expected = [f"{letter}{i}" for letter in "ABCD" for i in range(1,33)] + ["EXG1","EXG2"]
        if len(set(self.labels))!=len(self.labels) or not set(expected)<=set(self.labels):
            raise ValueError(f"Unexpected electrode labels: {self.labels}")
        self.picks = [self.labels.index(x) for x in expected]
        if not all(units[i] in ("uV","µV") for i in self.picks):
            raise ValueError("Expected microvolt EEG scales")
        self.gain = (pmax[self.picks]-pmin[self.picks])/(dmax[self.picks]-dmin[self.picks])
        self.offset = pmin[self.picks]-dmin[self.picks]*self.gain
        self.block = int(samples[0])
        n, self.trailing_bytes = divmod(path.stat().st_size-header,nchan*self.block*3)
        self.data = np.memmap(path,dtype="u1",mode="r",offset=header,shape=(n,nchan,self.block,3))
        self.header_records = int(head[236:244])
        self.complete_records = n

    def events(self):
        trigger = self.data[:,-1,:,0].astype("u2") + (self.data[:,-1,:,1].astype("u2")<<8)
        trigger = trigger.reshape(-1)
        changed = np.flatnonzero((trigger!=0)&(trigger!=np.r_[np.uint16(0),trigger[:-1]]))
        return [(int(i),int(trigger[i])) for i in changed]

    def window(self,start):
        end = start+2048
        if start<0 or end>self.complete_records*self.block:
            raise ValueError("Epoch out of complete BDF records")
        lo,hi = start//self.block,(end-1)//self.block+1
        raw = self.data[lo:hi][:,self.picks].transpose(1,0,2,3).reshape(len(self.picks),-1,3)
        raw = raw[:,start%self.block:start%self.block+2048]
        uv = decode_i24(raw)*self.gain[:,None]+self.offset[:,None]
        return uv[:128]-uv[-2:].mean(0)


def inner_trials(events):
    condition = None
    selected = []
    for i,(cue,code) in enumerate(events):
        if code in (21,22,23):
            condition = code
        elif code==16:
            condition = None
        elif code in (31,32,33,34) and condition==22:
            later = events[i+1:i+3]
            if len(later)!=2 or [e[1] for e in later]!=[44,45]:
                raise ValueError(f"Ambiguous inner-speech action markers after sample {cue}: {later}")
            action,relax = later[0][0],later[1][0]
            if not .3*1024<=action-cue<=.8*1024 or relax-action<2304:
                raise ValueError("Action window intersects a cue or relax interval")
            selected.append({"cue":cue,"action":action+256,"precue":cue-2176,"class":code-31})
    if len(selected)==0 or set(r["class"] for r in selected)!=set(range(4)):
        raise ValueError("Incomplete four-way inner-speech event mapping")
    return selected


def spectral_features(x):
    f,p = welch(x,fs=1024.,window="hann",nperseg=512,noverlap=256,detrend="constant",axis=1)
    return np.concatenate([np.log(np.maximum(p[:,(f>=a)&(f<b)].mean(1),1e-24)) for a,b in BANDS])


def fit(x,y):
    avg = x.mean(0)
    scale = x.std(0)
    scale[scale==0] = 1
    z = (x-avg)/scale
    labels = np.eye(4)[y]
    intercept = labels.mean(0)
    w = z.T @ np.linalg.solve(z @ z.T + len(y)*np.eye(len(y)),labels-intercept)
    return (avg,scale,w,intercept)


def predict(x,model):
    mean,std,w,b = model
    return int(np.argmax((x-mean)/std @ w + b))


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a Gilbreth allocation")
    args.output.mkdir(parents=True,exist_ok=False)
    receipt = json.loads(args.acquisition.read_text())
    if receipt["status"]!="complete" or receipt["tree"]!="0c394b1a8b7aaf9d5e3a94ff17b7c10eb41df07f":
        raise ValueError("Need the verified raw acquisition")
    write_json(args.output/"provenance.json",{"source":receipt,"script_sha256":sha256(Path(__file__)),
                                             "preregistration_sha256":sha256(args.registration)})
    write_json(args.output/"config.resolved.yaml",{"train_sessions":[1,2],"test_session":3,"subject":"sub-01",
        "classes":WORDS,"action_offset_from_44_samples":256,"control_offset_from_cue_samples":-2176,
        "rate_hz":1024,"window_samples":2048,"nperseg":512,"noverlap":256,"bands_hz":BANDS,
        "regularization":"alpha = number of training trials","features":512,"ref":"mean EXG1 and EXG2"})
    status = {"status":"started","sessions":{}}
    models = {}
    try:
        with ResourceMonitor(args.output/"resources.json",max_seconds=900) as monitor:
            paths = [args.root / f"sub-01_ses-{n:02d}_task-innerspeech_eeg.bdf" for n in (1,2,3)]
            for path,manifest in zip(paths,receipt["files"],strict=True):
                if path.name!=manifest["file"] or sha256(path)!=manifest["sha256"]:
                    raise ValueError("Input no longer matches acquisition")
            train = {m:[] for m in ("action","precue")}
            labels = []
            load_seconds, fit_seconds = 0.,0.
            for n,path in enumerate(paths,1):
                begin = monotonic()
                rec = Recording(path)
                trials = inner_trials(rec.events())
                if len(trials) != (80 if n in (1,2) else 40):
                    raise ValueError(f"Unexpected session {n} inner-speech coverage: {len(trials)}")
                status["sessions"][str(n)] = {"trials":len(trials),"complete_records":rec.complete_records,
                    "header_records":rec.header_records,"incomplete_trailing_bytes":rec.trailing_bytes}
                load_seconds += monotonic()-begin
                if n<=2:
                    for row in trials:
                        monitor.check()
                        for m in train:
                            train[m].append(spectral_features(rec.window(row[m])))
                        labels.append(row["class"])
                    if n==2:
                        begin = monotonic()
                        for m in train:
                            models[m] = fit(np.asarray(train[m]),np.asarray(labels))
                            np.savez(args.output/f"{m}.npz",mean=models[m][0],scale=models[m][1],
                                     weights=models[m][2],intercept=models[m][3])
                        fit_seconds = monotonic()-begin
                    del rec
            # Only held-out session 03 reaches the following evaluation.
            cache = {}
            by_mode = {m:[] for m in models}
            with (args.output/"predictions.jsonl").open("w",encoding="utf8",buffering=1) as pred, (
                  args.output/"timing.jsonl").open("w",encoding="utf8",buffering=1) as timing:
                for rep in range(4):
                    for i,row in enumerate(trials):
                        trial = f"sub-01/ses-03/cue-{row['cue']}"
                        for offset in range(2):
                            m = ("action","precue")[(i+rep+offset)%2]
                            monitor.check()
                            begin = monotonic()
                            estimate = predict(spectral_features(rec.window(row[m])),models[m])
                            dt = monotonic()-begin
                            if rep==0:
                                cache[trial,m] = estimate
                                pred.write(json.dumps({"trial_id":trial,"window":m,"prediction":WORDS[estimate]})+"\n")
                            else:
                                if cache[trial,m]!=estimate:
                                    raise ValueError("Deterministic replay mismatch")
                                by_mode[m].append(dt)
                                timing.write(json.dumps({"trial_id":trial,"window":m,"pass":rep,"seconds":dt})+"\n")
            status["conditions"] = {}
            for m in models:
                confusion = np.zeros((4,4),dtype=int)
                for row in trials:
                    trial = f"sub-01/ses-03/cue-{row['cue']}"
                    confusion[row["class"],cache[trial,m]] += 1
                correct,total = int(confusion.trace()),int(confusion.sum())
                interval = binomtest(correct,total).proportion_ci(.95,method="wilson")
                status["conditions"][m] = {"correct":correct,"total":total,"accuracy":correct/total,
                    "confusion_true_by_predicted":confusion.tolist(),"labels":WORDS,"uniform_chance":.25,
                    "wilson95":[interval.low,interval.high],"warm_requests":len(by_mode[m]),
                    "timing_s":{k:float(v) for k,v in zip(("p50","p90","p95","p99","max"),
                    np.quantile(by_mode[m],[.5,.9,.95,.99,1]),strict=True)},
                    "checkpoint_sha256":sha256(args.output/f"{m}.npz")}
            status.update(raw_header_and_event_index_s=load_seconds,fit_both_seconds=fit_seconds)
            monitor.check()
        if json.loads((args.output/"resources.json").read_text())["outcome"]!="complete":
            raise RuntimeError("Final sample exceeded resource budget")
        status["status"]="complete"
    except BaseException as exc:
        status.update(status="incomplete",error=repr(exc))
        raise
    finally:
        write_json(args.output/"metrics.json",status)
        (args.output/"report.md").write_text(f"# Four scalp EEG commands\n\nStatus {status['status']}. "
            "Predict one of four cued directions for participant 01, session 03. "
            f"Detailed counts and request times are in metrics.json.\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","registration","acquisition","output"):
        p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
