"""Summarize the locked two-rate deployment comparison without releasing individual trials."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render(data, path):
    methods=("native","merge_agree")
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.7),gridspec_kw={"width_ratios":[1,1]})
    fig.subplots_adjust(left=.105,bottom=.22,right=.975,top=.81,wspace=.58)
    x=np.arange(2)
    for label,values,color in (
            ("Compact beam 50",[data["compact_controls"][m]["wer"] for m in methods],"#93623f"),
            ("Expanded alignment + DistilGPT2",[data["counts"][m]["word_errors"]/data["counts"][m]["words"] for m in methods],"#12617d")):
        y=100*np.array(values)
        ax.plot(x,y,color=color,marker="o",ms=7,lw=2,label=label)
        for i,val in enumerate(y):
            ax.annotate(f"{val:.2f}%",(i,val),xytext=(2,8),textcoords="offset points",fontsize=9,
                        color=color,bbox={"facecolor":"white","pad":.2,"edgecolor":"none"})
    ax.set(xticks=x,xticklabels=["Every 40 ms frame","Merge agreeing pairs"],ylabel="Word error rate (%)",
           xlim=(-.16,1.36),ylim=(0,max(data["compact_controls"]["native"]["wer"]*100+4,32)))
    ax.set_title("Accuracy on the same 926 trials",loc="left",fontsize=11,fontweight="bold")
    ax.legend(frameon=False,fontsize=8,loc="lower left")
    for i,m in enumerate(methods):
        v=data["timing"][m]["seconds"]
        mid,high=1000*v["p50"],1000*v["p95"]
        bx.hlines(i,mid,high,color="#29708a",lw=2)
        bx.scatter(mid,i,color="#165c75",s=46)
        bx.scatter(high,i,facecolors="white",edgecolors="#165c75",lw=1.6,s=46)
        for z in (mid,high):
            bx.annotate(f"{z:.0f}",(z,i),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9)
    bx.set(yticks=x,yticklabels=["Every 40 ms frame","Merge agreeing pairs"],
           xlabel="Released features to final text (ms)",ylim=(1.55,-.55),
           xlim=(0,max(data["timing"][m]["seconds"]["p95"] for m in methods)*1150))
    bx.set_title("Median and p95 latency",loc="left",fontsize=11,fontweight="bold")
    for a in (ax,bx):
        a.spines[["top","right"]].set_visible(False)
        a.spines[["left","bottom"]].set_color("#9aa7ae")
        a.tick_params(length=0,pad=6,labelsize=9)
        a.grid(axis="y" if a is ax else "x",lw=.7,color="#e0e8eb")
        a.set_axisbelow(True)
    fig.savefig(path,dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main(a):
    data=load(a.study/"metrics.json")
    resource=load(a.study/"resources.json")
    if data["status"]!="complete" or resource["outcome"]!="complete":
        raise ValueError("Cannot publish incomplete comparison")
    rows=[json.loads(line) for line in (a.study/"predictions.jsonl").read_text().splitlines()]
    timings=[json.loads(line) for line in (a.study/"timing.jsonl").read_text().splitlines()]
    if len(rows)!=926 or len(set(r["trial_id"] for r in rows))!=926:
        raise ValueError("Incomplete tuning prediction coverage")
    cadence=load(a.compact)["conditions"]
    data["compact_controls"]={}
    cohorts=[]
    for m in ("native","merge_agree"):
        if data["timing"][m]["requests"]!=72:
            raise ValueError("Missing interleaved benchmark requests")
        warm=[v for v in timings if v["method"]==m and not v["warmup"]]
        keys={(v["pass"],v["trial_id"]) for v in warm}
        if (len(warm)!=72 or len(keys)!=72 or
                any(len({i for s,i in keys if s==k})!=24 for k in range(3)) or
                len({i for _,i in keys})!=24):
            raise ValueError("Timing cohort or three-pass coverage changed")
        cohorts.append(keys)
        tm=data["timing"][m]
        if tm["changed"]!=sum(not v["match"] for v in warm) or tm["empty"]!=sum(not v["text"] for v in warm):
            raise ValueError("Reported timing replay statuses differ")
        np.testing.assert_allclose(
            [tm["seconds"][k] for k in ("p50","p90","p95","p99","max")],
            np.percentile([v["elapsed_s"] for v in warm],[50,90,95,99,100]))
        c=data["counts"][m]
        for key in ("word_errors","words","phones","phone_errors","frames","oracle","candidates"):
            if sum(r["results"][m]["counts"][key] for r in rows)!=c[key]:
                raise ValueError(f"Per-trial {m}/{key} disagreement")
        prior=cadence["two_rate/"+m]
        data["compact_controls"][m]={k:prior[k] for k in ("counts","wer","per")}
        if c["phone_errors"]!=prior["counts"]["phone_errors"] or c["words"]!=prior["counts"]["word_reference_length"]:
            raise ValueError("The compact comparison did not use the same phone output or words")
    if cohorts[0]!=cohorts[1]:
        raise ValueError("Timing trajectories used different trials")
    data["admission"]=load(a.study/"admission.json")
    data["provenance"]=load(a.study/"provenance.json")
    data["resources"]={key:resource[key] for key in ("elapsed_seconds","method","allocated_cpus","peak_accounted_host_bytes",
          "host_ceiling_bytes","host_limit_source","slurm_requested_bytes","cgroup_effective_limit_bytes","cgroup_events_delta",
          "peak_cgroup_bytes","peak_tree_rss_bytes","peak_tree_pss_bytes","peak_tree_uss_bytes","peak_cuda_allocated_bytes",
          "peak_cuda_reserved_bytes","peak_device_used_bytes","cpu","sampler_seconds")}
    data["resources"]["gpu"]=resource["hardware"]["gpu"][0]["name"]
    data["resources"]["gpu_counters"]={k:v for k,v in resource["gpu_summary"][0].items() if k!="uuid"}
    with (a.study/"scheduler-end.psv").open(newline="") as receipt:
        job=next(csv.DictReader(receipt,delimiter="|"))
    if job["State"]!="COMPLETED" or job["ExitCode"]!="0:0":
        raise ValueError("Missing completed scheduler receipt")
    data["scheduler"]={k:job[k] for k in ("JobID","State","Elapsed","TotalCPU","ExitCode")}
    a.output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    render(data,a.figure)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--compact",type=Path,default=Path("reports/rate_decode_v4.json"))
    p.add_argument("--output",type=Path,default=Path("reports/two_rate_expanded.json"))
    p.add_argument("--figure",type=Path,default=Path("reports/figures/two_rate_expanded.png"))
    main(p.parse_args())
