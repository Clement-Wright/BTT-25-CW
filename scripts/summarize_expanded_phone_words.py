"""Export aggregate accuracy and matched HDF5-to-word timings for the expanded word decoder."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def block_intervals(rows, metadata):
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        rec = metadata[row["trial_id"]]
        groups[(rec["session_id"], rec["block_id"])].append(i)
    grouped = list(groups.values())
    denominator = np.array([sum(rows[i]["results"]["original_path"]["counts"]["reference_length"]
                                for i in indices) for indices in grouped])
    picks = np.random.default_rng(20261127).integers(0,len(grouped),size=(2000,len(grouped)))
    output={}
    for first,second in [("expanded_alignment","original_alignment"),("expanded_alignment","original_path"),
                         ("expanded_path","original_path")]:
        diff=np.array([sum(rows[i]["results"][first]["counts"]["errors"]-
                           rows[i]["results"][second]["counts"]["errors"] for i in indices) for indices in grouped])
        bootstrap=diff[picks].sum(1)/denominator[picks].sum(1)
        output[first+"_vs_"+second]={"blocks":len(grouped),"delta_errors":int(diff.sum()),
                                    "ci_95":np.quantile(bootstrap,[.025,.975]).tolist()}
    return output


def compact_resource(path):
    r=read(path)
    if r["outcome"]!="complete":
        raise ValueError("Incomplete resource trace")
    keys=("elapsed_seconds","allocated_cpus","host_ceiling_bytes","peak_accounted_host_bytes","peak_cgroup_bytes",
          "peak_tree_rss_bytes","peak_tree_pss_bytes","peak_cuda_allocated_bytes","peak_cuda_reserved_bytes",
          "peak_device_used_bytes","method","cpu")
    return {k:r[k] for k in keys}|{"gpu":r["hardware"]["gpu"][0]["name"]}


def chart(data, dest):
    fig,(a,b)=plt.subplots(1,2,figsize=(10.6,4.2),gridspec_kw={"width_ratios":[1.08,1]})
    fig.subplots_adjust(left=.10,right=.97,bottom=.21,top=.78,wspace=.58)
    metrics=data["evaluation"]["overall"]
    for j,(part,label,color) in enumerate([("path","Path score","#9d592b"),("alignment","Alignment sum","#14617a")]):
        nums=[100*metrics[f"{k}_{part}"]["errors"]/metrics[f"{k}_{part}"]["reference_length"]
              for k in ("original","expanded")]
        a.plot([0,1],nums,marker="o",color=color,lw=2.2,ms=7,label=label)
        for x,y in enumerate(nums):
            a.annotate(f"{y:.2f}%",(x,y),xytext=(3,11 if j==0 else -15),textcoords="offset points",fontsize=9,color=color,
                       bbox={"facecolor":"white","edgecolor":"none","pad":.4})
    a.set(xlim=(-.15,1.22),ylim=(12.35,14.65),xticks=[0,1],xticklabels=["Native list","With phone edits"],
          ylabel="Word error rate (%)")
    a.set_title("Adding sentences to the word search",loc="left",fontsize=12,fontweight="bold")
    a.legend(frameon=False,loc="upper left",fontsize=8)
    summary=data["deployment"]["timing"]
    for j,(name,label,color) in enumerate([("original_alignment","Native list","#809398"),
                                            ("expanded_alignment","With phone edits","#14617a")]):
        stats=summary[name]["p"]
        p50,p95=1000*stats["p50"],1000*stats["p95"]
        b.plot([p50,p95],[j,j],color=color,lw=2.4)
        b.scatter(p50,j,c=color,s=55)
        b.scatter(p95,j,facecolors="white",edgecolors=color,linewidths=1.7,s=55)
        b.annotate(f"{p50:.0f}",(p50,j),xytext=(0,8),ha="center",textcoords="offset points",fontsize=9)
        b.annotate(f"{p95:.0f}",(p95,j),xytext=(0,8),ha="center",textcoords="offset points",fontsize=9)
    b.set(yticks=[0,1],yticklabels=["Native list","With phone edits"],ylim=(1.55,-.55),
          xlim=(0,max(v["p"]["p95"] for v in summary.values())*1150),xlabel="Feature read to final text (ms)")
    b.set_title("Median and p95 latency",loc="left",fontsize=12,fontweight="bold")
    for axis in (a,b):
        axis.spines[["top","right"]].set_visible(False)
        axis.spines[["left","bottom"]].set_color("#95a3a7")
        axis.tick_params(length=0,pad=6,labelsize=9)
        axis.grid(axis="y" if axis is a else "x",color="#e3e9ea",lw=.7)
        axis.set_axisbelow(True)
    fig.savefig(dest,dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("study","timing","manifest","test"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("reports/expanded_phone_words.json"))
    p.add_argument("--figure",type=Path,default=Path("reports/figures/expanded_phone_words.png"))
    a=p.parse_args()
    metrics=read(a.study/"metrics.json")
    timing=read(a.timing/"metrics.json")
    rows=[json.loads(line) for line in (a.study/"predictions.jsonl").read_text().splitlines()]
    if metrics["status"]!="complete" or timing["status"]!="complete" or len({r["trial_id"] for r in rows})!=926:
        raise ValueError("Incomplete evaluation")
    timing.pop("decoder_startup")
    export=read(a.test/"metrics.json")
    if (export["status"]!="complete" or export["references_loaded"] or export["empty"]
            or export["saved_predictions"]!=export["expected_trials"]):
        raise ValueError("Incomplete blind export")
    data={"evaluation":metrics,"deployment":timing,"score_run_resources":compact_resource(a.study/"resources.json"),
          "deployment_resources":compact_resource(a.timing/"resources.json"),"run_provenance":read(a.study/"provenance.json"),
          "hidden_test_export":export,"export_resources":compact_resource(a.test/"resources.json"),
          "timing_provenance":read(a.timing/"provenance.json"),"paired_block_intervals":
          block_intervals(rows,{r["trial_id"]:r for r in read(a.manifest)})}
    a.output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    chart(data,a.figure)
