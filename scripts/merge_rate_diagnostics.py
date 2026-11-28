"""Join independent cadence audit receipts only when their shared control agrees."""
import argparse
import json
from pathlib import Path

import numpy as np


def merge(inputs, root):
    if not inputs:
        raise ValueError("Need at least one completed cadence audit")
    root.mkdir(parents=True,exist_ok=True)
    for source_name,public_name,conditions in (("cadence.json","cadence_v4.json","counts"),
            ("errors.json","cadence_phone_errors_v4.json","conditions")):
        result=json.loads((inputs[0]/source_name).read_text(encoding="utf8"))
        for directory in inputs[1:]:
            extra=json.loads((directory/source_name).read_text(encoding="utf8"))
            static=set(result)-{conditions,"frontends"}
            if any(extra.get(k)!=result[k] for k in static):
                raise ValueError(f"Mismatched analysis metadata for {source_name}")
            if result[conditions]["gru80"]!=extra[conditions]["gru80"]:
                raise ValueError("Matched control is different")
            for k,v in extra[conditions].items():
                if k in result[conditions] and result[conditions][k]!=v:
                    raise ValueError(f"Conflicting {k} receipts")
                result[conditions][k]=v
            if extra.get("frontends"):
                result.setdefault("frontends",{}).update(extra["frontends"])
        if source_name=="errors.json" and "gru40" in result[conditions]:
            base=result[conditions]["gru40"]["by_session"]
            sessions=sorted(base)
            draws=np.random.default_rng(20261127).integers(0,len(sessions),(2000,len(sessions)))
            for condition in result[conditions].values():
                if set(condition["by_session"])!=set(sessions):
                    raise ValueError("Session presence differs")
                paired={}
                for metric,error,total in (("wer","word_errors","words"),("per","phone_errors","phones")):
                    length=np.array([base[day][total] for day in sessions])
                    if [condition["by_session"][day][total] for day in sessions]!=length.tolist():
                        raise ValueError("Session denominators differ")
                    delta=np.array([condition["by_session"][day][error]-base[day][error] for day in sessions])
                    interval=np.quantile(delta[draws].sum(1)/length[draws].sum(1),[.025,.975])
                    paired[metric]={"delta":float(delta.sum()/length.sum()),"lower":float(interval[0]),
                                    "upper":float(interval[1]),"sessions":len(sessions),"replicates":len(draws)}
                condition["paired_session_bootstrap_vs_gru40"]=paired
        (root/public_name).write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("directories",nargs="+",type=Path)
    args=p.parse_args()
    merge(args.directories,args.output)
