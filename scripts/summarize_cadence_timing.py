"""Extract three matched warm passes already timed in the decoder pilot."""
import argparse
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path
from statistics import quantiles


def run(path, output):
    # The preregistered pilot loops 8 conditions inside 24 duration-ordered trials.
    # Ignore its first complete pass to warm the native decoder and file cache.
    count = 8 * 24
    with path.open(encoding="utf8") as file:
        rows = [json.loads(line) for line in islice(file, 4*count)]
    if len(rows)!=4*count:
        raise ValueError("Need four complete, matched passes")
    keys = [(r["encoder"],r["method"],r["trial_id"]) for r in rows]
    if len(set(keys[:count]))!=count or any(keys[i*count:(i+1)*count]!=keys[:count] for i in range(1,4)):
        raise ValueError("Pilot passes do not match")
    stats = defaultdict(list)
    for r in rows[count:]:
        stats[r["encoder"]+"/"+r["method"]].append(r["cached_logit_to_text_s"])
    if len(stats)!=8 or {len(v) for v in stats.values()}!={72}:
        raise ValueError("Incomplete matched timing roster")
    result={}
    for name,values in stats.items():
        cuts=quantiles(values,n=100,method="inclusive")
        result[name]={"requests":72,"p50":cuts[49],"p90":cuts[89],"p95":cuts[94],"p99":cuts[98],"max":max(values)}
    output.write_text(json.dumps({"timing_scope":"warm cached NPY read through lexical selection, seconds",
        "passes":3,"trials_per_pass_per_condition":24,"conditions":result},indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    run(args.source,args.output)
