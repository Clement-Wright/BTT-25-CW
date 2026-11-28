"""Check the lightweight ARPA reader against the official KenLM binary on study sentences."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from btt25.assets import sha256
from btt25.data import write_json
from btt25.phone_edit_candidates import Trigram


def run(args):
    lm = Trigram(args.arpa)
    observed = {}
    with args.source.open() as packets:
        for line in packets:
            packet = json.loads(line)
            if packet["input"] in ("neural", "reference_spikes"):
                for c in packet["candidates"]:
                    observed.setdefault(c["text"], []).append(c["kenlm_contribution"])
    sentences = list(observed)
    query = subprocess.run([str(args.query), "-v", "sentence", str(args.binary)],
                           input="\n".join(sentences)+"\n", text=True, capture_output=True, check=True)
    values = re.findall(r"Total:\s*([-+\d.eE]+)\s+OOV:", query.stdout)
    if len(values) != len(sentences):
        raise ValueError(f"KenLM query returned {len(values)} sentence scores for {len(sentences)} inputs")
    differences, stored = [], []
    for text, total in zip(sentences,values):
        value = 2 * np.log(10.) * float(total)
        differences.append(abs(2 * lm.sentence(text.split())-value))
        stored.extend(abs(val-value) for val in observed[text])
    def stats(values):
        a=np.array(values)
        return {"count":len(values),"max_nats":float(a.max()),"p95_nats":float(np.quantile(a,.95)),
                "over_one_cent_in_nats":int((a>.01).sum()),"over_one_nat":int((a>1.).sum())}
    write_json(args.output,{"distinct_sentences":len(sentences),"arpa_vs_query":stats(differences),
                            "native_component_vs_query":stats(stored),"hashes":{
                                 k:sha256(getattr(args,k)) for k in ("arpa","binary","query","source")}})
    if max(differences) > .01:
        raise ValueError("ARPA reader differs from reference executable by more than 0.01 weighted nats")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("arpa","binary","query","source","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    run(parser.parse_args())
