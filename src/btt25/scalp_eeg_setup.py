"""Acquire the three pinned raw BioSemi files used by the sub-01 experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import monotonic
from urllib.request import Request, urlopen


TREE = "0c394b1a8b7aaf9d5e3a94ff17b7c10eb41df07f"
FILES = [(771058176,"99304b3e9f43792d625ad36cf4153f3e"),
         (702878208,"e0ba741cc1f219f8a11c5ef183e6efc2"),
         (408271605,"4de9738d5fcc3fa13d2f522715bc9221")]


def acquire(root, out):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this acquisition in a Gilbreth setup allocation")
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    results = []
    try:
        for i,(size, expected) in enumerate(FILES,1):
            key = f"sub-01/ses-{i:02d}/eeg/sub-01_ses-{i:02d}_task-innerspeech_eeg.bdf"
            url = "https://s3.amazonaws.com/openneuro.org/ds003626/"+key
            file = root/Path(key).name
            tmp = file.with_suffix(".bdf.part")
            download = not file.exists()
            begin = monotonic()
            if download:
                offset = tmp.stat().st_size if tmp.exists() else 0
                if offset > size:
                    raise ValueError("Oversized partial BDF")
                if offset < size:
                    req = Request(url,headers={"Range":f"bytes={offset}-"} if offset else {})
                    with urlopen(req,timeout=75) as response, tmp.open("ab" if offset else "wb") as stream:
                        if offset and (response.status!=206 or response.headers["Content-Range"].split("/")[0] != f"bytes {offset}-{size-1}"):
                            raise ValueError("Server did not resume the expected byte range")
                        while chunk := response.read(2**22):
                            if monotonic()-start > 1200 or stream.tell()+len(chunk)>size:
                                raise TimeoutError("Setup wall/volume limit reached")
                            stream.write(chunk)
                            if stream.tell()//(128*2**20) != (stream.tell()-len(chunk))//(128*2**20):
                                print(file.name,stream.tell(),flush=True)
            source = tmp if download else file
            sha,md5 = hashlib.sha256(),hashlib.md5()
            with source.open("rb") as stream:
                while chunk := stream.read(2**22):
                    sha.update(chunk)
                    md5.update(chunk)
            if source.stat().st_size!=size or md5.hexdigest()!=expected:
                raise ValueError(f"Wrong OpenNeuro Git-annex size/content on {source.name}")
            if download:
                tmp.replace(file)
            results.append({"file":file.name,"size":size,"upstream_md5":expected,"sha256":sha.hexdigest(),
                            "downloaded":download,"seconds":monotonic()-begin,"url":url})
        status = "complete"
    except BaseException as exc:
        status = repr(exc)
        raise
    finally:
        (out/"acquisition.json").write_text(json.dumps({"status":status,"tree":TREE,"license":"CC0",
            "dataset":"OpenNeuro ds003626 version 2.0.0","files":results,"elapsed_seconds":monotonic()-start},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    acquire(args.root,args.output)
