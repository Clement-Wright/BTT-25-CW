"""Record and advise reclamation of a Slurm job's private training-file cache."""
import argparse
import json
import os
from pathlib import Path
from time import monotonic

from .host_accounting import HostAccounting


def measure(cgroup):
    stats = dict((key, int(value)) for key, value in
                 (line.split() for line in (cgroup / "memory.stat").read_text().splitlines()))
    return {key: stats[key] for key in ("anon", "file", "inactive_file", "active_file")} | {
        "current": int((cgroup / "memory.current").read_text())}


def run(manifest, scratch, output):
    root = scratch.resolve()
    records = json.loads(manifest.read_text(encoding="utf-8"))
    files = sorted({Path(row["source_file"]).resolve() for row in records if row["split"] == "train"})
    if not files or any(not file.is_relative_to(root) or file.suffix not in {".h5", ".hdf5"} for file in files):
        raise ValueError("Expected only private scratch training HDF5 files")
    group = HostAccounting().cgroup
    if group is None:
        raise ValueError("Run inside an allocation with verified whole-job cgroup accounting")
    with output.open("x", encoding="utf-8") as stream:
        before = measure(group)
        started = monotonic()
        for file in files:
            with file.open("rb") as source:
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        receipt = {"method": "POSIX_FADV_DONTNEED scoped to private training HDF5 files", "files": len(files),
                   "elapsed_seconds": monotonic() - started, "before": before, "after": measure(group)}
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, Path(os.environ["RCAC_SCRATCH"]), args.output)))


if __name__ == "__main__":
    main()
