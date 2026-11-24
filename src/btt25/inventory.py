"""Bounded project inventory; output contains private paths and stays local."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .assets import sha256
from .data import write_json
from .resources import hardware


def git_read(root, *args):
    command = ["git", "-c", f"safe.directory={Path(root).resolve().as_posix()}", "-C", str(root), *args]
    result = subprocess.run(command,capture_output=True,text=True,timeout=30)
    return {"returncode":result.returncode,"output":result.stdout.strip(),"error":result.stderr.strip()}


def inventory(roots, output_dir):
    roots = [Path(p).resolve() for p in roots]
    result = {"hardware":hardware(),"python":sys.executable,"versions":{name:importlib.metadata.version(name) for name in ("torch","numpy","scipy","h5py","psutil","PyYAML")},"roots":[],"repositories":[],"assets":[],"limitations":["Inventory is scoped to supplied project directories; no unrelated projects modified.","HDF5 files are counted/sized, not loaded or authenticated by these metadata alone.","Private paths and machine inventory must not be published."]}
    seen_paths = set()
    excluded = {".git",".venv",".decoder-venv","node_modules","__pycache__",".pytest_cache","uv-cache","python"}
    suffixes = {".hdf5",".h5",".pt",".pth",".ckpt",".pkl",".arpa",".binary",".fst",".safetensors"}
    for root in roots:
        result["roots"].append({"path":str(root),"exists":root.exists(),"free_disk_bytes":shutil.disk_usage(root if root.exists() else root.parent).free})
        if not root.exists():
            continue
        for parent, dirs, files in os.walk(root):
            p = Path(parent)
            if ".git" in dirs or ".git" in files:
                result["repositories"].append({"path":str(p),"head":git_read(p,"rev-parse","HEAD"),"status":git_read(p,"status","--short","--branch"),"branches":git_read(p,"branch","-avv"),"worktrees":git_read(p,"worktree","list","--porcelain"),"remotes":git_read(p,"remote","-v")})
            dirs[:]=[d for d in dirs if d not in excluded]
            for name in files:
                file = p/name
                if (file.suffix.lower() not in suffixes and file.name!="best_checkpoint") or str(file) in seen_paths:
                    continue
                seen_paths.add(str(file))
                stat=file.stat()
                row={"path":str(file),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"origin_root":str(root)}
                if file.suffix in {".pt",".pth",".ckpt",".safetensors"} or file.name=="best_checkpoint":
                    row["sha256"]=sha256(file)
                result["assets"].append(row)
    write_json(Path(output_dir)/"inventory.json",result)
    return result
