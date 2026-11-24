"""Measured process-tree memory and independent CUDA accounting."""
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

import psutil
import torch

from .data import write_json


class BudgetExceeded(RuntimeError):
    pass


def hardware():
    vm = psutil.virtual_memory()
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpu.append({"name": p.name, "total_vram_bytes": p.total_memory, "capability": list(torch.cuda.get_device_capability(i))})
    return {"os": platform.platform(), "cpu_threads": os.cpu_count(), "visible_host_ram_bytes": vm.total, "available_host_ram_bytes": vm.available, "gpu": gpu, "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


class ResourceMonitor:
    def __init__(self, output, max_seconds=28200, soft_ram=12_000_000_000,
                 hard_ram=16_000_000_000, minimum_available=2_000_000_000):
        self.output = Path(output)
        allocation_mib=os.environ.get("SLURM_MEM_PER_NODE") if os.environ.get("SLURM_JOB_ID") else None
        self.ram_source="laptop defaults"
        if allocation_mib and allocation_mib.isdecimal() and int(allocation_mib)>0:
            hard_ram=int(allocation_mib)*1024**2
            soft_ram=int(hard_ram*.90)
            self.ram_source="Slurm node allocation"
        self.max_seconds, self.soft_ram, self.hard_ram = max_seconds, soft_ram, hard_ram
        self.minimum_available = minimum_available
        self.start = time.monotonic()
        self.samples = []
        self.reason = None
        self.stop_event = threading.Event()
        self.system = hardware()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def sample(self):
        process = psutil.Process()
        processes = [process, *process.children(recursive=True)]
        rss, private, alive = 0, 0, 0
        for p in processes:
            try:
                mem = p.memory_info()
                rss += mem.rss
                private += getattr(mem, "private", 0)
                alive += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        elapsed = time.monotonic() - self.start
        sample = {"elapsed_seconds": elapsed, "tree_rss_bytes": rss, "tree_private_bytes": private, "process_count": alive, "system_available_bytes": vm.available, "swap_used_bytes": swap.used}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            sample.update(cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(), device_used_bytes=total-free, device_free_bytes=free)
        self.samples.append(sample)
        if elapsed > self.max_seconds:
            self.reason = "whole-job deadline"
        elif rss > self.hard_ram:
            self.reason = "process-tree RAM ceiling exceeded"
        elif rss > self.soft_ram:
            self.reason = "process-tree RAM target exceeded"
        elif vm.available < self.minimum_available:
            self.reason = "OS available memory below 2 GB"
        return sample

    def _loop(self):
        while not self.stop_event.wait(1):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def check(self):
        self.sample()
        stop_file=os.environ.get("BTT_STOP_FILE")
        if stop_file and Path(stop_file).exists():
            self.reason="whole-job supervisor requested a safe stop"
        if self.reason:
            raise BudgetExceeded(self.reason)

    def __exit__(self, typ, error, tb):
        self.stop_event.set()
        self.thread.join(timeout=3)
        self.sample()
        if isinstance(error, (BudgetExceeded, KeyboardInterrupt)) or (self.reason and error is None):
            outcome = "budget_stopped"
        else:
            outcome = "failed" if error else "complete"
        summary = {"hardware": self.system, "method": "psutil sum of process-tree RSS/Windows working sets; shared pages may be double counted; private bytes recorded separately", "elapsed_seconds": time.monotonic()-self.start, "sample_count": len(self.samples), "sampling_seconds": 1, "peak_tree_rss_bytes": max(s["tree_rss_bytes"] for s in self.samples), "peak_tree_private_bytes": max(s["tree_private_bytes"] for s in self.samples), "minimum_system_available_bytes": min(s["system_available_bytes"] for s in self.samples), "peak_device_used_bytes": max(s.get("device_used_bytes",0) for s in self.samples), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0, "host_ceiling_bytes": self.hard_ram, "host_target_bytes": self.soft_ram, "host_limit_source":self.ram_source, "stop_reason": self.reason, "outcome": outcome}
        write_json(self.output, summary)
        self.output.with_suffix(".samples.jsonl").write_text("".join(json.dumps(s)+"\n" for s in self.samples), encoding="utf-8")
