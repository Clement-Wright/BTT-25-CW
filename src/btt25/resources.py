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
from .host_accounting import HostAccounting, WATCHDOG_SECONDS


class BudgetExceeded(RuntimeError):
    pass


def hardware():
    vm = psutil.virtual_memory()
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpu.append({"name": p.name, "total_vram_bytes": p.total_memory, "capability": list(torch.cuda.get_device_capability(i)), "uuid": str(p.uuid) if getattr(p, "uuid", None) else None})
    return {"os": platform.platform(), "cpu_threads": os.cpu_count(), "visible_host_ram_bytes": vm.total, "available_host_ram_bytes": vm.available, "gpu": gpu, "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


def _smi(query, fields):
    import csv
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-{query}={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.5, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))


def gpu_snapshot(owned_pids, allocated):
    start = time.monotonic()
    ids = {device["uuid"] for device in allocated if device["uuid"]}
    if ids:
        ids |= {f"GPU-{item}" for item in ids if not item.startswith("GPU-")}
    fields = ["uuid", "name", "memory.used", "memory.total", "utilization.gpu",
              "temperature.gpu", "power.draw", "clocks.current.sm", "clocks.current.memory"]
    names = ["used_mib", "total_mib", "utilization_percent", "temperature_c",
             "power_w", "sm_clock_mhz", "memory_clock_mhz"]

    def numeric(field):
        try:
            return float(field)
        except ValueError:
            return None

    devices = _smi("gpu", fields)
    processes = _smi("compute-apps", ["gpu_uuid", "pid", "used_gpu_memory"])
    return {"gpu_devices": [{"uuid": row[0], "model": row[1],
                              **dict(zip(names, map(numeric, row[2:]), strict=True))}
                             for row in devices or [] if len(row) == len(fields) and row[0] in ids],
            "gpu_owned_processes": [{"uuid": row[0], "pid": int(row[1]), "used_mib": numeric(row[2])}
                                     for row in processes or [] if len(row) == 3 and row[1].isdigit()
                                     and int(row[1]) in owned_pids and row[0] in ids] if processes is not None and ids else None,
            "gpu_sampler_seconds": time.monotonic() - start}


def summarize_gpu(samples, allocated):
    summary = []
    for device in allocated:
        uuid = device["uuid"]
        matches = {uuid, f"GPU-{uuid}"} if uuid else set()
        values = [d for row in samples for d in row.get("gpu_devices", []) if d["uuid"] in matches]
        entry = {**device, "device_wide_samples": len(values)}
        for key in ("used_mib", "utilization_percent", "temperature_c", "power_w", "sm_clock_mhz", "memory_clock_mhz"):
            readings = [d[key] for d in values if d[key] is not None]
            entry[key] = {"mean": sum(readings) / len(readings), "peak": max(readings)} if readings else None
        owned = []
        for row in samples:
            processes = row.get("gpu_owned_processes")
            if processes is None:
                continue
            measured = [proc["used_mib"] for proc in processes if proc["uuid"] in matches]
            if measured and None not in measured:
                owned.append(sum(measured))
        entry["peak_sampled_owned_processes_mib"] = max(owned, default=None)
        summary.append(entry)
    return summary


class ResourceMonitor:
    def __init__(self, output, max_seconds=28200, soft_ram=12_000_000_000,
                 hard_ram=16_000_000_000, minimum_available=2_000_000_000):
        self.output = Path(output)
        self.host = HostAccounting(hard_ram, soft_ram)
        self.max_seconds = min(max_seconds, WATCHDOG_SECONDS)
        self.soft_ram, self.hard_ram = self.host.soft, self.host.hard
        self.minimum_available = minimum_available
        self.start = time.monotonic()
        self.samples = []
        self.lock = threading.Lock()
        self.reason = None
        self.stop_event = threading.Event()
        self.system = hardware()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

    def sample(self):
        with self.lock:
            sample = self.host.sample()
            if torch.cuda.is_available():
                sample["pytorch_devices"] = [
                    {"logical_index": i, "allocated_bytes": torch.cuda.memory_allocated(i),
                     "reserved_bytes": torch.cuda.memory_reserved(i),
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(i)}
                    for i in range(torch.cuda.device_count())
                ]
                sample.update(gpu_snapshot(sample["owned_pids"], self.system["gpu"]))
                for device in sample["pytorch_devices"]:
                    free, total = torch.cuda.mem_get_info(device["logical_index"])
                    device.update(visible_device_used_bytes=total - free,
                                  visible_device_free_bytes=free)
            self.samples.append(sample)
            self.stream.write(json.dumps(sample) + "\n")
            self.stream.flush()
            if self.reason is None:
                self.reason = ("whole-job deadline" if sample["elapsed_seconds"] > self.max_seconds
                               else self.host.stop_reason(sample, self.minimum_available))
            return sample

    def _loop(self):
        while not self.stop_event.wait(1):
            self.sample()

    def __enter__(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.output.with_suffix(".samples.jsonl").open("w", encoding="utf-8")
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def check(self):
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
        def peak(key):
            return max((s[key] for s in self.samples if s.get(key) is not None), default=None)

        final = self.samples[-1]
        initial_events, final_events = self.samples[0]["cgroup_events"], final["cgroup_events"]
        cpu = {k: v for k, v in final.items() if k.startswith("cpu_")}
        if self.host.allocated_cpus and cpu["cpu_busy_cores_average"] is not None:
            cpu["cpu_average_utilization_of_allocation"] = cpu["cpu_busy_cores_average"] / self.host.allocated_cpus
        device_used = [device["visible_device_used_bytes"] for row in self.samples
                       for device in row.get("pytorch_devices", [])]
        summary = {**self.host.describe(), "hardware": self.system,
                   "elapsed_seconds": time.monotonic() - self.start,
                   "sample_count": len(self.samples), "sampling_seconds": 1,
                   "peak_tree_rss_bytes": peak("tree_rss_bytes"),
                   "peak_tree_pss_bytes": peak("tree_pss_bytes"),
                   "peak_tree_uss_bytes": peak("tree_uss_bytes"),
                   "peak_tree_private_bytes": peak("tree_private_bytes"),
                   "peak_accounted_host_bytes": peak("accounted_host_bytes"),
                   "peak_cgroup_bytes": peak("cgroup_peak_bytes"),
                   "cgroup_events": final_events,
                   "cgroup_events_delta": {key: count - initial_events.get(key, 0) for key, count in final_events.items()}
                   if initial_events is not None and final_events is not None else None,
                   "minimum_system_available_bytes": min(s["system_available_bytes"] for s in self.samples),
                   "peak_device_used_bytes": max(device_used, default=None),
                   "peak_cuda_allocated_bytes": max((d["peak_allocated_bytes"] for d in final.get("pytorch_devices", [])), default=None),
                   "peak_cuda_reserved_bytes": max((d["peak_reserved_bytes"] for d in final.get("pytorch_devices", [])), default=None),
                   "cpu": cpu, "gpu_summary": summarize_gpu(self.samples, self.system["gpu"]),
                   "sampler_seconds": sum(s["sampler_seconds"] + s.get("gpu_sampler_seconds", 0) for s in self.samples),
                   "stop_reason": self.reason, "outcome": outcome}
        self.stream.close()
        write_json(self.output, summary)
