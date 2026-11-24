"""Host resource policy shared by the training and native decoder environments."""
from pathlib import Path
import os
import re
import time

import psutil


TASK_LIMIT = 16_000_000_000
OPERATING_TARGET = 12_000_000_000
WATCHDOG_SECONDS = 28_200


def _read(path):
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return None


def _number(value):
    return int(value) if value and value.isdecimal() else None


def _counters(path):
    raw = _read(path)
    return dict((key, int(value)) for key, value in
                (line.split() for line in raw.splitlines())) if raw else None


class HostAccounting:
    def __init__(self, hard=TASK_LIMIT, soft=OPERATING_TARGET, *,
                 proc=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
        self.cgroup = None
        self.cgroup_limit = None
        self.slurm_limit = None
        self.allocated_cpus = _number(os.environ.get("SLURM_CPUS_PER_TASK"))
        job = os.environ.get("SLURM_JOB_ID")
        membership = _read(proc / "self/cgroup")
        if membership:
            for line in membership.splitlines():
                if line.startswith("0::/"):
                    parts = Path(line[4:]).parts
                    if ".." in parts:
                        break
                    for depth in range(len(parts), -1, -1):
                        path = cgroup_root.joinpath(*parts[:depth])
                        limit = _number(_read(path / "memory.max"))
                        if limit is not None:
                            self.cgroup_limit = min(self.cgroup_limit or limit, limit)
                        named_job = bool(job and depth and re.fullmatch(
                            rf"job[_-]{re.escape(job)}(?:\.scope)?", parts[depth - 1]
                        ))
                        # Gilbreth's Slurm uses a per-job opaque systemd path.
                        # A sibling extern step verifies this is the entire job.
                        systemd_job = (job and 2 <= depth < len(parts)
                                       and parts[depth - 2] == "slurmstepd.scope"
                                       and re.fullmatch(r"step_(?:batch|extern|[0-9]+)", parts[depth])
                                       and (path / "step_extern").is_dir())
                        if (named_job or systemd_job) and _number(_read(path / "memory.current")) is not None:
                            self.cgroup = path
                    break
        if job:
            mem = _number(os.environ.get("SLURM_MEM_PER_NODE"))
            if not mem:
                per_cpu = _number(os.environ.get("SLURM_MEM_PER_CPU"))
                cpus_on_node = _number(os.environ.get("SLURM_CPUS_ON_NODE"))
                mem = per_cpu * cpus_on_node if per_cpu and cpus_on_node else None
            self.slurm_limit = mem * 1024**2 if mem else None
        limits = {"task": min(hard, TASK_LIMIT), "Slurm allocation": self.slurm_limit,
                  "effective cgroup": self.cgroup_limit}
        self.limit_source, self.hard = min(
            ((name, limit) for name, limit in limits.items() if limit is not None),
            key=lambda pair: pair[1],
        )
        self.soft = min(soft, OPERATING_TARGET, int(self.hard * .9))
        self.process = psutil.Process()
        cpu = self.process.cpu_times()
        self._initial_process_cpu = (cpu.user, cpu.system)
        self._cpu = {}
        self._started = time.monotonic()
        self._initial_cpu = self._job_cpu()
        self._previous = (0., 0.)

    def _job_cpu(self):
        return _counters(self.cgroup / "cpu.stat") if self.cgroup else None

    def describe(self):
        return {"host_target_bytes": self.soft, "host_ceiling_bytes": self.hard,
                "host_limit_source": self.limit_source, "slurm_requested_bytes": self.slurm_limit,
                "cgroup_effective_limit_bytes": self.cgroup_limit,
                "cgroup_scope": str(self.cgroup) if self.cgroup else None,
                "method": "job cgroup memory charge including page cache" if self.cgroup else
                          "process-tree RSS fallback; shared pages can count more than once and file cache is outside the measurement",
                "allocated_cpus": self.allocated_cpus}

    def sample(self, full=True):
        started = time.monotonic()
        rss, pss, uss, private = [], [], [], []
        pids = []
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                with process.oneshot():
                    mem = process.memory_info()
                    times = process.cpu_times()
                    self._cpu[(process.pid, process.create_time())] = (times.user, times.system)
                    rss.append(mem.rss)
                    private.append(getattr(mem, "private", None))
                    pids.append(process.pid)
                    if full:
                        metrics = process.memory_full_info()
                        pss.append(getattr(metrics, "pss", None))
                        uss.append(getattr(metrics, "uss", None))
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        def sum_if_complete(values):
            return sum(values) if values and len(values) == len(rss) and None not in values else None

        current = peak = events = None
        if self.cgroup:
            current = _number(_read(self.cgroup / "memory.current"))
            peak = _number(_read(self.cgroup / "memory.peak"))
            events = _counters(self.cgroup / "memory.events")
        cpu = self._job_cpu()
        if cpu and self._initial_cpu and all(
            key in cpu and key in self._initial_cpu for key in ("user_usec", "system_usec")
        ):
            user, system = ((cpu[key] - self._initial_cpu[key]) / 1e6
                            for key in ("user_usec", "system_usec"))
            cpu_scope = "job cgroup"
        else:
            user = max(0., sum(values[0] for values in self._cpu.values()) - self._initial_process_cpu[0])
            system = max(0., sum(values[1] for values in self._cpu.values()) - self._initial_process_cpu[1])
            cpu_scope = "observed process tree; short-lived workers can escape sampling"
        elapsed = started - self._started
        old_t, old_cpu = self._previous
        busy = max(0., (user + system - old_cpu) / (elapsed - old_t)) if elapsed > old_t else None
        self._previous = (elapsed, user + system)
        return {"elapsed_seconds": elapsed, "tree_rss_bytes": sum(rss),
                "tree_pss_bytes": sum_if_complete(pss), "tree_uss_bytes": sum_if_complete(uss),
                "tree_private_bytes": sum_if_complete(private), "process_count": len(pids),
                "owned_pids": pids, "system_available_bytes": psutil.virtual_memory().available,
                "cgroup_current_bytes": current, "cgroup_peak_bytes": peak, "cgroup_events": events,
                "accounted_host_bytes": current if current is not None else sum(rss),
                "enforcement_scope": "job cgroup" if current is not None else "process tree RSS",
                "cpu_scope": cpu_scope, "cpu_user_core_seconds": user,
                "cpu_system_core_seconds": system, "cpu_busy_cores_sample": busy,
                "cpu_busy_cores_average": (user + system) / elapsed if elapsed > 0 else None,
                "cpu_utilization_of_allocation": busy / self.allocated_cpus
                if busy is not None and self.allocated_cpus else None,
                "sampler_seconds": time.monotonic() - started}

    def stop_reason(self, sample, minimum_available):
        used = max(sample["accounted_host_bytes"], sample.get("cgroup_peak_bytes") or 0)
        if used > self.hard:
            return f"{self.hard / 1e9:g} GB host RAM ceiling exceeded"
        if used > self.soft:
            return f"{self.soft / 1e9:g} GB host RAM target exceeded"
        if not os.environ.get("SLURM_JOB_ID") and sample["system_available_bytes"] < minimum_available:
            return f"OS available memory below {minimum_available / 1e9:g} GB"
        return None
