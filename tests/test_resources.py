import json
from btt25.resources import ResourceMonitor


def test_caught_cooperative_stop_is_not_recorded_as_complete(tmp_path,monkeypatch):
    import btt25.resources as resources
    monkeypatch.setattr(resources.torch.cuda,'is_available',lambda:False)
    with ResourceMonitor(tmp_path/'resources.json') as monitor:
        monitor.reason='requested safe stop'
    result=json.loads((tmp_path/'resources.json').read_text())
    assert result['outcome']=='budget_stopped'


def test_unrelated_failure_is_failed(tmp_path,monkeypatch):
    import btt25.resources as resources
    monkeypatch.setattr(resources.torch.cuda,'is_available',lambda:False)
    try:
        with ResourceMonitor(tmp_path/'resources.json'):
            raise RuntimeError('broken checkpoint')
    except RuntimeError: pass
    assert json.loads((tmp_path/'resources.json').read_text())['outcome']=='failed'


def test_gilbreth_systemd_scope_tracks_whole_job(tmp_path, monkeypatch):
    from btt25.host_accounting import HostAccounting

    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    cgroup = tmp_path / "cgroup"
    job = cgroup / "system.slice/slurmstepd.scope/opaque123"
    (job / "step_batch/user").mkdir(parents=True)
    (job / "step_extern").mkdir()
    (proc / "self/cgroup").write_text("0::/system.slice/slurmstepd.scope/opaque123/step_batch/user\n")
    (job / "memory.current").write_text("1234")
    (job / "memory.max").write_text("15728640000")
    (job / "memory.peak").write_text("5678")
    (job / "memory.events").write_text("oom 0\noom_kill 0\n")
    (job / "cpu.stat").write_text("user_usec 1000000\nsystem_usec 1000000\n")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "20480")
    monitor = HostAccounting(proc=proc, cgroup_root=cgroup)
    (job / "cpu.stat").write_text("user_usec 2500000\nsystem_usec 1500000\n")
    assert monitor.hard == 15_728_640_000
    assert monitor.soft == 12_000_000_000
    assert monitor.cgroup == job
    row = monitor.sample()
    assert row["accounted_host_bytes"] == 1234
    assert row["cgroup_peak_bytes"] == 5678
    assert row["cgroup_events"]["oom_kill"] == 0
    assert row["cpu_user_core_seconds"] == 1.5
    assert row["cpu_system_core_seconds"] == .5
