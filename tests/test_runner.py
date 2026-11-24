import sys
import pytest
from btt25.runner import supervise


def test_watchdog_terminates_only_owned_process(tmp_path):
    result=supervise([sys.executable,"-c","import time; time.sleep(30)"],tmp_path,target_seconds=0.1,hard_seconds=0.4)
    assert result["state"]=="watchdog_stopped"
    assert (tmp_path/"STOP_REQUESTED").is_file()


def test_failed_stage_is_not_complete(tmp_path):
    result=supervise([sys.executable,"-c","raise SystemExit(7)"],tmp_path,target_seconds=5,hard_seconds=10)
    assert result["state"]=="failed"
    assert result["returncode"]==7
