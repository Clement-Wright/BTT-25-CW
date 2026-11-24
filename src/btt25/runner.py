"""A whole-job watchdog around sequential, owned subprocess stages."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

import psutil

from .data import write_json


def supervise(command, run_dir, target_seconds=27000, hard_seconds=28200):
    if not 0<target_seconds<hard_seconds<=28200:
        raise ValueError("Whole-job target must precede <=7h50m absolute watchdog")
    run_dir=Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    stop=run_dir/"STOP_REQUESTED"
    outer=os.environ.get("BTT_STOP_FILE")
    if stop.exists():
        stop.unlink()
    environment={**os.environ,"BTT_STOP_FILE":str(stop.resolve())}
    start=time.monotonic()
    with (run_dir/"supervisor.log").open("a",encoding="utf-8") as output:
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=environment)
        state={"state":"running","pid":process.pid,"command":command,"started_at":datetime.now(timezone.utc).isoformat(),"target_seconds":target_seconds,"hard_seconds":hard_seconds}
        write_json(run_dir/"supervisor.json",state)
        while process.poll() is None:
            elapsed=time.monotonic()-start
            outer_requested=bool(outer and Path(outer).exists())
            if (elapsed>=target_seconds or outer_requested) and not stop.exists():
                stop.write_text("Whole-job target reached; save at a valid boundary.\n",encoding="utf-8")
                state["stop_cause"]="outer scheduler signal" if outer_requested else "whole-job target"
            if elapsed>=hard_seconds:
                owned=psutil.Process(process.pid)
                for child in owned.children(recursive=True):
                    try: child.kill()
                    except psutil.NoSuchProcess: pass
                try: owned.kill()
                except psutil.NoSuchProcess: pass
                process.wait(timeout=10)
                state["state"]="watchdog_stopped"
                break
            time.sleep(min(.25,max(.01,hard_seconds-elapsed)))
        if state["state"]=="running":
            state["state"]="complete" if process.returncode==0 and not stop.exists() else "budget_stopped" if stop.exists() else "failed"
        state.update(returncode=process.returncode,elapsed_seconds=time.monotonic()-start)
        write_json(run_dir/"supervisor.json",state)
    return state
