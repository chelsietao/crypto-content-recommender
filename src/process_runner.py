"""Stop launcher descendants on timeout or caller interruption."""
from __future__ import annotations

import subprocess
import os
import signal

import psutil


def stop_process_tree(process):
    try:
        parent = psutil.Process(process.pid)
        try:
            targets = [parent, *parent.children(recursive=True)]
        except (psutil.AccessDenied, PermissionError):
            targets = [parent]
    except psutil.NoSuchProcess:
        targets = []
    def signal_group(sig):
        if os.name == 'posix':
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

    signal_group(signal.SIGTERM)
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=3)
    signal_group(signal.SIGKILL)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)
    process.wait()


def run_training_process(command, cwd, env, log, timeout=300):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=log,
                               start_new_session=os.name == 'posix')
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        stop_process_tree(process)
        raise
