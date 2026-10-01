"""Kill leftover Pion servers by EXECUTABLE name, never by command line (gh #347).

`pkill -f 'pion-serve[r]'` matches any process whose command line contains
"pion-server" -- including a running `mojo build ... -o pion-server`, which
ends in exactly that. A vector bench started during a build SIGKILLed the
build (exit 137) twice on 2026-09-25.

Matching the process's `comm` instead cannot hit a build: the compiler's comm
is `mojo`. The match is a basename PREFIX, `pion-server*`, so it also covers
the variants `pkill -x pion-server` misses:

  pion-server-dev       dev build; `-x pion-server` leaves it holding the port
  pion-server-iouring   19 chars; Linux truncates comm to 15 (`pion-server-iou`)
"""
import os
import signal
import subprocess

SERVER_PREFIX = "pion-server"


def pion_server_pids():
    """PIDs of running processes whose executable basename starts with pion-server."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,comm="], capture_output=True,
                             text=True, check=False).stdout
    except OSError:
        return []
    pids = []
    me = os.getpid()
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, comm = int(parts[0]), parts[1].strip()
        # macOS `comm` is the full executable path; Linux is the bare name.
        if pid != me and os.path.basename(comm).startswith(SERVER_PREFIX):
            pids.append(pid)
    return pids


def kill_pion_servers():
    """SIGKILL every pion-server* process. Returns the PIDs signalled."""
    pids = pion_server_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return pids


def kill_inference_workers():
    """SIGKILL the PyTorch sidecar (a python process, so it has to be -f).

    The bracket keeps the pattern from matching the `sh -c` that runs it.
    """
    subprocess.run(["pkill", "-9", "-f", "inference/worke[r]"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
