#!/usr/bin/env python3
"""gh #282 — the listen socket must open BEFORE the sidecar readiness wait.

The listen-before-init design (doc/architecture.md) exists so a client queues in
the kernel backlog instead of seeing "Connection refused" while the server
finishes starting. That property was inverted for the one configuration a
first-time reader starts with: `--kvcache`/`--inference` enable the sidecar, and
the up-to-30 s readiness poll ran ~230 lines before create_listen_socket().

So the failure was not "startup is slow". It was: run the README's first
command, run the README's second command, get `Connection refused`, conclude the
binary is broken.

WHY THE ASSERTION IS RELATIVE, NOT A DEADLINE. The first version of this test
gave the accept a 10 s budget and PASSED on the pre-fix binary: this Mac imports
torch in ~4 s, so the pre-fix wait fit inside the budget and the test proved
nothing. An absolute deadline measures how fast the machine is; the property is
that accept time is INDEPENDENT of sidecar readiness. So the test reads the
server's own "sidecar ready (… N.Ns)" line and requires the accept to land
clearly before it. Pre-fix those two are the same instant by construction —
accept cannot happen until the poll returns — so the ordering is what separates
them on any machine, fast or slow.

Usage:  python3 tests/test_gh282_listen_before_sidecar.py [--binary ./pion-server-dev]
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

# The poll itself is bounded at 30 s; allow it plus room to start.
OVERALL_BUDGET_S = 45.0
# The accept must beat sidecar-ready by more than scheduling noise. Pre-fix the
# gap is <= 0, so anything positive discriminates; 0.5 s keeps it from flapping.
MIN_LEAD_S = 0.5
# Only used when there is no sidecar at all (a release tarball ships no
# worker.py). Then there is no "ready" line to race and the absolute bound is
# the only thing left to check — it is generous on purpose.
NO_SIDECAR_DEADLINE_S = 10.0

READY_RE = re.compile(r"Inference sidecar ready\b")
NO_SIDECAR_RE = re.compile(r"no inference sidecar found|Auto-embed unavailable")
GAVE_UP_RE = re.compile(r"sidecar not ready after|failed to spawn inference sidecar")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", "./pion-server"))
    ap.add_argument("--port", type=int, default=17974)
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(f"SKIP: {args.binary} not built")
        return 0

    cmd = [args.binary, "-p", str(args.port), "-w", "1", "--inference"]
    print(f"$ {' '.join(cmd)}")

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)

    # Stamp each line as it arrives; the server prints elapsed seconds itself but
    # we want both clocks on the same origin as the accept.
    marks: dict[str, float] = {}
    lines: list[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            now = time.monotonic() - t0
            lines.append(line.rstrip())
            if "ready" not in marks and READY_RE.search(line):
                marks["ready"] = now
            elif "none" not in marks and NO_SIDECAR_RE.search(line):
                marks["none"] = now
            elif "gaveup" not in marks and GAVE_UP_RE.search(line):
                marks["gaveup"] = now

    threading.Thread(target=reader, daemon=True).start()

    failures: list[str] = []
    accepted: float | None = None
    try:
        while time.monotonic() - t0 < OVERALL_BUDGET_S:
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=0.5):
                    accepted = time.monotonic() - t0
                    break
            except OSError:
                time.sleep(0.02)

        if accepted is None:
            failures.append(f"never accepted a connection within {OVERALL_BUDGET_S:.0f}s")
        else:
            print(f"   accept at {accepted:.2f}s")
            # Let the sidecar resolve one way or the other.
            deadline = time.monotonic() + OVERALL_BUDGET_S
            while time.monotonic() < deadline and not marks:
                time.sleep(0.05)

            if "ready" in marks:
                lead = marks["ready"] - accepted
                print(f"   sidecar ready at {marks['ready']:.2f}s  (lead {lead:+.2f}s)")
                if lead < MIN_LEAD_S:
                    failures.append(
                        f"accepted at {accepted:.2f}s but the sidecar only became ready "
                        f"at {marks['ready']:.2f}s — a lead of {lead:+.2f}s means the "
                        f"listen socket is opening after the sidecar poll, which is gh #282")
                else:
                    print(f"PASS: accepted {lead:.2f}s before the sidecar was ready")
            elif "none" in marks or "gaveup" in marks:
                # No sidecar to wait for. Fall back to the absolute bound.
                print("   no sidecar resolved — checking the absolute bound instead")
                if accepted > NO_SIDECAR_DEADLINE_S:
                    failures.append(
                        f"no sidecar ran, yet accept took {accepted:.2f}s "
                        f"(limit {NO_SIDECAR_DEADLINE_S:.0f}s)")
                else:
                    print(f"PASS: accepted in {accepted:.2f}s with no sidecar")
            else:
                failures.append(
                    "could not tell what the sidecar did — no ready, no give-up and no "
                    "not-found line. The test cannot conclude anything; check the log "
                    "below rather than trusting a pass.")

            # A queued connection is only worth anything if it gets served.
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=10) as s:
                    s.sendall(b"PING\r\n")
                    if not s.recv(64).startswith(b"+PONG"):
                        failures.append("connection accepted but PING did not answer +PONG")
                    else:
                        print("PASS: PING answered +PONG")
            except OSError as exc:
                failures.append(f"connection accepted but PING failed: {exc}")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    if failures:
        print("\n--- server output ---")
        for line in lines[:40]:
            print(f"  {line}")
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\ngh #282: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
