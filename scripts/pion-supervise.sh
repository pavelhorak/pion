#!/usr/bin/env bash
# pion-supervise.sh — gh #138: supervised serving for pion-server.
#
# Keeps a pion-server alive: starts it, waits for it to answer PING, then polls
# liveness on an interval and restarts it (with backoff) if it dies or stops
# responding. Every death is written to the supervisor log with a postmortem:
# exit status, the last heartbeat record from the server's status file (RSS as
# of ~1s before death), and — on macOS — whether the kernel's jetsam subsystem
# reported killing that pid.
#
# The postmortem is the point. A `pion-server` reaped by jetsam or the Linux
# OOM killer gets SIGKILL, which no in-process handler can ever observe; the
# only evidence is the last heartbeat plus the OS log. This script collects both.
#
# Usage:
#   scripts/pion-supervise.sh [supervisor options] [-- <pion-server args>]
#
# Supervisor options:
#   --binary PATH        pion-server binary            (default ./pion-server)
#   --port N             health-check port             (default 1974, or -p/--port from server args)
#   --interval SECONDS   health poll interval          (default 5)
#   --timeout SECONDS    per-probe response timeout    (default 3)
#   --start-timeout SEC  readiness wait after start    (default 120; big keyspaces init slowly)
#   --max-restarts N     give up after N restarts, 0 = unlimited (default 0)
#   --backoff SECONDS    initial restart backoff, doubles to 60s max (default 1)
#   --log PATH           supervisor log (default pion-supervisor-<port>.log)
#   --once               do not restart; exit with the server's status
#
# Everything after `--` is passed to pion-server verbatim.
#
# Examples:
#   scripts/pion-supervise.sh -- --profile vector -w 1 --nle-embed -p 1974
#   scripts/pion-supervise.sh --interval 2 --max-restarts 10 -- --profile kv -w 4
#
set -uo pipefail

BINARY="./pion-server"
PORT=""
INTERVAL=5
TIMEOUT=3
START_TIMEOUT=120
MAX_RESTARTS=0
BACKOFF_INIT=1
LOGFILE=""
ONCE=0
SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)        BINARY="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --interval)      INTERVAL="$2"; shift 2 ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        --start-timeout) START_TIMEOUT="$2"; shift 2 ;;
        --max-restarts)  MAX_RESTARTS="$2"; shift 2 ;;
        --backoff)       BACKOFF_INIT="$2"; shift 2 ;;
        --log)           LOGFILE="$2"; shift 2 ;;
        --once)          ONCE=1; shift ;;
        -h|--help)       sed -n '2,40p' "$0"; exit 0 ;;
        --)              shift; SERVER_ARGS=("$@"); break ;;
        *)               echo "pion-supervise: unknown option '$1' (server args go after --)" >&2; exit 2 ;;
    esac
done

# Infer the port from the server args when not given explicitly, so the health
# probe and the status-file path always match what the server actually binds.
if [[ -z "$PORT" ]]; then
    for ((i = 0; i < ${#SERVER_ARGS[@]}; i++)); do
        case "${SERVER_ARGS[$i]}" in
            -p|--port) PORT="${SERVER_ARGS[$((i + 1))]:-}" ;;
        esac
    done
fi
PORT="${PORT:-1974}"
LOGFILE="${LOGFILE:-pion-supervisor-${PORT}.log}"
STATUS_FILE="pion-${PORT}.status"

# --status-file may be overridden in the server args; follow it.
for ((i = 0; i < ${#SERVER_ARGS[@]}; i++)); do
    if [[ "${SERVER_ARGS[$i]}" == "--status-file" ]]; then
        STATUS_FILE="${SERVER_ARGS[$((i + 1))]:-$STATUS_FILE}"
    fi
done

if [[ ! -x "$BINARY" ]]; then
    echo "pion-supervise: '$BINARY' is not an executable (build with: pixi run build)" >&2
    exit 2
fi

slog() {
    local line="[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*"
    echo "$line" | tee -a "$LOGFILE"
}

# ── Health probe: RESP inline PING over a raw TCP socket ─────────────────────
# Uses bash's /dev/tcp so the supervisor needs no redis-cli. The probe runs in a
# background subshell with its own deadline: a server that accepts the
# connection but never replies (wedged event loop) must fail the check, not hang
# the supervisor. `timeout(1)` is not installed by default on macOS, hence the
# manual poll-and-kill.
probe_pong() {
    local tmp
    tmp=$(mktemp "${TMPDIR:-/tmp}/pion-probe.XXXXXX") || return 1
    (
        exec 3<>"/dev/tcp/127.0.0.1/${PORT}" || exit 1
        printf 'PING\r\n' >&3 || exit 1
        head -c 7 <&3 > "$tmp"     # "+PONG\r\n" is exactly 7 bytes
    ) 2>/dev/null &
    local pp=$! waited=0
    local limit=$((TIMEOUT * 10))
    while kill -0 "$pp" 2>/dev/null; do
        if (( waited >= limit )); then
            kill -9 "$pp" 2>/dev/null
            wait "$pp" 2>/dev/null
            rm -f "$tmp"
            return 1
        fi
        sleep 0.1
        waited=$((waited + 1))
    done
    wait "$pp" 2>/dev/null
    local reply
    reply=$(cat "$tmp" 2>/dev/null)
    rm -f "$tmp"
    [[ "$reply" == *PONG* ]]
}

status_field() {
    [[ -r "$STATUS_FILE" ]] || return 1
    awk -F= -v k="$1" '$1 == k { print $2; exit }' "$STATUS_FILE"
}

# ── Postmortem ───────────────────────────────────────────────────────────────
# `wait` gives us the exit status; >128 means a signal. But an OS memory kill is
# SIGKILL (137) with no in-process trace, so we also report the last heartbeat
# and consult the platform's memory-pressure log.
postmortem() {
    local pid="$1" status="$2"
    local detail="exit_status=$status"
    if (( status > 128 )); then
        detail="$detail signal=$((status - 128))"
    fi

    local state rss peak total pct hb uptime ticks
    state=$(status_field state || echo "?")
    rss=$(status_field rss_bytes || echo 0)
    peak=$(status_field rss_peak_bytes || echo 0)
    total=$(status_field total_ram_bytes || echo 0)
    pct=$(status_field rss_pct || echo 0)
    hb=$(status_field heartbeat_unix_s || echo 0)
    uptime=$(status_field uptime_s || echo 0)
    ticks=$(status_field ticks || echo 0)

    slog "POSTMORTEM pid=$pid $detail last_state=$state uptime_s=$uptime ticks=$ticks"
    slog "POSTMORTEM rss_mb=$((rss / 1048576)) peak_mb=$((peak / 1048576)) of total_mb=$((total / 1048576)) (${pct}%) last_heartbeat_unix_s=$hb"

    # Signal 9 with no crash-log line = killed from outside. Ask the OS who did it.
    if (( status == 137 )) || [[ "$state" == "running" ]]; then
        if [[ "$(uname -s)" == "Darwin" ]]; then
            local jetsam
            # Match on the pid, and also on the binary name — a jetsam record
            # sometimes names the victim without repeating its pid.
            local bname
            bname=$(basename "$BINARY")
            jetsam=$(/usr/bin/log show --style compact --last 10m \
                        --predicate 'eventMessage CONTAINS "jetsam" OR eventMessage CONTAINS "memorystatus" OR eventMessage CONTAINS "lowswap"' \
                        2>/dev/null | grep -E "(^|[^0-9])${pid}([^0-9]|$)|${bname}" | head -5)
            if [[ -n "$jetsam" ]]; then
                slog "POSTMORTEM CONFIRMED OS memory kill (jetsam) for pid=$pid:"
                echo "$jetsam" | while IFS= read -r l; do slog "POSTMORTEM   $l"; done
            else
                slog "POSTMORTEM no jetsam record for pid=$pid in the last 10m — killed by something else (manual kill -9, or a crash before the handler ran)"
            fi
        elif [[ -r /var/log/kern.log || -r /dev/kmsg ]]; then
            local oom
            oom=$( { dmesg 2>/dev/null || cat /var/log/kern.log 2>/dev/null; } \
                   | grep -iE "out of memory|oom-kill|killed process" | grep -E "(^|[^0-9])${pid}([^0-9]|$)" | tail -5 )
            if [[ -n "$oom" ]]; then
                slog "POSTMORTEM CONFIRMED Linux OOM kill for pid=$pid:"
                echo "$oom" | while IFS= read -r l; do slog "POSTMORTEM   $l"; done
            else
                slog "POSTMORTEM no OOM-killer record for pid=$pid — killed by something else"
            fi
        fi
    fi

    if [[ -r "pion-${PORT}.crash.log" ]]; then
        local last
        last=$(tail -2 "pion-${PORT}.crash.log")
        slog "POSTMORTEM crash-log tail:"
        echo "$last" | while IFS= read -r l; do slog "POSTMORTEM   $l"; done
    fi
}

# ── Supervision loop ─────────────────────────────────────────────────────────
SERVER_PID=""
SHUTTING_DOWN=0

on_signal() {
    SHUTTING_DOWN=1
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        slog "supervisor received shutdown signal — stopping pion-server pid=$SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    exit 0
}
trap on_signal INT TERM

restarts=0
backoff="$BACKOFF_INIT"

slog "supervisor start: binary=$BINARY port=$PORT interval=${INTERVAL}s args=${SERVER_ARGS[*]:-<none>}"

while true; do
    "$BINARY" "${SERVER_ARGS[@]}" &
    SERVER_PID=$!
    slog "started pion-server pid=$SERVER_PID"

    # Readiness: the listen socket is up before the 10M-slot hash map finishes
    # initialising, so connect success alone is not readiness — wait for +PONG.
    ready=0
    for ((w = 0; w < START_TIMEOUT; w++)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
        if probe_pong; then ready=1; break; fi
        sleep 1
    done

    if (( ready )); then
        slog "pion-server pid=$SERVER_PID is serving on port $PORT"
        backoff="$BACKOFF_INIT"   # a successful start resets the backoff
    else
        slog "pion-server pid=$SERVER_PID did not answer PING within ${START_TIMEOUT}s"
    fi

    # Health loop: exit as soon as the process dies OR stops answering.
    unhealthy=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        sleep "$INTERVAL"
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
        if ! probe_pong; then
            # One retry — a single dropped probe under load is not a death.
            sleep 1
            if ! probe_pong; then
                slog "health check FAILED (no +PONG on port $PORT) — killing pid=$SERVER_PID"
                unhealthy=1
                kill -TERM "$SERVER_PID" 2>/dev/null
                sleep 2
                kill -9 "$SERVER_PID" 2>/dev/null
                break
            fi
        fi
    done

    wait "$SERVER_PID" 2>/dev/null
    status=$?
    dead_pid="$SERVER_PID"
    SERVER_PID=""

    (( SHUTTING_DOWN )) && exit 0

    if (( unhealthy )); then
        slog "pion-server pid=$dead_pid was alive but not serving (hung event loop?)"
    else
        slog "pion-server pid=$dead_pid exited unexpectedly"
    fi
    postmortem "$dead_pid" "$status"

    if (( ONCE )); then
        slog "--once: not restarting"
        exit "$status"
    fi

    restarts=$((restarts + 1))
    if (( MAX_RESTARTS > 0 && restarts > MAX_RESTARTS )); then
        slog "giving up after $((restarts - 1)) restarts (--max-restarts $MAX_RESTARTS)"
        exit 1
    fi

    slog "restart #$restarts in ${backoff}s"
    sleep "$backoff"
    backoff=$((backoff * 2))
    (( backoff > 60 )) && backoff=60
done
