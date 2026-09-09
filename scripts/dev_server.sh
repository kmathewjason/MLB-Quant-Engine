#!/usr/bin/env bash
# scripts/dev_server.sh — manage the FastAPI dev server lifecycle
#
# Usage:
#   ./scripts/dev_server.sh start   — kill any stale process on :8000, launch
#                                     uvicorn fully detached, return immediately
#   ./scripts/dev_server.sh stop    — kill the PID recorded in logs/api.pid
#   ./scripts/dev_server.sh status  — report PID liveness + /health HTTP check
#
# All output from uvicorn goes to logs/api.log.
# The server PID is tracked in logs/api.pid.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (relative to the project root, which is the parent of this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOGS_DIR}/api.pid"
LOG_FILE="${LOGS_DIR}/api.log"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PORT=8000
HOST="127.0.0.1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ensure_logs_dir() {
    mkdir -p "${LOGS_DIR}"
}

# Kill any process currently bound to $PORT (stale uvicorn, test server, etc.)
_kill_port() {
    local pids
    pids="$(lsof -ti tcp:${PORT} 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        echo "  → Killing stale process(es) on port ${PORT}: ${pids}"
        # shellcheck disable=SC2086
        kill -9 ${pids} 2>/dev/null || true
        sleep 0.3
    fi
}

_read_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        cat "${PID_FILE}"
    else
        echo ""
    fi
}

_pid_alive() {
    local pid="${1:-}"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
cmd_start() {
    _ensure_logs_dir
    _kill_port

    # Also kill any PID we have on record (may be dead already — ignore errors)
    local old_pid
    old_pid="$(_read_pid)"
    if _pid_alive "${old_pid}"; then
        echo "  → Stopping previously recorded server PID ${old_pid}"
        kill -9 "${old_pid}" 2>/dev/null || true
        sleep 0.2
    fi

    echo "  → Starting uvicorn on ${HOST}:${PORT} …"

    # nohup + & + disown: three-layer detachment so the shell exits immediately
    # and the server keeps running after this script returns.
    # stdout/stderr both go to api.log; the PID is captured before disown.
    (
        cd "${PROJECT_ROOT}"
        nohup "${PYTHON}" -m uvicorn src.api:app \
            --host "${HOST}" \
            --port "${PORT}" \
            --log-level info \
            >> "${LOG_FILE}" 2>&1 &
        echo $! > "${PID_FILE}"
        disown $!
    )

    # Give the process ~0.5 s to start (just enough to confirm it didn't
    # immediately crash), then read back the PID and report.
    sleep 0.5
    local new_pid
    new_pid="$(_read_pid)"

    if _pid_alive "${new_pid}"; then
        echo "  ✓ Server started (PID ${new_pid})"
        echo "    Log:  ${LOG_FILE}"
        echo "    PID:  ${PID_FILE}"
        echo ""
        echo "  Wait 2-3 s then verify:"
        echo "    curl http://${HOST}:${PORT}/health"
    else
        echo "  ✗ Server failed to start — last 20 lines of log:"
        echo "  ---"
        tail -20 "${LOG_FILE}" 2>/dev/null || echo "  (log empty)"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------
cmd_stop() {
    local pid
    pid="$(_read_pid)"

    # Also kill anything on the port (in case PID file is stale)
    _kill_port

    if [[ -z "${pid}" ]]; then
        echo "  → No PID file found at ${PID_FILE} — nothing to stop"
        return 0
    fi

    if _pid_alive "${pid}"; then
        echo "  → Stopping server PID ${pid} …"
        kill -15 "${pid}" 2>/dev/null || true
        sleep 0.5
        if _pid_alive "${pid}"; then
            kill -9 "${pid}" 2>/dev/null || true
        fi
        echo "  ✓ Stopped"
    else
        echo "  → PID ${pid} is not running"
    fi

    rm -f "${PID_FILE}"
}

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
cmd_status() {
    local pid
    pid="$(_read_pid)"

    echo "── Process ──────────────────────────────────────────────"
    if [[ -z "${pid}" ]]; then
        echo "  PID file: not found"
    elif _pid_alive "${pid}"; then
        echo "  PID ${pid}: ✓ running"
        ps -p "${pid}" -o pid,etime,command | tail -1
    else
        echo "  PID ${pid}: ✗ dead (stale pid file)"
    fi

    echo ""
    echo "── HTTP ─────────────────────────────────────────────────"
    local http_out
    if http_out="$(curl -s --max-time 3 "http://${HOST}:${PORT}/health" 2>&1)"; then
        echo "  GET /health → 200 OK"
        echo "  ${http_out}"
    else
        echo "  GET /health → ✗ no response (server not ready or not running)"
    fi

    echo ""
    echo "── Port ─────────────────────────────────────────────────"
    local port_pids
    port_pids="$(lsof -ti tcp:${PORT} 2>/dev/null || true)"
    if [[ -n "${port_pids}" ]]; then
        echo "  :${PORT} is in use by PID(s): ${port_pids}"
    else
        echo "  :${PORT} is free"
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
CMD="${1:-}"
case "${CMD}" in
    start)  cmd_start  ;;
    stop)   cmd_stop   ;;
    status) cmd_status ;;
    *)
        echo "Usage: $(basename "$0") {start|stop|status}"
        echo ""
        echo "  start   Kill anything on :${PORT}, launch uvicorn detached"
        echo "  stop    Gracefully stop the recorded server PID"
        echo "  status  Show PID liveness and /health HTTP response"
        exit 1
        ;;
esac
