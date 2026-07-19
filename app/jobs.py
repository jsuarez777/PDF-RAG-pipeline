"""Background job infrastructure for the PDF viewer.

Long-running work (conversion, OCR, chunking, indexing, evals, pipelines)
is persisted to the jobs table and executed one at a time by a worker
thread; endpoints enqueue and the frontend polls /api/jobs/<id>.

Log routing is per user: every log record emitted while a request or job
is executing carries the owning user's id (a contextvar), and the SSE
handler fans it out only to that user's connected browsers.
"""

import contextvars
import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque

import db

log = logging.getLogger(__name__)

# The user on whose behalf the current thread is working (set per request
# by the app, and per job by the worker). Routes log lines to that user's
# SSE stream.
log_uid: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "log_uid", default=None
)


class JobError(Exception):
    """User-facing job failure message (shown in the UI verbatim)."""


class LogBroadcaster:
    """Fan-out of log lines to every connected SSE client, with replay history."""

    def __init__(self, history: int = 200):
        self._clients: list[queue.Queue] = []
        self._history: deque[str] = deque(maxlen=history)
        self._lock = threading.Lock()

    def publish(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        stamped = f"[{time.strftime('%H:%M:%S')}] {line}"
        with self._lock:
            self._history.append(stamped)
            for q in self._clients:
                q.put(stamped)

    def listen(self) -> tuple[queue.Queue, list[str]]:
        q: queue.Queue = queue.Queue()
        with self._lock:
            backlog = list(self._history)
            self._clients.append(q)
        return q, backlog

    def drop(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)


_broadcasters: dict[int, LogBroadcaster] = {}
_broadcasters_lock = threading.Lock()


def broadcaster_for(uid: int) -> LogBroadcaster:
    with _broadcasters_lock:
        if uid not in _broadcasters:
            _broadcasters[uid] = LogBroadcaster()
        return _broadcasters[uid]


class BroadcastHandler(logging.Handler):
    """Mirrors log records to the SSE pane of the user who caused them."""

    def emit(self, record: logging.LogRecord) -> None:
        uid = log_uid.get()
        if uid is None:
            return
        bc = broadcaster_for(uid)
        for line in self.format(record).splitlines():
            bc.publish(line)


def run_and_stream(cmd: list[str], env: dict | None = None) -> int:
    """Run a command, streaming its output through the viewer's logging."""
    log.info("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.info(line.rstrip())
    code = proc.wait()
    log.info(f"[exit code {code}]")
    return code


# ------------------------------------------------------------------ worker

_handlers: dict[str, callable] = {}
_wake = threading.Event()
_worker_started = threading.Lock()
_worker_thread: threading.Thread | None = None


def handler(kind: str):
    """Register a function(uid, params) -> result as the runner for a job kind."""

    def register(fn):
        _handlers[kind] = fn
        return fn

    return register


def enqueue(uid: int, kind: str, params: dict) -> int:
    if kind not in _handlers:
        raise ValueError(f"no handler for job kind {kind!r}")
    job_id = db.create_job(uid, kind, params)
    log.info(f"Queued job {job_id}: {kind}")
    _wake.set()
    return job_id


def _run_one(job) -> None:
    uid = job["user_id"]
    params = json.loads(job["params"])
    token = log_uid.set(uid)  # route this job's logs to its owner's SSE pane
    try:
        log.info(f"Starting job {job['id']}: {job['kind']}")
        result = _handlers[job["kind"]](uid, params)
        db.finish_job(job["id"], result=result)
        log.info(f"Job {job['id']} done")
    except JobError as exc:
        db.finish_job(job["id"], error=str(exc))
        log.error(f"Job {job['id']} failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - worker must survive anything
        db.finish_job(job["id"], error=f"internal error: {exc}")
        log.exception(f"Job {job['id']} crashed")
    finally:
        log_uid.reset(token)


def _worker_loop() -> None:
    while True:
        job = db.claim_next_job()
        if job is None:
            _wake.wait(timeout=2.0)
            _wake.clear()
            continue
        _run_one(job)


def start_worker() -> None:
    """Start the single job worker thread (idempotent)."""
    global _worker_thread
    with _worker_started:
        if _worker_thread is not None:
            return
        stale = db.fail_stale_jobs()
        if stale:
            log.warning(f"Marked {stale} stale job(s) from a previous run as failed")
        _worker_thread = threading.Thread(
            target=_worker_loop, name="job-worker", daemon=True
        )
        _worker_thread.start()
