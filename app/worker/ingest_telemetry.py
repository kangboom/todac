"""Opt-in ingest measurements. No document text is written to logs.

Enable with INGEST_MEASUREMENTS_DIR. Each attempt gets its own JSONL file.
Stage snapshots complement the container metrics collected by cAdvisor;
process_peak_rss_bytes is a process-lifetime high-water mark, not a stage peak.
"""
import hashlib
import inspect
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)
_active = ContextVar("ingest_measurement", default=None)


def process_memory(pid="self"):
    """Linux process RSS and lifetime high-water mark, in bytes; None elsewhere."""
    values = {"process_rss_bytes": None, "process_peak_rss_bytes": None}
    try:
        fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
        for key, field in (("process_rss_bytes", "VmRSS"), ("process_peak_rss_bytes", "VmHWM")):
            if field in fields:
                values[key] = int(fields[field].split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def chunk_digest(chunks):
    """Order-independent digest of stored chunk identities/content (not vectors)."""
    digest = hashlib.sha256()
    for index, content in sorted((c.chunk_index, c.text[:65535]) for c in chunks):
        digest.update(json.dumps([index, content], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class Measurement:
    def __init__(self, doc_id):
        self.doc_id = str(uuid.UUID(str(doc_id)))
        self.attempt_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.stage_name = "setup"
        self.completed = False
        self.directory = Path(os.environ["INGEST_MEASUREMENTS_DIR"])

    def emit(self, event, **details):
        # Measurement failures must not change ingestion behavior.
        try:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event, "doc_id": self.doc_id, "attempt_id": self.attempt_id,
                "pid": os.getpid(), "stage": self.stage_name,
                "elapsed_seconds": round(time.monotonic() - self.started, 6),
                **process_memory(), **details,
            }
            payload = json.dumps(record, ensure_ascii=False)
            # Keep stdout evidence even when the measurement volume is not writable.
            logger.info("ingest_measurement %s", payload)
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"{self.doc_id}.{self.attempt_id}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        except Exception:
            logger.exception("Failed to persist ingest measurement")

    @contextmanager
    def stage(self, name, **details):
        self.stage_name = name
        started = time.monotonic()
        self.emit("stage_start", **details)
        try:
            yield
        except BaseException as error:
            self.emit("stage_error", error_type=type(error).__name__, stage_seconds=time.monotonic() - started)
            raise
        else:
            self.emit("stage_end", stage_seconds=time.monotonic() - started)


@contextmanager
def stage(name, **details):
    measurement = _active.get()
    if measurement is None:
        yield
    else:
        with measurement.stage(name, **details):
            yield


def record(event, **details):
    measurement = _active.get()
    if measurement is not None:
        measurement.emit(event, **details)


def capture_documents(documents):
    """Persist the exact pre-chunking input only when explicitly enabled."""
    measurement = _active.get()
    if measurement is None or os.getenv("INGEST_CAPTURE_MARKDOWN", "false").lower() != "true":
        return
    try:
        folder = measurement.directory / "captures"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{measurement.doc_id}.{measurement.attempt_id}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump([doc.model_dump() for doc in documents], handle, ensure_ascii=False)
        with path.open("rb") as handle:
            input_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        measurement.emit("input_captured", capture_file=path.name, input_sha256=input_sha256)
    except (OSError, TypeError, ValueError):
        logger.exception("Failed to capture pre-chunking input")


def completed(chunks, markdown_bytes):
    measurement = _active.get()
    if measurement is not None:
        measurement.completed = True
        details = {}
        try:
            details = {"chunk_count": len(chunks), "chunks_sha256": chunk_digest(chunks),
                       "markdown_sha256": hashlib.sha256(markdown_bytes).hexdigest()}
        except Exception as error:
            # Failure to collect evidence after DB commit must not trigger rollback.
            measurement.emit("verification_evidence_error", error_type=type(error).__name__)
        measurement.emit("storage_committed", **details)


def measure_ingest(function):
    signature = inspect.signature(function)

    @wraps(function)
    async def measured(*args, **kwargs):
        if not os.getenv("INGEST_MEASUREMENTS_DIR"):
            return await function(*args, **kwargs)
        arguments = signature.bind(*args, **kwargs).arguments
        measurement = Measurement(arguments["doc_id_str"])
        token = _active.set(measurement)
        measurement.emit("task_start", file_size=arguments.get("file_size"),
                         raw_sha256=arguments.get("doc_hash"))
        outcome = "returned_without_commit"
        try:
            result = await function(*args, **kwargs)
            outcome = "completed" if measurement.completed else "returned_without_commit"
            return result
        except BaseException as error:
            outcome = "raised"
            measurement.emit("task_error", error_type=type(error).__name__)
            raise
        finally:
            # A SIGKILL produces no task_end: correlate cAdvisor and kernel OOM records.
            measurement.emit("task_end", outcome=outcome)
            _active.reset(token)

    return measured
