"""Opt-in, in-process release experiment; never retain the observed models."""
import ctypes
import gc
import sys
import time
import weakref
from pathlib import Path


def memory_details():
    """Current-process snapshots in bytes (not process-lifetime peak RSS)."""
    values = {}
    for filename, fields in (
        ("status", {"VmRSS": "process_rss_bytes", "RssAnon": "process_anon_bytes"}),
        ("smaps_rollup", {"Pss": "process_pss_bytes", "Private_Dirty": "process_private_dirty_bytes"}),
    ):
        try:
            for line in Path(f"/proc/self/{filename}").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in fields:
                    values[fields[key]] = int(value.split()[0]) * 1024
        except (OSError, ValueError):
            pass
    return values


def trim_glibc():
    """Ask this process's glibc to release free pages; no effect on live objects."""
    if sys.platform != "linux":
        return {"trim_supported": False, "trim_skip_reason": "not_linux"}
    libc = ctypes.CDLL(None)
    if not hasattr(libc, "gnu_get_libc_version") or not hasattr(libc, "malloc_trim"):
        return {"trim_supported": False, "trim_skip_reason": "glibc_symbols_unavailable"}
    trim = libc.malloc_trim
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return {"trim_supported": True, "trim_return_code": trim(0)}


class MemoryProbe:
    def __init__(self):
        self.objects = {}

    def _track(self, name, obj):
        if obj is None:
            return
        type_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
        try:
            ref = weakref.ref(obj)
        except TypeError:
            ref = None
        # Store strings and weak references only, including for unsupported objects.
        self.objects[name] = (type_name, ref)

    def track_parser(self, parser):
        """Track known Docling model owners, without importing Torch or Docling."""
        self._track("parser", parser)
        converter = getattr(parser, "converter", None)
        self._track("converter", converter)
        pipelines = getattr(converter, "initialized_pipelines", {})
        for index, pipeline in enumerate(pipelines.values()):
            prefix = f"pipeline[{index}]"
            self._track(prefix, pipeline)
            layout = getattr(pipeline, "layout_model", None)
            self._track(f"{prefix}.layout", layout)
            engine = getattr(layout, "engine", None)
            self._track(f"{prefix}.layout.engine", engine)
            self._track(f"{prefix}.layout.model", getattr(engine, "_model", None))
            table = getattr(pipeline, "table_model", None)
            self._track(f"{prefix}.table", table)
            predictor = getattr(table, "tf_predictor", None)
            self._track(f"{prefix}.table.predictor", predictor)
            self._track(f"{prefix}.table.model", getattr(predictor, "_model", None))

    def snapshot(self, emit, checkpoint, previous=None, **details):
        memory = memory_details()
        current = memory.get("process_rss_bytes")
        previous_rss = (previous or {}).get("process_rss_bytes")
        if current is not None and previous_rss is not None:
            details["rss_released_since_previous_bytes"] = previous_rss - current
        emit(
            "memory_probe",
            checkpoint=checkpoint,
            objects={
                name: {"type": kind, "alive": ref() is not None if ref is not None else None}
                for name, (kind, ref) in self.objects.items()
            },
            **memory,
            **details,
        )
        return memory

    def run(self, emit):
        """Called after the task coroutine returns, in the same worker, without awaits."""
        try:
            baseline = self.snapshot(emit, "after_return")
            started = time.monotonic()
            collected = gc.collect()
            after_gc = self.snapshot(
                emit, "after_gc", previous=baseline,
                gc_collected=collected, operation_seconds=time.monotonic() - started,
            )
            started = time.monotonic()
            trim_result = trim_glibc()
            self.snapshot(
                emit, "after_trim", previous=after_gc,
                operation_seconds=time.monotonic() - started, **trim_result,
            )
        except Exception as error:
            # Diagnostics must not turn a committed ingest into a failed task.
            emit("memory_probe_error", error_type=type(error).__name__)
