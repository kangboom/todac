"""Standalone stdlib tests: python -m unittest discover -s tests -p test_ingest_memory_probe.py."""
import asyncio
import gc
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from app.worker import ingest_memory_probe as probe_module
from app.worker import ingest_telemetry as telemetry


class Owner:
    pass


class MemoryProbeTests(unittest.TestCase):
    def test_returned_task_cycle_is_observed_then_collected_in_same_process(self):
        @telemetry.measure_ingest
        async def task(doc_id_str):
            parser = Owner()
            parser.cycle = parser
            parser.converter = Owner()
            pipeline = Owner()
            parser.converter.initialized_pipelines = {"pdf": pipeline}
            pipeline.layout_model = Owner()
            pipeline.layout_model.engine = Owner()
            pipeline.layout_model.engine._model = Owner()
            pipeline.table_model = Owner()
            pipeline.table_model.tf_predictor = Owner()
            pipeline.table_model.tf_predictor._model = Owner()
            telemetry.track_parser_memory(parser)
            # Simulate the existing cleanup: the local parser is still reachable.
            gc.collect()
            telemetry._active.get().completed = True

        enabled = gc.isenabled()
        gc.disable()
        try:
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "INGEST_MEASUREMENTS_DIR": folder, "INGEST_MEMORY_PROBE": "true",
            }), patch.object(probe_module, "trim_glibc", return_value={"trim_supported": True, "trim_return_code": 1}) as trim:
                asyncio.run(task(str(uuid.uuid4())))
                events = [json.loads(line) for line in next(Path(folder).glob("*.jsonl")).read_text().splitlines()]
                checkpoints = {event["checkpoint"]: event for event in events if event["event"] == "memory_probe"}
                self.assertEqual(len(checkpoints["after_return"]["objects"]), 9)
                self.assertTrue(all(obj["alive"] for obj in checkpoints["after_return"]["objects"].values()))
                self.assertTrue(all(obj["alive"] is False for obj in checkpoints["after_gc"]["objects"].values()))
                self.assertGreater(checkpoints["after_gc"]["gc_collected"], 0)
                self.assertEqual({event["pid"] for event in events}, {os.getpid()})
                names = [event.get("checkpoint", event["event"]) for event in events]
                self.assertLess(names.index("task_end"), names.index("after_return"))
                self.assertLess(names.index("after_gc"), names.index("after_trim"))
                trim.assert_called_once_with()
                self.assertIsNone(telemetry._active.get())
        finally:
            if enabled:
                gc.enable()

    def test_probe_does_not_keep_noncyclic_objects_alive(self):
        probe = probe_module.MemoryProbe()
        parser = Owner()
        probe.track_parser(parser)
        del parser
        emit = Mock()
        probe.snapshot(emit, "check")
        self.assertFalse(emit.call_args.kwargs["objects"]["parser"]["alive"])

    def test_disabled_probe_never_collects_or_trims(self):
        @telemetry.measure_ingest
        async def task(doc_id_str):
            telemetry.track_parser_memory(Owner())
            telemetry._active.get().completed = True
            return "original result"

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "INGEST_MEASUREMENTS_DIR": folder, "INGEST_MEMORY_PROBE": "false",
        }), patch.object(probe_module.MemoryProbe, "run") as run:
            self.assertEqual(asyncio.run(task(str(uuid.uuid4()))), "original result")
            run.assert_not_called()

    def test_raised_task_preserves_exception_and_skips_probe(self):
        @telemetry.measure_ingest
        async def task(doc_id_str):
            telemetry.track_parser_memory(Owner())
            raise ValueError("original failure")

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "INGEST_MEASUREMENTS_DIR": folder, "INGEST_MEMORY_PROBE": "true",
        }), patch.object(probe_module.MemoryProbe, "run") as run:
            with self.assertRaisesRegex(ValueError, "original failure"):
                asyncio.run(task(str(uuid.uuid4())))
            run.assert_not_called()
            self.assertIsNone(telemetry._active.get())

    def test_trim_failure_is_logged_without_raising(self):
        emit = Mock()
        with patch.object(probe_module, "trim_glibc", side_effect=OSError("unavailable")):
            probe_module.MemoryProbe().run(emit)
        self.assertEqual(emit.call_args.args, ("memory_probe_error",))
        self.assertEqual(emit.call_args.kwargs["error_type"], "OSError")

    def test_non_linux_never_loads_native_library(self):
        with patch.object(probe_module.sys, "platform", "win32"), patch.object(probe_module.ctypes, "CDLL") as cdll:
            self.assertFalse(probe_module.trim_glibc()["trim_supported"])
            cdll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
