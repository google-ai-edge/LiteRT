# Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthetic capture tests; no models, builds, or devices are used."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import summarize_performance as perf


class PerformanceSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.make_capture("native")
        self.make_capture("litert-lm")

    def directory(self, role):
        return self.root / role / "128"

    def save(self, path, data):
        path.write_text(json.dumps(data))

    def mutate(self, role, name, change):
        path = self.directory(role) / name
        data = json.loads(path.read_text())
        change(data)
        self.save(path, data)

    def make_capture(self, role, ynnpack=False):
        directory = self.directory(role)
        directory.mkdir(parents=True)
        prompt, forced = [2, 3], [4, 5]
        filenames = []
        for index, (warmup, timings) in enumerate([
                (True, [90000, 90000, 90000]),
                (False, [100, 10, 90]),
                (False, [300, 20, 20]),
                (False, [200, 25, 25])]):
            n = 0 if warmup else index - 1
            name = f"case.{'warmup' if warmup else 'run'}_{n:03d}.json"
            filenames.append(name)
            record = {
                "schema_version": 1, "case_id": "case", "warmup": warmup,
                "run_index": -1 if warmup and role == "litert-lm" else n,
                "runner": "xnnpack_tensor_live_int8" if role == "native" else "litert_lm_cpu",
                "num_threads": 4, "reuse_runtimes": True, "vocab_size": 262144,
                "prompt_token_ids": prompt, "forced_decode_token_ids": forced,
                "timings_valid_for_benchmark": True, "passes": [],
            }
            if role == "native":
                record.update(kv_capacity=8448, memory_report=False,
                              prefill_rows=128, kv_alignment=32,
                              fixed_attention_extent=False,
                              preserve_static_int2=True, share_workspace=True,
                              static_int2_tensor_count=60, static_int2_operator_count=60,
                              static_int2_compact_bytes=283115520, kv_dtype="int8")
            else:
                record.update(max_num_tokens=8448, status="completed",
                              enable_ynnpack=ynnpack, enable_profiling=False,
                              context_magic_number_configuration_verified=True)
            for step, elapsed in enumerate(timings):
                record["passes"].append({
                    "kind": "prefill" if step == 0 else "decode",
                    "decode_index": step,
                    "input_ids": prompt if step == 0 else [forced[step - 1]],
                    "context_length_after": len(prompt) + step,
                    "logits_position": len(prompt) + step - 1,
                    "argmax_id": 6, "elapsed_ms": elapsed, "forward_ms": 1,
                    "logits_file": None,
                })
            self.save(directory / name, record)
        completion = {"status": "completed", "cases": 1}
        if role == "litert-lm":
            completion.update(measured_runs=3, warmup_runs=1, runs=filenames,
                              num_threads=4, max_num_tokens=8448,
                              reuse_runtimes=True, enable_ynnpack=ynnpack,
                              enable_profiling=False, timings_valid_for_benchmark=True)
            self.save(directory / "manifest.json", completion)
        self.save(directory / "run.json", completion)

    def assert_rejected(self):
        with self.assertRaises(perf.CaptureError):
            perf.summarize(self.root)

    def test_total_decode_time_then_session_median_excludes_warmup(self):
        summary = perf.summarize(self.root)
        self.assertEqual(summary["excluded_warmups_per_case"], 1)
        for row in summary["rows"]:
            self.assertEqual(row["median_prefill_seconds"], 0.2)
            self.assertEqual(row["median_decode_tokens_per_second"], 40)
            self.assertEqual([r["decode_tokens_per_second"] for r in row["sessions"]],
                             [20, 50, 40])
            self.assertEqual(len(row["sessions"]), 3)
        self.assertEqual(summary["rows"][1]["runner"], "LiteRT-LM CPU (XNNPACK)")

    def test_ynnpack_label_is_explicit(self):
        for path in self.directory("litert-lm").glob("*.json"):
            record = json.loads(path.read_text())
            record["enable_ynnpack"] = True
            self.save(path, record)
        self.assertEqual(perf.summarize(self.root)["rows"][1]["runner"],
                         "LiteRT-LM CPU (YNNPACK)")

    def test_missing_native_final_repetition_is_rejected(self):
        (self.directory("native") / "case.run_002.json").unlink()
        self.assert_rejected()

    def test_missing_warmup_is_rejected(self):
        (self.directory("native") / "case.warmup_000.json").unlink()
        self.assert_rejected()

    def test_missing_completion_is_rejected(self):
        (self.directory("native") / "run.json").unlink()
        self.assert_rejected()

    def test_partial_completion_is_rejected(self):
        self.mutate("native", "run.json", lambda d: d.update(status="running"))
        self.assert_rejected()

    def test_lm_index_must_list_all_sessions(self):
        for name in ("manifest.json", "run.json"):
            self.mutate("litert-lm", name, lambda d: d["runs"].pop())
        self.assert_rejected()

    def test_incomplete_passes_are_rejected(self):
        self.mutate("native", "case.run_000.json", lambda d: d["passes"].pop())
        self.assert_rejected()

    def test_forced_history_mismatch_is_rejected(self):
        for path in self.directory("native").glob("case.*.json"):
            data = json.loads(path.read_text())
            data["forced_decode_token_ids"][0] = 9
            data["passes"][1]["input_ids"] = [9]
            self.save(path, data)
        self.assert_rejected()

    def test_pass_history_mismatch_is_rejected(self):
        self.mutate("native", "case.run_000.json",
                    lambda d: d["passes"][1].update(input_ids=[9]))
        self.assert_rejected()

    def test_capacity_threads_and_reuse_must_match_between_runners(self):
        paths = list(self.directory("native").glob("case.*.json"))
        originals = {p: p.read_text() for p in paths}
        for key, value in [("kv_capacity", 4096), ("num_threads", 2),
                           ("reuse_runtimes", False)]:
            with self.subTest(key=key):
                for p, text in originals.items():
                    data = json.loads(text)
                    data[key] = value
                    self.save(p, data)
                self.assert_rejected()
        for p, text in originals.items():
            p.write_text(text)

    def test_nonfinite_negative_zero_and_boolean_timings_are_rejected(self):
        path = self.directory("native") / "case.run_000.json"
        original = json.loads(path.read_text())
        for bad in [float("nan"), float("inf"), -1, 0, True]:
            with self.subTest(value=bad):
                data = copy.deepcopy(original)
                data["passes"][0]["elapsed_ms"] = bad
                self.save(path, data)
                self.assert_rejected()

    def test_profiling_memory_and_dump_records_are_rejected(self):
        path = self.directory("native") / "case.run_000.json"
        original = json.loads(path.read_text())
        for key in ["enable_profiling", "memory_report", "dump_full_logits"]:
            with self.subTest(key=key):
                data = copy.deepcopy(original)
                data[key] = True
                self.save(path, data)
                self.assert_rejected()
        data = copy.deepcopy(original)
        data["passes"][0]["logits_file"] = "case.prefill.f32"
        self.save(path, data)
        self.assert_rejected()

    def test_diagnostic_command_without_dump_files_is_rejected(self):
        self.save(self.directory("native") / "command.json",
                  {"argv": ["runner", "--trace_position=5"]})
        self.assert_rejected()

    def test_trace_files_are_rejected(self):
        (self.directory("native") / "trace.5.hidden.f32").write_bytes(b"test")
        self.assert_rejected()

    def test_inconsistent_native_optimization_flags_are_rejected(self):
        self.mutate("native", "case.run_001.json", lambda d: d.update(share_workspace=False))
        self.assert_rejected()

    def test_malformed_json_and_duplicate_keys_are_rejected(self):
        path = self.directory("native") / "case.run_000.json"
        for text in ['{"passes":', '{"x": 1, "x": 2}']:
            with self.subTest(text=text):
                path.write_text(text)
                self.assert_rejected()

    def test_malformed_completion_file_list_is_rejected(self):
        for name in ("manifest.json", "run.json"):
            self.mutate("litert-lm", name, lambda d: d.update(runs=[{}]))
        self.assert_rejected()

    def test_missing_native_optimization_metadata_is_rejected(self):
        self.mutate("native", "case.run_000.json", lambda d: d.pop("share_workspace"))
        self.assert_rejected()

    def test_overflowing_derived_throughput_is_rejected(self):
        self.mutate("native", "case.run_000.json",
                    lambda d: [item.update(elapsed_ms=1e-310) for item in d["passes"][1:]])
        self.assert_rejected()

    def test_cli_writes_json_and_refuses_overwrite(self):
        output = self.root / "summary.json"
        command = [sys.executable, "-B", str(Path(perf.__file__)), str(self.root),
                   "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LiteRT-LM CPU (XNNPACK)", result.stdout)
        self.assertEqual(len(json.loads(output.read_text())["rows"]), 2)
        repeat = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(repeat.returncode, 0)


if __name__ == "__main__":
    unittest.main()
