# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for workload comparability, capture integrity and private metadata."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from common import (
    bundle_identity,
    bundle_payloads,
    read_cases,
    save_json,
    sha256,
    write_case,
)
from make_cases import generate
import run_phone
from summarize import capture_name, markdown, summarize, validate_rows

ROOT = Path(__file__).parent
SEED = ROOT / "fixtures/performance_1024_64.tsv"
CASE = {"id": "test", "prompt": [2, 10, 20], "forced": [30, 40]}


def rows(runner, scale=1):
    return [
        {
            "case": "test",
            "repetition": rep,
            "step": step,
            "context": 3 + step,
            "argmax": 42 + step,
            "setup_ms": 1000,
            "prefix_ms": (90 if rep < 0 else 10) * scale,
            "token_ms": (100 if rep < 0 else [5, 2, 4][step]) * scale,
            "peak_rss_kib": 2048 if rep < 0 else 1024,
            **({"consumed_token_ids_verified": True} if runner == "litert_lm" else {}),
        }
        for rep in range(-1, 2)
        for step in range(3)
    ]


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_fixture_identity(self):
        self.assertEqual(
            sha256(SEED),
            "08f95c4270b7eee5cb116ba6a6eb9d2b9ad541070c29e000ccfb61178825a3fd",
        )
        (case,) = read_cases(SEED)
        self.assertEqual((len(case["prompt"]), len(case["forced"])), (1024, 64))

    def test_generated_cases_share_prefix_and_continuation(self):
        path = generate(SEED, [1, 128, 1024, 4096], 64, self.root / "generated")
        cases = read_cases(path)
        (seed,) = read_cases(SEED)
        self.assertEqual(cases[2]["prompt"], seed["prompt"])
        for case in cases:
            self.assertEqual(case["prompt"], cases[-1]["prompt"][: len(case["prompt"])])
            self.assertEqual(case["forced"], seed["forced"])
        with self.assertRaises(ValueError):
            generate(SEED, [0], 64, self.root / "invalid")
        with self.assertRaises(ValueError):
            generate(SEED, [128, 128], 64, self.root / "duplicate")

    def test_malformed_fixtures_are_rejected(self):
        path = self.root / "cases.tsv"
        for text in (
            "../case\t2\t3\n",
            "case\t2\t262144\n",
            "case\t2\t-\ncase\t2\t-\n",
            "",
        ):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_cases(path)

    def test_plan_alternates_and_skips_insufficient_capacity(self):
        plan, skipped = run_phone.plan_runs([CASE], [4, 5], 2)
        self.assertEqual(
            [item["runner"] for item in plan],
            ["litert", "litert_lm", "litert_lm", "litert"],
        )
        self.assertTrue(
            all(item["case"] == "test" and item["capacity"] == 5 for item in plan)
        )
        self.assertEqual(skipped[0]["capacity"], 4)
        with self.assertRaises(ValueError):
            run_phone.plan_runs([CASE], [4], 2)
        with self.assertRaises(ValueError):
            run_phone.plan_runs([CASE], [5, 5], 2)

    def test_commands_use_matched_options_and_separate_caches(self):
        args = SimpleNamespace(
            mask="f0",
            threads=4,
            warmups=1,
            runs=3,
            compact_int2=True,
            dynamic_prefill=True,
            strict_shapes=True,
            lm_prefill="prefill_128",
        )
        candidate = run_phone.runner_command(
            "litert", "artifacts", "out", "case", "cache", 8448, args
        )
        baseline = run_phone.runner_command(
            "litert_lm", "artifacts", "out", "case", "cache", 8448, args, prime=True
        )
        self.assertEqual(candidate[:2], ["taskset", "f0"])
        for flag in (
            "experimental_compact_int2",
            "experimental_dynamic_prefill",
            "experimental_explicit_dynamic_shapes",
        ):
            self.assertIn("--" + flag, candidate)
            self.assertNotIn("--" + flag, baseline)
        self.assertEqual(
            candidate[candidate.index("--weight_cache_path") + 1],
            "cache/litert.xnnpack_cache",
        )
        self.assertEqual(baseline[baseline.index("--cache_dir") + 1], "cache/lm")
        self.assertEqual(
            baseline[baseline.index("--prefill_signature") + 1], "prefill_128"
        )
        self.assertEqual(baseline[baseline.index("--warmup_runs") + 1], "0")
        self.assertEqual(baseline[baseline.index("--measured_runs") + 1], "1")

    def test_metrics_exclude_warmup_and_first_step_from_decode(self):
        sessions, peak = validate_rows(rows("litert"), CASE, 1, 2, "litert")
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["first_logits_ms"], 15)
        self.assertEqual(sessions[0]["decode_ms"], 3)
        self.assertEqual(sessions[0]["request_ms"], 21)
        self.assertEqual(peak, 2048)  # Process high-water mark includes warmup.

    def test_invalid_measurements_are_rejected(self):
        original = rows("litert_lm")
        corruptions = [original[:-1], original + [original[0]]]
        for field, value in (
            ("token_ms", float("nan")),
            ("prefix_ms", -1),
            ("context", 999),
            ("argmax", -1),
            ("consumed_token_ids_verified", False),
        ):
            corrupt = copy.deepcopy(original)
            corrupt[0][field] = value
            corruptions.append(corrupt)
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt[0]):
                with self.assertRaises(ValueError):
                    validate_rows(corrupt, CASE, 1, 2, "litert_lm")

    def make_campaign(self):
        write_case(self.root / "workload.tsv", CASE)
        plan, skipped = run_phone.plan_runs([CASE], [5], 2)
        campaign = {
            "schema_version": 1,
            "status": "completed",
            "fixture_sha256": sha256(self.root / "workload.tsv"),
            "capacities": [5],
            "blocks": 2,
            "runs": 2,
            "warmups": 1,
            "threads": 4,
            "lm_prefill": "auto",
            "plan": plan,
            "skipped": skipped,
        }
        save_json(self.root / "campaign.json", campaign)
        save_json(self.root / "device.json", {"model": "test-device"})
        for item in plan:
            target = self.root / capture_name(item)
            (target / "capture").mkdir(parents=True)
            save_json(
                target / "capture.json",
                {**item, "fixture_sha256": campaign["fixture_sha256"]},
            )
            save_json(target / "status.json", {"completed": True, "returncode": 0})
            measurements = rows(
                item["runner"], 2 if item["runner"] == "litert_lm" else 1
            )
            if item["runner"] == "litert_lm":
                measurements[-1]["argmax"] += 1
                save_json(
                    target / "capture/manifest.json",
                    {
                        "schema_version": 1,
                        "status": "completed",
                        "runner": "litert_lm",
                        "backend": "cpu_xnnpack",
                        "speculative_decoding": False,
                        "timings_valid_for_benchmark": True,
                        "cache_policy": "persistent",
                        "threads": 4,
                        "capacity": 5,
                        "warmups": 1,
                        "runs": 2,
                        "prefill_signature": "auto",
                    },
                )
            (target / "capture/timings.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in measurements)
            )
            # A summary must never include private diagnostics.
            (target / "process.log").write_text(str(self.root))
        return campaign

    def test_summary_pairs_blocks_and_reports_disagreement(self):
        self.make_campaign()
        result = summarize(self.root)
        (comparison,) = result["comparisons"]
        self.assertEqual(comparison["runners"]["litert"]["measured_sessions"], 4)
        self.assertEqual(
            comparison["comparison"]["decode_ms"]["baseline_over_candidate"], 2
        )
        self.assertEqual(comparison["argmax_agreement"], {"matches": 10, "total": 12})
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertIn("10/12", markdown(result))

    def test_summary_rejects_traced_or_mismatched_lm_runs(self):
        campaign = self.make_campaign()
        item = next(item for item in campaign["plan"] if item["runner"] == "litert_lm")
        path = self.root / capture_name(item) / "capture/manifest.json"
        original = json.loads(path.read_text())
        for field, value in (
            ("timings_valid_for_benchmark", False),
            ("capacity", 6),
            ("prefill_signature", "prefill_128"),
        ):
            with self.subTest(field=field):
                save_json(path, {**original, field: value})
                with self.assertRaises(ValueError):
                    summarize(self.root)

    def test_summary_rejects_incomplete_campaign_and_fixture_changes(self):
        campaign = self.make_campaign()
        save_json(self.root / "campaign.json", {**campaign, "status": "running"})
        with self.assertRaises(ValueError):
            summarize(self.root)
        save_json(self.root / "campaign.json", campaign)
        write_case(self.root / "workload.tsv", {**CASE, "prompt": [2]})
        with self.assertRaises(ValueError):
            summarize(self.root)

    def make_bundle(self):
        bundle = self.root / "weights"
        bundle.mkdir()
        source = self.root / "source.litertlm"
        source.write_bytes(b"test source")
        weights = bundle / "weight.bin"
        weights.write_bytes(b"test coefficients")
        manifest = {
            "schema_version": 2,
            "status": "complete",
            "source_bundle": {
                "path": str(source),
                "sha256": sha256(source),
                "bytes": source.stat().st_size,
            },
            "tensors": [{"file": weights.name, "sha256": sha256(weights)}],
            "constants": [],
        }
        save_json(bundle / "manifest.json", manifest)
        return bundle, source

    def test_bundle_identity_excludes_private_source_path_and_validates_payloads(self):
        bundle, _ = self.make_bundle()
        self.assertNotIn(str(self.root), json.dumps(bundle_identity(bundle)))
        self.assertEqual(len(bundle_payloads(bundle)), 1)
        (bundle / "weight.bin").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            bundle_payloads(bundle)

    def test_dry_run_never_connects_and_omits_private_arguments(self):
        bundle, source = self.make_bundle()
        lm_source = self.root / "official"
        (lm_source / "gemma4_benchmark").mkdir(parents=True)
        shutil.copyfile(
            ROOT / "litert_lm_harness/adapter.cc",
            lm_source / "gemma4_benchmark/adapter.cc",
        )
        binary = self.root / "binary"
        binary.write_bytes(b"test binary")
        output = self.root / "plan"
        args = [
            "run_phone.py",
            "--dry-run",
            "--serial",
            "test-private-transport",
            "--remote",
            "/test-device-directory",
            "--output",
            str(output),
            "--litert-src",
            str(self.root),
            "--lm-src",
            str(lm_source),
            "--runner-bin",
            str(binary),
            "--lm-bin",
            str(binary),
            "--model",
            str(binary),
            "--source-model",
            str(source),
            "--bundle-dir",
            str(bundle),
            "--mask",
            "f0",
        ]
        with (
            mock.patch("sys.argv", args),
            mock.patch.object(run_phone, "Device") as device,
            mock.patch.object(
                run_phone,
                "source_identity",
                return_value={"commit": "a" * 40, "tracked_changes": False},
            ),
        ):
            run_phone.main()
            device.assert_not_called()
        text = (output / "campaign.json").read_text()
        self.assertNotIn(str(self.root), text)
        self.assertNotIn("test-private-transport", text)
        self.assertNotIn("/test-device-directory", text)
        self.assertEqual(sha256(output / "workload.tsv"), sha256(SEED))

    def test_thermal_aggregates_ignore_sensor_identifiers(self):
        sample = run_phone.parse_thermal(
            "Current temperatures from HAL:\n"
            "Temperature{mValue=39.0, mType=0, mName=cpu-zone}\n"
            "Temperature{mValue=42.0, mType=0, mName=another-zone}\n"
            "Temperature{mValue=33.0, mType=3, mName=skin-zone}\n"
            "Current cooling devices from HAL:\nCoolingDevice{mValue=1, mType=2, mName=cpu-cooling}"
        )
        self.assertTrue(sample["available"])
        self.assertTrue(sample["cpu_cooling_active"])
        self.assertEqual(sample["cpu_c"], 42)
        self.assertNotIn("zone", json.dumps(sample))
        self.assertFalse(run_phone.parse_thermal("unsupported")["available"])

    def test_thermal_gate_ignores_stale_cached_cooling(self):
        sample = run_phone.parse_thermal(
            "Cached cooling devices:\nCoolingDevice{mValue=1, mType=2,}\n"
            "Current temperatures from HAL:\n"
            "Temperature{mValue=40, mType=0,}\nTemperature{mValue=30, mType=3,}\n"
            "Current cooling devices from HAL:\nCoolingDevice{mValue=0, mType=2,}"
        )
        self.assertTrue(sample["available"])
        self.assertFalse(sample["cpu_cooling_active"])


if __name__ == "__main__":
    unittest.main()
