#!/usr/bin/env python3
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
"""Summarize completed native/LiteRT-LM fixed-history CPU timing captures.

Reads RESULTS/native and RESULTS/litert-lm recursively, with one completed
capture directory per case. Defaults require three measured sessions and one
warmup; override the expected counts explicitly for a different protocol.
Prefill includes the first prediction. Decode throughput uses the total time
of all forced decode calls, never an average of per-token rates. This is an
offline metadata check, not proof of model identity, CPU affinity, or correctness.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys


class CaptureError(ValueError):
    """A capture cannot support the requested comparison."""


def require(condition, message):
    if not condition:
        raise CaptureError(message)


def read_json(path):
    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), f"{path}: nonfinite number")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=unique_object)
    except (OSError, ValueError) as error:
        raise CaptureError(f"{path}: {error}") from error
    finite(value)
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def integer(value, field, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {field}: {value!r}")
    return value


def boolean(record, field):
    value = record.get(field)
    require(type(value) is bool, f"missing/invalid boolean {field}")
    return value


def no_diagnostics(record, path):
    for key in ("enable_profiling", "memory_report", "dump_full_logits",
                "dump_logits", "dump_intermediates", "dump_cache", "profile_file"):
        require(not record.get(key), f"{path}: diagnostic capture ({key})")
    require(record.get("trace_position", -1) == -1, f"{path}: trace capture")
    if "timings_valid_for_benchmark" in record:
        require(record["timings_valid_for_benchmark"] is True,
                f"{path}: timings not valid for benchmark")
    # Prior launchers record argv; parse both --name=value and --name value.
    argv = record.get("argv", [])
    if argv:
        require(isinstance(argv, list) and all(isinstance(x, str) for x in argv),
                f"{path}: invalid argv")
        flags = {}
        for index, arg in enumerate(argv):
            if arg.startswith("--"):
                name, sep, value = arg[2:].partition("=")
                if not sep:
                    value = argv[index + 1] if index + 1 < len(argv) else "true"
                flags[name] = value
        for key in ("enable_profiling", "memory_report", "dump_full_logits",
                    "dump_logits", "dump_intermediates", "dump_cache"):
            require(flags.get(key, "false").lower() in ("false", "0", ""),
                    f"{path}: diagnostic command flag {key}")
        require(flags.get("trace_position", "-1") == "-1",
                f"{path}: trace command flag")


def parse_session(path, role):
    record = read_json(path)
    no_diagnostics(record, path)
    require(type(record.get("schema_version")) is int
            and record["schema_version"] == 1, f"{path}: unknown schema")
    require(record.get("timings_valid_for_benchmark") is True,
            f"{path}: missing benchmark eligibility")
    expected_runner = "xnnpack_tensor_live_int8" if role == "native" else "litert_lm_cpu"
    require(record.get("runner") == expected_runner, f"{path}: unexpected runner")
    if role == "litert-lm":
        require(record.get("status") == "completed", f"{path}: incomplete session")
        backend = "YNNPACK" if boolean(record, "enable_ynnpack") else "XNNPACK"
        label = f"LiteRT-LM CPU ({backend})"
        require(record.get("context_magic_number_configuration_verified") is True,
                f"{path}: unverified LiteRT-LM context configuration")
    else:
        label = "Native XNNPACK"
    case = record.get("case_id")
    require(isinstance(case, str) and re.fullmatch(r"[A-Za-z0-9_-]+", case),
            f"{path}: invalid case_id")
    warmup = boolean(record, "warmup")
    match = re.fullmatch(re.escape(case) + r"\.(warmup|run)_(\d+)\.json", path.name)
    require(match is not None and (match[1] == "warmup") == warmup,
            f"{path}: filename/session mismatch")
    file_index = int(match[2])
    run_index = record.get("run_index")
    require(type(run_index) is int, f"{path}: invalid run_index")
    if not warmup or role == "native":
        require(run_index == file_index, f"{path}: wrong run_index")
    else:
        require(run_index < 0, f"{path}: wrong LM warmup index")
    vocab = integer(record.get("vocab_size"), "vocab_size", 1)
    histories = []
    for key in ("prompt_token_ids", "forced_decode_token_ids"):
        values = record.get(key)
        require(isinstance(values, list) and values, f"{path}: empty/invalid {key}")
        require(all(type(x) is int and 0 <= x < vocab for x in values),
                f"{path}: invalid token history")
        histories.append(values)
    prompt, forced = histories
    capacity_key = "kv_capacity" if role == "native" else "max_num_tokens"
    capacity = integer(record.get(capacity_key), capacity_key, 1)
    threads = integer(record.get("num_threads"), "num_threads", 1)
    reuse = boolean(record, "reuse_runtimes")
    require(len(prompt) + len(forced) <= capacity, f"{path}: history exceeds capacity")
    passes = record.get("passes")
    require(isinstance(passes, list) and len(passes) == len(forced) + 1,
            f"{path}: incomplete pass sequence")
    elapsed = []
    for index, item in enumerate(passes):
        require(isinstance(item, dict), f"{path}: invalid pass")
        require(item.get("kind") == ("prefill" if index == 0 else "decode")
                and type(item.get("decode_index")) is int
                and item["decode_index"] == index,
                f"{path}: pass ordering mismatch")
        require(item.get("input_ids") == (prompt if index == 0 else [forced[index - 1]])
                and item.get("context_length_after") == len(prompt) + index
                and item.get("logits_position") == len(prompt) + index - 1,
                f"{path}: pass history mismatch")
        require("logits_file" in item and item["logits_file"] is None,
                f"{path}: missing dump state or logits dump capture")
        integer(item.get("argmax_id"), "argmax_id")
        require(item["argmax_id"] < vocab, f"{path}: argmax outside vocabulary")
        value = item.get("elapsed_ms")
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                f"{path}: invalid elapsed_ms")
        elapsed.append(value)
    try:
        total_decode = math.fsum(elapsed[1:])
        prefill_seconds = elapsed[0] / 1000
        decode_rate = len(forced) * 1000 / total_decode
    except OverflowError as error:
        raise CaptureError(f"{path}: timing overflow") from error
    require(all(math.isfinite(x) and x > 0
                for x in (total_decode, prefill_seconds, decode_rate)),
            f"{path}: invalid derived timing")
    settings = {
        "num_threads": threads, "kv_capacity": capacity,
        "reuse_runtimes": reuse, "vocab_size": vocab,
        "prompt_token_ids": prompt, "forced_decode_token_ids": forced,
    }
    native_options = {key: record.get(key) for key in (
        "prefill_rows", "kv_alignment", "fixed_attention_extent",
        "preserve_static_int2", "share_workspace", "static_int2_tensor_count",
        "static_int2_operator_count", "static_int2_compact_bytes", "kv_dtype")}
    if role == "native":
        for field in ("fixed_attention_extent", "preserve_static_int2",
                      "share_workspace", "memory_report"):
            boolean(record, field)
        for field in ("prefill_rows", "kv_alignment"):
            integer(record.get(field), field, 1)
        for field in ("static_int2_tensor_count", "static_int2_operator_count",
                      "static_int2_compact_bytes"):
            integer(record.get(field), field)
        require(record.get("kv_dtype") == "int8", f"{path}: unexpected native KV type")
    return {
        "case_id": case, "runner": label, "warmup": warmup,
        "run_index": file_index, "source": str(path), "settings": settings,
        "native_options": native_options if role == "native" else None,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": total_decode / 1000,
        "decode_tokens_per_second": decode_rate,
    }


def load_capture(directory, role, measured_runs, warmup_runs):
    completion = read_json(directory / "run.json")
    require(completion.get("status") == "completed", f"{directory}: incomplete capture")
    no_diagnostics(completion, directory)
    for name in ("manifest.json", "command.json", "status.json"):
        path = directory / name
        if path.exists():
            metadata = read_json(path)
            no_diagnostics(metadata, path)
            if name == "manifest.json":
                require(metadata == completion, f"{path}: completion/manifest disagree")
            if name == "status.json":
                require(metadata.get("returncode") == 0
                        and metadata.get("pull_returncode", 0) == 0,
                        f"{path}: failed execution or pull")
    for path in directory.iterdir():
        require(not (path.suffix in (".f32", ".i8") or ".profile." in path.name
                     or path.name.startswith("trace.")),
                f"{directory}: diagnostic output {path.name}")
    files = sorted(list(directory.glob("*.run_*.json"))
                   + list(directory.glob("*.warmup_*.json")))
    require(files, f"{directory}: no session files")
    if role == "litert-lm":
        require(completion.get("measured_runs") == measured_runs
                and completion.get("warmup_runs") == warmup_runs,
                f"{directory}: repetition count differs from requested protocol")
        listed = completion.get("runs")
        require(isinstance(listed, list) and all(isinstance(x, str) for x in listed)
                and len(listed) == len(set(listed))
                and set(listed) == {p.name for p in files},
                f"{directory}: completion index does not match session files")
    sessions = [parse_session(path, role) for path in files]
    cases = defaultdict(list)
    for session in sessions:
        cases[session["case_id"]].append(session)
    require(type(completion.get("cases")) is int and completion["cases"] == len(cases),
            f"{directory}: completed case count mismatch")
    for case, records in cases.items():
        for warmup, count in ((False, measured_runs), (True, warmup_runs)):
            indices = [r["run_index"] for r in records if r["warmup"] == warmup]
            require(sorted(indices) == list(range(count)),
                    f"{directory}/{case}: missing/duplicate repetitions")
        first = records[0]
        for record in records[1:]:
            require(all(record[k] == first[k] for k in ("settings", "runner", "native_options")),
                    f"{directory}/{case}: inconsistent session settings/history")
        if role == "litert-lm":
            require(completion.get("num_threads") == first["settings"]["num_threads"]
                    and completion.get("max_num_tokens") == first["settings"]["kv_capacity"]
                    and completion.get("reuse_runtimes") == first["settings"]["reuse_runtimes"]
                    and type(completion.get("enable_ynnpack")) is bool
                    and ("YNNPACK" in first["runner"]) == completion["enable_ynnpack"],
                    f"{directory}: index/session configuration mismatch")
    return cases


def collect(root, role, measured_runs, warmup_runs):
    require(root.is_dir(), f"missing {role} directory: {root}")
    directories = sorted({p.parent for p in root.rglob("*.run_*.json")}
                         | {p.parent for p in root.rglob("*.warmup_*.json")}
                         | {p.parent for p in root.rglob("run.json")})
    require(directories, f"{root}: no captures")
    result = {}
    for directory in directories:
        for case, records in load_capture(directory, role, measured_runs, warmup_runs).items():
            require(case not in result, f"{root}: duplicate case capture: {case}")
            result[case] = records
    return result


def summarize(results, native_dir=None, litert_lm_dir=None, measured_runs=3, warmup_runs=1):
    integer(measured_runs, "measured_runs", 1)
    integer(warmup_runs, "warmup_runs")
    native = collect(Path(native_dir) if native_dir else results / "native",
                     "native", measured_runs, warmup_runs)
    baseline = collect(Path(litert_lm_dir) if litert_lm_dir else results / "litert-lm",
                       "litert-lm", measured_runs, warmup_runs)
    require(native.keys() == baseline.keys(), "native/LiteRT-LM case sets differ")
    rows = []
    for case in sorted(native, key=lambda x: (len(native[x][0]["settings"]["prompt_token_ids"]), x)):
        require(native[case][0]["settings"] == baseline[case][0]["settings"],
                f"{case}: native/LiteRT-LM histories/capacity/threads/reuse differ")
        for records in (native[case], baseline[case]):
            measured = [r for r in records if not r["warmup"]]
            first = measured[0]
            rows.append({
                "case_id": case, "runner": first["runner"],
                "prompt_tokens": len(first["settings"]["prompt_token_ids"]),
                "forced_decode_tokens": len(first["settings"]["forced_decode_token_ids"]),
                "settings": first["settings"], "native_options": first["native_options"],
                "median_prefill_seconds": statistics.median(r["prefill_seconds"] for r in measured),
                "median_decode_tokens_per_second": statistics.median(r["decode_tokens_per_second"] for r in measured),
                "sessions": [{k: r[k] for k in ("run_index", "source", "prefill_seconds",
                                                "decode_seconds", "decode_tokens_per_second")}
                             for r in measured],
            })
    return {"schema_version": 1, "measured_runs": measured_runs,
            "excluded_warmups_per_case": warmup_runs,
            "timing_scope": "Prefill includes first prediction; decode uses forced tokens / sum of decode elapsed times.",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--litert-lm-dir", type=Path)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output", type=Path, help="New JSON output file; console shows medians")
    args = parser.parse_args()
    try:
        summary = summarize(args.results, args.native_dir, args.litert_lm_dir,
                            args.measured_runs, args.warmup_runs)
        if args.output:
            with args.output.open("x") as output:
                json.dump(summary, output, indent=2, allow_nan=False)
                output.write("\n")
        print("Case\tRunner\tPrefill s (median)\tDecode tok/s (median)")
        for row in summary["rows"]:
            print(f"{row['case_id']}\t{row['runner']}\t"
                  f"{row['median_prefill_seconds']:.6f}\t"
                  f"{row['median_decode_tokens_per_second']:.3f}")
    except (CaptureError, OSError, OverflowError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
