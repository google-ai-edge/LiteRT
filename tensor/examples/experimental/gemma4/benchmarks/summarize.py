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

"""Validate a complete campaign and summarize paired warm CPU measurements."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from common import read_cases, require, save_json, sha256
from run_phone import plan_runs


def read_json(path):
    return json.loads(path.read_text())


def capture_name(item):
    return f"{item['case']}-c{item['capacity']}-b{item['block']}-{item['runner']}"


def validate_rows(rows, case, warmups, runs, runner):
    indexed = {}
    for row in rows:
        key = (row["repetition"], row["step"])
        require(all(type(x) is int for x in key), "Invalid repetition or step")
        require(
            row["case"] == case["id"] and key not in indexed,
            "Mixed cases or duplicate measurement",
        )
        require(
            row["context"] == len(case["prompt"]) + row["step"], "Wrong token context"
        )
        require(
            type(row["argmax"]) is int and 0 <= row["argmax"] < 262144, "Invalid argmax"
        )
        require(
            type(row["peak_rss_kib"]) is int and row["peak_rss_kib"] > 0, "Invalid RSS"
        )
        for field in ("token_ms", "prefix_ms", "setup_ms"):
            require(
                isinstance(row[field], (int, float))
                and math.isfinite(row[field])
                and row[field] >= 0,
                "Invalid timing",
            )
        require(row["token_ms"] > 0, "Zero token latency")
        if runner == "litert_lm":
            require(
                row.get("consumed_token_ids_verified") is True,
                "Unverified LM input history",
            )
        indexed[key] = row
    expected = {
        (rep, step)
        for rep in range(-warmups, runs)
        for step in range(len(case["forced"]) + 1)
    }
    require(set(indexed) == expected, "Missing or extra measurements")
    sessions = []
    for rep in range(runs):
        first = indexed[rep, 0]
        decode = [
            indexed[rep, step]["token_ms"] for step in range(1, len(case["forced"]) + 1)
        ]
        ttft = first["prefix_ms"] + first["token_ms"]
        sessions.append(
            {
                "repetition": rep,
                "first_logits_ms": ttft,
                "decode_ms": statistics.mean(decode),
                "request_ms": ttft + sum(decode),
                "argmax": [
                    indexed[rep, step]["argmax"] for step in range(len(decode) + 1)
                ],
            }
        )
    return sessions, max(row["peak_rss_kib"] for row in rows)


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(directory):
    campaign = read_json(directory / "campaign.json")
    require(
        campaign["schema_version"] == 1 and campaign["status"] == "completed",
        "Campaign is incomplete",
    )
    fixture = directory / "workload.tsv"
    require(sha256(fixture) == campaign["fixture_sha256"], "Fixture hash mismatch")
    cases = {case["id"]: case for case in read_cases(fixture)}
    require(
        all(case["forced"] for case in cases.values()),
        "Need forced decode measurements",
    )
    capacities = campaign["capacities"]
    expected, skipped = plan_runs(list(cases.values()), capacities, campaign["blocks"])
    require(
        campaign["plan"] == expected and campaign["skipped"] == skipped,
        "Invalid campaign plan",
    )
    groups = defaultdict(lambda: defaultdict(list))
    for item in expected:
        root = directory / capture_name(item)
        require(
            read_json(root / "capture.json")
            == {**item, "fixture_sha256": campaign["fixture_sha256"]},
            "Capture identity mismatch",
        )
        status = read_json(root / "status.json")
        require(
            status.get("completed") is True and status.get("returncode") == 0,
            "Failed capture",
        )
        if item["runner"] == "litert_lm":
            manifest = read_json(root / "capture/manifest.json")
            fields = {
                "schema_version": 1,
                "status": "completed",
                "runner": "litert_lm",
                "backend": "cpu_xnnpack",
                "speculative_decoding": False,
                "timings_valid_for_benchmark": True,
                "cache_policy": "persistent",
                "threads": campaign["threads"],
                "capacity": item["capacity"],
                "warmups": campaign["warmups"],
                "runs": campaign["runs"],
                "prefill_signature": campaign["lm_prefill"],
            }
            require(
                all(manifest.get(k) == v for k, v in fields.items()),
                "LM settings differ or diagnostic tracing was enabled",
            )
        rows = [
            json.loads(line)
            for line in (root / "capture/timings.jsonl").read_text().splitlines()
            if line
        ]
        sessions, rss = validate_rows(
            rows,
            cases[item["case"]],
            campaign["warmups"],
            campaign["runs"],
            item["runner"],
        )
        telemetry = (
            read_json(root / "telemetry.json")
            if (root / "telemetry.json").exists()
            else []
        )
        thermal = {}
        for field in ("cpu_c", "skin_c", "soc_c"):
            values = [
                sample[field] for sample in telemetry if sample.get(field) is not None
            ]
            thermal[field] = max(values) if values else None
        thermal["cpu_cooling_observed"] = any(
            s.get("cpu_cooling_active", False) for s in telemetry
        )
        thermal["samples"] = len(telemetry)
        groups[item["case"], item["capacity"]][item["runner"]].append(
            {
                "block": item["block"],
                "sessions": sessions,
                "peak_rss_kib": rss,
                "thermal": thermal,
            }
        )
    comparisons = []
    for (case, capacity), runners in groups.items():
        result = {
            "case": case,
            "capacity": capacity,
            "prompt_tokens": len(cases[case]["prompt"]),
            "forced_tokens": len(cases[case]["forced"]),
            "runners": {},
        }
        for kind, captures in runners.items():
            sessions = [
                session for capture in captures for session in capture["sessions"]
            ]
            stats = {
                field: distribution([s[field] for s in sessions])
                for field in ("first_logits_ms", "decode_ms", "request_ms")
            }
            stats["decode_tokens_per_second"] = 1000 / stats["decode_ms"]["median"]
            stats["peak_rss_kib"] = distribution([c["peak_rss_kib"] for c in captures])
            stats["processes"] = len(captures)
            stats["measured_sessions"] = len(sessions)
            stats["captures"] = captures
            result["runners"][kind] = stats
        candidate, baseline = (
            result["runners"]["litert"],
            result["runners"]["litert_lm"],
        )
        result["comparison"] = {
            field: {
                "baseline_over_candidate": baseline[field]["median"]
                / candidate[field]["median"],
                "candidate_reduction_percent": 100
                * (1 - candidate[field]["median"] / baseline[field]["median"]),
            }
            for field in ("first_logits_ms", "decode_ms", "request_ms", "peak_rss_kib")
        }
        matches, total = 0, 0
        for block in range(campaign["blocks"]):
            left = next(c for c in runners["litert"] if c["block"] == block)["sessions"]
            right = next(c for c in runners["litert_lm"] if c["block"] == block)[
                "sessions"
            ]
            for a, b in zip(left, right):
                matches += sum(x == y for x, y in zip(a["argmax"], b["argmax"]))
                total += len(a["argmax"])
        result["argmax_agreement"] = {"matches": matches, "total": total}
        comparisons.append(result)
    # These inputs are produced from allowlisted fields by run_phone.py. Private
    # process.log files and raw model manifests are deliberately never read.
    return {
        "schema_version": 1,
        "campaign": campaign,
        "device": read_json(directory / "device.json"),
        "comparisons": comparisons,
    }


def markdown(summary):
    lines = [
        "# Gemma4 CPU measurements",
        "",
        "Medians across measured sessions; ranges and per-process thermal/RSS data are in the JSON.",
        "",
        "| Case / capacity | Runner | First logits (ms) | Decode (ms/token) | Decode (token/s) | Request (ms) | Peak RSS (MiB, max process) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    agreements = []
    for comparison in summary["comparisons"]:
        for kind, stats in comparison["runners"].items():
            lines.append(
                f"| {comparison['case']} / {comparison['capacity']} | {kind} | "
                f"{stats['first_logits_ms']['median']:.3f} | {stats['decode_ms']['median']:.3f} | "
                f"{stats['decode_tokens_per_second']:.2f} | {stats['request_ms']['median']:.3f} | "
                f"{stats['peak_rss_kib']['max'] / 1024:.1f} |"
            )
        agreement = comparison["argmax_agreement"]
        agreements += [
            "",
            f"{comparison['case']} / {comparison['capacity']}: argmax agreement "
            f"{agreement['matches']}/{agreement['total']} (diagnostic, not an accuracy metric).",
            "",
        ]
    lines += agreements
    lines += [
        "Runtime setup and reset are excluded from latency; peak RSS includes setup and warmup.",
        "The fixed inputs measure performance and do not establish model quality.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    args = parser.parse_args()
    summary = summarize(args.input)
    save_json(args.output, summary)
    report = markdown(summary)
    if args.markdown:
        args.markdown.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
