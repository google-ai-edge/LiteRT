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


"""Deploy and compare Tensor API LiteRT and official LiteRT-LM on Android."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from common import (
    bundle_identity,
    bundle_payloads,
    read_cases,
    require,
    save_json,
    sha256,
    source_identity,
    write_case,
)


def plan_runs(cases, capacities, blocks):
    require(blocks > 0 and capacities and min(capacities) > 0, "Invalid run grid")
    require(len(set(capacities)) == len(capacities), "Duplicate capacities")
    plan, skipped = [], []
    for case in cases:
        for capacity in capacities:
            if len(case["prompt"]) + len(case["forced"]) > capacity:
                skipped.append(
                    {
                        "case": case["id"],
                        "capacity": capacity,
                        "reason": "prompt plus continuation exceeds capacity",
                    }
                )
                continue
            for block in range(blocks):
                order = (
                    ("litert", "litert_lm")
                    if block % 2 == 0
                    else ("litert_lm", "litert")
                )
                for kind in order:
                    plan.append(
                        {
                            "case": case["id"],
                            "capacity": capacity,
                            "block": block,
                            "runner": kind,
                        }
                    )
    require(plan, "No valid configurations")
    return plan, skipped


def parse_thermal(raw):
    # Store aggregates, not raw service output or device identifiers.
    live = raw.split("Current temperatures from HAL:", 1)
    if len(live) != 2:
        return {"available": False}
    live = live[1].split("Current cooling devices from HAL:", 1)[0]
    readings = [
        (float(v), int(t))
        for v, t in re.findall(r"Temperature\{mValue=([-\d.]+), mType=([-\d]+),", live)
    ]
    result = {
        "available": any(t == 0 for _, t in readings)
        and any(t == 3 for _, t in readings)
    }
    for kind, field in ((0, "cpu_c"), (3, "skin_c"), (13, "soc_c")):
        values = [v for v, t in readings if t == kind and math.isfinite(v)]
        result[field] = max(values) if values else None
    cooling = raw.split("Current cooling devices from HAL:", 1)
    result["cpu_cooling_active"] = any(
        int(v) > 0
        for v in re.findall(
            r"CoolingDevice\{mValue=(\d+), mType=2,",
            cooling[1] if len(cooling) == 2 else "",
        )
    )
    return result


class Device:
    def __init__(self, serial):
        # Serial is used only for transport; never write it to a report.
        self.adb = ["adb"] + (["-s", serial] if serial else [])

    def shell(self, argv, check=True, timeout=60):
        return subprocess.run(
            self.adb + ["shell", shlex.join(map(str, argv))],
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        )

    def push(self, source, destination):
        subprocess.run(
            self.adb + ["push", str(source), destination],
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=1800,
        )

    def thermal(self):
        result = self.shell(["dumpsys", "thermalservice"], check=False)
        return (
            parse_thermal(result.stdout)
            if result.returncode == 0
            else {"available": False}
        )


def runner_command(
    kind, artifacts, output, case_path, cache, capacity, args, prime=False
):
    command = [
        "taskset",
        args.mask,
        artifacts + ("/runner.android" if kind == "litert" else "/litertlm.android"),
        "--cases_file",
        case_path,
        "--output_dir",
        output,
        "--num_threads",
        str(args.threads),
        "--warmup_runs",
        "0" if prime else str(args.warmups),
        "--measured_runs",
        "1" if prime else str(args.runs),
    ]
    if kind == "litert":
        command += [
            "--bundle_dir",
            artifacts + "/weights",
            "--model_path",
            artifacts + "/model.tflite",
            "--cache_capacity",
            str(capacity),
            "--weight_cache_path",
            cache + "/litert.xnnpack_cache",
        ]
        for enabled, flag in (
            (args.compact_int2, "experimental_compact_int2"),
            (args.dynamic_prefill, "experimental_dynamic_prefill"),
            (args.strict_shapes, "experimental_explicit_dynamic_shapes"),
        ):
            if enabled:
                command += ["--" + flag]
    else:
        command += [
            "--model_path",
            artifacts + "/model.litertlm",
            "--max_num_tokens",
            str(capacity),
            "--cache_dir",
            cache + "/lm",
            "--prefill_signature",
            args.lm_prefill,
        ]
    return command


def wait_for_gate(device, args, directory, last_end):
    samples, consecutive = [], 0
    started = time.monotonic()
    if args.thermal_mode != "gate":
        return
    while time.monotonic() - started < args.gate_timeout:
        sample = device.thermal()
        require(
            sample.get("available"),
            "Thermal gate needs CPU/skin readings; choose record mode explicitly if unavailable",
        )
        eligible = (
            sample["cpu_c"] is not None
            and sample["skin_c"] is not None
            and sample["cpu_c"] <= args.cpu_max
            and sample["skin_c"] <= args.skin_max
            and not sample["cpu_cooling_active"]
        )
        samples.append(
            {**sample, "elapsed_s": time.monotonic() - started, "eligible": eligible}
        )
        save_json(directory / "start_gate.json", samples)
        consecutive = consecutive + 1 if eligible else 0
        if consecutive >= 2 and time.monotonic() - last_end >= args.cooldown:
            return
        time.sleep(args.sample_seconds)
    raise RuntimeError("Thermal gate timed out")


def invoke(device, command, directory, args):
    samples = []
    start = time.monotonic()
    with (directory / "process.log").open("w") as log:
        process = subprocess.Popen(
            device.adb + ["shell", shlex.join(command)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while True:
                if args.thermal_mode != "off":
                    samples.append(
                        {**device.thermal(), "elapsed_s": time.monotonic() - start}
                    )
                    save_json(directory / "telemetry.json", samples)
                remaining = args.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("Benchmark process timed out")
                try:
                    code = process.wait(timeout=min(args.sample_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            save_json(
                directory / "status.json",
                {"completed": False, "wall_s": time.monotonic() - start},
            )
            # Closing adb does not guarantee termination of the remote process.
            # Stop the campaign rather than running another overlapping capture.
            raise
    save_json(
        directory / "status.json",
        {
            "completed": code == 0,
            "returncode": code,
            "wall_s": time.monotonic() - start,
        },
    )
    require(code == 0, "Benchmark failed; inspect the private process.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="ADB transport selector; omitted from reports")
    parser.add_argument("--remote", required=True, help="Writable device directory")
    parser.add_argument(
        "--output", type=Path, required=True, help="New local capture directory"
    )
    parser.add_argument("--litert-src", type=Path, required=True)
    parser.add_argument("--lm-src", type=Path, required=True)
    parser.add_argument("--runner-bin", type=Path, required=True)
    parser.add_argument("--lm-bin", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, required=True, help="Exported Tensor API .tflite"
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        required=True,
        help="Original .litertlm used to prepare weights",
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).parent / "fixtures/performance_1024_64.tsv",
    )
    parser.add_argument("--capacities", type=int, nargs="+", default=[8448])
    parser.add_argument(
        "--mask", required=True, help="Hexadecimal taskset CPU affinity mask"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--lm-prefill", choices=["auto", "prefill_128"], default="auto")
    parser.add_argument(
        "--prefill-rows",
        type=int,
        default=128,
        help="Record the exported Tensor API row limit",
    )
    parser.add_argument("--compact-int2", action="store_true")
    parser.add_argument("--dynamic-prefill", action="store_true")
    parser.add_argument("--strict-shapes", action="store_true")
    parser.add_argument(
        "--thermal-mode", choices=["gate", "record", "off"], default="record"
    )
    parser.add_argument("--cpu-max", type=float, default=45)
    parser.add_argument("--skin-max", type=float, default=34)
    parser.add_argument("--gate-timeout", type=float, default=600)
    parser.add_argument("--cooldown", type=float, default=15)
    parser.add_argument("--sample-seconds", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument(
        "--dry-run", action="store_true", help="Write a plan without invoking adb"
    )
    args = parser.parse_args()
    require(
        re.fullmatch(r"[0-9a-fA-F]+", args.mask) and int(args.mask, 16) > 0,
        "Invalid affinity mask",
    )
    require(
        args.threads > 0 and args.warmups >= 1 and args.runs > 0,
        "Use positive threads/runs and at least one warmup",
    )
    require(
        args.prefill_rows in [32, 64, 128, 256, 512, 1024], "Invalid exported row limit"
    )
    require(
        all(
            math.isfinite(x) and x > 0
            for x in [args.timeout, args.gate_timeout, args.sample_seconds]
        ),
        "Invalid timeout",
    )
    require(
        math.isfinite(args.cooldown)
        and args.cooldown >= 0
        and math.isfinite(args.cpu_max)
        and math.isfinite(args.skin_max),
        "Invalid thermal settings",
    )
    require(
        args.remote.startswith("/")
        and ".." not in args.remote.split("/")
        and "\n" not in args.remote,
        "Use an absolute device directory without parent traversal",
    )
    cases = read_cases(args.cases)
    require(
        all(c["forced"] for c in cases), "Performance cases need forced continuations"
    )
    plan, skipped = plan_runs(cases, args.capacities, args.blocks)
    identity = bundle_identity(args.bundle_dir)
    files = {
        "runner.android": args.runner_bin,
        "litertlm.android": args.lm_bin,
        "model.tflite": args.model,
        "model.litertlm": args.source_model,
        "weights/manifest.json": args.bundle_dir / "manifest.json",
    }
    hashes = {name: sha256(path) for name, path in files.items()}
    hashes.update(
        {
            "weights/" + name: digest
            for name, digest in bundle_payloads(args.bundle_dir).items()
        }
    )
    require(
        hashes["model.litertlm"] == identity["source_sha256"],
        "LM source model differs from the prepared coefficient bundle",
    )
    require(
        (identity["storage"] == "compact_int2_v1") == args.compact_int2,
        "Compact storage flag differs from bundle metadata",
    )
    sources = {
        "litert": source_identity(args.litert_src),
        "litert_lm": source_identity(args.lm_src),
    }
    # The adapter is an intentionally added untracked package. Record its exact
    # source separately without copying any Git remote or personal metadata.
    adapter = args.lm_src / "gemma4_benchmark" / "adapter.cc"
    require(
        adapter.is_file(), "Expected the documented gemma4_benchmark adapter package"
    )
    sources["litert_lm"]["adapter_sha256"] = sha256(adapter)
    expected_adapter = Path(__file__).parent / "litert_lm_harness/adapter.cc"
    require(
        sha256(adapter) == sha256(expected_adapter),
        "LM adapter source differs from this benchmark package",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    # Local capture paths and transport serials deliberately stay out of JSON.
    run_id = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]
    manifest = {
        "schema_version": 1,
        "status": "planned",
        "sources": sources,
        "artifacts": hashes,
        "bundle": identity,
        "fixture_sha256": sha256(args.cases),
        "threads": args.threads,
        "mask": args.mask,
        "warmups": args.warmups,
        "runs": args.runs,
        "blocks": args.blocks,
        "capacities": args.capacities,
        "lm_prefill": args.lm_prefill,
        "prefill_rows": args.prefill_rows,
        "dynamic_prefill": args.dynamic_prefill,
        "strict_shapes": args.strict_shapes,
        "compact_int2": args.compact_int2,
        "cache_policy": "persistent; untimed prime per configuration; warm measured sessions",
        "thermal": {
            "mode": args.thermal_mode,
            "cpu_max": args.cpu_max,
            "skin_max": args.skin_max,
            "cooldown_s": args.cooldown,
            "sample_seconds": args.sample_seconds,
        },
        "plan": plan,
        "skipped": skipped,
    }
    save_json(args.output / "campaign.json", manifest)
    shutil.copyfile(args.cases, args.output / "workload.tsv")
    if args.dry_run:
        print(
            f"Planned {len(plan)} processes; {len(skipped)} invalid configurations skipped. No adb calls made."
        )
        return
    device = Device(args.serial)
    save_json(
        args.output / "device.json",
        {
            "model": device.shell(["getprop", "ro.product.model"]).stdout.strip(),
            "soc": device.shell(["getprop", "ro.soc.model"]).stdout.strip(),
            "android_release": device.shell(
                ["getprop", "ro.build.version.release"]
            ).stdout.strip(),
        },
    )
    tag = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
    artifacts = args.remote.rstrip("/") + "/artifacts-" + tag
    device.shell(["mkdir", "-p", artifacts])
    if device.shell(["test", "-f", artifacts + "/ready"], check=False).returncode:
        for name, path in files.items():
            if not name.startswith("weights/"):
                device.push(path, artifacts + "/" + name)
        device.shell(["mkdir", "-p", artifacts + "/weights"])
        device.push(str(args.bundle_dir) + "/.", artifacts + "/weights")
        device.shell(
            [
                "chmod",
                "755",
                artifacts + "/runner.android",
                artifacts + "/litertlm.android",
            ]
        )
    names = list(hashes)
    for offset in range(0, len(names), 32):
        batch = names[offset : offset + 32]
        lines = device.shell(
            ["sha256sum"] + [artifacts + "/" + name for name in batch]
        ).stdout.splitlines()
        require(len(lines) == len(batch), "Incomplete device artifact hash check")
        for name, line in zip(batch, lines):
            require(
                line.split()[0] == hashes[name],
                "Device artifact hash mismatch: " + name,
            )
    device.shell(["touch", artifacts + "/ready"])
    remote_run = args.remote.rstrip("/") + "/run-" + run_id
    device.shell(["mkdir", remote_run])
    for case in cases:
        path = args.output / (case["id"] + ".tsv")
        write_case(path, case)
        device.push(path, remote_run + "/" + path.name)
    manifest["status"] = "running"
    save_json(args.output / "campaign.json", manifest)
    primed, last_end = set(), 0
    for item in plan:
        name = f"{item['case']}-c{item['capacity']}-b{item['block']}-{item['runner']}"
        directory = args.output / name
        directory.mkdir()
        save_json(
            directory / "capture.json",
            {**item, "fixture_sha256": manifest["fixture_sha256"]},
        )
        kind, capacity = item["runner"], item["capacity"]
        cache = artifacts + f"/cache-c{capacity}-{args.lm_prefill}"
        device.shell(["mkdir", "-p", cache + "/lm"])
        case_path = remote_run + "/" + item["case"] + ".tsv"
        key = (item["case"], capacity, kind)
        if key not in primed:
            prime = directory / "prime"
            prime.mkdir()
            command = runner_command(
                kind,
                artifacts,
                remote_run + "/prime-" + name,
                case_path,
                cache,
                capacity,
                args,
                prime=True,
            )
            invoke(device, command, prime, args)
            primed.add(key)
            last_end = time.monotonic()
        wait_for_gate(device, args, directory, last_end)
        command = runner_command(
            kind, artifacts, remote_run + "/" + name, case_path, cache, capacity, args
        )
        invoke(device, command, directory, args)
        last_end = time.monotonic()
        # Pull into a fixed relative location. Do not store adb's serial or argv.
        subprocess.run(
            device.adb
            + ["pull", remote_run + "/" + name + "/.", str(directory / "capture")],
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=1800,
        )
    manifest["status"] = "completed"
    save_json(args.output / "campaign.json", manifest)
    print("Completed. Run summarize.py on this campaign directory.")


if __name__ == "__main__":
    main()
