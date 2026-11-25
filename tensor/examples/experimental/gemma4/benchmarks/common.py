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

"""Shared fixture, hashing and capture utilities; only the standard library."""

import hashlib
import json
import re
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cases(path):
    cases = []
    seen = set()
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        require(len(fields) == 3, "Expected three TSV columns")
        name, prompt, forced = fields
        require(
            re.fullmatch(r"[A-Za-z0-9_-]+", name) and name not in seen,
            "Invalid or duplicate case ID",
        )
        seen.add(name)
        prompt = [int(x) for x in prompt.split(",")]
        forced = [] if forced == "-" else [int(x) for x in forced.split(",")]
        require(
            prompt and all(0 <= x < 262144 for x in prompt + forced),
            "Expected Gemma4 E2B token IDs",
        )
        cases.append({"id": name, "prompt": prompt, "forced": forced})
    require(cases, "No cases")
    return cases


def write_case(path, case):
    path.write_text(
        case["id"]
        + "\t"
        + ",".join(map(str, case["prompt"]))
        + "\t"
        + (",".join(map(str, case["forced"])) or "-")
        + "\n"
    )


def source_identity(path):
    """Record code identity without remotes, authors, host paths or usernames."""

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def bundle_identity(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Do not copy the manifest: its source_bundle.path can be a private host path.
    source = manifest["source_bundle"]
    return {
        "manifest_sha256": sha256(manifest_path),
        "source_sha256": source["sha256"],
        "source_bytes": source["bytes"],
        "schema_version": manifest["schema_version"],
        "storage": manifest.get("experimental_storage", "default"),
    }


def bundle_payloads(directory):
    """Validate exported payloads and return their relative paths and digests."""
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["status"] == "complete", "Incomplete coefficient bundle")
    payloads = {}
    for record in manifest["tensors"] + manifest["constants"]:
        payloads[record["file"]] = record["sha256"]
        quantization = record.get("quantization", {})
        if "scales_file" in quantization:
            payloads[quantization["scales_file"]] = quantization["scales_sha256"]
    require(payloads, "No coefficient payloads")
    for name, digest in payloads.items():
        require(
            re.fullmatch(r"[A-Za-z0-9_./-]+", name)
            and not name.startswith("/")
            and ".." not in name.split("/"),
            "Invalid bundle-relative file",
        )
        require(sha256(directory / name) == digest, "Coefficient hash mismatch")
    return payloads
