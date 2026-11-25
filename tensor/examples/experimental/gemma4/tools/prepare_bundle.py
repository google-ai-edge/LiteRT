#!/usr/bin/env python3
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract and validate the published Gemma4 E2B/E4B bundle for the LiteRT example."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--schema-dir",
        type=Path,
        required=True,
        help="Parent of generated tflite/Model.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; contains audit/ and weights/",
    )
    parser.add_argument("--model-variant", choices=("e2b", "e4b"), default="e2b")
    parser.add_argument("--compact-int2", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run Python without -O: export assertions validate the source")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    schema = args.schema_dir.resolve()
    tools = Path(__file__).resolve().parent
    audit, weights = root / "audit", root / "weights"

    def run(script, *flags):
        subprocess.run(
            [sys.executable, str(tools / script), *map(str, flags)], check=True
        )

    inventory = audit / "published-inventory.json"
    trace = audit / "kv-graph-trace.json"
    kv = audit / "kv-placement-summary.json"
    run(
        "inspect_bundle.py",
        "--model",
        args.model.resolve(),
        "--schema-dir",
        schema,
        "--output-dir",
        audit,
    )
    run(
        "trace_kv.py",
        "--inventory",
        inventory,
        "--schema-dir",
        schema,
        "--output",
        trace,
    )
    run(
        "summarize_kv.py",
        "--trace",
        trace,
        "--output",
        kv,
        "--model-variant",
        args.model_variant,
    )
    common = [
        "--schema-dir",
        schema,
        "--inventory",
        inventory,
        "--trace",
        trace,
        "--kv",
        kv,
    ]
    run(
        "export_bundle.py",
        *common,
        "--output-dir",
        weights,
        "--model-variant",
        args.model_variant,
        *(["--compact-int2"] if args.compact_int2 else []),
    )
    run(
        "validate_export.py",
        *common,
        "--bundle-dir",
        weights,
        "--output",
        audit / "export-validation.json",
    )
    print(weights)


if __name__ == "__main__":
    main()
