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

"""Generate shared prompt prefixes and fixed continuations for CPU benchmarks."""

import argparse
from pathlib import Path

from common import read_cases, require, save_json, sha256


def generate(seed, lengths, continuation, output):
    require(
        lengths and len(set(lengths)) == len(lengths) and min(lengths) >= 1,
        "Prompt lengths must be distinct positive integers",
    )
    require(continuation >= 1, "Need at least one forced continuation")
    source = read_cases(seed)
    require(len(source) == 1, "Seed must have one case")
    source = source[0]
    body = source["prompt"][1:] + source["forced"]
    require(body, "Seed must contain tokens after BOS")
    # All prompts are prefixes of one stream. All cases use the same forced
    # continuation, making the context-length sweep independent of sampling.
    forced = (
        [source["forced"][i % len(source["forced"])] for i in range(continuation)]
        if source["forced"]
        else [body[i % len(body)] for i in range(continuation)]
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for length in lengths:
        prompt = [source["prompt"][0]] + [
            body[i % len(body)] for i in range(length - 1)
        ]
        rows.append(
            f"prompt_{length}\t"
            + ",".join(map(str, prompt))
            + "\t"
            + ",".join(map(str, forced))
        )
    target = output / "cases.tsv"
    target.write_text("\n".join(rows) + "\n")
    save_json(
        output / "fixture.json",
        {
            "schema_version": 1,
            "seed_sha256": sha256(seed),
            "cases_sha256": sha256(target),
            "prompt_lengths": lengths,
            "forced_tokens": continuation,
            "synthetic_workload": True,
        },
    )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path(__file__).parent / "fixtures/performance_1024_64.tsv",
    )
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", default=[1, 128, 1024, 4096]
    )
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    args = parser.parse_args()
    print(generate(args.seed, args.prompt_lengths, args.decode_tokens, args.output))


if __name__ == "__main__":
    main()
