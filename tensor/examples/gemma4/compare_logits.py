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

"""Compare matching autoregressive steps from two Gemma4 logits dumps.

Requires NumPy. Both prefixes must have .json manifests and little-endian FP32
logits files. Later steps are comparable only while their preceding generated
IDs agree. A reference may contain extra steps after the C++ run's stop token;
only the explicitly requested number of steps is compared.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-rmse", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.min_cosine <= 1 or args.max_rmse < 0:
        parser.error("Invalid step count or comparison threshold")
    prefixes = [args.expected, args.actual]
    reports = [json.loads(Path(str(prefix) + ".json").read_text()) for prefix in prefixes]
    if reports[0]["input_token_ids"] != reports[1]["input_token_ids"]:
        parser.error("Prompt token IDs differ")
    for report in reports:
        if min(len(report["logits_suffixes"]), len(report["generated_token_ids"])) < args.steps:
            parser.error("A report has fewer than the requested steps")
    results = []
    passed = True
    for step in range(args.steps):
        if reports[0]["generated_token_ids"][:step] != reports[1]["generated_token_ids"][:step]:
            parser.error(f"Step {step} has different input history; later logits are not comparable")
        vectors = []
        for prefix, report in zip(prefixes, reports):
            shape = report.get("logits_shape", [report.get("vocab_size")])
            if (not isinstance(shape, list) or len(shape) != 1
                    or type(shape[0]) is not int or shape[0] <= 0):
                parser.error("Invalid vocabulary dimensions in manifest")
            path = Path(str(prefix) + report["logits_suffixes"][step])
            if path.stat().st_size != shape[0] * 4:
                parser.error(f"Logit file length disagrees with its manifest at step {step}")
            vectors.append(np.fromfile(path, dtype="<f4").astype(np.float64))
        expected, actual = vectors
        if expected.shape != actual.shape or expected.size == 0:
            parser.error(f"Logit shapes differ or are empty at step {step}")
        if not all(np.isfinite(value).all() for value in vectors):
            parser.error(f"Nonfinite logits at step {step}")
        for value, report in zip(vectors, reports):
            if int(value.argmax()) != report["generated_token_ids"][step]:
                parser.error(f"Manifest argmax disagrees with logits at step {step}")
        difference = expected - actual
        denominator = np.linalg.norm(expected) * np.linalg.norm(actual)
        cosine = float(np.dot(expected, actual) / denominator) if denominator else float(np.array_equal(expected, actual))
        rmse = float(np.sqrt(np.mean(difference ** 2)))
        token_match = int(expected.argmax()) == int(actual.argmax())
        ok = token_match and cosine >= args.min_cosine and rmse <= args.max_rmse
        passed = passed and ok
        results.append({"step": step, "elements": expected.size,
                        "expected_token": int(expected.argmax()), "actual_token": int(actual.argmax()),
                        "token_match": token_match, "cosine": cosine, "rmse": rmse,
                        "max_abs_error": float(np.max(np.abs(difference))), "passed": ok})
    result = {"expected": str(args.expected), "actual": str(args.actual),
              "steps": results, "min_cosine": args.min_cosine,
              "max_rmse": args.max_rmse, "passed": passed}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
