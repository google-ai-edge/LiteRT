# Copyright 2026 The LiteRT Authors. All Rights Reserved.
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
"""Compare production convolution lowering with integer and LiteRT oracles.

The fixture executable uses the production analyzer/emitter and exports a
separate accumulator output for testing through LiteRT direct dispatch.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import hashlib
from emulation_environment import layer_environment

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(argv, directory, env, name):
    with (directory / (name + ".log")).open("w") as log:
        subprocess.run([str(a) for a in argv], cwd=directory, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=300)


def integer_oracle(directory, metadata, inp):
    weights = np.fromfile(directory / ("weights.bin" if (directory / "weights.bin").exists() else "constant_0.bin"), dtype=np.int8).reshape(
        metadata["weight_shape"]).astype(np.int64)
    if metadata.get("depthwise") and not (directory / "weights.bin").exists():
        c, h, w, _ = metadata["weight_shape"]
        weights = weights.reshape(h, w, c).transpose(2, 0, 1)[..., None]
    bias = np.fromfile(directory / ("bias.bin" if (directory / "bias.bin").exists() else "constant_1.bin"), dtype="<i4").astype(np.int64)
    centered = inp.astype(np.int64) - metadata["input_zero_point"]
    top, bottom, left, right = metadata["padding"]
    centered = np.pad(centered, ((0, 0), (top, bottom), (left, right), (0, 0)))
    batch, oh, ow, channels = metadata["output_shape"]
    _, kh, kw, _ = weights.shape
    sy, sx = metadata["stride"]
    dy, dx = metadata["dilation"]
    # Wide-integer oracle works directly from LiteRT padding/stride semantics;
    # it does not reuse the compiler's slicing or graph construction.
    acc = np.broadcast_to(bias, (batch, oh, ow, channels)).copy()
    for ky in range(kh):
        for kx in range(kw):
            patch = centered[:, ky*dy:ky*dy+oh*sy:sy,
                             kx*dx:kx*dx+ow*sx:sx, :]
            if metadata.get("depthwise"):
                acc += patch * weights[:, ky, kx, 0]
            else:
                groups = inp.shape[-1] // weights.shape[-1]
                ic, oc = weights.shape[-1], channels // groups
                for group in range(groups):
                    acc[...,group*oc:(group+1)*oc] += np.matmul(
                        patch[...,group*ic:(group+1)*ic],
                        weights[group*oc:(group+1)*oc, ky, kx, :].T)
    mult = np.array(metadata["multipliers"], dtype=np.int64)
    shift = np.array(metadata["shifts"], dtype=np.int64)
    result = (acc * mult + np.left_shift(np.int64(1), shift - 1)) // np.left_shift(np.int64(1), shift)
    if metadata.get("staged_rounding_channels") or metadata.get("staged_rounding"):
        first = np.minimum(shift, 31)
        tail = shift - first
        value = (acc * mult + (np.int64(1) << (first - 1))) // (np.int64(1) << first)
        # Rounding division, ties away from zero, independent of graph SELECT.
        magnitude = (np.abs(value) + np.where(tail > 0, np.int64(1) << np.maximum(tail - 1, 0), 0)) // (np.int64(1) << tail)
        staged_result = np.where(value < 0, -magnitude, magnitude)
        result = np.where(np.array(metadata.get("staged_rounding_channels", [1]), dtype=bool), staged_result, result)
    result += metadata["output_zero_point"]
    result = np.clip(result, metadata["activation_min"], metadata["activation_max"])
    return acc.astype(np.int32), result.astype(np.int8)


def run(args, directory):
    env = layer_environment(args.icd.resolve())
    invoke([args.fixture.resolve(), args.model.resolve(), directory, args.op_index,
            "--vulkan-emulation"], directory, env, "fixture")
    metadata = json.loads((directory / "graph.json").read_text())
    invoke([args.spirv_val, directory / "graph.spv"], directory, env, "spirv_val")
    cases = []
    for case_id in range(4):
        inp = np.fromfile(directory / f"input_{case_id}.bin", dtype=np.int8).reshape(metadata["input_shape"])
        accumulator, expected = integer_oracle(directory, metadata, inp)
        actual = np.fromfile(directory / f"vulkan_{case_id}.bin", dtype=np.int8).reshape(expected.shape)
        actual_acc = np.fromfile(directory / f"accumulator_{case_id}.bin", dtype="<i4").reshape(accumulator.shape)
        np.testing.assert_array_equal(actual_acc, accumulator)
        np.testing.assert_array_equal(actual, expected)
        reference = np.fromfile(directory / f"reference_{case_id}.bin", dtype=np.int8).reshape(expected.shape)
        error = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
        if error.max() > 1: raise AssertionError(f"CPU difference exceeds one code: {error.max()}")
        cases.append(dict(case=case_id, input_sha256=sha256(directory / f"input_{case_id}.bin"),
                          elements=actual.size, max_cpu_code_error=int(error.max()),
                          mean_cpu_code_error=float(error.mean()), exact_accumulator=True,
                          exact_tosa_rescale=True))
    return dict(model_sha256=sha256(args.model), op_index=args.op_index,
                spirv_sha256=sha256(directory / "graph.spv"), reference=metadata["reference"],
                single_rounding=metadata["single_rounding"], cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--op-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spirv-val", default="spirv-val")
    parser.add_argument("--icd", type=Path, default=Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="conv_", dir=args.output_dir.resolve()))
    print(f"Evidence: {directory}", flush=True)
    report = {"status": "failed"}
    try:
        report.update(run(args, directory))
        report["status"] = "passed"
    finally:
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
