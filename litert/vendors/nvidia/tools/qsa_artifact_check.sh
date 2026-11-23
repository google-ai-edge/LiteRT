#!/usr/bin/env bash
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# qsa_artifact_check.sh RUNTIME_LIBS OUT_DIR: writes the QSA artifact
# (qsa_artifact.cc), runs it on LiteRT's CPU and through the NVIDIA compiler
# plugin and dispatch library in RUNTIME_LIBS with FP32 and with BF16
# activations, and compares each NVIDIA run with the CPU's. The FP32 run must
# match (exit status); the BF16 run is reported (its rounding can move a
# selection between two blocks of nearly equal scores).
#
# QSA_ARTIFACT names the tool (default: this tree's bazel-bin), and
# LD_LIBRARY_PATH must reach the TensorRT-RTX and CUDA libraries. The
# partition policy is the one the Qwen 3.8 runs use (whole-signature
# islands; the default policy leaves composites to the CPU).
set -euo pipefail
libs=$(realpath "$1")
out=$(realpath -m "$2")
tool=${QSA_ARTIFACT:-$(dirname "$0")/../../../../bazel-bin/litert/vendors/nvidia/tools/qsa_artifact}
mkdir -p "$out/cpu" "$out/npu_fp32" "$out/npu_bf16"
"$tool" --write_model="$out/qsa_block.tflite"
"$tool" --model="$out/qsa_block.tflite" --accelerator=cpu --out_dir="$out/cpu"
for activations in fp32 bf16; do
  (
    for variable in ${!LITERT_NVIDIA_@}; do unset "$variable"; done
    export LD_LIBRARY_PATH=$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LITERT_NVIDIA_TENSORRT_PARTITION_POLICY=qwen3_5
    [[ $activations == fp32 ]] || export LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=bf16
    "$tool" --model="$out/qsa_block.tflite" --accelerator=npu \
      --runtime_libs="$libs" --out_dir="$out/npu_$activations"
  )
done
status=0
echo "== FP32 activations"
"$tool" --compare --cpu_dir="$out/cpu" --npu_dir="$out/npu_fp32" --tolerance=1e-3 || status=1
echo "== BF16 activations (reported)"
"$tool" --compare --cpu_dir="$out/cpu" --npu_dir="$out/npu_bf16" --tolerance=2e-2 || true
exit $status
