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

# qwen4_artifact_check.sh RUNTIME_LIBS OUT_DIR: writes the reduced whole Qwen 4
# model (qwen4_artifact.cc) with each Gated DeltaNet output gate (sigmoid,
# Qwen 4's; silu, Qwen 3.8's), runs it on LiteRT's CPU and through the NVIDIA
# compiler plugin and dispatch library in RUNTIME_LIBS with FP32 and BF16
# activations, and compares each NVIDIA run with the CPU's: FP32 to 1e-3, BF16
# to 3e-2 (both must match; exit status; a QSA selection moved by BF16
# rounding between near-tied blocks is reported, not failed). Then, with the
# sigmoid gate and BF16 activations: INT4 block weights (as real exports
# carry them) against the CPU run of the same weights' values in FP32, and
# the hyper-connection mix in its other spelling (--hc_spelling=sum).
#
# QWEN4_ARTIFACT names the tool (default: this tree's bazel-bin), and
# LD_LIBRARY_PATH must reach the TensorRT-RTX and CUDA libraries.
set -euo pipefail
libs=$(realpath "$1")
out=$(realpath -m "$2")
tool=${QWEN4_ARTIFACT:-$(dirname "$0")/../../../../bazel-bin/litert/vendors/nvidia/tools/qwen4_artifact}

run_npu() {
  local model=$1 dir=$2 activations=$3
  mkdir -p "$dir"
  (
    for variable in ${!LITERT_NVIDIA_@}; do unset "$variable"; done
    export LD_LIBRARY_PATH=$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=$activations
    "$tool" --model="$model" --accelerator=npu --runtime_libs="$libs" \
      --out_dir="$dir"
  )
}

status=0
for gate in sigmoid silu; do
  dir=$out/$gate
  mkdir -p "$dir/cpu"
  "$tool" --gdn_gate=$gate --write_model="$dir/model.tflite"
  "$tool" --model="$dir/model.tflite" --accelerator=cpu --out_dir="$dir/cpu"
  run_npu "$dir/model.tflite" "$dir/npu_fp32" 0
  run_npu "$dir/model.tflite" "$dir/npu_bf16" bf16
  echo "== $gate gate, FP32 activations"
  "$tool" --compare --cpu_dir="$dir/cpu" --npu_dir="$dir/npu_fp32" --tolerance=1e-3 || status=1
  echo "== $gate gate, BF16 activations"
  "$tool" --compare --cpu_dir="$dir/cpu" --npu_dir="$dir/npu_bf16" --tolerance=3e-2 \
    --strict_selection=false || status=1
done
# One more model of the sigmoid gate: `name`, the flags that write the NVIDIA
# run's model, and the flags that write its CPU reference.
variant() {
  local name=$1 npu_flags=$2 cpu_flags=$3
  local dir=$out/$name
  mkdir -p "$dir/cpu"
  "$tool" $npu_flags --write_model="$dir/model.tflite"
  "$tool" $cpu_flags --write_model="$dir/reference.tflite"
  "$tool" --model="$dir/reference.tflite" --accelerator=cpu --out_dir="$dir/cpu"
  run_npu "$dir/model.tflite" "$dir/npu_bf16" bf16
  echo "== $name, BF16 activations"
  "$tool" --compare --cpu_dir="$dir/cpu" --npu_dir="$dir/npu_bf16" --tolerance=3e-2 \
    --strict_selection=false || status=1
}
variant int4 --weights=int4 --weights=int4_fp32
variant hc_sum --hc_spelling=sum --hc_spelling=sum
exit $status
