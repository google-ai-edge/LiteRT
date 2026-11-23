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
# Then the same with each form of the quantized attention layer
# (--quant_layer: int4, uint8, int8, fused_int4), which the NVIDIA backend
# lowers with LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION=1 and 16-bit
# activations only: the FP16 run must match, the BF16 run is reported.
#
# QSA_ARTIFACT names the tool (default: this tree's bazel-bin), and
# LD_LIBRARY_PATH must reach the TensorRT-RTX and CUDA libraries. The
# compiler's default partition policy takes the composites.
set -euo pipefail
libs=$(realpath "$1")
out=$(realpath -m "$2")
tool=${QSA_ARTIFACT:-$(dirname "$0")/../../../../bazel-bin/litert/vendors/nvidia/tools/qsa_artifact}

# run_npu MODEL DIR ACTIVATIONS [VARIABLE=VALUE...]
run_npu() {
  local model=$1 dir=$2 activations=$3
  shift 3
  mkdir -p "$dir"
  (
    for variable in ${!LITERT_NVIDIA_@}; do unset "$variable"; done
    export LD_LIBRARY_PATH=$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=$activations
    for assignment in "$@"; do export "$assignment"; done
    "$tool" --model="$model" --accelerator=npu --runtime_libs="$libs" \
      --out_dir="$dir"
  )
}

status=0
mkdir -p "$out/cpu"
"$tool" --write_model="$out/qsa_block.tflite"
"$tool" --model="$out/qsa_block.tflite" --accelerator=cpu --out_dir="$out/cpu"
run_npu "$out/qsa_block.tflite" "$out/npu_fp32" 0
run_npu "$out/qsa_block.tflite" "$out/npu_bf16" bf16
echo "== FP32 activations"
"$tool" --compare --cpu_dir="$out/cpu" --npu_dir="$out/npu_fp32" --tolerance=1e-3 || status=1
echo "== BF16 activations (reported)"
"$tool" --compare --cpu_dir="$out/cpu" --npu_dir="$out/npu_bf16" --tolerance=2e-2 || true

for form in int4 uint8 int8 fused_int4; do
  dir=$out/quant_$form
  mkdir -p "$dir/cpu"
  "$tool" --quant_layer=$form --write_model="$dir/model.tflite"
  "$tool" --model="$dir/model.tflite" --accelerator=cpu --out_dir="$dir/cpu"
  run_npu "$dir/model.tflite" "$dir/npu_fp16" 1 LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION=1
  run_npu "$dir/model.tflite" "$dir/npu_bf16" bf16 LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION=1
  echo "== $form, FP16 activations"
  "$tool" --compare --quant_layer=$form --cpu_dir="$dir/cpu" \
    --npu_dir="$dir/npu_fp16" --tolerance=1e-2 || status=1
  echo "== $form, BF16 activations (reported)"
  "$tool" --compare --quant_layer=$form --cpu_dir="$dir/cpu" \
    --npu_dir="$dir/npu_bf16" --tolerance=3e-2 || true
done
exit $status
