// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// g5_runner: run a Google Tensor G5 AOT-compiled TFLite model on the NPU.
//
// Loads a .tflite file that was AOT-compiled with the Google Tensor SDK
// (all ops replaced by a single DISPATCH_OP), runs one forward pass on the
// Tensor G5 NPU with zero-filled inputs, and prints the first few output
// values. Intended for verifying that a compiled model dispatches correctly
// to the NPU.
//
// Usage (on-device, run from a directory containing the dispatch library):
//   LD_LIBRARY_PATH=/data/local/tmp \
//   ./g5_runner [--cpu] [--dispatch_lib_dir=<dir>] <model.tflite>
//
// Flags:
//   --dispatch_lib_dir=<dir>   Directory containing
//                              libLiteRtDispatch_GoogleTensor.so
//                              (default: /data/local/tmp)
//   --cpu                      Force CPU mode instead of NPU. Note: G5-compiled
//                              models contain only DISPATCH_OP nodes and have
//                              no CPU kernels — useful only to check env init.
//   --print=<n>                Print first N float32 values per output tensor
//                              (default: 8, 0 = none).
//
// Prerequisites on device:
//   - libLiteRtDispatch_GoogleTensor.so in --dispatch_lib_dir
//   - Model compiled with ai-edge-litert for the Tensor G5 target

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

namespace {

void PrintUsage(const char* prog) {
  fprintf(stderr,
          "Usage: %s [--cpu] [--dispatch_lib_dir=<dir>] [--print=<n>] "
          "<model.tflite>\n",
          prog);
}

// Returns the total byte size of a ranked tensor.
size_t TensorBytes(const LiteRtRankedTensorType& t) {
  size_t elem_bytes = 4;  // default: float32 / int32
  switch (t.element_type) {
    case kLiteRtElementTypeInt8:
    case kLiteRtElementTypeUInt8:
      elem_bytes = 1;
      break;
    case kLiteRtElementTypeInt16:
    case kLiteRtElementTypeFloat16:
      elem_bytes = 2;
      break;
    case kLiteRtElementTypeInt64:
    case kLiteRtElementTypeFloat64:
      elem_bytes = 8;
      break;
    default:
      elem_bytes = 4;
      break;
  }
  size_t total = elem_bytes;
  for (LiteRtParamIndex i = 0; i < t.layout.rank; ++i) {
    total *= static_cast<size_t>(t.layout.dimensions[i]);
  }
  return total;
}

// Allocates a managed AHWB-backed tensor buffer.
//
// The G5 dispatch layer calls get_tensor_buffer_ahwb() on every tensor during
// LiteRtRunCompiledModel. Passing a HostMemory buffer causes a null-pointer
// deref inside the SouthBound firmware. AHWB buffers are required.
//
// Tries the compiled model's buffer requirements API first to get the correct
// NPU-aligned size. Falls back to direct AHWB allocation with the raw tensor
// size padded to a 4096-byte minimum (some driver versions reject very small
// AHardwareBuffer allocations).
LiteRtStatus AllocBuffer(LiteRtEnvironment env, LiteRtCompiledModel cm,
                         int sig_idx, int tensor_idx, bool is_input,
                         const LiteRtRankedTensorType& tensor_type,
                         LiteRtTensorBuffer* out) {
  LiteRtTensorBufferRequirements reqs;
  LiteRtStatus rs =
      is_input ? LiteRtGetCompiledModelInputBufferRequirements(
                     cm, sig_idx, tensor_idx, &reqs)
               : LiteRtGetCompiledModelOutputBufferRequirements(
                     cm, sig_idx, tensor_idx, &reqs);

  if (rs == kLiteRtStatusOk) {
    rs = LiteRtCreateManagedTensorBufferFromRequirements(env, &tensor_type,
                                                        reqs, out);
    if (rs == kLiteRtStatusOk) return kLiteRtStatusOk;
    fprintf(stderr,
            "  [warn] requirements-based alloc failed (status %d), "
            "retrying with direct AHWB\n",
            rs);
  }

  // Fallback: AHWB with size padded to a safe minimum.
  size_t sz = TensorBytes(tensor_type);
  if (sz < 4096) sz = 4096;
  return LiteRtCreateManagedTensorBuffer(
      env, kLiteRtTensorBufferTypeAhwb, &tensor_type, sz, out);
}

}  // namespace

int main(int argc, char** argv) {
  // --- Argument parsing ---
  const char* model_path = nullptr;
  const char* dispatch_lib_dir = "/data/local/tmp";
  bool force_cpu = false;
  int print_n = 8;

  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (strcmp(arg, "--cpu") == 0) {
      force_cpu = true;
    } else if (strncmp(arg, "--dispatch_lib_dir=", 19) == 0) {
      dispatch_lib_dir = arg + 19;
    } else if (strncmp(arg, "--print=", 8) == 0) {
      print_n = atoi(arg + 8);
    } else if (arg[0] != '-') {
      model_path = arg;
    } else {
      fprintf(stderr, "Unknown flag: %s\n", arg);
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (!model_path) {
    fprintf(stderr, "Error: model path is required.\n");
    PrintUsage(argv[0]);
    return 1;
  }

  // --- Dispatch library ---
  // dlopen makes the dispatch library globally visible before
  // LiteRtCreateEnvironment, which also loads it internally once
  // dispatch_lib_dir is set. Doing it here first gives a readable early error.
  void* dispatch_handle =
      dlopen("libLiteRtDispatch_GoogleTensor.so", RTLD_NOW | RTLD_GLOBAL);
  fprintf(stderr, "dispatch lib: %s\n",
          dispatch_handle ? "loaded" : "not found (NPU will not work)");

  // --- Environment ---
  // kLiteRtEnvOptionTagDispatchLibraryDir is required for the NPU accelerator
  // to register. Without it LiteRtCreateEnvironment returns
  // kLiteRtStatusErrorInvalidArgument for NPU registration, leaving the
  // dispatch runtime context uninitialised and causing a segfault later inside
  // LiteRtRunCompiledModel.
  LiteRtEnvOption env_opt;
  env_opt.tag = kLiteRtEnvOptionTagDispatchLibraryDir;
  env_opt.value.type = kLiteRtAnyTypeString;
  env_opt.value.str_value = dispatch_lib_dir;

  LiteRtEnvironment env;
  LiteRtStatus s = LiteRtCreateEnvironment(1, &env_opt, &env);
  fprintf(stderr, "env: status=%d\n", s);
  if (s != kLiteRtStatusOk) {
    fprintf(stderr, "Fatal: failed to create LiteRT environment (status=%d).\n",
            s);
    return 1;
  }

  // --- Model ---
  LiteRtModel model;
  s = LiteRtCreateModelFromFile(model_path, &model);
  fprintf(stderr, "model load: status=%d\n", s);
  if (s != kLiteRtStatusOk) {
    fprintf(stderr, "Fatal: failed to load model from '%s' (status=%d).\n",
            model_path, s);
    LiteRtDestroyEnvironment(env);
    return 1;
  }

  // Inspect input/output tensor shapes from the first subgraph.
  LiteRtSubgraph sg;
  LiteRtGetModelSubgraph(model, 0, &sg);
  LiteRtParamIndex num_inputs, num_outputs;
  LiteRtGetNumSubgraphInputs(sg, &num_inputs);
  LiteRtGetNumSubgraphOutputs(sg, &num_outputs);
  fprintf(stderr, "subgraph: %d input(s), %d output(s)\n", (int)num_inputs,
          (int)num_outputs);

  std::vector<LiteRtRankedTensorType> in_types(num_inputs);
  for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
    LiteRtTensor t;
    LiteRtGetSubgraphInput(sg, i, &t);
    LiteRtGetRankedTensorType(t, &in_types[i]);
    fprintf(stderr, "  in[%d]: rank=%d dims=[", (int)i,
            (int)in_types[i].layout.rank);
    for (LiteRtParamIndex d = 0; d < in_types[i].layout.rank; ++d)
      fprintf(stderr, "%s%d", d ? "," : "",
              (int)in_types[i].layout.dimensions[d]);
    fprintf(stderr, "] bytes=%zu\n", TensorBytes(in_types[i]));
  }

  std::vector<LiteRtRankedTensorType> out_types(num_outputs);
  for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
    LiteRtTensor t;
    LiteRtGetSubgraphOutput(sg, i, &t);
    LiteRtGetRankedTensorType(t, &out_types[i]);
    fprintf(stderr, "  out[%d]: rank=%d dims=[", (int)i,
            (int)out_types[i].layout.rank);
    for (LiteRtParamIndex d = 0; d < out_types[i].layout.rank; ++d)
      fprintf(stderr, "%s%d", d ? "," : "",
              (int)out_types[i].layout.dimensions[d]);
    fprintf(stderr, "] bytes=%zu\n", TensorBytes(out_types[i]));
  }

  // --- Compiled model (NPU, with CPU fallback on failure) ---
  LiteRtOptions options;
  LiteRtCreateOptions(&options);
  LiteRtSetOptionsHardwareAccelerators(
      options,
      force_cpu ? kLiteRtHwAcceleratorCpu : kLiteRtHwAcceleratorNpu);

  LiteRtCompiledModel cm;
  s = LiteRtCreateCompiledModel(env, model, options, &cm);
  if (s != kLiteRtStatusOk && !force_cpu) {
    fprintf(stderr,
            "  NPU compiled model failed (status=%d), trying CPU fallback\n",
            s);
    LiteRtDestroyOptions(options);
    LiteRtCreateOptions(&options);
    LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu);
    s = LiteRtCreateCompiledModel(env, model, options, &cm);
  }
  fprintf(stderr, "compiled model: status=%d\n", s);
  LiteRtDestroyOptions(options);
  LiteRtDestroyModel(model);
  if (s != kLiteRtStatusOk) {
    fprintf(stderr, "Fatal: failed to create compiled model (status=%d).\n", s);
    LiteRtDestroyEnvironment(env);
    return 1;
  }

  // --- Buffers ---
  std::vector<LiteRtTensorBuffer> input_bufs(num_inputs);
  for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
    s = AllocBuffer(env, cm, 0, i, /*is_input=*/true, in_types[i],
                    &input_bufs[i]);
    if (s != kLiteRtStatusOk) {
      fprintf(stderr,
              "Fatal: failed to allocate input buffer %d (status=%d).\n",
              (int)i, s);
      LiteRtDestroyCompiledModel(cm);
      LiteRtDestroyEnvironment(env);
      return 1;
    }
    void* data;
    LiteRtLockTensorBuffer(input_bufs[i], &data,
                           kLiteRtTensorBufferLockModeReadWrite);
    memset(data, 0, TensorBytes(in_types[i]));
    LiteRtUnlockTensorBuffer(input_bufs[i]);
  }

  std::vector<LiteRtTensorBuffer> output_bufs(num_outputs);
  for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
    s = AllocBuffer(env, cm, 0, i, /*is_input=*/false, out_types[i],
                    &output_bufs[i]);
    if (s != kLiteRtStatusOk) {
      fprintf(stderr,
              "Fatal: failed to allocate output buffer %d (status=%d).\n",
              (int)i, s);
      for (auto& b : input_bufs) LiteRtDestroyTensorBuffer(b);
      LiteRtDestroyCompiledModel(cm);
      LiteRtDestroyEnvironment(env);
      return 1;
    }
  }

  // --- Inference ---
  fprintf(stderr, "running...\n");
  s = LiteRtRunCompiledModel(cm, /*signature_index=*/0, num_inputs,
                             input_bufs.data(), num_outputs,
                             output_bufs.data());
  fprintf(stderr, "run: status=%d\n", s);

  if (s == kLiteRtStatusOk && print_n > 0) {
    for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
      void* odata;
      LiteRtLockTensorBuffer(output_bufs[i], &odata,
                             kLiteRtTensorBufferLockModeRead);
      const float* f = static_cast<const float*>(odata);
      size_t total_floats = TensorBytes(out_types[i]) / sizeof(float);
      size_t n = (static_cast<size_t>(print_n) < total_floats)
                     ? static_cast<size_t>(print_n)
                     : total_floats;
      fprintf(stderr, "  out[%d]:", (int)i);
      for (size_t j = 0; j < n; ++j) fprintf(stderr, " %.4f", f[j]);
      if (total_floats > n) fprintf(stderr, " ...");
      fprintf(stderr, "\n");
      LiteRtUnlockTensorBuffer(output_bufs[i]);
    }
  }

  // --- Cleanup ---
  for (auto& b : input_bufs) LiteRtDestroyTensorBuffer(b);
  for (auto& b : output_bufs) LiteRtDestroyTensorBuffer(b);
  LiteRtDestroyCompiledModel(cm);
  LiteRtDestroyEnvironment(env);

  return s == kLiteRtStatusOk ? 0 : 1;
}
