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

#include "litert/c/internal/litert_accelerator_def.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"

extern "C" {

#if defined(FAKE_WEBGPU)
constexpr char kAcceleratorName[] = "GPU WebGPU";
#elif defined(FAKE_UNSUPPORTED)
constexpr char kAcceleratorName[] = "GPU Unsupported";
#else
constexpr char kAcceleratorName[] = "GPU OpenCL";
#endif

static LiteRtStatus GetName(LiteRtAcceleratorConst accelerator,
                            const char** name) {
  *name = kAcceleratorName;
  return kLiteRtStatusOk;
}

static LiteRtStatus GetVersion(LiteRtAcceleratorConst accelerator,
                               LiteRtApiVersion* version) {
  version->major = 1;
  version->minor = 0;
  version->patch = 0;
  return kLiteRtStatusOk;
}

static LiteRtStatus GetHardwareSupport(
    LiteRtAcceleratorConst accelerator,
    LiteRtHwAcceleratorSet* supported_hardware) {
#if defined(FAKE_UNSUPPORTED)
  *supported_hardware = kLiteRtHwAcceleratorNone;
#else
  *supported_hardware = kLiteRtHwAcceleratorGpu;
#endif
  return kLiteRtStatusOk;
}

static LiteRtStatus IsJit(LiteRtAcceleratorConst accelerator,
                          bool* does_jit_compilation) {
  *does_jit_compilation = false;
  return kLiteRtStatusOk;
}

static LiteRtStatus CreateDelegate(LiteRtRuntimeContext* runtime_context,
                                   LiteRtEnvironment env,
                                   LiteRtAcceleratorConst accelerator,
                                   LiteRtOptions options,
                                   LiteRtDelegateWrapper* delegate_wrapper) {
  *delegate_wrapper = nullptr;
  return kLiteRtStatusOk;
}

#if defined(_WIN32)
#define FAKE_ACCELERATOR_EXPORT __declspec(dllexport)
#else
#define FAKE_ACCELERATOR_EXPORT __attribute__((visibility("default")))
#endif

FAKE_ACCELERATOR_EXPORT extern const LiteRtAcceleratorDef LiteRtAcceleratorImpl;

const LiteRtAcceleratorDef LiteRtAcceleratorImpl = {
    .abi_header =
        {
            .struct_size = sizeof(LiteRtAcceleratorDefV1),
            .major_version = 1,
            .minor_version = 0,
            .reserved = 0,
        },
    .get_name = GetName,
    .get_version = GetVersion,
    .get_hardware_support = GetHardwareSupport,
    .is_tflite_delegate_responsible_for_jit_compilation = IsJit,
    .create_delegate = CreateDelegate,
    .start_metrics_collection = nullptr,
    .stop_metrics_collection = nullptr,
    .buffer_handlers = {},
};

}  // extern "C"
