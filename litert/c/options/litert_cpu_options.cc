// Copyright 2025 Google LLC.
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

#include "litert/c/options/litert_cpu_options.h"

#include <stdint.h>

#include <memory>

#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/litert_cpu_options.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"

LiteRtStatus LiteRtCreateCpuOptions(LiteRtOpaqueOptions* options) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  auto options_data = std::make_unique<LiteRtCpuOptionsT>();
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGetCpuOptionsIdentifier(), options_data.get(),
      [](void* payload) { delete reinterpret_cast<LiteRtCpuOptions>(payload); },
      options));
  options_data.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFindCpuOptions(LiteRtOpaqueOptions opaque_options,
                                  LiteRtCpuOptions* cpu_options) {
  LITERT_RETURN_IF_ERROR(cpu_options,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "cpu_options is null.";
  void* options_data = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtFindOpaqueOptionsData(
      opaque_options, LiteRtGetCpuOptionsIdentifier(), &options_data));
  *cpu_options = reinterpret_cast<LiteRtCpuOptions>(options_data);
  return kLiteRtStatusOk;
}

const char* LiteRtGetCpuOptionsIdentifier() { return "xnnpack"; }

LiteRtStatus LiteRtSetCpuOptionsNumThread(LiteRtCpuOptions options,
                                          int num_threads) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  options->xnn.num_threads = num_threads;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCpuOptionsNumThread(LiteRtCpuOptionsConst options,
                                          int* const num_threads) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(num_threads,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "num_threads is null.";
  *num_threads = options->xnn.num_threads;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetCpuOptionsXNNPackFlags(LiteRtCpuOptions options,
                                             uint32_t flags) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  options->xnn.flags = flags;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCpuOptionsXNNPackFlags(LiteRtCpuOptionsConst options,
                                             uint32_t* const flags) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(flags, litert::ErrorStatusBuilder::InvalidArgument())
      << "flags is null.";
  *flags = options->xnn.flags;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetCpuOptionsXnnPackWeightCachePath(LiteRtCpuOptions options,
                                                       const char* path) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(options->xnn.weight_cache_file_descriptor <= 0,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "weight cache file descriptor and path cannot both be set.";
  options->xnn.weight_cache_file_path = path;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCpuOptionsXnnPackWeightCachePath(
    LiteRtCpuOptionsConst options, const char** const path) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(path, litert::ErrorStatusBuilder::InvalidArgument())
      << "path is null.";
  *path = options->xnn.weight_cache_file_path;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetCpuOptionsXnnPackWeightCacheFileDescriptor(
    LiteRtCpuOptions options, int fd) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(options->xnn.weight_cache_file_path == nullptr,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "weight cache file descriptor and path cannot both be set.";
  options->xnn.weight_cache_file_descriptor = fd;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCpuOptionsXnnPackWeightCacheFileDescriptor(
    LiteRtCpuOptionsConst options, int* const fd) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(fd, litert::ErrorStatusBuilder::InvalidArgument())
      << "fd is null.";
  *fd = options->xnn.weight_cache_file_descriptor;
  return kLiteRtStatusOk;
}
