// Copyright 2024 Google LLC.
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

#include "litert/c/options/litert_darwinn_device_options.h"

#include <memory>
#include <string>

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/litert_darwinn_options.h"

LiteRtStatus LiteRtCreateDarwinnDeviceOptions(LiteRtOpaqueOptions* options) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto options_data = std::make_unique<litert::LiteRtDarwinnDeviceOptionsT>();
  
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGetDarwinnDeviceOptionsIdentifier(), options_data.get(),
      [](void* payload) { 
        delete reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(payload); 
      },
      options));
  
  options_data.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFindDarwinnDeviceOptions(
    LiteRtOpaqueOptions opaque_options,
    LiteRtDarwinnDeviceOptions* device_options) {
  LITERT_RETURN_IF_ERROR(device_options,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "device_options is null.";
  
  void* options_data = nullptr;
  LITERT_RETURN_IF_ERROR(LiteRtFindOpaqueOptionsData(
      opaque_options, LiteRtGetDarwinnDeviceOptionsIdentifier(), &options_data));
  
  *device_options = reinterpret_cast<LiteRtDarwinnDeviceOptions>(options_data);
  return kLiteRtStatusOk;
}

const char* LiteRtGetDarwinnDeviceOptionsIdentifier() {
  return litert::LiteRtDarwinnDeviceOptionsT::Identifier();
}

// Device selection setters/getters
LiteRtStatus LiteRtSetDarwinnDeviceType(LiteRtDarwinnDeviceOptions options,
                                        const char* device_type) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(device_type,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "device_type is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->device_type = device_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnDeviceType(
    LiteRtDarwinnDeviceOptionsConst options, const char** device_type) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(device_type,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "device_type is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *device_type = opts->device_type.c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnDevicePath(LiteRtDarwinnDeviceOptions options,
                                        const char* device_path) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->device_path = device_path ? device_path : "";
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnDevicePath(
    LiteRtDarwinnDeviceOptionsConst options, const char** device_path) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(device_path,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "device_path is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *device_path = opts->device_path.c_str();
  return kLiteRtStatusOk;
}

// Compilation options
LiteRtStatus LiteRtSetDarwinnEnableMultipleSubgraphs(
    LiteRtDarwinnDeviceOptions options, bool enable) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->enable_multiple_subgraphs = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnEnableMultipleSubgraphs(
    LiteRtDarwinnDeviceOptionsConst options, bool* enable) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(enable, litert::ErrorStatusBuilder::InvalidArgument())
      << "enable is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *enable = opts->enable_multiple_subgraphs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnCompileIfResize(
    LiteRtDarwinnDeviceOptions options, bool compile_if_resize) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->compile_if_resize = compile_if_resize;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnCompileIfResize(
    LiteRtDarwinnDeviceOptionsConst options, bool* compile_if_resize) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(compile_if_resize,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "compile_if_resize is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *compile_if_resize = opts->compile_if_resize;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnAllowCpuFallback(
    LiteRtDarwinnDeviceOptions options, bool allow_cpu_fallback) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->allow_cpu_fallback = allow_cpu_fallback;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnAllowCpuFallback(
    LiteRtDarwinnDeviceOptionsConst options, bool* allow_cpu_fallback) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(allow_cpu_fallback,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "allow_cpu_fallback is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *allow_cpu_fallback = opts->allow_cpu_fallback;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnSkipOpFilter(LiteRtDarwinnDeviceOptions options,
                                          bool skip_op_filter) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->skip_op_filter = skip_op_filter;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnSkipOpFilter(
    LiteRtDarwinnDeviceOptionsConst options, bool* skip_op_filter) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(skip_op_filter,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "skip_op_filter is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *skip_op_filter = opts->skip_op_filter;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnNumInterpreters(
    LiteRtDarwinnDeviceOptions options, int num_interpreters) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->num_interpreters = num_interpreters;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnNumInterpreters(
    LiteRtDarwinnDeviceOptionsConst options, int* num_interpreters) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(num_interpreters,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "num_interpreters is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *num_interpreters = opts->num_interpreters;
  return kLiteRtStatusOk;
}

// Memory configuration
LiteRtStatus LiteRtSetDarwinnAvoidBounceBuffer(
    LiteRtDarwinnDeviceOptions options, bool avoid_bounce_buffer) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->avoid_bounce_buffer = avoid_bounce_buffer;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnAvoidBounceBuffer(
    LiteRtDarwinnDeviceOptionsConst options, bool* avoid_bounce_buffer) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(avoid_bounce_buffer,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "avoid_bounce_buffer is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *avoid_bounce_buffer = opts->avoid_bounce_buffer;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnRegisterGraphDuringModify(
    LiteRtDarwinnDeviceOptions options, bool register_graph) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->register_graph_during_modify_graph_with_delegate = register_graph;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnRegisterGraphDuringModify(
    LiteRtDarwinnDeviceOptionsConst options, bool* register_graph) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(register_graph,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "register_graph is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *register_graph = opts->register_graph_during_modify_graph_with_delegate;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnInKernelFence(LiteRtDarwinnDeviceOptions options,
                                           bool in_kernel_fence) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->in_kernel_fence = in_kernel_fence;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnInKernelFence(
    LiteRtDarwinnDeviceOptionsConst options, bool* in_kernel_fence) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(in_kernel_fence,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "in_kernel_fence is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *in_kernel_fence = opts->in_kernel_fence;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnSkipIntermediateBufferAllocation(
    LiteRtDarwinnDeviceOptions options, bool skip_allocation) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->skip_intermediate_buffer_allocation = skip_allocation;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnSkipIntermediateBufferAllocation(
    LiteRtDarwinnDeviceOptionsConst options, bool* skip_allocation) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(skip_allocation,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "skip_allocation is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *skip_allocation = opts->skip_intermediate_buffer_allocation;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnGraphBuffersDonatable(
    LiteRtDarwinnDeviceOptions options, bool donatable) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->graph_buffers_donatable = donatable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnGraphBuffersDonatable(
    LiteRtDarwinnDeviceOptionsConst options, bool* donatable) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(donatable,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "donatable is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *donatable = opts->graph_buffers_donatable;
  return kLiteRtStatusOk;
}

// Async/Tachyon configuration
LiteRtStatus LiteRtSetDarwinnUseAsyncApi(LiteRtDarwinnDeviceOptions options,
                                         bool use_async_api) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->use_async_api = use_async_api;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnUseAsyncApi(
    LiteRtDarwinnDeviceOptionsConst options, bool* use_async_api) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(use_async_api,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "use_async_api is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *use_async_api = opts->use_async_api;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetDarwinnUseTachyon(LiteRtDarwinnDeviceOptions options,
                                       bool use_tachyon) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->use_tachyon = use_tachyon;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnUseTachyon(
    LiteRtDarwinnDeviceOptionsConst options, bool* use_tachyon) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(use_tachyon,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "use_tachyon is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *use_tachyon = opts->use_tachyon;
  return kLiteRtStatusOk;
}

// Logging
LiteRtStatus LiteRtSetDarwinnDisableLogInfo(
    LiteRtDarwinnDeviceOptions options, bool disable_log_info) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  
  auto* opts = reinterpret_cast<litert::LiteRtDarwinnDeviceOptionsT*>(options);
  opts->disable_log_info = disable_log_info;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDarwinnDisableLogInfo(
    LiteRtDarwinnDeviceOptionsConst options, bool* disable_log_info) {
  LITERT_RETURN_IF_ERROR(options, litert::ErrorStatusBuilder::InvalidArgument())
      << "options is null.";
  LITERT_RETURN_IF_ERROR(disable_log_info,
                         litert::ErrorStatusBuilder::InvalidArgument())
      << "disable_log_info is null.";
  
  auto* opts = reinterpret_cast<const litert::LiteRtDarwinnDeviceOptionsT*>(options);
  *disable_log_info = opts->disable_log_info;
  return kLiteRtStatusOk;
}