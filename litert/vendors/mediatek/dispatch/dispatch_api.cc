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

// This is the migrated version using the template framework

#include <any>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_mediatek_options.h"
#include "litert/vendors/common/vendor_dispatch_base.h"
#include "litert/vendors/common/vendor_traits.h"
#include "litert/vendors/mediatek/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/mediatek/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert {
namespace vendors {

static std::unique_ptr<mediatek::NeuronAdapterApi> TheNeuronAdapterApi;
static std::string TheBuildId;

// Implement trait methods for MediaTek
LiteRtStatus VendorTraits<MediaTekTag>::Initialize(const std::string& lib_dir) {
  if (!TheNeuronAdapterApi) {
    // Create MediaTek options
    auto mediatek_options = mediatek::MediatekOptions::Create();
    if (!mediatek_options) {
      LITERT_LOG(LITERT_ERROR, "Failed to create MediaTek options: %s",
                 mediatek_options.Error().Message().c_str());
      return mediatek_options.Error().Status();
    }

    // Create NeuronAdapter API with options
    auto api = mediatek::NeuronAdapterApi::Create(
        lib_dir.empty() ? std::nullopt : std::make_optional(lib_dir),
        mediatek_options);
    if (!api) {
      LITERT_LOG(LITERT_ERROR, "Failed to create NeuronAdapter API: %s",
                 api.Error().Message().c_str());
      return api.Error().Status();
    }
    TheNeuronAdapterApi = std::move(api.Value());
  }

  // Build version string
  TheBuildId = absl::StrFormat(
      "NeuronAdapter | LiteRT Dispatch v%d.%d.%d", LITERT_API_VERSION_MAJOR,
      LITERT_API_VERSION_MINOR, LITERT_API_VERSION_PATCH);

  return kLiteRtStatusOk;
}

std::string VendorTraits<MediaTekTag>::GetBuildId() { return TheBuildId; }

Expected<std::unique_ptr<VendorDeviceContext>>
VendorTraits<MediaTekTag>::CreateDeviceContext(
    const LiteRtDispatchDeviceContext* device_context_options) {
  if (!TheNeuronAdapterApi) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "NeuronAdapter API not initialized");
  }

  auto result = LiteRtDispatchDeviceContextT::Create(*TheNeuronAdapterApi,
                                                     *device_context_options);

  if (!result) {
    return Unexpected(result.Error());
  }

  return std::unique_ptr<VendorDeviceContext>(result.Value().release());
}

LiteRtStatus VendorTraits<MediaTekTag>::RegisterTensorBuffer(
    VendorDeviceContext* context, LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  auto* mtk_context = static_cast<LiteRtDispatchDeviceContextT*>(context);

  auto handle = mtk_context->RegisterTensorBuffer(tensor_buffer);
  if (!handle) {
    return handle.Error().Status();
  }

  *tensor_buffer_handle = handle.Value();
  return kLiteRtStatusOk;
}

LiteRtStatus VendorTraits<MediaTekTag>::UnregisterTensorBuffer(
    VendorDeviceContext* context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto* mtk_context = static_cast<LiteRtDispatchDeviceContextT*>(context);

  auto result = mtk_context->UnregisterTensorBuffer(tensor_buffer_handle);
  return result ? kLiteRtStatusOk : result.Error().Status();
}

Expected<std::unique_ptr<VendorInvocationContext>>
VendorTraits<MediaTekTag>::CreateInvocationContext(
    VendorDeviceContext* device_context, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name) {
  if (!TheNeuronAdapterApi) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "NeuronAdapter API not initialized");
  }

  auto* mtk_device_context =
      static_cast<LiteRtDispatchDeviceContextT*>(device_context);

  // Create LiteRtMemBuffer from the raw pointer and size
  LiteRtMemBuffer mem_buffer = {.fd = -1,
                                .base_addr = exec_bytecode_ptr,
                                .offset = 0,
                                .size = exec_bytecode_size};

  // Get num_inputs and num_outputs from the bytecode
  // For MediaTek, we need to parse the schema to get these values
  // For now, we'll use default values - this should be updated based on actual
  // schema parsing
  int num_inputs = 2;   // Default for testing
  int num_outputs = 1;  // Default for testing

  auto result = LiteRtDispatchInvocationContextT::Create(
      *TheNeuronAdapterApi, *mtk_device_context,
      kLiteRtDispatchExecutableTypeMlModel, &mem_buffer, function_name,
      num_inputs, num_outputs);

  if (!result) {
    return Unexpected(result.Error());
  }

  return std::unique_ptr<VendorInvocationContext>(result.Value().release());
}

}  // namespace vendors
}  // namespace litert

// Use the macro to define the dispatch entry point
DEFINE_VENDOR_DISPATCH_ENTRY_POINT(litert::vendors::MediaTekTag)
