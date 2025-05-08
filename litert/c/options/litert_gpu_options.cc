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

#include "litert/c/options/litert_gpu_options.h"

#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

using ::litert::ErrorStatusBuilder;

struct LiteRtGpuOptionsPayloadT {
  // Increment the minor version every time a field is added.
  static constexpr const absl::string_view kIdentifier = "gpu_payload";

  bool enable_constant_tensor_sharing = false;
  bool enable_infinite_float_capping = false;
  bool benchmark_mode = false;
  // Added in version 1.2.0.
  bool allow_src_quantized_fc_conv_ops = false;
  LiteRtDelegatePrecision precision = kLiteRtDelegatePrecisionDefault;
  LiteRtDelegateBufferStorageType buffer_storage_type =
      kLiteRtDelegateBufferStorageTypeDefault;
};

namespace litert {
namespace {

litert::Expected<LiteRtGpuOptionsPayloadT*> GetPayload(
    LiteRtOpaqueOptions options) {
  const char* identifier = nullptr;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsIdentifier(options, &identifier));
  LITERT_RETURN_IF_ERROR(identifier == LiteRtGpuOptionsPayloadT::kIdentifier,
                         ErrorStatusBuilder::InvalidArgument())
      << "Payload stored in accelerator options is incompatible. Got "
      << identifier << ", expected " << LiteRtGpuOptionsPayloadT::kIdentifier
      << ".";

  LiteRtGpuOptionsPayloadT* payload;
  LITERT_RETURN_IF_ERROR(
      LiteRtGetOpaqueOptionsData(options, reinterpret_cast<void**>(&payload)));
  return payload;
}

}  // namespace
}  // namespace litert

LiteRtStatus LiteRtCreateGpuOptions(LiteRtOpaqueOptions* options) {
  auto payload = std::make_unique<LiteRtGpuOptionsPayloadT>();
  LITERT_RETURN_IF_ERROR(LiteRtCreateOpaqueOptions(
      LiteRtGpuOptionsPayloadT::kIdentifier.data(), payload.get(),
      [](void* payload) {
        delete reinterpret_cast<LiteRtGpuOptionsPayloadT*>(payload);
      },
      options));
  payload.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsConstantTensorSharing(
    LiteRtOpaqueOptions gpu_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->enable_constant_tensor_sharing = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsInfiniteFloatCapping(
    LiteRtOpaqueOptions gpu_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->enable_infinite_float_capping = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuOptionsBenchmarkMode(LiteRtOpaqueOptions gpu_options,
                                              bool enable) {
  LITERT_ASSIGN_OR_RETURN(LiteRtGpuOptionsPayloadT * payload,
                          litert::GetPayload(gpu_options));
  payload->benchmark_mode = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtSetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    LiteRtOpaqueOptions gpu_accelerator_options, bool enable) {
  LITERT_ASSIGN_OR_RETURN(
      LiteRtGpuOptionsPayloadT * payload,
      litert::GetPayload(gpu_accelerator_options));
  payload->allow_src_quantized_fc_conv_ops = enable;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegatePrecision precision) {
  LITERT_ASSIGN_OR_RETURN(
      LiteRtGpuOptionsPayloadT * payload,
      litert::GetPayload(gpu_accelerator_options));
  payload->precision = precision;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetGpuAcceleratorCompilationOptionsUseBufferStorageType(
    LiteRtOpaqueOptions gpu_accelerator_options,
    LiteRtDelegateBufferStorageType buffer_storage_type) {
  LITERT_ASSIGN_OR_RETURN(
      LiteRtGpuOptionsPayloadT * payload,
      litert::GetPayload(gpu_accelerator_options));
  payload->buffer_storage_type = buffer_storage_type;
  return kLiteRtStatusOk;
}

const char* LiteRtGetGpuOptionsPayloadIdentifier() {
  return LiteRtGpuOptionsPayloadT::kIdentifier.data();
}

LiteRtStatus LiteRtGetGpuOptionsConstantTensorSharing(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->enable_constant_tensor_sharing;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsInfiniteFloatCapping(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->enable_infinite_float_capping;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuOptionsBenchmarkMode(bool* enabled,
                                              LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->benchmark_mode;
  return kLiteRtStatusOk;
}

LiteRtStatus
LiteRtGetGpuAcceleratorCompilationOptionsAllowSrcQuantizedFcConvOps(
    bool* enabled, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(enabled, ErrorStatusBuilder::InvalidArgument())
      << "`enabled` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *enabled = payload->allow_src_quantized_fc_conv_ops;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsPrecision(
    LiteRtDelegatePrecision* precision, LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(precision, ErrorStatusBuilder::InvalidArgument())
      << "`precision` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *precision = payload->precision;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGpuAcceleratorCompilationOptionsBufferStorageType(
    LiteRtDelegateBufferStorageType* buffer_storage_type,
    LiteRtGpuOptionsPayload payload) {
  LITERT_RETURN_IF_ERROR(buffer_storage_type,
                         ErrorStatusBuilder::InvalidArgument())
      << "`use_buffer_storage_type` cannot be null.";
  LITERT_RETURN_IF_ERROR(payload, ErrorStatusBuilder::InvalidArgument())
      << "`payload` cannot be null.";
  *buffer_storage_type = payload->buffer_storage_type;
  return kLiteRtStatusOk;
}
