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
#include <memory>
#include <optional>
#include <string>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/common/vendor_dispatch_base.h"
#include "litert/vendors/common/vendor_traits.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/dispatch/litert_dispatch_device_context.h"
#include "litert/vendors/qualcomm/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/qualcomm/qnn_manager.h"
#include "QnnCommon.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt

namespace litert {
namespace vendors {

// Static storage for QNN manager
static std::unique_ptr<litert::qnn::QnnManager> TheQnnManager;
static std::string TheBuildId;

// Implement trait methods for Qualcomm
template <>
LiteRtStatus VendorTraits<QualcommTag>::Initialize(const std::string& lib_dir) {
  std::optional<std::string> shared_library_dir = 
      !lib_dir.empty() ? std::make_optional(lib_dir) : std::nullopt;
  
  auto configs = litert::qnn::QnnManager::DefaultBackendConfigs();
  ::qnn::Options qnn_options;
  qnn_options.SetHtpPerformanceMode(::qnn::HtpPerformanceMode::kBurst);
  qnn_options.SetLogLevel(::qnn::LogLevel::kOff);
  
  auto qnn_manager = litert::qnn::QnnManager::Create(
      /*configs=*/configs,
      /*options=*/qnn_options,
      /*shared_library_dir=*/shared_library_dir,
      /*soc_model=*/std::nullopt);
  
  if (!qnn_manager) {
    LITERT_LOG(LITERT_ERROR, "%s", qnn_manager.Error().Message().c_str());
    return qnn_manager.Error().Status();
  }
  
  TheQnnManager = std::move(*qnn_manager);
  
  // Build version string
  Qnn_ApiVersion_t qnn_api_version;
  if (auto status = TheQnnManager->Api()->backendGetApiVersion(&qnn_api_version);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get QNN API version: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  
  const char* backend_build_id = nullptr;
  if (auto status = TheQnnManager->Api()->backendGetBuildId(&backend_build_id);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get backend build ID: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }
  
  char build_id[256];
  std::snprintf(build_id, sizeof(build_id),
                "QNN API v%u.%u.%u | Backend %s | LiteRT Dispatch v%d.%d.%d",
                qnn_api_version.coreApiVersion.major,
                qnn_api_version.coreApiVersion.minor,
                qnn_api_version.coreApiVersion.patch,
                backend_build_id != nullptr ? backend_build_id : "Unknown",
                LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
                LITERT_API_VERSION_PATCH);
  TheBuildId = build_id;
  
  return kLiteRtStatusOk;
}

template <>
std::string VendorTraits<QualcommTag>::GetBuildId() {
  return TheBuildId;
}

template <>
Expected<std::unique_ptr<VendorDeviceContext>> 
VendorTraits<QualcommTag>::CreateDeviceContext(
    const LiteRtDispatchDeviceContext* device_context_options) {
  if (!TheQnnManager) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                     "QNN Manager not initialized");
  }
  
  auto result = litert::qnn::LiteRtDispatchDeviceContextT::Create(
      *TheQnnManager, *device_context_options);
  
  if (!result.ok()) {
    return Unexpected(result.Error());
  }
  
  return std::unique_ptr<VendorDeviceContext>(result.Value().release());
}

template <>
LiteRtStatus VendorTraits<QualcommTag>::RegisterTensorBuffer(
    VendorDeviceContext* context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  auto* qnn_context = static_cast<litert::qnn::LiteRtDispatchDeviceContextT*>(context);
  
  LITERT_ASSIGN_OR_RETURN(auto handle,
                          qnn_context->RegisterTensorBuffer(tensor_buffer));
  *tensor_buffer_handle = handle;
  return kLiteRtStatusOk;
}

template <>
LiteRtStatus VendorTraits<QualcommTag>::UnregisterTensorBuffer(
    VendorDeviceContext* context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto* qnn_context = static_cast<litert::qnn::LiteRtDispatchDeviceContextT*>(context);
  return qnn_context->UnregisterTensorBuffer(tensor_buffer_handle);
}

template <>
Expected<std::unique_ptr<VendorInvocationContext>>
VendorTraits<QualcommTag>::CreateInvocationContext(
    VendorDeviceContext* device_context,
    const void* exec_bytecode_ptr,
    size_t exec_bytecode_size,
    const char* function_name) {
  auto* qnn_device_context = 
      static_cast<litert::qnn::LiteRtDispatchDeviceContextT*>(device_context);
  
  // Create LiteRtMemBuffer from the raw pointer and size
  LiteRtMemBuffer mem_buffer = {
      .fd = -1,
      .base_addr = exec_bytecode_ptr,
      .offset = 0,
      .size = exec_bytecode_size
  };
  
  auto result = litert::qnn::LiteRtDispatchInvocationContextT::Create(
      *TheQnnManager, *qnn_device_context, &mem_buffer, function_name);
  
  if (!result.ok()) {
    return Unexpected(result.Error());
  }
  
  return std::unique_ptr<VendorInvocationContext>(result.Value().release());
}

// Implement GetInputRequirements and GetOutputRequirements specializations
template <>
LiteRtStatus VendorTraits<QualcommTag>::GetInputRequirements(
    VendorInvocationContext* invocation_context,
    int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  
  auto* qnn_invocation_context = 
      static_cast<litert::qnn::LiteRtDispatchInvocationContextT*>(
          invocation_context);
  
  auto requirements = qnn_invocation_context->GetInputRequirements(
      input_index, *tensor_type);
  
  if (!requirements.ok()) {
    return requirements.Error().Status();
  }
  
  *tensor_buffer_requirements = requirements.Value();
  return kLiteRtStatusOk;
}

template <>
LiteRtStatus VendorTraits<QualcommTag>::GetOutputRequirements(
    VendorInvocationContext* invocation_context,
    int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  
  auto* qnn_invocation_context = 
      static_cast<litert::qnn::LiteRtDispatchInvocationContextT*>(
          invocation_context);
  
  auto requirements = qnn_invocation_context->GetOutputRequirements(
      output_index, *tensor_type);
  
  if (!requirements.ok()) {
    return requirements.Error().Status();
  }
  
  *tensor_buffer_requirements = requirements.Value();
  return kLiteRtStatusOk;
}

}  // namespace vendors
}  // namespace litert

// Define the entry point using the macro
extern "C" {
DEFINE_VENDOR_DISPATCH_ENTRY_POINT(litert::vendors::QualcommTag)
}