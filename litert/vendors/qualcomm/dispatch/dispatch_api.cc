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

using qnn::QnnManager;

static std::unique_ptr<QnnManager> TheQnnManager;
static std::string TheBuildId;

// Implement trait methods for Qualcomm
LiteRtStatus VendorTraits<QualcommTag>::Initialize(const std::string& lib_dir) {
  std::optional<std::string> shared_library_dir =
      !lib_dir.empty() ? std::make_optional(lib_dir) : std::nullopt;

  auto configs = QnnManager::DefaultBackendConfigs();
  // TODO(Alen): initialize qnn_options from LiteRtOptions
  ::qnn::Options qnn_options;
  qnn_options.SetHtpPerformanceMode(::qnn::HtpPerformanceMode::kBurst);
  qnn_options.SetLogLevel(::qnn::LogLevel::kOff);

  auto qnn_manager = QnnManager::Create(
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
  if (auto status =
          TheQnnManager->Api()->backendGetApiVersion(&qnn_api_version);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get QNN API version: %d", status);
    return kLiteRtStatusErrorRuntimeFailure;
  }

  const char* backend_build_id = nullptr;
  if (auto build_status =
          TheQnnManager->Api()->backendGetBuildId(&backend_build_id);
      build_status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get backend build ID: %d",
               build_status);
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

std::string VendorTraits<QualcommTag>::GetBuildId() { return TheBuildId; }

Expected<std::unique_ptr<VendorDeviceContext>>
VendorTraits<QualcommTag>::CreateDeviceContext(
    const LiteRtDispatchDeviceContext* device_context_options) {
  if (!TheQnnManager) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "QNN Manager not initialized");
  }

  auto result = LiteRtDispatchDeviceContextT::Create(*TheQnnManager,
                                                     *device_context_options);

  if (!result) {
    return Unexpected(result.Error());
  }

  return std::unique_ptr<VendorDeviceContext>(result.Value().release());
}

LiteRtStatus VendorTraits<QualcommTag>::RegisterTensorBuffer(
    VendorDeviceContext* context, LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  auto* qnn_context = static_cast<LiteRtDispatchDeviceContextT*>(context);

  LITERT_ASSIGN_OR_RETURN(auto handle,
                          qnn_context->RegisterTensorBuffer(tensor_buffer));
  *tensor_buffer_handle = handle;
  return kLiteRtStatusOk;
}

LiteRtStatus VendorTraits<QualcommTag>::UnregisterTensorBuffer(
    VendorDeviceContext* context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto* qnn_context = static_cast<LiteRtDispatchDeviceContextT*>(context);
  return qnn_context->UnregisterTensorBuffer(tensor_buffer_handle);
}

Expected<std::unique_ptr<VendorInvocationContext>>
VendorTraits<QualcommTag>::CreateInvocationContext(
    VendorDeviceContext* device_context, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name) {
  auto* qnn_device_context =
      static_cast<LiteRtDispatchDeviceContextT*>(device_context);

  // Create LiteRtMemBuffer from the raw pointer and size
  LiteRtMemBuffer mem_buffer = {.fd = -1,
                                .base_addr = exec_bytecode_ptr,
                                .offset = 0,
                                .size = exec_bytecode_size};

  auto result = LiteRtDispatchInvocationContextT::Create(
      *TheQnnManager, *qnn_device_context, &mem_buffer, function_name);

  if (!result) {
    return Unexpected(result.Error());
  }

  return std::unique_ptr<VendorInvocationContext>(result.Value().release());
}

// Use the macro to define the dispatch entry point
DEFINE_VENDOR_DISPATCH_ENTRY_POINT(QualcommTag)

}  // namespace vendors
}  // namespace litert
