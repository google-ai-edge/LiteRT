// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/c/ov_core.h>

#include <openvino/openvino.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>

#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

namespace litert {

namespace openvino {

// Initialize the Dispatch API runtime.
// This function should be called before calling any other Dispatch API
// functions.
LiteRtStatus DispatchInitialize(LiteRtEnvironmentOptions environment_options,
                                LiteRtOptions options) {
  ov::Core core;
  std::vector<std::string> availableDevices = core.get_available_devices();
  for (auto&& device : availableDevices)
    LITERT_LOG(LITERT_INFO, "Openvino found device: %s", device.c_str());

  //    ov_litert_init();//TBD

  return kLiteRtStatusOk;
}

// Return the vendor id of the Dispatch API runtime.
// This function returns a pointer to a statically allocated string that is the
// ID of vendor providing the Dispatch API runtime.
LiteRtStatus DispatchGetVendorId(const char** vendor_id) {
  *vendor_id = "Intel_OpenVino";
  return kLiteRtStatusOk;
}

// Return the build ID of the Dispatch API runtime.
// This function returns a pointer to a statically allocated string that is the
// ID of the Dispatch API runtime build.
LiteRtStatus DispatchGetBuildId(const char** build_id) {
  *build_id = "1.0";
  return kLiteRtStatusOk;
}

// Return the capabilities supported by the Dispatch API runtime as a set of the
// values specified in LiteRtDispatchCapabilities.
LiteRtStatus DispatchGetCapabilities(int* capabilities) {
  // TBD: Check if higher capabilities can be supported like asyn & graph
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

// Create a `LiteRtDispatchDeviceContext` object.
// The returned object is used to talk with the underlying HW. The caller owns
// the memory associated with the context and should call
// LiteRtDispatchDeviceContextDestroy() to release it. Return NULL in case of
// error.
LiteRtStatus DispatchDeviceContextCreate(
    LiteRtDispatchDeviceContext* device_context) {
  // TODO: Get the device from env/config options.
  char* device = "NPU";  // can be CPU/GPU/NPU;
  ov::Core core;

  ////START NPU Specific plugin logic
  const auto arch = core.get_property("NPU", ov::device::architecture);
  const auto maxTiles = core.get_property("NPU", ov::intel_npu::max_tiles);
  bool compilerDQ = false;
  const auto supported_properties =
      core.get_property("NPU", ov::supported_properties);
  if (std::find(supported_properties.begin(), supported_properties.end(),
                "NPU_COMPILER_DYNAMIC_QUANTIZATION") !=
      supported_properties.end()) {
    compilerDQ = true;
  }

  //////END NPU Specific plugin logic

  return kLiteRtStatusOk;
}

// Release a `LiteRtDispatchDeviceContext` object.
// The given context should be release only after releasing all associated
// objects.
LiteRtStatus DispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  return kLiteRtStatusOk;
}

// Given a tensor type for an invocation context input, obtain the attributes
// the HW requires for the associated tensor buffer. The returned
// `tensor_buffer_requirements` object is owned by the caller.
LiteRtStatus DispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  return kLiteRtStatusOk;
}

// Given a tensor type for an invocation context output, obtain the attributes
// the HW requires for the associated tensor buffer. The returned
// `tensor_buffer_requirements` object is owned by the caller.
LiteRtStatus DispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  return kLiteRtStatusOk;
}

// Registers a buffer with the given device context.
// Note: The memory backing the buffer should be valid until
// `LiteRtDispatchUnregisterTensorBuffer` is called.
LiteRtStatus DispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  return kLiteRtStatusOk;
}

// Unregisters the registered buffer associated with the given
// `LiteRtTensorBufferHandle`.
// Note: The registered `LiteRtTensorBufferHandle` is supposed to be
// unregistered with this function before the associated `ThrContext` is deleted
// by calling `LiteRtDispatchDeviceContextDestroy`.
LiteRtStatus DispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusOk;
}

// Create an invocation context to run a given function from a given
// executable. Parameter `function_name` is required if the provided executable
// includes multiple functions.
LiteRtStatus DispatchInvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context) {
  return kLiteRtStatusOk;
}

// Start collection of HW-specific metrics at a specific level of detail (>= 0).
LiteRtStatus DispatchStartMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context, int detail_level) {
  return kLiteRtStatusOk;
}

// Stop collection of HW-specific metrics and report the collected
// metrics. Note: The caller is responsible for deallocating the returned
// metrics by calling `LiteRtDispatchDestroyMetrics`.
LiteRtStatus DispatchStopMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchMetrics* metrics) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchGetNumMetrics(LiteRtDispatchMetrics metrics,
                                   int* num_metrics) {
  return kLiteRtStatusOk;
}

// Fetch a specific metric. The runtime owns the returned object.
LiteRtStatus DispatchGetMetric(LiteRtDispatchMetrics metrics, int metric_index,
                               LiteRtMetric* metric) {
  return kLiteRtStatusOk;
}

LiteRtStatus DispatchDestroyMetrics(LiteRtDispatchMetrics metrics) {
  return kLiteRtStatusOk;
}

}  // namespace openvino
}  // namespace litert

namespace {

LiteRtDispatchInterface TheInterface = {
    .initialize = litert::openvino::DispatchInitialize,
    .get_vendor_id = litert::openvino::DispatchGetVendorId,
    .get_build_id = litert::openvino::DispatchGetBuildId,
    .get_capabilities = litert::openvino::DispatchGetCapabilities,
    .device_context_create = litert::openvino::DispatchDeviceContextCreate,
    .device_context_destroy = litert::openvino::DispatchDeviceContextDestroy,
    .get_input_requirements = litert::openvino::DispatchGetInputRequirements,
    .get_output_requirements = litert::openvino::DispatchGetOutputRequirements,
    .register_tensor_buffer = litert::openvino::DispatchRegisterTensorBuffer,
    .unregister_tensor_buffer =
        litert::openvino::DispatchUnregisterTensorBuffer,
    .invocation_context_create =
        litert::openvino::DispatchInvocationContextCreate,
    .invocation_context_destroy =
        litert::openvino::DispatchInvocationContextDestroy,
    .attach_input = litert::openvino::DispatchAttachInput,
    .attach_output = litert::openvino::DispatchAttachOutput,
    .detach_input = litert::openvino::DispatchDetachInput,
    .detach_output = litert::openvino::DispatchDetachOutput,
    .invoke = litert::openvino::DispatchInvoke,
    .start_metrics_collection =
        litert::openvino::DispatchStartMetricsCollection,
    .stop_metrics_collection = litert::openvino::DispatchStopMetricsCollection,
    .get_num_metrics = litert::openvino::DispatchGetNumMetrics,
    .get_metric = litert::openvino::DispatchGetMetric,
    .destroy_metrics = litert::openvino::DispatchDestroyMetrics,
};

LiteRtDispatchApi TheApi = {
    .version = {.major = LITERT_API_VERSION_MAJOR,
                .minor = LITERT_API_VERSION_MINOR,
                .patch = LITERT_API_VERSION_PATCH},
    .interface = &TheInterface,
    .async_interface = nullptr,
    .graph_interface = nullptr,
};

}  // namespace

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) {
  *api = TheApi;
  return kLiteRtStatusOk;
}
