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

#ifndef LITERT_VENDORS_COMMON_VENDOR_DISPATCH_BASE_H_
#define LITERT_VENDORS_COMMON_VENDOR_DISPATCH_BASE_H_

#include <any>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/common/vendor_traits.h"

namespace litert::vendors {

// Common metrics implementation that vendors can use
class VendorMetrics {
 public:
  struct Metric {
    std::string name;
    int64_t value;
  };

  void AddMetric(const std::string& name, int64_t value) {
    metrics_.push_back({name, value});
  }

  int GetNumMetrics() const { return metrics_.size(); }

  LiteRtMetric GetMetric(int index) const {
    if (index < 0 || index >= static_cast<int>(metrics_.size())) {
      return LiteRtMetric{.name = nullptr,
                          .value = LiteRtAny{.type = kLiteRtAnyTypeNone}};
    }
    return LiteRtMetric{.name = metrics_[index].name.c_str(),
                        .value = LiteRtAny{.type = kLiteRtAnyTypeInt,
                                           .int_value = metrics_[index].value}};
  }

 private:
  std::vector<Metric> metrics_;
};

// Opaque handle type for metrics
struct LiteRtDispatchMetricsT : public VendorMetrics {};

// Base class for vendor device context
class VendorDeviceContext {
 public:
  virtual ~VendorDeviceContext() = default;

  // Get the underlying vendor-specific context
  virtual void* GetBackendContext() = 0;

  // Get the dispatch device context info
  const LiteRtDispatchDeviceContext& GetDeviceContext() const {
    return device_context_;
  }

 protected:
  explicit VendorDeviceContext(
      const LiteRtDispatchDeviceContext& device_context)
      : device_context_(device_context) {}

 private:
  LiteRtDispatchDeviceContext device_context_;
};

// Base class for vendor invocation context
class VendorInvocationContext {
 public:
  virtual ~VendorInvocationContext() = default;

  // Attach/detach buffers
  virtual LiteRtStatus AttachInput(int graph_input_idx,
                                   LiteRtTensorBufferHandle handle) = 0;
  virtual LiteRtStatus AttachOutput(int graph_output_idx,
                                    LiteRtTensorBufferHandle handle) = 0;
  virtual LiteRtStatus DetachInput(int graph_input_idx,
                                   LiteRtTensorBufferHandle handle) = 0;
  virtual LiteRtStatus DetachOutput(int graph_output_idx,
                                    LiteRtTensorBufferHandle handle) = 0;

  // Execute the model
  virtual LiteRtStatus Invoke() = 0;

  // Metrics collection methods
  virtual LiteRtStatus StartMetricsCollection(int detail_level) {
    return kLiteRtStatusErrorUnsupported;
  }

  virtual LiteRtStatus StopMetricsCollection(LiteRtDispatchMetrics* metrics) {
    return kLiteRtStatusErrorUnsupported;
  }

  // Get the underlying vendor-specific context
  virtual void* GetBackendContext() = 0;

 protected:
  VendorInvocationContext() = default;
};

// Template class for vendor dispatch implementation
template <typename VendorTag>
class VendorDispatch {
 public:
  using Traits = VendorTraits<VendorTag>;

  // Initialize the vendor dispatch
  static LiteRtStatus Initialize(LiteRtEnvironmentOptions environment_options,
                                 LiteRtOptions options) {
    // Extract library directory from environment options
    auto env_options = litert::EnvironmentOptions(environment_options);
    auto dispatch_lib_dir_any =
        env_options.GetOption(kLiteRtEnvOptionTagDispatchLibraryDir);

    if (!dispatch_lib_dir_any) {
      LITERT_LOG(LITERT_ERROR, "Dispatch library directory not provided");
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* dispatch_lib_dir = std::any_cast<const char*>(*dispatch_lib_dir_any);
    if (!dispatch_lib_dir) {
      LITERT_LOG(LITERT_ERROR, "Failed to get dispatch library directory");
      return kLiteRtStatusErrorInvalidArgument;
    }

    // Initialize vendor-specific backend
    return Traits::Initialize(dispatch_lib_dir);
  }

  // Get vendor ID
  static LiteRtStatus GetVendorId(const char** vendor_id) {
    if (!vendor_id) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    *vendor_id = Traits::kVendorId;
    return kLiteRtStatusOk;
  }

  // Get build ID
  static LiteRtStatus GetBuildId(const char** build_id) {
    if (!build_id) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    static std::string build_id_str = Traits::GetBuildId();
    *build_id = build_id_str.c_str();
    return kLiteRtStatusOk;
  }

  // Get capabilities
  static LiteRtStatus GetCapabilities(int* capabilities) {
    if (!capabilities) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    *capabilities = static_cast<int>(Traits::kCapabilities);
    return kLiteRtStatusOk;
  }

  // Create device context
  static LiteRtStatus DeviceContextCreate(
      LiteRtDispatchDeviceContext* device_context_handle) {
    if (!device_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    // Create empty device context info
    LiteRtDispatchDeviceContext device_context = nullptr;
    auto result = Traits::CreateDeviceContext(&device_context);
    if (!result) {
      LITERT_LOG(LITERT_ERROR, "Failed to create device context: %s",
                 result.Error().Message().c_str());
      return result.Error().Status();
    }

    *device_context_handle =
        reinterpret_cast<LiteRtDispatchDeviceContext>(result.Value().release());
    return kLiteRtStatusOk;
  }

  // Destroy device context
  static LiteRtStatus DeviceContextDestroy(
      LiteRtDispatchDeviceContext device_context_handle) {
    if (!device_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorDeviceContext*>(device_context_handle);
    delete context;
    return kLiteRtStatusOk;
  }

  // Register tensor buffer
  static LiteRtStatus RegisterTensorBuffer(
      LiteRtDispatchDeviceContext device_context_handle,
      LiteRtTensorBuffer tensor_buffer,
      LiteRtTensorBufferHandle* tensor_buffer_handle) {
    if (!device_context_handle || !tensor_buffer || !tensor_buffer_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorDeviceContext*>(device_context_handle);
    return Traits::RegisterTensorBuffer(context, tensor_buffer,
                                        tensor_buffer_handle);
  }

  // Unregister tensor buffer
  static LiteRtStatus UnregisterTensorBuffer(
      LiteRtDispatchDeviceContext device_context_handle,
      LiteRtTensorBufferHandle tensor_buffer_handle) {
    if (!device_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorDeviceContext*>(device_context_handle);
    return Traits::UnregisterTensorBuffer(context, tensor_buffer_handle);
  }

  // Get input/output requirements
  static LiteRtStatus GetInputRequirements(
      LiteRtDispatchInvocationContext invocation_context, int input_index,
      const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
    if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    // Default implementation - host memory only
    BufferRequirements reqs;
    reqs.supported_types = {kLiteRtTensorBufferTypeHostMemory};

    // Calculate buffer size based on tensor type
    size_t element_size = 0;
    switch (tensor_type->element_type) {
      case kLiteRtElementTypeFloat32:
        element_size = sizeof(float);
        break;
      case kLiteRtElementTypeInt32:
        element_size = sizeof(int32_t);
        break;
      case kLiteRtElementTypeInt8:
        element_size = sizeof(int8_t);
        break;
      case kLiteRtElementTypeUInt8:
        element_size = sizeof(uint8_t);
        break;
      case kLiteRtElementTypeFloat16:
        element_size = 2;
        break;
      case kLiteRtElementTypeInt16:
        element_size = sizeof(int16_t);
        break;
      case kLiteRtElementTypeInt64:
        element_size = sizeof(int64_t);
        break;
      default:
        return kLiteRtStatusErrorInvalidArgument;
    }

    size_t num_elements = 1;
    for (int i = 0; i < tensor_type->layout.rank; ++i) {
      if (tensor_type->layout.dimensions[i] > 0) {
        num_elements *= tensor_type->layout.dimensions[i];
      }
    }
    reqs.buffer_size = element_size * num_elements;

    auto result = reqs.ToLiteRtRequirements();
    if (!result) {
      return result.Error().Status();
    }

    *tensor_buffer_requirements = result.Value();
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetOutputRequirements(
      LiteRtDispatchInvocationContext invocation_context, int output_index,
      const LiteRtRankedTensorType* tensor_type,
      LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
    // Same as input requirements for most vendors
    return GetInputRequirements(invocation_context, output_index, tensor_type,
                                tensor_buffer_requirements);
  }

  // Create invocation context
  static LiteRtStatus InvocationContextCreate(
      LiteRtDispatchDeviceContext device_context_handle,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
      int num_inputs, int num_outputs,
      LiteRtDispatchInvocationContext* invocation_context_handle) {
    if (!device_context_handle || !exec_bytecode_buffer ||
        !invocation_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* device_context =
        reinterpret_cast<VendorDeviceContext*>(device_context_handle);
    auto result = Traits::CreateInvocationContext(
        device_context, exec_bytecode_buffer->base_addr,
        exec_bytecode_buffer->size, function_name);

    if (!result) {
      LITERT_LOG(LITERT_ERROR, "Failed to create invocation context: %s",
                 result.Error().Message().c_str());
      return result.Error().Status();
    }

    *invocation_context_handle =
        reinterpret_cast<LiteRtDispatchInvocationContext>(
            result.Value().release());
    return kLiteRtStatusOk;
  }

  // Destroy invocation context
  static LiteRtStatus InvocationContextDestroy(
      LiteRtDispatchInvocationContext invocation_context_handle) {
    if (!invocation_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorInvocationContext*>(invocation_context_handle);
    delete context;
    return kLiteRtStatusOk;
  }

  // Attach/detach buffers
  static LiteRtStatus AttachInput(
      LiteRtDispatchInvocationContext invocation_context_handle,
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
    if (!invocation_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorInvocationContext*>(invocation_context_handle);
    return context->AttachInput(graph_input_index, tensor_buffer_handle);
  }

  static LiteRtStatus AttachOutput(
      LiteRtDispatchInvocationContext invocation_context_handle,
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
    if (!invocation_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorInvocationContext*>(invocation_context_handle);
    return context->AttachOutput(graph_output_index, tensor_buffer_handle);
  }

  static LiteRtStatus DetachInput(
      LiteRtDispatchInvocationContext invocation_context_handle,
      int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
    if (!invocation_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorInvocationContext*>(invocation_context_handle);
    return context->DetachInput(graph_input_index, tensor_buffer_handle);
  }

  static LiteRtStatus DetachOutput(
      LiteRtDispatchInvocationContext invocation_context_handle,
      int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
    if (!invocation_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorInvocationContext*>(invocation_context_handle);
    return context->DetachOutput(graph_output_index, tensor_buffer_handle);
  }

  // Invoke
  static LiteRtStatus Invoke(
      LiteRtDispatchInvocationContext invocation_context_handle) {
    if (!invocation_context_handle) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* context =
        reinterpret_cast<VendorInvocationContext*>(invocation_context_handle);
    return context->Invoke();
  }

  // Metrics functions
  static LiteRtStatus StartMetricsCollection(
      LiteRtDispatchInvocationContext invocation_context, int detail_level) {
    if (!invocation_context) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    auto* context =
        reinterpret_cast<VendorInvocationContext*>(invocation_context);
    return context->StartMetricsCollection(detail_level);
  }

  static LiteRtStatus StopMetricsCollection(
      LiteRtDispatchInvocationContext invocation_context,
      LiteRtDispatchMetrics* metrics) {
    if (!invocation_context || !metrics) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    auto* context =
        reinterpret_cast<VendorInvocationContext*>(invocation_context);
    return context->StopMetricsCollection(metrics);
  }

  static LiteRtStatus GetNumMetrics(LiteRtDispatchMetrics metrics,
                                    int* num_metrics) {
    if (!metrics || !num_metrics) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    auto* m = reinterpret_cast<LiteRtDispatchMetricsT*>(metrics);
    *num_metrics = m->GetNumMetrics();
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetMetric(LiteRtDispatchMetrics metrics, int metric_index,
                                LiteRtMetric* metric) {
    if (!metrics || !metric) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    auto* m = reinterpret_cast<LiteRtDispatchMetricsT*>(metrics);
    *metric = m->GetMetric(metric_index);
    return kLiteRtStatusOk;
  }

  static LiteRtStatus DestroyMetrics(LiteRtDispatchMetrics metrics) {
    if (!metrics) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    auto* m = reinterpret_cast<LiteRtDispatchMetricsT*>(metrics);
    delete m;
    return kLiteRtStatusOk;
  }

  // Get the dispatch interface
  static constexpr LiteRtDispatchInterface GetInterface() {
    return LiteRtDispatchInterface{
        .initialize = Initialize,
        .get_vendor_id = GetVendorId,
        .get_build_id = GetBuildId,
        .get_capabilities = GetCapabilities,
        .device_context_create = DeviceContextCreate,
        .device_context_destroy = DeviceContextDestroy,
        .get_input_requirements = GetInputRequirements,
        .get_output_requirements = GetOutputRequirements,
        .register_tensor_buffer = RegisterTensorBuffer,
        .unregister_tensor_buffer = UnregisterTensorBuffer,
        .invocation_context_create = InvocationContextCreate,
        .invocation_context_destroy = InvocationContextDestroy,
        .attach_input = AttachInput,
        .attach_output = AttachOutput,
        .detach_input = DetachInput,
        .detach_output = DetachOutput,
        .invoke = Invoke,
        .start_metrics_collection = StartMetricsCollection,
        .stop_metrics_collection = StopMetricsCollection,
        .get_num_metrics = GetNumMetrics,
        .get_metric = GetMetric,
        .destroy_metrics = DestroyMetrics,
    };
  }

  // Get the dispatch API structure
  static LiteRtDispatchApi GetApi() {
    static const LiteRtDispatchInterface interface = GetInterface();
    return LiteRtDispatchApi{
        .version = {.major = LITERT_API_VERSION_MAJOR,
                    .minor = LITERT_API_VERSION_MINOR,
                    .patch = LITERT_API_VERSION_PATCH},
        .interface = const_cast<LiteRtDispatchInterface*>(&interface),
        .async_interface =
            const_cast<LiteRtDispatchAsyncInterface*>(GetAsyncInterface()),
        .graph_interface =
            const_cast<LiteRtDispatchGraphInterface*>(GetGraphInterface()),
    };
  }

 private:
  // Get async interface if supported
  static const LiteRtDispatchAsyncInterface* GetAsyncInterface() {
    if constexpr (Traits::kSupportsAsync) {
      return &GetAsyncInterfaceImpl();
    } else {
      return nullptr;
    }
  }

  // Get graph interface if supported
  static const LiteRtDispatchGraphInterface* GetGraphInterface() {
    if constexpr (Traits::kSupportsGraph) {
      return &GetGraphInterfaceImpl();
    } else {
      return nullptr;
    }
  }

 private:
  // Async interface implementation (only compiled if supported)
  template <typename T = Traits>
  static typename std::enable_if<T::kSupportsAsync,
                                 const LiteRtDispatchAsyncInterface&>::type
  GetAsyncInterfaceImpl() {
    static const LiteRtDispatchAsyncInterface async_interface = {
        .attach_input_event = Traits::AttachInputEvent,
        .invoke_async = Traits::InvokeAsync,
    };
    return async_interface;
  }

  // Graph interface implementation (only compiled if supported)
  template <typename T = Traits>
  static typename std::enable_if<T::kSupportsGraph,
                                 const LiteRtDispatchGraphInterface&>::type
  GetGraphInterfaceImpl() {
    static const LiteRtDispatchGraphInterface graph_interface = {
        .graph_create = Traits::GraphCreate,
        .graph_destroy = Traits::GraphDestroy,
        .add_node = Traits::AddNode,
        .add_edge = Traits::AddEdge,
        .connect_node_input = Traits::ConnectNodeInput,
        .connect_node_output = Traits::ConnectNodeOutput,
        .connect_graph_input = Traits::ConnectGraphInput,
        .connect_graph_output = Traits::ConnectGraphOutput,
        .load_executable = Traits::LoadExecutable,
        .unload_executable = Traits::UnloadExecutable,
        .assign_node_function = Traits::AssignNodeFunction,
        .annotate_graph = Traits::AnnotateGraph,
        .annotate_node = Traits::AnnotateNode,
        .annotate_edge = Traits::AnnotateEdge,
        .invocation_context_create_from_graph =
            Traits::InvocationContextCreateFromGraph,
    };
    return graph_interface;
  }
};

// Helper macro to define vendor dispatch entry point
#define DEFINE_VENDOR_DISPATCH_ENTRY_POINT(VendorTag)         \
  extern "C" {                                                \
  LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api) { \
    if (!api) {                                               \
      return kLiteRtStatusErrorInvalidArgument;               \
    }                                                         \
    static auto vendor_api =                                  \
        litert::vendors::VendorDispatch<VendorTag>::GetApi(); \
    *api = vendor_api;                                        \
    return kLiteRtStatusOk;                                   \
  }                                                           \
  }

}  // namespace litert::vendors

#endif  // LITERT_VENDORS_COMMON_VENDOR_DISPATCH_BASE_H_
