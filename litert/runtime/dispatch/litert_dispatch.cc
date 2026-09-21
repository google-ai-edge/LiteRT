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

#include "litert/vendors/c/litert_dispatch.h"

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_custom_tensor_buffer_handlers_def.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/internal/litert_scheduling_info.h"
#include "litert/c/internal/litert_tensor_buffer_registry.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_custom_tensor_buffer.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_profiler_types.h"
#include "litert/cc/internal/litert_shared_library.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/dynamic_loading.h"
#include "litert/core/util/perfetto_profiling.h"
#include "litert/core/version.h"
#include "litert/vendors/c/litert_dispatch_api.h"

#define INVOKE_FUNC(function, ...)                                      \
  const auto* basic_iface =                                             \
      TheBasicInterface_V1.load(std::memory_order_acquire);             \
  if (!basic_iface) {                                                   \
    LITERT_LOG(LITERT_ERROR, "Dispatch API basic interface not found"); \
    return kLiteRtStatusErrorRuntimeFailure;                            \
  }                                                                     \
  if (!LITERT_ABI_HAS_API(basic_iface, 1, function)) {                  \
    LITERT_LOG(LITERT_ERROR, #function " not found");                   \
    return kLiteRtStatusErrorUnsupported;                               \
  }                                                                     \
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API " #function);               \
  return basic_iface->function(__VA_ARGS__);

#define INVOKE_ASYNC_FUNC(function, ...)                                \
  const auto* async_iface =                                             \
      TheAsyncInterface_V1.load(std::memory_order_acquire);             \
  if (!async_iface) {                                                   \
    LITERT_LOG(LITERT_ERROR, "Dispatch API async interface not found"); \
    return kLiteRtStatusErrorUnsupported;                               \
  }                                                                     \
  if (!LITERT_ABI_HAS_API(async_iface, 1, function)) {                  \
    LITERT_LOG(LITERT_ERROR, #function " not found");                   \
    return kLiteRtStatusErrorUnsupported;                               \
  }                                                                     \
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API " #function);               \
  return async_iface->function(__VA_ARGS__);

#define INVOKE_GRAPH_FUNC(function, ...)                                \
  const auto* graph_iface =                                             \
      TheGraphInterface_V1.load(std::memory_order_acquire);             \
  if (!graph_iface) {                                                   \
    LITERT_LOG(LITERT_ERROR, "Dispatch API graph interface not found"); \
    return kLiteRtStatusErrorUnsupported;                               \
  }                                                                     \
  if (!LITERT_ABI_HAS_API(graph_iface, 1, function)) {                  \
    LITERT_LOG(LITERT_ERROR, #function " not found");                   \
    return kLiteRtStatusErrorUnsupported;                               \
  }                                                                     \
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API " #function);               \
  return graph_iface->function(__VA_ARGS__);

extern "C" {
// Set during static initialization by the vendor's Dispatch API implementation.
LiteRtStatus (*LiteRtStaticLinkedDispatchQueryInterface)(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version,
    LiteRtInterface* out_interface) = nullptr;
}

namespace {

litert::SharedLibrary* DispatchSharedLibrary = nullptr;
std::string* TheLoadedLibraryPath = nullptr;
std::atomic<bool> IsTheApiInitialized{false};
LiteRtDispatchQueryInterfaceT TheQueryInterface = nullptr;

std::atomic<const LiteRtDispatchInterface_V1*> TheBasicInterface_V1{nullptr};
std::atomic<const LiteRtDispatchAsyncInterface_V1*> TheAsyncInterface_V1{
    nullptr};
std::atomic<const LiteRtDispatchGraphInterface_V1*> TheGraphInterface_V1{
    nullptr};
std::atomic<const LiteRtCustomTensorBufferHandlersDef_V1*>
    TheCustomTensorBufferHandlers_V1{nullptr};

LiteRtStatus RegisterCustomTensorBufferHandlers(
    const LiteRtCustomTensorBufferHandlersDef_V1* handlers_def,
    LiteRtEnvironment env) {
  if (handlers_def != nullptr) {
    if (!LITERT_ABI_HAS_MEMBER(handlers_def, 1, device_tag) ||
        !LITERT_ABI_HAS_MEMBER(handlers_def, 1, queue_tag) ||
        !LITERT_ABI_HAS_MEMBER(handlers_def, 1, num_supported_buffer_types) ||
        !LITERT_ABI_HAS_MEMBER(handlers_def, 1, supported_buffer_types) ||
        !LITERT_ABI_HAS_API(handlers_def, 1, create_func) ||
        !LITERT_ABI_HAS_API(handlers_def, 1, destroy_func) ||
        !LITERT_ABI_HAS_API(handlers_def, 1, lock_func) ||
        !LITERT_ABI_HAS_API(handlers_def, 1, unlock_func)) {
      return kLiteRtStatusErrorWrongVersion;
    }
    ClearCustomTensorBuffer clear_func =
        LITERT_ABI_HAS_API(handlers_def, 1, clear_func)
            ? handlers_def->clear_func
            : nullptr;
    ImportCustomTensorBuffer import_func =
        LITERT_ABI_HAS_API(handlers_def, 1, import_func)
            ? handlers_def->import_func
            : nullptr;
    for (size_t i = 0;
         i < handlers_def->num_supported_buffer_types &&
         i < LITERT_CUSTOM_BUFFER_HANDLERS_DEF_MAX_SUPPORTED_BUFFER_TYPES;
         ++i) {
      LITERT_RETURN_IF_ERROR(LiteRtRegisterTensorBufferHandlers(
          env, handlers_def->supported_buffer_types[i],
          handlers_def->create_func, handlers_def->destroy_func,
          handlers_def->lock_func, handlers_def->unlock_func, clear_func,
          import_func, handlers_def->device_tag, handlers_def->queue_tag));
    }
  }
  return kLiteRtStatusOk;
}

litert::Expected<std::string> GetSharedLibraryPath(
    LiteRtEnvironmentOptions env_options) {
  std::vector<std::string> dispatch_lib_paths;
  LiteRtAny dispatch_lib_dir;
  auto status = LiteRtGetEnvironmentOptionsValue(
      env_options, kLiteRtEnvOptionTagDispatchLibraryDir, &dispatch_lib_dir);
  if (status != kLiteRtStatusOk) {
    return litert::Error(status, "Dispatch library directory option not set.");
  }
  litert::internal::FindLiteRtDispatchSharedLibs(dispatch_lib_dir.str_value,
                                                 dispatch_lib_paths);
  if (dispatch_lib_paths.empty()) {
    return litert::Error(kLiteRtStatusErrorNotFound,
                         "Dispatch library not found.");
  }
  if (dispatch_lib_paths.size() > 1) {
    LITERT_LOG(LITERT_WARNING,
               "Multiple dispatch libraries found, loading the first one: %s",
               dispatch_lib_paths.front().c_str());
  }
  return dispatch_lib_paths.front();
}

}  // namespace

LiteRtStatus LiteRtDispatchInitialize(
    const LiteRtRuntimeContext* runtime_context, LiteRtEnvironment env,
    LiteRtOptions options) {
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API Initialization");

  LiteRtEnvironmentOptions env_options;
  LITERT_RETURN_IF_ERROR(LiteRtGetEnvironmentOptions(env, &env_options));

  // If already initialized and static override has not changed, initialize the
  // new environment against the already-negotiated dispatch interface.
  if (IsTheApiInitialized.load(std::memory_order_relaxed) &&
      (LiteRtStaticLinkedDispatchQueryInterface == nullptr ||
       TheQueryInterface == LiteRtStaticLinkedDispatchQueryInterface)) {
    if (LiteRtStaticLinkedDispatchQueryInterface == nullptr &&
        TheLoadedLibraryPath != nullptr) {
      auto requested_path = GetSharedLibraryPath(env_options);
      if (requested_path && *requested_path != *TheLoadedLibraryPath) {
        LITERT_LOG(LITERT_WARNING,
                   "Dispatch API already initialized from '%s'; ignoring "
                   "different requested library '%s'",
                   TheLoadedLibraryPath->c_str(), requested_path->c_str());
      }
    }
    LITERT_RETURN_IF_ERROR(RegisterCustomTensorBufferHandlers(
        TheCustomTensorBufferHandlers_V1.load(std::memory_order_relaxed), env));
    const auto* basic = TheBasicInterface_V1.load(std::memory_order_relaxed);
    if (!LITERT_ABI_HAS_API(basic, 1, initialize)) {
      return kLiteRtStatusErrorRuntimeFailure;
    }
    return basic->initialize(runtime_context, env, options);
  }

  // 1. Resolve QueryInterface function (static or dynamic)
  LiteRtDispatchQueryInterfaceT query_interface = nullptr;
  std::string loaded_path;
  if (LiteRtStaticLinkedDispatchQueryInterface != nullptr) {
    query_interface = LiteRtStaticLinkedDispatchQueryInterface;
  } else {
    LITERT_ASSIGN_OR_RETURN(loaded_path, GetSharedLibraryPath(env_options));

    LITERT_LOG(LITERT_INFO, "Loading shared library: %s", loaded_path.c_str());

    if (!DispatchSharedLibrary) {
      DispatchSharedLibrary = new litert::SharedLibrary();
    }

    LITERT_ASSIGN_OR_RETURN(
        *DispatchSharedLibrary,
        litert::SharedLibrary::Load(
            loaded_path, litert::RtldFlags::Now().Local().NoDelete()));

    auto query_res =
        DispatchSharedLibrary->LookupSymbol<LiteRtDispatchQueryInterfaceT>(
            kLiteRtDispatchQueryInterface.data());
    if (!query_res) {
      if (DispatchSharedLibrary->LookupSymbol<void*>(
              "LiteRtDispatchInitialize")) {
        LITERT_LOG(LITERT_WARNING,
                   "Vendor library '%s' exports legacy pre-Option-B symbol "
                   "'LiteRtDispatchInitialize'. Please recompile vendor "
                   "library with LiteRT Option B ABI support.",
                   loaded_path.c_str());
      }
      return query_res.Error().Status();
    }
    query_interface = *query_res;
  }

  // 2. Negotiate Basic Interface into local temporaries first (prevent partial
  // initialization poisoning if subsequent steps fail).
  LiteRtApiVersion runtime_ver = {LITERT_DISPATCH_ABI_VERSION_MAJOR,
                                  LITERT_DISPATCH_ABI_VERSION_MINOR,
                                  LITERT_DISPATCH_ABI_VERSION_PATCH};
  LiteRtInterface basic_iface = nullptr;
  LiteRtStatus status = litert::internal::NegotiateInterface(
      query_interface, kLiteRtInterfaceBasic, runtime_ver,
      /*expected_abi_major=*/LITERT_DISPATCH_ABI_VERSION_MAJOR, &basic_iface);
  if (status != kLiteRtStatusOk || basic_iface == nullptr) {
    LITERT_LOG(LITERT_ERROR, "Failed to negotiate basic interface version");
    return kLiteRtStatusErrorWrongVersion;
  }
  const auto* local_basic =
      reinterpret_cast<const LiteRtDispatchInterface_V1*>(basic_iface);
  if (!LITERT_ABI_HAS_API(local_basic, 1, initialize)) {
    LITERT_LOG(LITERT_ERROR, "Dispatch initialize entry point not found");
    return kLiteRtStatusErrorWrongVersion;
  }

  // 3. Query optional sub-interfaces into local temporaries
  const LiteRtDispatchAsyncInterface_V1* local_async = nullptr;
  LiteRtInterface async_iface = nullptr;
  if (litert::internal::NegotiateInterface(
          query_interface, kLiteRtInterfaceAsync, runtime_ver,
          /*expected_abi_major=*/LITERT_DISPATCH_ABI_VERSION_MAJOR,
          &async_iface) == kLiteRtStatusOk) {
    local_async =
        reinterpret_cast<const LiteRtDispatchAsyncInterface_V1*>(async_iface);
  }

  const LiteRtDispatchGraphInterface_V1* local_graph = nullptr;
  LiteRtInterface graph_iface = nullptr;
  if (litert::internal::NegotiateInterface(
          query_interface, kLiteRtInterfaceGraph, runtime_ver,
          /*expected_abi_major=*/LITERT_DISPATCH_ABI_VERSION_MAJOR,
          &graph_iface) == kLiteRtStatusOk) {
    local_graph =
        reinterpret_cast<const LiteRtDispatchGraphInterface_V1*>(graph_iface);
  }

  const LiteRtCustomTensorBufferHandlersDef_V1* local_handlers = nullptr;
  LiteRtInterface handlers_iface = nullptr;
  if (litert::internal::NegotiateInterface(
          query_interface, kLiteRtInterfaceCustomTensorBufferHandlers,
          runtime_ver, /*expected_abi_major=*/LITERT_DISPATCH_ABI_VERSION_MAJOR,
          &handlers_iface) == kLiteRtStatusOk) {
    local_handlers =
        reinterpret_cast<const LiteRtCustomTensorBufferHandlersDef_V1*>(
            handlers_iface);
  }

  // 4. Register custom tensor buffer handlers and initialize the environment.
  LITERT_RETURN_IF_ERROR(
      RegisterCustomTensorBufferHandlers(local_handlers, env));

  status = local_basic->initialize(runtime_context, env, options);
  if (status != kLiteRtStatusOk) {
    return status;
  }

  // 5. Commit negotiated interfaces atomically after full initialization
  // succeeds.
  if (!loaded_path.empty()) {
    if (!TheLoadedLibraryPath) {
      TheLoadedLibraryPath = new std::string(loaded_path);
    } else {
      *TheLoadedLibraryPath = loaded_path;
    }
  }
  TheQueryInterface = query_interface;
  TheAsyncInterface_V1.store(local_async, std::memory_order_release);
  TheGraphInterface_V1.store(local_graph, std::memory_order_release);
  TheCustomTensorBufferHandlers_V1.store(local_handlers,
                                         std::memory_order_release);
  TheBasicInterface_V1.store(local_basic, std::memory_order_release);
  IsTheApiInitialized.store(true, std::memory_order_release);
  return kLiteRtStatusOk;
}

namespace litert::internal {

void ResetDispatchForTest() {
  IsTheApiInitialized.store(false, std::memory_order_release);
  TheBasicInterface_V1.store(nullptr, std::memory_order_release);
  TheAsyncInterface_V1.store(nullptr, std::memory_order_release);
  TheGraphInterface_V1.store(nullptr, std::memory_order_release);
  TheCustomTensorBufferHandlers_V1.store(nullptr, std::memory_order_release);
  TheQueryInterface = nullptr;
  LiteRtStaticLinkedDispatchQueryInterface = nullptr;
  if (DispatchSharedLibrary) {
    delete DispatchSharedLibrary;
    DispatchSharedLibrary = nullptr;
  }
  if (TheLoadedLibraryPath) {
    delete TheLoadedLibraryPath;
    TheLoadedLibraryPath = nullptr;
  }
}

}  // namespace litert::internal

LiteRtStatus LiteRtDispatchGetApiVersion(LiteRtApiVersion* api_version) {
  if (!api_version) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* basic = TheBasicInterface_V1.load(std::memory_order_acquire);
  if (!basic) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  *api_version = {basic->abi_header.major_version,
                  basic->abi_header.minor_version, 0};
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDispatchGetVendorId(const char** vendor_id) {
  if (!vendor_id) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *vendor_id = nullptr;
  INVOKE_FUNC(get_vendor_id, vendor_id);
}

LiteRtStatus LiteRtDispatchGetBuildId(const char** build_id) {
  if (!build_id) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *build_id = nullptr;
  INVOKE_FUNC(get_build_id, build_id);
}

LiteRtStatus LiteRtDispatchGetCapabilities(int* capabilities) {
  if (!capabilities) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *capabilities = 0;
  INVOKE_FUNC(get_capabilities, capabilities);
}

LiteRtStatus LiteRtDispatchDeviceContextCreate(
    const LiteRtRuntimeContext* runtime_context, LiteRtOptions options,
    LiteRtDispatchDeviceContext* device_context) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  *device_context = nullptr;
  INVOKE_FUNC(device_context_create, runtime_context, options, device_context);
}

LiteRtStatus LiteRtDispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(device_context_destroy, device_context);
}

LiteRtStatus LiteRtDispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_input_requirements, invocation_context, input_index,
              tensor_type, tensor_buffer_requirements);
}

LiteRtStatus LiteRtDispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_output_requirements, invocation_context, output_index,
              tensor_type, tensor_buffer_requirements);
}

LiteRtStatus LiteRtDispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (!device_context || !tensor_buffer || !tensor_buffer_handle) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(register_tensor_buffer, device_context, tensor_buffer,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(unregister_tensor_buffer, device_context, tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchInvocationContextCreate(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (!device_context || !exec_bytecode_buffer || !invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invocation_context_create, runtime_context, device_context,
              exec_type, exec_bytecode_buffer, function_name, num_inputs,
              num_outputs, invocation_context);
}

LiteRtStatus LiteRtDispatchInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invocation_context_destroy, invocation_context);
}

LiteRtStatus LiteRtDispatchInvocationContextSetOptions(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* basic = TheBasicInterface_V1.load(std::memory_order_acquire);
  if (!basic) {
    LITERT_LOG(LITERT_ERROR, "Dispatch API basic interface not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (!LITERT_ABI_HAS_API(basic, 1, invocation_context_set_options)) {
    return kLiteRtStatusErrorUnsupported;
  }
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API invocation_context_set_options");
  return basic->invocation_context_set_options(invocation_context, options);
}

LiteRtStatus LiteRtDispatchInvocationContextSetSchedulingInfo(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* basic = TheBasicInterface_V1.load(std::memory_order_acquire);
  if (!basic) {
    return kLiteRtStatusErrorUnsupported;
  }
  if (!LITERT_ABI_HAS_API(basic, 1, invocation_context_set_scheduling_info)) {
    return kLiteRtStatusErrorUnsupported;
  }
  LITERT_PERFETTO_TRACE_EVENT(
      "Dispatch API invocation_context_set_scheduling_info");
  return basic->invocation_context_set_scheduling_info(invocation_context,
                                                       scheduling_info);
}

LiteRtStatus LiteRtDispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(attach_input, invocation_context, graph_input_index,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(attach_output, invocation_context, graph_output_index,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(detach_input, invocation_context, graph_input_index,
              tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(detach_output, invocation_context, graph_output_index,
              tensor_buffer_handle);
}

#if defined(LITERT_ENABLE_FABRIC_INTEGRATION)
LiteRtStatus LiteRtDispatchAttachEdgeBuffer(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchEdgeId edge_id,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* basic_iface =
      TheBasicInterface_V1.load(std::memory_order_acquire);
  if (!LITERT_ABI_HAS_API(basic_iface, 1, attach_edge_buffer)) {
    LITERT_LOG(LITERT_ERROR, "attach_edge_buffer not found");
    return kLiteRtStatusErrorUnsupported;
  }
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API attach_edge_buffer");
  return basic_iface->attach_edge_buffer(invocation_context, edge_id,
                                         tensor_buffer_handle);
}

LiteRtStatus LiteRtDispatchDetachEdgeBuffer(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchEdgeId edge_id,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* basic_iface =
      TheBasicInterface_V1.load(std::memory_order_acquire);
  if (!LITERT_ABI_HAS_API(basic_iface, 1, detach_edge_buffer)) {
    LITERT_LOG(LITERT_ERROR, "detach_edge_buffer not found");
    return kLiteRtStatusErrorUnsupported;
  }
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API detach_edge_buffer");
  return basic_iface->detach_edge_buffer(invocation_context, edge_id,
                                         tensor_buffer_handle);
}
#endif  // defined(LITERT_ENABLE_FABRIC_INTEGRATION)

LiteRtStatus LiteRtDispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invoke, invocation_context);
}

LiteRtStatus LiteRtDispatchCheckRuntimeCompatibility(
    LiteRtApiVersion api_version, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  INVOKE_FUNC(check_runtime_compatibility, api_version, env, options);
}

LiteRtStatus LiteRtDispatchGetHooks(LiteRtDispatchDeviceContext device_context,
                                    LiteRtHook* hook, void** user_data) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!hook) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* basic = TheBasicInterface_V1.load(std::memory_order_acquire);
  if (!basic) {
    LITERT_LOG(LITERT_ERROR, "Dispatch API interface not found");
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (!LITERT_ABI_HAS_API(basic, 1, get_hooks)) {
    *hook = nullptr;
    if (user_data) {
      *user_data = nullptr;
    }
    return kLiteRtStatusOk;
  }
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API get_hooks");
  return basic->get_hooks(device_context, hook, user_data);
}

LiteRtStatus LiteRtDispatchStartMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context, int detail_level) {
  if (!invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  } else if (detail_level < 0) {
    LITERT_LOG(LITERT_ERROR, "Invalid detail level");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(start_metrics_collection, invocation_context, detail_level);
}

LiteRtStatus LiteRtDispatchStopMetricsCollection(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchMetrics* metrics) {
  if (!invocation_context || !metrics) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(stop_metrics_collection, invocation_context, metrics);
}

LiteRtStatus LiteRtDispatchGetNumMetrics(LiteRtDispatchMetrics metrics,
                                         int* num_metrics) {
  if (!metrics || !num_metrics) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_num_metrics, metrics, num_metrics);
}

LiteRtStatus LiteRtDispatchGetMetric(LiteRtDispatchMetrics metrics,
                                     int metric_index, LiteRtMetric* metric) {
  if (!metrics || !metric) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_metric, metrics, metric_index, metric);
}

LiteRtStatus LiteRtDispatchDestroyMetrics(LiteRtDispatchMetrics metrics) {
  if (!metrics) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(destroy_metrics, metrics);
}

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus LiteRtDispatchAttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event) {
  if (!invocation_context || !input_event) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_ASYNC_FUNC(attach_input_event, invocation_context, graph_input_index,
                    input_event);
}

LiteRtStatus LiteRtDispatchInvokeAsync(
    LiteRtDispatchInvocationContext invocation_context, int num_output_events,
    LiteRtEvent* output_events) {
  if (!invocation_context || !output_events) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_ASYNC_FUNC(invoke_async, invocation_context, num_output_events,
                    output_events);
}

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus LiteRtDispatchGraphCreate(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph* graph) {
  if (!device_context || !graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(graph_create, device_context, graph);
}

LiteRtStatus LiteRtDispatchGraphDestroy(LiteRtDispatchGraph graph) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(graph_destroy, graph);
}

LiteRtStatus LiteRtDispatchAddNode(LiteRtDispatchGraph graph,
                                   LiteRtDispatchNodeId node_id,
                                   LiteRtDispatchNodeType node_type) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(add_node, graph, node_id, node_type);
}

LiteRtStatus LiteRtDispatchAddEdge(LiteRtDispatchGraph graph,
                                   LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(add_edge, graph, edge_id);
}

LiteRtStatus LiteRtDispatchConnectNodeInput(LiteRtDispatchGraph graph,
                                            LiteRtDispatchNodeId node_id,
                                            int input_index,
                                            LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_node_input, graph, node_id, input_index, edge_id);
}

LiteRtStatus LiteRtDispatchConnectNodeOutput(LiteRtDispatchGraph graph,
                                             LiteRtDispatchNodeId node_id,
                                             int output_index,
                                             LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_node_output, graph, node_id, output_index, edge_id);
}

LiteRtStatus LiteRtDispatchConnectGraphInput(LiteRtDispatchGraph graph,
                                             int input_index,
                                             LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_graph_input, graph, input_index, edge_id);
}

LiteRtStatus LiteRtDispatchConnectGraphOutput(LiteRtDispatchGraph graph,
                                              int output_index,
                                              LiteRtDispatchEdgeId edge_id) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_graph_output, graph, output_index, edge_id);
}

LiteRtStatus LiteRtDispatchLoadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType type, const LiteRtMemBuffer* bytecode_buffer,
    LiteRtDispatchExecutableHandle* exec_handle) {
  if (!device_context || !bytecode_buffer || !exec_handle) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(load_executable, device_context, type, bytecode_buffer,
                    exec_handle);
}

LiteRtStatus LiteRtDispatchUnloadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(unload_executable, device_context, exec_handle);
}

#if defined(LITERT_ENABLE_FABRIC_INTEGRATION)
LiteRtStatus LiteRtDispatchGetScratchpadRequirements(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name,
    LiteRtTensorBufferRequirements* scratchpad_requirements) {
  if (!device_context || !scratchpad_requirements) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* graph_iface =
      TheGraphInterface_V1.load(std::memory_order_acquire);
  if (!LITERT_ABI_HAS_API(graph_iface, 1, get_scratchpad_requirements)) {
    LITERT_LOG(LITERT_ERROR, "get_scratchpad_requirements not found");
    return kLiteRtStatusErrorUnsupported;
  }
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API get_scratchpad_requirements");
  return graph_iface->get_scratchpad_requirements(
      device_context, exec_handle, function_name, scratchpad_requirements);
}

LiteRtStatus LiteRtDispatchAttachScratchpadBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name,
    LiteRtTensorBufferHandle scratchpad_buffer_handle) {
  if (!device_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* graph_iface =
      TheGraphInterface_V1.load(std::memory_order_acquire);
  if (!LITERT_ABI_HAS_API(graph_iface, 1, attach_scratchpad_buffer)) {
    LITERT_LOG(LITERT_ERROR, "attach_scratchpad_buffer not found");
    return kLiteRtStatusErrorUnsupported;
  }
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API attach_scratchpad_buffer");
  return graph_iface->attach_scratchpad_buffer(
      device_context, exec_handle, function_name, scratchpad_buffer_handle);
}
#endif  // defined(LITERT_ENABLE_FABRIC_INTEGRATION)

LiteRtStatus LiteRtDispatchAssignNodeFunction(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(assign_node_function, graph, node_id, exec_handle,
                    function_name);
}

LiteRtStatus LiteRtDispatchAnnotateGraph(LiteRtDispatchGraph graph,
                                         const char* key, const char* value) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_graph, graph, key, value);
}

LiteRtStatus LiteRtDispatchAnnotateNode(LiteRtDispatchGraph graph,
                                        LiteRtDispatchNodeId node_id,
                                        const char* key, const char* value) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_node, graph, node_id, key, value);
}

LiteRtStatus LiteRtDispatchAnnotateEdge(LiteRtDispatchGraph graph,
                                        LiteRtDispatchEdgeId edge_id,
                                        const char* key, const char* value) {
  if (!graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_edge, graph, edge_id, key, value);
}

LiteRtStatus LiteRtDispatchInvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (!device_context || !graph || !invocation_context) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(invocation_context_create_from_graph, device_context, graph,
                    invocation_context);
}

LiteRtStatus LiteRtDispatchInvocationContextGetGraph(
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchGraph* graph) {
  if (!invocation_context || !graph) {
    LITERT_LOG(LITERT_ERROR, "Null input");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto* graph_iface =
      TheGraphInterface_V1.load(std::memory_order_acquire);
  if (!graph_iface) {
    LITERT_LOG(LITERT_ERROR, "invocation_context_get_graph not found");
    return kLiteRtStatusErrorUnsupported;
  }
  if (!LITERT_ABI_HAS_API(graph_iface, 1, invocation_context_get_graph)) {
    LITERT_LOG(LITERT_ERROR, "invocation_context_get_graph not found");
    return kLiteRtStatusErrorUnsupported;
  }
  LITERT_PERFETTO_TRACE_EVENT("Dispatch API invocation_context_get_graph");
  return graph_iface->invocation_context_get_graph(invocation_context, graph);
}
