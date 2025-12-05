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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "litert/vendors/google_tensor/dispatch/sb_api.h"

// Places a log message in the standard output file.
#define THR_LOG_INFO(...) fprintf(stdout, __VA_ARGS__)

// Places a log message in the standard error file.
#define THR_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)

namespace hardware::gchips::hetero_runtime::star_ship {
namespace {

// SouthBound shared object handle.
static void* gSouthBoundLibHandle = nullptr;

// Dynamically resolves a symbol via the loaded SouthBound shared object and
// stores the result in a pointer.
#define THR_RESOLVE_SYM(ptr, sym) \
  ptr = reinterpret_cast<decltype(&sym)>(dlsym(gSouthBoundLibHandle, #sym))

// A convenience struct for holding pointers to SouthBound symbols.
//
// These pointers are initialized on the first call to `thrInitialize`.
struct ThrFunctions {
  decltype(&thrInitialize) thr_initialize = nullptr;

  decltype(&thrGetVendorApiVersion) thr_get_vendor_api_version = nullptr;
  decltype(&thrGetVendorId) thr_get_vendor_id = nullptr;

  decltype(&thrContextCreate) thr_context_create = nullptr;
  decltype(&thrContextDelete) thr_context_delete = nullptr;

  decltype(&thrGraphCreate) thr_graph_create = nullptr;
  decltype(&thrGraphDelete) thr_graph_delete = nullptr;

  decltype(&thrGraphAddEdge) thr_graph_add_edge = nullptr;
  decltype(&thrGraphAddSqNode) thr_graph_add_sq_node = nullptr;

  decltype(&thrGraphConnectNodeInput) thr_graph_connect_node_input = nullptr;
  decltype(&thrGraphConnectNodeOutput) thr_graph_connect_node_output = nullptr;

  decltype(&thrGraphSetInputEdge) thr_graph_set_input_edge = nullptr;
  decltype(&thrGraphSetOutputEdge) thr_graph_set_output_edge = nullptr;

  decltype(&thrGraphAnnotateGraph) thr_graph_annotate_graph = nullptr;
  decltype(&thrGraphAnnotateEdge) thr_graph_annotate_edge = nullptr;
  decltype(&thrGraphAnnotateNode) thr_graph_annotate_node = nullptr;

  decltype(&thrLoadSqContainer) thr_load_sq_container = nullptr;
  decltype(&thrLoadSqContainerFd) thr_load_sq_container_fd = nullptr;
  decltype(&thrLoadSqContainerFdWithLoadingType)
      thr_load_sq_container_fd_with_loading_type = nullptr;
  decltype(&thrLoadSqContainerFdWithOffset)
      thr_load_sq_container_fd_with_offset = nullptr;
  decltype(&thrLoadSqContainerFile) thr_load_sq_container_file = nullptr;
  decltype(&thrUnloadSqContainer) thr_unload_sq_container = nullptr;
  decltype(&thrPinSqContainer) thr_pin_sq_container = nullptr;
  decltype(&thrUnpinSqContainer) thr_unpin_sq_container = nullptr;

  decltype(&thrGraphAssignSq) thr_graph_assign_sq = nullptr;
  decltype(&thrSqQueryScratchPad) thr_sq_query_scratch_pad = nullptr;
  decltype(&thrSqAttachScratchPadBuffer) thr_sq_attach_scratch_pad_buffer =
      nullptr;

  decltype(&thrRegisterBuffer) thr_register_buffer = nullptr;
  decltype(&thrRegisterBufferWithOffset) thr_register_buffer_with_offset =
      nullptr;
  decltype(&thrUnregisterBuffer) thr_unregister_buffer = nullptr;

  decltype(&thrRegisterFence) thr_register_fence = nullptr;
  decltype(&thrUnregisterFence) thr_unregister_fence = nullptr;
  decltype(&thrFenceGetDupFd) thr_fence_get_dup_fd = nullptr;

  decltype(&thrInvocationContextGet) thr_invocation_context_get = nullptr;
  decltype(&thrInvocationContextDelete) thr_invocation_context_delete = nullptr;

  decltype(&thrInvocationContextAttachBuffer)
      thr_invocation_context_attach_buffer = nullptr;
  decltype(&thrInvocationContextDetachBuffer)
      thr_invocation_context_detach_buffer = nullptr;

  decltype(&thrInvocationContextPrepareForInvoke)
      thr_invocation_context_prepare_for_invoke = nullptr;
  decltype(&thrInvocationContextPrepareForInvoke2)
      thr_invocation_context_prepare_for_invoke2 = nullptr;
  decltype(&thrInvocationContextInvokeOnce) thr_invocation_context_invoke_once =
      nullptr;
  decltype(&thrInvocationContextWait) thr_invocation_context_wait = nullptr;
  decltype(&thrInvocationContextCancel) thr_invocation_context_cancel = nullptr;

  decltype(&thrInvocationContextAttachInputBufferSyncFence)
      thr_invocation_context_attach_input_buffer_sync_fence = nullptr;
  decltype(&thrInvocationContextAttachInputBufferFence)
      thr_invocation_context_attach_input_buffer_fence = nullptr;
  decltype(&thrInvocationContextDetachInputBufferSyncFence)
      thr_invocation_context_detach_input_buffer_sync_fence = nullptr;
  decltype(&thrInvocationContextDetachInputBufferFence)
      thr_invocation_context_detach_input_buffer_fence = nullptr;
  decltype(&thrInvocationContextGetOutputBufferSyncFence)
      thr_invocation_context_get_output_buffer_sync_fence = nullptr;
  decltype(&thrInvocationContextGetOutputBufferFence)
      thr_invocation_context_get_output_buffer_fence = nullptr;

  decltype(&thrInvocationContextQueryNodeScratchPad)
      thr_invocation_context_query_node_scratch_pad = nullptr;
  decltype(&thrInvocationContextAttachScratchPadBuffer)
      thr_invocation_context_attach_scratch_pad_buffer = nullptr;
  decltype(&thrInvocationContextGetMetrics) thr_invocation_context_get_metrics =
      nullptr;
  decltype(&thrInvocationContextResetMetrics)
      thr_invocation_context_reset_metrics = nullptr;
  decltype(&thrInvocationContextStartMetricsCollection)
      thr_invocation_context_start_metrics_collection = nullptr;
  decltype(&thrInvocationContextStopMetricsCollection)
      thr_invocation_context_stop_metrics_collection = nullptr;

  decltype(&thrVendorSetSystemAttributeStr)
      thr_vendor_set_system_attribute_str = nullptr;
  decltype(&thrVendorSetSystemAttributeInt64)
      thr_vendor_set_system_attribute_int64 = nullptr;

  decltype(&thrGraphAddSqNodeWithInterfaceBindingMode)
      thr_graph_add_sq_node_with_interface_binding_mode = nullptr;
  decltype(&thrGraphConnectNodeInputWithPortName)
      thr_graph_connect_node_input_with_port_name = nullptr;
  decltype(&thrGraphConnectNodeOutputWithPortName)
      thr_graph_connect_node_output_with_port_name = nullptr;
  decltype(&thrGraphConnectNodeInputWithPortIndex)
      thr_graph_connect_node_input_with_port_index = nullptr;
  decltype(&thrGraphConnectNodeOutputWithPortIndex)
      thr_graph_connect_node_output_with_port_index = nullptr;
};

// Stores the dynamically resolved SouthBound symbols.
static ThrFunctions* gSouthBoundFns = nullptr;

// Calls a dynamically resolved SouthBound function if it is available, else
// logs an error message and returns if the function is not available.
#define THR_CALL_DYN_FN(fn, ...)                                  \
  [&]() {                                                         \
    using ReturnType = decltype(gSouthBoundFns->fn(__VA_ARGS__)); \
                                                                  \
    if (gSouthBoundFns == nullptr) {                              \
      THR_LOG_ERROR("SouthBound is not initialized");             \
      return GetValForSymUnavailable<ReturnType>();               \
    }                                                             \
    if (gSouthBoundFns->fn == nullptr) {                          \
      THR_LOG_ERROR("Symbol for '%s' is not available", #fn);     \
      return GetValForSymUnavailable<ReturnType>();               \
    }                                                             \
                                                                  \
    return gSouthBoundFns->fn(__VA_ARGS__);                       \
  }()

// A companion function for `THR_CALL_DYN_FN` that is responsible for returning
// a value of type `T` when a symbol that returns a value of type `T` is not
// available.
template <typename T>
T GetValForSymUnavailable() {
  if constexpr (std::is_same_v<T, ThrStatus>) {
    return kThrStatusFail;
  } else {
    // In the "default" case, just return a default-initialized value.
    return T{};
  }
}

constexpr char kDefaultSouthBoundLibPath[] =
#ifdef __ANDROID__
    "/vendor/lib64/libedgetpu_litert.so";
#else
    "third_party/darwinn/litert/dispatch/libedgetpu_litert.so";
#endif  // __ANDROID__

// Defining this environment variable is a means to override
// `kDefaultSouthBoundLibPath` for debug builds.
#ifndef NDEBUG
constexpr char kSouthBoundLibPathEnvVar[] = "SOUTH_BOUND_LIB_PATH";
#endif  // NDEBUG

// Dynamically loads SouthBound symbols on the first call, else a no-op.
//
// NOTE: this function is thread-compatible.
ThrStatus LoadSouthBoundSyms() {
  if (gSouthBoundFns != nullptr) {
    return kThrStatusSuccess;
  }

  const char* south_bound_lib_path = kDefaultSouthBoundLibPath;
#ifndef NDEBUG
  if (char* path = getenv(kSouthBoundLibPathEnvVar)) {
    south_bound_lib_path = path;
  }
#endif  // NDEBUG

  dlerror();
  gSouthBoundLibHandle = dlopen(south_bound_lib_path, RTLD_NOW | RTLD_GLOBAL);
  if (gSouthBoundLibHandle == nullptr) {
    THR_LOG_ERROR("Failed to dlopen '%s': %s", south_bound_lib_path, dlerror());
    return kThrStatusFail;
  }

  gSouthBoundFns = new ThrFunctions();

  // Resolve all available symbols from the SouthBound shared library.
  //
  // NOTE: in most cases, this file will be more recent than the SouthBound
  // shared object in the system image, resulting in some symbols failing to
  // resolve - i.e. have an address of 0.
  //
  // Please consult SouthBound's semantic versioning documentation to determine
  // which symbols are available in a particular version of the library.

  THR_RESOLVE_SYM(gSouthBoundFns->thr_initialize, thrInitialize);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_get_vendor_api_version,
                  thrGetVendorApiVersion);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_get_vendor_id, thrGetVendorId);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_context_create, thrContextCreate);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_context_delete, thrContextDelete);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_create, thrGraphCreate);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_delete, thrGraphDelete);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_add_edge, thrGraphAddEdge);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_add_sq_node, thrGraphAddSqNode);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_connect_node_input,
                  thrGraphConnectNodeInput);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_connect_node_output,
                  thrGraphConnectNodeOutput);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_set_input_edge,
                  thrGraphSetInputEdge);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_set_output_edge,
                  thrGraphSetOutputEdge);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_annotate_graph,
                  thrGraphAnnotateGraph);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_annotate_edge,
                  thrGraphAnnotateEdge);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_annotate_node,
                  thrGraphAnnotateNode);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_load_sq_container, thrLoadSqContainer);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_load_sq_container_fd,
                  thrLoadSqContainerFd);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_load_sq_container_fd_with_offset,
                  thrLoadSqContainerFdWithOffset);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_load_sq_container_file,
                  thrLoadSqContainerFile);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_unload_sq_container,
                  thrUnloadSqContainer);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_pin_sq_container, thrPinSqContainer);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_unpin_sq_container, thrUnpinSqContainer);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_assign_sq, thrGraphAssignSq);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_sq_query_scratch_pad,
                  thrSqQueryScratchPad);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_sq_attach_scratch_pad_buffer,
                  thrSqAttachScratchPadBuffer);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_register_buffer, thrRegisterBuffer);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_register_buffer_with_offset,
                  thrRegisterBufferWithOffset);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_unregister_buffer, thrUnregisterBuffer);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_register_fence, thrRegisterFence);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_unregister_fence, thrUnregisterFence);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_fence_get_dup_fd, thrFenceGetDupFd);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_get,
                  thrInvocationContextGet);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_delete,
                  thrInvocationContextDelete);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_attach_buffer,
                  thrInvocationContextAttachBuffer);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_detach_buffer,
                  thrInvocationContextDetachBuffer);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_prepare_for_invoke,
                  thrInvocationContextPrepareForInvoke);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_prepare_for_invoke2,
                  thrInvocationContextPrepareForInvoke2);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_invoke_once,
                  thrInvocationContextInvokeOnce);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_wait,
                  thrInvocationContextWait);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_cancel,
                  thrInvocationContextCancel);

  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_attach_input_buffer_sync_fence,
      thrInvocationContextAttachInputBufferSyncFence);
  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_attach_input_buffer_fence,
      thrInvocationContextAttachInputBufferFence);
  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_detach_input_buffer_sync_fence,
      thrInvocationContextDetachInputBufferSyncFence);
  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_detach_input_buffer_fence,
      thrInvocationContextDetachInputBufferFence);
  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_get_output_buffer_sync_fence,
      thrInvocationContextGetOutputBufferSyncFence);
  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_get_output_buffer_fence,
      thrInvocationContextGetOutputBufferFence);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_query_node_scratch_pad,
                  thrInvocationContextQueryNodeScratchPad);
  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_attach_scratch_pad_buffer,
      thrInvocationContextAttachScratchPadBuffer);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_get_metrics,
                  thrInvocationContextGetMetrics);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_invocation_context_reset_metrics,
                  thrInvocationContextResetMetrics);
  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_start_metrics_collection,
      thrInvocationContextStartMetricsCollection);
  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_invocation_context_stop_metrics_collection,
      thrInvocationContextStopMetricsCollection);

  THR_RESOLVE_SYM(gSouthBoundFns->thr_vendor_set_system_attribute_str,
                  thrVendorSetSystemAttributeStr);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_vendor_set_system_attribute_int64,
                  thrVendorSetSystemAttributeInt64);

  THR_RESOLVE_SYM(
      gSouthBoundFns->thr_graph_add_sq_node_with_interface_binding_mode,
      thrGraphAddSqNodeWithInterfaceBindingMode);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_connect_node_input_with_port_name,
                  thrGraphConnectNodeInputWithPortName);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_connect_node_output_with_port_name,
                  thrGraphConnectNodeOutputWithPortName);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_connect_node_input_with_port_index,
                  thrGraphConnectNodeInputWithPortIndex);
  THR_RESOLVE_SYM(gSouthBoundFns->thr_graph_connect_node_output_with_port_index,
                  thrGraphConnectNodeOutputWithPortIndex);

  THR_LOG_INFO("SouthBound symbols resolved by '%s'", south_bound_lib_path);
  return ThrStatus::kThrStatusSuccess;
}

}  // namespace
}  // namespace hardware::gchips::hetero_runtime::star_ship

using ::hardware::gchips::hetero_runtime::star_ship::GetValForSymUnavailable;
using ::hardware::gchips::hetero_runtime::star_ship::gSouthBoundFns;
using ::hardware::gchips::hetero_runtime::star_ship::LoadSouthBoundSyms;

ThrStatus thrInitialize() {
  if (static const ThrStatus status = LoadSouthBoundSyms();
      status != kThrStatusSuccess) {
    THR_LOG_ERROR("Failed to load SouthBound symbols");
    return status;
  }

  return THR_CALL_DYN_FN(thr_initialize);
}

const char* thrGetVendorApiVersion() {
  return THR_CALL_DYN_FN(thr_get_vendor_api_version);
}

const char* thrGetVendorId() { return THR_CALL_DYN_FN(thr_get_vendor_id); }

ThrContext* thrContextCreate() { return THR_CALL_DYN_FN(thr_context_create); }

ThrStatus thrContextDelete(ThrContext* context) {
  return THR_CALL_DYN_FN(thr_context_delete, context);
}

ThrGraph* thrGraphCreate(ThrContext* context) {
  return THR_CALL_DYN_FN(thr_graph_create, context);
}

ThrStatus thrGraphDelete(ThrGraph* graph) {
  return THR_CALL_DYN_FN(thr_graph_delete, graph);
}

ThrStatus thrGraphAddEdge(ThrGraph* graph, ThrEdgeId edge_id,
                          ThrEdgeType type) {
  return THR_CALL_DYN_FN(thr_graph_add_edge, graph, edge_id, type);
}

ThrStatus thrGraphAddSqNode(ThrGraph* graph, ThrNodeId node_id,
                            ThrNodeType type) {
  return THR_CALL_DYN_FN(thr_graph_add_sq_node, graph, node_id, type);
}

ThrStatus thrGraphConnectNodeInput(ThrGraph* graph, ThrNodeId node_id,
                                   ThrEdgeId edge_id) {
  return THR_CALL_DYN_FN(thr_graph_connect_node_input, graph, node_id, edge_id);
}

ThrStatus thrGraphConnectNodeOutput(ThrGraph* graph, ThrNodeId node_id,
                                    ThrEdgeId edge_id) {
  return THR_CALL_DYN_FN(thr_graph_connect_node_output, graph, node_id,
                         edge_id);
}

ThrStatus thrGraphSetInputEdge(ThrGraph* graph, ThrEdgeId edge_id) {
  return THR_CALL_DYN_FN(thr_graph_set_input_edge, graph, edge_id);
}

ThrStatus thrGraphSetOutputEdge(ThrGraph* graph, ThrEdgeId edge_id) {
  return THR_CALL_DYN_FN(thr_graph_set_output_edge, graph, edge_id);
}

ThrStatus thrGraphAnnotateGraph(ThrGraph* graph, const char* key,
                                const char* value) {
  return THR_CALL_DYN_FN(thr_graph_annotate_graph, graph, key, value);
}

ThrStatus thrGraphAnnotateEdge(ThrGraph* graph, ThrEdgeId edge_id,
                               const char* key, const char* value) {
  return THR_CALL_DYN_FN(thr_graph_annotate_edge, graph, edge_id, key, value);
}

ThrStatus thrGraphAnnotateNode(ThrGraph* graph, ThrNodeId node_id,
                               const char* key, const char* value) {
  return THR_CALL_DYN_FN(thr_graph_annotate_node, graph, node_id, key, value);
}

ThrStatus thrLoadSqContainer(ThrContext* context, ThrSqContainerType type,
                             const void* sq_bytecode, size_t size,
                             ThrSqContainerHandle* sq_handle) {
  return THR_CALL_DYN_FN(thr_load_sq_container, context, type, sq_bytecode,
                         size, sq_handle);
}

ThrStatus thrLoadSqContainerFd(ThrContext* context, ThrSqContainerType type,
                               int fd, size_t size, bool lazy_loading,
                               ThrSqContainerHandle* sq_handle) {
  return THR_CALL_DYN_FN(thr_load_sq_container_fd, context, type, fd, size,
                         lazy_loading, sq_handle);
}

ThrStatus thrLoadSqContainerFdWithLoadingType(
    ThrContext* context, ThrSqContainerType type, int fd, size_t size,
    bool lazy_loading, ThrSqContainerHandle* sq_handle,
    ThrSqContainerLoadingType loading_type) {
  return THR_CALL_DYN_FN(thr_load_sq_container_fd_with_loading_type, context,
                         type, fd, size, lazy_loading, sq_handle, loading_type);
}

ThrStatus thrLoadSqContainerFdWithOffset(ThrContext* context,
                                         ThrSqContainerType type, int fd,
                                         size_t size, size_t offset,
                                         bool lazy_loading,
                                         ThrSqContainerHandle* sq_handle) {
  return THR_CALL_DYN_FN(thr_load_sq_container_fd_with_offset, context, type,
                         fd, size, offset, lazy_loading, sq_handle);
}

ThrStatus thrLoadSqContainerFile(ThrContext* context, ThrSqContainerType type,
                                 const char* filename,
                                 ThrSqContainerHandle* sq_handle) {
  return THR_CALL_DYN_FN(thr_load_sq_container_file, context, type, filename,
                         sq_handle);
}

ThrStatus thrUnloadSqContainer(ThrContext* context,
                               ThrSqContainerHandle handle) {
  return THR_CALL_DYN_FN(thr_unload_sq_container, context, handle);
}

ThrStatus thrGraphAssignSq(ThrGraph* graph, ThrNodeId node_id,
                           ThrSqContainerHandle sq_handle,
                           const char* func_name) {
  return THR_CALL_DYN_FN(thr_graph_assign_sq, graph, node_id, sq_handle,
                         func_name);
}

ThrStatus thrSqQueryScratchPad(ThrContext* context,
                               ThrSqContainerHandle sq_handle,
                               const char* func_name, uint64_t* buffer_size) {
  return THR_CALL_DYN_FN(thr_sq_query_scratch_pad, context, sq_handle,
                         func_name, buffer_size);
}

ThrStatus thrSqAttachScratchPadBuffer(ThrContext* context,
                                      ThrSqContainerHandle sq_handle,
                                      const char* func_name,
                                      ThrBufferHandle handle) {
  return THR_CALL_DYN_FN(thr_sq_attach_scratch_pad_buffer, context, sq_handle,
                         func_name, handle);
}

ThrStatus thrRegisterBuffer(ThrContext* context, ThrBufferType type,
                            void* buffer, size_t size,
                            ThrBufferHandle* handle) {
  return THR_CALL_DYN_FN(thr_register_buffer, context, type, buffer, size,
                         handle);
}

ThrStatus thrRegisterBufferWithOffset(ThrContext* context, ThrBufferType type,
                                      void* buffer, size_t offset, size_t size,
                                      ThrBufferHandle* handle) {
  return THR_CALL_DYN_FN(thr_register_buffer_with_offset, context, type, buffer,
                         offset, size, handle);
}

ThrStatus thrUnregisterBuffer(ThrContext* context, ThrBufferHandle handle) {
  return THR_CALL_DYN_FN(thr_unregister_buffer, context, handle);
}

ThrStatus thrRegisterFence(ThrContext* context, ThrFenceType type, int fence_fd,
                           ThrFenceHandle* handle) {
  return THR_CALL_DYN_FN(thr_register_fence, context, type, fence_fd, handle);
}

ThrStatus thrUnregisterFence(ThrContext* context, ThrFenceHandle handle) {
  return THR_CALL_DYN_FN(thr_unregister_fence, context, handle);
}

ThrStatus thrFenceGetDupFd(ThrContext* context, ThrFenceHandle handle,
                           int* fence_fd) {
  return THR_CALL_DYN_FN(thr_fence_get_dup_fd, context, handle, fence_fd);
}

ThrInvocationContext* thrInvocationContextGet(ThrGraph* graph,
                                              ThrContext* context) {
  return THR_CALL_DYN_FN(thr_invocation_context_get, graph, context);
}

ThrStatus thrInvocationContextDelete(ThrGraph* graph,
                                     ThrInvocationContext* icontext) {
  return THR_CALL_DYN_FN(thr_invocation_context_delete, graph, icontext);
}

ThrStatus thrInvocationContextAttachBuffer(ThrInvocationContext* icontext,
                                           ThrContext* context,
                                           ThrEdgeId edge_id,
                                           ThrBufferHandle handle) {
  return THR_CALL_DYN_FN(thr_invocation_context_attach_buffer, icontext,
                         context, edge_id, handle);
}

ThrStatus thrInvocationContextDetachBuffer(ThrInvocationContext* icontext,
                                           ThrContext* context,
                                           ThrEdgeId edge_id,
                                           ThrBufferHandle handle) {
  return THR_CALL_DYN_FN(thr_invocation_context_detach_buffer, icontext,
                         context, edge_id, handle);
}

ThrStatus thrInvocationContextPrepareForInvoke(ThrInvocationContext* icontext,
                                               bool create_output_sync_fence) {
  return THR_CALL_DYN_FN(thr_invocation_context_prepare_for_invoke, icontext,
                         create_output_sync_fence);
}

ThrStatus thrInvocationContextPrepareForInvoke2(
    ThrInvocationContext* icontext, ThrFenceType output_fence_type) {
  return THR_CALL_DYN_FN(thr_invocation_context_prepare_for_invoke2, icontext,
                         output_fence_type);
}

ThrStatus thrInvocationContextInvokeOnce(ThrInvocationContext* icontext) {
  return THR_CALL_DYN_FN(thr_invocation_context_invoke_once, icontext);
}

ThrStatus thrInvocationContextWait(ThrInvocationContext* icontext) {
  return THR_CALL_DYN_FN(thr_invocation_context_wait, icontext);
}

ThrStatus thrInvocationContextCancel(ThrInvocationContext* icontext) {
  return THR_CALL_DYN_FN(thr_invocation_context_cancel, icontext);
}

ThrStatus thrInvocationContextAttachInputBufferSyncFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, int fence_fd) {
  return THR_CALL_DYN_FN(thr_invocation_context_attach_input_buffer_sync_fence,
                         icontext, edge_id, fence_fd);
}

ThrStatus thrInvocationContextAttachInputBufferFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, ThrFenceHandle handle) {
  return THR_CALL_DYN_FN(thr_invocation_context_attach_input_buffer_fence,
                         icontext, edge_id, handle);
}

ThrStatus thrInvocationContextDetachInputBufferSyncFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, int fence_fd) {
  return THR_CALL_DYN_FN(thr_invocation_context_detach_input_buffer_sync_fence,
                         icontext, edge_id, fence_fd);
}

ThrStatus thrInvocationContextDetachInputBufferFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, ThrFenceHandle handle) {
  return THR_CALL_DYN_FN(thr_invocation_context_detach_input_buffer_fence,
                         icontext, edge_id, handle);
}

ThrStatus thrInvocationContextGetOutputBufferSyncFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, int* fence_fd) {
  return THR_CALL_DYN_FN(thr_invocation_context_get_output_buffer_sync_fence,
                         icontext, edge_id, fence_fd);
}

ThrStatus thrInvocationContextGetOutputBufferFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, ThrFenceHandle* handle) {
  return THR_CALL_DYN_FN(thr_invocation_context_get_output_buffer_fence,
                         icontext, edge_id, handle);
}

ThrStatus thrInvocationContextQueryNodeScratchPad(
    ThrInvocationContext* icontext, ThrContext* context, ThrNodeId node_id,
    uint64_t* buffer_size) {
  return THR_CALL_DYN_FN(thr_invocation_context_query_node_scratch_pad,
                         icontext, context, node_id, buffer_size);
}

ThrStatus thrInvocationContextAttachScratchPadBuffer(
    ThrInvocationContext* icontext, ThrContext* context, ThrNodeId node_id,
    ThrBufferHandle handle) {
  return THR_CALL_DYN_FN(thr_invocation_context_attach_scratch_pad_buffer,
                         icontext, context, node_id, handle);
}

ThrStatus thrInvocationContextGetMetrics(ThrInvocationContext* icontext,
                                         ThrInvocationMetrics* metrics) {
  return THR_CALL_DYN_FN(thr_invocation_context_get_metrics, icontext, metrics);
}

ThrStatus thrInvocationContextResetMetrics(ThrInvocationContext* icontext) {
  return THR_CALL_DYN_FN(thr_invocation_context_reset_metrics, icontext);
}

ThrStatus thrInvocationContextStartMetricsCollection(
    ThrInvocationContext* icontext, int detail_level) {
  return THR_CALL_DYN_FN(thr_invocation_context_start_metrics_collection,
                         icontext, detail_level);
}

ThrStatus thrInvocationContextStopMetricsCollection(
    ThrInvocationContext* icontext, ThrInvocationMetrics* metrics) {
  return THR_CALL_DYN_FN(thr_invocation_context_stop_metrics_collection,
                         icontext, metrics);
}

ThrStatus thrVendorSetSystemAttributeStr(ThrContext* context, const char* key,
                                         const char* value) {
  return THR_CALL_DYN_FN(thr_vendor_set_system_attribute_str, context, key,
                         value);
}

ThrStatus thrVendorSetSystemAttributeInt64(ThrContext* context, const char* key,
                                           int64_t value) {
  return THR_CALL_DYN_FN(thr_vendor_set_system_attribute_int64, context, key,
                         value);
}

ThrStatus thrGraphAddSqNodeWithInterfaceBindingMode(
    ThrGraph* graph, ThrNodeId node_id, ThrNodeType type,
    ThrNodeInterfaceBindingMode binding_mode) {
  return THR_CALL_DYN_FN(thr_graph_add_sq_node_with_interface_binding_mode,
                         graph, node_id, type, binding_mode);
}

ThrStatus thrGraphConnectNodeInputWithPortName(ThrGraph* graph,
                                               ThrNodeId node_id,
                                               ThrEdgeId edge_id,
                                               ThrNodeInterfaceId port_id) {
  return THR_CALL_DYN_FN(thr_graph_connect_node_input_with_port_name, graph,
                         node_id, edge_id, port_id);
}

ThrStatus thrGraphConnectNodeOutputWithPortName(ThrGraph* graph,
                                                ThrNodeId node_id,
                                                ThrEdgeId edge_id,
                                                ThrNodeInterfaceId port_id) {
  return THR_CALL_DYN_FN(thr_graph_connect_node_output_with_port_name, graph,
                         node_id, edge_id, port_id);
}

ThrStatus thrGraphConnectNodeInputWithPortIndex(ThrGraph* graph,
                                                ThrNodeId node_id,
                                                ThrEdgeId edge_id,
                                                unsigned int port_index) {
  return THR_CALL_DYN_FN(thr_graph_connect_node_input_with_port_index, graph,
                         node_id, edge_id, port_index);
}

ThrStatus thrGraphConnectNodeOutputWithPortIndex(ThrGraph* graph,
                                                 ThrNodeId node_id,
                                                 ThrEdgeId edge_id,
                                                 unsigned int port_index) {
  return THR_CALL_DYN_FN(thr_graph_connect_node_output_with_port_index, graph,
                         node_id, edge_id, port_index);
}
