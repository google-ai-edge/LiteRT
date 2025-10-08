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

#ifndef THIRD_PARTY_ODML_INFRA_SOUTHBOUND_SB_API_H_
#define THIRD_PARTY_ODML_INFRA_SOUTHBOUND_SB_API_H_

#include <cstddef>
#include <cstdint>

#if __ANDROID_API__ >= 26 || defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)
#include <android/hardware_buffer.h>
#else
extern "C" {
typedef struct AHardwareBuffer AHardwareBuffer;
}  // extern "C"
#endif  // #if __ANDROID_API__ >= 26 ||
        // defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
// Data types used in SouthBound APIs.

// Opaque object to store SouthBound runtime information.
// It's needed to create `ThrGraph`, `ThrSqContainerHandle`,
// `ThrBufferHandle`.
// The `ThrContext` must be valid while using these objects.
typedef struct ThrContext ThrContext;

// Opaque object to represents a SouthBound graph. This object is needed for
// Graph Builder APIs.
typedef struct ThrGraph ThrGraph;

// Opaque object to use for the ThrGraph. This object is needed for Graph
// Invocation APIs.
typedef struct ThrInvocationContext ThrInvocationContext;

// Handle that represents a SchedulingQuantum container.
typedef uint64_t ThrSqContainerHandle;

// Handle that represent a Buffer.
typedef uint64_t ThrBufferHandle;

// String ID of a graph edge.
// The `ThrEdgeId` used in SB APIs does not require persistent memory after API
// usage.
typedef const char* ThrEdgeId;

// String ID of a graph node.
// The `ThrNodeId` used in SB APIs does not require persistent memory after API
// usage.
typedef const char* ThrNodeId;

// String ID of a node interface's input or output.
typedef const char* ThrNodeInterfaceId;

enum ThrEdgeType : int {
  kThrEdgeNoType = 0,
  kThrEdgeTypeTensor = 1,
  kThrEdgeTypeImage = 2,
};

enum ThrNodeType : int {
  kThrNodeNoType = 0,
  kThrNodeTypeDsp = 1,  // Node for DSP. a function SQ is supported.
  kThrNodeTypeNpu = 2,  // Node for NPU. a ML model SQ is supported.
  kThrNodeTypeCpu = 3,  // Node for CPU. a TFLite model SQ is supported.
};

// Describes how arguments passed to this node should be internally mapped to
// the invocation interface.
enum ThrNodeInterfaceBindingMode : int {
  kThrNodeInterfaceBindingModeNoType = 0,      // invalid value.
  kThrNodeInterfaceBindingModePositional = 1,  // e.g. `d, e = f(a, b, c)`
  kThrNodeInterfaceBindingModeNamed =
      2,  // e.g. `d = f(arg2=b, arg1=a)['out_y23']`
};

enum ThrStatus : int {
  kThrStatusSuccess = 0,
  kThrStatusFail = -1,
};

// Content types of a SQ container.
enum ThrSqContainerType : int {
  kThrSqContainerNoType = 0,
  kThrSqContainerTypeFunctionLibrary = 1,  // Shared Library
  kThrSqContainerTypeMlModel = 2,          // Vendor specific ML model
  kThrSqContainerTypeTflite = 1000,        // TFLite model.
};

// Loading types of a SQ container.
// Current behavior is that the loading type will override the system attribute.
enum ThrSqContainerLoadingType : int {
  kThrSqContainerLoadingTypeNone = 0,    // Not set specifically, use default
  kThrSqContainerLoadingTypeMmap = 1,    // Mmap the file
  kThrSqContainerLoadingTypeDmaBuf = 2,  // DmaBuf
};

// Buffer type.
enum ThrBufferType : int {
  kThrBufferTypeHostMemory = 0,       // Host side memory
  kThrBufferTypeAHardwareBuffer = 1,  // Android Hardware Buffer
  kThrBufferTypeFastRpcMem =
      2,  // Shared memory used for FastRPC.
          // https://android.googlesource.com/platform/external/fastrpc/
};

// Version of the `ThrInvocationMetrics` struct defined in this header file.
// Increment this value whenever the struct's binary layout changes.
//
// This constant is used by clients to set the `version` field of the
// `ThrInvocationMetrics` struct. The backend maintains its own internal
// versioning scheme and does not access this constant. The backend typically
// supports all past struct versions for backward compatibility, selecting the
// correct one based on the client-set `version` field.
static const int kThrInvocationMetricsStructVersion = 0;

// Structure holding a read-only summary of invocation metrics.
//
// Clients must initialize the `version` field of this struct to
// `kThrInvocationMetricsStructVersion` and pass the struct to
// `thrInvocationContextStopMetricsCollection()`. The function will read the
// version and populate the corresponding remaining fields.
//
// Note: The backend owns the memory pointed to by `metric_keys` and
// `metric_values`. Clients should use the returned data promptly because it
// becomes invalid when any of these functions are called:
// `thrInvocationContextDelete()`,
// `thrInvocationContextStartMetricsCollection()`, or
// `thrInvocationContextStopMetricsCollection()`.
//
// The specific metrics provided are vendor-dependent. For example,
// one vendor might populate the struct as follows:
// ```
//   * num_metrics = 3
//   * metric_keys = {"npu_execution_time_us", "avg_power_usage_mw",
//                    "num_invocations"}
//   * metric_values = {1500, 45, 10}
// ```
// Other vendors may provide different metrics. Consult vendor documentation for
// details.
struct ThrInvocationMetrics {
  // Struct version. Must be set to `kThrInvocationMetricsStructVersion`.
  int version;

  // The number of metrics collected. This will be the number of elements in
  // `metric_keys` and `metric_values`.
  int num_metrics;

  // Array of null-terminated strings identifying each metric. The backend
  // owns the memory for these strings.
  const char** metric_keys;

  // Array of int64_t values corresponding to `metric_keys`. The backend owns
  // the memory for this array.
  const int64_t* metric_values;
};

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// SouthBound APIs.
//
// Here are SouthBound APIs to delegate compute graph to a THR runtime.
// - Management
// - Graph Builder
// - SchedulingQuantum Management
// - Buffer Management
// - InvocationContext
//
// Note: SouthBound API doesn't provide thread-safety except async API which
// ending with "Async"
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Management APIs.

// Initialize SouthBound API.
// This API should be called to use remaining SouthBound APIs.
ThrStatus thrInitialize();

// Returns version of SouthBound API.
//
// This API returns a pointer to a statically allocated string that is the
// version number of SouthBound API vendor implemented. The return value should
// be in semver 2 format <http://semver.org>, starting with MAJOR.MINOR.PATCH,
// e.g. "1.1.0" or "0.1.0-rc2".
const char* thrGetVendorApiVersion();

// Returns string id of vendor's SouthBound API.
//
// This API returns a pointer to a statically allocated string that is the
// id of SouthBound API vendor implementation. The returned value is used to
// determine the vendor of runtime SouthBound APIs.
const char* thrGetVendorId();

// Creates `ThrContext` object.
//
// These objects are used by Vendor's SouthBound implementation to manage
// objects such as `ThrGraph`, `ThrSqContainerHandle`, `ThrBufferHandle`
// Caller owns memory and should call thrContextDelete() to release it.
ThrContext* thrContextCreate();

// Releases the ThrContext object.
//
// The given `ThrContext` should be deleted after all associated objects are
// released.
// Note: Associated `ThrGraph`, `ThrSqContainerHandle`, `ThrBufferHandle`
// and `ThrInvocationContext` should be cleaned up before calling this API.
// If this API is called without proper cleanup, the API will either clean up
// automatically or return an error depending on SB implementation.
ThrStatus thrContextDelete(ThrContext* context);

// --------------------------------------------------------------------------
// Graph Builder APIs.
//
// These APIs are used to construct the `ThrGraph`. `ThrGraph` represents a
// user's compute pipeline. A `ThrGraph` has edges and nodes. Edges are mapped
// to data buffers and nodes are mapped to compute logic (SchedulingQuantum).
// You can get `ThrInvocationContext` to run the graph.
// Once a `ThrInvocationContext` object is
// obtained, the graph becomes immutable, and the graph management APIs will no
// longer function until the `ThrInvocationContext` object is deleted.

// Creates `ThrGraph` object.
//
// You need to provide `ThrContext` to create a `ThrGraph` object. Once it's
// created, you can add edges and nodes to the graph.
// The created `ThrGraph` object is belong to the given `ThrContext`. So it must
// be valid while using the `ThrGraph` object.
ThrGraph* thrGraphCreate(ThrContext* context);

// Deletes the given `ThrGraph` object.
// The associated `ThrContext` must be valid.
// Note: Created `ThrGraph` is supposed to be delete with this API before the
// associated `ThrContext` is deleted by `thrContextDelete` API.
ThrStatus thrGraphDelete(ThrGraph* graph);

// Adds an edge to the given `ThrGraph` object. The given `ThrEdgeId` should be
// unique on the given `ThrGraph` object.
ThrStatus thrGraphAddEdge(ThrGraph* graph, ThrEdgeId edge_id, ThrEdgeType type);

// Adds a compute (SchedulingQuantum) node to the given `ThrGraph` object. The
// given `ThrNodeId` should be unique on the given `ThrGraph` object.
ThrStatus thrGraphAddSqNode(ThrGraph* graph, ThrNodeId node_id,
                            ThrNodeType type);

// Adds a node and specifies the interface binding mode.
ThrStatus thrGraphAddSqNodeWithInterfaceBindingMode(
    ThrGraph* graph, ThrNodeId node_id, ThrNodeType type,
    ThrNodeInterfaceBindingMode binding_mode);

// Set input edges of the given node.
// Can be called multiple times for multiple inputs.
// Use this function for nodes with kThrNodeInterfaceBindingModePositional.
ThrStatus thrGraphConnectNodeInput(ThrGraph* graph, ThrNodeId node_id,
                                   ThrEdgeId edge_id);

// Variant for nodes with kThrNodeInterfaceBindingModeNamed.
ThrStatus thrGraphConnectNodeInputWithPortName(ThrGraph* graph,
                                               ThrNodeId node_id,
                                               ThrEdgeId edge_id,
                                               ThrNodeInterfaceId port_id);

// Set output edges of the given node.
// Can be called multiple time for multiple outputs.
// Use this function for nodes with kThrNodeInterfaceBindingModePositional.
ThrStatus thrGraphConnectNodeOutput(ThrGraph* graph, ThrNodeId node_id,
                                    ThrEdgeId edge_id);

// Variant for nodes with kThrNodeInterfaceBindingModeNamed.
ThrStatus thrGraphConnectNodeOutputWithPortName(ThrGraph* graph,
                                                ThrNodeId node_id,
                                                ThrEdgeId edge_id,
                                                ThrNodeInterfaceId port_id);

// Set input edges of the given `ThrGraph`.
// Can be called multiple time for multiple inputs.
ThrStatus thrGraphSetInputEdge(ThrGraph* graph, ThrEdgeId edge_id);

// Set output edges of the given `ThrGraph`.
// Can be called multiple time for multiple outputs.
ThrStatus thrGraphSetOutputEdge(ThrGraph* graph, ThrEdgeId edge_id);

// --------------------------------------------------------------------------
// Graph Annotation APIs.
//
// These APIs are used to provide annotation to nodes/edges, which allow
// users to specify QoS to the certain nodes/edges.

// Adds an annotation to the entire graph. No additional settings are required
// by default.
ThrStatus thrGraphAnnotateGraph(ThrGraph* graph, const char* key,
                                const char* value);

// Adds an annotation to the specified edge. No additional settings are required
// by default.
ThrStatus thrGraphAnnotateEdge(ThrGraph* graph, ThrEdgeId edge_id,
                               const char* key, const char* value);

// Adds an annotation to the specified node.
ThrStatus thrGraphAnnotateNode(ThrGraph* graph, ThrNodeId node_id,
                               const char* key, const char* value);

// --------------------------------------------------------------------------
// SchedulingQuantum Management APIs.
//
// This API are used to upload and assign SchedulingQuantum to a compute graph
// node which is mapped to a actual compute H/W such as a DSP or NPU.

// Uploads SchedulingQuantumContainer in `ThrSqContainerType`
// and returns `ThrSqContainerHandle`.
// Note: The memory of `sq_bytecode` should be valid until
// `thrUnloadSqContainer` is called.
ThrStatus thrLoadSqContainer(ThrContext* context, ThrSqContainerType type,
                             const void* sq_bytecode, size_t size,
                             ThrSqContainerHandle* sq_handle);

// Uploads SchedulingQuantumContainer in `ThrSqContainerType`
// with a file descriptor and returns `ThrSqContainerHandle`.
// `lazy_loading` tells the implementation to actually load the container
// contents when `thrInvocationContextPrepareForInvoke` is called.
// Note: The file descriptor should be remain open until
// `thrUnloadSqContainer` is called.
// Note: Return kThrStatusFail if this function is called multiple times with
// the same 'fd'.
ThrStatus thrLoadSqContainerFd(ThrContext* context, ThrSqContainerType type,
                               int fd, size_t size, bool lazy_loading,
                               ThrSqContainerHandle* sq_handle);

// Uploads SchedulingQuantumContainer in `ThrSqContainerType`
// with a file descriptor and returns `ThrSqContainerHandle`.
// `lazy_loading` tells the implementation to actually load the container
// contents when `thrInvocationContextPrepareForInvoke` is called.
// `loading_type` tells the implementation how to load the container contents.
// Note: The file descriptor should be remain open until
// `thrUnloadSqContainer` is called.
// Note: Return kThrStatusFail if this function is called multiple times with
// the same 'fd'.
ThrStatus thrLoadSqContainerFdWithLoadingType(
    ThrContext* context, ThrSqContainerType type, int fd, size_t size,
    bool lazy_loading, ThrSqContainerHandle* sq_handle,
    ThrSqContainerLoadingType loading_type);

// The same with `thrLoadSqContainerFd` with additional `offset` parameter.
// This API enables to use part of the given `fd` memory as a
// SchedulingQuantumContainer.
// Note: Unlike thrLoadSqContainerFd, this function can be called multiple times
// with the same 'fd'.
// Note: The file descriptor should be remain open until `thrUnloadSqContainer`
// is called for all ThrSqContainerHandle created from the same fd.
ThrStatus thrLoadSqContainerFdWithOffset(ThrContext* context,
                                         ThrSqContainerType type, int fd,
                                         size_t size, size_t offset,
                                         bool lazy_loading,
                                         ThrSqContainerHandle* sq_handle);

// Uploads SchedulingQuantumContainer for NPU ML models from the given file
// and returns `ThrSqContainerHandle`.
// WARNING: Experimental. Valid only for testing.
ThrStatus thrLoadSqContainerFile(ThrContext* context, ThrSqContainerType type,
                                 const char* filename,

                                 ThrSqContainerHandle* sq_handle);

// Unloads SchedulingQuantumContainer associated with the given
// `ThrSqContainerHandle`.
// Note: Loaded `ThrSqContainerHandle` is supposed to be unloaded with this API
// before the associated `ThrContext` is deleted by `thrContextDelete` API.
ThrStatus thrUnloadSqContainer(ThrContext* context,
                               ThrSqContainerHandle handle);

// Pins the resources of the SqContainer associated with the given
// `ThrSqContainerHandle`. This can be used to preload the resources of the
// SqContainer and prevent them from being released when they are not in use.
// Note: Pinning a pinned SqContainer is a no-op.
ThrStatus thrPinSqContainer(ThrContext* context, ThrSqContainerHandle handle);

// Unpins the resources of the SqContainer associated with the given
// `ThrSqContainerHandle`.
// Note: Unpinning an unpinned SqContainer is a no-op.
ThrStatus thrUnpinSqContainer(ThrContext* context, ThrSqContainerHandle handle);

// Assigns a SchedulingQuantum for the node of `ThrNodeId`.
// `func_name` needs to be provided for a DSP function.
//
// Example:
//   auto status = thrGraphAssignSq(thr_graph, "node_id", sq_handle,
//       "dsp_function");
ThrStatus thrGraphAssignSq(ThrGraph* graph, ThrNodeId node_id,
                           ThrSqContainerHandle sq_handle,
                           const char* func_name);

// SchedulingQuantum ScratchPad APIs
// Scratchpad is a temporary memory for intermediate data during computation of
// a SqNode. User might need to provide scratchpad memory for the SQ execution.
//
// Steps:
// 1. A user query the required scratchpad buffer size
//    via `thrSqQueryScratchPad` API.
//    Note: When the returned size is zero, you can skip the remaining steps.
// 2. The user allocates the scratchpad buffer with the required size
//    via `thrRegisterBuffer` API.
// 3. The user attaches the scratchpad buffer to the SQ
//    via `thrSqAttachScratchPadBuffer` API.

//
// WARNING: These APIs are experimental and subject to change.

// Query the required scratchpad buffer size of the given ThrNodeId `node_id`.
// The returned `buffer_size` could be 0 if the node doesn't require scratchpad
// memory.
// Parameters:
// - context: ThrContext holds the target SchedulingQuantumContainer.
// - sq_handle: ThrSqContainerHandle of the target SQ which the scratchpad
//              buffer needs to be attached.
// - func_name: If the target SchedulingQuantumContainer has multiple SQ,
//              `func_name` is used to identify the target SQ
//              otherwise it should be nullptr.
// - buffer_size: Pointer to the required buffer size of the scratchpad memory.
//                It returns 0 if the node doesn't require scratchpad memory.
//
// WARNING: This API is experimental and subject to change.
ThrStatus thrSqQueryScratchPad(ThrContext* context,
                               ThrSqContainerHandle sq_handle,
                               const char* func_name, uint64_t* buffer_size);

// Attaches the given ThrBufferHandle `handle` to the given ThrNodeId `node_id`
// for ScratchPad memory.
//
// Parameters:
// - context: ThrContext holds the target SchedulingQuantumContainer.
// - sq_handle: ThrSqContainerHandle of the target SQ which the scratchpad
//              buffer will be attached.
// - func_name: If the target SchedulingQuantumContainer has multiple SQ,
//              `func_name` is used to identify the target SQ
//              otherwise it should be nullptr.
// - handle: The ThrBufferHandle to be used as scratchpad buffer for the node.
//
// WARNING: This API is experimental and subject to change.
ThrStatus thrSqAttachScratchPadBuffer(ThrContext* context,
                                      ThrSqContainerHandle sq_handle,
                                      const char* func_name,
                                      ThrBufferHandle handle);
// --------------------------------------------------------------------------
// Buffer Management APIs.
//
// These APIs register / unregister buffers to SouthBound runtime.
// Registered buffers will be assigned to an edge of a `ThrGraph` via
// `thrInvocationContextAttachBuffer` API.
// To execute a `ThrGraph` with InvocationContext APIs, all edges of the
// `ThrGraph` should have assign buffers.
// After the execution, buffers can be detached by
// `thrInvocationContextDetachBuffer` API when they're not used anymore.

// Registers the given `buffer` in `type` to the given `ThrContext`.
// The `size` is the size of the given `buffer`. You can omit it for
// `AHardwareBuffer`.
// Note: The memory of `buffer` should be valid until `thrUnregisterBuffer` is
// called.
ThrStatus thrRegisterBuffer(ThrContext* context, ThrBufferType type,
                            void* buffer, size_t size, ThrBufferHandle* handle);

// Registers `AHardwareBuffer` to the given `ThrContext`.
//
// Warning: This is NOT a SB API but a `thrRegisterBuffer` wrapper for
// `AHardwareBuffer` buffer type.
inline ThrStatus thrRegisterBufferAhwb(ThrContext* context,
                                       AHardwareBuffer* ahwb,
                                       ThrBufferHandle* handle) {
  return thrRegisterBuffer(context, kThrBufferTypeAHardwareBuffer, ahwb,
                           /*size=*/0, handle);
}

// The same with `thrRegisterBuffer` with additional `offset` parameter.
// This API enables to use part of the given `buffer` memory as a SouthBound
// Buffer.
//
// Note: The offset + size should be less than or equal to the size of the
// given `buffer`.
ThrStatus thrRegisterBufferWithOffset(ThrContext* context, ThrBufferType type,
                                      void* buffer, size_t offset, size_t size,
                                      ThrBufferHandle* handle);

// The same with `thrRegisterBufferAhwb` with additional `offset` parameter.
inline ThrStatus thrRegisterBufferAhwbWithOffset(ThrContext* context,
                                                 AHardwareBuffer* ahwb,
                                                 size_t offset,
                                                 ThrBufferHandle* handle) {
  return thrRegisterBufferWithOffset(context, kThrBufferTypeAHardwareBuffer,
                                     ahwb, offset,
                                     /*size=*/0, handle);
}

// Unregisters the registered buffer associated with the given
// `ThrBufferHandle`.
// Note: Registered `ThrBufferHandle` is supposed to be unregistered with this
// API before the associated `ThrContext` is deleted by `thrContextDelete` API.
ThrStatus thrUnregisterBuffer(ThrContext* context, ThrBufferHandle handle);

// InvocationContext APIs
// These APIs are used to control the execution of the graph.
//
// Once the InvocationContext object is obtained, the graph becomes frozen so
// graph management APIs won't work anymore until the
// `thrInvocationContextDelete` is called.
ThrInvocationContext* thrInvocationContextGet(ThrGraph* graph,
                                              ThrContext* context);

// Releases the ThrInvocationContext object, enabling modifications to the
// graph.
// Note: Acquired `ThrInvocationContext` is supposed to be deleted with this API
// before the associated `ThrContext` is deleted by `thrContextDelete` API.
ThrStatus thrInvocationContextDelete(ThrGraph* graph,
                                     ThrInvocationContext* icontext);

// Attaches buffer to the ThrInvocationContext for execution. The provided
// `ThrEdgeId` and `ThrBufferHandle` must be valid.
// Note: The attached buffer should be detached via
// `thrInvocationContextDetachBuffer` when it's not used anymore.
ThrStatus thrInvocationContextAttachBuffer(ThrInvocationContext* icontext,
                                           ThrContext* context,
                                           ThrEdgeId edge_id,
                                           ThrBufferHandle handle);

// Detaches the previously attached buffer `ThrBufferHandle` from the
// ThrInvocationContext. The provided `ThrEdgeId` and `ThrBufferHandle` must be
// valid.
ThrStatus thrInvocationContextDetachBuffer(ThrInvocationContext* icontext,
                                           ThrContext* context,
                                           ThrEdgeId edge_id,
                                           ThrBufferHandle handle);

// Let THR Proxy Scheduler prepare resources for execution.
// At this moment, all the required buffers must be attached. All the used
// SchedulingQuantum must be assigned.
// When `create_output_sync_fence` is true, an output fence will be created for
// synchronization and you can get it with
// thrInvocationContextGetOutputBufferSyncFence() API.
ThrStatus thrInvocationContextPrepareForInvoke(ThrInvocationContext* icontext,
                                               bool create_output_sync_fence);

// Expects the user has already attached the required arguments. Can be called
// again to re-invoke the executor, but only after the previous invocation has
// completed.
ThrStatus thrInvocationContextInvokeOnce(ThrInvocationContext* icontext);

// Waits for the ThrInvocationContext to complete execution. Returns immediately
// if the ThrInvocationContext is not currently executing.
//
// Note: `thrInvocationContextWait` can be called multiple times concurrently or
// sequentially.
// Note: When an output sync fence is not in use, users must call
// thrInvocationContextWait after thrInvocationContextInvokeOnce.

ThrStatus thrInvocationContextWait(ThrInvocationContext* icontext);

// Cancels the invocation of the ThrInvocationContext.
//
// Note: Even `thrInvocationContextCancel` is called, `thrInvocationContextWait`
// must be called for the proper cleanup.
// Note: It will be no-op if the underlying environment does not support
// cancellation is not supported.
// Note: It will be no-op if the requests are already completed.

ThrStatus thrInvocationContextCancel(ThrInvocationContext* icontext);

// Attaches the given sync fence to the graph input edge. The graph execution
// will be resumed after the sync fence is fired.
//
// Note: User must keep `fence_fd` alive and valid for the duration of the
// invocation.
// Note: User should detach `fence_fd` using
// `thrInvocationContextDetachInputBufferSyncFence` and then can close it when
// the invocation is complete (output sync from
// `thrInvocationContextGetOutputBufferSyncFence` is signaled or
// `thrInvocationContextWait` returns)
ThrStatus thrInvocationContextAttachInputBufferSyncFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, int fence_fd);

// Detaches the given sync fence from the graph input edge.
//
ThrStatus thrInvocationContextDetachInputBufferSyncFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, int fence_fd);

// Returns an output sync fence to the graph output edge. When graph execution
// is finished, the sync fence will be fired.
//
// Note: User needs to close the returned fd when no longer in use.
ThrStatus thrInvocationContextGetOutputBufferSyncFence(
    ThrInvocationContext* icontext, ThrEdgeId edge_id, int* fence_fd);

// InvocationContext ScratchPad APIs
//
// WARNING: APIs are obsolete and will be removed soon.
// TODO(b/360930144): Remove these APIs.

// Query the required scratchpad buffer size of the given ThrNodeId `node_id`.
// The returned `buffer_size` could be 0 if the node doesn't require scratchpad
// memory.
// Parameters:
// - icontext: Target ThrInvocationContext that controls the graph execution.
// - context: ThrContext holds the target Southbound Graph.
// - node_id: ThrNodeId of the target node which the scratchpad buffer is
//            attached.
// - buffer_size: Pointer to the required buffer size of the scratchpad memory.
//                It returns 0 if the node doesn't require scratchpad memory.
//
// WARNING: This API is obsolete and will be removed soon.
ThrStatus thrInvocationContextQueryNodeScratchPad(
    ThrInvocationContext* icontext, ThrContext* context, ThrNodeId node_id,
    uint64_t* buffer_size);

// Attaches the given ThrBufferHandle `handle` to the given ThrNodeId `node_id`
// for ScratchPad memory.
//
// Parameters:
// - icontext: Target ThrInvocationContext that controls the graph execution.
// - context: ThrContext holds the target Southbound Graph.
// - node_id: ThrNodeId of the target node which the scratchpad buffer is
//            attached.
// - handle: The ThrBufferHandle to be used as scratchpad buffer for the node.
//
// WARNING: This API is obsolete and will be removed soon.
ThrStatus thrInvocationContextAttachScratchPadBuffer(
    ThrInvocationContext* icontext, ThrContext* context, ThrNodeId node_id,
    ThrBufferHandle handle);

// Starts the metrics collection for the given `ThrInvocationContext`.
//
// Parameters:
//   - icontext: The `ThrInvocationContext` for which to start metrics
//     collection.
//   - detail_level: The detail level of the metrics collection.
//     The exact meaning of detail level is vendor-dependent.
ThrStatus thrInvocationContextStartMetricsCollection(
    ThrInvocationContext* icontext, int detail_level);

// Stops the metrics collection for the given `ThrInvocationContext` and returns
// the collected metrics.
//
// Parameters:
//   - icontext: The `ThrInvocationContext` for which to stop metrics
//     collection.
//   - metrics: A pointer to a `ThrInvocationMetrics` struct, where
//     the `version` field is used as input and other fields are used as output.
//     On success, the remaining fields of `metrics` are filled with
//     data. See the `ThrInvocationMetrics` struct comment for details about the
//     lifetime of returned pointers and example values.
ThrStatus thrInvocationContextStopMetricsCollection(
    ThrInvocationContext* icontext, ThrInvocationMetrics* metrics);

// --------------------------------------------------------------------------
// Vendor APIs.
//
// These APIs are used to retrieve and set vendor specific attributes to
// SouthBound runtime.
//
// WARNING: APIs are experimental and subject to change.

// Sets the given string `key` attribute of the vendor SouthBound
// implementation associated with the given `context`.
//
// WARNING: This API is experimental and subject to change.
ThrStatus thrVendorSetSystemAttributeStr(ThrContext* context, const char* key,
                                         const char* value);

// Sets the given int64_t `key` attribute of the vendor SouthBound
// implementation associated with the given `context`
//
// WARNING: This API is experimental and subject to change.
ThrStatus thrVendorSetSystemAttributeInt64(ThrContext* context, const char* key,
                                           int64_t value);

// --------------------------------------------------------------------------
// Deprecated APIs.

// Retrieves a read-only snapshot of the invocation metrics.
// WARNING: Deprecated. Use `thrInvocationContextStartMetricsCollection` and
// `thrInvocationContextStopMetricsCollection` instead.
ThrStatus thrInvocationContextGetMetrics(ThrInvocationContext* icontext,
                                         ThrInvocationMetrics* metrics);

// Resets the invocation metrics for the given `ThrInvocationContext`.
// WARNING: Deprecated. Use `thrInvocationContextStartMetricsCollection` and
// `thrInvocationContextStopMetricsCollection` instead.
ThrStatus thrInvocationContextResetMetrics(ThrInvocationContext* icontext);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_INFRA_SOUTHBOUND_SB_API_H_
