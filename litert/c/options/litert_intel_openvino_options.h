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
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
#include "litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LrtIntelOpenVinoOptions);

// Create an Intel OpenVINO options object.
LiteRtStatus LrtIntelOpenVinoOptionsCreate(LrtIntelOpenVinoOptions* options);

// Destroy the options object.
void LrtDestroyIntelOpenVinoOptions(LrtIntelOpenVinoOptions options);

// Serializes intel openvino options and returns the components needed to create
// opaque options. The caller is responsible for passing these to
// `LiteRtCreateOpaqueOptions`.
LiteRtStatus LrtGetOpaqueIntelOpenVinoOptionsData(
    LrtIntelOpenVinoOptions options, const char** identifier, void** payload,
    void (**payload_deleter)(void*));

// Gets the identifier for Intel OpenVINO options stored in opaque options.
const char* LrtGetIntelOpenVinoOptionsIdentifier();

// Parses a TOML string into the C API representation.
LiteRtStatus LrtCreateIntelOpenVinoOptionsFromToml(
    const char* payload, LrtIntelOpenVinoOptions* options);

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// graph_backend -----------------------------------------------------------------
//
// The OpenVINO target device for a graph (partition).  Each partition in a
// compiled model carries its own graph type and is dispatched on the
// corresponding OpenVINO device.  There is no longer a model-wide "device
// type"; configure each partition individually via the per-graph API below.
typedef enum LiteRtIntelOpenVinoGraphBackend {
  kLiteRtIntelOpenVinoGraphBackendCPU = 0,
  kLiteRtIntelOpenVinoGraphBackendGPU = 1,
  kLiteRtIntelOpenVinoGraphBackendNPU = 2,
  kLiteRtIntelOpenVinoGraphBackendAUTO = 3,
} LiteRtIntelOpenVinoGraphBackend;

// performance_mode -----------------------------------------------------------

// Configures OpenVINO devices to optimize for performance or efficiency.
// See ov::hint::PerformanceMode in OpenVINO. By default, it
// will use LATENCY mode.

typedef enum LiteRtIntelOpenVinoPerformanceMode {
  /* Optimize for low latency */
  kLiteRtIntelOpenVinoPerformanceModeLatency = 0,
  /* Optimize for high throughput */
  kLiteRtIntelOpenVinoPerformanceModeThroughput = 1,
  /* Optimize for cumulative throughput */
  kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput = 2,
} LiteRtIntelOpenVinoPerformanceMode;

LiteRtStatus LrtIntelOpenVinoOptionsSetPerformanceMode(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode performance_mode);

LiteRtStatus LrtIntelOpenVinoOptionsGetPerformanceMode(
    LrtIntelOpenVinoOptions options,
    LiteRtIntelOpenVinoPerformanceMode* performance_mode);

// configs_map ----------------------------------------------------------------

// Set a custom configuration option with a string key-value pair.
// The key and value strings are copied internally, so their lifetime does not
// need to extend beyond this function call.
LiteRtStatus LrtIntelOpenVinoOptionsSetConfigsMapOption(
    LrtIntelOpenVinoOptions options, const char* key, const char* value);

// Get the number of custom configuration options
LiteRtStatus LrtIntelOpenVinoOptionsGetNumConfigsMapOptions(
    LrtIntelOpenVinoOptions options, int* num_options);

// Get a custom configuration option by index.
// The returned key and value pointers point to internal string data
// and are valid for the lifetime of the options object.
// The caller should not free these pointers.
LiteRtStatus LrtIntelOpenVinoOptionsGetConfigsMapOption(
    LrtIntelOpenVinoOptions options, int index, const char** key,
    const char** value);

// per-graph backend overrides ------------------------------------------------
//
// The OpenVINO compiler plugin compiles each partition for its configured
// graph type (device), and the dispatcher imports each partition's bytecode
// on that same device automatically.
//
// `graph_index` corresponds to the partition index produced by the LiteRT
// partitioner (i.e. the order of subgraphs in the partitioned model passed
// to `LiteRtCompilerPluginCompile`).  Applications that want to map a model
// signature key to a graph index should resolve that mapping themselves and
// pass the resulting integer index here.

// Sets the OpenVINO graph backend (target device) for a specific graph
// (partition) index.  Pass `graph_index = -1` as a wildcard to set the default
// backend used by all graphs that do not have an explicit per-index override.
// Partitions without either an explicit or wildcard override fall back to NPU.
LiteRtStatus LrtIntelOpenVinoOptionsSetGraphBackend(
    LrtIntelOpenVinoOptions options, int graph_index,
    enum LiteRtIntelOpenVinoGraphBackend graph_backend);

// Gets the graph backend override for a specific graph index.  Falls back to
// the wildcard (`graph_index = -1`) entry when no per-index override exists.
// Returns `kLiteRtStatusErrorNotFound` when neither is set.
LiteRtStatus LrtIntelOpenVinoOptionsGetGraphBackend(
    LrtIntelOpenVinoOptions options, int graph_index,
    enum LiteRtIntelOpenVinoGraphBackend* graph_backend);

// Sets an OpenVINO config map option for a specific graph index.  These
// per-graph configs are merged on top of the model-wide configs at compile
// time, with the per-graph values taking precedence.
LiteRtStatus LrtIntelOpenVinoOptionsSetGraphConfigsMapOption(
    LrtIntelOpenVinoOptions options, int graph_index, const char* key,
    const char* value);

// Number of graphs that have at least one per-graph override (device and/or
// config map entries).
LiteRtStatus LrtIntelOpenVinoOptionsGetNumGraphOverrides(
    LrtIntelOpenVinoOptions options, int* num_overrides);

// Returns the graph index at slot `slot_index`.  Use together with
// `LrtIntelOpenVinoOptionsGetNumGraphOverrides` to enumerate overrides.
LiteRtStatus LrtIntelOpenVinoOptionsGetGraphOverrideIndex(
    LrtIntelOpenVinoOptions options, int slot_index, int* graph_index);

// Number of per-graph config map entries set for `graph_index`.  Returns 0
// if no overrides are set for that graph.
LiteRtStatus LrtIntelOpenVinoOptionsGetNumGraphConfigsMapOptions(
    LrtIntelOpenVinoOptions options, int graph_index, int* num_options);

// Returns the (key, value) of the `index`-th config map entry for
// `graph_index`.
LiteRtStatus LrtIntelOpenVinoOptionsGetGraphConfigsMapOption(
    LrtIntelOpenVinoOptions options, int graph_index, int index,
    const char** key, const char** value);

#ifdef __cplusplus

}  // extern "C"

#endif  // __cplusplus
#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_OPTIONS_LITERT_INTEL_OPENVINO_OPTIONS_H_
