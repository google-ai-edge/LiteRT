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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_EVENT_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_EVENT_H_

#include <stddef.h>
#include <stdint.h>

// C-compatible version of 'tflite::Profiler::EventType'
typedef enum LiteRtProfilerEventType {
  // Default event type, the metadata field has no special significance.
  DEFAULT = 1,

  // The event is an operator invocation and the event_metadata field is the
  // index of operator node.
  OPERATOR_INVOKE_EVENT = 1 << 1,

  // The event is an invocation for an internal operator of a TFLite delegate.
  // The event_metadata field is the index of operator node that's specific to
  // the delegate.
  DELEGATE_OPERATOR_INVOKE_EVENT = 1 << 2,

  // A delegate op invoke event that profiles a delegate op in the
  // Operator-wise Profiling section and not in the Delegate internal section.
  DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT = 1 << 3,

  // The event is a recording of runtime instrumentation such as the overall
  // TFLite runtime status, the TFLite delegate status (if a delegate
  // is applied), and the overall model inference latency etc.
  // Note, the delegate status and overall status are stored as separate
  // event_metadata fields. In particular, the delegate status is encoded
  // as DelegateStatus::full_status().
  GENERAL_RUNTIME_INSTRUMENTATION_EVENT = 1 << 4,

  // Telemetry events. Users and code instrumentations should invoke Telemetry
  // calls instead of using the following types directly.
  // See experimental/telemetry:profiler for definition of each metadata.
  //
  // A telemetry event that reports model and interpreter level events.
  TELEMETRY_EVENT = 1 << 5,
  // A telemetry event that reports model and interpreter level settings.
  TELEMETRY_REPORT_SETTINGS = 1 << 6,
  // A telemetry event that reports delegate level events.
  TELEMETRY_DELEGATE_EVENT = 1 << 7,
  // A telemetry event that reports delegate settings.
  TELEMETRY_DELEGATE_REPORT_SETTINGS = 1 << 8,
} LiteRtProfilerEventType;

// C-compatible version of 'tflite::profiling::memory::MemoryUsage'
// We define a C struct that matches the memory layout of the C++ one.
typedef struct LiteRtMemoryUsage {
  size_t total_allocated_bytes;
  size_t in_use_allocated_bytes;
  size_t private_footprint_bytes;
} LiteRtMemoryUsage;

// C-compatible version of 'Source'
typedef enum ProfiledEventSource {
  LITERT,
  TFLITE_INTERPRETER,
  TFLITE_DELEGATE,
} ProfiledEventSource;

// The C version of the struct uses only pure C types.
typedef struct ProfiledEventData {
  const char* tag;  // The tag of the event. The tag is copied and owned by the
                    // profiler, the caller does not need to keep the string
                    // alive.
  LiteRtProfilerEventType event_type;
  uint64_t start_timestamp_us;
  uint64_t elapsed_time_us;
  LiteRtMemoryUsage begin_mem_usage;
  LiteRtMemoryUsage end_mem_usage;
  uint64_t event_metadata1;
  uint64_t event_metadata2;
  ProfiledEventSource event_source;
} ProfiledEventData;

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_C_LITERT_PROFILER_EVENT_H_
