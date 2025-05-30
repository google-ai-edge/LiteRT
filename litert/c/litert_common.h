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

#ifndef ODML_LITERT_LITERT_C_LITERT_COMMON_H_
#define ODML_LITERT_LITERT_C_LITERT_COMMON_H_

#include <stddef.h>

#ifdef __cplusplus
// =========================================================================
//  C++ DEFINITION: Visible only to C++ compilers
// =========================================================================
#include <cstdint>

#include "tflite/core/api/profiler.h"
#include "tflite/profiling/memory_info.h"

// The original C++ struct definition is kept here.
struct ProfiledEventData {
  const char* tag;  // The tag of the event. The tag is copied and owned by the
              // profiler, the caller does not need to keep the string alive.
  tflite::Profiler::EventType event_type;
  uint64_t start_timestamp_us;
  uint64_t elapsed_time_us;
  tflite::profiling::memory::MemoryUsage begin_mem_usage;
  tflite::profiling::memory::MemoryUsage end_mem_usage;
  uint64_t event_metadata1;
  uint64_t event_metadata2;
  enum class Source {
    LITERT,
    TFLITE_INTERPRETER,
    TFLITE_DELEGATE,
  };
  Source event_source;
};

#else
// =========================================================================
//  C DEFINITION: Visible only to C compilers
// =========================================================================
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
  int64_t total_rss_kb;
  int64_t total_hoard_kb;
} LiteRtMemoryUsage;

// C-compatible version of 'Source'
typedef enum LiteRtProfilerSource {
  LITERT,
  TFLITE_INTERPRETER,
  TFLITE_DELEGATE,
} LiteRtProfilerSource;

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
  LiteRtProfilerSource event_source;
} ProfiledEventData;

#endif  // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Declares canonical opaque type.
#define LITERT_DEFINE_HANDLE(name) \
  typedef struct name##T* name;    \
  typedef const struct name##T* name##Const

// LiteRT Accelerator object. (litert_accelerator.h)
LITERT_DEFINE_HANDLE(LiteRtAccelerator);
// LiteRT CompiledModel object. (litert_compiled_model.h)
LITERT_DEFINE_HANDLE(LiteRtCompiledModel);
// LiteRT Environment object. (litert_environment.h)
LITERT_DEFINE_HANDLE(LiteRtEnvironment);
// LiteRT EnvironmentOptions object. (litert_environment_options.h)
LITERT_DEFINE_HANDLE(LiteRtEnvironmentOptions);
// LiteRT Event object. (litert_event.h)
LITERT_DEFINE_HANDLE(LiteRtEvent);
// LiteRT Logger object. (litert_logging.h)
LITERT_DEFINE_HANDLE(LiteRtLogger);
// LiteRT Metrics object. (litert_metrics.h)
LITERT_DEFINE_HANDLE(LiteRtMetrics);

// Constant data behind a tensor stored in the model. (litert_model.h)
LITERT_DEFINE_HANDLE(LiteRtWeights);
// Values/edges of the model's graph. (litert_model.h)
LITERT_DEFINE_HANDLE(LiteRtTensor);
// Operations/nodes of the model's graph. (litert_model.h)
LITERT_DEFINE_HANDLE(LiteRtOp);
// Fundamental block of program, i.e. a function body. (litert_model.h)
LITERT_DEFINE_HANDLE(LiteRtSubgraph);
// Signature of the model.  (litert_model.h)
LITERT_DEFINE_HANDLE(LiteRtSignature);
// A collection of subgraph + metadata + signature.  (litert_model.h)
LITERT_DEFINE_HANDLE(LiteRtModel);
// Append only list of ops.  (litert_model.h)
LITERT_DEFINE_HANDLE(LiteRtOpList);
// Representations of an custom op.  (litert_op_options.h)
LITERT_DEFINE_HANDLE(LiteRtOp);
// A linked list of type erased opaque options. These are added to the
// LiteRtOptions object. (litert_opaque_options.h)
LITERT_DEFINE_HANDLE(LiteRtOpaqueOptions);
// The compilation options for the LiteRtCompiledModel. (litert_options.h)
LITERT_DEFINE_HANDLE(LiteRtOptions);
// LiteRT TensorBuffer object. (litert_tensor_buffer.h)
LITERT_DEFINE_HANDLE(LiteRtTensorBuffer);
// LiteRT TensorBufferRequirements object. (litert_tensor_buffer_requirements.h)
LITERT_DEFINE_HANDLE(LiteRtTensorBufferRequirements);
// LiteRT Profiler object. (litert_profiler.h)
LITERT_DEFINE_HANDLE(LiteRtProfiler);
// LiteRT Profiler event object. (litert_profiler.h)
LITERT_DEFINE_HANDLE(LiteRtProfilerEvent);

#if __ANDROID_API__ >= 26
#define LITERT_HAS_AHWB_SUPPORT 1
#else
#define LITERT_HAS_AHWB_SUPPORT 0
#endif  // __ANDROID_API__ >= 26

#if defined(__linux__) || defined(__ANDROID__)
#define LITERT_HAS_SYNC_FENCE_SUPPORT 1
#else
#define LITERT_HAS_SYNC_FENCE_SUPPORT 0
#endif

#if defined(__ANDROID__)
#define LITERT_HAS_ION_SUPPORT 1
#define LITERT_HAS_DMABUF_SUPPORT 1
#define LITERT_HAS_FASTRPC_SUPPORT 1
#define LITERT_HAS_OPENGL_SUPPORT 1
#define LITERT_HAS_OPENCL_SUPPORT_DEFAULT 1
// copybara:comment_begin(google-only)
#elif defined(GOOGLE_UNSUPPORTED_OS_LOONIX)
#define LITERT_HAS_ION_SUPPORT 0
#define LITERT_HAS_DMABUF_SUPPORT 1
#define LITERT_HAS_FASTRPC_SUPPORT 0
#define LITERT_HAS_OPENCL_SUPPORT_DEFAULT 1
// copybara:comment_end
#else
#define LITERT_HAS_ION_SUPPORT 0
#define LITERT_HAS_DMABUF_SUPPORT 0
#define LITERT_HAS_FASTRPC_SUPPORT 0
#define LITERT_HAS_OPENCL_SUPPORT_DEFAULT 1
#define LITERT_HAS_OPENGL_SUPPORT 0
#endif

#if defined(LITERT_DISABLE_OPENCL_SUPPORT)
#define LITERT_HAS_OPENCL_SUPPORT 0
#else
#define LITERT_HAS_OPENCL_SUPPORT LITERT_HAS_OPENCL_SUPPORT_DEFAULT
#endif

#define LITERT_API_VERSION_MAJOR 0
#define LITERT_API_VERSION_MINOR 1
#define LITERT_API_VERSION_PATCH 0

typedef struct LiteRtApiVersion {
  int major;
  int minor;
  int patch;
} LiteRtApiVersion;

// Compares `v1` and `v2`.
//
// Returns 0 if they are the same, a negative number if v1 < v2 and a positive
// number if v1 > v2.
int LiteRtCompareApiVersion(LiteRtApiVersion version,
                            LiteRtApiVersion reference);

// LINT.IfChange(status_codes)
typedef enum {
  kLiteRtStatusOk = 0,

  // Generic errors.
  kLiteRtStatusErrorInvalidArgument = 1,
  kLiteRtStatusErrorMemoryAllocationFailure = 2,
  kLiteRtStatusErrorRuntimeFailure = 3,
  kLiteRtStatusErrorMissingInputTensor = 4,
  kLiteRtStatusErrorUnsupported = 5,
  kLiteRtStatusErrorNotFound = 6,
  kLiteRtStatusErrorTimeoutExpired = 7,
  kLiteRtStatusErrorWrongVersion = 8,
  kLiteRtStatusErrorUnknown = 9,

  // File and loading related errors.
  kLiteRtStatusErrorFileIO = 500,
  kLiteRtStatusErrorInvalidFlatbuffer = 501,
  kLiteRtStatusErrorDynamicLoading = 502,
  kLiteRtStatusErrorSerialization = 503,
  kLiteRtStatusErrorCompilation = 504,

  // IR related errors.
  kLiteRtStatusErrorIndexOOB = 1000,
  kLiteRtStatusErrorInvalidIrType = 1001,
  kLiteRtStatusErrorInvalidGraphInvariant = 1002,
  kLiteRtStatusErrorGraphModification = 1003,

  // Tool related errors.
  kLiteRtStatusErrorInvalidToolConfig = 1500,

  // Legalization related errors.
  kLiteRtStatusLegalizeNoMatch = 2000,
  kLiteRtStatusErrorInvalidLegalization = 2001,
} LiteRtStatus;
// LINT.ThenChange(../kotlin/src/main/kotlin/com/google/ai/edge/litert/LiteRtException.kt:status_codes)

// Returns a string describing the status value.
const char* LiteRtGetStatusString(LiteRtStatus status);

typedef enum : int {
  kLiteRtHwAcceleratorNone = 0,
  kLiteRtHwAcceleratorCpu = 1 << 0,
  kLiteRtHwAcceleratorGpu = 1 << 1,
  kLiteRtHwAcceleratorNpu = 1 << 2,
} LiteRtHwAccelerators;

typedef enum {
  kLiteRtDelegatePrecisionDefault = 0,
  kLiteRtDelegatePrecisionFp16 = 1,
  kLiteRtDelegatePrecisionFp32 = 2,
} LiteRtDelegatePrecision;

// Storage type needed by buffers to be allocated in the GPU delegate.
// Default means the storage type will be automatically determined by the
// delegate.
typedef enum {
  kLiteRtDelegateBufferStorageTypeDefault = 0,
  kLiteRtDelegateBufferStorageTypeBuffer = 1,
  kLiteRtDelegateBufferStorageTypeTexture2D = 2,
} LiteRtDelegateBufferStorageType;

// A bit field of `LiteRtHwAccelerators` values.
typedef int LiteRtHwAcceleratorSet;

// For indexing into LiteRT collections or counting LiteRT things.
typedef size_t LiteRtParamIndex;

#if defined(_WIN32)
// Provides posix_memalign() missing in Windows.
#include <errno.h>

#define posix_memalign(p, a, s) \
  (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
#endif  // defined(_WIN32)

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_COMMON_H_
