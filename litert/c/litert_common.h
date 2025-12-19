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

#include "litert/build_common/build_config.h"  // IWYU pragma: keep



// Define LITERT_WINDOWS_OS if the current OS is Windows.
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || \
    defined(__NT__) || defined(_WIN64)
#define LITERT_WINDOWS_OS 1
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Define LITERT_CAPI_EXPORT macro to export a function properly with a shared
// library.
#if defined(LITERT_WINDOWS_OS)
#ifdef LITERT_COMPILE_LIBRARY
#define LITERT_CAPI_EXPORT __declspec(dllexport)
#else
#define LITERT_CAPI_EXPORT __declspec(dllimport)
#endif  // LITERT_COMPILE_LIBRARY
#else
#define LITERT_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // LITERT_WINDOWS_OS

// Declares canonical opaque type.

#ifdef __cplusplus
#define LITERT_DEFINE_HANDLE(name) \
  typedef class name##T* name;     \
  typedef const class name##T* name##Const
#else  // __cplusplus
#define LITERT_DEFINE_HANDLE(name) \
  typedef struct name##T* name;    \
  typedef const struct name##T* name##Const
#endif  // __cplusplus

#define LITERT_DEFINE_HANDLE_STRUCT(name) \
  typedef struct name##T* name;           \
  typedef const struct name##T* name##Const

// LiteRT Accelerator object. (litert_accelerator.h)
LITERT_DEFINE_HANDLE(LiteRtAccelerator);
// LiteRT DelegateWrapper object. (litert_delegate_wrapper.h).
// This wraps a TF Lite delegate (TfLiteOpaqueDelegate*).
LITERT_DEFINE_HANDLE(LiteRtDelegateWrapper);
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
// LiteRT Builder object. (litert_builder.h)
LITERT_DEFINE_HANDLE(LiteRtBuilder);
// Representations of an custom op.  (litert_op_options.h)
LITERT_DEFINE_HANDLE(LiteRtOp);
// A linked list of type erased opaque options. These are added to the
// LiteRtOptions object. (litert_opaque_options.h)
LITERT_DEFINE_HANDLE(LiteRtOpaqueOptions);
// The compilation options for the LiteRtCompiledModel. (litert_options.h)
LITERT_DEFINE_HANDLE_STRUCT(LiteRtOptions);
// LiteRT TensorBuffer object. (litert_tensor_buffer.h)
LITERT_DEFINE_HANDLE(LiteRtTensorBuffer);
// LiteRT TensorBufferRequirements object. (litert_tensor_buffer_requirements.h)
LITERT_DEFINE_HANDLE(LiteRtTensorBufferRequirements);
// LiteRT Profiler object. (litert_profiler.h)
LITERT_DEFINE_HANDLE(LiteRtProfiler);
// LiteRT ExternalLiteRtBufferContext object.
// (litert_external_litert_buffer_context.h)
LITERT_DEFINE_HANDLE(LiteRtExternalLiteRtBufferContext);

#if defined(__linux__) || defined(__ANDROID__)
#define LITERT_HAS_SYNC_FENCE_SUPPORT 1
#else
#define LITERT_HAS_SYNC_FENCE_SUPPORT 0
#endif

#if defined(__ANDROID__)
#define LITERT_HAS_AHWB_SUPPORT_DEFAULT 1
#define LITERT_HAS_OPENGL_SUPPORT_DEFAULT 1
#define LITERT_HAS_ION_SUPPORT_DEFAULT 1
#define LITERT_HAS_DMABUF_SUPPORT_DEFAULT 1
#define LITERT_HAS_FASTRPC_SUPPORT_DEFAULT 1
// copybara:comment_begin(google-only)
#elif defined(GOOGLE_UNSUPPORTED_OS_LOONIX)
#define LITERT_HAS_OPENGL_SUPPORT_DEFAULT 0
#define LITERT_HAS_ION_SUPPORT_DEFAULT 0
#define LITERT_HAS_DMABUF_SUPPORT_DEFAULT 1
#define LITERT_HAS_FASTRPC_SUPPORT_DEFAULT 0
// copybara:comment_end
#else
#define LITERT_HAS_AHWB_SUPPORT_DEFAULT 0
#define LITERT_HAS_OPENGL_SUPPORT_DEFAULT 0
#define LITERT_HAS_ION_SUPPORT_DEFAULT 0
#define LITERT_HAS_DMABUF_SUPPORT_DEFAULT 0
#define LITERT_HAS_FASTRPC_SUPPORT_DEFAULT 0
#endif

#if defined(__APPLE__)
#define LITERT_HAS_METAL_SUPPORT_DEFAULT 1
#define LITERT_HAS_OPENCL_SUPPORT_DEFAULT 0
#define LITERT_HAS_VULKAN_SUPPORT_DEFAULT 0
#else
#define LITERT_HAS_METAL_SUPPORT_DEFAULT 0
#define LITERT_HAS_OPENCL_SUPPORT_DEFAULT 1
#define LITERT_HAS_VULKAN_SUPPORT_DEFAULT 1
#endif

#define LITERT_HAS_WEBGPU_SUPPORT_DEFAULT 1

#if defined(LITERT_DISABLE_NPU)
#define LITERT_DISABLE_AHWB_SUPPORT
#define LITERT_DISABLE_ION_SUPPORT
#define LITERT_DISABLE_DMABUF_SUPPORT
#define LITERT_DISABLE_FASTRPC_SUPPORT
#endif

#if defined(LITERT_DISABLE_GPU)
#define LITERT_DISABLE_METAL_SUPPORT
#define LITERT_DISABLE_OPENCL_SUPPORT
#define LITERT_DISABLE_WEBGPU_SUPPORT
#define LITERT_DISABLE_VULKAN_SUPPORT
#define LITERT_DISABLE_OPENGL_SUPPORT
#endif

#if defined(LITERT_DISABLE_METAL_SUPPORT)
#define LITERT_HAS_METAL_SUPPORT 0
#else
#define LITERT_HAS_METAL_SUPPORT LITERT_HAS_METAL_SUPPORT_DEFAULT
#endif

#if defined(LITERT_DISABLE_OPENCL_SUPPORT)
#define LITERT_HAS_OPENCL_SUPPORT 0
#else
#define LITERT_HAS_OPENCL_SUPPORT LITERT_HAS_OPENCL_SUPPORT_DEFAULT
#endif

#if defined(LITERT_DISABLE_WEBGPU_SUPPORT)
#define LITERT_HAS_WEBGPU_SUPPORT 0
#else
#define LITERT_HAS_WEBGPU_SUPPORT LITERT_HAS_WEBGPU_SUPPORT_DEFAULT
#endif

#if defined(LITERT_DISABLE_VULKAN_SUPPORT)
#define LITERT_HAS_VULKAN_SUPPORT 0
#else
#define LITERT_HAS_VULKAN_SUPPORT LITERT_HAS_VULKAN_SUPPORT_DEFAULT
#endif

#if defined(LITERT_DISABLE_OPENGL_SUPPORT)
#define LITERT_HAS_OPENGL_SUPPORT 0
#else
#define LITERT_HAS_OPENGL_SUPPORT LITERT_HAS_OPENGL_SUPPORT_DEFAULT
#endif

#if defined(LITERT_DISABLE_AHWB_SUPPORT)
#define LITERT_HAS_AHWB_SUPPORT 0
#else
#define LITERT_HAS_AHWB_SUPPORT LITERT_HAS_AHWB_SUPPORT_DEFAULT
#endif

#if defined(LITERT_DISABLE_ION_SUPPORT)
#define LITERT_HAS_ION_SUPPORT 0
#else
#define LITERT_HAS_ION_SUPPORT LITERT_HAS_ION_SUPPORT_DEFAULT
#endif

#if defined(LITERT_DISABLE_DMABUF_SUPPORT)
#define LITERT_HAS_DMABUF_SUPPORT 0
#else
#define LITERT_HAS_DMABUF_SUPPORT LITERT_HAS_DMABUF_SUPPORT_DEFAULT
#endif

#if defined(LITERT_DISABLE_FASTRPC_SUPPORT)
#define LITERT_HAS_FASTRPC_SUPPORT 0
#else
#define LITERT_HAS_FASTRPC_SUPPORT LITERT_HAS_FASTRPC_SUPPORT_DEFAULT
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
  kLiteRtStatusErrorAlreadyExists = 10,

  // Inference progression errors.
  kLiteRtStatusCancelled = 100,

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

  // Transformation related errors.
  kLiteRtStatusPatternNoMatch = 3000,
  kLiteRtStatusInvalidTransformation = 3001,
} LiteRtStatus;
// LINT.ThenChange(
//   ../kotlin/src/main/kotlin/com/google/ai/edge/litert/LiteRtException.kt:status_codes,
//   ../cc/litert_common.h:status_codes
// )

// Returns a string describing the status value.
const char* LiteRtGetStatusString(LiteRtStatus status);

typedef enum : int {
  kLiteRtHwAcceleratorNone = 0,
  kLiteRtHwAcceleratorCpu = 1 << 0,
  kLiteRtHwAcceleratorGpu = 1 << 1,
  kLiteRtHwAcceleratorNpu = 1 << 2,
#if defined(__EMSCRIPTEN__)
  kLiteRtHwAcceleratorWebNn = 1 << 3,
#endif  // __EMSCRIPTEN__
} LiteRtHwAccelerators;

typedef enum {
  kLiteRtDelegatePrecisionDefault = 0,
  kLiteRtDelegatePrecisionFp16 = 1,
  kLiteRtDelegatePrecisionFp32 = 2,
} LiteRtDelegatePrecision;

typedef enum {
  kLiteRtGpuPriorityDefault = 0,
  kLiteRtGpuPriorityLow = 1,
  kLiteRtGpuPriorityNormal = 2,
  kLiteRtGpuPriorityHigh = 3,
} LiteRtGpuPriority;

// Storage type needed by buffers to be allocated in the GPU delegate.
// Default means the storage type will be automatically determined by the
// delegate.
typedef enum {
  kLiteRtDelegateBufferStorageTypeDefault = 0,
  kLiteRtDelegateBufferStorageTypeBuffer = 1,
  kLiteRtDelegateBufferStorageTypeTexture2D = 2,
} LiteRtDelegateBufferStorageType;

// Lock mode for tensor buffer.
typedef enum {
  kLiteRtTensorBufferLockModeRead = 0,
  kLiteRtTensorBufferLockModeWrite = 1,
  kLiteRtTensorBufferLockModeReadWrite = 2,
} LiteRtTensorBufferLockMode;

// Backend option of GPU Accelerator.
// No need to specify on iOS or Metal since there is only one backend.
typedef enum {
  kLiteRtGpuBackendAutomatic = 0,
  kLiteRtGpuBackendOpenCl = 1,
  kLiteRtGpuBackendWebGpu = 2,
  kLiteRtGpuBackendOpenGl = 3,  // Experimental, do not use.
} LiteRtGpuBackend;

// GPU Wait type on synchronous execution.
// Values are 1:1 mapping to GpuDelegateWaitType.
typedef enum {
  // Wait type will be automatically determined by the delegate.
  kLiteRtGpuWaitTypeDefault = 0,
  // Blocked waiting for GPU to finish.
  kLiteRtGpuWaitTypePassive = 1,
  // Active busy-waiting for GPU to finish.
  kLiteRtGpuWaitTypeActive = 2,
  // Do not wait for GPU to finish. Relies on other synchronization ways like
  // barriers or in-order queue. As it's for backward compatibility, not
  // recommended for new use cases. Use asynchronous execution mode instead.
  kLiteRtGpuWaitTypeDoNotWait = 3,
} LiteRtGpuWaitType;

// Error reporter mode enum
typedef enum LiteRtErrorReporterMode {
  // No error reporting (errors are ignored)
  kLiteRtErrorReporterModeNone = 0,
  // Default stderr error reporter
  kLiteRtErrorReporterModeStderr = 1,
  // Buffer error reporter that collects errors for later retrieval
  kLiteRtErrorReporterModeBuffer = 2,
} LiteRtErrorReporterMode;

// A bit field of `LiteRtHwAccelerators` values.
typedef int LiteRtHwAcceleratorSet;

// For indexing into LiteRT collections or counting LiteRT things.
typedef size_t LiteRtParamIndex;

#if defined(_WIN32)
// Provides posix_memalign() missing in Windows.
#include <errno.h>

#define posix_memalign(p, a, s) \
  (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)

// Memory allocated by _aligned_malloc() on Windows needs to be freed by
// _aligned_free(). Use aligned_free() instead of free() for the memory
// allocated by posix_memalign() for cross-platform compatibility.
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc?view=msvc-170
// litert_ prefix is added to avoid name conflicts with one defined in
// base/port.h, for example, included in unittests.
#define litert_aligned_free _aligned_free

#else  // _WIN32
#define litert_aligned_free free
#endif  // _WIN32

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_COMMON_H_
