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

#ifndef ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/types/optional_ref.h"  // from @com_google_absl
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_google_tensor_options_type.h"
#include "litert/vendors/c/litert_dispatch.h"
#if LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
#include "litert/c/options/google/litert_google_tensor_privileged_options_type.h"
#endif  // LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
#include "litert/vendors/google_tensor/dispatch/sb_api.h"

// This class is thread-compatible.
class LiteRtDispatchDeviceContextT {
 public:
  // Stores Google Tensor Options for later annotation.
  struct GoogleTensorOptionsData {
    std::optional<LiteRtGoogleTensorOptionsPerformanceMode> performance_mode =
        std::nullopt;
  };

  static LiteRtStatus Create(const LiteRtRuntimeContext* runtime_context,
                             LiteRtOptions options,
                             LiteRtDispatchDeviceContext& device_context);

  LiteRtStatus Destroy();

  LiteRtStatus RegisterTensorBuffer(
      LiteRtTensorBuffer tensor_buffer,
      LiteRtTensorBufferHandle& tensor_buffer_handle);

  LiteRtStatus UnregisterTensorBuffer(
      LiteRtTensorBufferHandle tensor_buffer_handle);

  LiteRtStatus LoadExecutable(LiteRtDispatchExecutableType type,
                              const LiteRtMemBuffer& bytecode_buffer,
                              LiteRtDispatchExecutableHandle& exec_handle);

  LiteRtStatus UnloadExecutable(LiteRtDispatchExecutableHandle exec_handle);

  bool IsTfliteExecutable(LiteRtDispatchExecutableHandle exec_handle) const;

  // Registers a graph with the device context.
  //
  // This has the effect of guaranteeing that the device context remains alive
  // until the graph is unregistered, meaning that a subsequent call to
  // `UnregisterGraph` is required to permit the device context to be destroyed.
  //
  // NOTE: a graph may only be registered once.
  LiteRtStatus RegisterGraph(LiteRtDispatchGraph graph);

  LiteRtStatus UnregisterGraph(LiteRtDispatchGraph graph);

  ThrContext* absl_nonnull thr_context() { return thr_context_; }

#if LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
  absl::optional_ref<const LrtGoogleTensorPrivilegedOptionsT>
  GetGoogleTensorPrivilegedOptions() const {
    return google_tensor_privileged_options_;
  }

  void SetGoogleTensorPrivilegedOptions(
      std::optional<LrtGoogleTensorPrivilegedOptionsT>
          google_tensor_privileged_options) {
    google_tensor_privileged_options_ =
        std::move(google_tensor_privileged_options);
  }
#endif  // LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT

  absl::optional_ref<const GoogleTensorOptionsData> GetGoogleTensorOptions()
      const {
    return google_tensor_options_;
  }

  void SetGoogleTensorOptions(
      std::optional<GoogleTensorOptionsData> google_tensor_options) {
    google_tensor_options_ = std::move(google_tensor_options);
  }

  const LiteRtRuntimeContext* runtime_context() const {
    return runtime_context_;
  }

  LiteRtStatus AnnotateSystemAttribute(const char* absl_nonnull key,
                                       const char* absl_nonnull value);

 private:
  // Cache key for loaded executable files.
  struct ExecutableFileCacheKey {
    ExecutableFileCacheKey(int fd, size_t size, size_t offset)
        : fd(fd), size(size), offset(offset) {}

    template <typename H>
    friend H AbslHashValue(H h, const ExecutableFileCacheKey& other) {
      return H::combine(std::move(h), other.fd, other.size, other.offset);
    }

    friend bool operator==(const ExecutableFileCacheKey& lhs,
                           const ExecutableFileCacheKey& rhs) {
      return lhs.fd == rhs.fd && lhs.size == rhs.size &&
             lhs.offset == rhs.offset;
    }

    int fd;
    size_t size;
    size_t offset;
  };

  // Aggregates data associated with an executable.
  struct ExecutableData {
    explicit ExecutableData(LiteRtDispatchExecutableHandle handle)
        : handle(handle), ref_count(1) {}

    LiteRtDispatchExecutableHandle handle;
    int ref_count;
  };

  struct MmapRegion {
    LiteRtDispatchExecutableHandle exec_handle;
    void* addr;
    size_t length;
  };

  explicit LiteRtDispatchDeviceContextT(
      const LiteRtRuntimeContext* runtime_context,
      ThrContext* absl_nonnull thr_context)
      : runtime_context_(runtime_context), thr_context_(thr_context) {}

  // Consumers of this class must use `Destroy` to delete the instance.
  ~LiteRtDispatchDeviceContextT() = default;

  const LiteRtRuntimeContext* runtime_context_;
  ThrContext* absl_nonnull thr_context_;
#if LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
  std::optional<LrtGoogleTensorPrivilegedOptionsT>
      google_tensor_privileged_options_;
#endif  // LITERT_HAS_GOOGLE_TENSOR_PRIVILEGED_OPTIONS_SUPPORT
  std::optional<GoogleTensorOptionsData> google_tensor_options_;
  // A device context cannot be destroyed with any registered graphs.
  absl::flat_hash_set<LiteRtDispatchGraph> registered_graphs_;
  // Each region backs an executable's bytecode and must outlive any graph
  // referencing that executable via AssignNodeFunction. Destroy refuses
  // when non-empty; release order is graphs, then executables, then Destroy.
  // std::vector (not hash set): expected N=1.
  std::vector<MmapRegion> mmap_regions_;
  // Set of executables that are TFLite flatbuffers. This is in contrast to
  // other executable types like custom-compiled binaries. TFLite
  // flatbuffers can contain multiple signatures.
  absl::flat_hash_set<LiteRtDispatchExecutableHandle> tflite_executables_;
  // Associates an executable file's cache key with its handle.
  absl::flat_hash_map<ExecutableFileCacheKey, ExecutableData>
      cache_key_to_exec_data_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_DEVICE_CONTEXT_H_
