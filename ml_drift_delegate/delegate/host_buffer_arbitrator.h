// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_HOST_BUFFER_ARBITRATOR_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_HOST_BUFFER_ARBITRATOR_H_

#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

// Describes the properties of a host buffer used for dynamic lifecycle
// arbitration and zero-copy host pointer wrapping.
struct HostBufferDescriptor {
  void* host_ptr = nullptr;
  size_t bytes = 0;
  bool is_custom_allocation = false;
  ::ml_drift::DataType data_type = ::ml_drift::DataType::kUnknown;

  HostBufferDescriptor() = default;
  HostBufferDescriptor(
      void* host_ptr, size_t bytes, bool is_custom_allocation,
      ::ml_drift::DataType data_type = ::ml_drift::DataType::kUnknown)
      : host_ptr(host_ptr),
        bytes(bytes),
        is_custom_allocation(is_custom_allocation),
        data_type(data_type) {}

  // Constructs a descriptor from a TfLiteTensor.
  static HostBufferDescriptor FromTfLiteTensor(
      const TfLiteTensor& tensor,
      ::ml_drift::DataType data_type = ::ml_drift::DataType::kUnknown);
};

// Arbitrates buffer lifecycle, memory reuse, and host-device synchronization
// policies for GPU backend execution. Centralizes zero-copy qualification,
// reallocation decision logic, and fallback staging buffer creation across
// both LiteRT and legacy TfLite delegate kernels.
class HostBufferArbitrator {
 public:
  struct Options {
    // If true, enables host pointer I/O buffers if supported by the backend.
    bool enable_host_ptr_io_buffers = false;
    // If true, forces backend to report host pointer I/O buffer support
    // (useful for CI/test coverage across heterogeneous devices).
    bool force_host_ptr_io_buffers = false;
  };

  HostBufferArbitrator() = default;
  explicit HostBufferArbitrator(GpuBackend* backend);
  HostBufferArbitrator(GpuBackend* backend, Options options);
  virtual ~HostBufferArbitrator() = default;

  // Determines if a host buffer qualifies for direct zero-copy wrapping via
  // OpenCL CL_MEM_USE_HOST_PTR (requires custom allocation, non-null pointer,
  // and OS physical page boundary alignment).
  static bool CanZeroCopy(const HostBufferDescriptor& desc, size_t alignment);
  static bool CanZeroCopy(const TfLiteTensor& tensor, size_t alignment);

  // Evaluates whether an I/O buffer must be recreated based on buffer byte
  // size, option toggles, custom allocation zero-copy eligibility, pointer
  // rebinding, or fallback requirements.
  virtual bool NeedsRecreateBuffer(const GpuIOBuffer* buffer,
                                   const HostBufferDescriptor& desc,
                                   bool is_input,
                                   const void* last_seen_ptr) const;
  bool NeedsRecreateBuffer(const GpuIOBuffer* buffer,
                           const TfLiteTensor& tensor, bool is_input,
                           const void* last_seen_ptr) const;

  // Creates or wraps a GpuIOBuffer for the given descriptor, prioritizing
  // zero-copy if eligible, then falling back to internal page-aligned host
  // memory, and finally standard device allocations.
  virtual absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateOrWrapBuffer(
      const HostBufferDescriptor& desc, bool is_input);
  absl::StatusOr<std::unique_ptr<GpuIOBuffer>> CreateOrWrapBuffer(
      const TfLiteTensor& tensor, ::ml_drift::DataType data_type,
      bool is_input);

  // Checks whether host pointer I/O buffers are supported and enabled for this
  // input/output direction.
  bool IsHostPtrSupported(bool is_input) const;

  // Mutators and accessors.
  void SetBackend(GpuBackend* backend);
  GpuBackend* backend() const { return backend_; }

  void SetOptions(const Options& options);
  const Options& options() const { return options_; }

 private:
  void ApplyBackendOptions();

  GpuBackend* backend_ = nullptr;
  Options options_;
};

}  // namespace litert::ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_HOST_BUFFER_ARBITRATOR_H_
