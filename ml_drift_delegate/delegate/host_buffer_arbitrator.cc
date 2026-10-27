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

#include "ml_drift_delegate/delegate/host_buffer_arbitrator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {

HostBufferDescriptor HostBufferDescriptor::FromTfLiteTensor(
    const TfLiteTensor& tensor, ::ml_drift::DataType data_type) {
  return HostBufferDescriptor(tensor.data.raw, tensor.bytes,
                              tensor.allocation_type == kTfLiteCustom,
                              data_type);
}

HostBufferArbitrator::HostBufferArbitrator(GpuBackend* backend)
    : backend_(backend), options_() {
  ApplyBackendOptions();
}

HostBufferArbitrator::HostBufferArbitrator(GpuBackend* backend, Options options)
    : backend_(backend), options_(options) {
  ApplyBackendOptions();
}

void HostBufferArbitrator::SetBackend(GpuBackend* backend) {
  backend_ = backend;
  ApplyBackendOptions();
}

void HostBufferArbitrator::SetOptions(const Options& options) {
  options_ = options;
  ApplyBackendOptions();
}

void HostBufferArbitrator::ApplyBackendOptions() {
  if (backend_ != nullptr && options_.force_host_ptr_io_buffers) {
    backend_->SetForceHostPtrIOBuffer(true);
  }
}

bool HostBufferArbitrator::CanZeroCopy(const HostBufferDescriptor& desc,
                                       size_t alignment) {
  if (!desc.is_custom_allocation || desc.host_ptr == nullptr) {
    return false;
  }
  return (alignment > 0) &&
         (reinterpret_cast<uintptr_t>(desc.host_ptr) % alignment == 0);
}

bool HostBufferArbitrator::CanZeroCopy(const TfLiteTensor& tensor,
                                       size_t alignment) {
  return CanZeroCopy(HostBufferDescriptor::FromTfLiteTensor(tensor), alignment);
}

bool HostBufferArbitrator::IsHostPtrSupported(bool is_input) const {
  if (backend_ == nullptr) {
    return false;
  }
  const bool enable_host_ptr =
      options_.enable_host_ptr_io_buffers || options_.force_host_ptr_io_buffers;
  return enable_host_ptr && backend_->SupportsHostPtrIOBuffer(is_input);
}

bool HostBufferArbitrator::NeedsRecreateBuffer(
    const GpuIOBuffer* buffer, const HostBufferDescriptor& desc, bool is_input,
    const void* last_seen_ptr) const {
  // Check 1: Existence and size initialization check.
  // Recreate on cold initialization (buffer is null) or when a previously
  // zero-sized buffer is now populated with non-zero byte size.
  if (buffer == nullptr || (buffer->GetSize() == 0 && desc.bytes > 0)) {
    return true;
  }
  // Dynamic shape resizing is explicitly rejected by EnsureTensorConverters;
  // do not recreate buffers for mismatched sizes as InferenceContext is
  // statically compiled.
  if (buffer->GetSize() > 0 && buffer->GetSize() != desc.bytes) {
    return false;
  }

  // Check 2: Option and hardware capability resolution.
  // Host pointer buffers require non-zero size and backend driver support.
  const bool host_ptr_supported =
      (desc.bytes > 0) && IsHostPtrSupported(is_input);
  if (!host_ptr_supported) {
    // If host pointers are disabled/unsupported on this platform, recreate
    // ONLY if the existing buffer was previously a host pointer (downgrading
    // back to a standard device buffer).
    return buffer->IsHostPtr();
  }

  // Check 3: Host pointer requirement.
  // If host pointers are supported but the existing buffer is a device buffer,
  // it must be upgraded to a host pointer buffer.
  if (!buffer->IsHostPtr()) {
    return true;
  }

  // Check 4: Zero-copy eligibility detection.
  // To wrap caller memory directly with CL_MEM_USE_HOST_PTR, memory must be
  // explicitly registered via custom allocation and page-aligned (typically
  // 4096B).
  const size_t alignment =
      backend_ != nullptr ? backend_->GetHostPtrAlignment() : 0;
  const bool can_zero_copy = CanZeroCopy(desc, alignment);

  // If eligible for zero-copy and the existing buffer already wraps this exact
  // caller pointer without taking ownership, no recreation is needed.
  if (can_zero_copy && !buffer->OwnsHostPtr() &&
      buffer->GetHostPtr() == desc.host_ptr) {
    return false;
  }

  // If not eligible for zero-copy and the existing buffer is already an
  // internally owned staging buffer, reuse it without recreation.
  if (!can_zero_copy && buffer->OwnsHostPtr()) {
    return false;
  }

  // If the caller provided the exact same pointer address as the previous
  // resolution, nothing has changed. Recreate is false.
  // This holds whether wrapping succeeded (zero-copy) or fell back to an
  // internal staging buffer, eliminating infinite recreation loops.
  if (desc.host_ptr == last_seen_ptr) {
    return false;
  }

  if (can_zero_copy) {
    // A new or changed pointer was provided and qualifies for zero-copy:
    // Recreate to attempt wrapping the new pointer.
    return true;
  }

  // Caller's pointer changed, but it does NOT qualify for zero-copy (e.g.
  // unaligned). If we already have an internal fallback buffer, reuse it
  // without recreation (memcpy will copy from the new unaligned pointer into
  // the same fallback buffer).
  return !buffer->OwnsHostPtr();
}

bool HostBufferArbitrator::NeedsRecreateBuffer(
    const GpuIOBuffer* buffer, const TfLiteTensor& tensor, bool is_input,
    const void* last_seen_ptr) const {
  return NeedsRecreateBuffer(buffer,
                             HostBufferDescriptor::FromTfLiteTensor(tensor),
                             is_input, last_seen_ptr);
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
HostBufferArbitrator::CreateOrWrapBuffer(const HostBufferDescriptor& desc,
                                         bool is_input) {
  if (backend_ == nullptr) {
    return absl::InternalError("GpuBackend is null.");
  }
  if (desc.bytes == 0) {
    return backend_->CreateIOBufferWithSize(desc.data_type, 0, is_input);
  }
  if (IsHostPtrSupported(is_input)) {
    const size_t alignment = backend_->GetHostPtrAlignment();
    if (CanZeroCopy(desc, alignment)) {
      auto wrap_or = backend_->WrapExternalHostBuffer(
          desc.data_type, desc.bytes, is_input, desc.host_ptr);
      if (wrap_or.ok() && *wrap_or != nullptr) {
        return std::move(*wrap_or);
      }
      ABSL_LOG(WARNING)
          << "Failed to wrap external host pointer with zero-copy ("
          << (wrap_or.ok() ? "backend returned null"
                           : wrap_or.status().ToString())
          << "). Falling back to internally-owned staging buffer.";
    }

    auto internal_or = backend_->CreateInternalHostBuffer(desc.data_type,
                                                          desc.bytes, is_input);
    if (internal_or.ok() && *internal_or != nullptr) {
      return std::move(*internal_or);
    }
  }
  return backend_->CreateIOBufferWithSize(desc.data_type, desc.bytes, is_input);
}

absl::StatusOr<std::unique_ptr<GpuIOBuffer>>
HostBufferArbitrator::CreateOrWrapBuffer(const TfLiteTensor& tensor,
                                         ::ml_drift::DataType data_type,
                                         bool is_input) {
  return CreateOrWrapBuffer(
      HostBufferDescriptor::FromTfLiteTensor(tensor, data_type), is_input);
}

}  // namespace litert::ml_drift
