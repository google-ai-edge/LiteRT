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

#include "litert/runtime/ahwb_buffer.h"

#include <cstddef>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/ahwb_wrapper.h"
#include "litert/runtime/event.h"

namespace litert {
namespace internal {

using ::litert::internal::AndroidHardwareBufferWrapper;
auto AhwbWrapper = AndroidHardwareBufferWrapper::Instance;

bool AhwbBuffer::IsSupported() { return AhwbWrapper().Supported(); }

Expected<AhwbBuffer> AhwbBuffer::Alloc(size_t size) {
  if (!IsSupported()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AHardwareBuffers are not supported on this platform");
  }

  AHardwareBuffer* ahwb;
  AHardwareBuffer_Desc ahwb_desc = {
      .width = static_cast<uint32_t>(size),
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY |
               AHARDWAREBUFFER_USAGE_CPU_READ_RARELY |
               AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER};
  if (AhwbWrapper().Allocate(&ahwb_desc, &ahwb) != 0) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate AHWB");
  }
  return AhwbBuffer{/*.ahwb=*/ahwb};
}

void AhwbBuffer::Free(AHardwareBuffer* ahwb) {
  if (!IsSupported()) {
    LITERT_LOG(LITERT_ERROR,
               "AHardwareBuffers are not supported on this platform");
    return;
  }
  AhwbWrapper().Release(ahwb);
}

Expected<size_t> AhwbBuffer::GetSize(AHardwareBuffer* ahwb) {
  if (!IsSupported()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AHardwareBuffers are not supported on this platform");
  }

  AHardwareBuffer_Desc ahwb_desc;
  AhwbWrapper().Describe(ahwb, &ahwb_desc);
  return static_cast<size_t>(ahwb_desc.width) * ahwb_desc.height *
         ahwb_desc.layers;
}

Expected<void*> AhwbBuffer::Lock(AHardwareBuffer* ahwb, LiteRtEventT* event) {
  if (!IsSupported()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AHardwareBuffers are not supported on this platform");
  }
  int fence = -1;
  if (event != nullptr) {
    LITERT_ASSIGN_OR_RETURN(fence, event->GetSyncFenceFd());
  }
  void* host_addr;
  LITERT_RETURN_IF_ERROR(
      AhwbWrapper().Lock(ahwb,
                         AHARDWAREBUFFER_USAGE_CPU_READ_RARELY |
                             AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
                         fence, /*rect=*/nullptr, &host_addr) == 0,
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to lock AHWB"));
  return host_addr;
}

Expected<void> AhwbBuffer::Unlock(AHardwareBuffer* ahwb) {
  if (!IsSupported()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "AHardwareBuffers are not supported on this platform");
  }
  if (AhwbWrapper().Unlock(ahwb, /*fence=*/nullptr) != 0) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to unlock AHWB");
  }
  return {};
}

}  // namespace internal
}  // namespace litert
