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

#ifndef THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_WEBGPU_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_WEBGPU_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager.h"

namespace ml_drift {
namespace webgpu {
class NotifiedTensor : public SpatialTensor {
 public:
#ifndef __EMSCRIPTEN__
  void set_upload_notification(
      std::shared_ptr<absl::Notification> upload_notification) {
    upload_notification_ = std::move(upload_notification);
  }

  std::shared_ptr<absl::Notification> upload_notification() const {
    return upload_notification_;
  }
#endif  // !__EMSCRIPTEN__

 private:
#ifndef __EMSCRIPTEN__
  std::shared_ptr<absl::Notification> upload_notification_;
#endif  // !__EMSCRIPTEN__
};
}  // namespace webgpu

namespace webgpu_internal {

// Copies the first `num_rows` rows of `row_size` bytes of `src`, which are
// tightly packed, into `dst`, where row i starts at byte i * `dst_row_pitch`.
// Bytes of `src` after the rows are ignored and the bytes between the rows of
// `dst` are left unchanged. This is the staging layout that
// CopyBufferToTexture() needs when `dst_row_pitch` is the 256-byte aligned
// bytesPerRow.
//
// Returns InvalidArgumentError unless src.size() >= num_rows * row_size,
// dst_row_pitch >= row_size and `dst` holds the last row.
absl::Status CopyRowsWithPitch(absl::Span<const uint8_t> src, size_t row_size,
                               size_t num_rows, size_t dst_row_pitch,
                               absl::Span<uint8_t> dst);

// Layout of the staging buffer that CopyBufferToTexture() reads when uploading
// texture data.
struct TextureStagingLayout {
  // Size of a tightly packed row: texel size * width.
  uint32_t bytes_per_row = 0;
  // bytes_per_row aligned to 256 bytes, as bytesPerRow requires.
  uint32_t bytes_per_row_aligned = 0;
  // Rows of all depth slices: height * depth.
  size_t num_rows = 0;
  // Bytes CopyBufferToTexture() reads: all rows at the aligned pitch, except
  // that only bytes_per_row bytes are needed for the last row.
  size_t min_buffer_size = 0;
  // Whether the rows must be copied at the aligned pitch with
  // CopyRowsWithPitch(). Otherwise the data is copied as is.
  bool repack_rows = false;
};

// Returns the staging layout for uploading `data_size` bytes of texture data
// covering `region` (width, height, depth) with texels of `texel_size` bytes.
// The data is copied as is, like it always was, unless that would upload the
// wrong texels:
//
// - Rows that are a multiple of 256 bytes: copied as is.
// - `data_size` is min_buffer_size or num_rows * bytes_per_row_aligned, i.e.
//   the data is already padded to the aligned pitch: copied as is.
// - Otherwise, `data_size` >= num_rows * bytes_per_row: the rows are tightly
//   packed, possibly followed by bytes that are not uploaded, and are repacked
//   to the aligned pitch.
//
// Returns InvalidArgumentError if the region is empty, the sizes overflow, or
// the rows are not a multiple of 256 bytes and `data_size` is smaller than
// num_rows * bytes_per_row.
absl::StatusOr<TextureStagingLayout> GetTextureStagingLayout(const int3& region,
                                                             size_t texel_size,
                                                             size_t data_size);

enum class UploadScheduling {
  kAllowInline,
  kRequireExecutor,
};

absl::Status CopyBufferToBuffer(
    const webgpu::ExecutionEnvironment* env, const TensorDescriptor& desc,
    size_t page_adjusted_offset,
    ::litert::ml_drift::ReleaseDataCallback release_data_callback,
    webgpu::SpatialTensor* tensor);

// Creates `tensor`, which must be null, from `tensor_desc` and uploads its
// data. On failure, `tensor` is reset to null so that the caller can try again,
// e.g. from other data.
absl::Status CreateSharedWebGpuTensor(
    const webgpu::ExecutionEnvironment& env, TensorDescriptor& tensor_desc,
    size_t page_adjusted_offset,
    ::litert::ml_drift::ReleaseDataCallback release_data_callback,
    bool has_prepacked_tflite_tensors, Executor* upload_executor,
    UploadScheduling upload_scheduling,
    std::unique_ptr<GpuSpatialTensor>& tensor);

}  // namespace webgpu_internal
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_WEBGPU_COMMON_H_
