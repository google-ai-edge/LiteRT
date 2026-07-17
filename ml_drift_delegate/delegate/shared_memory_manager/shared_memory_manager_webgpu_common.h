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
#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "ml_drift/common/executor.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
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

enum class UploadScheduling {
  kAllowInline,
  kRequireExecutor,
};

absl::Status CopyBufferToBuffer(
    const webgpu::ExecutionEnvironment* env, const TensorDescriptor& desc,
    size_t page_adjusted_offset,
    ml_drift_delegate::ReleaseDataCallback release_data_callback,
    webgpu::SpatialTensor* tensor);

absl::Status CreateSharedWebGpuTensor(
    const webgpu::ExecutionEnvironment& env, TensorDescriptor& tensor_desc,
    size_t page_adjusted_offset,
    ml_drift_delegate::ReleaseDataCallback release_data_callback,
    bool has_prepacked_tflite_tensors, Executor* upload_executor,
    UploadScheduling upload_scheduling,
    std::unique_ptr<GpuSpatialTensor>& tensor);

}  // namespace webgpu_internal
}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_ML_DRIFT_DELEGATE_SHARED_MEMORY_MANAGER_SHARED_MEMORY_MANAGER_WEBGPU_COMMON_H_
