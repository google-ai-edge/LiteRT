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

#include "litert/vendors/intel_openvino/dispatch/openvino_tensor_buffer.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/vendors/intel_openvino/dispatch/openvino_shared_core.h"
#include "litert/vendors/intel_openvino/utils.h"

litert::Expected<void> OpenVinoTensorBuffer::Alloc(
    const LiteRtRankedTensorType& tensor_type, size_t size) {
  absl::MutexLock lock(&mutex_);
  if (allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The OpenVino tensor has been allocated.");
  }
  const std::string device = OpenVINOSharedCore::GetInstance()->GetDevice();
  if (device != "NPU" && device != "GPU" && device != "CPU") {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Unsupported OpenVINO device: " + device);
  }
  element_type_ = litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
  std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
  for (size_t i = 0; i < ov_shape_vec.size(); i++)
    ov_shape_vec[i] = tensor_type.layout.dimensions[i];
  shape_ = ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()};
  allocated_ = true;
  return {};
}

litert::Expected<void> OpenVinoTensorBuffer::Wrap(
    const LiteRtRankedTensorType& tensor_type, void* host_memory, size_t size) {
  absl::MutexLock lock(&mutex_);
  if (allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The OpenVino tensor has been allocated.");
  }
  if (host_memory == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Cannot wrap null host memory.");
  }

  element_type_ = litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
  std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
  for (size_t i = 0; i < ov_shape_vec.size(); i++)
    ov_shape_vec[i] = tensor_type.layout.dimensions[i];
  shape_ = ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()};

  // ov::Tensor over an external pointer never frees it, so destroying this
  // wrapper leaves the owner's memory intact.
  ov_tensor_ = ov::Tensor(element_type_, shape_, host_memory);
  if (ov_tensor_.get_byte_size() > size) {
    ov_tensor_ = ov::Tensor();
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Wrapped memory is smaller than the tensor.");
  }
  allocated_ = true;
  wrapped_ = true;
  AnyWrappedFlag().store(true);
  return {};
}

bool OpenVinoTensorBuffer::AnyWrapped() { return AnyWrappedFlag().load(); }

std::atomic<bool>& OpenVinoTensorBuffer::AnyWrappedFlag() {
  static std::atomic<bool> any_wrapped{false};
  return any_wrapped;
}

litert::Expected<void> OpenVinoTensorBuffer::Materialize(const char* why) {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The tensor didn't allocate.");
  }
  if (ov_tensor_) return {};
  // TODO:: Release the shared OpenVINO Core.
  const std::string device = OpenVINOSharedCore::GetInstance()->GetDevice();
  if (device == "NPU" || device == "GPU") {
    auto context = OpenVINOSharedCore::GetInstance()->GetRemoteContext();
    ov_tensor_ = context.create_host_tensor(element_type_, shape_);
  } else {
    ov_tensor_ = ov::Tensor(element_type_, shape_);
  }
  std::memset(ov_tensor_.data(), 0, ov_tensor_.get_byte_size());
  LITERT_LOG(LITERT_VERBOSE, "OV buffer %s created on %s (%zu B)",
             shape_.to_string().c_str(), why, ov_tensor_.get_byte_size());
  return {};
}

litert::Expected<void*> OpenVinoTensorBuffer::GetTensorData() {
  absl::MutexLock lock(&mutex_);
  LITERT_RETURN_IF_ERROR(Materialize("lock"));
  return ov_tensor_.data();
}

litert::Expected<void> OpenVinoTensorBuffer::Clear() {
  absl::MutexLock lock(&mutex_);
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The tensor didn't allocate.");
  }
  if (ov_tensor_) std::memset(ov_tensor_.data(), 0, ov_tensor_.get_byte_size());
  return {};
}

litert::Expected<ov::Tensor> OpenVinoTensorBuffer::GetOVTensor() {
  absl::MutexLock lock(&mutex_);
  LITERT_RETURN_IF_ERROR(Materialize("register"));
  return ov_tensor_;
}
