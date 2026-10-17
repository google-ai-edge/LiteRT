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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/intel_openvino/dispatch/openvino_shared_core.h"
#include "litert/vendors/intel_openvino/utils.h"

namespace {

// Converts the tensor type's dimensions to an ov::Shape, rejecting ranks that
// exceed the fixed-size `dimensions` array.
litert::Expected<ov::Shape> ToOvShape(
    const LiteRtRankedTensorType& tensor_type) {
  if (tensor_type.layout.rank > LITERT_TENSOR_MAX_RANK) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Tensor rank exceeds LITERT_TENSOR_MAX_RANK.");
  }
  std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
  for (size_t i = 0; i < ov_shape_vec.size(); i++)
    ov_shape_vec[i] = tensor_type.layout.dimensions[i];
  return ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()};
}

}  // namespace

litert::Expected<void> OpenVinoTensorBuffer::Alloc(
    const LiteRtRankedTensorType& tensor_type, size_t size) {
  if (allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The OpenVino tensor has been allocated.");
  }

  // TODO:: Release the shared OpenVINO Core.
  std::string device = OpenVINOSharedCore::GetInstance()->GetDevice();
  ov::element::Type ov_element_type =
      litert::openvino::MapLiteTypeToOV(tensor_type.element_type);

  LITERT_ASSIGN_OR_RETURN(ov::Shape ov_shape, ToOvShape(tensor_type));

  if (device == "NPU" || device == "GPU") {
    auto context = OpenVINOSharedCore::GetInstance()->GetRemoteContext();
    ov_tensor_ = context.create_host_tensor(ov_element_type, ov_shape);
  } else if (device == "CPU") {
    ov_tensor_ = ov::Tensor(ov_element_type, ov_shape);
  } else {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Unsupported OpenVINO device: " + device);
  }
  allocated_ = true;

  return {};
}

litert::Expected<void> OpenVinoTensorBuffer::Wrap(
    const LiteRtRankedTensorType& tensor_type, void* host_memory, size_t size) {
  if (allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The OpenVino tensor has been allocated.");
  }
  if (host_memory == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Cannot wrap null host memory.");
  }

  ov::element::Type ov_element_type =
      litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
  LITERT_ASSIGN_OR_RETURN(ov::Shape ov_shape, ToOvShape(tensor_type));

  // ov::Tensor over an external pointer never frees it, so destroying this
  // wrapper leaves the owner's memory intact.
  ov_tensor_ = ov::Tensor(ov_element_type, ov_shape, host_memory);
  if (ov_tensor_.get_byte_size() > size) {
    ov_tensor_ = ov::Tensor();
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Wrapped memory is smaller than the tensor.");
  }
  allocated_ = true;
  return {};
}

litert::Expected<void*> OpenVinoTensorBuffer::GetTensorData() {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The tensor didn't allocate.");
  }
  return ov_tensor_.data();
}

litert::Expected<ov::Tensor> OpenVinoTensorBuffer::GetOVTensor() {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Failed to get tensor.");
  }
  return ov_tensor_;
}
