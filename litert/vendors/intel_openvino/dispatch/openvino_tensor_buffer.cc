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
#include <memory>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/intel_openvino/dispatch/openvino_shared_core.h"
#include "litert/vendors/intel_openvino/utils.h"

litert::Expected<void> OpenVinoTensorBuffer::Alloc(
    const LiteRtRankedTensorType& tensor_type, size_t size) {
  if (allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The OpenVino tensor has been allocated.");
  }

  // TODO:: Release the shared OpenVINO Core.
  std::shared_ptr<ov::Core> core = OpenVINOSharedCore::GetInstance()->getCore();
  auto context = core->get_default_context("NPU");
  ov::element::Type ov_element_type =
      litert::openvino::MapLiteTypeToOV(tensor_type.element_type);
  std::vector<int32_t> ov_shape_vec(tensor_type.layout.rank);
  for (size_t i = 0; i < ov_shape_vec.size(); i++)
    ov_shape_vec[i] = tensor_type.layout.dimensions[i];
  host_tensor_ = context.create_host_tensor(
       ov_element_type, ov::Shape{ov_shape_vec.begin(), ov_shape_vec.end()});
  allocated_ = true;

  return {};
}

litert::Expected<void*> OpenVinoTensorBuffer::GetTensorData() {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "The remote tensor didn't allocate.");
  }
  return host_tensor_.data();
}

litert::Expected<ov::Tensor> OpenVinoTensorBuffer::GetOVTensor() {
  if (!allocated_) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument,
                              "Failed to get zero buffer remote tensor.");
  }
  return host_tensor_;
}
