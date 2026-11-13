/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tflite/delegates/gpu/cl/cooperative_matrix.h"

#include <vector>

#include "tflite/delegates/gpu/common/data_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace cooperative_matrix {
namespace {

DataType ComponentType(cl_uint component) {
  // Order is the cl_device_cooperative_matrix_component_type_khr enum.
  constexpr DataType types[] = {
      DataType::FLOAT16, DataType::FLOAT32, DataType::FLOAT64,
      DataType::INT8, DataType::INT16, DataType::INT32, DataType::INT64,
      DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64};
  return component < sizeof(types) / sizeof(types[0])
             ? types[component] : DataType::UNKNOWN;
}

}  // namespace

void ReadInfo(cl_device_id device, PFN_clGetDeviceInfo get_device_info,
              GetInfo get_matrix_info, OpenClInfo* info) {
  info->cooperative_matrix_configs.clear();
  info->cooperative_matrix_pointer_alignment = 0;
  info->cooperative_matrix_stride_multiple = 0;
  if (!get_matrix_info || !get_device_info) return;

  cl_uint alignment = 0;
  cl_uint stride_multiple = 0;
  if (get_device_info(device, kPointerAlignment, sizeof(alignment), &alignment,
                      nullptr) != CL_SUCCESS ||
      get_device_info(device, kStrideMultiple, sizeof(stride_multiple),
                      &stride_multiple, nullptr) != CL_SUCCESS ||
      alignment == 0 || stride_multiple == 0) {
    return;
  }

  size_t bytes = 0;
  if (get_matrix_info(device, kDefaultSubGroupVariants, 0, nullptr, 0, nullptr,
                      &bytes) != CL_SUCCESS ||
      bytes == 0 || bytes % sizeof(Variant) != 0) {
    return;
  }
  std::vector<Variant> variants(bytes / sizeof(Variant));
  size_t returned_bytes = 0;
  if (get_matrix_info(device, kDefaultSubGroupVariants, 0, nullptr, bytes,
                      variants.data(), &returned_bytes) != CL_SUCCESS ||
      returned_bytes != bytes) {
    return;
  }

  info->cooperative_matrix_pointer_alignment = alignment;
  info->cooperative_matrix_stride_multiple = stride_multiple;
  for (const auto& variant : variants) {
    if (variant.saturating_accumulation != CL_FALSE &&
        variant.saturating_accumulation != CL_TRUE) {
      continue;
    }
    info->cooperative_matrix_configs.push_back(
        {variant.m_size, variant.n_size, variant.k_size,
         ComponentType(variant.a_type), ComponentType(variant.b_type),
         ComponentType(variant.c_type), ComponentType(variant.result_type),
         variant.saturating_accumulation == CL_TRUE});
  }
}

}  // namespace cooperative_matrix
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
