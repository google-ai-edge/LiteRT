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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_COOPERATIVE_MATRIX_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_COOPERATIVE_MATRIX_H_

#include <cstddef>
#include <type_traits>

#include "tflite/delegates/gpu/cl/opencl_wrapper.h"
#include "tflite/delegates/gpu/common/gpu_info.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace cooperative_matrix {

// Compatibility with the OpenCL headers pinned by LiteRT, which predate this
// extension. These declarations mirror the Khronos draft, not vendor guesses:
// OpenCL-Headers commit 8625d678fe52409580284a306e929cbd509836d0, CL/cl_ext.h
// (https://github.com/KhronosGroup/OpenCL-Headers/pull/305).
// Keep the adapter private to this namespace and validate its ABI when building
// with headers that already declare the extension.
constexpr cl_uint kDefaultSubGroupVariants = 0x1078;
constexpr cl_device_info kPointerAlignment = 0x107A;
constexpr cl_device_info kStrideMultiple = 0x107B;
constexpr cl_uint kFloat16 = 0;
constexpr cl_uint kFloat32 = 1;

struct Variant {
  cl_uint m_size;
  cl_uint n_size;
  cl_uint k_size;
  cl_uint a_type;
  cl_uint b_type;
  cl_uint c_type;
  cl_uint result_type;
  cl_bool saturating_accumulation;
};

using GetInfo = cl_int(CL_API_CALL*)(cl_device_id, cl_uint, size_t, const void*,
                                    size_t, void*, size_t*);

#if defined(cl_khr_cooperative_matrix)
static_assert(kDefaultSubGroupVariants ==
              CL_DEVICE_COOPERATIVE_MATRIX_DEFAULT_SUB_GROUP_VARIANTS_KHR);
static_assert(kPointerAlignment ==
              CL_DEVICE_COOPERATIVE_MATRIX_POINTER_ALIGNMENT_KHR);
static_assert(kStrideMultiple ==
              CL_DEVICE_COOPERATIVE_MATRIX_STRIDE_MULTIPLE_KHR);
static_assert(kFloat16 == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP16_KHR);
static_assert(kFloat32 == CL_DEVICE_COOPERATIVE_MATRIX_COMPONENT_TYPE_FP32_KHR);
static_assert(std::is_same<GetInfo,
                          clGetDeviceCooperativeMatrixInfoKHR_fn>::value);
static_assert(sizeof(Variant) == sizeof(cl_device_cooperative_matrix_variant_khr));
#define TFLITE_CHECK_COOP_MATRIX_OFFSET(field)                                \
  static_assert(offsetof(Variant, field) ==                                   \
                offsetof(cl_device_cooperative_matrix_variant_khr, field))
TFLITE_CHECK_COOP_MATRIX_OFFSET(m_size);
TFLITE_CHECK_COOP_MATRIX_OFFSET(n_size);
TFLITE_CHECK_COOP_MATRIX_OFFSET(k_size);
TFLITE_CHECK_COOP_MATRIX_OFFSET(a_type);
TFLITE_CHECK_COOP_MATRIX_OFFSET(b_type);
TFLITE_CHECK_COOP_MATRIX_OFFSET(c_type);
TFLITE_CHECK_COOP_MATRIX_OFFSET(result_type);
TFLITE_CHECK_COOP_MATRIX_OFFSET(saturating_accumulation);
#undef TFLITE_CHECK_COOP_MATRIX_OFFSET
#endif

// Caches supported default-subgroup variants and alignment requirements.
// Missing entry points, failed queries, and malformed results leave it empty.
// Callback parameters allow discovery to be tested without an OpenCL device.
void ReadInfo(cl_device_id device, PFN_clGetDeviceInfo get_device_info,
              GetInfo get_matrix_info, OpenClInfo* info);

}  // namespace cooperative_matrix
}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_COOPERATIVE_MATRIX_H_
