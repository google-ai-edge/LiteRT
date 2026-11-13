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

// Minimal fake OpenCL shared library (SONAME libOpenCL.so) for unit testing
// automatic OpenCL-vs-WebGPU runtime selection without a physical GPU.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/base/attributes.h"  // from @com_google_absl
#include <CL/cl.h>
#include <CL/cl_platform.h>

namespace {

ABSL_CONST_INIT std::atomic<unsigned int> fake_platform_count{1};
ABSL_CONST_INIT std::atomic<int> get_platform_ids_calls{0};
ABSL_CONST_INIT std::atomic<int> create_context_calls{0};

cl_int CopyStringInfo(const char* str, size_t param_value_size,
                      void* param_value, size_t* param_value_size_ret) {
  const size_t size = std::strlen(str) + 1;
  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = size;
  }
  if (param_value != nullptr) {
    if (param_value_size < size) {
      return CL_INVALID_VALUE;
    }
    std::memcpy(param_value, str, size);
  }
  return CL_SUCCESS;
}

template <typename T>
cl_int CopyScalarInfo(const T& value, size_t param_value_size,
                      void* param_value, size_t* param_value_size_ret) {
  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = sizeof(T);
  }
  if (param_value != nullptr) {
    if (param_value_size < sizeof(T)) {
      return CL_INVALID_VALUE;
    }
    std::memcpy(param_value, &value, sizeof(T));
  }
  return CL_SUCCESS;
}

}  // namespace

#define FAKE_OPENCL_EXPORT __attribute__((visibility("default")))

extern "C" {

FAKE_OPENCL_EXPORT void FakeOpenClSetPlatformCount(unsigned int count) {
  fake_platform_count.store(count, std::memory_order_relaxed);
}

FAKE_OPENCL_EXPORT int FakeOpenClGetPlatformIdsCalls() {
  return get_platform_ids_calls.load(std::memory_order_relaxed);
}

FAKE_OPENCL_EXPORT int FakeOpenClGetCreateContextCalls() {
  return create_context_calls.load(std::memory_order_relaxed);
}

FAKE_OPENCL_EXPORT cl_int CL_API_CALL clGetPlatformIDs(
    cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
  get_platform_ids_calls.fetch_add(1, std::memory_order_relaxed);
  const cl_uint count = fake_platform_count.load(std::memory_order_relaxed);
  if (num_platforms != nullptr) {
    *num_platforms = count;
  }
  if (platforms != nullptr && count > 0) {
    if (num_entries == 0) {
      return CL_INVALID_VALUE;
    }
    platforms[0] = reinterpret_cast<cl_platform_id>(static_cast<uintptr_t>(1));
  }
  return CL_SUCCESS;
}

FAKE_OPENCL_EXPORT cl_int CL_API_CALL clGetPlatformInfo(
    cl_platform_id /*platform*/, cl_platform_info param_name,
    size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
  if (param_name == CL_PLATFORM_VERSION) {
    return CopyStringInfo("OpenCL 2.0 FakePlatform", param_value_size,
                          param_value, param_value_size_ret);
  }
  return CopyStringInfo("", param_value_size, param_value,
                        param_value_size_ret);
}

FAKE_OPENCL_EXPORT cl_int CL_API_CALL clGetDeviceIDs(
    cl_platform_id /*platform*/, cl_device_type device_type,
    cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {
  if ((device_type & (CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ALL)) == 0) {
    if (num_devices != nullptr) {
      *num_devices = 0;
    }
    return CL_DEVICE_NOT_FOUND;
  }
  if (num_devices != nullptr) {
    *num_devices = 1;
  }
  if (devices != nullptr) {
    if (num_entries == 0) {
      return CL_INVALID_VALUE;
    }
    devices[0] = reinterpret_cast<cl_device_id>(static_cast<uintptr_t>(2));
  }
  return CL_SUCCESS;
}

FAKE_OPENCL_EXPORT cl_int CL_API_CALL clGetDeviceInfo(
    cl_device_id /*device*/, cl_device_info param_name, size_t param_value_size,
    void* param_value, size_t* param_value_size_ret) {
  switch (param_name) {
    case CL_DEVICE_NAME:
      return CopyStringInfo("FakeOpenCL GPU", param_value_size, param_value,
                            param_value_size_ret);
    case CL_DEVICE_VENDOR:
      return CopyStringInfo("FakeVendor", param_value_size, param_value,
                            param_value_size_ret);
    case CL_DEVICE_OPENCL_C_VERSION:
      return CopyStringInfo("OpenCL C 2.0 ", param_value_size, param_value,
                            param_value_size_ret);
    case CL_DRIVER_VERSION:
      return CopyStringInfo("1.0", param_value_size, param_value,
                            param_value_size_ret);
    case CL_DEVICE_EXTENSIONS:
      return CopyStringInfo("", param_value_size, param_value,
                            param_value_size_ret);
    case CL_DEVICE_IMAGE_SUPPORT:
      return CopyScalarInfo<cl_bool>(CL_TRUE, param_value_size, param_value,
                                     param_value_size_ret);
    case CL_DEVICE_SINGLE_FP_CONFIG:
      return CopyScalarInfo<cl_device_fp_config>(CL_FP_ROUND_TO_NEAREST,
                                                 param_value_size, param_value,
                                                 param_value_size_ret);
    case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
      return CopyScalarInfo<cl_uint>(3, param_value_size, param_value,
                                     param_value_size_ret);
    case CL_DEVICE_MAX_WORK_ITEM_SIZES: {
      const size_t sizes[3] = {64, 64, 64};
      return CopyScalarInfo(sizes, param_value_size, param_value,
                            param_value_size_ret);
    }
    case CL_DEVICE_MAX_WORK_GROUP_SIZE:
      return CopyScalarInfo<size_t>(64, param_value_size, param_value,
                                    param_value_size_ret);
    default:
      if (param_value_size_ret != nullptr) {
        *param_value_size_ret =
            param_value_size > 0 ? param_value_size : sizeof(cl_ulong);
      }
      if (param_value != nullptr && param_value_size > 0) {
        std::memset(param_value, 0, param_value_size);
      }
      return CL_SUCCESS;
  }
}

FAKE_OPENCL_EXPORT void* CL_API_CALL clGetExtensionFunctionAddressForPlatform(
    cl_platform_id /*platform*/, const char* /*funcname*/) {
  return nullptr;
}

FAKE_OPENCL_EXPORT cl_context CL_API_CALL clCreateContext(
    const cl_context_properties* /*properties*/, cl_uint /*num_devices*/,
    const cl_device_id* /*devices*/,
    void(CL_CALLBACK* /*pfn_notify*/)(const char*, const void*, size_t, void*),
    void* /*user_data*/, cl_int* errcode_ret) {
  create_context_calls.fetch_add(1, std::memory_order_relaxed);
  if (errcode_ret != nullptr) {
    *errcode_ret = CL_INVALID_CONTEXT;
  }
  return nullptr;
}

}  // extern "C"
