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

#include "litert/vendors/nvidia/cuda_vmm.h"

#include <dlfcn.h>

#include <cstdint>
#include <memory>
#include <string>

#include "third_party/gpus/cuda/include/cuda.h"
#include "cuda_runtime_api.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::nvidia {
namespace {

// The driver entry points used here. The CUDA runtime library does not export
// them, and linking the driver library's stub would add a build-time
// dependency that the rest of the NVIDIA backend does not have.
struct DriverApi {
  decltype(&cuMemGetAllocationGranularity) get_allocation_granularity = nullptr;
  decltype(&cuMemCreate) create = nullptr;
  decltype(&cuMemRelease) release = nullptr;
  decltype(&cuMemAddressReserve) address_reserve = nullptr;
  decltype(&cuMemAddressFree) address_free = nullptr;
  decltype(&cuMemMap) map = nullptr;
  decltype(&cuMemUnmap) unmap = nullptr;
  decltype(&cuMemSetAccess) set_access = nullptr;
  decltype(&cuCtxGetCurrent) context_get_current = nullptr;
  decltype(&cuCtxPushCurrent) context_push_current = nullptr;
  decltype(&cuCtxPopCurrent) context_pop_current = nullptr;
  decltype(&cuGetErrorName) get_error_name = nullptr;
};

const DriverApi* GetDriverApi() {
  static const DriverApi* const api = []() -> const DriverApi* {
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (library == nullptr) {
      return nullptr;
    }
    auto* loaded = new DriverApi();
    const auto resolve = [&](auto& function, const char* name) {
      function = reinterpret_cast<std::decay_t<decltype(function)>>(
          dlsym(library, name));
      return function != nullptr;
    };
    if (!resolve(loaded->get_allocation_granularity,
                 "cuMemGetAllocationGranularity") ||
        !resolve(loaded->create, "cuMemCreate") ||
        !resolve(loaded->release, "cuMemRelease") ||
        !resolve(loaded->address_reserve, "cuMemAddressReserve") ||
        !resolve(loaded->address_free, "cuMemAddressFree") ||
        !resolve(loaded->map, "cuMemMap") ||
        !resolve(loaded->unmap, "cuMemUnmap") ||
        !resolve(loaded->set_access, "cuMemSetAccess") ||
        !resolve(loaded->context_get_current, "cuCtxGetCurrent") ||
        !resolve(loaded->context_push_current, "cuCtxPushCurrent_v2") ||
        !resolve(loaded->context_pop_current, "cuCtxPopCurrent_v2") ||
        !resolve(loaded->get_error_name, "cuGetErrorName")) {
      delete loaded;
      return nullptr;
    }
    return loaded;
  }();
  return api;
}

Unexpected DriverError(const DriverApi& api, const char* call,
                       CUresult result) {
  const char* name = nullptr;
  api.get_error_name(result, &name);
  return Error(kLiteRtStatusErrorRuntimeFailure,
               std::string(call) + " failed: " +
                   (name != nullptr ? name : "unknown CUDA driver error"));
}

Expected<CUmemAllocationProp> AllocationProperties() {
  int device = 0;
  const cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        std::string("cudaGetDevice failed: ") + cudaGetErrorString(status));
  }
  CUmemAllocationProp properties{};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device;
  return properties;
}

}  // namespace

Expected<uint64_t> CudaVmmGranule() {
  const DriverApi* api = GetDriverApi();
  if (api == nullptr) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "The CUDA driver library lacks the virtual memory API");
  }
  LITERT_ASSIGN_OR_RETURN(auto properties, AllocationProperties());
  size_t granule = 0;
  CUresult result = api->get_allocation_granularity(
      &granule, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (result == CUDA_ERROR_NOT_INITIALIZED ||
      result == CUDA_ERROR_INVALID_CONTEXT) {
    // The first device call of the process: let the runtime library
    // initialize the driver and the context of this thread.
    if (cudaDeviceSynchronize() == cudaSuccess) {
      result = api->get_allocation_granularity(
          &granule, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    }
  }
  if (result != CUDA_SUCCESS) {
    return DriverError(*api, "cuMemGetAllocationGranularity", result);
  }
  if (granule == 0 || (granule & (granule - 1)) != 0) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Unexpected CUDA virtual memory granularity");
  }
  return static_cast<uint64_t>(granule);
}

Expected<std::unique_ptr<CudaVmmBlock>> CudaVmmBlock::Create(uint64_t size) {
  const DriverApi* api = GetDriverApi();
  if (api == nullptr) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "The CUDA driver library lacks the virtual memory API");
  }
  LITERT_ASSIGN_OR_RETURN(const uint64_t granule, CudaVmmGranule());
  if (size == 0 || size % granule != 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "A CUDA virtual memory block is a multiple of the granule");
  }
  LITERT_ASSIGN_OR_RETURN(auto properties, AllocationProperties());
  std::unique_ptr<CudaVmmBlock> block(new CudaVmmBlock());
  block->size_ = size;
  CUcontext context = nullptr;
  CUresult result = api->context_get_current(&context);
  if (result != CUDA_SUCCESS || context == nullptr) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "The thread has no current CUDA context");
  }
  block->context_ = context;
  CUmemGenericAllocationHandle handle = 0;
  result = api->create(&handle, size, &properties, 0);
  if (result != CUDA_SUCCESS) {
    return DriverError(*api, "cuMemCreate", result);
  }
  block->handle_ = handle;
  CUdeviceptr address = 0;
  result = api->address_reserve(&address, size, granule, 0, 0);
  if (result != CUDA_SUCCESS) {
    return DriverError(*api, "cuMemAddressReserve", result);
  }
  block->address_ = address;
  result = api->map(address, size, 0, handle, 0);
  if (result != CUDA_SUCCESS) {
    return DriverError(*api, "cuMemMap", result);
  }
  block->mapped_ = true;
  CUmemAccessDesc access{};
  access.location = properties.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  result = api->set_access(address, size, &access, 1);
  if (result != CUDA_SUCCESS) {
    return DriverError(*api, "cuMemSetAccess", result);
  }
  return block;
}

CudaVmmBlock::~CudaVmmBlock() {
  const DriverApi* api = GetDriverApi();
  if (api == nullptr || context_ == nullptr) {
    return;
  }
  // The thread that destroys the block may never have used CUDA: release it
  // in the context it was created in.
  if (api->context_push_current(static_cast<CUcontext>(context_)) !=
      CUDA_SUCCESS) {
    return;
  }
  if (mapped_) {
    api->unmap(address_, size_);
  }
  if (address_ != 0) {
    api->address_free(address_, size_);
  }
  if (handle_ != 0) {
    api->release(handle_);
  }
  CUcontext popped = nullptr;
  api->context_pop_current(&popped);
}

}  // namespace litert::nvidia
