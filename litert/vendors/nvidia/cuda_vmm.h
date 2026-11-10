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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_CUDA_VMM_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_CUDA_VMM_H_

#include <cstdint>
#include <memory>

#include "litert/cc/litert_expected.h"

namespace litert::nvidia {

// CUDA virtual memory management (cuMemCreate, cuMemMap, ...) for device
// memory that TensorRT maps into the weight memory of several engines
// (nvinfer1::IWeightsManager::restoreFromVmmAllocation). The entry points
// belong to the CUDA driver library, which the runtime library loads itself;
// they are looked up there at first use. A thread that creates a block needs
// a current CUDA context, which any CUDA runtime call that touches the device
// gives it; any thread may destroy the block.

// The size every allocation and mapping is a multiple of on the current
// device (2 MiB on the GPUs seen so far). An error if the driver library
// lacks the virtual memory API.
Expected<uint64_t> CudaVmmGranule();

// One physical allocation on the current device with a read-write mapping of
// its own for filling it. Copy into it with cudaMemcpyAsync on a stream, not
// with the synchronous cudaMemcpy: with driver 596.49 under WSL2 a
// synchronous copy into a mapping made after another block was unmapped
// fails with cudaErrorIllegalAddress every few dozen rounds, which a stream
// copy never did.
class CudaVmmBlock {
 public:
  // `size` is a positive multiple of CudaVmmGranule().
  static Expected<std::unique_ptr<CudaVmmBlock>> Create(uint64_t size);

  ~CudaVmmBlock();
  CudaVmmBlock(const CudaVmmBlock&) = delete;
  CudaVmmBlock& operator=(const CudaVmmBlock&) = delete;

  // The numeric value of the CUmemGenericAllocationHandle.
  uint64_t handle() const { return handle_; }
  // The device address of the block's own mapping.
  void* address() const { return reinterpret_cast<void*>(address_); }
  uint64_t size() const { return size_; }

 private:
  CudaVmmBlock() = default;

  void* context_ = nullptr;  // CUcontext
  uint64_t handle_ = 0;
  uint64_t address_ = 0;
  uint64_t size_ = 0;
  bool mapped_ = false;
};

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_CUDA_VMM_H_
