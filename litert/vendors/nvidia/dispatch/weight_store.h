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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_WEIGHT_STORE_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_WEIGHT_STORE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "cuda_runtime_api.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/cuda_vmm.h"
#include "NvInferRuntime.h"

namespace litert::nvidia {

// The device memory behind the weights of an engine whose plan was built
// without its packed plugin weights (bytecode.h, TensorRtWeightStore): a
// block per segment and a block per stretch of the engine's weight data
// between the segments.
//
// A segment block is filled from the model file and shared: every engine of
// the process whose store has a segment with the same key holds the same
// block, so the prefill and the decode engine of a model keep one copy of
// the weights they both read. LITERT_NVIDIA_DISPATCH_SHARED_WEIGHT_SEGMENTS=0
// gives every engine blocks of its own instead.
class EngineWeightMemory {
 public:
  // Allocates and fills the memory: the private stretches from the store's
  // runs, and from the model file the segments that no other engine holds.
  // The copies are ordered on `stream`, which is synchronized before this
  // returns. The store is not needed afterwards.
  static Expected<std::unique_ptr<EngineWeightMemory>> Create(
      const TensorRtWeightStore& store, cudaStream_t stream);

  struct Range {
    uint64_t offset = 0;  // in the engine's weight data
    uint64_t size = 0;
    std::shared_ptr<const CudaVmmBlock> block;
  };
  // In increasing offset order, covering the engine's weight data.
  const std::vector<Range>& ranges() const { return ranges_; }
  uint64_t weight_data_size() const { return weight_data_size_; }
  uint64_t private_bytes() const { return private_bytes_; }
  uint64_t segment_bytes() const { return segment_bytes_; }
  // The segment bytes this engine found on the device, held by another one.
  uint64_t shared_segment_bytes() const { return shared_segment_bytes_; }

 private:
  EngineWeightMemory() = default;

  uint64_t weight_data_size_ = 0;
  uint64_t private_bytes_ = 0;
  uint64_t segment_bytes_ = 0;
  uint64_t shared_segment_bytes_ = 0;
  std::vector<Range> ranges_;
};

// An EngineWeightMemory mapped into the weight memory of one engine
// (nvinfer1::IWeightsManager::restoreFromVmmAllocation, TensorRT-RTX 1.7).
// Destroy it before the engine.
class EngineWeightMapping {
 public:
  // `engine` was deserialized from the plan the memory's store describes and
  // has no weights loaded yet. kLiteRtStatusErrorUnsupported with an SDK
  // without a weights manager.
  static Expected<std::unique_ptr<EngineWeightMapping>> Create(
      nvinfer1::ICudaEngine& engine, const EngineWeightMemory& memory,
      cudaStream_t stream);

  ~EngineWeightMapping();
  EngineWeightMapping(const EngineWeightMapping&) = delete;
  EngineWeightMapping& operator=(const EngineWeightMapping&) = delete;

 private:
  struct Manager;
  EngineWeightMapping();

  std::unique_ptr<Manager> manager_;
};

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_DISPATCH_WEIGHT_STORE_H_
