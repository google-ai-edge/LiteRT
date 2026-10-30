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

#include "litert/vendors/nvidia/compiler/weight_holder_calibration.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl  // from @com_google_absl
#include "cuda_runtime_api.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"
#include "litert/vendors/nvidia/tensorrt_features.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

namespace litert::nvidia {

bool TensorRtWeightHoldersSupported() {
  return LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER != 0;
}

#if LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER

namespace {

// TensorRT aligns the weights of an engine to 128 bytes in its weight data.
constexpr uint64_t kWeightAlignment = 128;
constexpr uint64_t kPage = 4096;
// "LRTWHOLD": the first bytes of a holder during calibration, followed by the
// index of the holder.
constexpr uint64_t kMarker = 0x444c4f485754524cULL;

uint64_t RoundUp(uint64_t value, uint64_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// Receives the weight data in order: finds the markers and keeps the pages
// outside the segments that are not all zeros.
class WeightDataScanner final : public nvinfer1::IStreamWriter {
 public:
  WeightDataScanner(absl::Span<const TensorRtWeightHolderBuildData> holders,
                    uint64_t granule,
                    TensorRtWeightHolderCalibration* calibration)
      : holders_(holders),
        granule_(granule),
        calibration_(calibration),
        found_(holders.size(), false) {
    calibration_->holder_offsets.assign(holders.size(), 0);
  }

  int64_t write(void const* data, int64_t bytes) final {
    const auto* cur = static_cast<const uint8_t*>(data);
    int64_t remaining = bytes;
    while (remaining > 0) {
      if (pending_.empty() && remaining >= static_cast<int64_t>(kPage)) {
        Page(cur);
        cur += kPage;
        remaining -= kPage;
        continue;
      }
      const size_t take = static_cast<size_t>(std::min<int64_t>(
          remaining, static_cast<int64_t>(kPage - pending_.size())));
      pending_.insert(pending_.end(), cur, cur + take);
      cur += take;
      remaining -= take;
      if (pending_.size() == kPage) {
        Page(pending_.data());
        pending_.clear();
      }
    }
    return failed_ ? -1 : bytes;
  }

  uint64_t position() const { return position_ + pending_.size(); }
  bool complete() const {
    return !failed_ && pending_.empty() &&
           std::all_of(found_.begin(), found_.end(), [](bool f) { return f; });
  }

 private:
  // One page of the weight data, at position_.
  void Page(const uint8_t* page) {
    for (uint64_t at = 0; at < kPage; at += kWeightAlignment) {
      uint64_t marker = 0;
      std::memcpy(&marker, page + at, sizeof(marker));
      if (marker != kMarker) {
        continue;
      }
      uint64_t index = 0;
      std::memcpy(&index, page + at + sizeof(marker), sizeof(index));
      if (index >= holders_.size() || found_[index]) {
        failed_ = true;
        return;
      }
      found_[index] = true;
      const uint64_t offset = position_ + at;
      calibration_->holder_offsets[index] = offset;
      // Holders follow each other in the weight data, so their segments are
      // found in increasing order and do not overlap.
      segment_begin_ = RoundUp(offset, granule_);
      segment_end_ = segment_begin_ + holders_[index].bytes - granule_;
    }
    const bool in_segment =
        position_ >= segment_begin_ && position_ < segment_end_;
    if (!in_segment && std::any_of(page, page + kPage,
                                   [](uint8_t byte) { return byte != 0; })) {
      auto& runs = calibration_->private_runs;
      if (runs.empty() ||
          runs.back().offset + runs.back().data.size() != position_) {
        runs.push_back({position_, {}});
      }
      runs.back().data.insert(runs.back().data.end(), page, page + kPage);
    }
    position_ += kPage;
  }

  absl::Span<const TensorRtWeightHolderBuildData> holders_;
  const uint64_t granule_;
  TensorRtWeightHolderCalibration* calibration_;
  std::vector<bool> found_;
  std::vector<uint8_t> pending_;
  uint64_t position_ = 0;
  uint64_t segment_begin_ = 0;
  uint64_t segment_end_ = 0;
  bool failed_ = false;
};

struct Unmap {
  size_t size;
  void operator()(void* pointer) const { munmap(pointer, size); }
};

struct StreamDeleter {
  void operator()(cudaStream_t stream) const { cudaStreamDestroy(stream); }
};

}  // namespace

Expected<TensorRtWeightHolderCalibration> CalibrateTensorRtWeightHolders(
    const void* plan, size_t plan_size,
    absl::Span<const TensorRtWeightHolderBuildData> holders, uint64_t granule) {
  if (plan == nullptr || plan_size == 0 || holders.empty() || granule == 0 ||
      granule % kPage != 0) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid weight holder calibration request");
  }
  uint64_t source_bytes = 0;
  for (const auto& holder : holders) {
    if (holder.bytes <= granule || holder.bytes % granule != 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "A weight holder is a segment of granules plus one");
    }
    source_bytes += holder.bytes;
  }

  TensorRtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(
      nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create a TensorRT runtime for calibration");
  }
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(plan, plan_size));
  if (!engine) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to deserialize the plan for calibration");
  }
  std::unique_ptr<nvinfer1::IWeightsManager> manager(
      engine->createWeightsManager());
  if (!manager) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "TensorRT has no weights manager for this plan");
  }
  TensorRtWeightHolderCalibration calibration;
  calibration.weight_data_size = static_cast<uint64_t>(manager->getSize());
  if (calibration.weight_data_size == 0 ||
      calibration.weight_data_size % granule != 0) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Unexpected TensorRT weight data size");
  }

  // The refit source: zero pages that cost no memory until they are written,
  // which only the page with a holder's marker is.
  void* const mapped = mmap(nullptr, source_bytes, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (mapped == MAP_FAILED) {
    return Error(kLiteRtStatusErrorMemoryAllocationFailure,
                 "Failed to reserve the calibration refit source");
  }
  const std::unique_ptr<void, Unmap> zeros(mapped, Unmap{source_bytes});
  cudaStream_t raw_stream = nullptr;
  if (cudaStreamCreate(&raw_stream) != cudaSuccess) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to create a CUDA stream for calibration");
  }
  const std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter>
      stream(raw_stream);
  {
    std::unique_ptr<nvinfer1::IRefitter> refitter(
        nvinfer1::createInferRefitter(*engine, logger));
    if (!refitter) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create a TensorRT refitter for calibration");
    }
    uint64_t source_offset = 0;
    for (size_t i = 0; i < holders.size(); ++i) {
      auto* source = static_cast<uint8_t*>(mapped) + source_offset;
      const uint64_t index = i;
      std::memcpy(source, &kMarker, sizeof(kMarker));
      std::memcpy(source + sizeof(kMarker), &index, sizeof(index));
      const int64_t count =
          static_cast<int64_t>(holders[i].bytes / sizeof(int64_t));
      const nvinfer1::Weights prototype =
          refitter->getWeightsPrototype(holders[i].name.c_str());
      if (prototype.type != nvinfer1::DataType::kINT64 ||
          prototype.count != count ||
          !refitter->setNamedWeights(
              holders[i].name.c_str(),
              nvinfer1::Weights{nvinfer1::DataType::kINT64, source, count},
              nvinfer1::TensorLocation::kHOST)) {
        return Error(
            kLiteRtStatusErrorRuntimeFailure,
            "The plan does not hold the weight holder " + holders[i].name);
      }
      source_offset += holders[i].bytes;
    }
    if (refitter->getMissingWeights(0, nullptr) != 0) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "The plan has stripped weights besides its weight holders");
    }
    if (!refitter->refitCudaEngineAsync(stream.get()) ||
        cudaStreamSynchronize(stream.get()) != cudaSuccess) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to refit the weight holders for calibration");
    }
    refitter->releaseRefitResources();
  }

  WeightDataScanner scanner(holders, granule, &calibration);
  const bool saved = manager->saveToStream(scanner, stream.get());
  // Leave the device as it was found, whatever happened.
  const bool unloaded = manager->unload(stream.get());
  cudaStreamSynchronize(stream.get());
  if (!saved || !scanner.complete() ||
      scanner.position() != calibration.weight_data_size) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to locate the weight holders in the weight data");
  }
  if (!unloaded) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to unload the calibration weights");
  }
  for (size_t i = 0; i < holders.size(); ++i) {
    const uint64_t segment = RoundUp(calibration.holder_offsets[i], granule);
    if (segment + holders[i].bytes - granule > calibration.weight_data_size) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "A weight holder does not fit the weight data");
    }
  }
  return calibration;
}

#else  // !LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER

Expected<TensorRtWeightHolderCalibration> CalibrateTensorRtWeightHolders(
    const void* plan, size_t plan_size,
    absl::Span<const TensorRtWeightHolderBuildData> holders, uint64_t granule) {
  static_cast<void>(plan);
  static_cast<void>(plan_size);
  static_cast<void>(holders);
  static_cast<void>(granule);
  return Error(kLiteRtStatusErrorUnsupported,
               "This TensorRT SDK has no weights manager");
}

#endif  // LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER

}  // namespace litert::nvidia
