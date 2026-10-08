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

#include "litert/vendors/nvidia/dispatch/weight_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cuda_runtime_api.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/cuda_vmm.h"
#include "litert/vendors/nvidia/tensorrt_features.h"
#include "NvInferRuntime.h"

namespace litert::nvidia {
namespace {

Unexpected CudaError(const char* call, cudaError_t status) {
  return Error(kLiteRtStatusErrorRuntimeFailure,
               std::string(call) + " failed: " + cudaGetErrorString(status));
}

bool ShareSegments() {
  const char* value =
      std::getenv("LITERT_NVIDIA_DISPATCH_SHARED_WEIGHT_SEGMENTS");
  return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

// The segments on the device, by device, key and size. An entry is of use as
// long as an engine holds its block.
using SegmentId = std::tuple<int, uint64_t, uint64_t, uint64_t>;
struct SegmentRegistry {
  std::mutex mutex;
  std::map<SegmentId, std::weak_ptr<const CudaVmmBlock>> blocks;
};
SegmentRegistry& Registry() {
  static auto* const registry = new SegmentRegistry();
  return *registry;
}

// Copies host data into device memory on a stream through two page-locked
// buffers that take turns: while the device copies out of one, the next chunk
// is read into the other. The copies are stream copies from page-locked
// memory, the kind that CUDA virtual memory mappings take reliably
// (cuda_vmm.h). The destructor waits for the copies.
class Uploader {
 public:
  explicit Uploader(cudaStream_t stream) : stream_(stream) {}

  ~Uploader() {
    for (auto& buffer : buffers_) {
      if (buffer.pending) {
        cudaEventSynchronize(buffer.copied);
      }
      if (buffer.copied != nullptr) {
        cudaEventDestroy(buffer.copied);
      }
      if (buffer.data != nullptr) {
        cudaFreeHost(buffer.data);
      }
    }
  }

  Expected<void> FromMemory(const uint8_t* source, uint64_t size,
                            uint8_t* target) {
    for (uint64_t done = 0; done < size;) {
      const size_t bytes =
          static_cast<size_t>(std::min<uint64_t>(size - done, kChunkBytes));
      LITERT_ASSIGN_OR_RETURN(uint8_t* buffer, Next());
      std::memcpy(buffer, source + done, bytes);
      LITERT_RETURN_IF_ERROR(Send(target + done, bytes));
      done += bytes;
    }
    return {};
  }

  Expected<void> FromFile(int fd, uint64_t offset, uint64_t size,
                          uint8_t* target) {
    for (uint64_t done = 0; done < size;) {
      const size_t bytes =
          static_cast<size_t>(std::min<uint64_t>(size - done, kChunkBytes));
      LITERT_ASSIGN_OR_RETURN(uint8_t* buffer, Next());
      for (size_t read = 0; read < bytes;) {
        const ssize_t result = pread(fd, buffer + read, bytes - read,
                                     static_cast<off_t>(offset + done + read));
        if (result < 0 && errno == EINTR) {
          continue;
        }
        if (result <= 0) {
          return Error(kLiteRtStatusErrorFileIO,
                       "Failed to read the weight source: " +
                           std::string(result == 0 ? "unexpected end of file"
                                                   : std::strerror(errno)));
        }
        read += static_cast<size_t>(result);
      }
      // The device gets the only copy that is needed from here on: let the
      // page cache drop its own.
      posix_fadvise(fd, static_cast<off_t>(offset + done),
                    static_cast<off_t>(bytes), POSIX_FADV_DONTNEED);
      LITERT_RETURN_IF_ERROR(Send(target + done, bytes));
      done += bytes;
    }
    return {};
  }

 private:
  static constexpr size_t kChunkBytes = 32 << 20;

  struct Buffer {
    void* data = nullptr;
    cudaEvent_t copied = nullptr;
    bool pending = false;
  };

  // The buffer for the next chunk, once the device has copied what it held.
  Expected<uint8_t*> Next() {
    next_ ^= 1;
    Buffer& buffer = buffers_[next_];
    if (buffer.data == nullptr) {
      cudaError_t status = cudaMallocHost(&buffer.data, kChunkBytes);
      if (status != cudaSuccess) {
        buffer.data = nullptr;
        return CudaError("cudaMallocHost", status);
      }
      status = cudaEventCreateWithFlags(&buffer.copied, cudaEventDisableTiming);
      if (status != cudaSuccess) {
        buffer.copied = nullptr;
        return CudaError("cudaEventCreateWithFlags", status);
      }
    }
    if (buffer.pending) {
      const cudaError_t status = cudaEventSynchronize(buffer.copied);
      if (status != cudaSuccess) {
        return CudaError("cudaEventSynchronize", status);
      }
      buffer.pending = false;
    }
    return static_cast<uint8_t*>(buffer.data);
  }

  // Copies `bytes` of the buffer Next() returned to `target`.
  Expected<void> Send(void* target, size_t bytes) {
    Buffer& buffer = buffers_[next_];
    cudaError_t status = cudaMemcpyAsync(target, buffer.data, bytes,
                                         cudaMemcpyHostToDevice, stream_);
    if (status != cudaSuccess) {
      return CudaError("cudaMemcpyAsync", status);
    }
    status = cudaEventRecord(buffer.copied, stream_);
    if (status != cudaSuccess) {
      return CudaError("cudaEventRecord", status);
    }
    buffer.pending = true;
    return {};
  }

  cudaStream_t stream_;
  std::array<Buffer, 2> buffers_;
  int next_ = 1;
};

// The model file the segments are read from, checked to be the one the plans
// were built from.
class SourceFile {
 public:
  ~SourceFile() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  Expected<void> Open(const TensorRtWeightStore& store) {
    fd_ = open(store.source_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
      return Error(kLiteRtStatusErrorFileIO,
                   "Failed to open the weight source " + store.source_path +
                       ": " + std::strerror(errno));
    }
    struct stat stat_buffer{};
    if (fstat(fd_, &stat_buffer) != 0 || !S_ISREG(stat_buffer.st_mode)) {
      return Error(kLiteRtStatusErrorFileIO,
                   "Failed to inspect the weight source " + store.source_path);
    }
    const TensorRtAotFileIdentity identity = {
        static_cast<uint64_t>(stat_buffer.st_dev),
        static_cast<uint64_t>(stat_buffer.st_ino),
        static_cast<int64_t>(stat_buffer.st_mtim.tv_sec),
        static_cast<int64_t>(stat_buffer.st_mtim.tv_nsec),
        static_cast<int64_t>(stat_buffer.st_ctim.tv_sec),
        static_cast<int64_t>(stat_buffer.st_ctim.tv_nsec)};
    if (static_cast<uint64_t>(stat_buffer.st_size) != store.source_size ||
        !(identity == store.source_identity)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "The model file changed since its TensorRT plans were "
                   "built: " +
                       store.source_path);
    }
    return {};
  }

  bool is_open() const { return fd_ >= 0; }
  int fd() const { return fd_; }

 private:
  int fd_ = -1;
};

struct WeightRanges {
  std::vector<EngineWeightMemory::Range> ranges;
  uint64_t private_bytes = 0;
  uint64_t segment_bytes = 0;
  uint64_t shared_segment_bytes = 0;
};

// Allocates the blocks of `store` into `filled` and starts the copies that
// fill them on `stream`; the caller waits for the stream, also when this
// fails, before it lets go of a block. The caller holds the registry's mutex.
// `uploaded` receives the segments this call adds to the registry.
Expected<void> FillWeightRanges(const TensorRtWeightStore& store,
                                cudaStream_t stream, int device,
                                SegmentRegistry& registry, WeightRanges& filled,
                                std::vector<SegmentId>& uploaded) {
  Uploader uploader(stream);

  // The stretches of the weight data between the segments: zeros but for the
  // store's runs.
  size_t next_run = 0;
  const auto add_private = [&](uint64_t begin, uint64_t end) -> Expected<void> {
    if (begin == end) {
      return {};
    }
    LITERT_ASSIGN_OR_RETURN(auto block, CudaVmmBlock::Create(end - begin));
    auto* const target = static_cast<uint8_t*>(block->address());
    filled.private_bytes += end - begin;
    filled.ranges.push_back({begin, end - begin, std::move(block)});
    const cudaError_t status = cudaMemsetAsync(target, 0, end - begin, stream);
    if (status != cudaSuccess) {
      return CudaError("cudaMemsetAsync", status);
    }
    for (; next_run < store.private_runs.size() &&
           store.private_runs[next_run].offset < end;
         ++next_run) {
      const auto& run = store.private_runs[next_run];
      if (run.offset < begin || run.offset + run.size > end) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "A private weight run crosses a weight segment");
      }
      LITERT_RETURN_IF_ERROR(uploader.FromMemory(
          run.data, run.size, target + (run.offset - begin)));
    }
    return {};
  };

  SourceFile source;
  const bool share = ShareSegments();
  uint64_t position = 0;
  for (const auto& segment : store.segments) {
    if (segment.payload_offset < position) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "The weight segments of the plan overlap");
    }
    LITERT_RETURN_IF_ERROR(add_private(position, segment.payload_offset));
    position = segment.payload_offset + segment.size;
    filled.segment_bytes += segment.size;
    const SegmentId id = {device, segment.key.low, segment.key.high,
                          segment.size};
    if (share) {
      if (const auto found = registry.blocks.find(id);
          found != registry.blocks.end()) {
        if (auto block = found->second.lock()) {
          filled.shared_segment_bytes += segment.size;
          filled.ranges.push_back(
              {segment.payload_offset, segment.size, std::move(block)});
          continue;
        }
      }
    }
    if (!source.is_open()) {
      LITERT_RETURN_IF_ERROR(source.Open(store));
    }
    LITERT_ASSIGN_OR_RETURN(auto created, CudaVmmBlock::Create(segment.size));
    auto* const target = static_cast<uint8_t*>(created->address());
    std::shared_ptr<const CudaVmmBlock> block = std::move(created);
    filled.ranges.push_back({segment.payload_offset, segment.size, block});
    if (share) {
      registry.blocks[id] = block;
      uploaded.push_back(id);
    }
    for (const auto& piece : segment.pieces) {
      if (piece.segment_offset > segment.size ||
          piece.size > segment.size - piece.segment_offset) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "A weight piece lies outside its segment");
      }
      LITERT_RETURN_IF_ERROR(uploader.FromFile(source.fd(), piece.source_offset,
                                               piece.size,
                                               target + piece.segment_offset));
    }
  }
  if (position > store.weight_data_size) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "A weight segment ends outside the weight data of the plan");
  }
  LITERT_RETURN_IF_ERROR(add_private(position, store.weight_data_size));
  if (next_run != store.private_runs.size()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "A private weight run lies inside a weight segment");
  }
  return {};
}

}  // namespace

Expected<std::unique_ptr<EngineWeightMemory>> EngineWeightMemory::Create(
    const TensorRtWeightStore& store, cudaStream_t stream) {
  LITERT_ASSIGN_OR_RETURN(const uint64_t granule, CudaVmmGranule());
  if (granule != store.granule) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "The plan was built for another CUDA virtual memory granule");
  }
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return CudaError("cudaGetDevice", status);
  }
  WeightRanges filled;
  std::vector<SegmentId> uploaded;
  SegmentRegistry& registry = Registry();
  // Held across the uploads: an engine that needs a segment another one is
  // uploading waits for it instead of uploading a second copy.
  const std::lock_guard<std::mutex> lock(registry.mutex);
  const auto started =
      FillWeightRanges(store, stream, device, registry, filled, uploaded);
  // The copies write the blocks from the upload buffers: wait for them, also
  // before a failure releases the blocks.
  status = cudaStreamSynchronize(stream);
  if (!started || status != cudaSuccess) {
    // No other engine has seen the segments of this call: the mutex is held.
    for (const SegmentId& id : uploaded) {
      registry.blocks.erase(id);
    }
    if (!started) {
      return started.Error();
    }
    return CudaError("cudaStreamSynchronize", status);
  }
  // Forget the segments that no engine holds any more.
  for (auto entry = registry.blocks.begin(); entry != registry.blocks.end();) {
    entry = entry->second.expired() ? registry.blocks.erase(entry)
                                    : std::next(entry);
  }
  std::unique_ptr<EngineWeightMemory> memory(new EngineWeightMemory());
  memory->weight_data_size_ = store.weight_data_size;
  memory->private_bytes_ = filled.private_bytes;
  memory->segment_bytes_ = filled.segment_bytes;
  memory->shared_segment_bytes_ = filled.shared_segment_bytes;
  memory->ranges_ = std::move(filled.ranges);
  return memory;
}

#if LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER

struct EngineWeightMapping::Manager {
  std::unique_ptr<nvinfer1::IWeightsManager> manager;
};

EngineWeightMapping::EngineWeightMapping() = default;
EngineWeightMapping::~EngineWeightMapping() = default;

Expected<std::unique_ptr<EngineWeightMapping>> EngineWeightMapping::Create(
    nvinfer1::ICudaEngine& engine, const EngineWeightMemory& memory,
    cudaStream_t stream) {
  std::unique_ptr<EngineWeightMapping> mapping(new EngineWeightMapping());
  mapping->manager_ = std::make_unique<Manager>();
  mapping->manager_->manager.reset(engine.createWeightsManager());
  nvinfer1::IWeightsManager* manager = mapping->manager_->manager.get();
  if (manager == nullptr) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "TensorRT has no weights manager for this engine");
  }
  if (manager->getSize() < 0 ||
      static_cast<uint64_t>(manager->getSize()) != memory.weight_data_size()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "The weight data of the engine does not have the size its "
                 "plan was calibrated with");
  }
  for (const auto& range : memory.ranges()) {
    if (!manager->restoreFromVmmAllocation(
            range.block->handle(), static_cast<int64_t>(range.offset),
            static_cast<int64_t>(range.size), stream)) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "TensorRT failed to map the weights of the engine");
    }
  }
  const cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    return CudaError("cudaStreamSynchronize", status);
  }
  return mapping;
}

#else  // !LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER

struct EngineWeightMapping::Manager {};

EngineWeightMapping::EngineWeightMapping() = default;
EngineWeightMapping::~EngineWeightMapping() = default;

Expected<std::unique_ptr<EngineWeightMapping>> EngineWeightMapping::Create(
    nvinfer1::ICudaEngine& engine, const EngineWeightMemory& memory,
    cudaStream_t stream) {
  static_cast<void>(engine);
  static_cast<void>(memory);
  static_cast<void>(stream);
  return Error(kLiteRtStatusErrorUnsupported,
               "A plan with a weight store needs the weights manager of "
               "TensorRT-RTX 1.7");
}

#endif  // LITERT_NVIDIA_TENSORRT_WEIGHTS_MANAGER

}  // namespace litert::nvidia
