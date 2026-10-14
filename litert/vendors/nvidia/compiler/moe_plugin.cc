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

#include "litert/vendors/nvidia/compiler/moe_plugin.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <vector>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"

namespace litert::nvidia {
namespace {

// Debug: LITERT_NVIDIA_PLUGIN_SKIP, a comma list of plugin families whose
// launches are skipped (the outputs are garbage): measures what each family
// costs a step. Read once per library.
bool SkipPluginFamily(const char* family) {
  static const char* const value = std::getenv("LITERT_NVIDIA_PLUGIN_SKIP");
  if (value == nullptr || value[0] == '\0') return false;
  const char* at = value;
  const size_t length = std::strlen(family);
  while (*at != '\0') {
    const char* end = std::strchr(at, ',');
    const size_t item = end == nullptr ? std::strlen(at) : end - at;
    if (item == length && std::strncmp(at, family, length) == 0) return true;
    if (end == nullptr) break;
    at = end + 1;
  }
  return false;
}

constexpr char kPluginName[] = "LiteRtNvidiaMoe";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kExpertsField[] = "experts";
constexpr char kTopKField[] = "top_k";
constexpr char kModelDimField[] = "model_dim";
constexpr char kHiddenDimField[] = "hidden_dim";
constexpr char kGateField[] = "gate";
constexpr char kScaleTypeField[] = "scale_type";
constexpr char kModelSizeField[] = "model_size";
constexpr char kOffsetsField[] = "offsets";
constexpr char kSizesField[] = "sizes";
constexpr char kModelPathField[] = "model_path";
constexpr char kRouterScaleTypeField[] = "router_scale_type";
constexpr char kRoutedField[] = "routed";
constexpr char kNormEpsField[] = "norm_eps";
constexpr char kRouterMultiplierField[] = "router_multiplier";
constexpr char kDenseScaleTypeField[] = "dense_scale_type";
constexpr char kDenseHiddenField[] = "dense_hidden";
constexpr char kBlockField[] = "block";
constexpr char kDenseGateField[] = "dense_gate";
constexpr char kResidualMultiplierField[] = "residual_multiplier";
constexpr int kFields = 19;
constexpr int kPieces = 14;

int CompactScalesMode();
constexpr size_t kArenaAlignment = 256;
constexpr size_t kStagingBytes = 32u << 20;

std::atomic<bool> g_loading_enabled{false};

// Round-to-nearest-even FP32 -> FP16.
float HalfBitsToFloat(uint16_t bits) {
  const uint32_t sign = (bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1f;
  const uint32_t mantissa = bits & 0x3ff;
  uint32_t word;
  if (exponent == 0) {
    if (mantissa == 0) {
      word = sign;
    } else {
      // Subnormal: normalize.
      int shift = 0;
      uint32_t m = mantissa;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++shift;
      }
      word = sign | ((127 - 15 - shift + 1) << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (exponent == 0x1f) {
    word = sign | 0x7f800000u | (mantissa << 13);
  } else {
    word = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 112;
  uint32_t mantissa = bits & 0x7fffffu;
  if (((bits >> 23) & 0xff) == 0xff) {  // inf or nan
    return static_cast<uint16_t>(sign | 0x7c00u | (mantissa != 0 ? 0x200u : 0));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    const int shift = 14 - exponent;
    uint32_t half = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half & 1u))) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

size_t RoundUp(size_t bytes, size_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

int64_t EnvInt64(const char* name, int64_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  return end != nullptr && *end == '\0' ? parsed : default_value;
}

// An expert arena: the six pieces of a layer, contiguous, in device or
// pinned host memory.
struct Arena {
  ~Arena() {
    if (memory == nullptr) return;
    if (on_device) {
      cudaFree(memory);
    } else {
      cudaFreeHost(memory);
    }
  }
  void* memory = nullptr;
  size_t bytes = 0;
  bool on_device = false;
  LiteRtNvidiaMoeArena pointers{};
};

struct Registry {
  std::mutex mutex;
  std::map<std::string, std::shared_ptr<Arena>> arenas;
  size_t device_bytes = 0;
  size_t host_bytes = 0;
  // The layers whose plugins exist (their first piece's offset), for the
  // compact-scale decision; -1 undecided, else 0 FP16 / 1 compact.
  std::set<uint64_t> layers;
  int compact_decision = -1;
};

Registry& GetRegistry() {
  static auto* const registry = new Registry();
  return *registry;
}

std::string ArenaKey(const MoeExpertSource& source) {
  std::string key = source.model_path;
  key += '\0';
  key += std::to_string(source.model_size);
  key += ':';
  key += std::to_string(source.router_scale_type);
  key += ':';
  key += std::to_string(source.dense_scale_type);
  key += ':';
  key += std::to_string(source.dense_hidden);
  key += ":m";
  key += std::to_string(CompactScalesMode());
  for (int i = 0; i < kPieces; ++i) {
    key += ':';
    key += std::to_string(source.offsets[i]);
    key += '+';
    key += std::to_string(source.sizes[i]);
  }
  return key;
}

// The arena layout of a layer: the three INT4 pieces then the three FP16
// scale pieces, each 256-byte aligned.
struct ArenaLayout {
  std::array<size_t, kPieces> offsets{};
  std::array<size_t, kPieces> bytes{};
  // Compact expert scales: pieces 3-5 hold a byte per block, then the rows'
  // FP16 maxima at max_offsets[piece - 3].
  bool compact = false;
  std::array<size_t, 3> max_offsets{};
  size_t total = 0;
};

// LITERT_NVIDIA_MOE_COMPACT_SCALES: 1 (the default) stores a byte per block
// as the fraction of the row's FP16 maximum, half the bytes of FP16 scales
// (the 26B: 1.43 GB to 0.72, reproduced to 0.3% at the 99th percentile) at
// the same decode speed once the format is a compile-time choice of the
// kernels; 0 keeps FP16; auto takes the compact form only when the FP16
// arenas of every layer would not all fit on the device beside the reserve,
// decided at the first arena (which cannot see caches allocated later).
int CompactScalesMode() {
  static const int mode = [] {
    const char* value = std::getenv("LITERT_NVIDIA_MOE_COMPACT_SCALES");
    if (value == nullptr || value[0] == '\0') return 1;
    if (std::strcmp(value, "0") == 0) return 0;
    if (std::strcmp(value, "auto") == 0) return 2;
    return 1;
  }();
  return mode;
}

// Whether piece i holds FP32 scales to convert; the three INT4 pieces and
// the router's rows are copied.
bool ConvertsPiece(const MoeExpertSource& source, int i) {
  if (i >= 3 && i < 6) return source.scale_type == 0;
  if (i == 7) return source.router_scale_type == 0;
  if (i == 10 || i == 11 || i == 13) return source.dense_scale_type == 0;
  return false;
}

bool LayoutArena(const MoeExpertSource& source, int32_t experts,
                 int32_t model_dim, int32_t hidden_dim, bool compact,
                 ArenaLayout* layout) {
  const size_t gate_rows = static_cast<size_t>(hidden_dim) * experts;
  const size_t down_rows = static_cast<size_t>(model_dim) * experts;
  const size_t scale_in = source.scale_type == 0 ? 4 : 2;
  const bool routed = source.router_scale_type >= 0;
  const size_t router_scale_in = source.router_scale_type == 0 ? 4 : 2;
  const bool dense = source.dense_scale_type >= 0;
  const size_t dense_rows =
      dense ? static_cast<size_t>(source.dense_hidden) : 0;
  const size_t dense_scale_in = source.dense_scale_type == 0 ? 4 : 2;
  const std::array<size_t, kPieces> expected = {
      gate_rows * model_dim / 2,
      gate_rows * model_dim / 2,
      down_rows * hidden_dim / 2,
      gate_rows * (model_dim / 32) * scale_in,
      gate_rows * (model_dim / 32) * scale_in,
      down_rows * (hidden_dim / 32) * scale_in,
      routed ? static_cast<size_t>(experts) * model_dim / 2 : 0,
      routed ? static_cast<size_t>(experts) * (model_dim / 32) * router_scale_in
             : 0,
      dense_rows * model_dim / 2,
      dense_rows * model_dim / 2,
      dense_rows * (model_dim / 32) * dense_scale_in,
      dense_rows * (model_dim / 32) * dense_scale_in,
      dense ? static_cast<size_t>(model_dim) * dense_rows / 2 : 0,
      dense
          ? static_cast<size_t>(model_dim) * (dense_rows / 32) * dense_scale_in
          : 0,
  };
  layout->compact = compact;
  size_t total = 0;
  for (int i = 0; i < kPieces; ++i) {
    if (source.sizes[i] != expected[i]) {
      return false;
    }
    // Scales land as FP16, or compact for the experts' (a byte per block
    // and the rows' maxima after).
    layout->bytes[i] = ConvertsPiece(source, i) ? expected[i] / 2 : expected[i];
    if (layout->compact && i >= 3 && i < 6) {
      const size_t rows = i < 5 ? gate_rows : down_rows;
      const size_t blocks = expected[i] / scale_in / rows;
      const size_t fractions = RoundUp(rows * blocks, kArenaAlignment);
      layout->bytes[i] = fractions + rows * sizeof(uint16_t);
    }
    if (i == 9 || i == 11) {
      // The dense up rows follow the gate rows without a gap, and their
      // scales the gate scales: the gated GEMV and GEMM read them stacked.
      layout->offsets[i] = layout->offsets[i - 1] + layout->bytes[i - 1];
      total = layout->offsets[i] + layout->bytes[i];
    } else {
      total = RoundUp(total, kArenaAlignment);
      layout->offsets[i] = total;
      total += layout->bytes[i];
    }
    if (layout->compact && i >= 3 && i < 6) {
      const size_t rows = i < 5 ? gate_rows : down_rows;
      layout->max_offsets[i - 3] =
          layout->offsets[i] + layout->bytes[i] - rows * sizeof(uint16_t);
    }
  }
  layout->total = RoundUp(total, kArenaAlignment);
  return true;
}

class FileReader {
 public:
  ~FileReader() {
    if (fd_ >= 0) close(fd_);
  }
  bool Open(const MoeExpertSource& source, std::string* error) {
    fd_ = open(source.model_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
      *error = "cannot open " + source.model_path + ": " + std::strerror(errno);
      return false;
    }
    struct stat info{};
    if (fstat(fd_, &info) != 0 || !S_ISREG(info.st_mode)) {
      *error = "cannot inspect " + source.model_path;
      return false;
    }
    if (static_cast<uint64_t>(info.st_size) != source.model_size) {
      *error = "the model file changed size since its plan was built: " +
               source.model_path;
      return false;
    }
    return true;
  }
  bool Read(uint64_t offset, void* buffer, size_t bytes, std::string* error) {
    auto* out = static_cast<uint8_t*>(buffer);
    for (size_t done = 0; done < bytes;) {
      const ssize_t result = pread(fd_, out + done, bytes - done,
                                   static_cast<off_t>(offset + done));
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) {
        *error =
            std::string("read failed: ") +
            (result == 0 ? "unexpected end of file" : std::strerror(errno));
        return false;
      }
      done += static_cast<size_t>(result);
    }
    return true;
  }

 private:
  int fd_ = -1;
};

// Loads the pieces of `source` into a new arena.
std::shared_ptr<Arena> LoadArena(const MoeExpertSource& source, int32_t experts,
                                 int32_t model_dim, int32_t hidden_dim,
                                 std::string* error) {
  const uint64_t reserve = static_cast<uint64_t>(EnvInt64(
                               "LITERT_NVIDIA_MOE_DEVICE_RESERVE_MB", 3072))
                           << 20;
  const char* placement = std::getenv("LITERT_NVIDIA_MOE_PLACEMENT");
  const bool by_memory =
      placement == nullptr || (std::strcmp(placement, "host") != 0 &&
                               std::strcmp(placement, "device") != 0);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    *error = "cudaMemGetInfo failed";
    return nullptr;
  }
  // The scale format, once for every layer (the caller holds the registry).
  Registry& registry = GetRegistry();
  if (registry.compact_decision < 0) {
    const int mode = CompactScalesMode();
    bool compact = mode == 1;
    if (mode == 2) {
      ArenaLayout half;
      if (!LayoutArena(source, experts, model_dim, hidden_dim, false, &half)) {
        *error = "the expert pieces do not match the layer's shape";
        return nullptr;
      }
      const size_t layers = std::max<size_t>(1, registry.layers.size());
      const uint64_t all_half = static_cast<uint64_t>(half.total) * layers;
      compact = by_memory && free_bytes < all_half + reserve;
      std::fprintf(stderr,
                   "[LiteRtNvidiaMoe] expert scales %s: %zu layers x %.1f MB "
                   "as FP16 = %.2f GB against %.2f GB free less a %.2f GB "
                   "reserve\n",
                   compact ? "compact (a byte per block)" : "FP16", layers,
                   half.total / 1048576.0, all_half / 1073741824.0,
                   free_bytes / 1073741824.0, reserve / 1073741824.0);
    }
    registry.compact_decision = compact ? 1 : 0;
  }
  ArenaLayout layout;
  if (!LayoutArena(source, experts, model_dim, hidden_dim,
                   registry.compact_decision == 1, &layout)) {
    *error = "the expert pieces do not match the layer's shape";
    return nullptr;
  }
  FileReader file;
  if (!file.Open(source, error)) return nullptr;

  auto arena = std::make_shared<Arena>();
  arena->bytes = layout.total;
  // Placement: device while a reserve stays free, pinned host otherwise.
  bool on_device = true;
  if (placement != nullptr && std::strcmp(placement, "host") == 0) {
    on_device = false;
  } else if (by_memory) {
    on_device = free_bytes >= layout.total + reserve;
  }
  cudaError_t status =
      on_device ? cudaMalloc(&arena->memory, layout.total)
                : cudaHostAlloc(&arena->memory, layout.total,
                                cudaHostAllocMapped | cudaHostAllocPortable);
  if (status != cudaSuccess) {
    arena->memory = nullptr;
    *error = std::string(on_device ? "cudaMalloc" : "cudaHostAlloc") +
             " failed: " + cudaGetErrorString(status);
    return nullptr;
  }
  arena->on_device = on_device;
  void* device_base = arena->memory;
  if (!on_device &&
      cudaHostGetDevicePointer(&device_base, arena->memory, 0) != cudaSuccess) {
    *error = "cudaHostGetDevicePointer failed";
    return nullptr;
  }

  // Staging for device arenas and for the scale conversion.
  std::vector<uint8_t> staging;
  std::vector<uint16_t> halves;
  auto* base = static_cast<uint8_t*>(arena->memory);
  for (int i = 0; i < kPieces; ++i) {
    if (source.sizes[i] == 0) continue;
    uint8_t* target = base + layout.offsets[i];
    const bool convert = ConvertsPiece(source, i);
    const size_t in_bytes = source.sizes[i];
    if (layout.compact && i >= 3 && i < 6) {
      // Row by row: the maximum as FP16, the blocks as fractions of it.
      const size_t rows = i < 5 ? static_cast<size_t>(hidden_dim) * experts
                                : static_cast<size_t>(model_dim) * experts;
      const size_t scale_in = source.scale_type == 0 ? 4 : 2;
      const size_t blocks = in_bytes / scale_in / rows;
      const size_t row_in = blocks * scale_in;
      const size_t rows_per_chunk = std::max<size_t>(1, kStagingBytes / row_in);
      std::vector<uint16_t> maxima(rows);
      std::vector<float> row(blocks);
      std::vector<uint8_t> fractions;
      for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
        const size_t n = std::min(rows_per_chunk, rows - r0);
        staging.resize(n * row_in);
        if (!file.Read(source.offsets[i] + r0 * row_in, staging.data(),
                       n * row_in, error)) {
          return nullptr;
        }
        fractions.resize(n * blocks);
        for (size_t r = 0; r < n; ++r) {
          for (size_t b = 0; b < blocks; ++b) {
            if (scale_in == 4) {
              std::memcpy(&row[b], staging.data() + (r * blocks + b) * 4, 4);
            } else {
              uint16_t bits = 0;
              std::memcpy(&bits, staging.data() + (r * blocks + b) * 2, 2);
              row[b] = HalfBitsToFloat(bits);
            }
          }
          LiteRtNvidiaCompactScaleRow(row.data(), static_cast<int32_t>(blocks),
                                      fractions.data() + r * blocks,
                                      maxima.data() + r0 + r);
        }
        if (on_device) {
          status = cudaMemcpy(target + r0 * blocks, fractions.data(),
                              n * blocks, cudaMemcpyHostToDevice);
          if (status != cudaSuccess) {
            *error =
                std::string("cudaMemcpy failed: ") + cudaGetErrorString(status);
            return nullptr;
          }
        } else {
          std::memcpy(target + r0 * blocks, fractions.data(), n * blocks);
        }
      }
      uint8_t* max_target = base + layout.max_offsets[i - 3];
      if (on_device) {
        status = cudaMemcpy(max_target, maxima.data(), rows * sizeof(uint16_t),
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
          *error =
              std::string("cudaMemcpy failed: ") + cudaGetErrorString(status);
          return nullptr;
        }
      } else {
        std::memcpy(max_target, maxima.data(), rows * sizeof(uint16_t));
      }
      continue;
    }
    for (size_t done = 0; done < in_bytes;) {
      const size_t chunk = std::min(in_bytes - done, kStagingBytes);
      const uint8_t* ready = nullptr;
      size_t ready_bytes = chunk;
      if (!convert && !on_device) {
        if (!file.Read(source.offsets[i] + done, target + done, chunk, error)) {
          return nullptr;
        }
        done += chunk;
        continue;
      }
      staging.resize(chunk);
      if (!file.Read(source.offsets[i] + done, staging.data(), chunk, error)) {
        return nullptr;
      }
      ready = staging.data();
      if (convert) {
        const size_t count = chunk / sizeof(float);
        halves.resize(count);
        for (size_t v = 0; v < count; ++v) {
          float value = 0.0f;
          std::memcpy(&value, staging.data() + v * sizeof(float),
                      sizeof(float));
          halves[v] = FloatToHalfBits(value);
        }
        ready = reinterpret_cast<const uint8_t*>(halves.data());
        ready_bytes = count * sizeof(uint16_t);
      }
      const size_t out_done = convert ? done / 2 : done;
      if (on_device) {
        status = cudaMemcpy(target + out_done, ready, ready_bytes,
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
          *error =
              std::string("cudaMemcpy failed: ") + cudaGetErrorString(status);
          return nullptr;
        }
      } else {
        std::memcpy(target + out_done, ready, ready_bytes);
      }
      done += chunk;
    }
  }
  auto* device_bytes = static_cast<uint8_t*>(device_base);
  const bool routed = source.router_scale_type >= 0;
  const bool dense = source.dense_scale_type >= 0;
  arena->pointers = {
      device_bytes + layout.offsets[0],
      device_bytes + layout.offsets[1],
      device_bytes + layout.offsets[2],
      device_bytes + layout.offsets[3],
      device_bytes + layout.offsets[4],
      device_bytes + layout.offsets[5],
      routed ? device_bytes + layout.offsets[6] : nullptr,
      routed ? device_bytes + layout.offsets[7] : nullptr,
      dense ? device_bytes + layout.offsets[8] : nullptr,
      dense ? device_bytes + layout.offsets[10] : nullptr,
      dense ? device_bytes + layout.offsets[12] : nullptr,
      dense ? device_bytes + layout.offsets[13] : nullptr,
      layout.compact ? device_bytes + layout.max_offsets[0] : nullptr,
      layout.compact ? device_bytes + layout.max_offsets[1] : nullptr,
      layout.compact ? device_bytes + layout.max_offsets[2] : nullptr,
      layout.compact ? 1 : 0};
  return arena;
}

std::shared_ptr<Arena> AcquireArena(const MoeExpertSource& source,
                                    int32_t experts, int32_t model_dim,
                                    int32_t hidden_dim) {
  Registry& registry = GetRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const std::string key = ArenaKey(source);
  if (const auto found = registry.arenas.find(key);
      found != registry.arenas.end()) {
    return found->second;
  }
  const auto start = std::chrono::steady_clock::now();
  std::string error;
  auto arena = LoadArena(source, experts, model_dim, hidden_dim, &error);
  if (arena == nullptr) {
    std::fprintf(stderr,
                 "[LiteRtNvidiaMoe] loading the experts at offset %llu of %s "
                 "failed: %s\n",
                 static_cast<unsigned long long>(source.offsets[0]),
                 source.model_path.c_str(), error.c_str());
    return nullptr;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  (arena->on_device ? registry.device_bytes : registry.host_bytes) +=
      arena->bytes;
  std::fprintf(stderr,
               "[LiteRtNvidiaMoe] experts at offset %llu: %.1f MB in %s "
               "memory, %.2f s (arenas now %.2f GB device, %.2f GB host)\n",
               static_cast<unsigned long long>(source.offsets[0]),
               arena->bytes / 1048576.0, arena->on_device ? "device" : "host",
               seconds, registry.device_bytes / 1073741824.0,
               registry.host_bytes / 1073741824.0);
  registry.arenas.emplace(key, arena);
  return arena;
}

int64_t ActivationRows(const nvinfer1::Dims& dims, int32_t model_dim) {
  if (dims.nbDims < 1 || dims.d[dims.nbDims - 1] != model_dim) {
    return 0;
  }
  int64_t rows = 1;
  for (int i = 0; i + 1 < dims.nbDims; ++i) {
    if (dims.d[i] <= 0 || dims.d[i] > std::numeric_limits<int32_t>::max() ||
        rows > std::numeric_limits<int32_t>::max() / dims.d[i]) {
      return 0;
    }
    rows *= dims.d[i];
  }
  return rows;
}

class MoePlugin final : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuild,
                        public nvinfer1::IPluginV3OneRuntime {
 public:
  MoePlugin(int32_t experts, int32_t top_k, int32_t model_dim,
            int32_t hidden_dim, int32_t gate, MoeExpertSource source,
            int32_t routed, float norm_eps, float router_multiplier,
            int32_t block, int32_t dense_gate,
            float residual_multiplier) noexcept
      : experts_(experts),
        top_k_(top_k),
        model_dim_(model_dim),
        hidden_dim_(hidden_dim),
        gate_(gate),
        source_(std::move(source)),
        routed_(routed),
        norm_eps_(norm_eps),
        router_multiplier_(router_multiplier),
        block_(block),
        dense_gate_(dense_gate),
        residual_multiplier_(residual_multiplier),
        model_size_(static_cast<int64_t>(source_.model_size)) {
    for (int i = 0; i < kPieces; ++i) {
      offsets_[i] = static_cast<int64_t>(source_.offsets[i]);
      sizes_[i] = static_cast<int64_t>(source_.sizes[i]);
    }
    {
      // One layer per distinct expert piece, for the scale-format decision.
      Registry& registry = GetRegistry();
      std::lock_guard<std::mutex> lock(registry.mutex);
      registry.layers.insert(source_.offsets[0]);
    }
    fields_ = {{
        {kExpertsField, &experts_, nvinfer1::PluginFieldType::kINT32, 1},
        {kTopKField, &top_k_, nvinfer1::PluginFieldType::kINT32, 1},
        {kModelDimField, &model_dim_, nvinfer1::PluginFieldType::kINT32, 1},
        {kHiddenDimField, &hidden_dim_, nvinfer1::PluginFieldType::kINT32, 1},
        {kGateField, &gate_, nvinfer1::PluginFieldType::kINT32, 1},
        {kScaleTypeField, &source_.scale_type,
         nvinfer1::PluginFieldType::kINT32, 1},
        {kModelSizeField, &model_size_, nvinfer1::PluginFieldType::kINT64, 1},
        {kOffsetsField, offsets_.data(), nvinfer1::PluginFieldType::kINT64,
         kPieces},
        {kSizesField, sizes_.data(), nvinfer1::PluginFieldType::kINT64,
         kPieces},
        {kModelPathField, source_.model_path.c_str(),
         nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(source_.model_path.size())},
        {kRouterScaleTypeField, &source_.router_scale_type,
         nvinfer1::PluginFieldType::kINT32, 1},
        {kRoutedField, &routed_, nvinfer1::PluginFieldType::kINT32, 1},
        {kNormEpsField, &norm_eps_, nvinfer1::PluginFieldType::kFLOAT32, 1},
        {kRouterMultiplierField, &router_multiplier_,
         nvinfer1::PluginFieldType::kFLOAT32, 1},
        {kDenseScaleTypeField, &source_.dense_scale_type,
         nvinfer1::PluginFieldType::kINT32, 1},
        {kDenseHiddenField, &source_.dense_hidden,
         nvinfer1::PluginFieldType::kINT32, 1},
        {kBlockField, &block_, nvinfer1::PluginFieldType::kINT32, 1},
        {kDenseGateField, &dense_gate_, nvinfer1::PluginFieldType::kINT32, 1},
        {kResidualMultiplierField, &residual_multiplier_,
         nvinfer1::PluginFieldType::kFLOAT32, 1},
    }};
    field_collection_ = {static_cast<int32_t>(fields_.size()), fields_.data()};
  }

  nvinfer1::IPluginCapability* getCapabilityInterface(
      nvinfer1::PluginCapabilityType type) noexcept override {
    switch (type) {
      case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
      case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
      case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    auto* copy = new (std::nothrow)
        MoePlugin(experts_, top_k_, model_dim_, hidden_dim_, gate_, source_,
                  routed_, norm_eps_, router_multiplier_, block_, dense_gate_,
                  residual_multiplier_);
    if (copy != nullptr) {
      copy->arena_ = arena_;
    }
    return copy;
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != InputCount() || num_outputs != 1) {
      return 1;
    }
    for (int i = 0; i < InputCount(); ++i) {
      if (input_types[i] != InputType(i)) return 1;
    }
    output_types[0] = nvinfer1::DataType::kBF16;
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != InputCount() || num_outputs != 1 ||
        position < 0 || position > InputCount()) {
      return false;
    }
    const nvinfer1::DataType type = position < InputCount()
                                        ? InputType(position)
                                        : nvinfer1::DataType::kBF16;
    return in_out[position].desc.type == type &&
           in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expr_builder);
    if (inputs == nullptr || outputs == nullptr || num_inputs != InputCount() ||
        num_shape_inputs != 0 || num_outputs != 1 || inputs[0].nbDims < 1) {
      return 1;
    }
    outputs[0] = inputs[0];
    return 0;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == InputCount() && num_outputs == 1 &&
                   (block_ == 0 || EnsureSideResources())
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == InputCount() && num_outputs == 1
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(outputs);
    if (inputs == nullptr || num_inputs != InputCount() || num_outputs != 1) {
      return 0;
    }
    const LiteRtNvidiaMoeShape shape =
        Shape(ActivationRows(inputs[0].max, model_dim_));
    return block_ != 0 ? LiteRtNvidiaMoeBlockWorkspaceBytes(
                             &shape, source_.dense_hidden, dense_gate_)
                       : LiteRtNvidiaMoeWorkspaceBytes(&shape);
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    if (SkipPluginFamily("moe")) return 0;
    if (arena_ == nullptr) {
      if (!MoeArenaLoadingEnabled()) {
        // An engine build times the plugin without experts: nothing to do.
        return 0;
      }
      arena_ = AcquireArena(source_, experts_, model_dim_, hidden_dim_);
      if (arena_ == nullptr) {
        return 1;
      }
    }
    const int64_t rows = ActivationRows(input_desc[0].dims, model_dim_);
    const LiteRtNvidiaMoeShape shape = Shape(rows);
    if (rows < 1 || !LiteRtNvidiaMoeSupports(&shape) || workspace == nullptr) {
      return 1;
    }
    cudaError_t status;
    if (block_ != 0) {
      if (!EnsureSideResources()) return 1;
      const LiteRtNvidiaMoeBlockParams block = {
          source_.dense_hidden,
          dense_gate_,
          residual_multiplier_,
          static_cast<const float*>(inputs[4]),
          static_cast<const float*>(inputs[5]),
          static_cast<const float*>(inputs[6]),
          static_cast<const float*>(inputs[7]),
          side_stream_,
          fork_,
          join_};
      status = LiteRtNvidiaLaunchMoeLayerBlock(
          &shape, &arena_->pointers, inputs[0],
          static_cast<const float*>(inputs[1]),
          static_cast<const float*>(inputs[2]), router_multiplier_,
          static_cast<const float*>(inputs[3]), norm_eps_, /*renormalize=*/1,
          &block, workspace, outputs[0], stream);
    } else if (routed_ != 0) {
      status = LiteRtNvidiaLaunchMoeInt4BlockFused(
          &shape, &arena_->pointers, inputs[0],
          static_cast<const float*>(inputs[1]),
          static_cast<const float*>(inputs[2]), router_multiplier_,
          static_cast<const float*>(inputs[3]), norm_eps_, /*renormalize=*/1,
          workspace, outputs[0], stream);
    } else {
      status = LiteRtNvidiaLaunchMoeInt4Block(
          &shape, &arena_->pointers, inputs[0],
          static_cast<const int32_t*>(inputs[2]),
          static_cast<const float*>(inputs[1]),
          static_cast<const float*>(inputs[3]), workspace, outputs[0], stream);
    }
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaMoe] CUDA launch failed: %s (%d): %s\n",
                   cudaGetErrorName(status), static_cast<int>(status),
                   cudaGetErrorString(status));
      return 1;
    }
    return 0;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    auto* copy = static_cast<MoePlugin*>(clone());
    if (copy != nullptr && copy->arena_ == nullptr &&
        MoeArenaLoadingEnabled()) {
      copy->arena_ = AcquireArena(source_, experts_, model_dim_, hidden_dim_);
      arena_ = copy->arena_;
    }
    if (copy != nullptr && block_ != 0) {
      // Made here, outside any graph capture of the context's stream.
      copy->EnsureSideResources();
    }
    return copy;
  }

  ~MoePlugin() override { ReleaseSideResources(); }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return &field_collection_;
  }

 private:
  // Plain: x, top_weights, top_indices, per_expert_scale. Routed: x,
  // per_expert_scale, router norm scale, expert norm scale; a layer block
  // adds the dense input norm's, dense output norm's, experts' output
  // norm's and the sum's norm's scales.
  int32_t InputCount() const noexcept { return block_ != 0 ? 8 : 4; }

  nvinfer1::DataType InputType(int position) const noexcept {
    if (position == 0) return nvinfer1::DataType::kBF16;
    if (position == 2 && routed_ == 0) return nvinfer1::DataType::kINT32;
    return nvinfer1::DataType::kFLOAT;
  }

  // The side stream and events of a layer block's dense feed-forward.
  bool EnsureSideResources() noexcept {
    if (side_stream_ != nullptr) return true;
    if (cudaStreamCreateWithFlags(&side_stream_, cudaStreamNonBlocking) !=
            cudaSuccess ||
        cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming) !=
            cudaSuccess ||
        cudaEventCreateWithFlags(&join_, cudaEventDisableTiming) !=
            cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      ReleaseSideResources();
      return false;
    }
    return true;
  }

  void ReleaseSideResources() noexcept {
    if (join_ != nullptr) cudaEventDestroy(join_);
    if (fork_ != nullptr) cudaEventDestroy(fork_);
    if (side_stream_ != nullptr) cudaStreamDestroy(side_stream_);
    join_ = nullptr;
    fork_ = nullptr;
    side_stream_ = nullptr;
  }

  LiteRtNvidiaMoeShape Shape(int64_t rows) const noexcept {
    return {static_cast<int32_t>(
                std::min<int64_t>(rows, std::numeric_limits<int32_t>::max())),
            top_k_,
            experts_,
            model_dim_,
            hidden_dim_,
            gate_};
  }

  int32_t experts_;
  int32_t top_k_;
  int32_t model_dim_;
  int32_t hidden_dim_;
  int32_t gate_;
  MoeExpertSource source_;
  int32_t routed_;
  float norm_eps_;
  float router_multiplier_;
  int32_t block_;
  int32_t dense_gate_;
  float residual_multiplier_;
  cudaStream_t side_stream_ = nullptr;
  cudaEvent_t fork_ = nullptr;
  cudaEvent_t join_ = nullptr;
  int64_t model_size_;
  std::array<int64_t, kPieces> offsets_{};
  std::array<int64_t, kPieces> sizes_{};
  std::shared_ptr<Arena> arena_;
  std::array<nvinfer1::PluginField, kFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class MoePluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  MoePluginCreator() noexcept
      : fields_{{
            {kExpertsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kTopKField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kModelDimField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kHiddenDimField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kGateField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kScaleTypeField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kModelSizeField, nullptr, nvinfer1::PluginFieldType::kINT64, 1},
            {kOffsetsField, nullptr, nvinfer1::PluginFieldType::kINT64,
             kPieces},
            {kSizesField, nullptr, nvinfer1::PluginFieldType::kINT64, kPieces},
            {kModelPathField, nullptr, nvinfer1::PluginFieldType::kCHAR, 1},
            {kRouterScaleTypeField, nullptr, nvinfer1::PluginFieldType::kINT32,
             1},
            {kRoutedField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kNormEpsField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
            {kRouterMultiplierField, nullptr,
             nvinfer1::PluginFieldType::kFLOAT32, 1},
            {kDenseScaleTypeField, nullptr, nvinfer1::PluginFieldType::kINT32,
             1},
            {kDenseHiddenField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kBlockField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kDenseGateField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {kResidualMultiplierField, nullptr,
             nvinfer1::PluginFieldType::kFLOAT32, 1},
        }},
        field_collection_{static_cast<int32_t>(fields_.size()),
                          fields_.data()} {}

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    if (fields == nullptr || fields->fields == nullptr) {
      return nullptr;
    }
    int32_t experts = 0;
    int32_t top_k = 0;
    int32_t model_dim = 0;
    int32_t hidden_dim = 0;
    int32_t gate = 0;
    int32_t routed = 0;
    float norm_eps = -1.0f;
    float router_multiplier = 1.0f;
    int32_t block = 0;
    int32_t dense_gate = 0;
    float residual_multiplier = 1.0f;
    MoeExpertSource source;
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr) {
        continue;
      }
      if (field.type == nvinfer1::PluginFieldType::kFLOAT32 &&
          field.length == 1) {
        const float value = *static_cast<const float*>(field.data);
        if (std::strcmp(field.name, kNormEpsField) == 0) {
          norm_eps = value;
        } else if (std::strcmp(field.name, kRouterMultiplierField) == 0) {
          router_multiplier = value;
        } else if (std::strcmp(field.name, kResidualMultiplierField) == 0) {
          residual_multiplier = value;
        }
        continue;
      }
      if (field.type == nvinfer1::PluginFieldType::kCHAR) {
        if (std::strcmp(field.name, kModelPathField) == 0 && field.length > 0) {
          source.model_path.assign(static_cast<const char*>(field.data),
                                   static_cast<size_t>(field.length));
        }
        continue;
      }
      if (field.type == nvinfer1::PluginFieldType::kINT64) {
        const auto* wide = static_cast<const int64_t*>(field.data);
        if (std::strcmp(field.name, kModelSizeField) == 0 &&
            field.length == 1) {
          source.model_size = static_cast<uint64_t>(wide[0]);
        } else if (std::strcmp(field.name, kOffsetsField) == 0 &&
                   field.length == kPieces) {
          for (int p = 0; p < kPieces; ++p) {
            source.offsets[p] = static_cast<uint64_t>(wide[p]);
          }
        } else if (std::strcmp(field.name, kSizesField) == 0 &&
                   field.length == kPieces) {
          for (int p = 0; p < kPieces; ++p) {
            source.sizes[p] = static_cast<uint64_t>(wide[p]);
          }
        }
        continue;
      }
      if (field.type != nvinfer1::PluginFieldType::kINT32 ||
          field.length != 1) {
        continue;
      }
      const int32_t value = *static_cast<const int32_t*>(field.data);
      if (std::strcmp(field.name, kExpertsField) == 0) {
        experts = value;
      } else if (std::strcmp(field.name, kTopKField) == 0) {
        top_k = value;
      } else if (std::strcmp(field.name, kModelDimField) == 0) {
        model_dim = value;
      } else if (std::strcmp(field.name, kHiddenDimField) == 0) {
        hidden_dim = value;
      } else if (std::strcmp(field.name, kGateField) == 0) {
        gate = value;
      } else if (std::strcmp(field.name, kScaleTypeField) == 0) {
        source.scale_type = value;
      } else if (std::strcmp(field.name, kRouterScaleTypeField) == 0) {
        source.router_scale_type = value;
      } else if (std::strcmp(field.name, kRoutedField) == 0) {
        routed = value;
      } else if (std::strcmp(field.name, kDenseScaleTypeField) == 0) {
        source.dense_scale_type = value;
      } else if (std::strcmp(field.name, kDenseHiddenField) == 0) {
        source.dense_hidden = value;
      } else if (std::strcmp(field.name, kBlockField) == 0) {
        block = value;
      } else if (std::strcmp(field.name, kDenseGateField) == 0) {
        dense_gate = value;
      }
    }
    return CreateMoePlugin(experts, top_k, model_dim, hidden_dim, gate, source,
                           routed != 0, norm_eps, router_multiplier, block != 0,
                           dense_gate, residual_multiplier);
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &field_collection_;
  }
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  std::array<nvinfer1::PluginField, kFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

}  // namespace

nvinfer1::IPluginV3* CreateMoePlugin(int32_t experts, int32_t top_k,
                                     int32_t model_dim, int32_t hidden_dim,
                                     int32_t gate,
                                     const MoeExpertSource& source, bool routed,
                                     float norm_eps, float router_multiplier,
                                     bool block, int32_t dense_gate,
                                     float residual_multiplier) noexcept {
  const LiteRtNvidiaMoeShape shape = {1,         top_k,      experts,
                                      model_dim, hidden_dim, gate};
  if (!LiteRtNvidiaMoeSupports(&shape) || source.model_path.empty() ||
      source.model_size == 0 ||
      (source.scale_type != 0 && source.scale_type != 1) ||
      source.router_scale_type < -1 || source.router_scale_type > 1 ||
      routed != (source.router_scale_type >= 0) ||
      (routed && (norm_eps < 0.0f || model_dim > 4096 ||
                  !(router_multiplier == router_multiplier))) ||
      source.dense_scale_type < -1 || source.dense_scale_type > 1 ||
      block != (source.dense_scale_type >= 0) ||
      (block &&
       (!routed || source.dense_hidden < 32 || source.dense_hidden % 32 != 0 ||
        (dense_gate != 1 && dense_gate != 2) ||
        !(residual_multiplier == residual_multiplier) ||
        LiteRtNvidiaMoeBlockWorkspaceBytes(&shape, source.dense_hidden,
                                           dense_gate) == 0))) {
    return nullptr;
  }
  ArenaLayout layout;
  if (!LayoutArena(source, experts, model_dim, hidden_dim, false, &layout)) {
    return nullptr;
  }
  for (int i = 0; i < kPieces; ++i) {
    if (source.offsets[i] > source.model_size ||
        source.sizes[i] > source.model_size - source.offsets[i]) {
      return nullptr;
    }
  }
  return new (std::nothrow)
      MoePlugin(experts, top_k, model_dim, hidden_dim, gate, source,
                routed ? 1 : 0, norm_eps, router_multiplier, block ? 1 : 0,
                dense_gate, residual_multiplier);
}

void EnsureMoePluginRegistered() noexcept {}

void SetMoeArenaLoadingEnabled(bool enabled) noexcept {
  g_loading_enabled.store(enabled);
  setenv("LITERT_NVIDIA_MOE_LOAD_EXPERTS", enabled ? "1" : "0", 1);
}

// The plugin library is linked into both the compiler and the dispatch
// library, and TensorRT's registry keeps whichever creator registered first,
// so the flag is also carried in the environment, which every copy sees.
bool MoeArenaLoadingEnabled() noexcept {
  if (g_loading_enabled.load()) return true;
  const char* value = std::getenv("LITERT_NVIDIA_MOE_LOAD_EXPERTS");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void MoeArenaBytes(size_t* device_bytes, size_t* host_bytes) noexcept {
  Registry& registry = GetRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  if (device_bytes != nullptr) *device_bytes = registry.device_bytes;
  if (host_bytes != nullptr) *host_bytes = registry.host_bytes;
}

REGISTER_TENSORRT_PLUGIN(MoePluginCreator);

}  // namespace litert::nvidia
