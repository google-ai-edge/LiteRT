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

#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper_with_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_c_types_printing.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/compiler/plugin/algo.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"
#include "litert/vendors/nvidia/memory_profile.h"
#include "litert/vendors/nvidia/tensorrt_rtx/include/NvInferRuntime.h"
#include "NvInferVersion.h"

namespace {

using litert::Error;
using litert::Expected;

constexpr char kPluginManufacturer[] = "NVIDIA";
constexpr char kPluginSocModel[] = "tensorrt-rtx";
constexpr uint32_t kNvidiaCompilerCacheSchemaVersion = 3;

enum class PartitionPolicy {
  kSafe,
  kGemma4,
  kCompute,
  kAll,
};

std::string PartitionName(int index) {
  return "tensorrt_partition_" + std::to_string(index);
}

uint64_t HashSharedWeight(
    const litert::nvidia::TensorRtRefitWeightBuildData& weight) {
  constexpr uint64_t kOffset = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t hash = kOffset;
  auto add_bytes = [&](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= kPrime;
    }
  };
  add_bytes(&weight.data_type, sizeof(weight.data_type));
  add_bytes(&weight.count, sizeof(weight.count));
  add_bytes(weight.data.data(), weight.data.size());
  return hash;
}

class SharedWeightDeduper {
 public:
  uint32_t Add(litert::nvidia::TensorRtRefitWeightBuildData weight) {
    logical_bytes_ += weight.data.size();
    const uint64_t hash = HashSharedWeight(weight);
    auto& candidates = indexes_by_hash_[hash];
    for (uint32_t index : candidates) {
      const auto& existing = weights_[index];
      if (existing.data_type == weight.data_type &&
          existing.count == weight.count && existing.data == weight.data) {
        return index;
      }
    }
    const uint32_t index = static_cast<uint32_t>(weights_.size());
    unique_bytes_ += weight.data.size();
    weights_.push_back(
        {weight.data_type, weight.count, std::move(weight.data)});
    candidates.push_back(index);
    return index;
  }

  const std::vector<litert::nvidia::TensorRtSharedWeight>& weights() const {
    return weights_;
  }
  size_t logical_bytes() const { return logical_bytes_; }
  size_t unique_bytes() const { return unique_bytes_; }

 private:
  std::vector<litert::nvidia::TensorRtSharedWeight> weights_;
  std::unordered_map<uint64_t, std::vector<uint32_t>> indexes_by_hash_;
  size_t logical_bytes_ = 0;
  size_t unique_bytes_ = 0;
};

struct PendingBundleEntry {
  std::string function_name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<uint8_t> engine;
  std::optional<litert::nvidia::TensorRtLlmHeadBuildData> trtllm_head;
  std::vector<litert::nvidia::TensorRtSharedWeightRef> refit_weights;
};

std::string TensorSummary(const litert::compiler::Tensor& tensor) {
  std::string summary =
      "tensor=" + std::to_string(tensor.TensorIndex()) +
      " type=" + std::to_string(static_cast<int>(tensor.ElementType()));
  auto ranked_type = tensor.RankedTensorType();
  if (ranked_type) {
    summary += " shape=[";
    const auto dims = ranked_type->Layout().Dimensions();
    for (int i = 0; i < dims.size(); ++i) {
      if (i > 0) {
        summary += "x";
      }
      summary += std::to_string(dims[i]);
    }
    summary += "]";
  } else {
    summary += " shape=<unranked>";
  }
  summary += tensor.HasWeights() ? " const" : " runtime";
  summary += " qtype=" + std::to_string(tensor.QTypeId());
  return summary;
}

std::string OpSummary(const litert::compiler::Op& op) {
  std::string summary =
      std::string(litert::GetOpCodeStringView(op.Code())) + " inputs={";
  for (int i = 0; i < op.Inputs().size(); ++i) {
    if (i > 0) {
      summary += "; ";
    }
    summary += TensorSummary(op.Inputs()[i]);
  }
  summary += "} outputs={";
  for (int i = 0; i < op.Outputs().size(); ++i) {
    if (i > 0) {
      summary += "; ";
    }
    summary += TensorSummary(op.Outputs()[i]);
  }
  summary += "}";
  return summary;
}

bool LogPartitionOps() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_LOG_PARTITION_OPS");
  return value != nullptr && value[0] != '\0';
}

bool LogUnsupportedOps() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_LOG_UNSUPPORTED_OPS");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

PartitionPolicy GetPartitionPolicy() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_PARTITION_POLICY");
  if (value == nullptr || value[0] == '\0') {
    return PartitionPolicy::kSafe;
  }
  const std::string policy(value);
  if (policy == "all") {
    return PartitionPolicy::kAll;
  }
  if (policy == "gemma4") {
    return PartitionPolicy::kGemma4;
  }
  if (policy == "compute") {
    return PartitionPolicy::kCompute;
  }
  return PartitionPolicy::kSafe;
}

const char* PartitionPolicyName(PartitionPolicy policy) {
  switch (policy) {
    case PartitionPolicy::kAll:
      return "all";
    case PartitionPolicy::kCompute:
      return "compute";
    case PartitionPolicy::kGemma4:
      return "gemma4";
    case PartitionPolicy::kSafe:
      return "safe";
  }
}

bool IsComputeHeavyOp(const litert::compiler::Op& op) {
  switch (op.Code()) {
    case kLiteRtOpCodeTflBatchMatmul:
    case kLiteRtOpCodeTflConv2d:
    case kLiteRtOpCodeTflFullyConnected:
    case kLiteRtOpCodeTflSoftmax:
      return true;
    default:
      return false;
  }
}

bool IsSafeGemma4Op(const litert::compiler::Op& op) {
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflMul:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflDiv:
    case kLiteRtOpCodeTflMaximum:
    case kLiteRtOpCodeTflRsqrt:
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflSum:
      return true;
    default:
      return false;
  }
}

bool IsValidatedGemma4TensorRtOp(const litert::compiler::Op& op) {
  // The full Gemma4 transformer stack. Delegating whole layers as few large
  // islands is required for output quality: every extra island boundary is a
  // CPU/TensorRT requantization seam, and past ~two dozen seams per token the
  // accumulated drift corrupts greedy decoding.
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflBatchMatmul:
    case kLiteRtOpCodeTflCast:
    case kLiteRtOpCodeTflConcatenation:
    case kLiteRtOpCodeTflCos:
    case kLiteRtOpCodeTflDequantize:
    case kLiteRtOpCodeTflDynamicUpdateSlice:
    case kLiteRtOpCodeTflFill:
    case kLiteRtOpCodeTflFullyConnected:
    case kLiteRtOpCodeTflGelu:
    case kLiteRtOpCodeTflGreaterEqual:
    case kLiteRtOpCodeTflLess:
    case kLiteRtOpCodeTflLogicalAnd:
    case kLiteRtOpCodeTflMaximum:
    case kLiteRtOpCodeTflMul:
    case kLiteRtOpCodeTflNotEqual:
    case kLiteRtOpCodeTflPack:
    case kLiteRtOpCodeTflQuantize:
    case kLiteRtOpCodeTflReduceMax:
    case kLiteRtOpCodeTflReshape:
    case kLiteRtOpCodeTflRsqrt:
    case kLiteRtOpCodeTflSelectV2:
    case kLiteRtOpCodeShloComposite:
    case kLiteRtOpCodeTflSin:
    case kLiteRtOpCodeTflSlice:
    case kLiteRtOpCodeTflSoftmax:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflSum:
    case kLiteRtOpCodeTflTanh:
    case kLiteRtOpCodeTflTranspose:
    case kLiteRtOpCodeTflUnpack:
      return true;
    default:
      return false;
  }
}

bool MatchesOpNameToken(const litert::compiler::Op& op,
                        const std::string& token) {
  if (token.empty()) {
    return false;
  }
  const std::string op_name(litert::GetOpCodeStringView(op.Code()));
  if (token == op_name) {
    return true;
  }
  const size_t dot = op_name.find('.');
  return dot != std::string::npos && token == op_name.substr(dot + 1);
}

bool EnvOpListContains(const char* env_name, const litert::compiler::Op& op) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const std::string list(value);
  size_t begin = 0;
  while (begin < list.size()) {
    size_t end = list.find(',', begin);
    if (end == std::string::npos) {
      end = list.size();
    }
    if (MatchesOpNameToken(op, list.substr(begin, end - begin))) {
      return true;
    }
    begin = end + 1;
  }
  return false;
}

// True if `name` appears in the comma-separated list env var `env_name`.
bool EnvNameListContains(const char* env_name, const char* name) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0' || name == nullptr) {
    return false;
  }
  const std::string list(value);
  const std::string target(name);
  size_t begin = 0;
  while (begin < list.size()) {
    size_t end = list.find(',', begin);
    if (end == std::string::npos) {
      end = list.size();
    }
    if (list.compare(begin, end - begin, target) == 0) {
      return true;
    }
    begin = end + 1;
  }
  return false;
}

bool EnvEnableOpListIsSet() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_ENABLE_OPS");
  return value != nullptr && value[0] != '\0';
}

bool IsPolicyAllowedOp(const litert::compiler::Op& op, PartitionPolicy policy) {
  if (EnvEnableOpListIsSet()) {
    return EnvOpListContains("LITERT_NVIDIA_TENSORRT_ENABLE_OPS", op);
  }
  switch (policy) {
    case PartitionPolicy::kAll:
      return true;
    case PartitionPolicy::kCompute:
      return IsComputeHeavyOp(op);
    case PartitionPolicy::kGemma4:
      return IsValidatedGemma4TensorRtOp(op);
    case PartitionPolicy::kSafe:
      return IsSafeGemma4Op(op);
  }
}

bool IsEnvDisabledOp(const litert::compiler::Op& op) {
  return EnvOpListContains("LITERT_NVIDIA_TENSORRT_DISABLE_OPS", op);
}

size_t EnvSizeT(const char* name, size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const uint64_t parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

bool EnvEnabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  return std::strcmp(value, "0") != 0;
}

std::string Hex64(uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(value));
  return buffer;
}

std::string BuildCompilerSdkVersion() {
  const std::string base_version = "TensorRT-RTX via NvInfer headers " +
                                   std::to_string(NV_TENSORRT_MAJOR) + "." +
                                   std::to_string(NV_TENSORRT_MINOR) + "." +
                                   std::to_string(NV_TENSORRT_PATCH);
  const char* aot_cache_dir =
      std::getenv("LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR");
  if (aot_cache_dir == nullptr || aot_cache_dir[0] == '\0') {
    return base_version;
  }

  // LiteRT includes the SDK-version string in its compilation-cache key. Raw
  // environment values are intentional: an explicit default may cause an
  // extra cache miss, but can never reuse an engine compiled under a different
  // NVIDIA configuration.
  constexpr const char* kCompileEnvironmentVariables[] = {
      "CUDA_VISIBLE_DEVICES",
      "LITERT_NVIDIA_TENSORRT_ALLOW_TF32",
      "LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR",
      "LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH",
      "LITERT_NVIDIA_TENSORRT_BUILDER_OPT_LEVEL",
      "LITERT_NVIDIA_TENSORRT_DISABLE_FC_SHAPES",
      "LITERT_NVIDIA_TENSORRT_DISABLE_INT8_ELEMENTWISE",
      "LITERT_NVIDIA_TENSORRT_DISABLE_OPS",
      "LITERT_NVIDIA_TENSORRT_DISABLE_SUBBYTE_WEIGHTS",
      "LITERT_NVIDIA_TENSORRT_ENABLE_OPS",
      "LITERT_NVIDIA_TENSORRT_FILTER_SMALL_PARTITIONS",
      "LITERT_NVIDIA_TENSORRT_FOLD_INPUT_SCALE",
      "LITERT_NVIDIA_TENSORRT_FP16",
      "LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS",
      "LITERT_NVIDIA_TENSORRT_JIT_HANDLE",
      "LITERT_NVIDIA_TENSORRT_MAX_BATCH_MATMUL_OUTPUT_ELEMENTS",
      "LITERT_NVIDIA_TENSORRT_MAX_FC_WEIGHT_BYTES",
      "LITERT_NVIDIA_TENSORRT_MAX_FILL_BYTES",
      "LITERT_NVIDIA_TENSORRT_MAX_PARTITIONS",
      "LITERT_NVIDIA_TENSORRT_MAX_SELECTED_OPS",
      "LITERT_NVIDIA_TENSORRT_MAX_SOFTMAX_ELEMENTS",
      "LITERT_NVIDIA_TENSORRT_MAX_SUBGRAPH_OPS",
      "LITERT_NVIDIA_TENSORRT_MIN_PARTITION_OPS",
      "LITERT_NVIDIA_TENSORRT_MIN_PARTITION_OUTPUT_BYTES",
      "LITERT_NVIDIA_TENSORRT_MIN_SUBGRAPH_OPS",
      "LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES",
      "LITERT_NVIDIA_TENSORRT_PARTITION_POLICY",
      "LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS",
      "LITERT_NVIDIA_TENSORRT_RUNTIME_BMM_CONTEXT_LIMIT",
      "LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS",
      "LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS",
      "LITERT_NVIDIA_TENSORRT_SYNC_ALLOCATOR",
      "LITERT_NVIDIA_TENSORRT_TACTIC_DRAM_MB",
      "LITERT_NVIDIA_TENSORRT_WORKSPACE_MB",
  };

  std::string configuration =
      "schema=" + std::to_string(kNvidiaCompilerCacheSchemaVersion) + "\n";
  configuration +=
      "tensorrt_runtime=" + std::to_string(getInferLibVersion()) + "\n";
  for (const char* name : kCompileEnvironmentVariables) {
    configuration += name;
    configuration += '=';
    const char* value = std::getenv(name);
    configuration += value == nullptr ? "<unset>" : value;
    configuration += '\n';
  }

  int device = -1;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) == cudaSuccess &&
      cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
    configuration += "gpu=";
    configuration += properties.name;
    configuration += ";sm=" + std::to_string(properties.major) +
                     std::to_string(properties.minor);
    configuration += ";sms=" + std::to_string(properties.multiProcessorCount);
    configuration +=
        ";memory=" + std::to_string(properties.totalGlobalMem) + "\n";
  } else {
    cudaGetLastError();
    configuration += "gpu=<unknown>\n";
  }
  int driver_version = 0;
  if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
    configuration += "cuda_driver=" + std::to_string(driver_version) + "\n";
  }

  const auto fingerprint = litert::nvidia::FingerprintTensorRtArtifact(
      configuration.data(), configuration.size());
  return base_version +
         " cache-schema=" + std::to_string(kNvidiaCompilerCacheSchemaVersion) +
         " config=" + Hex64(fingerprint.low) + Hex64(fingerprint.high);
}

std::string TensorRtAotCacheDir() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR");
  return value == nullptr ? std::string() : std::string(value);
}

Expected<std::string> CanonicalAotCacheDir(const std::string& cache_dir) {
  if (cache_dir.empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT AOT cache directory is empty");
  }
  LITERT_RETURN_IF_ERROR(litert::internal::MkDir(cache_dir));
  char resolved[PATH_MAX];
  if (realpath(cache_dir.c_str(), resolved) == nullptr) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to resolve TensorRT AOT cache directory " + cache_dir +
                     ": " + std::strerror(errno));
  }
  return std::string(resolved);
}

litert::nvidia::TensorRtAotFileIdentity AotFileIdentity(
    const struct stat& stat_buffer) {
  return {static_cast<uint64_t>(stat_buffer.st_dev),
          static_cast<uint64_t>(stat_buffer.st_ino),
          static_cast<int64_t>(stat_buffer.st_mtim.tv_sec),
          static_cast<int64_t>(stat_buffer.st_mtim.tv_nsec),
          static_cast<int64_t>(stat_buffer.st_ctim.tv_sec),
          static_cast<int64_t>(stat_buffer.st_ctim.tv_nsec)};
}

Expected<litert::nvidia::TensorRtAotFileIdentity> SealAndDescribeAotArtifactFd(
    int fd, const std::string& path, size_t expected_size) {
  struct stat stat_buffer{};
  if (fstat(fd, &stat_buffer) != 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to describe TensorRT AOT artifact " + path + ": " +
                     std::strerror(errno));
  }
  if (!S_ISREG(stat_buffer.st_mode) || stat_buffer.st_size < 0 ||
      static_cast<uint64_t>(stat_buffer.st_size) != expected_size) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "TensorRT AOT artifact size or file type changed before "
                 "sealing: " +
                     path);
  }
  if ((stat_buffer.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0 &&
      fchmod(fd, S_IRUSR) != 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to seal TensorRT AOT artifact " + path + ": " +
                     std::strerror(errno));
  }
  // Capture identity from the same descriptor whose bytes were written or
  // compared. If the pathname is concurrently replaced, dispatch observes a
  // different inode and falls back to content validation.
  if (fstat(fd, &stat_buffer) != 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to describe sealed TensorRT AOT artifact " + path +
                     ": " + std::strerror(errno));
  }
  return AotFileIdentity(stat_buffer);
}

Expected<std::optional<litert::nvidia::TensorRtAotFileIdentity>>
ExistingArtifactIdentity(const std::string& path, const uint8_t* data,
                         size_t size, bool* exists) {
  *exists = false;
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    if (errno == ENOENT) {
      return std::optional<litert::nvidia::TensorRtAotFileIdentity>();
    }
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to open TensorRT AOT artifact " + path + ": " +
                     std::strerror(errno));
  }
  *exists = true;
  struct stat stat_buffer{};
  if (fstat(fd, &stat_buffer) != 0) {
    const int saved_errno = errno;
    close(fd);
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to inspect TensorRT AOT artifact " + path + ": " +
                     std::strerror(saved_errno));
  }
  if (!S_ISREG(stat_buffer.st_mode)) {
    close(fd);
    return Error(kLiteRtStatusErrorFileIO,
                 "TensorRT AOT artifact is not a regular file: " + path);
  }
  if (stat_buffer.st_size < 0 || static_cast<uint64_t>(stat_buffer.st_size) !=
                                     static_cast<uint64_t>(size)) {
    close(fd);
    return std::optional<litert::nvidia::TensorRtAotFileIdentity>();
  }
  std::vector<uint8_t> buffer(
      std::min(size, litert::nvidia::kTensorRtAotFingerprintChunkBytes));
  size_t offset = 0;
  while (offset < size) {
    const size_t requested = std::min(buffer.size(), size - offset);
    size_t received = 0;
    while (received < requested) {
      const ssize_t result =
          pread(fd, buffer.data() + received, requested - received,
                static_cast<off_t>(offset + received));
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        const int saved_errno = errno;
        close(fd);
        return Error(
            kLiteRtStatusErrorFileIO,
            "Failed to compare TensorRT AOT artifact " + path + ": " +
                (result == 0 ? std::string("unexpected EOF")
                             : std::string(std::strerror(saved_errno))));
      }
      received += static_cast<size_t>(result);
    }
    if (std::memcmp(buffer.data(), data + offset, received) != 0) {
      close(fd);
      return std::optional<litert::nvidia::TensorRtAotFileIdentity>();
    }
    posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(received),
                  POSIX_FADV_DONTNEED);
    offset += received;
  }
  auto identity = SealAndDescribeAotArtifactFd(fd, path, size);
  if (!identity) {
    close(fd);
    return identity.Error();
  }
  if (close(fd) != 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to close TensorRT AOT artifact " + path + ": " +
                     std::strerror(errno));
  }
  return std::optional<litert::nvidia::TensorRtAotFileIdentity>(*identity);
}

Expected<void> WriteAll(int fd, const uint8_t* data, size_t size,
                        const std::string& path) {
  size_t written = 0;
  while (written < size) {
    const size_t chunk = std::min<size_t>(size - written, size_t{1} << 30);
    const ssize_t result = write(fd, data + written, chunk);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      return Error(kLiteRtStatusErrorFileIO,
                   "Failed to write TensorRT AOT artifact " + path + ": " +
                       std::strerror(errno));
    }
    if (result == 0) {
      return Error(
          kLiteRtStatusErrorFileIO,
          "Short write while persisting TensorRT AOT artifact " + path);
    }
    written += static_cast<size_t>(result);
  }
  return {};
}

Expected<litert::nvidia::TensorRtAotFileIdentity> PersistArtifactFile(
    const std::string& path, const uint8_t* data, size_t size) {
  bool exists = false;
  LITERT_ASSIGN_OR_RETURN(auto matches,
                          ExistingArtifactIdentity(path, data, size, &exists));
  if (matches.has_value()) {
    return *matches;
  }
  if (exists) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Content-addressed TensorRT AOT artifact differs: " + path);
  }

  static std::atomic<uint64_t> sequence{0};
  const std::string temporary_path =
      path + ".tmp." + std::to_string(getpid()) + "." +
      std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
  const int fd = open(temporary_path.c_str(),
                      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to create TensorRT AOT artifact " + temporary_path +
                     ": " + std::strerror(errno));
  }
  auto write_status = WriteAll(fd, data, size, temporary_path);
  if (!write_status) {
    close(fd);
    unlink(temporary_path.c_str());
    return write_status.Error();
  }
  if (fchmod(fd, S_IRUSR) != 0) {
    const int saved_errno = errno;
    close(fd);
    unlink(temporary_path.c_str());
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to seal TensorRT AOT artifact " + temporary_path +
                     ": " + std::strerror(saved_errno));
  }
  if (fsync(fd) != 0) {
    const int saved_errno = errno;
    close(fd);
    unlink(temporary_path.c_str());
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to sync TensorRT AOT artifact " + temporary_path +
                     ": " + std::strerror(saved_errno));
  }
  // Hard-linking publishes the immutable content-addressed artifact without
  // overwriting a file concurrently published by another compiler process.
  if (link(temporary_path.c_str(), path.c_str()) != 0) {
    const int link_errno = errno;
    close(fd);
    unlink(temporary_path.c_str());
    if (link_errno != EEXIST) {
      return Error(kLiteRtStatusErrorFileIO,
                   "Failed to publish TensorRT AOT artifact " + path + ": " +
                       std::strerror(link_errno));
    }
    bool concurrent_exists = false;
    LITERT_ASSIGN_OR_RETURN(
        auto concurrent_matches,
        ExistingArtifactIdentity(path, data, size, &concurrent_exists));
    if (!concurrent_exists || !concurrent_matches.has_value()) {
      return Error(kLiteRtStatusErrorFileIO,
                   "Concurrent TensorRT AOT artifact differs: " + path);
    }
    return *concurrent_matches;
  }
  if (unlink(temporary_path.c_str()) != 0) {
    LITERT_LOG(LITERT_WARNING,
               "NVIDIA TensorRT-RTX could not remove AOT temporary link %s: "
               "%s",
               temporary_path.c_str(), std::strerror(errno));
  }
  auto identity = SealAndDescribeAotArtifactFd(fd, path, size);
  if (!identity) {
    close(fd);
    return identity.Error();
  }
  if (close(fd) != 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to close TensorRT AOT artifact " + path + ": " +
                     std::strerror(errno));
  }
  return *identity;
}

Expected<std::pair<size_t, size_t>> PersistAotArtifact(
    const std::string& canonical_dir, std::vector<uint8_t>& bytecode,
    size_t ordinal, size_t artifact_count) {
  if (bytecode.empty()) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Cannot persist empty TensorRT AOT artifact");
  }
  const auto fingerprint = litert::nvidia::FingerprintTensorRtAotArtifact(
      bytecode.data(), bytecode.size());
  const std::string filename =
      "tensorrt_aot_v" +
      std::to_string(litert::nvidia::kTensorRtAotLocatorVersion) + "_" +
      std::to_string(bytecode.size()) + "_" + Hex64(fingerprint.low) +
      Hex64(fingerprint.high) + ".bin";
  const std::string path = litert::internal::Join({canonical_dir, filename});
  LITERT_ASSIGN_OR_RETURN(
      auto file_identity,
      PersistArtifactFile(path, bytecode.data(), bytecode.size()));
  const size_t artifact_bytes = bytecode.size();
  LITERT_ASSIGN_OR_RETURN(auto locator,
                          litert::nvidia::PackTensorRtAotLocator(
                              {path, static_cast<uint64_t>(artifact_bytes),
                               fingerprint, file_identity}));
  const size_t locator_bytes = locator.size();
  LITERT_LOG(LITERT_INFO,
             "NVIDIA TensorRT-RTX persisted AOT artifact %zu/%zu: "
             "artifact_bytes=%zu locator_bytes=%zu path=%s",
             ordinal + 1, artifact_count, artifact_bytes, locator_bytes,
             path.c_str());
  bytecode = std::move(locator);
  return std::pair<size_t, size_t>{artifact_bytes, locator_bytes};
}

Expected<void> PersistAotArtifacts(
    const std::string& cache_dir,
    std::vector<std::vector<uint8_t>>& bytecodes) {
  LITERT_ASSIGN_OR_RETURN(const std::string canonical_dir,
                          CanonicalAotCacheDir(cache_dir));
  size_t total_artifact_bytes = 0;
  size_t total_locator_bytes = 0;
  for (size_t i = 0; i < bytecodes.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(
        auto persisted,
        PersistAotArtifact(canonical_dir, bytecodes[i], i, bytecodes.size()));
    total_artifact_bytes += persisted.first;
    total_locator_bytes += persisted.second;
  }
  LITERT_LOG(LITERT_INFO,
             "NVIDIA TensorRT-RTX AOT artifacts ready: modules=%zu "
             "artifact_bytes=%zu locator_bytes=%zu",
             bytecodes.size(), total_artifact_bytes, total_locator_bytes);
  return {};
}

template <typename T>
void AppendCacheKeyScalar(std::string& key, T value) {
  const auto* bytes = reinterpret_cast<const char*>(&value);
  key.append(bytes, sizeof(T));
}

void AppendCacheKeyString(std::string& key, absl::string_view value) {
  AppendCacheKeyScalar<uint64_t>(key, value.size());
  key.append(value.data(), value.size());
}

Expected<void> AppendTensorStructure(std::string& key,
                                     const litert::compiler::Tensor& tensor) {
  AppendCacheKeyScalar<uint32_t>(key, tensor.TensorIndex());
  AppendCacheKeyString(key, tensor.Name());
  AppendCacheKeyScalar<int32_t>(key, tensor.TypeId());
  auto ranked_type = tensor.RankedTensorType();
  AppendCacheKeyScalar<uint8_t>(key, ranked_type.HasValue() ? 1 : 0);
  if (ranked_type.HasValue()) {
    AppendCacheKeyScalar<int32_t>(
        key, static_cast<int32_t>(ranked_type->ElementType()));
    const auto& layout = ranked_type->Layout();
    AppendCacheKeyScalar<uint32_t>(key, layout.Rank());
    for (int32_t dimension : layout.Dimensions()) {
      AppendCacheKeyScalar<int32_t>(key, dimension);
    }
    AppendCacheKeyScalar<uint8_t>(key, layout.HasStrides() ? 1 : 0);
    for (uint32_t stride : layout.Strides()) {
      AppendCacheKeyScalar<uint32_t>(key, stride);
    }
  } else {
    auto unranked_type = tensor.UnrankedTensorType();
    AppendCacheKeyScalar<int32_t>(
        key, unranked_type.HasValue()
                 ? static_cast<int32_t>(unranked_type->element_type)
                 : static_cast<int32_t>(kLiteRtElementTypeNone));
  }
  AppendCacheKeyScalar<uint8_t>(key, tensor.HasWeights() ? 1 : 0);
  const auto quantization_type = tensor.QTypeId();
  AppendCacheKeyScalar<int32_t>(key, quantization_type);
  switch (quantization_type) {
    case kLiteRtQuantizationNone:
      break;
    case kLiteRtQuantizationPerTensor: {
      const auto quantization = tensor.PerTensorQuantization();
      AppendCacheKeyScalar<float>(key, quantization.scale);
      AppendCacheKeyScalar<int64_t>(key, quantization.zero_point);
      break;
    }
    case kLiteRtQuantizationPerChannel: {
      const auto quantization = tensor.PerChannelQuantization();
      if (quantization.num_channels >
              std::numeric_limits<size_t>::max() / sizeof(int64_t) ||
          (quantization.num_channels != 0 &&
           (quantization.scales == nullptr ||
            quantization.zero_points == nullptr))) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Invalid per-channel quantization in AOT cache key");
      }
      AppendCacheKeyScalar<int32_t>(key, quantization.quantized_dimension);
      AppendCacheKeyScalar<uint64_t>(key, quantization.num_channels);
      if (quantization.num_channels != 0) {
        key.append(reinterpret_cast<const char*>(quantization.scales),
                   quantization.num_channels * sizeof(float));
        key.append(reinterpret_cast<const char*>(quantization.zero_points),
                   quantization.num_channels * sizeof(int64_t));
      }
      break;
    }
    case kLiteRtQuantizationBlockWise: {
      const auto quantization = tensor.BlockWiseQuantization();
      if (quantization.scales == nullptr ||
          quantization.zero_points == nullptr) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Invalid block-wise quantization in AOT cache key");
      }
      AppendCacheKeyScalar<int32_t>(key, quantization.block_size);
      AppendCacheKeyScalar<uint32_t>(
          key, litert::compiler::Tensor(tensor.ctx(), quantization.scales)
                   .TensorIndex());
      AppendCacheKeyScalar<uint32_t>(
          key, litert::compiler::Tensor(tensor.ctx(), quantization.zero_points)
                   .TensorIndex());
      break;
    }
    default:
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unknown quantization type in AOT cache key");
  }
  return {};
}

Expected<void> AppendPartitionStructure(std::string& key,
                                        const litert::compiler::Model& model) {
  AppendCacheKeyScalar<uint64_t>(key, model.NumSubgraphs());
  for (size_t subgraph_index = 0; subgraph_index < model.NumSubgraphs();
       ++subgraph_index) {
    LITERT_ASSIGN_OR_RETURN(auto subgraph, model.Subgraph(subgraph_index));
    std::map<uint32_t, litert::compiler::Tensor> tensors;
    auto remember_tensor = [&](const litert::compiler::Tensor& tensor) {
      tensors.emplace(tensor.TensorIndex(), tensor);
    };

    const auto inputs = subgraph.Inputs();
    AppendCacheKeyScalar<uint64_t>(key, inputs.size());
    for (const auto& tensor : inputs) {
      AppendCacheKeyScalar<uint32_t>(key, tensor.TensorIndex());
      remember_tensor(tensor);
    }
    const auto outputs = subgraph.Outputs();
    AppendCacheKeyScalar<uint64_t>(key, outputs.size());
    for (const auto& tensor : outputs) {
      AppendCacheKeyScalar<uint32_t>(key, tensor.TensorIndex());
      remember_tensor(tensor);
    }

    const auto ops = subgraph.Ops();
    AppendCacheKeyScalar<uint64_t>(key, ops.size());
    for (const auto& op : ops) {
      AppendCacheKeyScalar<int32_t>(key, op.Code());
      const auto op_inputs = op.Inputs();
      AppendCacheKeyScalar<uint64_t>(key, op_inputs.size());
      for (const auto& tensor : op_inputs) {
        AppendCacheKeyScalar<uint32_t>(key, tensor.TensorIndex());
        remember_tensor(tensor);
      }
      const auto op_outputs = op.Outputs();
      AppendCacheKeyScalar<uint64_t>(key, op_outputs.size());
      for (const auto& tensor : op_outputs) {
        AppendCacheKeyScalar<uint32_t>(key, tensor.TensorIndex());
        remember_tensor(tensor);
      }
      if (op.Code() == kLiteRtOpCodeTflCustom) {
        LITERT_ASSIGN_OR_RETURN(auto custom_code, op.CustomCode());
        LITERT_ASSIGN_OR_RETURN(auto custom_options, op.CustomOptions());
        AppendCacheKeyString(key, custom_code);
        AppendCacheKeyScalar<uint64_t>(key, custom_options.size());
        if (!custom_options.empty()) {
          key.append(reinterpret_cast<const char*>(custom_options.data()),
                     custom_options.size());
        }
      }
    }

    AppendCacheKeyScalar<uint64_t>(key, tensors.size());
    for (const auto& [tensor_index, tensor] : tensors) {
      LITERT_RETURN_IF_ERROR(AppendTensorStructure(key, tensor));
    }
  }
  return {};
}

std::string TensorRtAotModelPath() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("G4MODEL");
  }
  return value == nullptr ? std::string() : std::string(value);
}

Expected<std::optional<litert::nvidia::TensorRtArtifactFingerprint>>
BuildAotCacheKey(const std::string& sdk_version,
                 const litert::compiler::Model& model) {
  const std::string source_path = TensorRtAotModelPath();
  if (source_path.empty()) {
    return std::optional<litert::nvidia::TensorRtArtifactFingerprint>();
  }
  char resolved[PATH_MAX];
  if (realpath(source_path.c_str(), resolved) == nullptr) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to resolve TensorRT AOT source model " + source_path +
                     ": " + std::strerror(errno));
  }
  struct stat stat_buffer{};
  if (stat(resolved, &stat_buffer) != 0 || !S_ISREG(stat_buffer.st_mode) ||
      stat_buffer.st_size < 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "TensorRT AOT source model is not a readable regular file: " +
                     std::string(resolved));
  }

  std::string key;
  key.reserve(1 << 20);
  AppendCacheKeyString(key, sdk_version);
  AppendCacheKeyString(key, resolved);
  AppendCacheKeyScalar<uint64_t>(key, stat_buffer.st_dev);
  AppendCacheKeyScalar<uint64_t>(key, stat_buffer.st_ino);
  AppendCacheKeyScalar<uint64_t>(key, stat_buffer.st_size);
  AppendCacheKeyScalar<int64_t>(key, stat_buffer.st_mtim.tv_sec);
  AppendCacheKeyScalar<int64_t>(key, stat_buffer.st_mtim.tv_nsec);
  AppendCacheKeyScalar<int64_t>(key, stat_buffer.st_ctim.tv_sec);
  AppendCacheKeyScalar<int64_t>(key, stat_buffer.st_ctim.tv_nsec);
  LITERT_RETURN_IF_ERROR(AppendPartitionStructure(key, model));
  return std::optional<litert::nvidia::TensorRtArtifactFingerprint>(
      litert::nvidia::FingerprintTensorRtArtifact(key.data(), key.size()));
}

std::string AotManifestPath(
    const std::string& canonical_dir,
    litert::nvidia::TensorRtArtifactFingerprint cache_key) {
  return litert::internal::Join(
      {canonical_dir,
       "tensorrt_aot_index_v" +
           std::to_string(litert::nvidia::kTensorRtAotManifestVersion) + "_" +
           Hex64(cache_key.low) + Hex64(cache_key.high) + ".bin"});
}

Expected<std::optional<litert::nvidia::TensorRtAotManifest>> TryLoadAotManifest(
    const std::string& manifest_path,
    litert::nvidia::TensorRtArtifactFingerprint expected_cache_key) {
  if (!litert::internal::Exists(manifest_path)) {
    return std::optional<litert::nvidia::TensorRtAotManifest>();
  }
  auto manifest_bytes = litert::internal::LoadBinaryFile(manifest_path);
  if (!manifest_bytes) {
    return manifest_bytes.Error();
  }
  auto manifest = litert::nvidia::ParseTensorRtAotManifest(
      manifest_bytes->Data(), manifest_bytes->Size());
  if (!manifest) {
    LITERT_LOG(LITERT_WARNING,
               "NVIDIA TensorRT-RTX ignored invalid AOT index %s: %s",
               manifest_path.c_str(), manifest.Error().Message().c_str());
    return std::optional<litert::nvidia::TensorRtAotManifest>();
  }
  if (!(manifest->cache_key == expected_cache_key)) {
    LITERT_LOG(LITERT_WARNING,
               "NVIDIA TensorRT-RTX ignored AOT index with a mismatched key: "
               "%s",
               manifest_path.c_str());
    return std::optional<litert::nvidia::TensorRtAotManifest>();
  }
  for (const auto& locator_bytes : manifest->locators) {
    LITERT_ASSIGN_OR_RETURN(auto locator,
                            litert::nvidia::TryParseTensorRtAotLocator(
                                locator_bytes.data(), locator_bytes.size()));
    if (!locator.has_value()) {
      return std::optional<litert::nvidia::TensorRtAotManifest>();
    }
    struct stat artifact_stat{};
    if (stat(locator->path.c_str(), &artifact_stat) != 0 ||
        !S_ISREG(artifact_stat.st_mode) || artifact_stat.st_size < 0 ||
        static_cast<uint64_t>(artifact_stat.st_size) !=
            locator->artifact_size) {
      LITERT_LOG(LITERT_WARNING,
                 "NVIDIA TensorRT-RTX AOT index references a missing or "
                 "size-mismatched artifact: %s",
                 locator->path.c_str());
      return std::optional<litert::nvidia::TensorRtAotManifest>();
    }
  }
  return std::optional<litert::nvidia::TensorRtAotManifest>(
      std::move(*manifest));
}

Expected<void> ReplaceSmallFileAtomically(const std::string& path,
                                          const uint8_t* data, size_t size) {
  static std::atomic<uint64_t> sequence{0};
  const std::string temporary_path =
      path + ".tmp." + std::to_string(getpid()) + "." +
      std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
  const int fd = open(temporary_path.c_str(),
                      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) {
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to create TensorRT AOT index " + temporary_path +
                     ": " + std::strerror(errno));
  }
  auto write_status = WriteAll(fd, data, size, temporary_path);
  if (!write_status) {
    close(fd);
    unlink(temporary_path.c_str());
    return write_status.Error();
  }
  if (fsync(fd) != 0) {
    const int saved_errno = errno;
    close(fd);
    unlink(temporary_path.c_str());
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to sync TensorRT AOT index " + temporary_path + ": " +
                     std::strerror(saved_errno));
  }
  if (close(fd) != 0) {
    const int saved_errno = errno;
    unlink(temporary_path.c_str());
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to close TensorRT AOT index " + temporary_path + ": " +
                     std::strerror(saved_errno));
  }
  if (rename(temporary_path.c_str(), path.c_str()) != 0) {
    const int saved_errno = errno;
    unlink(temporary_path.c_str());
    return Error(kLiteRtStatusErrorFileIO,
                 "Failed to publish TensorRT AOT index " + path + ": " +
                     std::strerror(saved_errno));
  }
  return {};
}

size_t EnvSizeTAllowZero(const char* name, size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const uint64_t parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0') {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

size_t MinPartitionOps(PartitionPolicy policy) {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_MIN_PARTITION_OPS", 8);
}

size_t MinPartitionOutputBytes(PartitionPolicy policy) {
  const size_t safe_default = 1ULL << 20;
  const size_t broad_default = 16ULL << 10;
  return EnvSizeT(
      "LITERT_NVIDIA_TENSORRT_MIN_PARTITION_OUTPUT_BYTES",
      policy == PartitionPolicy::kSafe ? safe_default : broad_default);
}

size_t MaxPartitions(PartitionPolicy policy) {
  const size_t default_value =
      policy == PartitionPolicy::kSafe ? 8 : std::numeric_limits<size_t>::max();
  return EnvSizeTAllowZero("LITERT_NVIDIA_TENSORRT_MAX_PARTITIONS",
                           default_value);
}

size_t MaxSelectedOpsPerSubgraph(PartitionPolicy policy) {
  return EnvSizeTAllowZero("LITERT_NVIDIA_TENSORRT_MAX_SELECTED_OPS",
                           /*default_value=*/0);
}

size_t ElementByteSize(litert::ElementType type) {
  switch (type) {
    case litert::ElementType::Float32:
    case litert::ElementType::Int32:
      return 4;
    case litert::ElementType::Float16:
    case litert::ElementType::BFloat16:
      return 2;
    case litert::ElementType::Float8E4M3FN:
    case litert::ElementType::Int8:
    case litert::ElementType::Bool:
      return 1;
    case litert::ElementType::Int4:
      return 1;
    case litert::ElementType::Int64:
      return 8;
    default:
      return 0;
  }
}

size_t TensorByteSize(const litert::compiler::Tensor& tensor) {
  auto ranked_type = tensor.RankedTensorType();
  if (!ranked_type) {
    return 0;
  }
  auto num_elements = ranked_type->Layout().NumElements();
  if (!num_elements) {
    return 0;
  }
  if (ranked_type->ElementType() == litert::ElementType::Int4) {
    return (*num_elements + 1) / 2;
  }
  return *num_elements * ElementByteSize(ranked_type->ElementType());
}

size_t PartitionOutputBytes(const LiteRtCompilerContext* ctx,
                            const std::vector<LiteRtOp>& partition) {
  size_t bytes = 0;
  for (auto* raw_op : partition) {
    litert::compiler::Op op(ctx, raw_op);
    for (const auto& output : op.Outputs()) {
      bytes += TensorByteSize(output);
    }
  }
  return bytes;
}

bool ContainsComputeDenseOp(const LiteRtCompilerContext* ctx,
                            const std::vector<LiteRtOp>& partition) {
  for (auto* raw_op : partition) {
    switch (litert::compiler::Op(ctx, raw_op).Code()) {
      case kLiteRtOpCodeTflBatchMatmul:
      case kLiteRtOpCodeTflConv2d:
      case kLiteRtOpCodeTflFullyConnected:
        return true;
      default:
        break;
    }
  }
  return false;
}

bool ContainsReductionOp(const LiteRtCompilerContext* ctx,
                         const std::vector<LiteRtOp>& partition) {
  for (auto* raw_op : partition) {
    switch (litert::compiler::Op(ctx, raw_op).Code()) {
      case kLiteRtOpCodeTflSoftmax:
      case kLiteRtOpCodeTflSum:
      case kLiteRtOpCodeTflReduceMax:
        return true;
      default:
        break;
    }
  }
  return false;
}

bool ShouldKeepPartition(const LiteRtCompilerContext* ctx,
                         const std::vector<LiteRtOp>& partition,
                         PartitionPolicy policy, size_t output_bytes) {
  if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_FILTER_SMALL_PARTITIONS",
                  /*default_value=*/true)) {
    return true;
  }
  if (ContainsComputeDenseOp(ctx, partition)) {
    return true;
  }
  if (ContainsReductionOp(ctx, partition) &&
      output_bytes >= MinPartitionOutputBytes(policy)) {
    return true;
  }
  return partition.size() >= MinPartitionOps(policy) &&
         output_bytes >= MinPartitionOutputBytes(policy);
}

std::vector<LiteRtOpWithPartitionIndex> FilterSmallPartitions(
    const LiteRtCompilerContext* ctx,
    const std::vector<LiteRtOpWithPartitionIndex>& selected_ops,
    LiteRtSubgraph subgraph, PartitionPolicy policy) {
  auto islands = litert::internal::GroupPartitionsV2(selected_ops, subgraph);
  struct IslandInfo {
    size_t index;
    size_t ops;
    size_t output_bytes;
  };
  std::vector<IslandInfo> keep_candidates;
  std::unordered_set<LiteRtOp> kept_ops;
  size_t dropped_islands = 0;
  size_t dropped_ops = 0;
  for (size_t i = 0; i < islands.size(); ++i) {
    const auto& island = islands[i];
    const size_t output_bytes = PartitionOutputBytes(ctx, island);
    if (ShouldKeepPartition(ctx, island, policy, output_bytes)) {
      keep_candidates.push_back({i, island.size(), output_bytes});
    } else {
      ++dropped_islands;
      dropped_ops += island.size();
    }
  }

  const size_t max_partitions = MaxPartitions(policy);
  size_t capped_islands = 0;
  size_t capped_ops = 0;
  std::unordered_set<size_t> kept_island_indexes;
  if (max_partitions != 0 && keep_candidates.size() > max_partitions) {
    std::vector<IslandInfo> ranked = keep_candidates;
    std::sort(ranked.begin(), ranked.end(),
              [](const IslandInfo& lhs, const IslandInfo& rhs) {
                if (lhs.output_bytes != rhs.output_bytes) {
                  return lhs.output_bytes > rhs.output_bytes;
                }
                if (lhs.ops != rhs.ops) {
                  return lhs.ops > rhs.ops;
                }
                return lhs.index < rhs.index;
              });
    for (size_t i = 0; i < max_partitions; ++i) {
      kept_island_indexes.insert(ranked[i].index);
    }
    for (size_t i = max_partitions; i < ranked.size(); ++i) {
      ++capped_islands;
      capped_ops += ranked[i].ops;
    }
  } else {
    for (const auto& candidate : keep_candidates) {
      kept_island_indexes.insert(candidate.index);
    }
  }

  for (const auto& index : kept_island_indexes) {
    const auto& island = islands[index];
    kept_ops.insert(island.begin(), island.end());
  }

  std::vector<LiteRtOpWithPartitionIndex> filtered;
  filtered.reserve(kept_ops.size());
  for (const auto& selected_op : selected_ops) {
    if (kept_ops.find(selected_op.first) != kept_ops.end()) {
      filtered.push_back(selected_op);
    }
  }
  LITERT_LOG(LITERT_INFO,
             "NVIDIA TensorRT-RTX partition filter: candidate_islands=%zu "
             "candidate_ops=%zu kept_islands=%zu kept_ops=%zu "
             "dropped_islands=%zu dropped_ops=%zu capped_islands=%zu "
             "capped_ops=%zu min_ops=%zu min_output_bytes=%zu "
             "max_partitions=%zu",
             islands.size(), selected_ops.size(), kept_island_indexes.size(),
             filtered.size(), dropped_islands, dropped_ops, capped_islands,
             capped_ops, MinPartitionOps(policy),
             MinPartitionOutputBytes(policy), max_partitions);
  return filtered;
}

}  // namespace

struct LiteRtCompiledResultT {
  std::vector<std::vector<uint8_t>> bytecodes;
  std::vector<std::string> call_infos;
  std::vector<LiteRtParamIndex> bytecode_indices;
  std::vector<litert::nvidia::TensorRtJitExecutable> jit_executables;
};

struct LiteRtCompilerPluginT {
  explicit LiteRtCompilerPluginT(const LiteRtCompilerContext* ctx)
      : ctx(ctx), sdk_version(BuildCompilerSdkVersion()) {}
  const LiteRtCompilerContext* ctx = nullptr;
  std::string sdk_version;
};

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = static_cast<LiteRtHwAccelerators>(
      kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorNpu);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (!compiler_plugin || !num_supported_soc_models) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = 1;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (!compiler_plugin || !soc_model_name || soc_model_idx != 0) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = kPluginSocModel;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (!compiler_plugin || !sdk_version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version = compiler_plugin->sdk_version.c_str();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->bytecodes.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size ||
      byte_code_idx >= compiled_result->bytecodes.size()) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& bytecode = compiled_result->bytecodes[byte_code_idx];
  *byte_code = bytecode.data();
  *byte_code_size = bytecode.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size || !byte_code_idx) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (call_idx >= compiled_result->call_infos.size() ||
      call_idx >= compiled_result->bytecode_indices.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  const auto& info = compiled_result->call_infos[call_idx];
  *call_info = info.data();
  *call_info_size = info.size();
  *byte_code_idx = compiled_result->bytecode_indices[call_idx];
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->call_infos.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

LiteRtStatus LiteRtGetCompiledResultHandle(LiteRtCompiledResult compiled_result,
                                           LiteRtParamIndex byte_code_idx,
                                           LiteRtJitExecutable* handle) {
  if (!compiled_result || !handle) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (compiled_result->jit_executables.empty()) {
    *handle = nullptr;
    return kLiteRtStatusOk;
  }
  if (byte_code_idx >= compiled_result->jit_executables.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *handle = reinterpret_cast<LiteRtJitExecutable>(
      &compiled_result->jit_executables[byte_code_idx]);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  if (compiler_context == nullptr || compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtPropagateMinLoggerSeverityWithCompilerContext(compiler_context, env);
  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context);
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  if (!compiler_plugin || !subgraph || !selected_ops) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  litert::compiler::Subgraph graph(compiler_plugin->ctx, subgraph);
  // Leave named subgraphs on CPU (comma list). Each delegated signature's
  // engines embed their own weight copies, so skipping rarely-used signatures
  // (e.g. "verify", "prefill_1024") cuts device memory roughly per signature.
  {
    const std::string subgraph_name(subgraph->Name());
    if (!subgraph_name.empty() &&
        EnvNameListContains("LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS",
                            subgraph_name.c_str())) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX skipping subgraph `%s` "
                 "(LITERT_NVIDIA_TENSORRT_SKIP_SUBGRAPHS)",
                 subgraph_name.c_str());
      return kLiteRtStatusOk;
    }
  }
  // Debug bounds to delegate only subgraphs within an op-count range (e.g. to
  // isolate decode-vs-prefill behavior); 0 disables each bound.
  const size_t min_subgraph_ops =
      EnvSizeTAllowZero("LITERT_NVIDIA_TENSORRT_MIN_SUBGRAPH_OPS", 0);
  const size_t max_subgraph_ops =
      EnvSizeTAllowZero("LITERT_NVIDIA_TENSORRT_MAX_SUBGRAPH_OPS", 0);
  const size_t num_graph_ops = graph.Ops().size();
  if ((min_subgraph_ops != 0 && num_graph_ops < min_subgraph_ops) ||
      (max_subgraph_ops != 0 && num_graph_ops > max_subgraph_ops)) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX skipping subgraph with %zu ops "
               "(bounds min=%zu max=%zu)",
               num_graph_ops, min_subgraph_ops, max_subgraph_ops);
    return kLiteRtStatusOk;
  }
  const PartitionPolicy policy = GetPartitionPolicy();
  int total_ops = 0;
  int selected_op_count = 0;
  int policy_rejected_op_count = 0;
  int cap_rejected_op_count = 0;
  std::map<LiteRtOpCode, int> unsupported_counts;
  std::map<LiteRtOpCode, int> policy_rejected_counts;
  std::map<LiteRtOpCode, int> cap_rejected_counts;
  std::vector<LiteRtOpWithPartitionIndex> candidate_ops;
  const size_t max_selected_ops = MaxSelectedOpsPerSubgraph(policy);
  for (const auto& op : graph.Ops()) {
    ++total_ops;
    const bool supported = litert::nvidia::IsTensorRtOpSupported(op);
    const bool policy_allowed = IsPolicyAllowedOp(op, policy);
    const bool env_disabled = IsEnvDisabledOp(op);
    if (supported && policy_allowed && !env_disabled) {
      if (max_selected_ops == 0 || candidate_ops.size() < max_selected_ops) {
        candidate_ops.push_back({op.Get(), 0});
        ++selected_op_count;
      } else {
        ++cap_rejected_op_count;
        ++cap_rejected_counts[op.Code()];
      }
    } else if (!supported &&
               (policy == PartitionPolicy::kAll || policy_allowed)) {
      ++unsupported_counts[op.Code()];
      if (LogUnsupportedOps()) {
        const std::string op_summary = OpSummary(op);
        LITERT_LOG(LITERT_INFO, "NVIDIA TensorRT-RTX unsupported op detail: %s",
                   op_summary.c_str());
      }
    } else {
      ++policy_rejected_op_count;
      ++policy_rejected_counts[op.Code()];
    }
  }
  LITERT_LOG(LITERT_INFO,
             "NVIDIA TensorRT-RTX partition scan: soc=%s total_ops=%d "
             "selected_ops=%d unsupported_kinds=%zu policy=%s "
             "policy_rejected_ops=%d policy_rejected_kinds=%zu "
             "cap_rejected_ops=%d cap_rejected_kinds=%zu max_selected_ops=%zu",
             soc_model ? soc_model : "(null)", total_ops, selected_op_count,
             unsupported_counts.size(), PartitionPolicyName(policy),
             policy_rejected_op_count, policy_rejected_counts.size(),
             cap_rejected_op_count, cap_rejected_counts.size(),
             max_selected_ops);
  for (const auto& [code, count] : unsupported_counts) {
    LITERT_LOG(LITERT_INFO, "NVIDIA TensorRT-RTX unsupported op: %s count=%d",
               std::string(litert::GetOpCodeStringView(code)).c_str(), count);
  }
  for (const auto& [code, count] : policy_rejected_counts) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX policy skipped op: %s count=%d",
               std::string(litert::GetOpCodeStringView(code)).c_str(), count);
  }
  for (const auto& [code, count] : cap_rejected_counts) {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX selected-op cap skipped op: %s count=%d",
               std::string(litert::GetOpCodeStringView(code)).c_str(), count);
  }
  auto filtered_ops = FilterSmallPartitions(compiler_plugin->ctx, candidate_ops,
                                            subgraph, policy);
  for (const auto& op : filtered_ops) {
    LITERT_RETURN_IF_ERROR(
        compiler_plugin->ctx->push_op(selected_ops, op.first, op.second));
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (!compiler_plugin || !partitions || !compiled_result) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const litert::nvidia::MemoryProfiler memory_profiler("compiler");
  litert::compiler::Model model(compiler_plugin->ctx, partitions);
  auto result = std::make_unique<LiteRtCompiledResultT>();
  const auto num_partitions = model.NumSubgraphs();
  const bool shared_weights = litert::nvidia::TensorRtSharedWeightsEnabled();
  const std::string aot_cache_dir = TensorRtAotCacheDir();
  std::string canonical_aot_cache_dir;
  std::optional<litert::nvidia::TensorRtArtifactFingerprint> aot_cache_key;
  std::string aot_manifest_path;
  if (!aot_cache_dir.empty()) {
    memory_profiler.Log("aot_cache_lookup_begin", soc_model);
    LITERT_ASSIGN_OR_RETURN(canonical_aot_cache_dir,
                            CanonicalAotCacheDir(aot_cache_dir));
    LITERT_ASSIGN_OR_RETURN(
        aot_cache_key, BuildAotCacheKey(compiler_plugin->sdk_version, model));
    if (aot_cache_key.has_value()) {
      aot_manifest_path =
          AotManifestPath(canonical_aot_cache_dir, *aot_cache_key);
      LITERT_ASSIGN_OR_RETURN(
          auto cached_manifest,
          TryLoadAotManifest(aot_manifest_path, *aot_cache_key));
      if (cached_manifest.has_value()) {
        bool call_layout_matches =
            cached_manifest->call_infos.size() == num_partitions;
        for (size_t i = 0; call_layout_matches && i < num_partitions; ++i) {
          call_layout_matches =
              cached_manifest->call_infos[i] == PartitionName(i);
        }
        if (call_layout_matches) {
          result->bytecodes = std::move(cached_manifest->locators);
          result->call_infos = std::move(cached_manifest->call_infos);
          result->bytecode_indices.reserve(
              cached_manifest->bytecode_indices.size());
          for (uint32_t index : cached_manifest->bytecode_indices) {
            result->bytecode_indices.push_back(index);
          }
          LITERT_LOG(
              LITERT_INFO,
              "NVIDIA TensorRT-RTX AOT cache hit: partitions=%zu modules=%zu "
              "index=%s",
              num_partitions, result->bytecodes.size(),
              aot_manifest_path.c_str());
          memory_profiler.Log("aot_cache_hit", soc_model);
          *compiled_result = result.release();
          return kLiteRtStatusOk;
        }
        LITERT_LOG(LITERT_WARNING,
                   "NVIDIA TensorRT-RTX ignored AOT index with mismatched "
                   "partition calls: %s",
                   aot_manifest_path.c_str());
      }
      LITERT_LOG(LITERT_INFO, "NVIDIA TensorRT-RTX AOT cache miss: index=%s",
                 aot_manifest_path.c_str());
      memory_profiler.Log("aot_cache_miss", soc_model);
    } else {
      LITERT_LOG(
          LITERT_INFO,
          "NVIDIA TensorRT-RTX AOT direct lookup is disabled because neither "
          "LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH nor G4MODEL is set; the "
          "locator remains compatible with LiteRT's compilation cache");
    }
  }
  result->bytecodes.reserve(
      shared_weights && aot_cache_dir.empty() ? 1 : num_partitions);
  result->call_infos.reserve(num_partitions);
  result->bytecode_indices.reserve(num_partitions);
  SharedWeightDeduper shared_weight_store;
  std::vector<PendingBundleEntry> pending_bundle_entries;
  pending_bundle_entries.reserve(shared_weights ? num_partitions : 0);
  size_t total_bytecode_bytes = 0;
  bool aot_artifacts_persisted = false;
  memory_profiler.Log("compile_begin", soc_model);

  for (LiteRtParamIndex i = 0; i < num_partitions; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto subgraph, model.Subgraph(i));
    const auto function_name = PartitionName(i);
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX compiling partition %d/%d with %d ops",
               static_cast<int>(i + 1), static_cast<int>(num_partitions),
               static_cast<int>(subgraph.Ops().size()));
    if (LogPartitionOps()) {
      for (const auto& op : subgraph.Ops()) {
        const std::string op_summary = OpSummary(op);
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT-RTX partition %d op detail: %s",
                   static_cast<int>(i + 1), op_summary.c_str());
      }
    }
    memory_profiler.Log("partition_build_begin", function_name.c_str());
    auto engine_or = litert::nvidia::BuildTensorRtEngine(subgraph);
    if (!engine_or) {
      std::string op_codes;
      for (const auto& op : subgraph.Ops()) {
        if (!op_codes.empty()) {
          op_codes += ", ";
        }
        op_codes += std::string(litert::GetOpCodeStringView(op.Code()));
      }
      LITERT_LOG(LITERT_ERROR,
                 "NVIDIA TensorRT-RTX failed to compile partition %d/%d "
                 "with %d ops: %s. Ops: [%s]",
                 static_cast<int>(i + 1), static_cast<int>(num_partitions),
                 static_cast<int>(subgraph.Ops().size()),
                 engine_or.Error().Message().c_str(), op_codes.c_str());
      for (const auto& op : subgraph.Ops()) {
        const std::string op_summary = OpSummary(op);
        LITERT_LOG(LITERT_ERROR, "NVIDIA TensorRT-RTX failed op detail: %s",
                   op_summary.c_str());
      }
      return engine_or.Error().Status();
    }
    auto engine = std::move(*engine_or);
    memory_profiler.Log("partition_build_end", function_name.c_str());
    if (shared_weights) {
      PendingBundleEntry pending;
      pending.function_name = function_name;
      pending.input_names = std::move(engine.input_names);
      pending.output_names = std::move(engine.output_names);
      pending.engine = std::move(engine.engine);
      pending.trtllm_head = std::move(engine.trtllm_head);
      pending.refit_weights.reserve(engine.refit_weights.size());
      size_t partition_logical_weight_bytes = 0;
      for (auto& weight : engine.refit_weights) {
        partition_logical_weight_bytes += weight.data.size();
        const std::string name = weight.name;
        const uint32_t shared_index =
            shared_weight_store.Add(std::move(weight));
        pending.refit_weights.push_back({name, shared_index});
      }
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX compiled %s partition %d/%d: "
                 "plan_bytes=%zu refit_weights=%zu logical_weight_bytes=%zu "
                 "cumulative_unique_weight_bytes=%zu",
                 engine.is_stripped_plan ? "stripped" : "self-contained",
                 static_cast<int>(i + 1), static_cast<int>(num_partitions),
                 pending.engine.size(), pending.refit_weights.size(),
                 partition_logical_weight_bytes,
                 shared_weight_store.unique_bytes());
      result->call_infos.push_back(function_name);
      pending_bundle_entries.push_back(std::move(pending));
      memory_profiler.Log("partition_retained", function_name.c_str());
      continue;
    }
    litert::nvidia::TensorRtLlmHead trtllm_head;
    const litert::nvidia::TensorRtLlmHead* trtllm_head_ptr = nullptr;
    if (engine.trtllm_head.has_value()) {
      trtllm_head.hidden_output_port = engine.trtllm_head->hidden_output_port;
      trtllm_head.logits_output_port = engine.trtllm_head->logits_output_port;
      trtllm_head.k = engine.trtllm_head->k;
      trtllm_head.n = engine.trtllm_head->n;
      trtllm_head.soft_cap = engine.trtllm_head->soft_cap;
      trtllm_head.weight_format = engine.trtllm_head->weight_format;
      trtllm_head.packed_weights = engine.trtllm_head->packed_weights.data();
      trtllm_head.packed_weights_size =
          engine.trtllm_head->packed_weights.size();
      trtllm_head.bf16_scales = engine.trtllm_head->bf16_scales.data();
      trtllm_head.bf16_scales_size = engine.trtllm_head->bf16_scales.size();
      trtllm_head_ptr = &trtllm_head;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto packed,
        litert::nvidia::PackTensorRtBytecode(
            function_name, engine.input_names, engine.output_names,
            engine.engine.data(), engine.engine.size(), trtllm_head_ptr));
    total_bytecode_bytes += packed.size();
    LITERT_LOG(
        LITERT_INFO,
        "NVIDIA TensorRT-RTX compiled partition %d/%d: "
        "engine_bytes=%zu bytecode_bytes=%zu cumulative_bytecode_bytes=%zu",
        static_cast<int>(i + 1), static_cast<int>(num_partitions),
        engine.engine.size(), packed.size(), total_bytecode_bytes);
    result->call_infos.push_back(function_name);
    result->bytecode_indices.push_back(result->bytecodes.size());
    result->bytecodes.push_back(std::move(packed));
    memory_profiler.Log("partition_retained", function_name.c_str());
  }

  if (shared_weights) {
    memory_profiler.Log("bundle_pack_begin", soc_model);
    std::vector<litert::nvidia::TensorRtLlmHead> trtllm_heads(
        pending_bundle_entries.size());
    std::vector<litert::nvidia::TensorRtBundleEntry> bundle_entries;
    bundle_entries.reserve(pending_bundle_entries.size());
    for (size_t i = 0; i < pending_bundle_entries.size(); ++i) {
      auto& pending = pending_bundle_entries[i];
      const litert::nvidia::TensorRtLlmHead* head_ptr = nullptr;
      if (pending.trtllm_head.has_value()) {
        auto& head = trtllm_heads[i];
        head.hidden_output_port = pending.trtllm_head->hidden_output_port;
        head.logits_output_port = pending.trtllm_head->logits_output_port;
        head.k = pending.trtllm_head->k;
        head.n = pending.trtllm_head->n;
        head.soft_cap = pending.trtllm_head->soft_cap;
        head.weight_format = pending.trtllm_head->weight_format;
        head.packed_weights = pending.trtllm_head->packed_weights.data();
        head.packed_weights_size = pending.trtllm_head->packed_weights.size();
        head.bf16_scales = pending.trtllm_head->bf16_scales.data();
        head.bf16_scales_size = pending.trtllm_head->bf16_scales.size();
        head_ptr = &head;
      }
      bundle_entries.push_back({pending.function_name, pending.input_names,
                                pending.output_names, pending.engine.data(),
                                pending.engine.size(), head_ptr,
                                pending.refit_weights});
    }
    if (aot_cache_dir.empty()) {
      LITERT_ASSIGN_OR_RETURN(
          auto packed_bundle,
          litert::nvidia::PackTensorRtSharedWeightBundle(
              shared_weight_store.weights(), bundle_entries));
      total_bytecode_bytes = packed_bundle.size();
      result->bytecodes.push_back(std::move(packed_bundle));
      result->bytecode_indices.assign(bundle_entries.size(), 0);
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX packed shared-weight bundle: engines=%zu "
                 "shared_weights=%zu logical_weight_bytes=%zu "
                 "unique_weight_bytes=%zu saved_serialized_weight_bytes=%zu "
                 "bytecode_bytes=%zu",
                 bundle_entries.size(), shared_weight_store.weights().size(),
                 shared_weight_store.logical_bytes(),
                 shared_weight_store.unique_bytes(),
                 shared_weight_store.logical_bytes() -
                     shared_weight_store.unique_bytes(),
                 total_bytecode_bytes);
      memory_profiler.Log("bundle_pack_end", soc_model);
    } else {
      memory_profiler.Log("aot_persist_begin", soc_model);
      size_t total_locator_bytes = 0;
      size_t total_shard_weight_bytes = 0;
      for (size_t i = 0; i < bundle_entries.size(); ++i) {
        const auto& entry = bundle_entries[i];
        memory_profiler.Log("aot_shard_pack_begin",
                            entry.function_name.c_str());
        LITERT_ASSIGN_OR_RETURN(auto packed_shard,
                                litert::nvidia::PackTensorRtSharedWeightShard(
                                    shared_weight_store.weights(), entry));
        memory_profiler.Log("aot_shard_pack_end", entry.function_name.c_str());
        size_t shard_weight_bytes = 0;
        std::unordered_set<uint32_t> shard_weight_indices;
        for (const auto& ref : entry.refit_weights) {
          if (shard_weight_indices.insert(ref.shared_weight_index).second) {
            shard_weight_bytes +=
                shared_weight_store.weights()[ref.shared_weight_index]
                    .data.size();
          }
        }
        const size_t shard_bytes = packed_shard.size();
        LITERT_ASSIGN_OR_RETURN(
            auto persisted,
            PersistAotArtifact(canonical_aot_cache_dir, packed_shard, i,
                               bundle_entries.size()));
        total_bytecode_bytes += persisted.first;
        total_locator_bytes += persisted.second;
        total_shard_weight_bytes += shard_weight_bytes;
        result->bytecode_indices.push_back(result->bytecodes.size());
        result->bytecodes.push_back(std::move(packed_shard));
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT-RTX packed AOT shard %zu/%zu: function=%s "
                   "artifact_bytes=%zu referenced_weight_bytes=%zu",
                   i + 1, bundle_entries.size(), entry.function_name.c_str(),
                   shard_bytes, shard_weight_bytes);
        memory_profiler.Log("aot_shard_persisted", entry.function_name.c_str());
      }
      LITERT_LOG(
          LITERT_INFO,
          "NVIDIA TensorRT-RTX AOT shards ready: modules=%zu "
          "artifact_bytes=%zu locator_bytes=%zu referenced_weight_bytes=%zu "
          "cross_shard_duplicate_weight_bytes=%zu",
          result->bytecodes.size(), total_bytecode_bytes, total_locator_bytes,
          total_shard_weight_bytes,
          total_shard_weight_bytes - shared_weight_store.unique_bytes());
      memory_profiler.Log("bundle_pack_end", soc_model);
      memory_profiler.Log("aot_persist_end", soc_model);
      aot_artifacts_persisted = true;
    }
  }

  if (!aot_cache_dir.empty() && !aot_artifacts_persisted) {
    memory_profiler.Log("aot_persist_begin", soc_model);
    LITERT_RETURN_IF_ERROR(
        PersistAotArtifacts(aot_cache_dir, result->bytecodes));
    memory_profiler.Log("aot_persist_end", soc_model);
  }
  if (!aot_cache_dir.empty() && aot_cache_key.has_value()) {
    litert::nvidia::TensorRtAotManifest manifest;
    manifest.cache_key = *aot_cache_key;
    manifest.locators = result->bytecodes;
    manifest.call_infos = result->call_infos;
    manifest.bytecode_indices.reserve(result->bytecode_indices.size());
    for (LiteRtParamIndex index : result->bytecode_indices) {
      if (index > std::numeric_limits<uint32_t>::max()) {
        return kLiteRtStatusErrorInvalidArgument;
      }
      manifest.bytecode_indices.push_back(static_cast<uint32_t>(index));
    }
    LITERT_ASSIGN_OR_RETURN(auto packed_manifest,
                            litert::nvidia::PackTensorRtAotManifest(manifest));
    LITERT_RETURN_IF_ERROR(ReplaceSmallFileAtomically(
        aot_manifest_path, packed_manifest.data(), packed_manifest.size()));
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX published AOT index: bytes=%zu path=%s",
               packed_manifest.size(), aot_manifest_path.c_str());
    memory_profiler.Log("aot_index_published", soc_model);
  }

  if (!aot_cache_dir.empty() && EnvEnabled("LITERT_NVIDIA_TENSORRT_JIT_HANDLE",
                                           /*default_value=*/false)) {
    LITERT_LOG(LITERT_WARNING,
               "NVIDIA TensorRT-RTX AOT artifacts supersede in-memory JIT "
               "handles so LiteRT can persist the rewritten model cache");
  } else if (EnvEnabled("LITERT_NVIDIA_TENSORRT_JIT_HANDLE",
                        /*default_value=*/false)) {
    result->jit_executables.reserve(result->bytecodes.size());
    for (const auto& bytecode : result->bytecodes) {
      result->jit_executables.push_back(
          {litert::nvidia::kTensorRtJitExecutableMagic,
           litert::nvidia::kTensorRtJitExecutableVersion,
           /*reserved=*/0, bytecode.data(),
           static_cast<uint64_t>(bytecode.size())});
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX exposing %zu in-memory JIT executable "
               "handle(s); bytecode will not be copied into the rewritten "
               "LiteRT FlatBuffer",
               result->jit_executables.size());
    memory_profiler.Log("jit_handles_ready", soc_model);
  }

  memory_profiler.Log("compile_end", soc_model);
  *compiled_result = result.release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  if (!transformations || !num_patterns) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *transformations = nullptr;
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  static constexpr LiteRtApiVersion kApiVersion{LITERT_API_VERSION_MAJOR,
                                                LITERT_API_VERSION_MINOR,
                                                LITERT_API_VERSION_PATCH};
  if (LiteRtCompareApiVersion(api_version, kApiVersion) > 0) {
    LITERT_LOG(LITERT_ERROR,
               "LiteRT caller API version is newer than NVIDIA compiler "
               "plugin API version");
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }
  if (soc_model_name != nullptr &&
      std::string(soc_model_name) != kPluginSocModel) {
    LITERT_LOG(LITERT_WARNING,
               "NVIDIA TensorRT-RTX compiler accepts RTX CUDA targets; "
               "requested SoC model was %s",
               soc_model_name);
  }
  return kLiteRtStatusOk;
}
