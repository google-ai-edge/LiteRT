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

#ifndef ODML_LITERT_LITERT_VENDORS_NVIDIA_BYTECODE_H_
#define ODML_LITERT_LITERT_VENDORS_NVIDIA_BYTECODE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "litert/cc/litert_expected.h"

namespace litert::nvidia {

inline constexpr uint32_t kTensorRtBytecodeVersion = 1;
inline constexpr uint32_t kTensorRtBytecodeVersionWithTrtLlmHead = 2;
inline constexpr uint32_t kTensorRtBytecodeVersionWithTypedHead = 3;
inline constexpr uint32_t kTensorRtBytecodeVersionWithSharedWeights = 4;

// Versioned POD payload behind LiteRtJitExecutable. The compiler plugin owns
// the bytecode for the lifetime of the compiled result; dispatch reads it
// directly instead of forcing LiteRT to copy it into the JIT-rewritten
// FlatBuffer.
inline constexpr uint64_t kTensorRtJitExecutableMagic =
    0x4a49544e56545254ULL;  // "JITNVTRT"
inline constexpr uint32_t kTensorRtJitExecutableVersion = 1;

// A small, serializable locator used by the persistent AOT path. The LiteRT
// model cache stores this locator while the large TensorRT bytecode bundle
// remains in a separate immutable, memory-mappable artifact file.
inline constexpr uint32_t kTensorRtAotLocatorVersion = 2;
inline constexpr uint32_t kTensorRtAotManifestVersion = 1;
inline constexpr size_t kTensorRtAotFingerprintChunkBytes = 4ULL << 20;

struct TensorRtJitExecutable {
  uint64_t magic = kTensorRtJitExecutableMagic;
  uint32_t version = kTensorRtJitExecutableVersion;
  uint32_t reserved = 0;
  const void* bytecode_data = nullptr;
  uint64_t bytecode_size = 0;
};

struct TensorRtArtifactFingerprint {
  uint64_t low = 0;
  uint64_t high = 0;

  friend bool operator==(const TensorRtArtifactFingerprint& lhs,
                         const TensorRtArtifactFingerprint& rhs) {
    return lhs.low == rhs.low && lhs.high == rhs.high;
  }
};

// File identity captured after an AOT artifact has been atomically published
// and sealed read-only. Dispatch can trust the content fingerprint already
// computed by that writer while every field still matches the opened inode.
struct TensorRtAotFileIdentity {
  uint64_t device = 0;
  uint64_t inode = 0;
  int64_t mtime_seconds = 0;
  int64_t mtime_nanoseconds = 0;
  int64_t ctime_seconds = 0;
  int64_t ctime_nanoseconds = 0;

  friend bool operator==(const TensorRtAotFileIdentity& lhs,
                         const TensorRtAotFileIdentity& rhs) {
    return lhs.device == rhs.device && lhs.inode == rhs.inode &&
           lhs.mtime_seconds == rhs.mtime_seconds &&
           lhs.mtime_nanoseconds == rhs.mtime_nanoseconds &&
           lhs.ctime_seconds == rhs.ctime_seconds &&
           lhs.ctime_nanoseconds == rhs.ctime_nanoseconds;
  }
};

struct TensorRtAotLocator {
  std::string path;
  uint64_t artifact_size = 0;
  TensorRtArtifactFingerprint fingerprint;
  std::optional<TensorRtAotFileIdentity> file_identity;
  uint32_t version = kTensorRtAotLocatorVersion;
};

struct TensorRtAotManifest {
  TensorRtArtifactFingerprint cache_key;
  std::vector<std::vector<uint8_t>> locators;
  std::vector<std::string> call_infos;
  std::vector<uint32_t> bytecode_indices;
};

enum class TensorRtLlmHeadWeightFormat : uint32_t {
  kInvalid = 0,
  kInt4ColumnMajorInterleaved = 1,
  kInt2TfliteRowMajor = 2,
};

// TensorRT-LLM's ColumnMajorInterleaved W4A16 layout is shared by the
// upstream SM80 and SM120 kernels. Hopper and datacenter Blackwell use
// different layouts, so they must fall back to the ordinary TensorRT engine.
inline constexpr bool IsTensorRtLlmHeadComputeCapabilitySupported(
    int compute_capability) {
  return (compute_capability >= 80 && compute_capability < 90) ||
         compute_capability >= 120;
}

// The local row-major W2 kernel carries compute_80 PTX and does not depend on
// TensorRT-LLM's architecture-specific interleaved weight layout.
inline constexpr bool IsInt2GemvComputeCapabilitySupported(
    int compute_capability) {
  return compute_capability >= 80;
}

// Optional TensorRT-LLM weight-only LM-head payload. The pointers alias the
// enclosing bytecode buffer, just like TensorRtBytecode::engine_data, and
// remain valid only while that buffer is alive. BF16 scales are serialized as
// little-endian uint16 bit patterns and exposed as bytes to avoid imposing an
// alignment requirement on the bytecode buffer.
struct TensorRtLlmHead {
  uint32_t hidden_output_port = 0;
  uint32_t logits_output_port = 0;
  uint32_t k = 0;
  uint32_t n = 0;
  float soft_cap = 0.0f;
  TensorRtLlmHeadWeightFormat weight_format =
      TensorRtLlmHeadWeightFormat::kInvalid;
  const uint8_t* packed_weights = nullptr;
  size_t packed_weights_size = 0;
  const uint8_t* bf16_scales = nullptr;
  size_t bf16_scales_size = 0;
};

// TensorRT weight types are kept independent of NvInfer headers so the
// bytecode parser remains usable by LiteRT tooling that does not link the
// TensorRT SDK. The numeric values intentionally match nvinfer1::DataType.
enum class TensorRtWeightDataType : int32_t {
  kFloat = 0,
  kHalf = 1,
  kInt8 = 2,
  kInt32 = 3,
  kBool = 4,
  kUint8 = 5,
  kFp8 = 6,
  kBf16 = 7,
  kInt64 = 8,
  kInt4 = 9,
  kFp4 = 10,
  kE8m0 = 11,
};

// A view of one named weight required to refit a stripped TensorRT plan. The
// data aliases the enclosing bytecode buffer and remains valid only while the
// buffer is alive.
struct TensorRtRefitWeight {
  std::string name;
  TensorRtWeightDataType data_type = TensorRtWeightDataType::kFloat;
  uint64_t count = 0;
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// Owning input used when packing the shared store. Multiple engine entries
// can reference the same element by index.
struct TensorRtSharedWeight {
  TensorRtWeightDataType data_type = TensorRtWeightDataType::kFloat;
  uint64_t count = 0;
  std::vector<uint8_t> data;
};

struct TensorRtSharedWeightRef {
  std::string name;
  uint32_t shared_weight_index = 0;
};

// Non-owning engine view used only for PackTensorRtSharedWeightBundle().
struct TensorRtBundleEntry {
  std::string function_name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  const void* engine_data = nullptr;
  size_t engine_size = 0;
  const TensorRtLlmHead* trtllm_head = nullptr;
  std::vector<TensorRtSharedWeightRef> refit_weights;
};

struct TensorRtBytecode {
  uint32_t version = 0;
  std::string function_name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  const uint8_t* engine_data = nullptr;
  size_t engine_size = 0;
  std::optional<TensorRtLlmHead> trtllm_head;
  std::vector<TensorRtRefitWeight> refit_weights;
};

Expected<std::vector<uint8_t>> PackTensorRtBytecode(
    const std::string& function_name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const void* engine_data,
    size_t engine_size, const TensorRtLlmHead* trtllm_head = nullptr);

// Packs several stripped TensorRT plans and one deduplicated weight store into
// a single bytecode module. LiteRT dispatch calls select an engine by their
// call-info function name while sharing the same serialized module.
Expected<std::vector<uint8_t>> PackTensorRtSharedWeightBundle(
    const std::vector<TensorRtSharedWeight>& shared_weights,
    const std::vector<TensorRtBundleEntry>& entries);

// Packs one engine with only the shared weights referenced by that engine.
// The references are remapped to the shard-local store without copying the
// source weights before serialization. Persistent AOT uses this form so each
// dispatch context maps only the bytes it can consume.
Expected<std::vector<uint8_t>> PackTensorRtSharedWeightShard(
    const std::vector<TensorRtSharedWeight>& shared_weights,
    const TensorRtBundleEntry& entry);

// Returns a stable, non-cryptographic 128-bit content fingerprint. The AOT
// writer additionally compares existing artifacts byte-for-byte before reuse;
// dispatch uses the fingerprint to detect accidental corruption.
TensorRtArtifactFingerprint FingerprintTensorRtArtifact(const void* data,
                                                        size_t size);

// Locator v2 uses a chunk-composable fingerprint so dispatch can validate a
// file through a small streaming buffer rather than faulting its entire mmap
// into process RSS. Call Add() with consecutive chunks and preserve boundaries
// when reproducing a fingerprint.
class TensorRtAotFingerprintBuilder {
 public:
  void Add(const void* data, size_t size);
  TensorRtArtifactFingerprint Finish() const;

 private:
  TensorRtArtifactFingerprint fingerprint_;
  uint64_t total_size_ = 0;
  uint64_t chunk_count_ = 0;
};

TensorRtArtifactFingerprint FingerprintTensorRtAotArtifact(const void* data,
                                                           size_t size);

Expected<std::vector<uint8_t>> PackTensorRtAotLocator(
    const TensorRtAotLocator& locator);

// Returns nullopt when data is ordinary TensorRT bytecode. Once the AOT magic
// is present, malformed locator data is reported as an error rather than being
// reinterpreted as engine bytecode.
Expected<std::optional<TensorRtAotLocator>> TryParseTensorRtAotLocator(
    const void* data, size_t size);

Expected<std::vector<uint8_t>> PackTensorRtAotManifest(
    const TensorRtAotManifest& manifest);
Expected<TensorRtAotManifest> ParseTensorRtAotManifest(const void* data,
                                                       size_t size);

// Legacy bytecodes contain one engine and ignore function_name. Version 4
// bundles require function_name when they contain more than one engine.
Expected<TensorRtBytecode> ParseTensorRtBytecode(
    const void* data, size_t size, const char* function_name = nullptr);

}  // namespace litert::nvidia

#endif  // ODML_LITERT_LITERT_VENDORS_NVIDIA_BYTECODE_H_
