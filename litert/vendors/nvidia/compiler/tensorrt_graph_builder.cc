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

#include "litert/vendors/nvidia/compiler/tensorrt_graph_builder.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/litert_op_options.h"
#include "litert/cc/internal/litert_op_options.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/nvidia/bytecode.h"
#include "litert/vendors/nvidia/cache_layout.h"
#include "litert/vendors/nvidia/compiler/cache_update_plugin.h"
#include "litert/vendors/nvidia/compiler/decode_attention_plugin.h"
#include "litert/vendors/nvidia/compiler/gated_delta_plugin.h"
#include "litert/vendors/nvidia/compiler/general_attention_plugin.h"
#include "litert/vendors/nvidia/compiler/int8_cache_write_plugin.h"
#include "litert/vendors/nvidia/compiler/moe_plugin.h"
#include "litert/vendors/nvidia/compiler/qkv_norm_rope_plugin.h"
#include "litert/vendors/nvidia/compiler/qsa_attention_plugin.h"
#include "litert/vendors/nvidia/compiler/quant_attention_plugin.h"
#include "litert/vendors/nvidia/compiler/quant_linear_plugin.h"
#include "litert/vendors/nvidia/compiler/slot_network.h"
#include "litert/vendors/nvidia/compiler/subbyte_gemv_plugin.h"
#include "litert/vendors/nvidia/compiler/tensorrt_rtx_plugin_compat.h"
#include "litert/vendors/nvidia/compiler/weight_store_builder.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel_lowering.h"
#include "litert/vendors/nvidia/experimental/megakernel/megakernel_plugin.h"
#include "litert/vendors/nvidia/memory_profile.h"
#include "litert/vendors/nvidia/tensorrt_logger.h"
#include "NvInfer.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/gated_delta.h"
#include "litert/vendors/nvidia/trtllm/gated_delta_net.h"
#include "litert/vendors/nvidia/trtllm/general_attention.h"
#include "litert/vendors/nvidia/trtllm/global_attention.h"
#include "litert/vendors/nvidia/trtllm/int2_gemv.h"
#include "litert/vendors/nvidia/trtllm/int4_block_gemv.h"
#include "litert/vendors/nvidia/trtllm/qkv_norm_rope.h"
#include "litert/vendors/nvidia/trtllm/qsa_indexer.h"
#include "litert/vendors/nvidia/trtllm/qsa_sparse_attention.h"
#include "litert/vendors/nvidia/trtllm/quant_linear.h"
#include "litert/vendors/nvidia/trtllm/subbyte_gemm.h"
#include "litert/vendors/nvidia/trtllm/tiled_attention.h"

namespace litert::nvidia {
namespace {

using ::litert::compiler::Op;
using ::litert::compiler::Subgraph;
using ::litert::compiler::Tensor;

// The budget for a partition's intermediate buffers. Myelin compiles a whole
// partition as one fused region, so this must cover the full activation arena
// of the largest island (multi-layer prefill attention islands need >1GB); it
// is a cap on tactic choice, not an upfront allocation.
constexpr size_t kDefaultTensorRtWorkspaceBytes = 4096ULL << 20;
constexpr size_t kDefaultMaxTensorRtFillBytes = 16ULL << 20;
// Float FC weights beyond this stay on the CPU; quantized weights are not
// capped (see IsFullyConnectedSupported).
constexpr size_t kDefaultMaxTensorRtFullyConnectedWeightBytes = 256ULL << 20;

template <typename T>
using TrtPtr = std::unique_ptr<T>;

size_t EnvSizeT(const char* name, size_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
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

// The type of the activations inside an island
// (LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS). BF16 by default ("bf16"), the
// type the Gemma 4 and Qwen 3.5 exports are validated in; "0" keeps FP32
// activations, and any other value runs them in FP16.
enum class ActivationMode { kFloat32, kFp16, kBf16 };

ActivationMode TensorRtActivationMode() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "bf16") == 0) {
    return ActivationMode::kBf16;
  }
  return std::strcmp(value, "0") == 0 ? ActivationMode::kFloat32
                                      : ActivationMode::kFp16;
}

// FP16-activation mode (FP16 or BF16): activations flow through the network
// as 16-bit real values; int8 activation quantization survives only at island
// boundaries (one dequantize per input, one quantize per output) and weights
// keep their weight-only-quantized form inside the matmuls. This collapses the
// per-op Cast/Mul Q-DQ chains that otherwise fragment Myelin fusion into
// thousands of kernels per step.
bool Fp16ActivationsEnabled() {
  return TensorRtActivationMode() != ActivationMode::kFloat32;
}

// How the products over integer weights run
// (LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS). Materializing dequantized
// weight constants removes the per-step Cast/Mul/Move weight-dequantize
// kernels that Myelin otherwise runs on every invocation (it has no fused
// sub-byte weight-only GEMV at M=1).
//   kFloat: fp16/bf16 constants (2 bytes/element).
//   kFp8:   fp8-e4m3 value constants + per-channel scale dequantize, which
//           Myelin fuses into a single GEMV that reads 1 byte/element.
//           int4/int2 integer values are exactly representable in e4m3.
// kCudaGemv, the default ("cuda_gemv"), keeps packed INT2/INT4 weights and
// performs dequantization and GEMV in one native CUDA plugin without
// materializing expanded weights; "0" leaves the products to TensorRT.
enum class PredequantMode { kOff, kFloat, kFp8, kCudaGemv };

PredequantMode PredequantizeFcWeightsMode() {
  const char* value =
      std::getenv("LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS");
  if (value == nullptr || value[0] == '\0' ||
      std::strcmp(value, "cuda_gemv") == 0) {
    return PredequantMode::kCudaGemv;
  }
  if (std::strcmp(value, "0") == 0) {
    return PredequantMode::kOff;
  }
  if (std::strcmp(value, "fp8") == 0) {
    return PredequantMode::kFp8;
  }
  return PredequantMode::kFloat;
}

size_t TensorRtWorkspaceBytes() {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_WORKSPACE_MB",
                  kDefaultTensorRtWorkspaceBytes >> 20)
         << 20;
}

size_t MaxTensorRtFillBytes() {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_MAX_FILL_BYTES",
                  kDefaultMaxTensorRtFillBytes);
}

size_t MaxTensorRtFullyConnectedWeightBytes() {
  return EnvSizeT("LITERT_NVIDIA_TENSORRT_MAX_FC_WEIGHT_BYTES",
                  kDefaultMaxTensorRtFullyConnectedWeightBytes);
}

// Block sizes tried, largest first, when re-encoding per-channel INT4 FC
// scales as TensorRT block scales along K. Larger blocks mean smaller scale
// constants; TensorRT requires the block to divide K and be smaller than K.
constexpr int32_t kFcBlockScaleSizes[] = {1280, 1024, 512, 256, 128, 64, 32};

bool FcBlockScalesEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_FC_BLOCK_SCALES",
                    /*default_value=*/true);
}

bool FuseGemvGroupsEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_FUSE_GEMV_GROUPS",
                    /*default_value=*/true);
}

// Whether the GEMMs over blockwise INT4 weights run on the int8 tensor cores
// (trtllm/subbyte_gemm_int8.cu), as the per-channel ones do: in a graph with
// Gated DeltaNet layers (Qwen 3.5's hybrid stack), whose greedy outputs it
// keeps, while Gemma's blockwise models stay on the fp16 kernel.
// LITERT_NVIDIA_TENSORRT_INT8_BLOCKWISE_GEMM says otherwise, and so does a
// partition policy that is set: qwen3_5 turns it on, any other off.
bool Int8BlockwiseGemmRequested(bool graph_has_gated_delta_net) {
  if (const char* value =
          std::getenv("LITERT_NVIDIA_TENSORRT_INT8_BLOCKWISE_GEMM");
      value != nullptr && value[0] != '\0') {
    return std::strcmp(value, "0") != 0;
  }
  if (const char* policy =
          std::getenv("LITERT_NVIDIA_TENSORRT_PARTITION_POLICY");
      policy != nullptr && policy[0] != '\0') {
    return std::strcmp(policy, "qwen3_5") == 0;
  }
  return graph_has_gated_delta_net;
}

bool NativeKvCacheUpdateEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_NATIVE_KV_CACHE_UPDATE",
                    /*default_value=*/true);
}

// Experimental: decode attention blocks (q @ K^T -> select(mask) -> softmax
// -> probs @ V over in-place KV caches) fused into one CUDA plugin launch
// that reads the cache update outputs directly, which spares the whole-cache
// copies Myelin makes for its own consumers (~75 us per 35 MB global cache).
// On by default since the kernel's chunked split (trtllm/decode_attention.cu):
// a Gemma 4 12B local block (8 KV heads, 2 rows, a 1,152-key ring) runs in
// about 12 us against the 19 us of Myelin's matmul / transpose / softmax
// chain, 48 of 48 blocks fuse and the 12B decodes 104 against 102 steps/s at
// 2k, 96.3 against 95.5 at 30k. Set
// LITERT_NVIDIA_TENSORRT_DECODE_ATTENTION_PLUGIN=0 to keep Myelin's lowering.
bool DecodeAttentionPluginEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_DECODE_ATTENTION_PLUGIN",
                    /*default_value=*/true);
}

// Decode attention over INT8 per-tensor KV caches (Gemma 4 E4B) runs on the
// same CUDA plugin, which dequantizes the caches as it reads them. On by
// default: Myelin fuses the dequantize, both products (M = the query heads
// of a KV head, a GEMM) and the softmax into its `_gemm_mha_v2` kernel,
// which takes 180 us per layer on the RTX 5080 where the plugin takes about
// 25 us. Set LITERT_NVIDIA_TENSORRT_INT8_CACHE_ATTENTION_PLUGIN=0 to keep
// Myelin's lowering.
// Longest cache at which single-KV-head attention stays on TensorRT's own
// lowering (see MatchInt8CacheAttentionBlock): its cost grows about 2 us per
// 1k keys per layer on E2B, the plugin's floor is about 24 us, so the plugin
// wins from the 16k preset on (E2B at the 8k preset: 270 steps/s on TensorRT,
// 255 on the plugin).
constexpr int64_t kSingleHeadNativeCacheLimit = 16384;

// Sliding-window attention blocks read only the keys their window can reach
// (see FindWindowedCacheReads).
bool WindowedCacheReadsEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_WINDOWED_CACHE_READS",
                    /*default_value=*/true);
}

bool Int8CacheAttentionPluginEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_INT8_CACHE_ATTENTION_PLUGIN",
                    /*default_value=*/true);
}

// INT8 cache updates through the cache write plugin (the step's rows
// quantized and written in place) instead of a scatter that copies the
// whole cache every step.
bool Int8CacheWriteEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_INT8_CACHE_WRITE",
                    /*default_value=*/true);
}

// Attention over a single-head cache of depth 512 (Gemma 4 global layers, in
// decode and in prefill) runs on the tensor-core kernels of
// trtllm/global_attention.h through the same plugin. Unlike the matmul /
// select / softmax / matmul lowering they only read the keys some query row
// can see, never materialize a [rows, context] tensor (so the activation
// arena stays small at long context), and read the in-place cache updates
// without the whole-cache copies Myelin makes for its own consumers. Set
// LITERT_NVIDIA_TENSORRT_GLOBAL_ATTENTION_PLUGIN=0 to keep the TensorRT
// lowering.
bool GlobalAttentionPluginEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_GLOBAL_ATTENTION_PLUGIN",
                    /*default_value=*/true);
}

// Prefill attention over a ring cache (Gemma 4 local layers) reads the cache
// before its update and the keys and values of the chunk itself:
// concatenated scores, select, softmax, and the sum of two value products. It
// runs on the tensor-core kernels of trtllm/tiled_attention.h as one launch
// of the same plugin, which only computes the tiles of keys some query row of
// a block can see (about half of them in a sliding window). Set
// LITERT_NVIDIA_TENSORRT_LOCAL_ATTENTION_PLUGIN=0 to keep the TensorRT
// lowering.
bool LocalAttentionPluginEnabled() {
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_LOCAL_ATTENTION_PLUGIN",
                    /*default_value=*/true);
}

// With the CUDA subbyte GEMV, fully connected layers over many activation
// rows (prefill) with INT4 weights run as a tensor-core GEMM in the same
// plugin (trtllm/subbyte_gemm.h) when they are large enough to keep the
// device busy. It accumulates in FP16 over 128 input dims at a time and in
// FP32 beyond, at one and a half times the throughput of TensorRT's
// FP32-accumulating matmul (Gemma 4 12B, 1024 rows on an RTX 5080: 175
// against 110 TFLOP/s), and takes the gate and up projections of a
// feed-forward block as one launch that also applies the GELU gate.
// LITERT_NVIDIA_TENSORRT_CUDA_GEMM=gated limits it to those pairs; =0 keeps
// the TensorRT lowering for every layer.
enum class CudaGemmMode { kOff, kGated, kAll };

CudaGemmMode GetCudaGemmMode() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_CUDA_GEMM");
  if (value == nullptr || value[0] == '\0') {
    return CudaGemmMode::kAll;
  }
  if (std::strcmp(value, "gated") == 0) {
    return CudaGemmMode::kGated;
  }
  return std::strcmp(value, "0") == 0 ? CudaGemmMode::kOff : CudaGemmMode::kAll;
}

// The tensor-core kernels skip keys that no query row can see, which is exact
// as long as a masked score underflows in the softmax.
constexpr float kMaxGlobalAttentionFill = -1000.0f;

// runtime_bmm multiplies activations against FP16 KV caches. By default the
// small activation operand is cast to the cache type, the product runs in
// the cache precision (the model's own), and the result is cast to the
// activation mode type (BF16). Besides avoiding a full-cache cast, the casts
// around the score/softmax chain keep Myelin from fusing it into its
// `_gemm_mha_v2` kernel, which is several times slower than plain matmul and
// softmax kernels for the decode shapes (probes/probe_layout.cc: local block
// 73 -> 19 us) and also slower for the prefill shapes. Set to "activation"
// to cast the cache to the activation type and keep the fused kernel.
bool RuntimeBmmInCachePrecision() {
  const char* value =
      std::getenv("LITERT_NVIDIA_TENSORRT_RUNTIME_BMM_PRECISION");
  return value == nullptr || value[0] == '\0' ||
         std::strcmp(value, "activation") != 0;
}

// Value caches the model stores as [B, H, D, S] and updates in place are held
// as [B, H, S, D] inside the engines (the same bytes; every reader and writer
// of those buffers is lowered here) unless this is set to "native". Myelin
// otherwise transposes every value cache on every invocation. Holding the
// keys transposed as well measured slower (the fused attention kernels want
// keys as [B, H, S, D]). Engines built with different settings must not
// share a cache buffer, and a signature left on the CPU must not touch these
// caches.
bool TransposedValueCacheEnabled() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_VALUE_CACHE_LAYOUT");
  return value == nullptr || value[0] == '\0' ||
         std::strcmp(value, "native") != 0;
}

bool LogUnsupportedOpDetails() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_LOG_UNSUPPORTED_OPS");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void LogFullyConnectedReject(const Op& op, const char* reason) {
  if (!LogUnsupportedOpDetails()) {
    return;
  }
  auto summarize_tensor = [](const Tensor& tensor) {
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
  };
  std::string detail = std::string(reason) + " inputs={";
  for (int i = 0; i < op.Inputs().size(); ++i) {
    if (i > 0) {
      detail += "; ";
    }
    detail += summarize_tensor(op.Inputs()[i]);
  }
  detail += "} outputs={";
  for (int i = 0; i < op.Outputs().size(); ++i) {
    if (i > 0) {
      detail += "; ";
    }
    detail += summarize_tensor(op.Outputs()[i]);
  }
  detail += "}";
  LITERT_LOG(LITERT_INFO, "NVIDIA TensorRT-RTX FC unsupported: %s",
             detail.c_str());
}

Expected<nvinfer1::DataType> ConvertDataType(litert::ElementType type) {
  switch (type) {
    case litert::ElementType::Float32:
      return nvinfer1::DataType::kFLOAT;
    case litert::ElementType::Float16:
      return nvinfer1::DataType::kHALF;
    case litert::ElementType::BFloat16:
      return nvinfer1::DataType::kBF16;
    case litert::ElementType::Float8E4M3FN:
      return nvinfer1::DataType::kFP8;
    case litert::ElementType::Int4:
      return nvinfer1::DataType::kINT4;
    case litert::ElementType::Int8:
      return nvinfer1::DataType::kINT8;
    case litert::ElementType::Int32:
      return nvinfer1::DataType::kINT32;
    case litert::ElementType::Int64:
      return nvinfer1::DataType::kINT64;
    case litert::ElementType::Bool:
      return nvinfer1::DataType::kBOOL;
    default:
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported TensorRT tensor element type: " +
                       std::to_string(static_cast<int>(type)));
  }
}

bool IsFloatLike(litert::ElementType type) {
  return type == litert::ElementType::Float32 ||
         type == litert::ElementType::Float16 ||
         type == litert::ElementType::BFloat16;
}

// Integer types of positions and indices; their float casts stay exact.
bool IsPositionElementType(litert::ElementType type) {
  return type == litert::ElementType::Int32 ||
         type == litert::ElementType::Int64;
}

bool IsTensorRtElementType(litert::ElementType type) {
  return IsFloatLike(type) || type == litert::ElementType::Float8E4M3FN ||
         type == litert::ElementType::Int4 ||
         type == litert::ElementType::Int8 ||
         type == litert::ElementType::Int32 ||
         type == litert::ElementType::Int64 ||
         type == litert::ElementType::Bool;
}

bool IsTensorRtArithmeticElementType(litert::ElementType type) {
  return IsFloatLike(type) || type == litert::ElementType::Int8 ||
         type == litert::ElementType::Int32 ||
         type == litert::ElementType::Int64;
}

Expected<nvinfer1::Dims> ConvertDims(const litert::RankedTensorType& type) {
  const auto& layout = type.Layout();
  if (layout.HasStrides()) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "TensorRT backend does not support strided LiteRT tensors");
  }
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int32_t>(layout.Rank());
  if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
    return Error(kLiteRtStatusErrorUnsupported, "Tensor rank is too large");
  }
  for (int i = 0; i < dims.nbDims; ++i) {
    const int32_t dim = layout.Dimensions()[i];
    if (dim < 0) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Dynamic shapes are not supported yet");
    }
    dims.d[i] = dim;
  }
  return dims;
}

Expected<size_t> NumElements(const litert::RankedTensorType& type) {
  return type.Layout().NumElements();
}

Expected<size_t> ElementByteSize(litert::ElementType type) {
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
    case litert::ElementType::Int64:
      return 8;
    default:
      return Error(kLiteRtStatusErrorUnsupported,
                   "Element type does not have a supported byte width");
  }
}

Expected<bool> IsStaticRankedTensor(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
  if (type.Layout().HasStrides()) {
    return false;
  }
  for (auto dim : type.Layout().Dimensions()) {
    if (dim < 0) {
      return false;
    }
  }
  return true;
}

Expected<bool> IsStaticTensorRtTensor(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(bool static_ranked, IsStaticRankedTensor(tensor));
  if (!static_ranked) {
    return false;
  }
  return IsTensorRtElementType(tensor.ElementType());
}

Expected<bool> IsStaticFloatTensor(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(bool static_ranked, IsStaticRankedTensor(tensor));
  if (!static_ranked) {
    return false;
  }
  return IsFloatLike(tensor.ElementType());
}

bool SameElementType(const Tensor& lhs, const Tensor& rhs) {
  return lhs.ElementType() == rhs.ElementType();
}

bool HasRuntimeInput(const Op& op) {
  for (const auto& input : op.Inputs()) {
    if (!input.HasWeights()) {
      return true;
    }
  }
  return false;
}

Expected<size_t> StaticTensorByteSize(const Tensor& tensor) {
  LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
  if (type.ElementType() == litert::ElementType::Int4) {
    return (num_elements + 1) / 2;
  }
  LITERT_ASSIGN_OR_RETURN(size_t element_size,
                          ElementByteSize(type.ElementType()));
  return num_elements * element_size;
}

bool IsTensorRtQuantizedElementType(litert::ElementType type) {
  return type == litert::ElementType::Int4 ||
         type == litert::ElementType::Int8 ||
         type == litert::ElementType::Float8E4M3FN;
}

Expected<bool> HasSymmetricTensorRtQuantization(const Tensor& tensor,
                                                bool allow_per_channel);
bool IsBlockwiseInt4Weights(const Tensor& tensor);
bool BlockwiseFcPathEnabled();

bool IsSupportedFusedActivation(uint32_t activation) {
  using Opt = litert::ActivationFunctionType;
  return activation == Opt::kActivationFunctionTypeNone ||
         activation == Opt::kActivationFunctionTypeRelu ||
         activation == Opt::kActivationFunctionTypeRelu6 ||
         activation == Opt::kActivationFunctionTypeReluN1To1;
}

Expected<uint32_t> GetFusedActivation(const Op& op) {
  uint32_t activation = litert::kActivationFunctionTypeNone;
  LiteRtStatus status = kLiteRtStatusOk;
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
      status = LiteRtGetAddFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflMul:
      status = LiteRtGetMulFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflSub:
      status = LiteRtGetSubFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflDiv:
      status = LiteRtGetDivFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflFullyConnected:
      status =
          LiteRtGetFullyConnectedFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflConv2d:
      status = LiteRtGetConv2dFusedActivationOption(op.Get(), &activation);
      break;
    case kLiteRtOpCodeTflConcatenation:
      status =
          LiteRtGetConcatenationFusedActivationOption(op.Get(), &activation);
      break;
    default:
      return activation;
  }
  if (status != kLiteRtStatusOk) {
    return Error(status, "Failed to read fused activation option");
  }
  if (!IsSupportedFusedActivation(activation)) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Fused activation is not supported");
  }
  return activation;
}

Expected<bool> HasSupportedFusedActivation(const Op& op) {
  auto activation = GetFusedActivation(op);
  return activation.HasValue();
}

Expected<std::vector<int32_t>> ReadInt32Constant(const Tensor& tensor) {
  if (!tensor.HasWeights() ||
      tensor.ElementType() != litert::ElementType::Int32) {
    return Error(kLiteRtStatusErrorUnsupported,
                 "Expected constant int32 tensor");
  }
  LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
  auto bytes = tensor.Weights().Bytes();
  if (bytes.size() != num_elements * sizeof(int32_t)) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Unexpected int32 constant size");
  }
  std::vector<int32_t> values(num_elements);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

bool IsElementwiseOp(LiteRtOpCode code) {
  return code == kLiteRtOpCodeTflAdd || code == kLiteRtOpCodeTflMul ||
         code == kLiteRtOpCodeTflSub || code == kLiteRtOpCodeTflDiv ||
         code == kLiteRtOpCodeTflMaximum || code == kLiteRtOpCodeTflMinimum;
}

Expected<bool> HasSymmetricPerTensorInt8Quantization(const Tensor& tensor) {
  if (tensor.ElementType() != litert::ElementType::Int8 ||
      tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
    return false;
  }
  return HasSymmetricTensorRtQuantization(tensor,
                                          /*allow_per_channel=*/false);
}

Expected<bool> IsElementwiseSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (!HasRuntimeInput(op)) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticTensorRtTensor(tensor));
    if (!supported || !IsTensorRtArithmeticElementType(tensor.ElementType())) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!output_supported ||
      !IsTensorRtArithmeticElementType(op.Outputs()[0].ElementType())) {
    return false;
  }
  if (!SameElementType(op.Inputs()[0], op.Inputs()[1]) ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  if (op.Code() == kLiteRtOpCodeTflMaximum ||
      op.Code() == kLiteRtOpCodeTflMinimum) {
    return true;
  }
  if (op.Inputs()[0].ElementType() == litert::ElementType::Int8) {
    if (EnvEnabled("LITERT_NVIDIA_TENSORRT_DISABLE_INT8_ELEMENTWISE",
                   /*default_value=*/false)) {
      return false;
    }
    // Symmetric per-tensor int8 elementwise is lowered as DQ -> op -> Q.
    for (const auto& tensor :
         {op.Inputs()[0], op.Inputs()[1], op.Outputs()[0]}) {
      LITERT_ASSIGN_OR_RETURN(bool symmetric,
                              HasSymmetricPerTensorInt8Quantization(tensor));
      if (!symmetric) {
        return false;
      }
    }
    return HasSupportedFusedActivation(op);
  }
  if (op.Inputs()[0].ElementType() == litert::ElementType::Int32 ||
      op.Inputs()[0].ElementType() == litert::ElementType::Int64) {
    // Integer index arithmetic (e.g. cache positions) with no activation.
    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    return activation == litert::kActivationFunctionTypeNone;
  }
  if (!IsFloatLike(op.Inputs()[0].ElementType())) {
    return false;
  }
  return HasSupportedFusedActivation(op);
}

Expected<bool> IsComparisonSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (!HasRuntimeInput(op)) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticTensorRtTensor(tensor));
    if (!supported || !IsTensorRtArithmeticElementType(tensor.ElementType())) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  return output_supported &&
         op.Outputs()[0].ElementType() == litert::ElementType::Bool &&
         SameElementType(op.Inputs()[0], op.Inputs()[1]);
}

Expected<bool> IsLogicalAndSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (!HasRuntimeInput(op)) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticTensorRtTensor(tensor));
    if (!supported || tensor.ElementType() != litert::ElementType::Bool) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  return output_supported &&
         op.Outputs()[0].ElementType() == litert::ElementType::Bool;
}

Expected<bool> IsUnaryActivationSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticFloatTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  return input_supported && output_supported;
}

Expected<bool> IsGeluSupported(const Op& op) {
  LITERT_ASSIGN_OR_RETURN(bool supported, IsUnaryActivationSupported(op));
  bool approximate = false;
  return supported && LiteRtGetGeluApproximateOption(op.Get(), &approximate) ==
                          kLiteRtStatusOk;
}

Expected<bool> IsSoftmaxSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  float beta = 1.0f;
  if (LiteRtGetSoftmaxBetaOption(op.Get(), &beta) != kLiteRtStatusOk ||
      beta != 1.0f) {
    return false;
  }
  return IsUnaryActivationSupported(op);
}

Expected<bool> IsReshapeSupported(const Op& op) {
  if (op.Inputs().size() < 1 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported) {
    return false;
  }
  if (!SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  // The shape operand (constant or runtime-computed) is ignored: the static
  // output shape fully determines the reshape.
  return true;
}

Expected<bool> IsCastSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  // FP16 cache-update decompositions cast an INT64 constant to FP16.
  // GetTensor/LowerCast can materialize this directly; rejecting it creates
  // a CPU boundary between otherwise supported cache operations.
  if (op.Inputs()[0].HasWeights() &&
      !(op.Inputs()[0].ElementType() == litert::ElementType::Int64 &&
        op.Outputs()[0].ElementType() == litert::ElementType::Float16)) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  return input_supported && output_supported;
}

Expected<bool> IsTransposeSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  auto perm = ReadInt32Constant(op.Inputs()[1]);
  return perm.HasValue();
}

Expected<bool> IsSliceSupported(const Op& op) {
  if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const size_t rank = input_type.Layout().Rank();
  // The size operand is ignored: the static output shape determines it. The
  // begin operand may be a constant or a runtime int32 vector of length rank.
  auto begin = ReadInt32Constant(op.Inputs()[1]);
  if (begin) {
    return begin->size() == rank;
  }
  if (op.Inputs()[1].HasWeights() ||
      op.Inputs()[1].ElementType() != litert::ElementType::Int32) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto begin_type, op.Inputs()[1].RankedTensorType());
  if (begin_type.Layout().Rank() != 1 ||
      begin_type.Layout().Dimensions()[0] != static_cast<int32_t>(rank)) {
    return false;
  }
  // Runtime offsets lower to a gather along the sliced axis; only one axis
  // may shrink (full-extent axes must have offset zero by slice semantics).
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  int shrinking_axes = 0;
  for (size_t i = 0; i < rank; ++i) {
    if (output_type.Layout().Dimensions()[i] !=
        input_type.Layout().Dimensions()[i]) {
      ++shrinking_axes;
    }
  }
  return shrinking_axes <= 1;
}

// Composite ops this backend can lower natively (instead of letting the
// runtime inline their decompositions, whose reshape/index glue fragments
// Myelin fusion into thousands of tiny kernels per step).
std::string CompositeOpName(const Op& op) {
  const char* name = nullptr;
  if (LiteRtGetSHLOCompositeOpName(op.Get(), &name) != kLiteRtStatusOk ||
      name == nullptr) {
    return "";
  }
  return name;
}

// Default: all supported composites lower natively. Set
// LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES to "none" (or "0") to inline
// everything, or to a comma list of short names (rms_norm,cache_update,
// runtime_bmm) to enable a subset.
bool NativeCompositeEnabled(const std::string& full_name) {
  const std::string short_name =
      full_name.rfind("odml.", 0) == 0 ? full_name.substr(5) : full_name;
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_NATIVE_COMPOSITES");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "none") == 0) {
    return false;
  }
  const std::string list(value);
  size_t begin = 0;
  while (begin < list.size()) {
    size_t end = list.find(',', begin);
    if (end == std::string::npos) {
      end = list.size();
    }
    if (list.compare(begin, end - begin, short_name) == 0) {
      return true;
    }
    begin = end + 1;
  }
  return false;
}

// Gemma 4 26B-A4B declares its KV caches as float32. Only our engines touch
// a cache, so an in-place cache declared float32 is held as FP16 in the
// (larger) buffer the runtime binds: half the attention traffic and the FP16
// cache kernels. LITERT_NVIDIA_TENSORRT_CACHE_STORAGE=native keeps float32.
bool Fp16CacheStorageEnabled() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_CACHE_STORAGE");
  return value == nullptr || value[0] == '\0' ||
         std::strcmp(value, "native") != 0;
}

bool IsFp16StoredCacheType(litert::ElementType type) {
  return type == litert::ElementType::Float16 ||
         (type == litert::ElementType::Float32 && Fp16CacheStorageEnabled());
}

// A cache update whose results attention reads in the same graph (decode,
// and the global layers of prefill): float updates [1, H, n, D] into float
// caches k [1, H, S, D] and v [1, H, D, S] that are subgraph inputs, at the
// start indices of inputs 5 and 6 (the ring position is already reduced in
// the graph), the updated caches being subgraph outputs. Returns n.
struct DecodeCacheUpdateLayout {
  int32_t heads = 0;
  int32_t rows = 0;  // n
  int32_t seq = 0;
  int32_t depth = 0;
};

std::optional<DecodeCacheUpdateLayout> DecodeCacheUpdateMode(const Op& op) {
  if (!NativeKvCacheUpdateEnabled() || op.Inputs().size() != 7 ||
      op.Outputs().size() != 2 || op.Outputs()[0].Uses().empty() ||
      op.Outputs()[1].Uses().empty()) {
    return std::nullopt;
  }
  std::array<std::array<int32_t, 4>, 5> dims;
  for (int i = 0; i < 5; ++i) {
    auto type = op.Inputs()[i].RankedTensorType();
    if (!type || type->Layout().Rank() != 4) {
      return std::nullopt;
    }
    const auto element_type = op.Inputs()[i].ElementType();
    if (i == 2 ? element_type != litert::ElementType::Int32
               : (i < 2 ? !IsFloatLike(element_type)
                        : !IsFp16StoredCacheType(element_type))) {
      return std::nullopt;
    }
    const auto shape = type->Layout().Dimensions();
    std::copy(shape.begin(), shape.end(), dims[i].begin());
    for (int d : dims[i]) {
      if (d <= 0) return std::nullopt;
    }
  }
  for (int i : {5, 6}) {
    auto type = op.Inputs()[i].RankedTensorType();
    if (!type || type->Layout().Rank() != 1 ||
        type->Layout().Dimensions()[0] != 4 ||
        op.Inputs()[i].ElementType() != litert::ElementType::Int32) {
      return std::nullopt;
    }
  }
  const auto& k = dims[3];
  const auto& v = dims[4];
  if (k[0] != 1 || v[0] != 1 || v[1] != k[1] || v[2] != k[3] || v[3] != k[2] ||
      dims[2][0] != 1 || dims[2][1] != 1 || dims[2][2] != 1 ||
      dims[2][3] != 7 || !op.Inputs()[3].IsSubgraphInput() ||
      !op.Inputs()[4].IsSubgraphInput()) {
    return std::nullopt;
  }
  for (int i = 0; i < 2; ++i) {
    if (dims[i][0] != 1 || dims[i][1] != k[1] || dims[i][2] < 1 ||
        dims[i][2] > k[2] || dims[i][3] != k[3] ||
        !std::equal(dims[0].begin(), dims[0].end(), dims[1].begin())) {
      return std::nullopt;
    }
    auto output_type = op.Outputs()[i].RankedTensorType();
    if (!output_type || output_type->Layout().Rank() != 4 ||
        output_type->ElementType() != op.Inputs()[3 + i].ElementType() ||
        !std::equal(dims[3 + i].begin(), dims[3 + i].end(),
                    output_type->Layout().Dimensions().begin())) {
      return std::nullopt;
    }
  }
  return DecodeCacheUpdateLayout{k[1], dims[0][2], k[2], k[3]};
}

bool TensorHasShape(const Tensor& tensor,
                    std::initializer_list<int32_t> expected);
bool IsUnquantizedFloat32(const Tensor& tensor);

// odml.qkv_norm_rope(qkv [1, n, (H + 2 Hkv) D] float, position int32 (n
// values), q_scale [D], k_scale [D]) -> q [1, H, n, D], k and v [1, Hkv, n,
// D]: the per-head RMS norms (values without a scale when has_v_norm) and
// the half-rotation RoPE over depth * proportion dims. The attributes are
// epsilon, has_v_norm, head_dim, max_timescale, min_timescale, num_heads,
// num_kv_heads and proportion.
struct QkvNormRopeOp {
  LiteRtNvidiaQkvNormRopeParams params{};
};

std::optional<QkvNormRopeOp> ParseQkvNormRope(const Op& op) {
  if (op.Inputs().size() != 4 || op.Outputs().size() != 3) {
    return std::nullopt;
  }
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value()) {
    return std::nullopt;
  }
  const auto& map = options.attributes_map.value();
  const auto heads = map["num_heads"];
  const auto kv_heads = map["num_kv_heads"];
  const auto depth = map["head_dim"];
  if (heads.IsNull() || kv_heads.IsNull() || depth.IsNull()) {
    return std::nullopt;
  }
  QkvNormRopeOp result;
  auto& p = result.params;
  p.heads = heads.AsInt32();
  p.kv_heads = kv_heads.AsInt32();
  p.depth = depth.AsInt32();
  p.has_v_norm = !map["has_v_norm"].IsNull() && map["has_v_norm"].AsBool();
  p.epsilon = map["epsilon"].IsNull() ? 1e-6f : map["epsilon"].AsFloat();
  p.min_timescale =
      map["min_timescale"].IsNull() ? 1.0f : map["min_timescale"].AsFloat();
  p.max_timescale =
      map["max_timescale"].IsNull() ? 10000.0f : map["max_timescale"].AsFloat();
  const float proportion =
      map["proportion"].IsNull() ? 1.0f : map["proportion"].AsFloat();
  if (p.depth <= 0 || proportion < 0.0f || proportion > 1.0f) {
    return std::nullopt;
  }
  p.rotary = static_cast<int32_t>(std::lround(p.depth * proportion));
  p.rotary -= p.rotary % 2;
  // Shapes: x [.., n, W], position with n values, scales [D], outputs.
  auto x_type = op.Inputs()[0].RankedTensorType();
  auto pos_type = op.Inputs()[1].RankedTensorType();
  if (!x_type || !pos_type || x_type->Layout().Rank() < 2 ||
      !IsFloatLike(op.Inputs()[0].ElementType()) ||
      op.Inputs()[1].ElementType() != litert::ElementType::Int32 ||
      op.Inputs()[0].HasWeights()) {
    return std::nullopt;
  }
  const auto x_dims = x_type->Layout().Dimensions();
  const int64_t width =
      static_cast<int64_t>(p.heads + 2 * p.kv_heads) * p.depth;
  if (x_dims.back() != width) return std::nullopt;
  int64_t tokens = 1;
  for (size_t d = 0; d + 1 < x_dims.size(); ++d) {
    if (x_dims[d] <= 0) return std::nullopt;
    tokens *= x_dims[d];
  }
  int64_t pos_count = 1;
  for (int32_t d : pos_type->Layout().Dimensions()) pos_count *= d;
  if (pos_count != tokens || tokens > std::numeric_limits<int32_t>::max()) {
    return std::nullopt;
  }
  p.tokens = static_cast<int32_t>(tokens);
  for (int i : {2, 3}) {
    const Tensor scale = op.Inputs()[i];
    if (scale.ElementType() != litert::ElementType::Float32 ||
        !scale.HasWeights() ||
        scale.Weights().Bytes().size() != static_cast<size_t>(p.depth) * 4) {
      return std::nullopt;
    }
  }
  const int32_t n = p.tokens;
  if (!TensorHasShape(op.Outputs()[0], {1, p.heads, n, p.depth}) ||
      !TensorHasShape(op.Outputs()[1], {1, p.kv_heads, n, p.depth}) ||
      !TensorHasShape(op.Outputs()[2], {1, p.kv_heads, n, p.depth})) {
    return std::nullopt;
  }
  if (!LiteRtNvidiaQkvNormRopeSupports(&p)) return std::nullopt;
  return result;
}

// odml.sdpa_transposed(q [1, Hq, n, D], k [1, H, S, D], v [1, H, D, S],
// mask bool [1, 1, 1 | n, S], params) -> [1, Hq, n, D]: the Hq / H query
// heads of a KV head attend over the whole cache with the mask (false
// scores take the decomposition's fill of -10000). Returns Hq / H.
constexpr float kSdpaTransposedFill = -10000.0f;

// The tokens per query head of an attention block the CUDA attention plugin
// runs, as given or after padding: the smallest n' >= `tokens` whose
// group * n' rows fit the decode kernel (at most 16) or the tensor-core kernel
// (a multiple of its row tile, n' dividing it). Gemma 4 26B's verify step
// has 8 query heads x 5 tokens per KV head: 40 rows run as 48. nullopt when
// no padding of up to 15 tokens does.
std::optional<int32_t> SdpaTransposedPaddedTokens(int32_t group, int32_t tokens,
                                                  int32_t depth) {
  if (group < 1 || tokens < 1) {
    return std::nullopt;
  }
  for (int32_t padded = tokens; padded < tokens + 16; ++padded) {
    const int32_t rows = group * padded;
    if (rows <= 16 ||
        (LiteRtNvidiaGlobalAttentionSupports(rows, padded, depth) &&
         kSdpaTransposedFill <= kMaxGlobalAttentionFill)) {
      return padded;
    }
  }
  return std::nullopt;
}

// Many query rows of a multi-head FP16 cache that already holds the chunk's
// keys (Qwen 3.5's prefill attends to its updated caches): the tiled
// tensor-core kernel over the whole cache, no padding (tiled_attention.h).
bool SdpaTransposedTiledCache(int32_t heads, int32_t rows, int32_t mask_rows,
                              int32_t depth, int32_t cache_len) {
  const LiteRtNvidiaAttentionShape shape = {heads, rows,      mask_rows,
                                            depth, cache_len, 0};
  return rows > 16 && kSdpaTransposedFill <= kMaxGlobalAttentionFill &&
         LiteRtNvidiaTiledAttentionSupports(&shape);
}

// An odml.swiglu composite (x [.., 2 * N]) -> act(x[.., :N]) * x[.., N:], the
// feed-forward gate over a fused gate/up projection. The composite keeps its
// name whatever the activation: its `activation` attribute names it, SiLU
// when absent or "silu" (Qwen 3.5's SwiGLU) and the tanh approximation of
// GELU for "gelu_tanh" or "gelu_pytorch_tanh" (Gemma 4's GeGLU). Any other
// activation is left to the composite's decomposition.
struct SwigluGate {
  int32_t gate_size = 0;
  LiteRtNvidiaGemmGate activation = kLiteRtNvidiaGemmGateSilu;
};

std::optional<SwigluGate> ParseSwiglu(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return std::nullopt;
  }
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value() ||
      options.attributes_map.value()["gate_size"].IsNull()) {
    return std::nullopt;
  }
  SwigluGate gate;
  gate.gate_size = options.attributes_map.value()["gate_size"].AsInt32();
  const auto activation = options.attributes_map.value()["activation"];
  if (!activation.IsNull()) {
    if (!activation.IsString()) return std::nullopt;
    const std::string name = activation.AsString().str();
    if (name == "gelu_tanh" || name == "gelu_pytorch_tanh") {
      gate.activation = kLiteRtNvidiaGemmGateGeluTanh;
    } else if (name != "silu") {
      return std::nullopt;
    }
  }
  auto in_type = op.Inputs()[0].RankedTensorType();
  if (!in_type || in_type->Layout().Rank() < 1 || gate.gate_size <= 0 ||
      in_type->Layout().Dimensions().back() != 2 * gate.gate_size ||
      !IsFloatLike(op.Inputs()[0].ElementType()) ||
      !IsFloatLike(op.Outputs()[0].ElementType())) {
    return std::nullopt;
  }
  return gate;
}

// A composite's boolean attribute, false when absent.
bool CompositeBoolAttribute(const Op& op, const char* name) {
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value()) {
    return false;
  }
  const auto value = options.attributes_map.value()[name];
  return value.IsBool() && value.AsBool();
}

std::optional<int32_t> SdpaTransposedGroup(const Op& op) {
  // q, k, v, then a bool mask and the params tensor, or the params tensor
  // alone (ML Drift's form without a mask: the keys up to the params' active
  // count, causal with is_causal; MaskFromSdpaParams).
  const bool masked = op.Inputs().size() == 5;
  if ((!masked && op.Inputs().size() != 4) || op.Outputs().size() != 1) {
    return std::nullopt;
  }
  if (!masked && !TensorHasShape(op.Inputs()[3], {1, 1, 1, 7})) {
    return std::nullopt;
  }
  std::array<std::array<int32_t, 4>, 4> dims;
  for (int i = 0; i < (masked ? 4 : 3); ++i) {
    auto type = op.Inputs()[i].RankedTensorType();
    if (!type || type->Layout().Rank() != 4) {
      return std::nullopt;
    }
    const auto element_type = op.Inputs()[i].ElementType();
    if (i == 0 ? !IsFloatLike(element_type)
               : (i == 3 ? element_type != litert::ElementType::Bool
                         : !IsFp16StoredCacheType(element_type))) {
      return std::nullopt;
    }
    const auto shape = type->Layout().Dimensions();
    std::copy(shape.begin(), shape.end(), dims[i].begin());
    for (int d : dims[i]) {
      if (d <= 0) return std::nullopt;
    }
  }
  const auto& q = dims[0];
  const auto& k = dims[1];
  const auto& v = dims[2];
  if (!masked) dims[3] = {1, 1, 1, k[2]};
  const auto& mask = dims[3];
  if (q[0] != 1 || k[0] != 1 || v[0] != 1 || k[1] < 1 || q[1] % k[1] != 0 ||
      k[3] != q[3] || v[1] != k[1] || v[2] != k[3] || v[3] != k[2] ||
      mask[0] != 1 || mask[1] != 1 || (mask[2] != 1 && mask[2] != q[2]) ||
      mask[3] != k[2] ||
      !TensorHasShape(op.Outputs()[0], {q[0], q[1], q[2], q[3]})) {
    return std::nullopt;
  }
  const int32_t group = q[1] / k[1];
  const int32_t rows = group * q[2];
  const int32_t depth = q[3];
  if (depth != 128 && depth != 256 && depth != 512) {
    return std::nullopt;
  }
  // Up to 16 rows run the decode kernel, a few tokens with more rows run it
  // in chunks, many tokens the tensor-core kernel (depth 512, rows a multiple
  // of its tile, padded otherwise: SdpaTransposedPaddedTokens).
  if (rows > 16 && q[2] > 16 &&
      !SdpaTransposedPaddedTokens(group, q[2], depth).has_value() &&
      !SdpaTransposedTiledCache(k[1], rows, mask[2], depth, k[2])) {
    return std::nullopt;
  }
  return group;
}

// The composites over quantized KV caches (INT8 queries over the INT8 ring
// caches of local layers and the INT4 caches of global ones) run on the
// quantized attention plugin (quant_attention_plugin.h), which reads and
// writes the caches as the model stores them. The plugin takes the float
// tensors behind the graph's QUANTIZE ops, so these lowerings need float
// activations.
bool QuantAttentionEnabled() { return Fp16ActivationsEnabled(); }

// The mask fills of the decompositions: FP16 0xF999 in odml.sdpa_transposed
// over quantized tensors, -10000 in odml.fused_sdpa_cache_update.
constexpr float kQuantSdpaFill = -45856.0f;
constexpr float kQuantFusedSdpaFill = -10000.0f;

// The scale and the bits of an INT8 or INT4 tensor quantized per tensor with
// a zero point of 0.
struct QuantTensor {
  float scale = 0.0f;
  int32_t bits = 0;
};

std::optional<QuantTensor> ParseQuantTensor(const Tensor& tensor) {
  const auto type = tensor.ElementType();
  if ((type != litert::ElementType::Int8 &&
       type != litert::ElementType::Int4) ||
      tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
    return std::nullopt;
  }
  const auto q = tensor.PerTensorQuantization();
  if (!std::isfinite(q.scale) || q.scale <= 0.0f || q.zero_point != 0) {
    return std::nullopt;
  }
  return QuantTensor{q.scale, type == litert::ElementType::Int8 ? 8 : 4};
}

bool QuantizedLike(const Tensor& tensor, float scale, int32_t bits) {
  const auto quant = ParseQuantTensor(tensor);
  return quant.has_value() && quant->bits == bits && quant->scale == scale;
}

std::optional<std::array<int32_t, 4>> StaticRank4Dims(const Tensor& tensor) {
  auto type = tensor.RankedTensorType();
  if (!type || type->Layout().Rank() != 4 || type->Layout().HasStrides()) {
    return std::nullopt;
  }
  std::array<int32_t, 4> dims;
  const auto shape = type->Layout().Dimensions();
  std::copy(shape.begin(), shape.end(), dims.begin());
  for (int32_t d : dims) {
    if (d <= 0) return std::nullopt;
  }
  return dims;
}

// A key cache [1, H, S, D] and a value cache [1, H, D, S] of one quantized
// type.
struct QuantCaches {
  int32_t heads = 0;
  int32_t seq = 0;
  int32_t depth = 0;
  int32_t bits = 0;
  float k_scale = 0.0f;
  float v_scale = 0.0f;
};

std::optional<QuantCaches> ParseQuantCaches(const Tensor& k, const Tensor& v) {
  const auto k_quant = ParseQuantTensor(k);
  const auto v_quant = ParseQuantTensor(v);
  const auto kd = StaticRank4Dims(k);
  const auto vd = StaticRank4Dims(v);
  if (!k_quant || !v_quant || !kd || !vd || k_quant->bits != v_quant->bits ||
      (*kd)[0] != 1 || (*vd)[0] != 1 || (*vd)[1] != (*kd)[1] ||
      (*vd)[2] != (*kd)[3] || (*vd)[3] != (*kd)[2] || (*kd)[3] % 2 != 0) {
    return std::nullopt;
  }
  return QuantCaches{(*kd)[1],      (*kd)[2],       (*kd)[3],
                     k_quant->bits, k_quant->scale, v_quant->scale};
}

// The plugin leaves the keys a mask hides out of the softmax, where the
// decompositions give their scores `fill`. That is the same result for every
// row that sees a key when the fill is so far below every score that its
// weight is no fp32 number (exp(-104) is none); a score is at most depth x
// 128 x the keys' limit x the two scales. A row that sees no key gets zero
// from the plugin and the average of all values from the decomposition: the
// graphs replace such a row (the SELECT after odml.sdpa_transposed) or never
// read it (the padding of a prefill chunk).
bool QuantFillHidesScores(float fill, const QuantCaches& caches,
                          float q_scale) {
  const float max_score = static_cast<float>(caches.depth) * 128.0f *
                          static_cast<float>(1 << (caches.bits - 1)) * q_scale *
                          caches.k_scale;
  return fill + 2.0f * max_score < -150.0f;
}

// odml.cache_update over quantized tensors, (k [1, H, n, D], v [1, H, n, D],
// params, k_cache, v_cache, k_start [4], v_start [4]) -> the two caches. The
// keys, the values and their caches share a quantization, so the
// decomposition's re-quantization of the rows is the identity and its two
// DYNAMIC_UPDATE_SLICE ops write them as they are. The caches are inputs of
// the graph.
struct QuantCacheUpdate {
  QuantCaches caches;
  int32_t tokens = 0;  // n
};

std::optional<QuantCacheUpdate> ParseQuantCacheUpdate(const Op& op) {
  if (!QuantAttentionEnabled() || op.Inputs().size() != 7 ||
      op.Outputs().size() != 2 || !op.Inputs()[3].IsSubgraphInput() ||
      !op.Inputs()[4].IsSubgraphInput()) {
    return std::nullopt;
  }
  const auto caches = ParseQuantCaches(op.Inputs()[3], op.Inputs()[4]);
  const auto kd = StaticRank4Dims(op.Inputs()[0]);
  const auto vd = StaticRank4Dims(op.Inputs()[1]);
  if (!caches || !kd || !vd || *kd != *vd || (*kd)[0] != 1 ||
      (*kd)[1] != caches->heads || (*kd)[2] > caches->seq ||
      (*kd)[3] != caches->depth) {
    return std::nullopt;
  }
  for (int i = 0; i < 2; ++i) {
    const float scale = i == 0 ? caches->k_scale : caches->v_scale;
    auto start_type = op.Inputs()[5 + i].RankedTensorType();
    if (!QuantizedLike(op.Inputs()[i], scale, caches->bits) ||
        !QuantizedLike(op.Outputs()[i], scale, caches->bits) || !start_type ||
        StaticRank4Dims(op.Outputs()[i]) !=
            StaticRank4Dims(op.Inputs()[3 + i]) ||
        op.Inputs()[5 + i].ElementType() != litert::ElementType::Int32 ||
        start_type->Layout().Rank() != 1 ||
        start_type->Layout().Dimensions()[0] != 4) {
      return std::nullopt;
    }
  }
  return QuantCacheUpdate{*caches, (*kd)[2]};
}

// odml.sdpa_transposed with an INT8 query, (q [1, Hq, n, D], k_cache,
// v_cache, mask [1, 1, n, S] bool, params) -> float [1, Hq, n, D]: the
// Hq / H query heads of a KV head attend over the whole cache under the
// mask. The caches are the results of an odml.cache_update that
// ParseQuantCacheUpdate takes, or inputs of the graph. Up to 16 rows per KV
// head run the plugin's decode kernel, more its tensor-core kernel.
struct QuantSdpa {
  QuantCaches caches;
  float q_scale = 0.0f;
  int32_t group = 0;   // Hq / H
  int32_t tokens = 0;  // n
};

std::optional<QuantSdpa> ParseQuantSdpa(const Op& op) {
  if (!QuantAttentionEnabled() || op.Inputs().size() != 5 ||
      op.Outputs().size() != 1) {
    return std::nullopt;
  }
  const auto q_quant = ParseQuantTensor(op.Inputs()[0]);
  const auto caches = ParseQuantCaches(op.Inputs()[1], op.Inputs()[2]);
  const auto qd = StaticRank4Dims(op.Inputs()[0]);
  if (!q_quant || q_quant->bits != 8 || !caches || !qd || (*qd)[0] != 1 ||
      (*qd)[1] % caches->heads != 0 || (*qd)[3] != caches->depth ||
      op.Inputs()[3].ElementType() != litert::ElementType::Bool ||
      !TensorHasShape(op.Inputs()[3], {1, 1, (*qd)[2], caches->seq}) ||
      !IsFloatLike(op.Outputs()[0].ElementType()) ||
      !TensorHasShape(op.Outputs()[0],
                      {(*qd)[0], (*qd)[1], (*qd)[2], (*qd)[3]})) {
    return std::nullopt;
  }
  for (int i : {1, 2}) {
    if (op.Inputs()[i].IsSubgraphInput()) {
      continue;
    }
    const auto producer = op.Inputs()[i].GetDefiningOp();
    if (!producer || producer->Code() != kLiteRtOpCodeShloComposite ||
        CompositeOpName(*producer) != "odml.cache_update" ||
        !ParseQuantCacheUpdate(*producer).has_value()) {
      return std::nullopt;
    }
  }
  // The query arrives scaled, and the caches have their sequence on axes 2
  // and 3.
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) == kLiteRtStatusOk &&
      options.attributes_map.has_value()) {
    const auto& map = options.attributes_map.value();
    if ((!map["scale"].IsNull() && map["scale"].AsFloat() != 1.0f) ||
        (!map["k_ts_idx"].IsNull() && map["k_ts_idx"].AsInt32() != 2) ||
        (!map["v_ts_idx"].IsNull() && map["v_ts_idx"].AsInt32() != 3)) {
      return std::nullopt;
    }
  }
  if (!QuantFillHidesScores(kQuantSdpaFill, *caches, q_quant->scale)) {
    return std::nullopt;
  }
  const int32_t group = (*qd)[1] / caches->heads;
  const int32_t rows = group * (*qd)[2];
  if (rows <= 16) {
    if (!LiteRtNvidiaQuantDecodeAttentionSupports(rows, caches->depth,
                                                  caches->bits)) {
      return std::nullopt;
    }
  } else {
    const LiteRtNvidiaAttentionShape shape = {
        caches->heads, rows, (*qd)[2], caches->depth, caches->seq, 0};
    if (!LiteRtNvidiaTiledQuantAttentionSupports(&shape, caches->bits) ||
        !LiteRtNvidiaTiledAttentionAvailable()) {
      return std::nullopt;
    }
  }
  return QuantSdpa{*caches, q_quant->scale, group, (*qd)[2]};
}

// odml.fused_sdpa_cache_update, (q [1, H, R, D] INT8, k_cache [1, H, S, D],
// v_cache [1, H, D, S], k [1, H, M, D], v [1, H, D, M], mask [1, 1, M, S + M]
// bool, params [1, 1, 1, 7]) -> (float [1, H, R, D], the two caches): the
// attention of R / M query heads per KV head over the INT8 ring caches
// followed by the chunk's own keys and values, and the chunk written into the
// ring (unless update_cache is false). The caches are inputs of the graph.
struct QuantFusedSdpa {
  QuantCaches caches;
  float q_scale = 0.0f;
  int32_t rows = 0;    // R
  int32_t tokens = 0;  // M
  bool update_cache = true;
};

std::optional<QuantFusedSdpa> ParseQuantFusedSdpa(const Op& op) {
  if (!QuantAttentionEnabled() || op.Inputs().size() != 7 ||
      op.Outputs().size() != 3 || !op.Inputs()[1].IsSubgraphInput() ||
      !op.Inputs()[2].IsSubgraphInput()) {
    return std::nullopt;
  }
  const auto q_quant = ParseQuantTensor(op.Inputs()[0]);
  const auto caches = ParseQuantCaches(op.Inputs()[1], op.Inputs()[2]);
  const auto qd = StaticRank4Dims(op.Inputs()[0]);
  const auto kd = StaticRank4Dims(op.Inputs()[3]);
  if (!q_quant || q_quant->bits != 8 || !caches || caches->bits != 8 || !qd ||
      !kd) {
    return std::nullopt;
  }
  const int32_t rows = (*qd)[2];
  const int32_t tokens = (*kd)[2];
  if ((*qd)[0] != 1 || (*qd)[1] != caches->heads || (*qd)[3] != caches->depth ||
      rows % tokens != 0 ||
      !QuantizedLike(op.Inputs()[3], caches->k_scale, 8) ||
      !TensorHasShape(op.Inputs()[3],
                      {1, caches->heads, tokens, caches->depth}) ||
      !QuantizedLike(op.Inputs()[4], caches->v_scale, 8) ||
      !TensorHasShape(op.Inputs()[4],
                      {1, caches->heads, caches->depth, tokens}) ||
      op.Inputs()[5].ElementType() != litert::ElementType::Bool ||
      !TensorHasShape(op.Inputs()[5], {1, 1, tokens, caches->seq + tokens}) ||
      op.Inputs()[6].ElementType() != litert::ElementType::Int32 ||
      !TensorHasShape(op.Inputs()[6], {1, 1, 1, 7}) ||
      !IsFloatLike(op.Outputs()[0].ElementType()) ||
      !TensorHasShape(op.Outputs()[0],
                      {1, caches->heads, rows, caches->depth}) ||
      !QuantizedLike(op.Outputs()[1], caches->k_scale, 8) ||
      !TensorHasShape(op.Outputs()[1],
                      {1, caches->heads, caches->seq, caches->depth}) ||
      !QuantizedLike(op.Outputs()[2], caches->v_scale, 8) ||
      !TensorHasShape(op.Outputs()[2],
                      {1, caches->heads, caches->depth, caches->seq})) {
    return std::nullopt;
  }
  bool update_cache = true;
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) == kLiteRtStatusOk &&
      options.attributes_map.has_value()) {
    const auto update = options.attributes_map.value()["update_cache"];
    if (update.IsBool()) {
      update_cache = update.AsBool();
    }
  }
  const LiteRtNvidiaAttentionShape shape = {caches->heads, rows,        tokens,
                                            caches->depth, caches->seq, tokens};
  if (!LiteRtNvidiaTiledQuantAttentionSupports(&shape, 8) ||
      !LiteRtNvidiaTiledAttentionAvailable() ||
      !QuantFillHidesScores(kQuantFusedSdpaFill, *caches, q_quant->scale)) {
    return std::nullopt;
  }
  return QuantFusedSdpa{*caches, q_quant->scale, rows, tokens, update_cache};
}

// The composites over KV caches in the forms the quantized attention plugin
// does not take (general_attention_plugin.h): float or INT8 queries, bool or
// additive float masks, a scale and a soft cap, head depths other than 256,
// packed INT4 caches in odml.fused_sdpa_cache_update, float or re-scaled
// keys and values into INT8, INT4 and UINT8 caches, and UINT8 caches. Off
// unless LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION=1: models whose composites
// the specialized kernels or the decompositions run today build the same
// engines without it. Like those kernels, the plugin takes the float tensors
// behind the graphs' QUANTIZE ops, so it needs 16-bit activations.
bool GeneralAttentionEnabled() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_GENERAL_ATTENTION");
  return value != nullptr && std::strcmp(value, "1") == 0 &&
         QuantAttentionEnabled();
}

// A cache tensor of the general forms: INT8 or INT4 with a zero point of 0,
// or UINT8 with 128, quantized per tensor.
struct GeneralCacheType {
  int32_t type = 0;  // LiteRtNvidiaGeneralType
  float scale = 0.0f;
};

std::optional<GeneralCacheType> ParseGeneralCacheType(const Tensor& tensor) {
  const auto type = tensor.ElementType();
  if (tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
    return std::nullopt;
  }
  const auto q = tensor.PerTensorQuantization();
  if (!std::isfinite(q.scale) || q.scale <= 0.0f) {
    return std::nullopt;
  }
  if (type == litert::ElementType::Int8 && q.zero_point == 0) {
    return GeneralCacheType{kLiteRtNvidiaGeneralInt8, q.scale};
  }
  if (type == litert::ElementType::Int4 && q.zero_point == 0) {
    return GeneralCacheType{kLiteRtNvidiaGeneralInt4, q.scale};
  }
  if (type == litert::ElementType::UInt8 && q.zero_point == 128) {
    return GeneralCacheType{kLiteRtNvidiaGeneralUint8, q.scale};
  }
  return std::nullopt;
}

// A key cache [1, H, S, D] and a value cache [1, H, D, S] of one quantized
// type.
struct GeneralCaches {
  int32_t type = 0;
  float k_scale = 0.0f;
  float v_scale = 0.0f;
  int32_t heads = 0;
  int32_t seq = 0;
  int32_t depth = 0;
};

std::optional<GeneralCaches> ParseGeneralCaches(const Tensor& k,
                                                const Tensor& v) {
  const auto kt = ParseGeneralCacheType(k);
  const auto vt = ParseGeneralCacheType(v);
  const auto kd = StaticRank4Dims(k);
  const auto vd = StaticRank4Dims(v);
  if (!kt || !vt || !kd || !vd || kt->type != vt->type || (*kd)[0] != 1 ||
      (*vd)[0] != 1 || (*vd)[1] != (*kd)[1] || (*vd)[2] != (*kd)[3] ||
      (*vd)[3] != (*kd)[2] ||
      (kt->type == kLiteRtNvidiaGeneralInt4 &&
       ((*kd)[2] % 2 != 0 || (*kd)[3] % 2 != 0))) {
    return std::nullopt;
  }
  return GeneralCaches{kt->type, kt->scale, vt->scale,
                       (*kd)[1], (*kd)[2],  (*kd)[3]};
}

// A float input of the general forms, or an INT8 one (the plugin takes the
// float the graph quantized and rounds it with this scale): the scale, 0 for
// a float. nullopt for anything else.
std::optional<float> GeneralFloatInput(const Tensor& tensor) {
  if (IsFloatLike(tensor.ElementType()) &&
      tensor.QTypeId() == kLiteRtQuantizationNone) {
    return 0.0f;
  }
  const auto quant = ParseQuantTensor(tensor);
  if (quant.has_value() && quant->bits == 8) {
    return quant->scale;
  }
  return std::nullopt;
}

// The keys or values a step writes into quantized caches: floats, or
// quantized values (an INT8 grid of any scale, or the cache's own grid).
std::optional<float> GeneralUpdateInput(const Tensor& tensor,
                                        const GeneralCaches& caches,
                                        float cache_scale) {
  if (const auto scale = GeneralFloatInput(tensor)) {
    return scale;
  }
  const auto own = ParseGeneralCacheType(tensor);
  if (own.has_value() && own->type == caches.type &&
      own->scale == cache_scale) {
    return 0.0f;  // already on the cache's grid
  }
  return std::nullopt;
}

// The attributes of the attention composites: scale (1), softcap or
// logit_cap (none), and the caches' time axes, which must be 2 and 3.
struct GeneralAttentionAttributes {
  float scale = 1.0f;
  float softcap = 0.0f;
};

std::optional<GeneralAttentionAttributes> ParseGeneralAttentionAttributes(
    const Op& op) {
  GeneralAttentionAttributes result;
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value()) {
    return result;
  }
  const auto& map = options.attributes_map.value();
  if ((!map["k_ts_idx"].IsNull() && map["k_ts_idx"].AsInt32() != 2) ||
      (!map["v_ts_idx"].IsNull() && map["v_ts_idx"].AsInt32() != 3)) {
    return std::nullopt;
  }
  if (!map["scale"].IsNull()) result.scale = map["scale"].AsFloat();
  const auto softcap =
      !map["softcap"].IsNull() ? map["softcap"] : map["logit_cap"];
  if (!softcap.IsNull()) result.softcap = softcap.AsFloat();
  if (!std::isfinite(result.scale) || !std::isfinite(result.softcap) ||
      result.softcap < 0.0f) {
    return std::nullopt;
  }
  return result;
}

bool IsGeneralMask(const Tensor& mask, int32_t rows, int32_t keys) {
  return (mask.ElementType() == litert::ElementType::Bool ||
          (IsFloatLike(mask.ElementType()) &&
           mask.QTypeId() == kLiteRtQuantizationNone)) &&
         (TensorHasShape(mask, {1, 1, rows, keys}) ||
          TensorHasShape(mask, {1, 1, 1, keys}));
}

// odml.cache_update into quantized caches in a general form: the forms
// ParseQuantCacheUpdate and the INT8 scatter path take stay theirs. The
// caches are inputs of the graph.
struct GeneralCacheUpdate {
  GeneralCaches caches;
  int32_t tokens = 0;
  float src_int8_scale = 0.0f;
};

std::optional<GeneralCacheUpdate> ParseGeneralCacheUpdate(const Op& op) {
  if (!GeneralAttentionEnabled() || op.Inputs().size() != 7 ||
      op.Outputs().size() != 2 || !op.Inputs()[3].IsSubgraphInput() ||
      !op.Inputs()[4].IsSubgraphInput() ||
      ParseQuantCacheUpdate(op).has_value()) {
    return std::nullopt;
  }
  const auto caches = ParseGeneralCaches(op.Inputs()[3], op.Inputs()[4]);
  const auto kd = StaticRank4Dims(op.Inputs()[0]);
  const auto vd = StaticRank4Dims(op.Inputs()[1]);
  if (!caches || !kd || !vd || *kd != *vd || (*kd)[0] != 1 ||
      (*kd)[1] != caches->heads || (*kd)[2] > caches->seq ||
      (*kd)[3] != caches->depth) {
    return std::nullopt;
  }
  // Float keys and values into INT8 caches are the scatter path's.
  if (caches->type == kLiteRtNvidiaGeneralInt8 &&
      IsFloatLike(op.Inputs()[0].ElementType()) &&
      IsFloatLike(op.Inputs()[1].ElementType())) {
    return std::nullopt;
  }
  std::optional<float> src_scale;
  for (int i = 0; i < 2; ++i) {
    const float cache_scale = i == 0 ? caches->k_scale : caches->v_scale;
    const auto scale = GeneralUpdateInput(op.Inputs()[i], *caches, cache_scale);
    const auto out = ParseGeneralCacheType(op.Outputs()[i]);
    auto start_type = op.Inputs()[5 + i].RankedTensorType();
    if (!scale || (src_scale.has_value() && *src_scale != *scale) || !out ||
        out->type != caches->type || out->scale != cache_scale ||
        StaticRank4Dims(op.Outputs()[i]) !=
            StaticRank4Dims(op.Inputs()[3 + i]) ||
        !start_type ||
        op.Inputs()[5 + i].ElementType() != litert::ElementType::Int32 ||
        start_type->Layout().Rank() != 1 ||
        start_type->Layout().Dimensions()[0] != 4) {
      return std::nullopt;
    }
    src_scale = scale;
  }
  return GeneralCacheUpdate{*caches, (*kd)[2], *src_scale};
}

// Whether `tensor`, a cache an attention composite reads, is a quantized
// cache an engine holds as the model stores it: an input of the graph, or
// the result of an odml.cache_update the quantized attention plugin, the
// general plugin or the INT8 scatter path writes.
bool IsGeneralCacheSource(const Tensor& tensor) {
  if (tensor.IsSubgraphInput()) {
    return true;
  }
  const auto producer = tensor.GetDefiningOp();
  if (!producer || producer->Code() != kLiteRtOpCodeShloComposite ||
      CompositeOpName(*producer) != "odml.cache_update") {
    return false;
  }
  if (ParseQuantCacheUpdate(*producer).has_value() ||
      ParseGeneralCacheUpdate(*producer).has_value()) {
    return true;
  }
  // The scatter path: float keys and values into INT8 caches.
  const auto& in = producer->Inputs();
  return in.size() == 7 && IsFloatLike(in[0].ElementType()) &&
         IsFloatLike(in[1].ElementType()) &&
         ParseGeneralCacheType(in[3]).has_value() &&
         in[3].ElementType() == litert::ElementType::Int8;
}

// odml.sdpa_transposed over quantized caches in a general form, (q
// [1, Hq, n, D] float or INT8, k_cache, v_cache, mask [1, 1, n | 1, S] bool
// or float, params) -> float [1, Hq, n, D]. The form ParseQuantSdpa takes
// stays its.
struct GeneralSdpa {
  GeneralCaches caches;
  GeneralAttentionAttributes attributes;
  float q_int8_scale = 0.0f;
  int32_t group = 0;
  int32_t tokens = 0;
};

std::optional<GeneralSdpa> ParseGeneralSdpa(const Op& op) {
  if (!GeneralAttentionEnabled() ||
      (op.Inputs().size() != 4 && op.Inputs().size() != 5) ||
      op.Outputs().size() != 1 || ParseQuantSdpa(op).has_value()) {
    return std::nullopt;
  }
  const auto& in = op.Inputs();
  const auto caches = ParseGeneralCaches(in[1], in[2]);
  const auto q_scale = GeneralFloatInput(in[0]);
  const auto qd = StaticRank4Dims(in[0]);
  const auto attributes = ParseGeneralAttentionAttributes(op);
  if (!caches || !q_scale || !qd || !attributes || (*qd)[0] != 1 ||
      (*qd)[1] % caches->heads != 0 || (*qd)[3] != caches->depth ||
      !IsGeneralMask(in[3], (*qd)[2], caches->seq) ||
      !IsFloatLike(op.Outputs()[0].ElementType()) ||
      !TensorHasShape(op.Outputs()[0],
                      {(*qd)[0], (*qd)[1], (*qd)[2], (*qd)[3]}) ||
      !IsGeneralCacheSource(in[1]) || !IsGeneralCacheSource(in[2])) {
    return std::nullopt;
  }
  GeneralSdpa result{*caches, *attributes, *q_scale, (*qd)[1] / caches->heads,
                     (*qd)[2]};
  LiteRtNvidiaGeneralAttentionParams p{};
  p.kv_heads = caches->heads;
  p.rows = result.group * result.tokens;
  p.depth = caches->depth;
  p.cache_len = caches->seq;
  p.mask_rows = 1;
  p.q = {kLiteRtNvidiaGeneralBFloat16, *q_scale};
  p.caches = {caches->type, caches->k_scale, caches->v_scale};
  p.scale = attributes->scale;
  p.softcap = attributes->softcap;
  if (!LiteRtNvidiaGeneralAttentionSupports(&p)) {
    return std::nullopt;
  }
  return result;
}

// odml.fused_sdpa_cache_update over quantized ring caches in a general form,
// (q [1, H, R, D], k_cache, v_cache, k [1, H, M, D], v [1, H, D, M], mask
// [1, 1, M, S + M] bool or float, params int32) -> (float [1, H, R, D] and,
// unless the composite has one output, the two caches). The caches are
// inputs of the graph. The form ParseQuantFusedSdpa takes stays its.
struct GeneralFusedSdpa {
  GeneralCaches caches;
  GeneralAttentionAttributes attributes;
  float q_int8_scale = 0.0f;
  float new_int8_scale = 0.0f;
  int32_t rows = 0;
  int32_t tokens = 0;
  bool update_cache = true;
};

// A float cache of the fused composite (no NVIDIA path has one: the float
// caches the other paths keep in place are FP16 storage of FP32 tensors): its
// type, scale 1.
std::optional<GeneralCacheType> ParseGeneralFloatCacheType(
    const Tensor& tensor) {
  if (tensor.QTypeId() != kLiteRtQuantizationNone) return std::nullopt;
  switch (tensor.ElementType()) {
    case litert::ElementType::Float32:
      return GeneralCacheType{kLiteRtNvidiaGeneralFloat32, 1.0f};
    case litert::ElementType::Float16:
      return GeneralCacheType{kLiteRtNvidiaGeneralFloat16, 1.0f};
    case litert::ElementType::BFloat16:
      return GeneralCacheType{kLiteRtNvidiaGeneralBFloat16, 1.0f};
    default:
      return std::nullopt;
  }
}

// The fused composite's caches: quantized (ParseGeneralCaches) or float, the
// key cache [1, H, S, D] and the value cache [1, H, D, S] of one type.
std::optional<GeneralCaches> ParseGeneralFusedCaches(const Tensor& k,
                                                     const Tensor& v) {
  if (const auto quantized = ParseGeneralCaches(k, v)) return quantized;
  const auto kt = ParseGeneralFloatCacheType(k);
  const auto vt = ParseGeneralFloatCacheType(v);
  const auto kd = StaticRank4Dims(k);
  const auto vd = StaticRank4Dims(v);
  if (!kt || !vt || !kd || !vd || kt->type != vt->type || (*kd)[0] != 1 ||
      (*vd)[0] != 1 || (*vd)[1] != (*kd)[1] || (*vd)[2] != (*kd)[3] ||
      (*vd)[3] != (*kd)[2]) {
    return std::nullopt;
  }
  return GeneralCaches{kt->type, 1.0f, 1.0f, (*kd)[1], (*kd)[2], (*kd)[3]};
}

std::optional<GeneralFusedSdpa> ParseGeneralFusedSdpa(const Op& op) {
  if (!GeneralAttentionEnabled() || op.Inputs().size() != 7 ||
      (op.Outputs().size() != 3 && op.Outputs().size() != 1) ||
      !op.Inputs()[1].IsSubgraphInput() || !op.Inputs()[2].IsSubgraphInput() ||
      ParseQuantFusedSdpa(op).has_value()) {
    return std::nullopt;
  }
  const auto& in = op.Inputs();
  const auto caches = ParseGeneralFusedCaches(in[1], in[2]);
  const auto q_scale = GeneralFloatInput(in[0]);
  const auto qd = StaticRank4Dims(in[0]);
  const auto kd = StaticRank4Dims(in[3]);
  const auto attributes = ParseGeneralAttentionAttributes(op);
  if (!caches || !q_scale || !qd || !kd || !attributes) {
    return std::nullopt;
  }
  const int32_t rows = (*qd)[2];
  const int32_t tokens = (*kd)[2];
  const auto k_scale = GeneralUpdateInput(in[3], *caches, caches->k_scale);
  const auto v_scale = GeneralUpdateInput(in[4], *caches, caches->v_scale);
  auto params_type = in[6].RankedTensorType();
  int64_t params_count = 1;
  if (params_type) {
    for (int32_t d : params_type->Layout().Dimensions()) params_count *= d;
  }
  if ((*qd)[0] != 1 || (*qd)[1] != caches->heads || (*qd)[3] != caches->depth ||
      tokens < 1 || rows % tokens != 0 || !k_scale || !v_scale ||
      *k_scale != *v_scale ||
      !TensorHasShape(in[3], {1, caches->heads, tokens, caches->depth}) ||
      !TensorHasShape(in[4], {1, caches->heads, caches->depth, tokens}) ||
      !IsGeneralMask(in[5], tokens, caches->seq + tokens) ||
      !TensorHasShape(in[5], {1, 1, tokens, caches->seq + tokens}) ||
      in[6].ElementType() != litert::ElementType::Int32 || !params_type ||
      params_count < 2 || !IsFloatLike(op.Outputs()[0].ElementType()) ||
      !TensorHasShape(op.Outputs()[0],
                      {1, caches->heads, rows, caches->depth})) {
    return std::nullopt;
  }
  const bool update_cache = op.Outputs().size() == 3;
  if (update_cache) {
    for (int i = 0; i < 2; ++i) {
      auto out = ParseGeneralCacheType(op.Outputs()[1 + i]);
      if (!out) out = ParseGeneralFloatCacheType(op.Outputs()[1 + i]);
      if (!out || out->type != caches->type ||
          out->scale != (i == 0 ? caches->k_scale : caches->v_scale) ||
          StaticRank4Dims(op.Outputs()[1 + i]) != StaticRank4Dims(in[1 + i])) {
        return std::nullopt;
      }
    }
  }
  GeneralFusedSdpa result{*caches, *attributes, *q_scale,    *k_scale,
                          rows,    tokens,      update_cache};
  LiteRtNvidiaGeneralAttentionParams p{};
  p.kv_heads = caches->heads;
  p.rows = rows;
  p.depth = caches->depth;
  p.cache_len = caches->seq;
  p.new_len = tokens;
  p.mask_rows = tokens;
  p.q = {kLiteRtNvidiaGeneralBFloat16, *q_scale};
  p.caches = {caches->type, caches->k_scale, caches->v_scale};
  p.new_kv = {kLiteRtNvidiaGeneralBFloat16, *k_scale};
  p.scale = attributes->scale;
  p.softcap = attributes->softcap;
  LiteRtNvidiaGeneralWriteParams w{};
  w.heads = caches->heads;
  w.cache_len = caches->seq;
  w.depth = caches->depth;
  w.tokens = tokens;
  w.mode = kLiteRtNvidiaGeneralWriteRing;
  w.caches = p.caches;
  w.src = p.new_kv;
  if (!LiteRtNvidiaGeneralAttentionSupports(&p) ||
      (update_cache && !LiteRtNvidiaGeneralCacheWriteSupports(&w))) {
    return std::nullopt;
  }
  return result;
}

// The "moe" custom op of Gemma 4 26B-A4B (the XNNPACK delegate's
// mixture-of-experts contract): inputs x [1, n, model_dim], top_weights
// [1, n, k], top_indices [1, n, k] int32, gate and up weights INT4
// [hidden, experts, 1, model_dim] with scales [hidden, experts, 1,
// model_dim / 32], down weights [model_dim, experts, 1, hidden] with scales
// [model_dim, experts, 1, hidden / 32], per_expert_scale [1, 1, 1, experts];
// a flexbuffer map of num_experts, num_active_experts, model_dim, hidden_dim,
// activation (gelu | gelu_tanh | silu | swish), weight_type (int4) and
// renormalized_top_weights.
struct MoeCustomOp {
  int32_t experts = 0;
  int32_t top_k = 0;
  int32_t model_dim = 0;
  int32_t hidden_dim = 0;
  int32_t gate = 0;        // LiteRtNvidiaMoeShape::gate
  int32_t rows = 0;        // n
  int32_t scale_type = 0;  // 0 FP32, 1 FP16
};

std::optional<MoeCustomOp> ParseMoeCustomOp(const Op& op) {
  const auto code = op.CustomCode();
  if (!code || *code != "moe" || op.Inputs().size() != 10 ||
      op.Outputs().size() != 1) {
    return std::nullopt;
  }
  const auto options = op.CustomOptions();
  if (!options || options->empty()) {
    return std::nullopt;
  }
  const flexbuffers::Map map =
      flexbuffers::GetRoot(options->data(), options->size()).AsMap();
  if (map.IsTheEmptyMap()) {
    return std::nullopt;
  }
  MoeCustomOp moe;
  moe.experts = map["num_experts"].AsInt32();
  moe.top_k = map["num_active_experts"].AsInt32();
  moe.model_dim = map["model_dim"].AsInt32();
  moe.hidden_dim = map["hidden_dim"].AsInt32();
  const std::string activation =
      map["activation"].IsNull() ? "gelu" : map["activation"].AsString().str();
  const std::string weight_type = map["weight_type"].IsNull()
                                      ? "int4"
                                      : map["weight_type"].AsString().str();
  if (activation == "gelu_tanh") {
    moe.gate = 1;
  } else if (activation == "gelu") {
    moe.gate = 2;
  } else if (activation == "silu" || activation == "swish") {
    // Qwen's experts (the GPU delegate takes both names since b32ee661b).
    moe.gate = 3;
  } else {
    return std::nullopt;
  }
  if (weight_type != "int4" || moe.experts <= 0 || moe.top_k <= 0 ||
      moe.model_dim <= 0 || moe.hidden_dim <= 0) {
    return std::nullopt;
  }
  // x, top_weights, top_indices and the output.
  auto x_type = op.Inputs()[0].RankedTensorType();
  auto w_type = op.Inputs()[1].RankedTensorType();
  auto i_type = op.Inputs()[2].RankedTensorType();
  auto o_type = op.Outputs()[0].RankedTensorType();
  if (!x_type || !w_type || !i_type || !o_type ||
      !IsFloatLike(op.Inputs()[0].ElementType()) ||
      !IsFloatLike(op.Inputs()[1].ElementType()) ||
      op.Inputs()[2].ElementType() != litert::ElementType::Int32 ||
      !IsFloatLike(op.Outputs()[0].ElementType()) ||
      x_type->Layout().Rank() < 2 || op.Inputs()[0].HasWeights()) {
    return std::nullopt;
  }
  const auto x_dims = x_type->Layout().Dimensions();
  const auto w_dims = w_type->Layout().Dimensions();
  const auto i_dims = i_type->Layout().Dimensions();
  const auto o_dims = o_type->Layout().Dimensions();
  if (x_dims.back() != moe.model_dim || w_dims.empty() ||
      w_dims.back() != moe.top_k || i_dims.empty() ||
      i_dims.back() != moe.top_k || o_dims.size() != x_dims.size()) {
    return std::nullopt;
  }
  int64_t rows = 1;
  int64_t w_rows = 1;
  int64_t i_rows = 1;
  for (size_t d = 0; d + 1 < x_dims.size(); ++d) {
    if (x_dims[d] <= 0 || o_dims[d] != x_dims[d]) return std::nullopt;
    rows *= x_dims[d];
  }
  if (o_dims.back() != moe.model_dim) return std::nullopt;
  for (size_t d = 0; d + 1 < w_dims.size(); ++d) w_rows *= w_dims[d];
  for (size_t d = 0; d + 1 < i_dims.size(); ++d) i_rows *= i_dims[d];
  if (rows < 1 || rows > std::numeric_limits<int32_t>::max() ||
      w_rows != rows || i_rows != rows) {
    return std::nullopt;
  }
  moe.rows = static_cast<int32_t>(rows);
  // The experts: INT4 weights and their scales, all constants.
  const size_t gate_values =
      static_cast<size_t>(moe.hidden_dim) * moe.experts * moe.model_dim;
  const size_t down_values =
      static_cast<size_t>(moe.model_dim) * moe.experts * moe.hidden_dim;
  for (int i : {3, 5, 7}) {
    const Tensor weights = op.Inputs()[i];
    const Tensor scales = op.Inputs()[i + 1];
    const size_t values = i == 7 ? down_values : gate_values;
    if (weights.ElementType() != litert::ElementType::Int4 ||
        !weights.HasWeights() ||
        weights.Weights().Bytes().size() != values / 2 ||
        !scales.HasWeights()) {
      return std::nullopt;
    }
    const auto scale_type = scales.ElementType();
    const int32_t type_code =
        scale_type == litert::ElementType::Float32
            ? 0
            : (scale_type == litert::ElementType::Float16 ? 1 : -1);
    if (type_code < 0 || (i > 3 && type_code != moe.scale_type) ||
        scales.Weights().Bytes().size() !=
            values / 32 * (type_code == 0 ? 4 : 2)) {
      return std::nullopt;
    }
    moe.scale_type = type_code;
  }
  const Tensor expert_scale = op.Inputs()[9];
  if (expert_scale.ElementType() != litert::ElementType::Float32 ||
      !expert_scale.HasWeights() ||
      expert_scale.Weights().Bytes().size() !=
          static_cast<size_t>(moe.experts) * sizeof(float)) {
    return std::nullopt;
  }
  const LiteRtNvidiaMoeShape shape = {moe.rows,       moe.top_k,
                                      moe.experts,    moe.model_dim,
                                      moe.hidden_dim, moe.gate};
  if (!LiteRtNvidiaMoeSupports(&shape)) {
    return std::nullopt;
  }
  return moe;
}

// A recurrent state of a linear-attention layer (Qwen 3.5's Gated DeltaNet
// and its short convolution): an FP32 subgraph input whose updated value is
// the op's output, read by nothing else in the graph. The runtime binds both
// to one buffer and the plugins update it in place in FP32.
bool IsInPlaceFp32State(const Tensor& input, const Tensor& output) {
  return input.IsSubgraphInput() && IsUnquantizedFloat32(input) &&
         IsUnquantizedFloat32(output) && output.Uses().empty() &&
         input.Uses().size() == 1;
}

// Qwen 3.5's gated_delta_update custom op (q, k [1, Hk, n, Dk], v [1, H, n,
// Dv], beta, g [1, H, n], state [1, H, Dk, Dv][, valid length int32]) ->
// (out [1, H, n, Dv], state'), see trtllm/gated_delta.h.
struct GatedDeltaUpdateOp {
  LiteRtNvidiaGatedDeltaParams params{};
  bool has_valid_len = false;
};

bool IsGatedDeltaUpdateOp(const Op& op) {
  const auto code = op.CustomCode();
  return code && (*code == "gated_delta_update" ||
                  *code == "custom_call.gated_delta_update");
}

std::optional<GatedDeltaUpdateOp> ParseGatedDeltaUpdate(const Op& op) {
  if (!IsGatedDeltaUpdateOp(op)) {
    return std::nullopt;
  }
  if ((op.Inputs().size() != 6 && op.Inputs().size() != 7) ||
      op.Outputs().size() != 2) {
    return std::nullopt;
  }
  if (const auto options = op.CustomOptions(); options && !options->empty()) {
    const flexbuffers::Map map =
        flexbuffers::GetRoot(options->data(), options->size()).AsMap();
    if (!map["state_dtype"].IsNull() &&
        map["state_dtype"].AsString().str() != "float32") {
      return std::nullopt;
    }
  }
  std::array<std::vector<int32_t>, 6> dims;
  for (int i = 0; i < 6; ++i) {
    auto type = op.Inputs()[i].RankedTensorType();
    if (!type || !IsFloatLike(op.Inputs()[i].ElementType()) ||
        op.Inputs()[i].HasWeights()) {
      return std::nullopt;
    }
    const auto d = type->Layout().Dimensions();
    dims[i].assign(d.begin(), d.end());
  }
  const auto& q = dims[0];
  const auto& v = dims[2];
  const auto& state = dims[5];
  if (q.size() != 4 || dims[1] != q || v.size() != 4 || state.size() != 4 ||
      q[0] != 1 || v[0] != 1 || q[2] != v[2] || state[0] != 1 ||
      state[1] != v[1] || state[2] != q[3] || state[3] != v[3] ||
      dims[3] != std::vector<int32_t>{1, v[1], v[2]} ||
      dims[4] != std::vector<int32_t>{1, v[1], v[2]}) {
    return std::nullopt;
  }
  if (!IsInPlaceFp32State(op.Inputs()[5], op.Outputs()[1]) ||
      !TensorHasShape(op.Outputs()[0], {1, v[1], v[2], v[3]})) {
    return std::nullopt;
  }
  GatedDeltaUpdateOp result;
  result.params = {v[1], q[1], v[2], q[3], v[3]};
  if (op.Inputs().size() == 7) {
    auto type = op.Inputs()[6].RankedTensorType();
    int64_t count = 1;
    if (type) {
      for (int32_t d : type->Layout().Dimensions()) count *= d;
    }
    if (!type || count != 1 ||
        op.Inputs()[6].ElementType() != litert::ElementType::Int32 ||
        op.Inputs()[6].HasWeights()) {
      return std::nullopt;
    }
    result.has_valid_len = true;
  }
  if (!LiteRtNvidiaGatedDeltaSupports(&result.params)) return std::nullopt;
  return result;
}

// odml.short_conv_step(x [1, n, C], state [1, C, L - 1], weight [C, 1, L][,
// valid length int32]) -> (y [1, n, C], state'): the causal depthwise
// convolution of a Gated DeltaNet layer, attributes conv_L_cache (L),
// use_silu and is_gated (unsupported), see trtllm/gated_delta.h.
struct ShortConvStepOp {
  LiteRtNvidiaShortConvParams params{};
  bool has_valid_len = false;
};

std::optional<ShortConvStepOp> ParseShortConvStep(const Op& op) {
  if ((op.Inputs().size() != 3 && op.Inputs().size() != 4) ||
      op.Outputs().size() != 2) {
    return std::nullopt;
  }
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value()) {
    return std::nullopt;
  }
  const auto& map = options.attributes_map.value();
  if (map["conv_L_cache"].IsNull() ||
      (!map["is_gated"].IsNull() && map["is_gated"].AsBool())) {
    return std::nullopt;
  }
  const int32_t taps = map["conv_L_cache"].AsInt32();
  const bool silu = map["use_silu"].IsNull() || map["use_silu"].AsBool();
  auto x_type = op.Inputs()[0].RankedTensorType();
  if (!x_type || x_type->Layout().Rank() != 3 ||
      !IsFloatLike(op.Inputs()[0].ElementType()) ||
      op.Inputs()[0].HasWeights()) {
    return std::nullopt;
  }
  const auto x = x_type->Layout().Dimensions();
  const int32_t tokens = x[1];
  const int32_t channels = x[2];
  if (x[0] != 1 || taps < 2 ||
      !TensorHasShape(op.Inputs()[1], {1, channels, taps - 1}) ||
      !IsInPlaceFp32State(op.Inputs()[1], op.Outputs()[1]) ||
      !TensorHasShape(op.Inputs()[2], {channels, 1, taps}) ||
      !IsUnquantizedFloat32(op.Inputs()[2]) || !op.Inputs()[2].HasWeights() ||
      op.Inputs()[2].Weights().Bytes().size() !=
          static_cast<size_t>(channels) * taps * sizeof(float) ||
      !TensorHasShape(op.Outputs()[0], {1, tokens, channels})) {
    return std::nullopt;
  }
  ShortConvStepOp result;
  result.params = {channels, tokens, taps, silu ? 1 : 0};
  if (op.Inputs().size() == 4) {
    auto type = op.Inputs()[3].RankedTensorType();
    int64_t count = 1;
    if (type) {
      for (int32_t d : type->Layout().Dimensions()) count *= d;
    }
    if (!type || count != 1 ||
        op.Inputs()[3].ElementType() != litert::ElementType::Int32 ||
        op.Inputs()[3].HasWeights()) {
      return std::nullopt;
    }
    result.has_valid_len = true;
  }
  if (!LiteRtNvidiaShortConvSupports(&result.params)) return std::nullopt;
  return result;
}

// The element count of a tensor of static shape, or -1.
int64_t StaticElementCount(const Tensor& tensor) {
  auto type = tensor.RankedTensorType();
  if (!type) return -1;
  int64_t count = 1;
  for (int32_t d : type->Layout().Dimensions()) {
    if (d <= 0) return -1;
    count *= d;
  }
  return count;
}

bool IsUnquantizedFloat32Or16(const Tensor& tensor) {
  return (tensor.ElementType() == litert::ElementType::Float32 ||
          tensor.ElementType() == litert::ElementType::Float16) &&
         tensor.QTypeId() == kLiteRtQuantizationNone;
}

// A state the QSA indexer updates in place: a float subgraph input whose
// updated value is a subgraph output read by nothing else (the runtime binds
// both to one buffer, as for the linear-attention states).
bool IsInPlaceQsaState(const Tensor& input, const Tensor& output) {
  return input.IsSubgraphInput() && IsUnquantizedFloat32Or16(input) &&
         output.ElementType() == input.ElementType() &&
         output.QTypeId() == kLiteRtQuantizationNone && output.Uses().empty() &&
         input.Uses().size() == 1;
}

// A float operand the QSA plugins take as FP32: a constant, or a runtime
// tensor.
bool IsQsaFloatOperand(const Tensor& tensor) {
  return IsFloatLike(tensor.ElementType()) &&
         tensor.QTypeId() == kLiteRtQuantizationNone &&
         (!tensor.HasWeights() ||
          tensor.ElementType() != litert::ElementType::BFloat16);
}

// The mask of the QSA composites: [1, 1, rows, len] bool, or float
// (additive).
bool IsQsaMask(const Tensor& mask, int32_t rows, int32_t len) {
  return (mask.ElementType() == litert::ElementType::Bool ||
          (IsFloatLike(mask.ElementType()) &&
           mask.QTypeId() == kLiteRtQuantizationNone)) &&
         TensorHasShape(mask, {1, 1, rows, len});
}

// The compress ratio of the QSA composites is 4 (the only one the composites
// define); returns false for any other.
bool QsaCompressRatioIsFour(const flexbuffers::Map& map) {
  return map["compress_ratio"].IsNull() || map["compress_ratio"].AsInt32() == 4;
}

// odml.qsa_indexer_update, Qwen 4's sparse-attention indexer (q_idx [1, S, H,
// D], k_idx_new [1, S, D], ring_k [1, 4, D], block_k_pool [1, B, D],
// k_norm_weight [D], block_cos and block_sin [1, B, R], start_pos (int32, one
// value), mask [1, 1, S, L]) -> (topk_block_indices [1, S, block_topk] int32,
// ring_k', block_k_pool'): trtllm/qsa_indexer.h. Attributes compress_ratio
// (4), block_topk (512), rms_norm_eps (1e-6) and scale (1). The ring and the
// pool are in-place states (IsInPlaceQsaState).
struct QsaIndexerOp {
  LiteRtNvidiaQsaIndexerParams params{};
};

std::optional<QsaIndexerOp> ParseQsaIndexerUpdate(const Op& op) {
  if (op.Inputs().size() != 9 || op.Outputs().size() != 3) {
    return std::nullopt;
  }
  QsaIndexerOp result;
  auto& p = result.params;
  p.block_topk = 512;
  p.rms_norm_eps = 1e-6f;
  p.scale = 1.0f;
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) == kLiteRtStatusOk &&
      options.attributes_map.has_value()) {
    const auto& map = options.attributes_map.value();
    if (!QsaCompressRatioIsFour(map)) return std::nullopt;
    if (!map["block_topk"].IsNull()) p.block_topk = map["block_topk"].AsInt32();
    if (!map["rms_norm_eps"].IsNull()) {
      p.rms_norm_eps = map["rms_norm_eps"].AsFloat();
    }
    if (!map["scale"].IsNull()) p.scale = map["scale"].AsFloat();
  }
  const auto& in = op.Inputs();
  auto q_type = in[0].RankedTensorType();
  auto pool_type = in[3].RankedTensorType();
  auto cos_type = in[5].RankedTensorType();
  auto mask_type = in[8].RankedTensorType();
  if (!q_type || !pool_type || !cos_type || !mask_type ||
      q_type->Layout().Rank() != 4 || pool_type->Layout().Rank() != 3 ||
      cos_type->Layout().Rank() != 3 || mask_type->Layout().Rank() != 4) {
    return std::nullopt;
  }
  const auto q = q_type->Layout().Dimensions();
  const int32_t rows = q[1];
  const int32_t dim = q[3];
  const int32_t blocks = pool_type->Layout().Dimensions()[1];
  const int32_t rotary = cos_type->Layout().Dimensions()[2];
  const int32_t len = mask_type->Layout().Dimensions()[3];
  if (q[0] != 1 || in[0].HasWeights() || !IsFloatLike(in[0].ElementType()) ||
      in[0].QTypeId() != kLiteRtQuantizationNone ||
      in[1].ElementType() != in[0].ElementType() || in[1].HasWeights() ||
      !TensorHasShape(in[1], {1, rows, dim}) ||
      !TensorHasShape(in[2], {1, 4, dim}) ||
      !TensorHasShape(in[3], {1, blocks, dim}) ||
      in[3].ElementType() != in[2].ElementType() ||
      !IsInPlaceQsaState(in[2], op.Outputs()[1]) ||
      !IsInPlaceQsaState(in[3], op.Outputs()[2]) ||
      !TensorHasShape(op.Outputs()[1], {1, 4, dim}) ||
      !TensorHasShape(op.Outputs()[2], {1, blocks, dim}) ||
      !TensorHasShape(in[4], {dim}) || !IsQsaFloatOperand(in[4]) ||
      !TensorHasShape(in[5], {1, blocks, rotary}) ||
      !TensorHasShape(in[6], {1, blocks, rotary}) ||
      !IsQsaFloatOperand(in[5]) || !IsQsaFloatOperand(in[6]) ||
      in[7].ElementType() != litert::ElementType::Int32 ||
      StaticElementCount(in[7]) != 1 || !IsQsaMask(in[8], rows, len) ||
      op.Outputs()[0].ElementType() != litert::ElementType::Int32 ||
      !TensorHasShape(op.Outputs()[0], {1, rows, p.block_topk})) {
    return std::nullopt;
  }
  p.rows = rows;
  p.index_heads = q[2];
  p.index_dim = dim;
  p.rotary_dim = rotary;
  p.max_blocks = blocks;
  p.mask_len = len;
  // The types the plugin sees: the activation type of q (any float), the
  // model's state type, a bool or FP32 mask.
  p.q_type = kLiteRtNvidiaQsaFloat32;
  p.state_type = in[2].ElementType() == litert::ElementType::Float16
                     ? kLiteRtNvidiaQsaFloat16
                     : kLiteRtNvidiaQsaFloat32;
  p.mask_type = in[8].ElementType() == litert::ElementType::Bool
                    ? kLiteRtNvidiaQsaMaskBool
                    : kLiteRtNvidiaQsaMaskFloat;
  if (!LiteRtNvidiaQsaIndexerSupports(&p)) return std::nullopt;
  return result;
}

// odml.sparse_sdpa_transposed, Qwen 4's sparse attention (query [1, Hq, S, D],
// key_cache [1, H, L, D], value_cache [1, H, D, L], topk_block_indices
// [1, S, block_topk] int32, mask [1, 1, S, L]) -> [1, S, Hq, D]:
// trtllm/qsa_sparse_attention.h. Attributes compress_ratio (4), block_topk,
// scale (1), k_ts_idx (2) and v_ts_idx (3). The caches are the results of an
// in-place odml.cache_update or inputs of the graph.
struct SparseSdpaOp {
  LiteRtNvidiaQsaAttentionParams params{};
};

std::optional<SparseSdpaOp> ParseSparseSdpaTransposed(const Op& op) {
  if (op.Inputs().size() != 5 || op.Outputs().size() != 1) {
    return std::nullopt;
  }
  SparseSdpaOp result;
  auto& p = result.params;
  p.scale = 1.0f;
  int32_t block_topk = 512;
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) == kLiteRtStatusOk &&
      options.attributes_map.has_value()) {
    const auto& map = options.attributes_map.value();
    if (!QsaCompressRatioIsFour(map) ||
        (!map["k_ts_idx"].IsNull() && map["k_ts_idx"].AsInt32() != 2) ||
        (!map["v_ts_idx"].IsNull() && map["v_ts_idx"].AsInt32() != 3)) {
      return std::nullopt;
    }
    if (!map["block_topk"].IsNull()) block_topk = map["block_topk"].AsInt32();
    if (!map["scale"].IsNull()) p.scale = map["scale"].AsFloat();
  }
  const auto& in = op.Inputs();
  const auto qd = StaticRank4Dims(in[0]);
  const auto kd = StaticRank4Dims(in[1]);
  if (!qd || !kd || !IsFloatLike(in[0].ElementType()) ||
      in[0].QTypeId() != kLiteRtQuantizationNone || in[0].HasWeights()) {
    return std::nullopt;
  }
  const int32_t heads = (*qd)[1], rows = (*qd)[2], dim = (*qd)[3];
  const int32_t kv_heads = (*kd)[1], len = (*kd)[2];
  if ((*qd)[0] != 1 || (*kd)[0] != 1 || (*kd)[3] != dim ||
      !IsFp16StoredCacheType(in[1].ElementType()) ||
      in[1].QTypeId() != kLiteRtQuantizationNone ||
      in[2].ElementType() != in[1].ElementType() ||
      !TensorHasShape(in[2], {1, kv_heads, dim, len}) ||
      in[3].ElementType() != litert::ElementType::Int32 ||
      !TensorHasShape(in[3], {1, rows, block_topk}) ||
      !IsQsaMask(in[4], rows, len) ||
      !IsFloatLike(op.Outputs()[0].ElementType()) ||
      !TensorHasShape(op.Outputs()[0], {1, rows, heads, dim})) {
    return std::nullopt;
  }
  p.rows = rows;
  p.q_heads = heads;
  p.kv_heads = kv_heads;
  p.head_dim = dim;
  p.cache_len = len;
  p.block_topk = block_topk;
  p.q_type = kLiteRtNvidiaQsaFloat32;
  p.cache_type = kLiteRtNvidiaQsaFloat16;
  p.mask_type = in[4].ElementType() == litert::ElementType::Bool
                    ? kLiteRtNvidiaQsaMaskBool
                    : kLiteRtNvidiaQsaMaskFloat;
  if (!LiteRtNvidiaQsaAttentionSupports(&p)) return std::nullopt;
  // The caches are what an in-place cache update wrote, or inputs of the
  // graph.
  for (int i : {1, 2}) {
    if (in[i].IsSubgraphInput()) continue;
    const auto producer = in[i].GetDefiningOp();
    if (!producer || producer->Code() != kLiteRtOpCodeShloComposite ||
        CompositeOpName(*producer) != "odml.cache_update" ||
        !DecodeCacheUpdateMode(*producer).has_value()) {
      return std::nullopt;
    }
  }
  return result;
}

// The experts stay in the model file until the engine runs: the plugin needs
// the file the weight store maps (LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH).
Expected<bool> IsCustomOpSupported(const Op& op) {
  if (ParseGatedDeltaUpdate(op).has_value()) {
    return true;
  }
  if (!ParseMoeCustomOp(op).has_value()) {
    return false;
  }
  const char* model_path = std::getenv("LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH");
  if (model_path == nullptr || model_path[0] == '\0') {
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX: the moe op needs the model path "
               "(LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH) for its experts");
    return false;
  }
  return BlockwiseFcPathEnabled();
}

// The GPU cache composite has a different contract from its fallback DUS:
// params[0] is the write position and params[3] the number of valid update
// rows. Ring caches wrap; padded prefill rows must never enter the cache.
std::optional<bool> PrefillCacheUpdateRingMode(const Op& op) {
  if (!NativeKvCacheUpdateEnabled() || op.Inputs().size() != 7 ||
      op.Outputs().size() != 2) {
    return std::nullopt;
  }
  // Prefill attends to the old cache and fresh K/V separately, then exports
  // the updated caches. Keep decode/verify's updated-cache reader lowering
  // unchanged: preparing a full local ring for every token would be wasteful.
  if (!op.Outputs()[0].Uses().empty() || !op.Outputs()[1].Uses().empty()) {
    return std::nullopt;
  }
  // Their complete result tensors also provide execution-order dependencies
  // before the in-place write. A view/reshape of the cache is not a completed
  // read and cannot serve as that dependency.
  for (int i : {3, 4}) {
    for (const auto& use : op.Inputs()[i].Uses()) {
      if (use.user.Get() == op.Get()) continue;
      if (use.user_arg_ind != 1 ||
          (use.user.Code() != kLiteRtOpCodeTflBatchMatmul &&
           !(use.user.Code() == kLiteRtOpCodeShloComposite &&
             CompositeOpName(use.user) == "odml.runtime_bmm"))) {
        return std::nullopt;
      }
    }
  }
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value()) {
    return std::nullopt;
  }
  const auto ring = options.attributes_map.value()["is_ring_buffer"];
  if (!ring.IsBool()) {
    return std::nullopt;
  }
  std::array<std::array<int32_t, 4>, 5> dims;
  for (int i = 0; i < 5; ++i) {
    auto type = op.Inputs()[i].RankedTensorType();
    if (!type || type->Layout().Rank() != 4 ||
        (i == 2 ? op.Inputs()[i].ElementType() != litert::ElementType::Int32
                : (i < 2 ? !IsFloatLike(op.Inputs()[i].ElementType())
                         : !IsFp16StoredCacheType(
                               op.Inputs()[i].ElementType())))) {
      return std::nullopt;
    }
    const auto shape = type->Layout().Dimensions();
    std::copy(shape.begin(), shape.end(), dims[i].begin());
    for (int d : dims[i]) {
      if (d <= 0) return std::nullopt;
    }
  }
  const auto& k = dims[3];
  const auto& v = dims[4];
  if (k[0] != 1 || v[0] != 1 || v[1] != k[1] || v[2] != k[3] || v[3] != k[2] ||
      dims[2][0] != 1 || dims[2][1] != 1 || dims[2][2] != 1 ||
      dims[2][3] != 7) {
    return std::nullopt;
  }
  for (int i = 0; i < 2; ++i) {
    if (dims[i][0] != 1 || dims[i][1] != k[1] || dims[i][2] <= 1 ||
        dims[i][2] > k[2] || dims[i][3] != k[3] ||
        !std::equal(dims[0].begin(), dims[0].end(), dims[1].begin()) ||
        !op.Inputs()[3 + i].IsSubgraphInput()) {
      return std::nullopt;
    }
    auto output_type = op.Outputs()[i].RankedTensorType();
    if (!output_type ||
        output_type->ElementType() != op.Inputs()[3 + i].ElementType() ||
        output_type->Layout().Rank() != 4 ||
        !std::equal(dims[3 + i].begin(), dims[3 + i].end(),
                    output_type->Layout().Dimensions().begin())) {
      return std::nullopt;
    }
  }
  return ring.AsBool();
}

bool HasNoFusedActivation(const Op& op);

// Statically quantized products (trtllm/quant_linear.h): INT8 activations times
// INT4 per-channel or centered INT2 weights give an INT8 result, rounded as
// LiteRT's CPU kernels round it. They take the quant plugin (FindQuantRegions),
// which needs the FP16-activation mode around it.
bool QuantProductsEnabled() { return Fp16ActivationsEnabled(); }

bool IsSymmetricInt8(const Tensor& tensor) {
  if (tensor.ElementType() != litert::ElementType::Int8 ||
      tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
    return false;
  }
  const auto q = tensor.PerTensorQuantization();
  return q.scale > 0.0f && q.zero_point == 0;
}

// The FP32 scales of centered INT2 weights, one per output of the product:
// INT2 values with blockwise quantization whose scale and zero point tensors
// have the weights' dims with 1 at `input_axis` (a scale per output, shared
// by its inputs), every zero point -0.5: a weight is scale * (q + 0.5).
std::optional<std::vector<float>> CenteredInt2Scales(const Tensor& weights,
                                                     size_t input_axis) {
  if (weights.ElementType() != litert::ElementType::Int2 ||
      weights.QTypeId() != kLiteRtQuantizationBlockWise ||
      !weights.HasWeights()) {
    return std::nullopt;
  }
  const auto q = weights.BlockWiseQuantization();
  const auto type = weights.RankedTensorType();
  if (q.scales == nullptr || q.zero_points == nullptr || !type) {
    return std::nullopt;
  }
  const auto dims = type->Layout().Dimensions();
  if (input_axis >= dims.size()) {
    return std::nullopt;
  }
  size_t elements = 1;
  size_t outputs = 1;
  for (size_t d = 0; d < dims.size(); ++d) {
    if (dims[d] <= 0) return std::nullopt;
    elements *= dims[d];
    if (d != input_axis) outputs *= dims[d];
  }
  if (weights.Weights().Bytes().size() != (elements + 3) / 4) {
    return std::nullopt;
  }
  const auto read = [&](LiteRtTensor raw) -> std::optional<std::vector<float>> {
    const Tensor tensor(weights.Context(), raw);
    const auto tensor_type = tensor.RankedTensorType();
    if (!tensor_type || tensor.ElementType() != litert::ElementType::Float32 ||
        !tensor.HasWeights()) {
      return std::nullopt;
    }
    const auto tensor_dims = tensor_type->Layout().Dimensions();
    if (tensor_dims.size() != dims.size()) return std::nullopt;
    for (size_t d = 0; d < dims.size(); ++d) {
      if (tensor_dims[d] != (d == input_axis ? 1 : dims[d])) {
        return std::nullopt;
      }
    }
    const auto bytes = tensor.Weights().Bytes();
    if (bytes.size() != outputs * sizeof(float)) return std::nullopt;
    std::vector<float> values(outputs);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
  };
  const auto zero_points = read(q.zero_points);
  if (!zero_points) return std::nullopt;
  for (const float zero_point : *zero_points) {
    if (zero_point != -0.5f) return std::nullopt;
  }
  auto scales = read(q.scales);
  if (!scales) return std::nullopt;
  for (const float scale : *scales) {
    if (!(scale > 0.0f) || !std::isfinite(scale)) return std::nullopt;
  }
  return scales;
}

// INT4 weights [rows, columns] with a symmetric scale per row.
bool IsPerRowInt4(const Tensor& weights, int32_t rows, int32_t columns) {
  if (weights.ElementType() != litert::ElementType::Int4 ||
      weights.QTypeId() != kLiteRtQuantizationPerChannel ||
      !weights.HasWeights()) {
    return false;
  }
  const auto q = weights.PerChannelQuantization();
  if (q.quantized_dimension != 0 ||
      q.num_channels != static_cast<uint64_t>(rows) || q.scales == nullptr ||
      q.zero_points == nullptr ||
      weights.Weights().Bytes().size() !=
          static_cast<size_t>(rows) * columns / 2) {
    return false;
  }
  for (int32_t r = 0; r < rows; ++r) {
    if (!(q.scales[r] > 0.0f) || q.zero_points[r] != 0) return false;
  }
  return true;
}

// The [rows, columns] of the weights of a FULLY_CONNECTED the quant kernels
// take (whole CTAs of eight rows, whole 32-bit words per row), without bias
// and fused activation.
std::optional<std::pair<int32_t, int32_t>> QuantFullyConnectedShape(
    const Op& op) {
  if (!QuantProductsEnabled() || op.Code() != kLiteRtOpCodeTflFullyConnected ||
      op.Inputs().size() != 2 || op.Outputs().size() != 1 ||
      op.Inputs()[0].HasWeights() || !HasNoFusedActivation(op)) {
    return std::nullopt;
  }
  uint32_t format = litert::kFullyConnectedOptionsWeightsFormatDefault;
  const auto type = op.Inputs()[1].RankedTensorType();
  if (LiteRtGetFullyConnectedWeightsFormatOption(op.Get(), &format) !=
          kLiteRtStatusOk ||
      format != litert::kFullyConnectedOptionsWeightsFormatDefault || !type ||
      type->Layout().Rank() != 2) {
    return std::nullopt;
  }
  const auto dims = type->Layout().Dimensions();
  if (dims[0] <= 0 || dims[0] % 8 != 0 || dims[1] <= 0 || dims[1] % 16 != 0) {
    return std::nullopt;
  }
  return std::make_pair(dims[0], dims[1]);
}

// Whether the INT8 tensors of the product `op` are the quantized form of
// FLOAT16 ones: the QUANTIZE that makes its input (through reshapes and
// transposes) reads FLOAT16, or the DEQUANTIZE behind its result (through
// reshapes) makes FLOAT16, and neither has another float type. A statically
// quantized model's products are; the Gemma 4 E2B and E4B have INT4 products
// between FLOAT32 tensors, which keep the FP16-activation lowering they run on
// (the quant kernels round as the CPU path rounds FLOAT16).
bool IsBetweenFloat16(const Op& op) {
  bool found = false;
  Tensor input = op.Inputs()[0];
  for (int hops = 0; hops < 6; ++hops) {
    auto producer = input.GetDefiningOp();
    if (!producer || producer->Inputs().empty()) break;
    if (producer->Code() == kLiteRtOpCodeTflQuantize) {
      if (producer->Inputs()[0].ElementType() != litert::ElementType::Float16) {
        return false;
      }
      found = true;
      break;
    }
    if (producer->Code() != kLiteRtOpCodeTflReshape &&
        producer->Code() != kLiteRtOpCodeTflTranspose) {
      break;
    }
    input = producer->Inputs()[0];
  }
  Tensor output = op.Outputs()[0];
  for (int hops = 0; hops < 3; ++hops) {
    const auto uses = output.Uses();
    if (uses.size() != 1 || uses[0].user.Outputs().size() != 1) break;
    const Op user = uses[0].user;
    if (user.Code() == kLiteRtOpCodeTflDequantize) {
      if (user.Outputs()[0].ElementType() != litert::ElementType::Float16) {
        return false;
      }
      found = true;
      break;
    }
    if (user.Code() != kLiteRtOpCodeTflReshape) break;
    output = user.Outputs()[0];
  }
  return found;
}

// FULLY_CONNECTED of INT8 activations to an INT8 result with centered INT2
// weights, or with INT4 per-channel weights between FLOAT16 tensors.
bool IsQuantFullyConnected(const Op& op) {
  const auto shape = QuantFullyConnectedShape(op);
  if (!shape || !IsSymmetricInt8(op.Inputs()[0]) ||
      !IsSymmetricInt8(op.Outputs()[0])) {
    return false;
  }
  return (IsPerRowInt4(op.Inputs()[1], shape->first, shape->second) &&
          IsBetweenFloat16(op)) ||
         CenteredInt2Scales(op.Inputs()[1], 1).has_value();
}

// FULLY_CONNECTED of FLOAT16 activations to FLOAT16 logits with centered
// INT2 weights (TFLite's dynamic range contract): the vocabulary head.
bool IsQuantLogitsFullyConnected(const Op& op) {
  const auto shape = QuantFullyConnectedShape(op);
  return shape.has_value() &&
         op.Inputs()[0].ElementType() == litert::ElementType::Float16 &&
         op.Outputs()[0].ElementType() == litert::ElementType::Float16 &&
         CenteredInt2Scales(op.Inputs()[1], 1).has_value();
}

// BATCH_MATMUL of INT8 rows [.., blocks, 1, columns] with constant centered
// INT2 weights [blocks, columns, outputs] (a leading 1 allowed) to INT8.
bool IsQuantBatchMatmul(const Op& op) {
  if (!QuantProductsEnabled() || op.Code() != kLiteRtOpCodeTflBatchMatmul ||
      op.Inputs().size() != 2 || op.Outputs().size() != 1 ||
      op.Inputs()[0].HasWeights() || !IsSymmetricInt8(op.Inputs()[0]) ||
      !IsSymmetricInt8(op.Outputs()[0])) {
    return false;
  }
  bool adj_x = false;
  bool adj_y = false;
  const auto lhs_type = op.Inputs()[0].RankedTensorType();
  const auto rhs_type = op.Inputs()[1].RankedTensorType();
  if (LiteRtGetBatchMatmulAdjXOption(op.Get(), &adj_x) != kLiteRtStatusOk ||
      LiteRtGetBatchMatmulAdjYOption(op.Get(), &adj_y) != kLiteRtStatusOk ||
      adj_x || adj_y || !lhs_type || !rhs_type) {
    return false;
  }
  const auto lhs = lhs_type->Layout().Dimensions();
  const auto rhs = rhs_type->Layout().Dimensions();
  const size_t rank = rhs.size();
  if ((rank != 3 && !(rank == 4 && rhs[0] == 1)) || lhs.size() != 4 ||
      lhs[1] != rhs[rank - 3] || lhs[2] != 1 || lhs[3] != rhs[rank - 2] ||
      rhs[rank - 1] % 16 != 0) {
    return false;
  }
  return CenteredInt2Scales(op.Inputs()[1], rank - 2).has_value();
}

// odml.rope(x FLOAT16 [1, heads, tokens, depth], positions INT32 [1, tokens])
// as a statically quantized model has it behind its q and k head norms
// (LiteRtNvidiaLaunchQuantQkvRope).
struct QuantRope {
  int32_t heads = 0;
  int32_t tokens = 0;
  int32_t depth = 0;
  float min_timescale = 1.0f;
  float max_timescale = 10000.0f;
  float proportion = 1.0f;
};

std::optional<QuantRope> ParseQuantRope(const Op& op) {
  if (!QuantProductsEnabled() || op.Code() != kLiteRtOpCodeShloComposite ||
      CompositeOpName(op) != "odml.rope" || op.Inputs().size() != 2 ||
      op.Outputs().size() != 1 ||
      op.Inputs()[0].ElementType() != litert::ElementType::Float16 ||
      op.Outputs()[0].ElementType() != litert::ElementType::Float16 ||
      op.Inputs()[0].HasWeights() ||
      op.Inputs()[1].ElementType() != litert::ElementType::Int32) {
    return std::nullopt;
  }
  const auto type = op.Inputs()[0].RankedTensorType();
  const auto positions = op.Inputs()[1].RankedTensorType();
  if (!type || !positions || type->Layout().Rank() != 4) {
    return std::nullopt;
  }
  const auto dims = type->Layout().Dimensions();
  const auto position_count = NumElements(*positions);
  QuantRope rope;
  rope.heads = dims[1];
  rope.tokens = dims[2];
  rope.depth = dims[3];
  if (dims[0] != 1 || rope.heads <= 0 || rope.tokens <= 0 || rope.depth < 2 ||
      rope.depth > 1024 || rope.depth % 2 != 0 || !position_count ||
      *position_count != static_cast<size_t>(rope.tokens) ||
      !TensorHasShape(op.Outputs()[0],
                      {1, rope.heads, rope.tokens, rope.depth})) {
    return std::nullopt;
  }
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value()) {
    return std::nullopt;
  }
  auto& attributes = options.attributes_map.value();
  if (!attributes["min_timescale"].IsNull()) {
    rope.min_timescale = attributes["min_timescale"].AsFloat();
  }
  if (!attributes["max_timescale"].IsNull()) {
    rope.max_timescale = attributes["max_timescale"].AsFloat();
  }
  if (!attributes["proportion"].IsNull()) {
    rope.proportion = attributes["proportion"].AsFloat();
  }
  if (!(rope.min_timescale > 0.0f) ||
      !(rope.max_timescale >= rope.min_timescale) ||
      !(rope.proportion > 0.0f) || rope.proportion > 1.0f) {
    return std::nullopt;
  }
  return rope;
}

// Whether the odml.rope `op` rotates the head norm of a quant projection, as
// MatchQuantQkvRope takes it: (TRANSPOSE | RESHAPE) <- CAST <- odml.rms_norm
// <- CAST <- DEQUANTIZE <- RESHAPE <- FULLY_CONNECTED. No other odml.rope
// has a lowering (the Gemma 4 26B's stays on the CPU).
bool IsRopeOfQuantProjection(const Op& op) {
  Tensor value = op.Inputs()[0];
  const auto producer =
      [&](std::initializer_list<LiteRtOpCode> codes) -> std::optional<Op> {
    auto defining = value.GetDefiningOp();
    if (!defining || defining->Inputs().empty() ||
        std::find(codes.begin(), codes.end(), defining->Code()) ==
            codes.end()) {
      return std::nullopt;
    }
    value = defining->Inputs()[0];
    return *defining;
  };
  if (!producer({kLiteRtOpCodeTflTranspose, kLiteRtOpCodeTflReshape}) ||
      !producer({kLiteRtOpCodeTflCast})) {
    return false;
  }
  const auto norm = producer({kLiteRtOpCodeShloComposite});
  if (!norm || CompositeOpName(*norm) != "odml.rms_norm" ||
      !producer({kLiteRtOpCodeTflCast}) ||
      !producer({kLiteRtOpCodeTflDequantize}) ||
      !producer({kLiteRtOpCodeTflReshape})) {
    return false;
  }
  const auto product = value.GetDefiningOp();
  return product && IsQuantFullyConnected(*product);
}

// odml.factorized_mlp(x FLOAT16 [.., 2 C], six centered INT2 weights) -> y
// FLOAT16: a factorized MLP carved into one composite (tools/carve), whose
// attributes carry the INT8 scales of x and of the stages' results in order
// (QuantFactorizedMlp::scales). The weights are up-1 a and b, up-2, down-1 and
// down-2 a and b.
struct QuantFactorizedMlpComposite {
  float scales[6] = {};
};

std::optional<QuantFactorizedMlpComposite> ParseQuantFactorizedMlpComposite(
    const Op& op) {
  if (!QuantProductsEnabled() || op.Code() != kLiteRtOpCodeShloComposite ||
      CompositeOpName(op) != "odml.factorized_mlp" || op.Inputs().size() != 7 ||
      op.Outputs().size() != 1 || op.Inputs()[0].HasWeights() ||
      op.Inputs()[0].ElementType() != litert::ElementType::Float16 ||
      op.Outputs()[0].ElementType() != litert::ElementType::Float16) {
    return std::nullopt;
  }
  for (int i = 1; i <= 6; ++i) {
    const auto type = op.Inputs()[i].RankedTensorType();
    if (!type) return std::nullopt;
    const size_t rank = type->Layout().Rank();
    const bool blocks = i == 3 || i == 4;
    if ((blocks ? rank < 3 : rank != 2) ||
        !CenteredInt2Scales(op.Inputs()[i], blocks ? rank - 2 : 1)) {
      return std::nullopt;
    }
  }
  CompositeOptions options;
  if (options.InitFromOp(op.Get()) != kLiteRtStatusOk ||
      !options.attributes_map.has_value()) {
    return std::nullopt;
  }
  auto& attributes = options.attributes_map.value();
  constexpr const char* kScales[6] = {"scale_in",  "scale_up1",   "scale_up2",
                                      "scale_act", "scale_down1", "scale_out"};
  QuantFactorizedMlpComposite parsed;
  for (int i = 0; i < 6; ++i) {
    if (attributes[kScales[i]].IsNull()) return std::nullopt;
    parsed.scales[i] = attributes[kScales[i]].AsFloat();
    if (!(parsed.scales[i] > 0.0f)) return std::nullopt;
  }
  // The kernels' table between the middle stages is the SiLU.
  if (!attributes["activation"].IsNull() &&
      std::strcmp(attributes["activation"].AsString().c_str(), "silu") != 0) {
    return std::nullopt;
  }
  return parsed;
}

Expected<bool> IsCompositeSupported(const Op& op) {
  const std::string name = CompositeOpName(op);
  if (name.empty() || !NativeCompositeEnabled(name)) {
    return false;
  }
  if (name == "odml.rms_norm") {
    if (op.Inputs().size() < 1 || op.Inputs().size() > 2 ||
        op.Outputs().size() != 1) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(bool x_ok, IsStaticFloatTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(bool out_ok, IsStaticFloatTensor(op.Outputs()[0]));
    if (!x_ok || !out_ok) {
      return false;
    }
    if (op.Inputs().size() == 2) {
      LITERT_ASSIGN_OR_RETURN(bool w_ok, IsStaticFloatTensor(op.Inputs()[1]));
      if (!w_ok) {
        return false;
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto x_type, op.Inputs()[0].RankedTensorType());
    return x_type.Layout().Rank() >= 1;
  }
  if (name == "odml.cache_update") {
    if (PrefillCacheUpdateRingMode(op).has_value() ||
        DecodeCacheUpdateMode(op).has_value() ||
        ParseQuantCacheUpdate(op).has_value() ||
        ParseGeneralCacheUpdate(op).has_value()) {
      return true;
    }
    // (update_a f32, update_b f32, _, cache_a i8, cache_b i8, start_a i32,
    //  start_b i32) -> (cache_a', cache_b'); see the odml decomposition.
    if (op.Inputs().size() < 7 || op.Outputs().size() != 2) {
      return false;
    }
    for (int i : {0, 1}) {
      LITERT_ASSIGN_OR_RETURN(bool ok, IsStaticFloatTensor(op.Inputs()[i]));
      if (!ok) {
        return false;
      }
    }
    for (int i : {3, 4}) {
      LITERT_ASSIGN_OR_RETURN(bool ok, IsStaticTensorRtTensor(op.Inputs()[i]));
      LITERT_ASSIGN_OR_RETURN(
          bool quant, HasSymmetricPerTensorInt8Quantization(op.Inputs()[i]));
      if (!ok || !quant) {
        return false;
      }
    }
    for (int i : {5, 6}) {
      if (op.Inputs()[i].ElementType() != litert::ElementType::Int32) {
        return false;
      }
    }
    for (int i : {0, 1}) {
      LITERT_ASSIGN_OR_RETURN(
          bool quant, HasSymmetricPerTensorInt8Quantization(op.Outputs()[i]));
      if (!quant) {
        return false;
      }
    }
    return true;
  }
  if (name == "odml.qkv_norm_rope") {
    return ParseQkvNormRope(op).has_value();
  }
  if (name == "odml.rope") {
    return ParseQuantRope(op).has_value() && IsRopeOfQuantProjection(op);
  }
  if (name == "odml.factorized_mlp") {
    return ParseQuantFactorizedMlpComposite(op).has_value();
  }
  if (name == "odml.short_conv_step") {
    return ParseShortConvStep(op).has_value();
  }
  if (name == "odml.swiglu") {
    return ParseSwiglu(op).has_value();
  }
  if (name == "odml.sdpa_transposed") {
    if (ParseQuantSdpa(op).has_value() || ParseGeneralSdpa(op).has_value()) {
      return true;
    }
    if (!SdpaTransposedGroup(op).has_value()) {
      return false;
    }
    // The caches it reads are the outputs of an in-place cache update, or
    // inputs of the graph: caches another engine keeps (a drafter reading
    // its base model's caches).
    for (int i : {1, 2}) {
      if (op.Inputs()[i].IsSubgraphInput()) {
        continue;
      }
      const auto producer = op.Inputs()[i].GetDefiningOp();
      if (!producer || producer->Code() != kLiteRtOpCodeShloComposite ||
          CompositeOpName(*producer) != "odml.cache_update" ||
          !DecodeCacheUpdateMode(*producer).has_value()) {
        return false;
      }
    }
    return true;
  }
  if (name == "odml.fused_sdpa_cache_update") {
    return ParseQuantFusedSdpa(op).has_value() ||
           ParseGeneralFusedSdpa(op).has_value();
  }
  if (name == "odml.qsa_indexer_update") {
    return ParseQsaIndexerUpdate(op).has_value();
  }
  if (name == "odml.sparse_sdpa_transposed") {
    return ParseSparseSdpaTransposed(op).has_value();
  }
  if (name == "odml.runtime_bmm") {
    // (a, cache, positions) -> a @ dequantize(cache)^T over the full
    // context. The decomposition computes only a sliding window and fills
    // the remainder with a placeholder, but every non-window position is
    // either masked downstream (score path) or multiplied by a zero
    // probability (probs x transposed-values path), so the full matmul is
    // equivalent.
    if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(bool a_ok, IsStaticFloatTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(bool out_ok, IsStaticFloatTensor(op.Outputs()[0]));
    LITERT_ASSIGN_OR_RETURN(bool cache_static,
                            IsStaticTensorRtTensor(op.Inputs()[1]));
    if (!a_ok || !out_ok || !cache_static) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(
        bool cache_quant,
        HasSymmetricPerTensorInt8Quantization(op.Inputs()[1]));
    if (!cache_quant && !IsFloatLike(op.Inputs()[1].ElementType())) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto a_type, op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto c_type, op.Inputs()[1].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto o_type, op.Outputs()[0].RankedTensorType());
    const auto a_dims = a_type.Layout().Dimensions();
    const auto c_dims = c_type.Layout().Dimensions();
    const auto o_dims = o_type.Layout().Dimensions();
    const int rank = a_type.Layout().Rank();
    if (rank < 2 || c_type.Layout().Rank() != rank ||
        o_type.Layout().Rank() != rank) {
      return false;
    }
    if (a_dims[rank - 1] != c_dims[rank - 1] ||
        o_dims[rank - 2] != a_dims[rank - 2] ||
        o_dims[rank - 1] != c_dims[rank - 2]) {
      return false;
    }
    return true;
  }
  return false;
}

Expected<bool> IsDynamicUpdateSliceSupported(const Op& op) {
  if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool operand_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool update_supported,
                          IsStaticTensorRtTensor(op.Inputs()[1]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!operand_supported || !update_supported || !output_supported ||
      !SameElementType(op.Inputs()[0], op.Inputs()[1]) ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  const auto element_type = op.Inputs()[0].ElementType();
  if (!IsFloatLike(element_type) && element_type != litert::ElementType::Int8 &&
      element_type != litert::ElementType::Int32) {
    return false;
  }
  if (op.Inputs()[2].ElementType() != litert::ElementType::Int32) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto operand_type, op.Inputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto update_type, op.Inputs()[1].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto indices_type, op.Inputs()[2].RankedTensorType());
  const int rank = operand_type.Layout().Rank();
  if (rank < 1 || rank > 4 || update_type.Layout().Rank() != rank ||
      output_type.Layout().Rank() != rank ||
      indices_type.Layout().Rank() != 1) {
    return false;
  }
  const auto operand_dims = operand_type.Layout().Dimensions();
  const auto update_dims = update_type.Layout().Dimensions();
  const auto output_dims = output_type.Layout().Dimensions();
  const auto indices_dims = indices_type.Layout().Dimensions();
  if (indices_dims[0] != rank || operand_dims != output_dims) {
    return false;
  }
  // The lowering scatters elements along a single partial axis; every other
  // axis must be written in full (its offset is then zero by construction).
  int partial_axes = 0;
  for (int i = 0; i < rank; ++i) {
    if (update_dims[i] > operand_dims[i]) {
      return false;
    }
    if (update_dims[i] < operand_dims[i]) {
      ++partial_axes;
    }
  }
  return partial_axes <= 1;
}

std::optional<float> ReadFloatScalar(const Tensor& tensor);

// ONE_HOT(indices, depth, on = 1, off = 0) along a new last axis whose only
// user is the left operand of a BATCH_MATMUL with a constant table [depth, C]
// or [1, depth, C] (FLOAT32, or INT8 with symmetric per-tensor scales or
// per-channel scales along C): the product is a lookup of the table's rows.
// The Gemma 4 vision export with an MTP drafter computes its patch position
// embeddings this way (two [2520, 10240] x [10240, 1152] products per
// 280-token image, which ran on the CPU; the earlier export used
// EMBEDDING_LOOKUP). Returns the BATCH_MATMUL.
std::optional<Op> OneHotTableLookup(const Op& one_hot) {
  if (one_hot.Code() != kLiteRtOpCodeTflOneHot ||
      one_hot.Inputs().size() != 4 || one_hot.Outputs().size() != 1) {
    return std::nullopt;
  }
  // Inputs() and Outputs() return copies: keep the tensors, not references.
  const Tensor indices = one_hot.Inputs()[0];
  const Tensor selection = one_hot.Outputs()[0];
  const auto indices_type = indices.RankedTensorType();
  const auto selection_type = selection.RankedTensorType();
  const auto depth = ReadInt32Constant(one_hot.Inputs()[1]);
  int32_t axis = 0;
  if (indices.HasWeights() ||
      indices.ElementType() != litert::ElementType::Int32 ||
      selection.ElementType() != litert::ElementType::Float32 ||
      !indices_type || !selection_type || !depth || depth->size() != 1 ||
      (*depth)[0] <= 0 || ReadFloatScalar(one_hot.Inputs()[2]) != 1.0f ||
      ReadFloatScalar(one_hot.Inputs()[3]) != 0.0f ||
      LiteRtGetOneHotAxisOption(one_hot.Get(), &axis) != kLiteRtStatusOk) {
    return std::nullopt;
  }
  const auto index_dims = indices_type->Layout().Dimensions();
  const int rank = static_cast<int>(index_dims.size());
  if ((axis != -1 && axis != rank) ||
      selection_type->Layout().Dimensions().size() !=
          static_cast<size_t>(rank) + 1) {
    return std::nullopt;
  }
  const auto uses = selection.Uses();
  if (uses.size() != 1 || uses[0].user_arg_ind != 0) {
    return std::nullopt;
  }
  const Op product = uses[0].user;
  bool adj_x = false;
  bool adj_y = false;
  if (product.Code() != kLiteRtOpCodeTflBatchMatmul ||
      product.Inputs().size() != 2 || product.Outputs().size() != 1 ||
      LiteRtGetBatchMatmulAdjXOption(product.Get(), &adj_x) !=
          kLiteRtStatusOk ||
      LiteRtGetBatchMatmulAdjYOption(product.Get(), &adj_y) !=
          kLiteRtStatusOk ||
      adj_x || adj_y) {
    return std::nullopt;
  }
  const Tensor table = product.Inputs()[1];
  const Tensor rows = product.Outputs()[0];
  const auto table_type = table.RankedTensorType();
  const auto rows_type = rows.RankedTensorType();
  if (!table.HasWeights() || !table_type || !rows_type ||
      rows.ElementType() != litert::ElementType::Float32) {
    return std::nullopt;
  }
  const auto table_dims = table_type->Layout().Dimensions();
  const int table_rank = static_cast<int>(table_dims.size());
  if ((table_rank != 2 && !(table_rank == 3 && table_dims[0] == 1)) ||
      table_dims[table_rank - 2] != (*depth)[0]) {
    return std::nullopt;
  }
  const int32_t columns = table_dims[table_rank - 1];
  const auto rows_dims = rows_type->Layout().Dimensions();
  if (rows_dims.size() != static_cast<size_t>(rank) + 1 ||
      rows_dims[rank] != columns) {
    return std::nullopt;
  }
  for (int i = 0; i < rank; ++i) {
    if (index_dims[i] <= 0 || rows_dims[i] != index_dims[i]) {
      return std::nullopt;
    }
  }
  if (table.ElementType() == litert::ElementType::Float32) {
    return table.QTypeId() == kLiteRtQuantizationNone
               ? std::optional<Op>(product)
               : std::nullopt;
  }
  if (table.ElementType() != litert::ElementType::Int8) {
    return std::nullopt;
  }
  auto symmetric =
      HasSymmetricTensorRtQuantization(table, /*allow_per_channel=*/true);
  if (!symmetric || !*symmetric) {
    return std::nullopt;
  }
  if (table.QTypeId() == kLiteRtQuantizationPerChannel) {
    const auto q = table.PerChannelQuantization();
    if (q.quantized_dimension != table_rank - 1 ||
        q.num_channels != static_cast<uint64_t>(columns)) {
      return std::nullopt;
    }
  }
  return product;
}

// The BATCH_MATMUL of a OneHotTableLookup: `op`'s left operand is the one-hot.
bool IsOneHotTableLookupProduct(const Op& op) {
  const auto selector = op.Inputs()[0].GetDefiningOp();
  if (!selector) {
    return false;
  }
  const auto product = OneHotTableLookup(*selector);
  return product && product->Get() == op.Get();
}

Expected<bool> IsBatchMatmulSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  // INT8 rows times centered INT2 weights: a product of the quant plugin.
  if (IsQuantBatchMatmul(op)) {
    return true;
  }
  if (IsOneHotTableLookupProduct(op)) {
    return true;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticFloatTensor(tensor));
    if (!supported) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    if (type.Layout().Rank() < 2) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  if (!output_supported) {
    return false;
  }
  return true;
}

Expected<bool> IsFullyConnectedSupported(const Op& op) {
  if (op.Inputs().size() < 2 || op.Inputs().size() > 3 ||
      op.Outputs().size() != 1) {
    LogFullyConnectedReject(op, "bad input/output count");
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported) {
    LogFullyConnectedReject(op,
                            "input or output is not static TensorRT tensor");
    return false;
  }
  const bool input_float = IsFloatLike(op.Inputs()[0].ElementType());
  const bool output_float = IsFloatLike(op.Outputs()[0].ElementType());
  bool input_quantized = false;
  bool output_quantized = false;
  if (op.Inputs()[0].ElementType() == litert::ElementType::Int8) {
    LITERT_ASSIGN_OR_RETURN(input_quantized, HasSymmetricTensorRtQuantization(
                                                 op.Inputs()[0],
                                                 /*allow_per_channel=*/false));
  }
  if (op.Outputs()[0].ElementType() == litert::ElementType::Int8) {
    LITERT_ASSIGN_OR_RETURN(output_quantized, HasSymmetricTensorRtQuantization(
                                                  op.Outputs()[0],
                                                  /*allow_per_channel=*/false));
  }
  if (!((input_float && output_float) ||
        (input_quantized && output_quantized))) {
    LogFullyConnectedReject(
        op, "input/output are neither float nor symmetric int8 quantized");
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool weights_supported,
                          IsStaticRankedTensor(op.Inputs()[1]));
  if (!weights_supported) {
    LogFullyConnectedReject(op, "weights are not static ranked");
    return false;
  }
  const bool weights_float = IsFloatLike(op.Inputs()[1].ElementType());
  // The byte cap keeps large float heads on the CPU; quantized weights are
  // admitted whatever their size (Gemma 4 12B's INT4 head packs to 503 MB,
  // the 26B-A4B's dense projections are blockwise INT4): they run on the
  // GEMV plugin from the shared weight store, not as TensorRT constants.
  if (weights_float && op.Inputs()[1].HasWeights() &&
      op.Inputs()[1].Weights().Bytes().size() >
          MaxTensorRtFullyConnectedWeightBytes()) {
    LogFullyConnectedReject(op, "float weights exceed configured byte cap");
    return false;
  }
  // Reject FCs whose weight shape is listed in this env (comma-separated
  // "OUTxIN" entries), keeping specific problematic matmuls on the CPU.
  if (const char* disabled_shapes =
          std::getenv("LITERT_NVIDIA_TENSORRT_DISABLE_FC_SHAPES");
      disabled_shapes != nullptr && disabled_shapes[0] != '\0') {
    auto weights_type = op.Inputs()[1].RankedTensorType();
    if (weights_type && weights_type->Layout().Rank() == 2) {
      const auto dims = weights_type->Layout().Dimensions();
      const std::string shape =
          std::to_string(dims[0]) + "x" + std::to_string(dims[1]);
      const std::string list(disabled_shapes);
      size_t begin = 0;
      while (begin < list.size()) {
        size_t end = list.find(',', begin);
        if (end == std::string::npos) {
          end = list.size();
        }
        if (list.compare(begin, end - begin, shape) == 0) {
          LogFullyConnectedReject(op, "weight shape disabled via env");
          return false;
        }
        begin = end + 1;
      }
    }
  }
  // Debug switch to keep sub-byte (INT4/INT2) weight matmuls on the CPU.
  if (EnvEnabled("LITERT_NVIDIA_TENSORRT_DISABLE_SUBBYTE_WEIGHTS",
                 /*default_value=*/false) &&
      (op.Inputs()[1].ElementType() == litert::ElementType::Int4 ||
       op.Inputs()[1].ElementType() == litert::ElementType::Int2)) {
    LogFullyConnectedReject(op, "sub-byte weights disabled via env");
    return false;
  }
  // Centered INT2 weights run on the quant plugin (FindQuantRegions), from
  // INT8 activations or, for the vocabulary head, FLOAT16 ones.
  if (op.Inputs()[1].ElementType() == litert::ElementType::Int2 &&
      op.Inputs()[1].QTypeId() == kLiteRtQuantizationBlockWise) {
    const bool quant =
        IsQuantFullyConnected(op) || IsQuantLogitsFullyConnected(op);
    if (!quant) {
      LogFullyConnectedReject(
          op, "blockwise INT2 weights are not a quant plugin product");
    }
    return quant;
  }
  bool weights_quantized = false;
  if (op.Inputs()[1].ElementType() == litert::ElementType::Int8 ||
      op.Inputs()[1].ElementType() == litert::ElementType::Int4 ||
      op.Inputs()[1].ElementType() == litert::ElementType::Int2) {
    LITERT_ASSIGN_OR_RETURN(weights_quantized, HasSymmetricTensorRtQuantization(
                                                   op.Inputs()[1],
                                                   /*allow_per_channel=*/true));
    // Sub-byte weights are repacked at engine-build time, so they must be
    // compile-time constants.
    if (op.Inputs()[1].ElementType() != litert::ElementType::Int8 &&
        !op.Inputs()[1].HasWeights()) {
      weights_quantized = false;
    }
  }
  bool weights_blockwise = false;
  if (!weights_float && !weights_quantized &&
      IsBlockwiseInt4Weights(op.Inputs()[1])) {
    if (!BlockwiseFcPathEnabled()) {
      LogFullyConnectedReject(op, "blockwise weights need the CUDA GEMV path");
      return false;
    }
    // The GEMV takes a few activation rows and the GEMM the shapes it
    // supports; no TensorRT lowering can take over.
    LITERT_ASSIGN_OR_RETURN(auto in_type, op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto w_type, op.Inputs()[1].RankedTensorType());
    const auto in_dims = in_type.Layout().Dimensions();
    const auto w_dims = w_type.Layout().Dimensions();
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < in_dims.size(); ++i) {
      rows *= in_dims[i];
    }
    const LiteRtNvidiaGemmShape shape = {
        static_cast<int32_t>(std::min<int64_t>(rows, INT32_MAX)), w_dims[1], 2,
        kLiteRtNvidiaGemmGateNone, 32};
    if (rows < 1 || in_dims.empty() || in_dims.back() != w_dims[1] ||
        (rows > kLiteRtNvidiaInt4BlockGemvMaxRows &&
         (rows > INT32_MAX || !LiteRtNvidiaSubbyteGemmSupports(&shape)))) {
      LogFullyConnectedReject(
          op, "blockwise weights: no CUDA kernel for the activation rows");
      return false;
    }
    weights_blockwise = true;
  }
  if (!weights_float && !weights_quantized && !weights_blockwise) {
    LogFullyConnectedReject(
        op, "weights are neither float nor supported quantized");
    return false;
  }
  if (op.Inputs().size() == 3) {
    LITERT_ASSIGN_OR_RETURN(bool bias_supported,
                            IsStaticFloatTensor(op.Inputs()[2]));
    if (!op.Inputs()[2].HasWeights() || !bias_supported) {
      LogFullyConnectedReject(op, "bias is not constant static float");
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto weights_type, op.Inputs()[1].RankedTensorType());
  if (input_type.Layout().Rank() < 2 || weights_type.Layout().Rank() != 2) {
    LogFullyConnectedReject(op, "input or weights rank is unsupported");
    return false;
  }
  uint32_t weights_format = litert::kFullyConnectedOptionsWeightsFormatDefault;
  if (LiteRtGetFullyConnectedWeightsFormatOption(op.Get(), &weights_format) !=
          kLiteRtStatusOk ||
      weights_format != litert::kFullyConnectedOptionsWeightsFormatDefault) {
    LogFullyConnectedReject(op, "unsupported weights format option");
    return false;
  }
  bool keep_num_dims = false;
  if (LiteRtGetFullyConnectedKeepNumDimsOption(op.Get(), &keep_num_dims) !=
      kLiteRtStatusOk) {
    LogFullyConnectedReject(op, "failed to read keep_num_dims option");
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool activation_supported,
                          HasSupportedFusedActivation(op));
  if (!activation_supported) {
    LogFullyConnectedReject(op, "unsupported fused activation");
  }
  return activation_supported;
}

Expected<bool> IsConcatenationSupported(const Op& op) {
  if (op.Inputs().empty() || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!output_supported) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  const int rank = output_type.Layout().Rank();
  for (const auto& input : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool input_supported,
                            IsStaticTensorRtTensor(input));
    if (!input_supported || !SameElementType(input, op.Outputs()[0])) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto input_type, input.RankedTensorType());
    if (input_type.Layout().Rank() != rank) {
      return false;
    }
  }
  int32_t axis = 0;
  if (LiteRtGetConcatenationAxisOption(op.Get(), &axis) != kLiteRtStatusOk) {
    return false;
  }
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return false;
  }
  return HasSupportedFusedActivation(op);
}

Expected<bool> IsSelectV2Supported(const Op& op) {
  if (op.Inputs().size() != 3 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool cond_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool then_supported,
                          IsStaticTensorRtTensor(op.Inputs()[1]));
  LITERT_ASSIGN_OR_RETURN(bool else_supported,
                          IsStaticTensorRtTensor(op.Inputs()[2]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!cond_supported || !then_supported || !else_supported ||
      !output_supported) {
    return false;
  }
  return op.Inputs()[0].ElementType() == litert::ElementType::Bool &&
         SameElementType(op.Inputs()[1], op.Inputs()[2]) &&
         SameElementType(op.Inputs()[1], op.Outputs()[0]);
}

Expected<bool> IsSumSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticFloatTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported) {
    return false;
  }
  auto axes = ReadInt32Constant(op.Inputs()[1]);
  if (!axes) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const int rank = input_type.Layout().Rank();
  if (rank <= 0 || rank >= 32) {
    return false;
  }
  for (int32_t axis : *axes) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      return false;
    }
  }
  bool keep_dims = false;
  return LiteRtGetSumKeepDimsOption(op.Get(), &keep_dims) == kLiteRtStatusOk;
}

// MEAN over constant axes: the inlined qkv_norm_rope composite of Gemma 4
// 26B-A4B takes the mean of squares over the head dim.
Expected<bool> IsMeanSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticFloatTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported) {
    return false;
  }
  auto axes = ReadInt32Constant(op.Inputs()[1]);
  if (!axes || axes->empty()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const int rank = input_type.Layout().Rank();
  if (rank <= 0 || rank >= 32) {
    return false;
  }
  for (int32_t axis : *axes) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      return false;
    }
  }
  bool keep_dims = false;
  return LiteRtGetMeanKeepDimsOption(op.Get(), &keep_dims) == kLiteRtStatusOk;
}

// TOPK_V2 over the last axis with a constant k: the router of a
// mixture-of-experts layer picks its experts with it.
Expected<bool> IsTopkV2Supported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 2) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticFloatTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool values_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool indices_supported,
                          IsStaticTensorRtTensor(op.Outputs()[1]));
  if (!input_supported || !values_supported || !indices_supported ||
      op.Outputs()[1].ElementType() != litert::ElementType::Int32) {
    return false;
  }
  auto k = ReadInt32Constant(op.Inputs()[1]);
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const auto dims = input_type.Layout().Dimensions();
  // TensorRT's TopK takes k up to 3840 and needs a leading axis.
  return k && k->size() == 1 && (*k)[0] > 0 && (*k)[0] <= 3840 &&
         dims.size() >= 2 && dims.back() >= (*k)[0];
}

Expected<bool> IsReduceMaxSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported ||
      !IsTensorRtArithmeticElementType(op.Inputs()[0].ElementType()) ||
      !SameElementType(op.Inputs()[0], op.Outputs()[0])) {
    return false;
  }
  auto axes = ReadInt32Constant(op.Inputs()[1]);
  if (!axes) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const int rank = input_type.Layout().Rank();
  if (rank <= 0 || rank >= 32) {
    return false;
  }
  for (int32_t axis : *axes) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      return false;
    }
  }
  bool keep_dims = false;
  return LiteRtGetReduceMaxKeepDimsOption(op.Get(), &keep_dims) ==
         kLiteRtStatusOk;
}

Expected<bool> HasSymmetricTensorRtQuantization(const Tensor& tensor,
                                                bool allow_per_channel) {
  switch (tensor.QTypeId()) {
    case kLiteRtQuantizationPerTensor: {
      auto q = tensor.PerTensorQuantization();
      return q.scale > 0.0f && q.zero_point == 0;
    }
    case kLiteRtQuantizationPerChannel: {
      if (!allow_per_channel) {
        return false;
      }
      auto q = tensor.PerChannelQuantization();
      if (q.num_channels == 0 || q.scales == nullptr ||
          q.zero_points == nullptr) {
        return false;
      }
      for (uint64_t i = 0; i < q.num_channels; ++i) {
        if (q.scales[i] <= 0.0f || q.zero_points[i] != 0) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

// TFLite blockwise quantization as Gemma 4 26B-A4B's weights carry it: INT4
// values with one scale per block of 32 input dims in a constant FP16 or
// FP32 tensor [rows, columns / 32] and no zero points. The CUDA GEMV and GEMM
// read it; TensorRT's own fully connected lowering does not.
bool IsBlockwiseInt4Weights(const Tensor& tensor) {
  if (tensor.ElementType() != litert::ElementType::Int4 ||
      tensor.QTypeId() != kLiteRtQuantizationBlockWise ||
      !tensor.HasWeights()) {
    return false;
  }
  const auto q = tensor.BlockWiseQuantization();
  if (q.block_size != 32 || q.scales == nullptr || q.zero_points != nullptr) {
    return false;
  }
  const auto type = tensor.RankedTensorType();
  if (!type || type->Layout().Rank() != 2) {
    return false;
  }
  const auto dims = type->Layout().Dimensions();
  if (dims[0] <= 0 || dims[1] <= 0 || dims[1] % 32 != 0) {
    return false;
  }
  const Tensor scales(tensor.Context(), q.scales);
  if (!scales.HasWeights() ||
      (scales.ElementType() != litert::ElementType::Float16 &&
       scales.ElementType() != litert::ElementType::Float32)) {
    return false;
  }
  const auto scales_type = scales.RankedTensorType();
  if (!scales_type) {
    return false;
  }
  const auto count = NumElements(*scales_type);
  if (!count || *count != static_cast<size_t>(dims[0]) *
                              static_cast<size_t>(dims[1] / 32)) {
    return false;
  }
  const size_t element_bytes =
      scales.ElementType() == litert::ElementType::Float16 ? 2 : 4;
  return scales.Weights().Bytes().size() == *count * element_bytes;
}

// Whether blockwise weights have a lowering: the CUDA GEMV and GEMM paths
// (16-bit activations, cuda_gemv predequantization) are on.
bool BlockwiseFcPathEnabled() {
  return Fp16ActivationsEnabled() &&
         PredequantizeFcWeightsMode() == PredequantMode::kCudaGemv &&
         GetCudaGemmMode() != CudaGemmMode::kOff;
}

Expected<bool> IsQuantizeSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported ||
      !IsFloatLike(op.Inputs()[0].ElementType()) ||
      !IsTensorRtQuantizedElementType(op.Outputs()[0].ElementType())) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  if (input_type.Layout().Rank() == 0) {
    return false;
  }
  return HasSymmetricTensorRtQuantization(
      op.Outputs()[0], /*allow_per_channel=*/op.Inputs()[0].HasWeights());
}

Expected<bool> IsDequantizeSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().size() != 1) {
    return false;
  }
  const Tensor input = op.Inputs()[0];
  LITERT_ASSIGN_OR_RETURN(bool input_static, IsStaticRankedTensor(input));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  // Runtime tensors must be a TensorRT-representable quantized type. INT2 is
  // additionally allowed for constants, which are repacked to INT4 weights.
  const bool input_type_ok =
      IsTensorRtQuantizedElementType(input.ElementType()) ||
      (input.HasWeights() && input.ElementType() == litert::ElementType::Int2);
  if (!input_static || !output_supported || !input_type_ok ||
      !IsFloatLike(op.Outputs()[0].ElementType())) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, input.RankedTensorType());
  if (input_type.Layout().Rank() == 0) {
    return false;
  }
  return HasSymmetricTensorRtQuantization(
      input, /*allow_per_channel=*/input.HasWeights());
}

Expected<bool> IsFillSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  if (!ReadInt32Constant(op.Inputs()[0])) {
    return false;
  }
  if (!op.Inputs()[1].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto value_type, op.Inputs()[1].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  LITERT_ASSIGN_OR_RETURN(size_t value_elements, NumElements(value_type));
  LITERT_ASSIGN_OR_RETURN(size_t output_bytes,
                          StaticTensorByteSize(op.Outputs()[0]));
  if (output_bytes > MaxTensorRtFillBytes()) {
    return false;
  }
  return output_supported && value_elements == 1 &&
         value_type.ElementType() == output_type.ElementType();
}

Expected<bool> IsPackSupported(const Op& op) {
  if (op.Inputs().empty() || op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!output_supported) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  const int output_rank = output_type.Layout().Rank();
  if (output_rank <= 0) {
    return false;
  }
  int32_t axis = 0;
  int32_t values_count = 0;
  if (LiteRtGetPackAxisOption(op.Get(), &axis) != kLiteRtStatusOk ||
      LiteRtGetPackValuesCountOption(op.Get(), &values_count) !=
          kLiteRtStatusOk ||
      values_count != static_cast<int32_t>(op.Inputs().size())) {
    return false;
  }
  if (axis < 0) {
    axis += output_rank;
  }
  if (axis < 0 || axis >= output_rank ||
      output_type.Layout().Dimensions()[axis] != values_count) {
    return false;
  }
  for (const auto& input : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool input_supported,
                            IsStaticTensorRtTensor(input));
    if (!input_supported || !SameElementType(input, op.Outputs()[0])) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto input_type, input.RankedTensorType());
    if (input_type.Layout().Rank() + 1 != output_rank) {
      return false;
    }
  }
  return true;
}

Expected<bool> IsUnpackSupported(const Op& op) {
  if (op.Inputs().size() != 1 || op.Outputs().empty()) {
    return false;
  }
  if (op.Inputs()[0].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticTensorRtTensor(op.Inputs()[0]));
  if (!input_supported) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  const int input_rank = input_type.Layout().Rank();
  if (input_rank <= 0) {
    return false;
  }
  int32_t axis = 0;
  int32_t num = 0;
  if (LiteRtGetUnpackAxisOption(op.Get(), &axis) != kLiteRtStatusOk ||
      LiteRtGetUnpackNumOption(op.Get(), &num) != kLiteRtStatusOk ||
      num != static_cast<int32_t>(op.Outputs().size())) {
    return false;
  }
  if (axis < 0) {
    axis += input_rank;
  }
  if (axis < 0 || axis >= input_rank ||
      input_type.Layout().Dimensions()[axis] != num) {
    return false;
  }
  for (const auto& output : op.Outputs()) {
    LITERT_ASSIGN_OR_RETURN(bool output_supported,
                            IsStaticTensorRtTensor(output));
    if (!output_supported || !SameElementType(op.Inputs()[0], output)) {
      return false;
    }
    LITERT_ASSIGN_OR_RETURN(auto output_type, output.RankedTensorType());
    if (output_type.Layout().Rank() + 1 != input_rank) {
      return false;
    }
  }
  return true;
}

Expected<bool> IsFloorModSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  for (const auto& tensor : op.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(bool supported, IsStaticTensorRtTensor(tensor));
    if (!supported || tensor.ElementType() != litert::ElementType::Int32) {
      return false;
    }
  }
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  return output_supported &&
         op.Outputs()[0].ElementType() == litert::ElementType::Int32;
}

// BITWISE_XOR of int32 tensors (Qwen 4's n-gram hash), as arithmetic.
Expected<bool> IsBitwiseXorSupported(const Op& op) {
  return IsFloorModSupported(op);
}

// The largest constant table a GATHER takes into the engine: Qwen 4's PLE
// tables of a real export (hundreds of millions of rows) stay on LiteRT's
// CPU, a reduced model's or an embedding of this size runs in the engine.
constexpr size_t kMaxGatherTableBytes = size_t{256} << 20;

// GATHER along one axis (batch_dims 0) of a float or int32 tensor, or of a
// constant table up to kMaxGatherTableBytes, with int32 indices.
Expected<bool> IsGatherSupported(const Op& op) {
  if (op.Inputs().size() != 2 || op.Outputs().size() != 1) {
    return false;
  }
  int32_t axis = 0, batch_dims = 0;
  if (LiteRtGetGatherAxisOption(op.Get(), &axis) != kLiteRtStatusOk ||
      LiteRtGetGatherBatchDimsOption(op.Get(), &batch_dims) !=
          kLiteRtStatusOk ||
      batch_dims != 0) {
    return false;
  }
  const Tensor data = op.Inputs()[0];
  const Tensor indices = op.Inputs()[1];
  LITERT_ASSIGN_OR_RETURN(bool data_static, IsStaticTensorRtTensor(data));
  LITERT_ASSIGN_OR_RETURN(bool indices_static, IsStaticTensorRtTensor(indices));
  LITERT_ASSIGN_OR_RETURN(bool output_static,
                          IsStaticTensorRtTensor(op.Outputs()[0]));
  if (!data_static || !indices_static || !output_static ||
      indices.ElementType() != litert::ElementType::Int32 ||
      indices.HasWeights() ||
      (data.ElementType() != litert::ElementType::Int32 &&
       !IsFloatLike(data.ElementType())) ||
      data.QTypeId() != kLiteRtQuantizationNone) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto type, data.RankedTensorType());
  const int rank = static_cast<int>(type.Layout().Rank());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return false;
  return !data.HasWeights() ||
         data.Weights().Bytes().size() <= kMaxGatherTableBytes;
}

Expected<bool> IsConv2dSupported(const Op& op) {
  if (op.Inputs().size() < 2 || op.Inputs().size() > 3 ||
      op.Outputs().size() != 1) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(bool input_supported,
                          IsStaticFloatTensor(op.Inputs()[0]));
  LITERT_ASSIGN_OR_RETURN(bool output_supported,
                          IsStaticFloatTensor(op.Outputs()[0]));
  if (!input_supported || !output_supported || !op.Inputs()[1].HasWeights()) {
    return false;
  }
  if (op.Inputs().size() == 3 && !op.Inputs()[2].HasWeights()) {
    return false;
  }
  LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto filter_type, op.Inputs()[1].RankedTensorType());
  LITERT_ASSIGN_OR_RETURN(auto output_type, op.Outputs()[0].RankedTensorType());
  if (input_type.Layout().Rank() != 4 || filter_type.Layout().Rank() != 4 ||
      output_type.Layout().Rank() != 4) {
    return false;
  }
  uint32_t padding = litert::kPaddingSame;
  if (LiteRtGetConv2dPaddingOption(op.Get(), &padding) != kLiteRtStatusOk ||
      padding > litert::kPaddingValid) {
    return false;
  }
  return HasSupportedFusedActivation(op);
}

std::string TensorName(const char* prefix, const Tensor& tensor, int index) {
  return std::string(prefix) + "_" + std::to_string(index) + "_tensor_" +
         std::to_string(tensor.TensorIndex());
}

bool TensorHasShape(const Tensor& tensor,
                    std::initializer_list<int32_t> expected) {
  auto type = tensor.RankedTensorType();
  if (!type) {
    return false;
  }
  const auto dimensions = type->Layout().Dimensions();
  return dimensions.size() == expected.size() &&
         std::equal(dimensions.begin(), dimensions.end(), expected.begin());
}

bool IsUnquantizedFloat32(const Tensor& tensor) {
  return tensor.ElementType() == litert::ElementType::Float32 &&
         tensor.QTypeId() == kLiteRtQuantizationNone;
}

bool IsDefinedBy(const Tensor& tensor, const Op& op) {
  const auto defining_op = tensor.DefiningOp();
  return defining_op.has_value() && defining_op->op == op.Get();
}

bool HasOnlyUse(const Tensor& tensor, const Op& op,
                LiteRtParamIndex input_index) {
  const auto uses = tensor.Uses();
  return uses.size() == 1 && uses[0].user.Get() == op.Get() &&
         uses[0].user_arg_ind == input_index;
}

std::optional<uint32_t> FindSubgraphOutputPort(const Subgraph& subgraph,
                                               const Tensor& tensor) {
  const auto outputs = subgraph.Outputs();
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].Get() == tensor.Get()) {
      return i;
    }
  }
  return std::nullopt;
}

std::optional<float> ReadFloat32Scalar(const Tensor& tensor) {
  if (!tensor.HasWeights() || !IsUnquantizedFloat32(tensor)) {
    return std::nullopt;
  }
  auto type = tensor.RankedTensorType();
  if (!type) {
    return std::nullopt;
  }
  auto num_elements = NumElements(*type);
  const auto bytes = tensor.Weights().Bytes();
  if (!num_elements || *num_elements != 1 || bytes.size() != sizeof(float)) {
    return std::nullopt;
  }
  float value = 0.0f;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

float HalfBitsToFloat(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
  const uint32_t exponent = (half >> 10) & 0x1F;
  const uint32_t mantissa = half & 0x3FF;
  uint32_t bits = 0;
  if (exponent == 0) {
    const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return (half & 0x8000) ? -magnitude : magnitude;
  }
  if (exponent == 31) {
    bits = sign | 0x7F800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// A single-element FP32 or FP16 constant as a float.
std::optional<float> ReadFloatScalar(const Tensor& tensor) {
  if (auto value = ReadFloat32Scalar(tensor)) {
    return value;
  }
  if (!tensor.HasWeights() ||
      tensor.ElementType() != litert::ElementType::Float16) {
    return std::nullopt;
  }
  auto type = tensor.RankedTensorType();
  if (!type) {
    return std::nullopt;
  }
  auto num_elements = NumElements(*type);
  const auto bytes = tensor.Weights().Bytes();
  if (!num_elements || *num_elements != 1 || bytes.size() != sizeof(uint16_t)) {
    return std::nullopt;
  }
  uint16_t half = 0;
  std::memcpy(&half, bytes.data(), sizeof(half));
  return HalfBitsToFloat(half);
}

uint16_t Float32ToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

bool HasNoFusedActivation(const Op& op) {
  auto activation = GetFusedActivation(op);
  return activation.HasValue() &&
         *activation == litert::kActivationFunctionTypeNone;
}

Expected<std::optional<TensorRtLlmHeadBuildData>> MatchTensorRtLlmHead(
    const Subgraph& subgraph) {
  constexpr uint32_t kGemma4HiddenSize = 1536;
  constexpr uint32_t kGemma4VocabSize = 262144;
  constexpr float kGemma4SoftCap = 30.0f;
  constexpr size_t kTailOpCount = 4;

  const auto no_match = [] {
    return std::optional<TensorRtLlmHeadBuildData>();
  };
  const auto ops = subgraph.Ops();
  if (ops.size() < kTailOpCount) {
    return no_match();
  }
  const Op& fc = ops[ops.size() - 4];
  const Op& scale_down = ops[ops.size() - 3];
  const Op& tanh = ops[ops.size() - 2];
  const Op& scale_up = ops[ops.size() - 1];
  if (fc.Code() != kLiteRtOpCodeTflFullyConnected ||
      scale_down.Code() != kLiteRtOpCodeTflMul ||
      tanh.Code() != kLiteRtOpCodeTflTanh ||
      scale_up.Code() != kLiteRtOpCodeTflMul) {
    return no_match();
  }

  const auto fc_inputs = fc.Inputs();
  const auto fc_outputs = fc.Outputs();
  const auto scale_down_inputs = scale_down.Inputs();
  const auto scale_down_outputs = scale_down.Outputs();
  const auto tanh_inputs = tanh.Inputs();
  const auto tanh_outputs = tanh.Outputs();
  const auto scale_up_inputs = scale_up.Inputs();
  const auto scale_up_outputs = scale_up.Outputs();
  if (fc_inputs.size() != 2 || fc_outputs.size() != 1 ||
      scale_down_inputs.size() != 2 || scale_down_outputs.size() != 1 ||
      tanh_inputs.size() != 1 || tanh_outputs.size() != 1 ||
      scale_up_inputs.size() != 2 || scale_up_outputs.size() != 1) {
    return no_match();
  }

  const Tensor& hidden = fc_inputs[0];
  const Tensor& weights = fc_inputs[1];
  const Tensor& fc_output = fc_outputs[0];
  const Tensor& scale_down_output = scale_down_outputs[0];
  const Tensor& tanh_output = tanh_outputs[0];
  const Tensor& logits = scale_up_outputs[0];
  if (scale_down_inputs[0].Get() != fc_output.Get() ||
      tanh_inputs[0].Get() != scale_down_output.Get() ||
      scale_up_inputs[0].Get() != tanh_output.Get() ||
      !IsDefinedBy(fc_output, fc) ||
      !IsDefinedBy(scale_down_output, scale_down) ||
      !IsDefinedBy(tanh_output, tanh) || !IsDefinedBy(logits, scale_up) ||
      !HasOnlyUse(fc_output, scale_down, 0) ||
      !HasOnlyUse(scale_down_output, tanh, 0) ||
      !HasOnlyUse(tanh_output, scale_up, 0) || !logits.Uses().empty()) {
    return no_match();
  }

  const auto hidden_port = FindSubgraphOutputPort(subgraph, hidden);
  const auto logits_port = FindSubgraphOutputPort(subgraph, logits);
  if (!hidden_port.has_value() || !logits_port.has_value() ||
      *hidden_port == *logits_port || !hidden.DefiningOp().has_value() ||
      !HasOnlyUse(hidden, fc, 0)) {
    return no_match();
  }

  if (!IsUnquantizedFloat32(hidden) || !IsUnquantizedFloat32(fc_output) ||
      !IsUnquantizedFloat32(scale_down_output) ||
      !IsUnquantizedFloat32(tanh_output) || !IsUnquantizedFloat32(logits) ||
      !TensorHasShape(hidden,
                      {1, 1, static_cast<int32_t>(kGemma4HiddenSize)}) ||
      !TensorHasShape(weights, {static_cast<int32_t>(kGemma4VocabSize),
                                static_cast<int32_t>(kGemma4HiddenSize)}) ||
      !TensorHasShape(fc_output,
                      {1, 1, static_cast<int32_t>(kGemma4VocabSize)}) ||
      !TensorHasShape(scale_down_output,
                      {1, 1, static_cast<int32_t>(kGemma4VocabSize)}) ||
      !TensorHasShape(tanh_output,
                      {1, 1, static_cast<int32_t>(kGemma4VocabSize)}) ||
      !TensorHasShape(logits, {1, 1, static_cast<int32_t>(kGemma4VocabSize)})) {
    return no_match();
  }

  if (!weights.HasWeights() ||
      weights.ElementType() != litert::ElementType::Int2 ||
      weights.QTypeId() != kLiteRtQuantizationPerChannel ||
      !HasNoFusedActivation(fc) || !HasNoFusedActivation(scale_down) ||
      !HasNoFusedActivation(scale_up)) {
    return no_match();
  }
  uint32_t weights_format = litert::kFullyConnectedOptionsWeightsFormatDefault;
  bool keep_num_dims = false;
  if (LiteRtGetFullyConnectedWeightsFormatOption(fc.Get(), &weights_format) !=
          kLiteRtStatusOk ||
      weights_format != litert::kFullyConnectedOptionsWeightsFormatDefault ||
      LiteRtGetFullyConnectedKeepNumDimsOption(fc.Get(), &keep_num_dims) !=
          kLiteRtStatusOk ||
      !keep_num_dims) {
    return no_match();
  }

  const auto down = ReadFloat32Scalar(scale_down_inputs[1]);
  const auto cap = ReadFloat32Scalar(scale_up_inputs[1]);
  if (!down.has_value() || !cap.has_value() || *cap != kGemma4SoftCap ||
      *down != 1.0f / kGemma4SoftCap) {
    return no_match();
  }

  const auto quantization = weights.PerChannelQuantization();
  if (quantization.quantized_dimension != 0 ||
      quantization.num_channels != kGemma4VocabSize ||
      quantization.scales == nullptr || quantization.zero_points == nullptr) {
    return no_match();
  }
  for (uint64_t i = 0; i < quantization.num_channels; ++i) {
    if (!std::isfinite(quantization.scales[i]) ||
        quantization.scales[i] <= 0.0f || quantization.zero_points[i] != 0) {
      return no_match();
    }
  }

  const auto raw_weights = weights.Weights().Bytes();
  const uint64_t num_elements =
      static_cast<uint64_t>(kGemma4HiddenSize) * kGemma4VocabSize;
  if (raw_weights.size() != (num_elements + 3) / 4) {
    return no_match();
  }
  TensorRtLlmHeadBuildData result;
  result.hidden_output_port = *hidden_port;
  result.logits_output_port = *logits_port;
  result.k = kGemma4HiddenSize;
  result.n = kGemma4VocabSize;
  result.soft_cap = kGemma4SoftCap;
  result.weight_format = TensorRtLlmHeadWeightFormat::kInt2TfliteRowMajor;
  result.packed_weights = raw_weights;
  result.bf16_scales.resize(kGemma4VocabSize * sizeof(uint16_t));
  for (uint32_t i = 0; i < kGemma4VocabSize; ++i) {
    const uint16_t bits = Float32ToBf16Bits(quantization.scales[i]);
    result.bf16_scales[2 * i] = static_cast<uint8_t>(bits);
    result.bf16_scales[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
  }
  return std::optional<TensorRtLlmHeadBuildData>(std::move(result));
}

std::optional<int> CurrentCudaComputeCapability() {
  int device = 0;
  cudaDeviceProp properties{};
  const cudaError_t get_device_status = cudaGetDevice(&device);
  if (get_device_status != cudaSuccess) {
    LITERT_LOG(LITERT_WARNING, "NVIDIA CUDA device query failed: %s",
               cudaGetErrorString(get_device_status));
    return std::nullopt;
  }
  const cudaError_t properties_status =
      cudaGetDeviceProperties(&properties, device);
  if (properties_status != cudaSuccess) {
    LITERT_LOG(LITERT_WARNING, "NVIDIA CUDA device properties query failed: %s",
               cudaGetErrorString(properties_status));
    return std::nullopt;
  }
  return properties.major * 10 + properties.minor;
}

bool RunningUnderWsl() {
  static const bool is_wsl = [] {
    std::FILE* f = std::fopen("/proc/version", "r");
    if (f == nullptr) {
      return false;
    }
    char buffer[512] = {};
    const size_t read = std::fread(buffer, 1, sizeof(buffer) - 1, f);
    std::fclose(f);
    (void)read;
    for (char& c : buffer) {
      c = std::tolower(static_cast<unsigned char>(c));
    }
    return std::strstr(buffer, "microsoft") != nullptr;
  }();
  return is_wsl;
}

// WSL2's stream-ordered allocator (cudaMallocAsync) pools cannot grow to the
// sizes the TensorRT builder requests for large partitions, so builder
// allocations fail with spurious OOMs while plain cudaMalloc of the same size
// succeeds. Route builder allocations through synchronous cudaMalloc there.
class SyncCudaGpuAllocator : public nvinfer1::IGpuAllocator {
 public:
  void* allocate(uint64_t const size, uint64_t const /*alignment*/,
                 nvinfer1::AllocatorFlags const /*flags*/) noexcept override {
    if (size == 0) {
      return nullptr;
    }
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
      return nullptr;
    }
    return ptr;
  }

  bool deallocate(void* const memory) noexcept override {
    return cudaFree(memory) == cudaSuccess;
  }

  void* allocateAsync(uint64_t const size, uint64_t const alignment,
                      nvinfer1::AllocatorFlags const flags,
                      cudaStream_t /*stream*/) noexcept override {
    return allocate(size, alignment, flags);
  }

  bool deallocateAsync(void* const memory,
                       cudaStream_t /*stream*/) noexcept override {
    // cudaFree synchronizes the device, so in-flight work that uses this
    // memory completes before it is released.
    return deallocate(memory);
  }
};

bool UseSyncCudaAllocator() {
  const char* value = std::getenv("LITERT_NVIDIA_TENSORRT_SYNC_ALLOCATOR");
  if (value != nullptr && value[0] != '\0') {
    return std::strcmp(value, "0") != 0;
  }
  return RunningUnderWsl();
}

struct OwnedRefitWeight {
  std::string name;
  int32_t data_type = 0;
  uint64_t count = 0;
  size_t owned_weight_index = 0;
};

nvinfer1::Permutation MakePermutation(std::initializer_list<int32_t> values) {
  nvinfer1::Permutation permutation{};
  int index = 0;
  for (int32_t value : values) {
    permutation.order[index++] = value;
  }
  return permutation;
}

class TensorRtGraphBuilder {
 public:
  Expected<TensorRtBuildResult> Build(
      const Subgraph& subgraph,
      absl::Span<const std::string> read_only_value_cache_inputs,
      TensorRtWeightStoreBuilder* weight_store,
      absl::Span<const std::string> shared_cache_inputs) {
    weight_store_ = weight_store;
    GraphComputesFloat16() = SubgraphComputesFloat16(subgraph);
    if (GraphComputesFloat16()) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX the graph computes in FLOAT16: FP16 "
                 "activations");
    }
    for (const auto& op : subgraph.Ops()) {
      graph_has_gated_delta_net_ |= IsGatedDeltaUpdateOp(op);
    }
    if (Int8BlockwiseGemmRequested(graph_has_gated_delta_net_)) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX blockwise INT4 GEMMs run on the int8 "
                 "tensor cores%s",
                 graph_has_gated_delta_net_ ? " (Gated DeltaNet layers)" : "");
    }
    LITERT_RETURN_IF_ERROR(
        FindReadOnlyValueCaches(subgraph, read_only_value_cache_inputs));
    LITERT_RETURN_IF_ERROR(
        FindSharedCacheInputs(subgraph, shared_cache_inputs));
    memory_profiler_.Log("graph_build_begin");
    builder_.reset(nvinfer1::createInferBuilder(logger_));
    if (!builder_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT builder");
    }
    if (UseSyncCudaAllocator()) {
      builder_->setGpuAllocator(&sync_allocator_);
    }

    network_.reset(builder_->createNetworkV2(/*flags=*/0));
    if (!network_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT network");
    }

    config_.reset(builder_->createBuilderConfig());
    if (!config_) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to create TensorRT builder config");
    }
    config_->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                TensorRtWorkspaceBytes());
    // Lower levels profile fewer tactics, which cuts engine-build time and
    // the autotuner's transient device-memory footprint on large partitions.
    const char* opt_level_env =
        std::getenv("LITERT_NVIDIA_TENSORRT_BUILDER_OPT_LEVEL");
    if (opt_level_env != nullptr && opt_level_env[0] != '\0') {
      char* end = nullptr;
      const long parsed = std::strtol(opt_level_env, &end, 10);
      if (end != opt_level_env && *end == '\0' && parsed >= 0 && parsed <= 5) {
        config_->setBuilderOptimizationLevel(static_cast<int32_t>(parsed));
      }
    }
    // TensorRT derives the tactic-DRAM budget from free device memory, which
    // is unreliable under WSL; allow pinning it explicitly.
    const size_t tactic_dram_mb =
        EnvSizeT("LITERT_NVIDIA_TENSORRT_TACTIC_DRAM_MB", 0);
    if (tactic_dram_mb > 0) {
      config_->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_DRAM,
                                  tactic_dram_mb << 20);
    }
    if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_ALLOW_TF32",
                    /*default_value=*/false)) {
      config_->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }
    if (TensorRtSharedWeightsEnabled()) {
      // Only constants explicitly registered below are refittable. Making all
      // builder-generated weights refittable produces an archive that the
      // TensorRT-RTX recorder cannot serialize for these large graphs.
      config_->setFlag(nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL);
    }
#if !defined(TRT_MAJOR_RTX)
    if (std::getenv("LITERT_NVIDIA_TENSORRT_FP16") != nullptr) {
      config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
#else
    if (std::getenv("LITERT_NVIDIA_TENSORRT_FP16") != nullptr) {
      LITERT_LOG(LITERT_WARNING,
                 "Ignoring LITERT_NVIDIA_TENSORRT_FP16 because TensorRT-RTX "
                 "uses strongly typed networks and does not support "
                 "BuilderFlag::kFP16");
    }
#endif

    memory_profiler_.Log("builder_configured");

    const auto ops = subgraph.Ops();
    std::optional<TensorRtLlmHeadBuildData> trtllm_head;
    const std::optional<int> compute_capability =
        CurrentCudaComputeCapability();
    // The external head carries a private copy of the vocabulary weights in
    // the shard of its engine. With a weight store the head's fully connected
    // op runs on the CUDA subbyte GEMV plugin instead, so every graph of the
    // model (decode and verify) reads the one shared copy.
    if (compute_capability.has_value() &&
        IsInt2GemvComputeCapabilitySupported(*compute_capability) &&
        weight_store_ == nullptr) {
      LITERT_ASSIGN_OR_RETURN(trtllm_head, MatchTensorRtLlmHead(subgraph));
      if (trtllm_head.has_value()) {
        LITERT_LOG(
            LITERT_INFO,
            "NVIDIA TensorRT-RTX external W2 head matched: hidden_port=%u "
            "logits_port=%u K=%u N=%u weights_bytes=%zu scales_bytes=%zu",
            trtllm_head->hidden_output_port, trtllm_head->logits_output_port,
            trtllm_head->k, trtllm_head->n, trtllm_head->packed_weights.size(),
            trtllm_head->bf16_scales.size());
      }
    }

    LITERT_RETURN_IF_ERROR(FindInPlaceCacheUpdates(subgraph));
    LITERT_RETURN_IF_ERROR(FindDecodeAttentionBlocks(subgraph));
    LITERT_RETURN_IF_ERROR(FindWindowedCacheReads(subgraph));
    LITERT_RETURN_IF_ERROR(FindInt8CacheAttentionBlocks(subgraph));
    FindPackedInt4Caches(subgraph);
    FindGeneralCaches(subgraph);
    LITERT_RETURN_IF_ERROR(AddInputs(subgraph));
    const size_t num_ops_to_lower =
        ops.size() - (trtllm_head.has_value() ? 4 : 0);
    for (size_t i = 0; i < num_ops_to_lower; ++i) {
      op_order_[ops[i].Get()] = i;
    }
    LITERT_RETURN_IF_ERROR(FindPrefillAttentionBlocks(subgraph));
    LITERT_RETURN_IF_ERROR(FindCompositeAttentionBlocks(subgraph));
    LITERT_RETURN_IF_ERROR(FindMoeBlocks(subgraph));
    LITERT_RETURN_IF_ERROR(FindGatedDeltaNetBlocks(subgraph));
    LITERT_RETURN_IF_ERROR(FindQuantRegions(subgraph));
    if (megakernel_body_) {
      LITERT_RETURN_IF_ERROR(LowerMegakernel(subgraph));
    }
    std::vector<Op> prefill_cache_updates;
    for (size_t i = 0; i < num_ops_to_lower; ++i) {
      if (megakernel_body_) break;
      if (ops[i].Code() == kLiteRtOpCodeShloComposite &&
          CompositeOpName(ops[i]) == "odml.cache_update" &&
          PrefillCacheUpdateRingMode(ops[i]).has_value() &&
          !FindPrefillAttentionJoin(ops[i]).has_value()) {
        // These outputs have no internal users. Finish lowering their old
        // readers first so the patch plugin can depend on the reader results.
        // This is graph construction order, not a final-layer execution fence.
        prefill_cache_updates.push_back(ops[i]);
      } else {
        LITERT_RETURN_IF_ERROR(LowerOp(ops[i]));
      }
    }
    for (const auto& op : prefill_cache_updates) {
      LITERT_RETURN_IF_ERROR(LowerOp(op));
    }
    LITERT_RETURN_IF_ERROR(AddUnjoinedQuantCacheUpdates());
    LITERT_RETURN_IF_ERROR(MarkOutputs(subgraph, trtllm_head));
    memory_profiler_.Log("graph_lowered");

    LITERT_RETURN_IF_ERROR(AddWeightHolders());

    // The CUDA GEMV plugin receives packed subbyte weights through an INT8
    // constant input. Those weights cannot use the safe stripping path below,
    // so keep the decode plan portable and self-contained instead of making it
    // refittable only for its comparatively tiny scale constants.
    const bool strip_plan =
        TensorRtSharedWeightsEnabled() && !uses_cuda_subbyte_gemv_;
    if (!weight_holders_.empty()) {
      // The plugin weights are placeholders for segments of a weight store.
      // Only they are stripped: the other constants of the partition are
      // small and stay in the plan, so the engine needs no refit.
      for (const auto& weight : owned_refit_weights_) {
        network_->unmarkWeightsRefittable(weight.name.c_str());
      }
      if (!config_->setNbComputeCapabilities(1) ||
          !config_->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT,
                                         0)) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to target the current GPU for a stripped plan");
      }
      config_->setFlag(nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL);
      config_->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
      uint64_t holder_bytes = 0;
      for (const auto& holder : weight_holders_) {
        holder_bytes += holder.bytes;
      }
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX building a plan without its packed "
                 "plugin weights: %zu weight holders (%llu bytes) read by %zu "
                 "plugins",
                 weight_holders_.size(),
                 static_cast<unsigned long long>(holder_bytes),
                 holder_readers_.size());
    } else if (strip_plan) {
      // Weight-stripped TensorRT-RTX plans require a GPU build targeting at
      // most one compute capability. Ordinary plans retain RTX's portable
      // multi-architecture default.
      if (!config_->setNbComputeCapabilities(1) ||
          !config_->setComputeCapability(nvinfer1::ComputeCapability::kCURRENT,
                                         0)) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to target the current GPU for a stripped plan");
      }
      config_->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
      size_t refit_weight_bytes = 0;
      for (const auto& weight : owned_refit_weights_) {
        refit_weight_bytes += owned_weights_[weight.owned_weight_index].size();
      }
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX building a stripped selectively "
                 "refittable plan with %zu weights (%zu bytes)",
                 owned_refit_weights_.size(), refit_weight_bytes);
    } else if (TensorRtSharedWeightsEnabled()) {
      config_->clearFlag(nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL);
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX keeping CUDA GEMV plugin plan "
                 "self-contained because its packed INT8 plugin weights "
                 "cannot be stripped safely by this SDK");
    }

    memory_profiler_.Log("engine_serialize_begin");
    TrtPtr<nvinfer1::IHostMemory> serialized(
        builder_->buildSerializedNetwork(*network_, *config_));
    if (!serialized) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to build serialized TensorRT engine");
    }
    memory_profiler_.Log("engine_serialize_end");
    if (EnvEnabled("LITERT_NVIDIA_TENSORRT_DUMP_ENGINE_INFO",
                   /*default_value=*/false)) {
      TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger_));
      if (runtime) {
        TrtPtr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
            serialized->data(), serialized->size()));
        if (engine) {
          TrtPtr<nvinfer1::IEngineInspector> inspector(
              engine->createEngineInspector());
          if (inspector) {
            const char* info = inspector->getEngineInformation(
                nvinfer1::LayerInformationFormat::kONELINE);
            LITERT_LOG(LITERT_INFO, "TensorRT engine layers:\n%s",
                       info != nullptr ? info : "(null)");
          }
        }
      }
    }

    TensorRtBuildResult result;
    result.input_names = input_names_;
    result.output_names = output_names_;
    result.trtllm_head = std::move(trtllm_head);
    LITERT_ASSIGN_OR_RETURN(result.step_feed, MatchStepFeed(subgraph));
    // The batched decode engines of a decode step (one that feeds itself),
    // built from its network while its constants are still alive.
    if (const int32_t max_slots = DecodeSlotsRequested();
        max_slots >= 2 && result.step_feed.has_value() &&
        !result.trtllm_head.has_value() && !megakernel_body_) {
      for (int32_t slots = 2; slots <= max_slots; ++slots) {
        auto plan = BuildSlotPlan(slots);
        if (!plan) {
          LITERT_LOG(LITERT_WARNING,
                     "NVIDIA TensorRT-RTX builds no batched decode engine of "
                     "%d slots: %s",
                     slots, plan.Error().Message().c_str());
          break;
        }
        result.slot_plans.push_back(std::move(*plan));
      }
    }
    result.is_stripped_plan = strip_plan;
    result.weight_holders = std::move(weight_holders_);
    // TensorRT-RTX has consumed every network constant at this point. Destroy
    // the network before copying its serialized plan into the result so the
    // compiler-owned constants do not overlap that additional engine copy.
    network_.reset();
    if (strip_plan) {
      result.refit_weights.reserve(owned_refit_weights_.size());
      for (auto& weight : owned_refit_weights_) {
        result.refit_weights.push_back(
            {std::move(weight.name),
             static_cast<TensorRtWeightDataType>(weight.data_type),
             weight.count,
             std::move(owned_weights_[weight.owned_weight_index])});
      }
    }
    owned_weights_.clear();
    owned_weights_.shrink_to_fit();
    result.engine.resize(serialized->size());
    std::memcpy(result.engine.data(), serialized->data(), serialized->size());
    memory_profiler_.Log("engine_copied");
    return result;
  }

 private:
  // Whether a signature tensor's name carries `key`: the tensors carry the
  // signature's key behind the signature's name (decode_embeddings,
  // prefill_1024_input_pos, decode_logits_output), some models with an output
  // index after it (decode_input_pos:0).
  static bool HasSignatureKey(std::string name, const char* key) {
    const size_t colon = name.rfind(':');
    if (colon != std::string::npos && colon + 1 < name.size() &&
        name.find_first_not_of("0123456789", colon + 1) == std::string::npos) {
      name.erase(colon);
    }
    const std::string suffix = std::string("_") + key;
    return name == key || (name.size() > suffix.size() &&
                           name.compare(name.size() - suffix.size(),
                                        suffix.size(), suffix) == 0);
  }

  // The batched decode engine of `slots` slots built from the step's network
  // (BuildSlotDecodeNetwork), with the step's builder configuration.
  Expected<TensorRtSlotPlanBuildData> BuildSlotPlan(int32_t slots) {
    int32_t merged = 0;
    std::vector<std::unique_ptr<nvinfer1::IPluginV3>> plugins;
    LITERT_ASSIGN_OR_RETURN(
        nvinfer1::INetworkDefinition * network_ptr,
        BuildSlotDecodeNetwork(*builder_, *network_, slots, &merged, &plugins));
    TrtPtr<nvinfer1::INetworkDefinition> network(network_ptr);
    TrtPtr<nvinfer1::IHostMemory> serialized(
        builder_->buildSerializedNetwork(*network, *config_));
    if (!serialized) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to build the batched decode engine");
    }
    TensorRtSlotPlanBuildData plan;
    plan.slots = slots;
    for (int32_t s = 0; s < slots; ++s) {
      for (const auto& name : input_names_) {
        plan.input_names.push_back(SlotTensorName(name, s));
      }
      for (const auto& name : output_names_) {
        plan.output_names.push_back(SlotTensorName(name, s));
      }
    }
    plan.engine.assign(
        static_cast<const uint8_t*>(serialized->data()),
        static_cast<const uint8_t*>(serialized->data()) + serialized->size());
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX built a batched decode engine of %d slots: "
               "%d projections read once for the slots, plan_bytes=%zu",
               slots, merged, plan.engine.size());
    return plan;
  }

  // A decode step that feeds itself (TensorRtStepFeed): one token's
  // embeddings in, the logits out, the vocabulary head a fully connected op
  // with INT4 per-channel or centered INT2 weights that the weight store maps
  // (the models tie them to the embedder's table; the embedder's multiplier
  // is sqrt(dim), and the dispatch verifies every embedding the host feeds
  // against what it ran, so an untied table only costs the speculation).
  // Nothing when the signature is not that shape.
  // LITERT_NVIDIA_TENSORRT_STEP_FEED=0 records none.
  Expected<std::optional<TensorRtStepFeedBuildData>> MatchStepFeed(
      const Subgraph& subgraph) {
    const auto none = [] { return std::optional<TensorRtStepFeedBuildData>(); };
    // A signature the feed cannot follow: the line that saw why.
    const auto refuse = [&none](int line) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX step feed: not this signature "
                 "(tensorrt_graph_builder.cc:%d)",
                 line);
      return none();
    };
    if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_STEP_FEED",
                    /*default_value=*/true)) {
      return none();
    }
    const auto is_key = &HasSignatureKey;
    TensorRtStepFeedBuildData feed;
    const auto inputs = subgraph.Inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      const Tensor& input = inputs[i];
      const std::string name(input.Name());
      auto type = input.RankedTensorType();
      if (!type) continue;
      const auto dims = type->Layout().Dimensions();
      const uint32_t port = static_cast<uint32_t>(i);
      const auto all_ones_but_last = [&] {
        for (size_t d = 0; d + 1 < dims.size(); ++d) {
          if (dims[d] != 1) return false;
        }
        return !dims.empty();
      };
      if (is_key(name, "per_layer_embeddings")) {
        // [.., layers, width], the per-layer embedder's lookups of the token.
        if (input.ElementType() != litert::ElementType::Float32 ||
            dims.size() < 2 || dims[dims.size() - 2] <= 0 || dims.back() <= 0) {
          return refuse(__LINE__);
        }
        for (size_t d = 0; d + 2 < dims.size(); ++d) {
          if (dims[d] != 1) return refuse(__LINE__);
        }
        feed.per_layer_port = port;
        feed.per_layer_count = static_cast<uint32_t>(dims[dims.size() - 2]);
        feed.per_layer_width = static_cast<uint32_t>(dims.back());
      } else if (is_key(name, "embeddings")) {
        if (input.ElementType() != litert::ElementType::Float32 ||
            !all_ones_but_last() || dims.back() <= 0 || dims.back() % 2 != 0) {
          return refuse(__LINE__);
        }
        feed.embeddings_port = port;
        feed.dim = static_cast<uint32_t>(dims.back());
      } else if (is_key(name, "positions") || is_key(name, "input_pos")) {
        if (input.ElementType() != litert::ElementType::Int32 ||
            !all_ones_but_last() || dims.back() != 1) {
          return refuse(__LINE__);
        }
        feed.positions_port = port;
      } else if (is_key(name, "param_tensor")) {
        if (input.ElementType() != litert::ElementType::Int32 ||
            !all_ones_but_last() || dims.back() < 3) {
          return refuse(__LINE__);
        }
        feed.param_port = port;
      } else if (is_key(name, "local_mask") ||
                 is_key(name, "attn_mask_local") ||
                 is_key(name, "local_attn_mask")) {
        if (input.ElementType() != litert::ElementType::Bool ||
            !all_ones_but_last() || dims.back() <= 0) {
          return refuse(__LINE__);
        }
        feed.local_mask_port = port;
        feed.ring = static_cast<uint32_t>(dims.back());
      } else if (is_key(name, "attn_mask") || is_key(name, "global_mask") ||
                 is_key(name, "mask")) {
        if (input.ElementType() != litert::ElementType::Bool ||
            !all_ones_but_last()) {
          return refuse(__LINE__);
        }
        feed.global_mask_port = port;
      }
    }
    if (feed.embeddings_port == kTensorRtStepFeedNoPort ||
        feed.positions_port == kTensorRtStepFeedNoPort) {
      std::string names;
      for (size_t i = 0; i < inputs.size() && i < 8; ++i) {
        names += (i == 0 ? "" : ", ") + std::string(inputs[i].Name());
      }
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX step feed: no embeddings and positions "
                 "inputs among %zu (%s%s)",
                 inputs.size(), names.c_str(),
                 inputs.size() > 8 ? ", ..." : "");
      return none();
    }
    // The logits output, one row: by its key, or, where the outputs carry
    // the exporter's names (StatefulPartitionedCall:49), the one row output
    // a fully connected op from the embedding dim to its width produces,
    // through the elementwise tail (the head).
    const auto outputs = subgraph.Outputs();
    std::optional<Tensor> logits;
    for (size_t i = 0; i < outputs.size(); ++i) {
      const std::string name(outputs[i].Name());
      const bool named =
          is_key(name, "logits") || is_key(name, "logits_output");
      auto type = outputs[i].RankedTensorType();
      if (!type) {
        if (named) return refuse(__LINE__);
        continue;
      }
      const auto dims = type->Layout().Dimensions();
      bool one_row = !dims.empty() && dims.back() > 0;
      for (size_t d = 0; d + 1 < dims.size(); ++d) {
        if (dims[d] != 1) one_row = false;
      }
      if (!one_row) {
        if (named) return refuse(__LINE__);
        continue;
      }
      if (!named) {
        auto head = FindHeadBehind(outputs[i]);
        if (!head.has_value() || head->Inputs().size() < 2) continue;
        auto weights_type = head->Inputs()[1].RankedTensorType();
        if (!weights_type || weights_type->Layout().Rank() != 2 ||
            weights_type->Layout().Dimensions()[0] != dims.back() ||
            weights_type->Layout().Dimensions()[1] !=
                static_cast<int32_t>(feed.dim)) {
          continue;
        }
      }
      feed.logits_output_port = static_cast<uint32_t>(i);
      feed.vocab = static_cast<uint32_t>(dims.back());
      logits = outputs[i];
      if (named) break;
    }
    if (!logits.has_value()) return refuse(__LINE__);
    // The token table: the head's weights when they are an INT4 per-channel
    // or a centered INT2 [vocab, dim] view of the model file (the embedder's
    // table is tied to them, with the multiplier sqrt(dim)); otherwise the
    // runtime registers the embedder's own table (an external head, a
    // blockwise embedder).
    feed.table_offset = kTensorRtStepFeedNoTable;
    feed.multiplier =
        static_cast<float>(std::sqrt(static_cast<double>(feed.dim)));
    MatchTiedHeadTable(*logits, feed);
    if (feed.local_mask_port != kTensorRtStepFeedNoPort) {
      // The sliding window is the runtime's (the model's metadata); the
      // graph shows it only where a block reads its cache through a window.
      // Unknown (0), the dispatch learns it from the host's masks.
      for (const auto& read : windowed_reads_) {
        feed.window = std::max(feed.window, static_cast<uint32_t>(read.window));
      }
      if (feed.window > feed.ring) feed.window = 0;
    }
    const std::string table_text =
        feed.table_offset == kTensorRtStepFeedNoTable
            ? "from the runtime"
            : "INT" + std::to_string(feed.table_bits) + " at " +
                  std::to_string(feed.table_offset);
    const std::string per_layer_text =
        feed.per_layer_port == kTensorRtStepFeedNoPort
            ? "-"
            : std::to_string(feed.per_layer_port) + " [" +
                  std::to_string(feed.per_layer_count) + ", " +
                  std::to_string(feed.per_layer_width) + "]";
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX step feed: embeddings port %u, positions "
               "%u, param %s, local mask %s (ring %u, window %u), global mask "
               "%s, per-layer %s, logits output %u, table [%u, %u] %s, "
               "multiplier %g",
               feed.embeddings_port, feed.positions_port,
               feed.param_port == kTensorRtStepFeedNoPort
                   ? "-"
                   : std::to_string(feed.param_port).c_str(),
               feed.local_mask_port == kTensorRtStepFeedNoPort
                   ? "-"
                   : std::to_string(feed.local_mask_port).c_str(),
               feed.ring, feed.window,
               feed.global_mask_port == kTensorRtStepFeedNoPort
                   ? "-"
                   : std::to_string(feed.global_mask_port).c_str(),
               per_layer_text.c_str(), feed.logits_output_port, feed.vocab,
               feed.dim, table_text.c_str(), feed.multiplier);
    return std::optional<TensorRtStepFeedBuildData>(std::move(feed));
  }

  // The fully connected op behind a tensor through the elementwise tail
  // (the soft cap, casts, reshapes), or none.
  static std::optional<Op> FindHeadBehind(const Tensor& tensor) {
    Tensor value = tensor;
    for (int hops = 0; hops < 12; ++hops) {
      auto producer = value.GetDefiningOp();
      if (!producer) return std::nullopt;
      switch (producer->Code()) {
        case kLiteRtOpCodeTflFullyConnected:
          return *producer;
        case kLiteRtOpCodeTflCast:
        case kLiteRtOpCodeTflReshape:
        case kLiteRtOpCodeTflTanh:
          if (producer->Inputs().empty()) return std::nullopt;
          value = producer->Inputs()[0];
          break;
        case kLiteRtOpCodeTflMul:
        case kLiteRtOpCodeTflDiv:
          if (producer->Inputs().size() != 2) return std::nullopt;
          // The other input is the scalar.
          value =
              producer
                  ->Inputs()[ReadFloatScalar(producer->Inputs()[0]) ? 1 : 0];
          break;
        default:
          return std::nullopt;
      }
    }
    return std::nullopt;
  }

  // The head's weights as the feed's table when they are the tied
  // embedder's: [vocab, dim] in the model file, INT4 per-channel, or
  // centered INT2 with a scale per row (CenteredInt2Scales: a weight is
  // scale * (q + 0.5), which the embedder's blockwise lookup computes too).
  void MatchTiedHeadTable(const Tensor& logits,
                          TensorRtStepFeedBuildData& feed) {
    auto head = FindHeadBehind(logits);
    if (!head.has_value() || head->Inputs().size() < 2) return;
    const Tensor weights = head->Inputs()[1];
    auto weights_type = weights.RankedTensorType();
    if (!weights_type || !weights.HasWeights() ||
        weights_type->Layout().Rank() != 2 ||
        weights_type->Layout().Dimensions()[0] !=
            static_cast<int32_t>(feed.vocab) ||
        weights_type->Layout().Dimensions()[1] !=
            static_cast<int32_t>(feed.dim)) {
      return;
    }
    if (weights.ElementType() == litert::ElementType::Int2) {
      const auto scales = CenteredInt2Scales(weights, /*input_axis=*/1);
      if (!scales.has_value() || scales->size() != feed.vocab ||
          feed.dim % 4 != 0 || weight_store_ == nullptr) {
        return;
      }
      auto offset = weight_store_->SourceOffset(weights.Weights().Bytes());
      if (!offset) {
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT-RTX step feed: the head's weights are not "
                   "a view of the model file (%s)",
                   offset.Error().Message().c_str());
        return;
      }
      feed.table_offset = *offset;
      feed.table_bits = 2;
      feed.table_zero_point = -0.5f;
      feed.row_scales.resize(scales->size() * sizeof(float));
      std::memcpy(feed.row_scales.data(), scales->data(),
                  feed.row_scales.size());
      return;
    }
    if (weights.ElementType() != litert::ElementType::Int4 ||
        weights.QTypeId() != kLiteRtQuantizationPerChannel) {
      return;
    }
    const auto q = weights.PerChannelQuantization();
    if (q.quantized_dimension != 0 || q.num_channels != feed.vocab ||
        q.scales == nullptr) {
      return;
    }
    // The feed dequantizes the table as symmetric: a head with a zero point
    // that is not zero gets no tied table (the runtime may register one).
    if (q.zero_points != nullptr) {
      for (uint64_t i = 0; i < q.num_channels; ++i) {
        if (q.zero_points[i] != 0) {
          LITERT_LOG(
              LITERT_INFO,
              "NVIDIA TensorRT-RTX step feed: the head's weights have a "
              "zero point that is not zero (channel %llu): no tied table",
              static_cast<unsigned long long>(i));
          return;
        }
      }
    }
    const absl::Span<const uint8_t> bytes = weights.Weights().Bytes();
    if (bytes.size() != static_cast<size_t>(feed.vocab) * feed.dim / 2 ||
        weight_store_ == nullptr) {
      return;  // without a weight store the table has no place in the file
    }
    auto offset = weight_store_->SourceOffset(bytes);
    if (!offset) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX step feed: the head's weights are not a "
                 "view of the model file (%s)",
                 offset.Error().Message().c_str());
      return;
    }
    feed.table_offset = *offset;
    feed.row_scales.resize(static_cast<size_t>(feed.vocab) * sizeof(float));
    std::memcpy(feed.row_scales.data(), q.scales, feed.row_scales.size());
  }

  // Limits odml.runtime_bmm to a static KV-cache prefix, then restores the
  // graph's fixed output shape where needed. This is correct only when the
  // runtime never attends beyond the configured prefix.
  static size_t RuntimeBmmContextLimit() {
    return EnvSizeT("LITERT_NVIDIA_TENSORRT_RUNTIME_BMM_CONTEXT_LIMIT",
                    /*default_value=*/0);
  }

  // bf16 keeps fp32's exponent range, sidestepping overflow in reductions and
  // exp-heavy regions at slightly coarser mantissa. That is the default, for
  // the Gemma 4 and Qwen 3.5 graphs, which compute in FLOAT32 (with FLOAT16
  // casts around some norms). A graph whose quantized products sit between
  // FLOAT16 tensors (a residual stream, attention results and head in FLOAT16,
  // every product INT8) has chosen its type: the values between its quant
  // layers are FP16, and BF16 would drop three bits of every one of them (the
  // first token of a reply changed with it).
  static nvinfer1::DataType ModeFloatType() {
    if (TensorRtActivationMode() == ActivationMode::kBf16 &&
        !GraphComputesFloat16()) {
      return nvinfer1::DataType::kBF16;
    }
    return nvinfer1::DataType::kHALF;
  }

  // Whether the subgraph being built computes in FLOAT16: set by Build.
  static bool& GraphComputesFloat16() {
    thread_local bool value = false;
    return value;
  }

  // A statically quantized product between FLOAT16 tensors (the ops of
  // FindQuantRegions), in the graph's own ops or carved into a composite.
  static bool SubgraphComputesFloat16(const Subgraph& subgraph) {
    for (const auto& op : subgraph.Ops()) {
      if (IsQuantFullyConnected(op) || IsQuantLogitsFullyConnected(op) ||
          IsQuantBatchMatmul(op) ||
          ParseQuantFactorizedMlpComposite(op).has_value()) {
        return true;
      }
    }
    return false;
  }

  const char* KeepName(std::string name) {
    names_.push_back(std::move(name));
    return names_.back().c_str();
  }

  std::string UniqueName(std::string base) {
    base += "_trt_";
    base += std::to_string(next_name_id_++);
    return base;
  }

  Expected<void> RegisterRefitWeight(nvinfer1::Weights weights,
                                     const std::string& base_name) {
    if (!TensorRtSharedWeightsEnabled() || weights.values == nullptr ||
        weights.count <= 0) {
      return {};
    }
    // TensorRT-RTX 1.5.0 build 114 fails in stdArchiveRecorder when an INT8
    // constant is marked for stripping, even though equivalent INT4 and
    // floating-point constants serialize and refit correctly. Leave INT8
    // weights embedded until the SDK recorder supports them.
    if (weights.type == nvinfer1::DataType::kINT8) {
      return {};
    }
    for (const auto& existing : owned_refit_weights_) {
      if (owned_weights_[existing.owned_weight_index].data() ==
          weights.values) {
        return {};
      }
    }
    size_t owned_weight_index = owned_weights_.size();
    for (size_t i = 0; i < owned_weights_.size(); ++i) {
      if (owned_weights_[i].data() == weights.values) {
        owned_weight_index = i;
        break;
      }
    }
    if (owned_weight_index == owned_weights_.size()) {
      return Error(kLiteRtStatusErrorCompilation,
                   "TensorRT refit weight is not compiler-owned");
    }
    std::string name = UniqueName("refit_" + base_name);
    const char* kept_name = KeepName(name);
    if (!network_->setWeightsName(weights, kept_name)) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to name TensorRT refit weight");
    }
    if (!network_->markWeightsRefittable(kept_name)) {
      LITERT_LOG(LITERT_VERBOSE,
                 "NVIDIA TensorRT-RTX weight cannot be refitted: %s",
                 kept_name);
      return {};
    }
    owned_refit_weights_.push_back(
        {std::move(name), static_cast<int32_t>(weights.type),
         static_cast<uint64_t>(weights.count), owned_weight_index});
    return {};
  }

  Expected<void> AddInputs(const Subgraph& subgraph) {
    auto inputs = subgraph.Inputs();
    input_names_.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      LITERT_ASSIGN_OR_RETURN(auto type, input.RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
      // A UINT8 cache of the general forms crosses as its bytes
      // (FindGeneralCaches).
      LITERT_ASSIGN_OR_RETURN(
          auto data_type,
          byte_caches_.count(input.Get()) != 0
              ? Expected<nvinfer1::DataType>(nvinfer1::DataType::kINT8)
              : ConvertDataType(type.ElementType()));
      std::string name = TensorName("input", input, i);
      // An INT4 cache of the composites over quantized caches is declared as
      // the INT8 bytes that hold it (FindPackedInt4Caches).
      if (const auto packed = packed_int4_caches_.find(input.Get());
          packed != packed_int4_caches_.end()) {
        data_type = nvinfer1::DataType::kINT8;
        dims.d[packed->second ? 2 : 3] /= 2;
      }
      // A value cache stored as [B, H, D, S] and updated in place is either
      // held as [B, H, S, D] (see TransposedValueCacheEnabled) or declared
      // through its free [B, H*D, S, 1] view so the native cache update can
      // write positions along its last axis. The same bytes either way, so
      // dispatch binds the model's buffer unchanged. Read-only caches opt in
      // explicitly when sharing a producer's transposed buffer; dispatch
      // validates that physical layout using the advertised element strides.
      const auto in_place = in_place_caches_.find(input.Get());
      // A float32 cache updated in place is held as FP16 (see
      // Fp16CacheStorageEnabled); the runtime's buffer is twice as large. A
      // cache another engine keeps in place (shared_cache_inputs_) is read
      // the way that engine stores it.
      if ((in_place != in_place_caches_.end() ||
           shared_cache_inputs_.count(input.Get()) != 0) &&
          type.ElementType() == litert::ElementType::Float32 &&
          Fp16CacheStorageEnabled()) {
        data_type = nvinfer1::DataType::kHALF;
        fp16_stored_caches_.insert(input.Get());
      }
      const bool value_cache = in_place != in_place_caches_.end() &&
                               in_place->second.partial_axis == 3;
      const bool value_transposed = transposed_caches_.count(input.Get()) != 0;
      const bool value_view = value_cache && !value_transposed;
      nvinfer1::Dims declared_dims = dims;
      if (value_transposed) {
        name += kTransposedValueCacheSuffix;
        declared_dims.d[2] = dims.d[3];
        declared_dims.d[3] = dims.d[2];
      } else if (value_view) {
        declared_dims.nbDims = 4;
        declared_dims.d[0] = dims.d[0];
        declared_dims.d[1] = dims.d[1] * dims.d[2];
        declared_dims.d[2] = dims.d[3];
        declared_dims.d[3] = 1;
      }
      auto* tensor =
          network_->addInput(KeepName(name), data_type, declared_dims);
      if (tensor == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT input");
      }
      raw_input_map_[input.Get()] = tensor;
      // The token positions of a prefill's rows: the CUDA GEMM's int8 path
      // takes them to tell a chunk's rows from its padding
      // (AddSubbyteGemvPlugin). Two such inputs leave both out.
      if (type.ElementType() == litert::ElementType::Int32 &&
          (HasSignatureKey(std::string(input.Name()), "input_pos") ||
           HasSignatureKey(std::string(input.Name()), "positions"))) {
        int64_t count = 1;
        for (int d = 0; d < dims.nbDims; ++d) count *= dims.d[d];
        row_positions_ = row_positions_count_ == 0 ? tensor : nullptr;
        row_positions_count_ = row_positions_count_ == 0 ? count : -1;
      }
      nvinfer1::ITensor* value = tensor;
      if (value_view) {
        LITERT_ASSIGN_OR_RETURN(value, ReshapeTensor(tensor, dims));
      }
      if (Fp16ActivationsEnabled()) {
        LITERT_ASSIGN_OR_RETURN(bool int8_quantized,
                                HasSymmetricPerTensorInt8Quantization(input));
        if (int8_quantized) {
          LITERT_ASSIGN_OR_RETURN(
              value, AddDequantizeTensor(input, tensor, ModeFloatType()));
        } else if (data_type == nvinfer1::DataType::kFLOAT) {
          LITERT_ASSIGN_OR_RETURN(value,
                                  AddCastTensor(tensor, ModeFloatType()));
        }
      }
      tensor_map_[input.Get()] = value;
      input_names_.push_back(name);
    }
    return {};
  }

  Expected<void> MarkOutputs(
      const Subgraph& subgraph,
      const std::optional<TensorRtLlmHeadBuildData>& trtllm_head) {
    auto outputs = subgraph.Outputs();
    output_names_.reserve(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      if (trtllm_head.has_value() && i == trtllm_head->logits_output_port) {
        // LiteRT still exposes this port; NVIDIA dispatch writes it with the
        // external W2 head after the TensorRT-RTX prefix.
        output_names_.emplace_back();
        continue;
      }
      const auto& output = outputs[i];
      auto it = tensor_map_.find(output.Get());
      if (it == tensor_map_.end() || it->second == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Subgraph output was not produced by TensorRT network");
      }
      nvinfer1::ITensor* out = it->second;
      // Prefer a raw boundary-format tensor when one was recorded (cache
      // updates); in FP16 mode otherwise convert the in-network FP16 value
      // back to the tensor's boundary format (int8 for quantized, fp32 for
      // float).
      if (auto raw = raw_output_map_.find(output.Get());
          raw != raw_output_map_.end()) {
        out = raw->second;
      }
      if (Fp16ActivationsEnabled()) {
        LITERT_ASSIGN_OR_RETURN(auto type, output.RankedTensorType());
        LITERT_ASSIGN_OR_RETURN(
            auto boundary_type,
            byte_caches_.count(output.Get()) != 0
                ? Expected<nvinfer1::DataType>(nvinfer1::DataType::kINT8)
                : ConvertDataType(type.ElementType()));
        if (const auto in_place = in_place_outputs_.find(output.Get());
            in_place != in_place_outputs_.end() &&
            type.ElementType() == litert::ElementType::Float32 &&
            Fp16CacheStorageEnabled()) {
          boundary_type = nvinfer1::DataType::kHALF;
        }
        if (packed_int4_caches_.count(output.Get()) != 0) {
          boundary_type = nvinfer1::DataType::kINT8;
        }
        if (out->getType() != boundary_type) {
          LITERT_ASSIGN_OR_RETURN(
              bool int8_quantized,
              HasSymmetricPerTensorInt8Quantization(output));
          if (int8_quantized && out->getType() != nvinfer1::DataType::kINT8) {
            LITERT_ASSIGN_OR_RETURN(
                out, AddQuantizeTensor(out, output, nvinfer1::DataType::kINT8));
          } else if (!int8_quantized) {
            LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, boundary_type));
          }
        }
      }
      std::string name = TensorName("output", output, i);
      if (transposed_caches_.count(output.Get()) != 0) {
        name += kTransposedValueCacheSuffix;
      }
      out->setName(KeepName(name));
      network_->markOutput(*out);
      output_names_.push_back(name);
    }
    return {};
  }

  Expected<nvinfer1::Weights> MakeWeights(const Tensor& tensor) {
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
    if (type.ElementType() == litert::ElementType::Int2) {
      return MakeInt2WeightsAsInt4(tensor, num_elements);
    }
    auto data_type_or = ConvertDataType(type.ElementType());
    if (!data_type_or) {
      return Error(data_type_or.Error().Status(),
                   data_type_or.Error().Message() + " for constant tensor " +
                       std::to_string(tensor.TensorIndex()));
    }
    auto data_type = *data_type_or;
    size_t expected_bytes = 0;
    if (type.ElementType() == litert::ElementType::Int4) {
      expected_bytes = (num_elements + 1) / 2;
    } else {
      LITERT_ASSIGN_OR_RETURN(size_t element_size,
                              ElementByteSize(type.ElementType()));
      expected_bytes = num_elements * element_size;
    }
    const auto bytes = tensor.Weights().Bytes();
    if (bytes.size() != expected_bytes) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected TensorRT weight buffer size");
    }
    owned_weights_.emplace_back(bytes.begin(), bytes.end());
    return nvinfer1::Weights{data_type, owned_weights_.back().data(),
                             static_cast<int64_t>(num_elements)};
  }

  // TensorRT has no 2-bit type; widen TFLite INT2 constants (four values per
  // byte, little-endian bit fields) to TensorRT INT4 nibbles (two values per
  // byte, low nibble first). Values are in [-2, 1], so they fit INT4 exactly
  // and the quantization scales carry over unchanged.
  Expected<nvinfer1::Weights> MakeInt2WeightsAsInt4(const Tensor& tensor,
                                                    size_t num_elements) {
    const auto bytes = tensor.Weights().Bytes();
    if (bytes.size() != (num_elements + 3) / 4) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected TensorRT INT2 weight buffer size");
    }
    owned_weights_.emplace_back((num_elements + 1) / 2);
    auto& packed = owned_weights_.back();
    for (size_t i = 0; i < num_elements; ++i) {
      const uint8_t byte = static_cast<uint8_t>(bytes[i / 4]);
      const int shift = static_cast<int>(i % 4) * 2;
      // Sign-extend the 2-bit field to int8, then keep its low nibble.
      const int8_t value =
          static_cast<int8_t>(static_cast<uint8_t>(byte << (6 - shift))) >> 6;
      const uint8_t nibble = static_cast<uint8_t>(value) & 0x0F;
      if (i % 2 == 0) {
        packed[i / 2] = nibble;
      } else {
        packed[i / 2] |= nibble << 4;
      }
    }
    return nvinfer1::Weights{nvinfer1::DataType::kINT4, packed.data(),
                             static_cast<int64_t>(num_elements)};
  }

  Expected<nvinfer1::Weights> MakeConv2dWeightsOihw(const Tensor& tensor) {
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    if (type.ElementType() != litert::ElementType::Float32 &&
        type.ElementType() != litert::ElementType::Float16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Conv2D weights must be FP32 or FP16");
    }
    const auto dims = type.Layout().Dimensions();
    const int out_channels = dims[0];
    const int kernel_h = dims[1];
    const int kernel_w = dims[2];
    const int in_channels = dims[3];
    LITERT_ASSIGN_OR_RETURN(size_t element_size,
                            ElementByteSize(type.ElementType()));
    auto src = tensor.Weights().Bytes();
    if (src.size() != static_cast<size_t>(out_channels) * kernel_h * kernel_w *
                          in_channels * element_size) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected Conv2D filter size");
    }
    owned_weights_.emplace_back(src.size());
    auto& dst = owned_weights_.back();
    for (int o = 0; o < out_channels; ++o) {
      for (int i = 0; i < in_channels; ++i) {
        for (int h = 0; h < kernel_h; ++h) {
          for (int w = 0; w < kernel_w; ++w) {
            const size_t src_index =
                (((o * kernel_h + h) * kernel_w + w) * in_channels + i) *
                element_size;
            const size_t dst_index =
                (((o * in_channels + i) * kernel_h + h) * kernel_w + w) *
                element_size;
            std::memcpy(dst.data() + dst_index, src.data() + src_index,
                        element_size);
          }
        }
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto data_type,
                            ConvertDataType(type.ElementType()));
    return nvinfer1::Weights{
        data_type, dst.data(),
        static_cast<int64_t>(out_channels) * in_channels * kernel_h * kernel_w};
  }

  static uint16_t Fp32ToFp16Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exponent == 128) {  // Inf/NaN.
      return sign | 0x7C00 | (mantissa != 0 ? 0x200 : 0);
    }
    if (exponent > 15) {  // Overflow -> Inf.
      return sign | 0x7C00;
    }
    if (exponent >= -14) {  // Normal half.
      // Round mantissa from 23 to 10 bits, to nearest even.
      uint32_t half = ((exponent + 15) << 10) | (mantissa >> 13);
      const uint32_t round_bits = mantissa & 0x1FFF;
      if (round_bits > 0x1000 || (round_bits == 0x1000 && (half & 1))) {
        ++half;  // Carry may bump the exponent; the encoding stays valid.
      }
      return sign | half;
    }
    if (exponent >= -24) {  // Subnormal half.
      mantissa |= 0x800000;
      const int shift = -14 - exponent;
      uint32_t half = mantissa >> (13 + shift);
      const uint32_t round_mask = (1u << (13 + shift)) - 1;
      const uint32_t round_bits = mantissa & round_mask;
      const uint32_t halfway = 1u << (12 + shift);
      if (round_bits > halfway || (round_bits == halfway && (half & 1))) {
        ++half;
      }
      return sign | half;
    }
    return sign;  // Underflow -> signed zero.
  }

  static uint16_t Fp32ToBf16Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + rounding) >> 16);
  }

  Expected<nvinfer1::ITensor*> AddFloatConstant(
      absl::Span<const float> values, nvinfer1::Dims dims,
      const std::string& name,
      nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT) {
    if (data_type == nvinfer1::DataType::kHALF) {
      owned_weights_.emplace_back(values.size() * sizeof(uint16_t));
      auto& storage = owned_weights_.back();
      auto* half_data = reinterpret_cast<uint16_t*>(storage.data());
      for (size_t i = 0; i < values.size(); ++i) {
        half_data[i] = Fp32ToFp16Bits(values[i]);
      }
    } else if (data_type == nvinfer1::DataType::kBF16) {
      // bf16 is the top 16 bits of the fp32 encoding (round-to-nearest-even).
      owned_weights_.emplace_back(values.size() * sizeof(uint16_t));
      auto& storage = owned_weights_.back();
      auto* bf16_data = reinterpret_cast<uint16_t*>(storage.data());
      for (size_t i = 0; i < values.size(); ++i) {
        bf16_data[i] = Fp32ToBf16Bits(values[i]);
      }
    } else if (data_type == nvinfer1::DataType::kFLOAT) {
      owned_weights_.emplace_back(values.size() * sizeof(float));
      auto& storage = owned_weights_.back();
      std::memcpy(storage.data(), values.data(), storage.size());
    } else {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported TensorRT float constant type");
    }
    nvinfer1::Weights weights{data_type, owned_weights_.back().data(),
                              static_cast<int64_t>(values.size())};
    auto* layer = network_->addConstant(dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT float constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(weights, name));
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    return layer->getOutput(0);
  }

  // An all-false boolean constant.
  Expected<nvinfer1::ITensor*> AddFalseConstant(nvinfer1::Dims dims,
                                                const std::string& name) {
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
      count *= static_cast<size_t>(dims.d[i]);
    }
    owned_weights_.emplace_back(count, 0);
    nvinfer1::Weights weights{nvinfer1::DataType::kBOOL,
                              owned_weights_.back().data(),
                              static_cast<int64_t>(count)};
    auto* layer = network_->addConstant(dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT boolean constant");
    }
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    return layer->getOutput(0);
  }

  Expected<int8_t> ReadQuantizedConstantValue(const Tensor& tensor,
                                              size_t index) {
    const auto bytes = tensor.Weights().Bytes();
    switch (tensor.ElementType()) {
      case litert::ElementType::Int8:
        if (index >= bytes.size()) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Unexpected INT8 weight index");
        }
        return static_cast<int8_t>(bytes[index]);
      case litert::ElementType::Int4: {
        if (index / 2 >= bytes.size()) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Unexpected INT4 weight index");
        }
        const uint8_t byte = static_cast<uint8_t>(bytes[index / 2]);
        const uint8_t nibble =
            (index % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        return static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
      }
      case litert::ElementType::Int2: {
        if (index / 4 >= bytes.size()) {
          return Error(kLiteRtStatusErrorInvalidArgument,
                       "Unexpected INT2 weight index");
        }
        const uint8_t byte = static_cast<uint8_t>(bytes[index / 4]);
        const int shift = static_cast<int>(index % 4) * 2;
        return static_cast<int8_t>(static_cast<uint8_t>(byte << (6 - shift))) >>
               6;
      }
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported predequantized FC weight type");
    }
  }

  Expected<nvinfer1::ITensor*> AddPredequantizedFcWeights(
      const Tensor& tensor, nvinfer1::DataType data_type) {
    if (data_type != nvinfer1::DataType::kHALF &&
        data_type != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Predequantized FC weights require FP16 or BF16");
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    if (tensor.QTypeId() != kLiteRtQuantizationPerChannel) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Predequantized FC weights require per-channel scales");
    }
    auto q = tensor.PerChannelQuantization();
    if (q.num_channels == 0 || q.scales == nullptr ||
        q.zero_points == nullptr) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Invalid predequantized FC weight quantization");
    }
    const auto dims = type.Layout().Dimensions();
    const int32_t axis = q.quantized_dimension;
    if (axis < 0 || axis >= dims.size() ||
        dims[axis] != static_cast<int32_t>(q.num_channels)) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Predequantized FC weight scale axis does not match shape");
    }
    for (uint64_t i = 0; i < q.num_channels; ++i) {
      if (q.scales[i] <= 0.0f || q.zero_points[i] != 0) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Predequantized FC weights require symmetric scales");
      }
    }
    LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
    size_t inner = 1;
    for (int i = axis + 1; i < dims.size(); ++i) {
      inner *= dims[i];
    }
    owned_weights_.emplace_back(num_elements * sizeof(uint16_t));
    auto& storage = owned_weights_.back();
    auto* dst = reinterpret_cast<uint16_t*>(storage.data());
    for (size_t i = 0; i < num_elements; ++i) {
      const size_t channel = (i / inner) % q.num_channels;
      LITERT_ASSIGN_OR_RETURN(const int8_t quantized,
                              ReadQuantizedConstantValue(tensor, i));
      const float value = static_cast<float>(quantized) * q.scales[channel];
      dst[i] = data_type == nvinfer1::DataType::kHALF ? Fp32ToFp16Bits(value)
                                                      : Fp32ToBf16Bits(value);
    }
    LITERT_ASSIGN_OR_RETURN(auto trt_dims, ConvertDims(type));
    nvinfer1::Weights weights{data_type, storage.data(),
                              static_cast<int64_t>(num_elements)};
    auto* layer = network_->addConstant(trt_dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add predequantized FC weight constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        weights,
        "predequantized_fc_weights_" + std::to_string(tensor.TensorIndex())));
    layer->getOutput(0)->setName(KeepName(UniqueName(
        "predequantized_fc_weights_" + std::to_string(tensor.TensorIndex()))));
    return layer->getOutput(0);
  }

  // Encodes int4/int2 weight values as fp8-e4m3 constants (exact for
  // integers in [-8, 7]) and dequantizes with the per-channel scales.
  // Myelin fuses this fp8-weight dequantize into the GEMV, so decode reads
  // 1 byte/element with no per-step weight materialization.
  Expected<nvinfer1::ITensor*> AddFp8PredequantizedFcWeights(
      const Tensor& tensor, nvinfer1::DataType compute_type) {
    if (compute_type != nvinfer1::DataType::kHALF &&
        compute_type != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "FP8 predequantized FC weights require FP16 or BF16");
    }
    const auto element_type = tensor.ElementType();
    if (element_type != litert::ElementType::Int4 &&
        element_type != litert::ElementType::Int2) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "FP8 predequantized FC weights require int4/int2 values");
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    if (tensor.QTypeId() != kLiteRtQuantizationPerChannel) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "FP8 predequantized FC weights require per-channel scales");
    }
    auto q = tensor.PerChannelQuantization();
    if (q.num_channels == 0 || q.scales == nullptr ||
        q.zero_points == nullptr) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Invalid FP8 predequantized FC weight quantization");
    }
    LITERT_ASSIGN_OR_RETURN(size_t num_elements, NumElements(type));
    // e4m3 encodings of the integers -8..7 (index = value + 8).
    static constexpr uint8_t kInt4ToE4m3[16] = {
        0xD0, 0xCE, 0xCC, 0xCA, 0xC8, 0xC4, 0xC0, 0xB8,
        0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E};
    const auto bytes = tensor.Weights().Bytes();
    owned_weights_.emplace_back(num_elements);
    auto& storage = owned_weights_.back();
    auto* dst = reinterpret_cast<uint8_t*>(storage.data());
    if (element_type == litert::ElementType::Int4) {
      if (bytes.size() * 2 < num_elements) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Unexpected INT4 weight byte count");
      }
      for (size_t i = 0; i < num_elements; ++i) {
        const uint8_t byte = static_cast<uint8_t>(bytes[i / 2]);
        const uint8_t nibble =
            (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        const int8_t value =
            static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
        dst[i] = kInt4ToE4m3[value + 8];
      }
    } else {
      if (bytes.size() * 4 < num_elements) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Unexpected INT2 weight byte count");
      }
      for (size_t i = 0; i < num_elements; ++i) {
        const uint8_t byte = static_cast<uint8_t>(bytes[i / 4]);
        const int shift = static_cast<int>(i % 4) * 2;
        const int8_t value =
            static_cast<int8_t>(static_cast<uint8_t>(byte << (6 - shift))) >> 6;
        dst[i] = kInt4ToE4m3[value + 8];
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto trt_dims, ConvertDims(type));
    nvinfer1::Weights weights{nvinfer1::DataType::kFP8, storage.data(),
                              static_cast<int64_t>(num_elements)};
    auto* constant = network_->addConstant(trt_dims, weights);
    if (constant == nullptr || constant->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add FP8 FC weight constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        weights, "fp8_fc_weights_" + std::to_string(tensor.TensorIndex())));
    constant->getOutput(0)->setName(KeepName(
        UniqueName("fp8_fc_weights_" + std::to_string(tensor.TensorIndex()))));
    int32_t axis = -1;
    LITERT_ASSIGN_OR_RETURN(
        auto* scale, AddQuantizationScaleTensor(
                         tensor, axis, QuantizationScaleType(compute_type)));
    auto* dq =
        network_->addDequantize(*constant->getOutput(0), *scale, compute_type);
    if (dq == nullptr || dq->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add FP8 FC weight dequantize layer");
    }
    if (axis >= 0) {
      dq->setAxis(axis);
    }
    return dq->getOutput(0);
  }

  struct SubbyteGemvWeights {
    int32_t bit_width = 0;
    int32_t rows = 0;
    int32_t columns = 0;
    // 0: per-channel scales. 32: blockwise, `scales` the FP16 or FP32
    // scales tensor [rows, columns / 32] (IsBlockwiseInt4Weights).
    int32_t scale_block = 0;
    std::optional<Tensor> scales;
  };

  // Validates a constant weight tensor for the CUDA subbyte GEMV plugin.
  Expected<SubbyteGemvWeights> InspectSubbyteGemvWeights(const Tensor& tensor) {
    const auto element_type = tensor.ElementType();
    if (element_type != litert::ElementType::Int2 &&
        element_type != litert::ElementType::Int4 &&
        element_type != litert::ElementType::Int8) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires INT2, INT4 or INT8 weights");
    }
    if (!tensor.HasWeights()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires constant weights");
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    const auto dims = type.Layout().Dimensions();
    if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0 || dims[1] % 16 != 0) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires aligned rank-2 weights");
    }
    if (tensor.QTypeId() == kLiteRtQuantizationBlockWise) {
      if (!IsBlockwiseInt4Weights(tensor)) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "CUDA subbyte GEMV requires blockwise INT4 weights with "
                     "FP16 or FP32 scales per 32 input dims");
      }
      SubbyteGemvWeights info;
      info.bit_width = 4;
      info.rows = dims[0];
      info.columns = dims[1];
      info.scale_block = 32;
      info.scales =
          Tensor(tensor.Context(), tensor.BlockWiseQuantization().scales);
      const size_t expected_bytes = static_cast<size_t>(dims[0]) * dims[1] / 2;
      if (tensor.Weights().Bytes().size() != expected_bytes) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Unexpected CUDA subbyte GEMV weight byte count");
      }
      return info;
    }
    if (tensor.QTypeId() != kLiteRtQuantizationPerChannel) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires per-channel scales");
    }
    const auto q = tensor.PerChannelQuantization();
    if (q.quantized_dimension != 0 || q.num_channels != dims[0] ||
        q.scales == nullptr || q.zero_points == nullptr) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Invalid CUDA subbyte GEMV weight quantization");
    }
    for (uint64_t i = 0; i < q.num_channels; ++i) {
      if (q.scales[i] <= 0.0f || q.zero_points[i] != 0) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "CUDA subbyte GEMV requires symmetric scales");
      }
    }
    SubbyteGemvWeights info;
    info.bit_width = element_type == litert::ElementType::Int2   ? 2
                     : element_type == litert::ElementType::Int4 ? 4
                                                                 : 8;
    info.rows = dims[0];
    info.columns = dims[1];
    const size_t expected_bytes =
        (static_cast<size_t>(dims[0]) * dims[1] * info.bit_width + 7) / 8;
    if (tensor.Weights().Bytes().size() != expected_bytes) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected CUDA subbyte GEMV weight byte count");
    }
    return info;
  }

  Expected<void> ValidateSubbyteGemvActivation(nvinfer1::ITensor* activation,
                                               nvinfer1::DataType compute_type,
                                               int32_t columns, int bit_width,
                                               int32_t scale_block = 0) {
    if (activation == nullptr || compute_type != nvinfer1::DataType::kBF16 ||
        activation->getType() != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires BF16 activations");
    }
    const auto dims = activation->getDimensions();
    if (dims.nbDims < 1 || dims.d[dims.nbDims - 1] != columns) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV activation shape does not match weights");
    }
    const int64_t rows = GemmRows(activation);
    if (!BatchedGemvTakes(rows) &&
        !CudaGemmTakes(rows, columns, bit_width, scale_block)) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV requires a few static rows, or rows the "
                   "CUDA GEMM takes");
    }
    return {};
  }

  // An RMS norm fused into a GEMV plugin: the activation the plugin reads is
  // the norm's input, and the kernel applies rsqrt(mean(x^2) + eps) * scale
  // * multiplier (the scalar multiplies between the norm and the projection,
  // Gemma 4 26B-A4B's router scale) itself.
  struct NormFusion {
    nvinfer1::ITensor* input = nullptr;  // the norm's input, BF16
    nvinfer1::ITensor* scale = nullptr;  // FP32 [columns]
    float eps = 1e-6f;
    float multiplier = 1.0f;
  };

  // A cast between float types, whose input the fusion reads instead, with
  // LITERT_NVIDIA_TENSORRT_NORM_THROUGH_CASTS=1. By default the fusion stops
  // at a cast: the projections behind the Gemma 4 12B's casts take
  // TensorRT's norm and the casts, as the model's graph has them. Through
  // the casts the 12B passed fewer LiveCodeBench v6 problems under greedy
  // decoding (with the int8 GEMM 96 against 103, the fp16 GEMM 92 against
  // 101, of 175). Of the four Gemma 4 models only the 12B has casts between
  // float types, so only its engines depend on the switch. It is part of the
  // AOT cache key (kCompileEnvironmentVariables in compiler_plugin.cc), which
  // needs LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR, as every variable in that
  // list does.
  static bool IsFloatCast(const Op& op) {
    return EnvEnabled("LITERT_NVIDIA_TENSORRT_NORM_THROUGH_CASTS",
                      /*default_value=*/false) &&
           op.Code() == kLiteRtOpCodeTflCast && op.Inputs().size() == 1 &&
           op.Outputs().size() == 1 &&
           IsFloatLike(op.Inputs()[0].ElementType()) &&
           IsFloatLike(op.Outputs()[0].ElementType());
  }

  // rms_norm(x, w) -> mul(c1) -> ... -> `tensor` with constant scalars c_i
  // and casts between float types: the fusion the projection reading
  // `tensor` can take, if its weights are INT4 (the fused kernels) and x has
  // `columns` dims. The Gemma 4 12B wraps its norms in casts (float16 to
  // float32 and back; the E2B, E4B and 26B have none), which the fusion
  // reads through only when IsFloatCast says so; the kernel then reads the
  // norm's input.
  Expected<std::optional<NormFusion>> FindNormFusion(const Tensor& tensor,
                                                     int32_t columns) {
    if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_FUSE_NORM_GEMV",
                    /*default_value=*/true)) {
      return std::optional<NormFusion>();
    }
    float multiplier = 1.0f;
    Tensor value = tensor;
    for (int hops = 0; hops < 6; ++hops) {
      auto producer = value.GetDefiningOp();
      if (!producer) return std::optional<NormFusion>();
      if (IsFloatCast(*producer)) {
        value = producer->Inputs()[0];
        continue;
      }
      if (producer->Code() == kLiteRtOpCodeTflMul &&
          producer->Inputs().size() == 2 && HasNoFusedActivation(*producer)) {
        std::optional<float> scalar;
        int other = -1;
        for (int i = 0; i < 2; ++i) {
          auto constant = ReadFloatScalar(producer->Inputs()[i]);
          if (constant) {
            scalar = *constant;
            other = 1 - i;
          }
        }
        if (!scalar.has_value()) return std::optional<NormFusion>();
        multiplier *= *scalar;
        value = producer->Inputs()[other];
        continue;
      }
      if (producer->Code() == kLiteRtOpCodeTflReshape &&
          producer->Inputs().size() >= 1) {
        value = producer->Inputs()[0];
        continue;
      }
      if (producer->Code() != kLiteRtOpCodeShloComposite ||
          CompositeOpName(*producer) != "odml.rms_norm" ||
          producer->Inputs().size() != 2 || producer->Outputs().size() != 1) {
        return std::optional<NormFusion>();
      }
      const Tensor x = producer->Inputs()[0];
      const Tensor w = producer->Inputs()[1];
      auto x_type = x.RankedTensorType();
      if (!x_type || x_type->Layout().Rank() < 1 ||
          x_type->Layout().Dimensions().back() != columns ||
          !IsFloatLike(x.ElementType()) || x.HasWeights() ||
          w.ElementType() != litert::ElementType::Float32 || !w.HasWeights() ||
          w.Weights().Bytes().size() != static_cast<size_t>(columns) * 4) {
        return std::optional<NormFusion>();
      }
      CompositeOptions options;
      float eps = 1e-6f;
      if (options.InitFromOp(producer->Get()) == kLiteRtStatusOk &&
          options.attributes_map.has_value() &&
          !options.attributes_map.value()["epsilon"].IsNull()) {
        eps = options.attributes_map.value()["epsilon"].AsFloat();
      }
      // The norm's input, before the casts that feed it (the kernel reads the
      // mode's float type either way).
      Tensor source = x;
      for (int casts = 0; casts < 2; ++casts) {
        auto cast = source.GetDefiningOp();
        if (!cast || !IsFloatCast(*cast)) break;
        source = cast->Inputs()[0];
      }
      NormFusion fusion;
      LITERT_ASSIGN_OR_RETURN(fusion.input, GetTensor(source));
      LITERT_ASSIGN_OR_RETURN(fusion.input,
                              AddCastTensor(fusion.input, ModeFloatType()));
      // The norm output keeps its own TensorRT lowering for other readers;
      // TensorRT drops it when nothing reads it.
      std::vector<float> values(columns);
      std::memcpy(values.data(), w.Weights().Bytes().data(),
                  values.size() * sizeof(float));
      nvinfer1::Dims dims{};
      dims.nbDims = 1;
      dims.d[0] = columns;
      LITERT_ASSIGN_OR_RETURN(
          fusion.scale, AddFloatConstant(values, dims,
                                         "fused_norm_scale_" +
                                             std::to_string(w.TensorIndex()),
                                         nvinfer1::DataType::kFLOAT));
      fusion.eps = eps;
      fusion.multiplier = multiplier;
      return std::optional<NormFusion>(fusion);
    }
    return std::optional<NormFusion>();
  }

  // The scales of a GEMV member: per-channel floats, or the FP16 bytes of
  // blockwise scales (FP32 scales are converted into owned storage).
  Expected<void> AppendGemvScales(
      const Tensor& weights, const SubbyteGemvWeights& info,
      std::vector<float>& per_channel,
      std::vector<absl::Span<const uint8_t>>& block_scales) {
    if (info.scale_block == 0) {
      const auto q = weights.PerChannelQuantization();
      per_channel.insert(per_channel.end(), q.scales,
                         q.scales + q.num_channels);
      return {};
    }
    if (!info.scales.has_value()) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Blockwise GEMV weights have no scales tensor");
    }
    const auto bytes = info.scales->Weights().Bytes();
    if (info.scales->ElementType() == litert::ElementType::Float16) {
      block_scales.push_back(bytes);
      return {};
    }
    const size_t count = bytes.size() / sizeof(float);
    owned_weights_.emplace_back(count * sizeof(uint16_t));
    auto& storage = owned_weights_.back();
    auto* halves = reinterpret_cast<uint16_t*>(storage.data());
    for (size_t i = 0; i < count; ++i) {
      float value = 0.0f;
      std::memcpy(&value, bytes.data() + i * sizeof(float), sizeof(float));
      halves[i] = Fp32ToFp16Bits(value);
    }
    block_scales.push_back(absl::MakeConstSpan(storage.data(), storage.size()));
    return {};
  }

  // The activation rows [..., columns] of a fully connected layer.
  static int64_t GemmRows(nvinfer1::ITensor* activation) {
    const auto dims = activation->getDimensions();
    int64_t rows = 1;
    for (int i = 0; i + 1 < dims.nbDims; ++i) {
      rows *= dims.d[i];
    }
    return rows;
  }

  // Whether `rows` static activation rows run on the batched GEMV: one row
  // (decode) or the few rows of a verify step, in INT2 or INT4.
  static bool BatchedGemvTakes(int64_t rows) {
    return rows >= 1 && rows <= kLiteRtNvidiaSubbyteGemvMaxRows;
  }

  // Whether `rows` activation rows of `columns` dims run on the CUDA GEMM.
  static bool CudaGemmTakes(int64_t rows, int64_t columns, int bit_width,
                            int32_t scale_block = 0) {
    if (GetCudaGemmMode() == CudaGemmMode::kOff || bit_width != 4 ||
        rows <= 1 || rows > std::numeric_limits<int32_t>::max() ||
        columns > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    // Any even number of output channels works.
    const LiteRtNvidiaGemmShape shape = {
        static_cast<int32_t>(rows), static_cast<int32_t>(columns), 2,
        kLiteRtNvidiaGemmGateNone, scale_block};
    return LiteRtNvidiaSubbyteGemmSupports(&shape) &&
           LiteRtNvidiaSubbyteGemmAvailable();
  }

  // Whether a product the CUDA GEMM supports runs on it although it leaves
  // part of the device idle. Without a weight store TensorRT's lowering is as
  // fast for such products; with one, the plugin keeps the weights in the
  // shared segments that every other graph of the model reads, where the
  // TensorRT lowering would hold a private copy of them in each engine.
  bool CudaGemmTakesSmallProducts() const { return weight_store_ != nullptr; }

  // A single projection and a row-concatenated group use the same plugin.
  // Keep constant names and layer creation order identical in both paths so
  // refit identities and TensorRT's generated graph are unchanged.
  //
  // One or a few activation rows run as a batched GEMV, many rows as a GEMM
  // over the same row-major weights.
  //
  // `members` are the row-major packed weights of the projections, in the
  // order of their rows. With a weight store they stay in the model file and
  // the plugin reads them in the holder of a segment; otherwise, or when a
  // member is not a view of the model file, they are copied into a constant.
  // With blockwise weights (info.scale_block 32) `block_scales` are the FP16
  // scales of the members in their order and `scales` is unused; they join
  // the weights in the store when they are views of the model file, and are
  // an FP16 constant otherwise.
  Expected<nvinfer1::ITensor*> AddSubbyteGemvPlugin(
      nvinfer1::ITensor* activation, const SubbyteGemvWeights& info,
      absl::Span<const absl::Span<const uint8_t>> members,
      absl::Span<const float> scales, const std::string& suffix,
      const std::string& output_name, int32_t gate = 0,
      absl::Span<const absl::Span<const uint8_t>> block_scales = {},
      const NormFusion* norm = nullptr) {
    const int64_t activation_rows = GemmRows(activation);
    const bool blockwise = info.scale_block != 0;
    // The blockwise GEMV has a gate mode; the per-channel one needs the GEMM.
    const bool gemm =
        !BatchedGemvTakes(activation_rows) || (gate != 0 && !blockwise);
    if (norm != nullptr && info.bit_width != 4) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "A fused norm needs INT4 weights");
    }
    size_t packed_bytes = 0;
    for (const auto& member : members) {
      packed_bytes += member.size();
    }
    if (gemm) {
      const LiteRtNvidiaGemmShape shape = {
          static_cast<int32_t>(activation_rows), info.columns,
          gate != 0 ? info.rows / 2 : info.rows, gate, info.scale_block};
      // Blockwise weights have no TensorRT lowering to leave small products
      // to.
      if (!CudaGemmTakes(activation_rows, info.columns, info.bit_width,
                         info.scale_block) ||
          !LiteRtNvidiaSubbyteGemmSupports(&shape) ||
          (!CudaGemmTakesSmallProducts() && !blockwise &&
           !LiteRtNvidiaSubbyteGemmFillsDevice(&shape))) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "The CUDA GEMM does not take this product");
      }
      if (LiteRtNvidiaSubbyteGemmWeightBytes(&shape) != packed_bytes) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Unexpected CUDA GEMM weight byte count");
      }
    }
    size_t scale_bytes = 0;
    for (const auto& member : block_scales) {
      scale_bytes += member.size();
    }
    if (blockwise && scale_bytes != static_cast<size_t>(info.rows) *
                                        (info.columns / info.scale_block) *
                                        sizeof(uint16_t)) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Unexpected blockwise scale byte count");
    }
    std::optional<TensorRtWeightLocation> location;
    bool scales_in_store = false;
    if (weight_store_ != nullptr) {
      std::vector<absl::Span<const uint8_t>> all(members.begin(),
                                                 members.end());
      // Blockwise scales join the weights in the store unless
      // LITERT_NVIDIA_TENSORRT_BLOCK_SCALES_IN_STORE=0 keeps them in the plan.
      const bool scales_to_store =
          blockwise &&
          EnvEnabled("LITERT_NVIDIA_TENSORRT_BLOCK_SCALES_IN_STORE",
                     /*default_value=*/true);
      if (scales_to_store) {
        all.insert(all.end(), block_scales.begin(), block_scales.end());
      }
      auto added = weight_store_->Add(all);
      if (added.HasValue()) {
        location = *added;
        scales_in_store = scales_to_store;
      } else if (added.Error().Status() != kLiteRtStatusErrorNotFound) {
        return added.Error();
      } else if (blockwise) {
        // Converted scales are not views of the file: they stay in the plan
        // while the weights go to the store.
        auto weights_only = weight_store_->Add(members);
        if (weights_only.HasValue()) {
          location = *weights_only;
        } else if (weights_only.Error().Status() !=
                   kLiteRtStatusErrorNotFound) {
          return weights_only.Error();
        } else {
          LITERT_LOG(LITERT_INFO,
                     "NVIDIA TensorRT-RTX keeping the CUDA subbyte weights%s "
                     "in the plan: %s",
                     suffix.c_str(), weights_only.Error().Message().c_str());
        }
      } else {
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT-RTX keeping the CUDA subbyte weights%s in "
                   "the plan: %s",
                   suffix.c_str(), added.Error().Message().c_str());
      }
    }
    nvinfer1::ITensor* packed_tensor = nullptr;
    if (location.has_value()) {
      // The holder of the segment replaces this once the segments of the
      // partition are final (AddWeightHolders).
      LITERT_ASSIGN_OR_RETURN(packed_tensor, ProvisionalWeightHolder());
    } else {
      std::vector<uint8_t> packed;
      packed.reserve(packed_bytes);
      for (const auto& member : members) {
        packed.insert(packed.end(), member.begin(), member.end());
      }
      owned_weights_.push_back(std::move(packed));
      nvinfer1::Dims packed_dims{};
      packed_dims.nbDims = 1;
      packed_dims.d[0] = static_cast<int32_t>(owned_weights_.back().size());
      nvinfer1::Weights packed_weights{
          nvinfer1::DataType::kINT8, owned_weights_.back().data(),
          static_cast<int64_t>(owned_weights_.back().size())};
      auto* packed_constant =
          network_->addConstant(packed_dims, packed_weights);
      if (packed_constant == nullptr ||
          packed_constant->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add packed CUDA subbyte GEMV weights");
      }
      LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
          packed_weights, "cuda_subbyte_gemv_weights" + suffix));
      packed_tensor = packed_constant->getOutput(0);
    }
    nvinfer1::ITensor* scale_tensor = nullptr;
    int64_t scale_offset = -1;
    if (scales_in_store) {
      // The scales follow the weights in the segment; the holder serves
      // both inputs (AddWeightHolders).
      scale_tensor = packed_tensor;
      scale_offset = static_cast<int64_t>(location->offset + packed_bytes);
    } else if (blockwise) {
      std::vector<uint8_t> halves;
      halves.reserve(scale_bytes);
      for (const auto& member : block_scales) {
        halves.insert(halves.end(), member.begin(), member.end());
      }
      owned_weights_.push_back(std::move(halves));
      nvinfer1::Dims scale_dims{};
      scale_dims.nbDims = 1;
      scale_dims.d[0] = static_cast<int64_t>(scale_bytes / sizeof(uint16_t));
      nvinfer1::Weights scale_weights{nvinfer1::DataType::kHALF,
                                      owned_weights_.back().data(),
                                      scale_dims.d[0]};
      auto* constant = network_->addConstant(scale_dims, scale_weights);
      if (constant == nullptr || constant->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add CUDA subbyte GEMV block scales");
      }
      LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
          scale_weights, "cuda_subbyte_gemv_block_scales" + suffix));
      scale_tensor = constant->getOutput(0);
    } else {
      nvinfer1::Dims scale_dims{};
      scale_dims.nbDims = 1;
      scale_dims.d[0] = info.rows;
      LITERT_ASSIGN_OR_RETURN(
          scale_tensor, AddFloatConstant(scales, scale_dims,
                                         "cuda_subbyte_gemv_scales" + suffix,
                                         nvinfer1::DataType::kBF16));
    }
    // A GEMM over a prefill chunk's rows reads their token positions when
    // its int8 path can take the product: with them that path chooses the
    // outlier dims over the chunk's rows, not over the padding past them,
    // whose attention averaged whatever the cache held
    // (LITERT_NVIDIA_GEMM_INT8_REAL_ROWS=0 returns to every row).
    const LiteRtNvidiaGemmShape int8_shape = {
        static_cast<int32_t>(activation_rows), info.columns,
        gate != 0 ? info.rows / 2 : info.rows, gate, info.scale_block};
    const bool row_positions = gemm && row_positions_ != nullptr &&
                               row_positions_count_ == activation_rows &&
                               LiteRtNvidiaInt8GemmSupports(&int8_shape);
    TrtPtr<nvinfer1::IPluginV3> plugin(CreateSubbyteGemvPlugin(
        info.bit_width, info.rows, info.columns, gate, gemm,
        location.has_value() ? static_cast<int64_t>(location->offset) : 0,
        location.has_value() ? static_cast<int64_t>(weight_store_->granule())
                             : 0,
        info.scale_block, scale_offset, norm != nullptr ? norm->eps : -1.0f,
        norm != nullptr ? norm->multiplier : 1.0f, row_positions,
        /*int8_blockwise=*/gemm && info.scale_block != 0 &&
            Int8BlockwiseGemmRequested(graph_has_gated_delta_net_)));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create CUDA subbyte GEMV plugin");
    }
    std::vector<nvinfer1::ITensor*> inputs = {activation, packed_tensor,
                                              scale_tensor};
    if (norm != nullptr) {
      inputs.push_back(norm->scale);
    }
    if (row_positions) {
      inputs.push_back(row_positions_);
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(*network_, inputs.data(),
                                                     inputs.size(), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add CUDA subbyte GEMV plugin layer");
    }
    if (location.has_value()) {
      holder_readers_.push_back({layer, location->segment, scales_in_store});
    }
    layer->setName(KeepName(UniqueName(output_name)));
    layer->getOutput(0)->setName(KeepName(UniqueName(output_name)));
    owned_plugins_.push_back(std::move(plugin));
    uses_cuda_subbyte_gemv_ = true;
    return layer->getOutput(0);
  }

  // A small INT64 constant that plugins take as their weights input until
  // the holder of their segment exists.
  Expected<nvinfer1::ITensor*> ProvisionalWeightHolder() {
    if (provisional_weight_holder_ != nullptr) {
      return provisional_weight_holder_;
    }
    constexpr int64_t kElements = 8;
    owned_weights_.emplace_back(kElements * sizeof(int64_t), 0);
    nvinfer1::Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = kElements;
    auto* layer = network_->addConstant(
        dims, nvinfer1::Weights{nvinfer1::DataType::kINT64,
                                owned_weights_.back().data(), kElements});
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the provisional weight holder");
    }
    provisional_weight_holder_ = layer->getOutput(0);
    return provisional_weight_holder_;
  }

  // Adds, for every segment of the weight store that this partition reads, a
  // placeholder constant ("holder") one granule larger than the segment, and
  // makes it the weights input of the plugins that read the segment. A
  // placeholder has no values at build time (TensorRT-RTX 1.7): the plan is
  // stripped of it, and the runtime maps the segment into the engine's weight
  // memory at the first granule boundary inside the holder.
  Expected<void> AddWeightHolders() {
    if (holder_readers_.empty()) {
      return {};
    }
    weight_store_->EndPartition();
    const uint64_t granule = weight_store_->granule();
    std::map<uint32_t, nvinfer1::ITensor*> holders;
    for (const auto& reader : holder_readers_) {
      auto [holder, inserted] = holders.try_emplace(reader.segment, nullptr);
      if (inserted) {
        const uint64_t bytes =
            weight_store_->segments()[reader.segment].size + granule;
        nvinfer1::Dims dims{};
        dims.nbDims = 1;
        dims.d[0] = static_cast<int64_t>(bytes / sizeof(int64_t));
        auto* layer = network_->addConstant(
            dims,
            nvinfer1::Weights{nvinfer1::DataType::kINT64, nullptr, dims.d[0]});
        if (layer == nullptr || layer->getOutput(0) == nullptr) {
          return Error(kLiteRtStatusErrorCompilation,
                       "Failed to add a weight holder placeholder");
        }
        std::string name =
            "litert_weight_holder_" + std::to_string(reader.segment);
        const char* kept_name = KeepName(name);
        // A placeholder is named by the descriptor its layer returns.
        if (!network_->setWeightsName(layer->getWeights(), kept_name)) {
          return Error(kLiteRtStatusErrorCompilation,
                       "Failed to name a weight holder placeholder");
        }
        layer->setName(kept_name);
        holder->second = layer->getOutput(0);
        weight_holders_.push_back({std::move(name), reader.segment, bytes});
      }
      reader.layer->setInput(reader.input, *holder->second);
      if (reader.scales_too) {
        reader.layer->setInput(2, *holder->second);
      }
    }
    return {};
  }

  // A gated feed-forward pair over many activation rows,
  //   mul(gelu(fully_connected(x, gate)), fully_connected(x, up))
  // with reshapes of single use in between, which the CUDA GEMM computes in
  // one launch.
  struct GatedFeedForward {
    std::optional<Op> up;
    std::optional<Op> multiply;
    std::vector<LiteRtOp> members;  // everything but the gate projection
    int32_t gate = 0;
  };

  std::optional<GatedFeedForward> FindGatedFeedForward(const Op& gate_op) {
    const auto gate_info = InspectSubbyteGemvWeights(gate_op.Inputs()[1]);
    if (!gate_info || gate_info->bit_width != 4 ||
        gate_op.Inputs().size() != 2 || !HasNoFusedActivation(gate_op)) {
      return std::nullopt;
    }
    GatedFeedForward found;
    // gate projection -> [reshape] -> gelu -> multiply.
    Tensor value = gate_op.Outputs()[0];
    auto user = SingleUser(value, 0);
    if (user && user->Code() == kLiteRtOpCodeTflReshape) {
      found.members.push_back(user->Get());
      value = user->Outputs()[0];
      user = SingleUser(value, 0);
    }
    bool approximate = false;
    if (!user || user->Code() != kLiteRtOpCodeTflGelu ||
        LiteRtGetGeluApproximateOption(user->Get(), &approximate) !=
            kLiteRtStatusOk) {
      return std::nullopt;
    }
    found.gate = approximate ? kLiteRtNvidiaGemmGateGeluTanh
                             : kLiteRtNvidiaGemmGateGeluErf;
    found.members.push_back(user->Get());
    value = user->Outputs()[0];
    if (value.Uses().size() != 1) {
      return std::nullopt;
    }
    const Op multiply = value.Uses()[0].user;
    if (multiply.Code() != kLiteRtOpCodeTflMul ||
        multiply.Inputs().size() != 2 || multiply.Outputs().size() != 1 ||
        !HasNoFusedActivation(multiply)) {
      return std::nullopt;
    }
    // up projection -> [reshape] -> the other operand.
    Tensor other = multiply.Inputs()[1 - value.Uses()[0].user_arg_ind];
    auto producer = SingleUseProducer(other);
    if (producer && producer->Code() == kLiteRtOpCodeTflReshape) {
      found.members.push_back(producer->Get());
      other = producer->Inputs()[0];
      producer = SingleUseProducer(other);
    }
    if (!producer || producer->Code() != kLiteRtOpCodeTflFullyConnected ||
        producer->Inputs().size() != 2 ||
        producer->Inputs()[0].Get() != gate_op.Inputs()[0].Get() ||
        !HasNoFusedActivation(*producer)) {
      return std::nullopt;
    }
    const auto up_info = InspectSubbyteGemvWeights(producer->Inputs()[1]);
    if (!up_info || up_info->bit_width != 4 ||
        up_info->rows != gate_info->rows ||
        up_info->columns != gate_info->columns ||
        up_info->scale_block != gate_info->scale_block) {
      return std::nullopt;
    }
    // Every member comes after the gate projection, which emits the launch,
    // and no intermediate result leaves the pair.
    const auto gate_order = op_order_.find(gate_op.Get());
    found.members.push_back(producer->Get());
    found.members.push_back(multiply.Get());
    for (LiteRtOp member : found.members) {
      const auto order = op_order_.find(member);
      if (gate_order == op_order_.end() || order == op_order_.end() ||
          order->second <= gate_order->second) {
        return std::nullopt;
      }
    }
    if (graph_outputs_.count(gate_op.Outputs()[0].Get()) != 0 ||
        graph_outputs_.count(multiply.Inputs()[0].Get()) != 0 ||
        graph_outputs_.count(multiply.Inputs()[1].Get()) != 0 ||
        graph_outputs_.count(producer->Outputs()[0].Get()) != 0) {
      return std::nullopt;
    }
    found.up = *producer;
    found.multiply = multiply;
    return found;
  }

  // Qwen 3.5's feed-forward, and Gemma 4's when it is exported fused: one INT4
  // fully connected op over the gate and up projections stacked [2 * N, K]
  // whose single user is odml.swiglu (gate_size N). The gated CUDA GEMM/GEMV
  // computes act(gate) * up in the composite's gate mode in one launch,
  // without the [.., 2 * N] intermediate.
  // LITERT_NVIDIA_TENSORRT_FUSE_SWIGLU=0 lowers them separately.
  std::optional<Op> FindFusedSwiglu(const Op& op) {
    if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_FUSE_SWIGLU",
                    /*default_value=*/true) ||
        op.Inputs().size() != 2 || op.Outputs().size() != 1 ||
        !HasNoFusedActivation(op)) {
      return std::nullopt;
    }
    const auto info = InspectSubbyteGemvWeights(op.Inputs()[1]);
    if (!info || info->bit_width != 4 || info->rows % 2 != 0) {
      return std::nullopt;
    }
    auto user = SingleUser(op.Outputs()[0], 0);
    if (!user || user->Code() != kLiteRtOpCodeShloComposite ||
        CompositeOpName(*user) != "odml.swiglu" ||
        graph_outputs_.count(op.Outputs()[0].Get()) != 0) {
      return std::nullopt;
    }
    const auto gate = ParseSwiglu(*user);
    if (!gate.has_value() || 2 * gate->gate_size != info->rows) {
      return std::nullopt;
    }
    return user;
  }

  Expected<void> AddFusedSwiglu(const Op& op, const Op& swiglu,
                                nvinfer1::ITensor* activation,
                                nvinfer1::DataType compute_type,
                                const NormFusion* norm = nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto info,
                            InspectSubbyteGemvWeights(op.Inputs()[1]));
    const auto gate = ParseSwiglu(swiglu);
    if (!gate.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported swiglu composite");
    }
    LITERT_RETURN_IF_ERROR(
        ValidateSubbyteGemvActivation(activation, compute_type, info.columns,
                                      info.bit_width, info.scale_block));
    std::vector<absl::Span<const uint8_t>> members = {
        op.Inputs()[1].Weights().Bytes()};
    std::vector<float> scales;
    std::vector<absl::Span<const uint8_t>> block_scales;
    LITERT_RETURN_IF_ERROR(
        AppendGemvScales(op.Inputs()[1], info, scales, block_scales));
    const std::string suffix =
        "_" + std::to_string(op.Inputs()[1].TensorIndex());
    LITERT_ASSIGN_OR_RETURN(
        auto* gated,
        AddSubbyteGemvPlugin(
            activation, info, members, absl::MakeConstSpan(scales), suffix,
            "cuda_subbyte_swiglu_gemm" + suffix, gate->activation,
            absl::MakeConstSpan(block_scales), norm));
    LITERT_ASSIGN_OR_RETURN(auto out_type,
                            swiglu.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto out_dims, ConvertDims(out_type));
    LITERT_ASSIGN_OR_RETURN(gated, ReshapeTensor(gated, out_dims));
    unlowered_ops_.insert(swiglu.Get());
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX fused CUDA SwiGLU GEMM: N=%d K=%d "
               "tensor=%s gate=%s",
               info.rows / 2, info.columns, suffix.c_str() + 1,
               gate->activation == kLiteRtNvidiaGemmGateGeluTanh ? "gelu_tanh"
                                                                 : "silu");
    return SetOutputTensor(swiglu.Outputs()[0], gated);
  }

  // Emits the launch of a gated feed-forward pair at its gate projection and
  // binds the result of the multiplication; the other members are not
  // lowered.
  Expected<void> AddGatedFeedForward(const Op& gate_op,
                                     const GatedFeedForward& pair,
                                     nvinfer1::ITensor* activation,
                                     nvinfer1::DataType compute_type,
                                     const NormFusion* norm = nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto info,
                            InspectSubbyteGemvWeights(gate_op.Inputs()[1]));
    LITERT_RETURN_IF_ERROR(
        ValidateSubbyteGemvActivation(activation, compute_type, info.columns,
                                      info.bit_width, info.scale_block));
    std::vector<absl::Span<const uint8_t>> members;
    std::vector<float> scales;
    std::vector<absl::Span<const uint8_t>> block_scales;
    std::string suffix;
    for (const Tensor& weights : {gate_op.Inputs()[1], pair.up->Inputs()[1]}) {
      LITERT_ASSIGN_OR_RETURN(auto member_info,
                              InspectSubbyteGemvWeights(weights));
      members.push_back(weights.Weights().Bytes());
      LITERT_RETURN_IF_ERROR(
          AppendGemvScales(weights, member_info, scales, block_scales));
      suffix += "_" + std::to_string(weights.TensorIndex());
    }
    SubbyteGemvWeights gated_info = info;
    gated_info.rows = 2 * info.rows;
    LITERT_ASSIGN_OR_RETURN(
        auto* gated,
        AddSubbyteGemvPlugin(activation, gated_info, members,
                             absl::MakeConstSpan(scales), suffix,
                             "cuda_subbyte_gated_gemm" + suffix, pair.gate,
                             absl::MakeConstSpan(block_scales), norm));
    LITERT_ASSIGN_OR_RETURN(auto out_type,
                            pair.multiply->Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto out_dims, ConvertDims(out_type));
    LITERT_ASSIGN_OR_RETURN(gated, ReshapeTensor(gated, out_dims));
    for (LiteRtOp member : pair.members) {
      unlowered_ops_.insert(member);
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX fused CUDA gated feed-forward GEMM: N=%d "
               "K=%d tensors=%s",
               info.rows, info.columns, suffix.c_str() + 1);
    return SetOutputTensor(pair.multiply->Outputs()[0], gated);
  }

  // Fully connected ops that read the same activation (Gemma's q/k/v and
  // gate/up projections) and qualify for the CUDA subbyte GEMV run as one
  // plugin over their row-concatenated weights: the activation is read once,
  // the small projections stop being latency-bound launches, and every step
  // issues fewer kernels. Later ops of the group take their slice of the
  // fused output from fused_gemv_outputs_ when the lowering reaches them.
  Expected<nvinfer1::ITensor*> AddCudaSubbyteGemvGroup(
      const Op& op, nvinfer1::ITensor* activation,
      nvinfer1::DataType compute_type, const NormFusion* norm = nullptr) {
    if (!FuseGemvGroupsEnabled()) {
      return AddCudaSubbyteGemv(op.Inputs()[1], activation, compute_type, norm);
    }
    LITERT_ASSIGN_OR_RETURN(auto first,
                            InspectSubbyteGemvWeights(op.Inputs()[1]));
    // Over many activation rows, the projections of a later gated
    // feed-forward pair run as a launch of their own.
    std::unordered_set<LiteRtOp> gated;
    if (!BatchedGemvTakes(GemmRows(activation))) {
      for (const auto& use : op.Inputs()[0].Uses()) {
        const Op& user = use.user;
        if (use.user_arg_ind != 0 || user.Get() == op.Get() ||
            user.Code() != kLiteRtOpCodeTflFullyConnected ||
            user.Inputs().size() < 2 || user.Outputs().size() != 1 ||
            op_order_.find(user.Get()) == op_order_.end() ||
            op_order_[user.Get()] < op_order_[op.Get()]) {
          continue;
        }
        if (const auto pair = FindGatedFeedForward(user);
            pair.has_value() && pair->up->Get() != op.Get()) {
          gated.insert(user.Get());
          gated.insert(pair->up->Get());
        }
      }
    }
    std::vector<Op> group;
    for (const auto& use : op.Inputs()[0].Uses()) {
      const Op& user = use.user;
      if (use.user_arg_ind != 0 ||
          user.Code() != kLiteRtOpCodeTflFullyConnected ||
          user.Inputs().size() < 2 || user.Outputs().size() != 1 ||
          op_order_.find(user.Get()) == op_order_.end() ||
          op_order_[user.Get()] < op_order_[op.Get()] ||
          fused_gemv_outputs_.find(user.Get()) != fused_gemv_outputs_.end() ||
          gated.count(user.Get()) != 0) {
        continue;
      }
      auto info = InspectSubbyteGemvWeights(user.Inputs()[1]);
      if (!info || info->bit_width != first.bit_width ||
          info->columns != first.columns ||
          info->scale_block != first.scale_block) {
        continue;
      }
      group.push_back(user);
    }
    if (group.size() <= 1) {
      return AddCudaSubbyteGemv(op.Inputs()[1], activation, compute_type, norm);
    }
    std::sort(group.begin(), group.end(), [&](const Op& a, const Op& b) {
      return op_order_[a.Get()] < op_order_[b.Get()];
    });
    if (group.front().Get() != op.Get()) {
      return Error(kLiteRtStatusErrorCompilation,
                   "CUDA subbyte GEMV group does not start at the current op");
    }
    LITERT_RETURN_IF_ERROR(
        ValidateSubbyteGemvActivation(activation, compute_type, first.columns,
                                      first.bit_width, first.scale_block));
    const auto activation_dims = activation->getDimensions();

    std::vector<absl::Span<const uint8_t>> members;
    std::vector<float> scales;
    std::vector<absl::Span<const uint8_t>> block_scales;
    std::vector<int32_t> row_counts;
    std::string suffix;
    for (const auto& member : group) {
      const Tensor weights = member.Inputs()[1];
      LITERT_ASSIGN_OR_RETURN(auto info, InspectSubbyteGemvWeights(weights));
      members.push_back(weights.Weights().Bytes());
      LITERT_RETURN_IF_ERROR(
          AppendGemvScales(weights, info, scales, block_scales));
      row_counts.push_back(info.rows);
      suffix += "_" + std::to_string(weights.TensorIndex());
    }
    const int64_t total_rows =
        std::accumulate(row_counts.begin(), row_counts.end(), int64_t{0});
    if (total_rows > std::numeric_limits<int32_t>::max()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "CUDA subbyte GEMV group is too large");
    }

    SubbyteGemvWeights group_info = first;
    group_info.rows = static_cast<int32_t>(total_rows);
    LITERT_ASSIGN_OR_RETURN(
        auto* fused,
        AddSubbyteGemvPlugin(activation, group_info, members,
                             absl::MakeConstSpan(scales), suffix,
                             "cuda_subbyte_gemv_group" + suffix, /*gate=*/0,
                             absl::MakeConstSpan(block_scales), norm));

    const int axis = activation_dims.nbDims - 1;
    nvinfer1::ITensor* first_output = nullptr;
    int32_t offset = 0;
    for (size_t i = 0; i < group.size(); ++i) {
      LITERT_ASSIGN_OR_RETURN(
          auto* slice,
          SliceTensorStatic(
              fused, axis, offset, row_counts[i],
              "cuda_subbyte_gemv_slice_" +
                  std::to_string(group[i].Inputs()[1].TensorIndex())));
      LITERT_ASSIGN_OR_RETURN(auto out_type,
                              group[i].Outputs()[0].RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto out_dims, ConvertDims(out_type));
      const auto slice_dims = slice->getDimensions();
      bool same_dims = slice_dims.nbDims == out_dims.nbDims;
      for (int d = 0; same_dims && d < out_dims.nbDims; ++d) {
        same_dims = slice_dims.d[d] == out_dims.d[d];
      }
      if (!same_dims) {
        LITERT_ASSIGN_OR_RETURN(slice, ReshapeTensor(slice, out_dims));
      }
      if (i == 0) {
        first_output = slice;
      } else {
        fused_gemv_outputs_[group[i].Get()] = slice;
      }
      gemv_group_slices_[group[i].Get()] = {fused, offset, row_counts[i],
                                            static_cast<int32_t>(total_rows)};
      offset += row_counts[i];
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX fused CUDA subbyte GEMV group: ops=%zu "
               "bits=%d N=%lld K=%d tensors=%s",
               group.size(), first.bit_width,
               static_cast<long long>(total_rows), first.columns,
               suffix.c_str() + 1);
    return first_output;
  }

  Expected<nvinfer1::ITensor*> AddCudaSubbyteGemv(
      const Tensor& tensor, nvinfer1::ITensor* activation,
      nvinfer1::DataType compute_type, const NormFusion* norm = nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto info, InspectSubbyteGemvWeights(tensor));
    LITERT_RETURN_IF_ERROR(
        ValidateSubbyteGemvActivation(activation, compute_type, info.columns,
                                      info.bit_width, info.scale_block));
    std::vector<float> scales;
    std::vector<absl::Span<const uint8_t>> block_scales;
    LITERT_RETURN_IF_ERROR(
        AppendGemvScales(tensor, info, scales, block_scales));
    const absl::Span<const uint8_t> members[] = {tensor.Weights().Bytes()};
    const std::string suffix = "_" + std::to_string(tensor.TensorIndex());
    LITERT_ASSIGN_OR_RETURN(
        auto* output,
        AddSubbyteGemvPlugin(activation, info, members,
                             absl::MakeConstSpan(scales), suffix,
                             "cuda_subbyte_gemv" + suffix, /*gate=*/0,
                             absl::MakeConstSpan(block_scales), norm));
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX CUDA subbyte GEMV: tensor=%u bits=%d "
               "N=%lld K=%lld",
               tensor.TensorIndex(), info.bit_width,
               static_cast<long long>(info.rows),
               static_cast<long long>(info.columns));
    return output;
  }

  Expected<nvinfer1::ITensor*> AddInt32Constant(
      absl::Span<const int32_t> values, nvinfer1::Dims dims,
      const std::string& name) {
    owned_weights_.emplace_back(values.size() * sizeof(int32_t));
    auto& storage = owned_weights_.back();
    std::memcpy(storage.data(), values.data(), storage.size());
    nvinfer1::Weights weights{nvinfer1::DataType::kINT32, storage.data(),
                              static_cast<int64_t>(values.size())};
    auto* layer = network_->addConstant(dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT int32 constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(weights, name));
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddQuantizationScaleTensor(
      const Tensor& quantized_tensor, int32_t& axis,
      nvinfer1::DataType scale_type, float scale_multiplier = 1.0f) {
    axis = -1;
    if (quantized_tensor.QTypeId() == kLiteRtQuantizationPerTensor) {
      auto q = quantized_tensor.PerTensorQuantization();
      if (q.scale <= 0.0f || q.zero_point != 0) {
        return Error(
            kLiteRtStatusErrorUnsupported,
            "TensorRT Q/DQ requires symmetric per-tensor quantization");
      }
      nvinfer1::Dims scalar_dims{};
      scalar_dims.nbDims = 0;
      const float scale = q.scale * scale_multiplier;
      return AddFloatConstant(
          absl::MakeConstSpan(&scale, 1), scalar_dims,
          "quant_scale_tensor_" +
              std::to_string(quantized_tensor.TensorIndex()),
          scale_type);
    }

    if (quantized_tensor.QTypeId() == kLiteRtQuantizationPerChannel) {
      auto q = quantized_tensor.PerChannelQuantization();
      if (q.num_channels == 0 || q.scales == nullptr ||
          q.zero_points == nullptr) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Invalid per-channel quantization metadata");
      }
      for (uint64_t i = 0; i < q.num_channels; ++i) {
        if (q.scales[i] <= 0.0f || q.zero_points[i] != 0) {
          return Error(
              kLiteRtStatusErrorUnsupported,
              "TensorRT Q/DQ requires symmetric per-channel quantization");
        }
      }
      if (q.num_channels >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Per-channel quantization scale count is too large");
      }
      axis = q.quantized_dimension;
      nvinfer1::Dims scale_dims{};
      scale_dims.nbDims = 1;
      scale_dims.d[0] = static_cast<int32_t>(q.num_channels);
      std::vector<float> scaled;
      absl::Span<const float> scales(q.scales, q.num_channels);
      if (scale_multiplier != 1.0f) {
        scaled.assign(scales.begin(), scales.end());
        for (float& scale : scaled) {
          scale *= scale_multiplier;
        }
        scales = absl::MakeConstSpan(scaled);
      }
      return AddFloatConstant(
          scales, scale_dims,
          "quant_scale_tensor_" +
              std::to_string(quantized_tensor.TensorIndex()),
          scale_type);
    }

    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported TensorRT quantization metadata");
  }

  // TensorRT requires the Q/DQ scale tensor type to match the float side of
  // the conversion (the dequantize output or the quantize input).
  static nvinfer1::DataType QuantizationScaleType(nvinfer1::DataType type) {
    if (type == nvinfer1::DataType::kHALF ||
        type == nvinfer1::DataType::kBF16) {
      return type;
    }
    return nvinfer1::DataType::kFLOAT;
  }

  Expected<nvinfer1::ITensor*> AddDequantizeTensor(
      const Tensor& quantized_tensor, nvinfer1::ITensor* input,
      nvinfer1::DataType output_type, float scale_multiplier = 1.0f) {
    int32_t axis = -1;
    LITERT_ASSIGN_OR_RETURN(
        auto* scale, AddQuantizationScaleTensor(
                         quantized_tensor, axis,
                         QuantizationScaleType(output_type), scale_multiplier));
    auto* layer = network_->addDequantize(*input, *scale, output_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT dequantize layer");
    }
    if (axis >= 0) {
      layer->setAxis(axis);
    }
    return layer->getOutput(0);
  }

  // Lowering per-channel INT4 FC weights as `constant -> dequantize -> matmul`
  // makes Myelin materialize the dequantized 16-bit weights (four times the
  // INT4 bytes, written and read back) on every invocation before a dense
  // GEMM. The same scales expressed as block scales along K, where every
  // block of a row carries that row's per-channel scale, are numerically
  // identical but let TensorRT-RTX emit one fused weight-only-quantized GEMM
  // kernel that reads the INT4 weights directly. Returns nullopt when the
  // weights cannot use this encoding (M=1 GEMVs never reach here).
  //
  // The fused kernel is not always the faster choice: on the RTX 5080 at
  // M=1024 it loses to the materialize-then-GEMM path: 858 vs 617 us for
  // K=8192 and 1573 vs 1185 us for K=15360, while winning or tying for
  // K<=4096, whereas at M<=512 it wins for every shape probed (M=128,
  // K=15360: 185 vs 340 us). `rows` is the static M of the activation.
  Expected<std::optional<nvinfer1::ITensor*>> AddBlockScaledDequantizeTensor(
      const Tensor& tensor, nvinfer1::ITensor* weights,
      nvinfer1::DataType output_type, int64_t rows) {
    const auto none = std::optional<nvinfer1::ITensor*>();
    if (!FcBlockScalesEnabled() ||
        weights->getType() != nvinfer1::DataType::kINT4 ||
        tensor.QTypeId() != kLiteRtQuantizationPerChannel) {
      return none;
    }
    const auto dims = weights->getDimensions();
    if (dims.nbDims != 2 || dims.d[0] <= 0 || dims.d[1] <= 0) {
      return none;
    }
    if (rows > 512 && dims.d[1] > 4096) {
      return none;
    }
    const auto q = tensor.PerChannelQuantization();
    if (q.quantized_dimension != 0 ||
        q.num_channels != static_cast<uint64_t>(dims.d[0]) ||
        q.scales == nullptr || q.zero_points == nullptr) {
      return none;
    }
    const int32_t k = dims.d[1];
    int32_t block = 0;
    for (int32_t candidate : kFcBlockScaleSizes) {
      if (candidate < k && k % candidate == 0) {
        block = candidate;
        break;
      }
    }
    if (block == 0) {
      return none;
    }
    const int32_t blocks_per_row = k / block;
    std::vector<float> scales(static_cast<size_t>(q.num_channels) *
                              blocks_per_row);
    for (uint64_t n = 0; n < q.num_channels; ++n) {
      if (q.scales[n] <= 0.0f || q.zero_points[n] != 0) {
        return none;
      }
      std::fill_n(scales.data() + n * blocks_per_row, blocks_per_row,
                  q.scales[n]);
    }
    nvinfer1::Dims scale_dims{};
    scale_dims.nbDims = 2;
    scale_dims.d[0] = dims.d[0];
    scale_dims.d[1] = blocks_per_row;
    LITERT_ASSIGN_OR_RETURN(
        auto* scale, AddFloatConstant(absl::MakeConstSpan(scales), scale_dims,
                                      "fc_block_scale_tensor_" +
                                          std::to_string(tensor.TensorIndex()),
                                      QuantizationScaleType(output_type)));
    auto* layer = network_->addDequantize(*weights, *scale, output_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT block-scaled dequantize layer");
    }
    layer->setAxis(1);
    return std::optional<nvinfer1::ITensor*>(layer->getOutput(0));
  }

  Expected<nvinfer1::ITensor*> AddQuantizeTensor(
      nvinfer1::ITensor* input, const Tensor& quantized_tensor,
      nvinfer1::DataType output_type) {
    // TensorRT's Q/DQ graph optimizer may treat a same-scale Q->DQ pair as an
    // identity, which drops the fake-quant saturation TFLite applies. Clamp
    // to the representable range explicitly so saturation semantics survive
    // any such rewrite.
    if (output_type == nvinfer1::DataType::kINT8 &&
        quantized_tensor.QTypeId() == kLiteRtQuantizationPerTensor) {
      const float scale = quantized_tensor.PerTensorQuantization().scale;
      auto* clip_layer =
          network_->addActivation(*input, nvinfer1::ActivationType::kCLIP);
      if (clip_layer == nullptr || clip_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT quantize clamp layer");
      }
      clip_layer->setAlpha(-128.0f * scale);
      clip_layer->setBeta(127.0f * scale);
      input = clip_layer->getOutput(0);
    }
    int32_t axis = -1;
    LITERT_ASSIGN_OR_RETURN(
        auto* scale,
        AddQuantizationScaleTensor(quantized_tensor, axis,
                                   QuantizationScaleType(input->getType())));
    auto* layer = network_->addQuantize(*input, *scale, output_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT quantize layer");
    }
    if (axis >= 0) {
      layer->setAxis(axis);
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddCastTensor(nvinfer1::ITensor* input,
                                             nvinfer1::DataType output_type) {
    if (input->getType() == output_type) {
      // Skip identity casts: an interposed cast layer can also break
      // TensorRT's dequantize->matmul weight-only-quantization fusion.
      return input;
    }
    auto* layer = network_->addCast(*input, output_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT cast layer");
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddZeroFloatTensor(nvinfer1::Dims dims,
                                                  nvinfer1::DataType data_type,
                                                  const std::string& name) {
    if (data_type != nvinfer1::DataType::kFLOAT &&
        data_type != nvinfer1::DataType::kHALF &&
        data_type != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Zero padding supports only TensorRT float tensors");
    }
    size_t elements = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
      if (dims.d[i] < 0) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Zero padding requires static dimensions");
      }
      elements *= static_cast<size_t>(dims.d[i]);
    }
    std::vector<float> zeros(elements, 0.0f);
    return AddFloatConstant(absl::MakeConstSpan(zeros), dims, name, data_type);
  }

  Expected<nvinfer1::ITensor*> SliceTensorStatic(nvinfer1::ITensor* input,
                                                 int axis, int start_index,
                                                 int size,
                                                 const std::string& name) {
    const auto input_dims = input->getDimensions();
    if (axis < 0 || axis >= input_dims.nbDims || size < 0 || start_index < 0 ||
        start_index + size > input_dims.d[axis]) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Invalid TensorRT static slice range");
    }
    nvinfer1::Dims start{};
    nvinfer1::Dims output_dims = input_dims;
    nvinfer1::Dims stride{};
    start.nbDims = input_dims.nbDims;
    stride.nbDims = input_dims.nbDims;
    for (int i = 0; i < input_dims.nbDims; ++i) {
      start.d[i] = 0;
      stride.d[i] = 1;
    }
    start.d[axis] = start_index;
    output_dims.d[axis] = size;
    auto* layer = network_->addSlice(*input, start, output_dims, stride);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT static slice layer");
    }
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> AddRepeatedScalarConstant(
      const Tensor& value_tensor, const Tensor& output_tensor) {
    LITERT_ASSIGN_OR_RETURN(auto value_type, value_tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_type, output_tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(size_t value_elements, NumElements(value_type));
    if (value_elements != 1 ||
        value_type.ElementType() != output_type.ElementType()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Fill expects a scalar value matching the output type");
    }
    LITERT_ASSIGN_OR_RETURN(size_t element_size,
                            ElementByteSize(output_type.ElementType()));
    auto value_bytes = value_tensor.Weights().Bytes();
    if (value_bytes.size() != element_size) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected fill scalar byte size");
    }
    LITERT_ASSIGN_OR_RETURN(size_t output_elements, NumElements(output_type));
    owned_weights_.emplace_back(output_elements * element_size);
    auto& storage = owned_weights_.back();
    for (size_t i = 0; i < output_elements; ++i) {
      std::memcpy(storage.data() + i * element_size, value_bytes.data(),
                  element_size);
    }
    LITERT_ASSIGN_OR_RETURN(auto data_type,
                            ConvertDataType(output_type.ElementType()));
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    nvinfer1::Weights weights{data_type, storage.data(),
                              static_cast<int64_t>(output_elements)};
    auto* layer = network_->addConstant(output_dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT fill constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        weights,
        "fill_constant_tensor_" + std::to_string(output_tensor.TensorIndex())));
    layer->getOutput(0)->setName(
        KeepName(UniqueName("fill_constant_tensor_" +
                            std::to_string(output_tensor.TensorIndex()))));
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> GetTensor(const Tensor& tensor) {
    if (auto it = tensor_map_.find(tensor.Get()); it != tensor_map_.end()) {
      return it->second;
    }
    if (auto value = quant_values_.find(tensor.Get());
        value != quant_values_.end()) {
      // An op outside the quant regions reads a quant layer's INT8 result.
      LITERT_ASSIGN_OR_RETURN(auto* view, QuantRealView(tensor, value->second));
      tensor_map_[tensor.Get()] = view;
      return view;
    }
    if (!tensor.HasWeights()) {
      return Error(kLiteRtStatusErrorCompilation,
                   "TensorRT tensor is not available");
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
    LITERT_ASSIGN_OR_RETURN(auto weights, MakeWeights(tensor));
    auto* layer = network_->addConstant(dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT constant");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        weights, "constant_tensor_" + std::to_string(tensor.TensorIndex())));
    auto* out = layer->getOutput(0);
    out->setName(KeepName(
        UniqueName("constant_tensor_" + std::to_string(tensor.TensorIndex()))));
    if (Fp16ActivationsEnabled()) {
      // Keep per-channel (weight) constants raw for the matmul WoQ pattern;
      // fold everything else into FP16 so downstream types match.
      LITERT_ASSIGN_OR_RETURN(bool int8_per_tensor,
                              HasSymmetricPerTensorInt8Quantization(tensor));
      if (int8_per_tensor) {
        LITERT_ASSIGN_OR_RETURN(
            out, AddDequantizeTensor(tensor, out, ModeFloatType()));
      } else if (out->getType() == nvinfer1::DataType::kFLOAT) {
        LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, ModeFloatType()));
      }
    }
    tensor_map_[tensor.Get()] = out;
    return out;
  }

  // The FP32 view of `tensor` on the position path (see exact_fp32_map_): a
  // value recorded there, or an unquantized FP32 constant such as the inverse
  // RoPE frequencies, built in FP32 on first use. Null otherwise.
  Expected<nvinfer1::ITensor*> GetExactFp32Tensor(const Tensor& tensor) {
    if (auto it = exact_fp32_map_.find(tensor.Get());
        it != exact_fp32_map_.end()) {
      return it->second;
    }
    if (auto it = exact_fp32_constants_.find(tensor.Get());
        it != exact_fp32_constants_.end()) {
      return it->second;
    }
    if (!tensor.HasWeights() ||
        tensor.ElementType() != litert::ElementType::Float32 ||
        tensor.QTypeId() != kLiteRtQuantizationNone) {
      return nullptr;
    }
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
    LITERT_ASSIGN_OR_RETURN(auto weights, MakeWeights(tensor));
    auto* layer = network_->addConstant(dims, weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT FP32 constant");
    }
    auto* out = layer->getOutput(0);
    out->setName(KeepName(UniqueName("exact_fp32_constant_tensor_" +
                                     std::to_string(tensor.TensorIndex()))));
    exact_fp32_constants_[tensor.Get()] = out;
    return out;
  }

  bool HasExactFp32Value(const Tensor& tensor) const {
    return exact_fp32_map_.find(tensor.Get()) != exact_fp32_map_.end();
  }

  Expected<nvinfer1::ITensor*> AddFusedActivation(nvinfer1::ITensor* input,
                                                  uint32_t activation) {
    using Opt = litert::ActivationFunctionType;
    if (activation == Opt::kActivationFunctionTypeNone) {
      return input;
    }
    nvinfer1::ActivationType type = nvinfer1::ActivationType::kRELU;
    float alpha = 0.0f;
    float beta = 0.0f;
    if (activation == Opt::kActivationFunctionTypeRelu) {
      type = nvinfer1::ActivationType::kRELU;
    } else if (activation == Opt::kActivationFunctionTypeRelu6) {
      type = nvinfer1::ActivationType::kCLIP;
      alpha = 0.0f;
      beta = 6.0f;
    } else if (activation == Opt::kActivationFunctionTypeReluN1To1) {
      type = nvinfer1::ActivationType::kCLIP;
      alpha = -1.0f;
      beta = 1.0f;
    } else {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported TensorRT fused activation");
    }
    auto* layer = network_->addActivation(*input, type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT activation");
    }
    if (type == nvinfer1::ActivationType::kCLIP) {
      layer->setAlpha(alpha);
      layer->setBeta(beta);
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> ExpandRankForElementwise(
      nvinfer1::ITensor* tensor, int target_rank) {
    const int rank = tensor->getDimensions().nbDims;
    if (rank == target_rank) {
      return tensor;
    }
    if (rank > target_rank || target_rank > nvinfer1::Dims::MAX_DIMS) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "TensorRT elementwise rank expansion is not supported");
    }

    nvinfer1::Dims reshape{};
    reshape.nbDims = target_rank;
    const int rank_delta = target_rank - rank;
    for (int i = 0; i < rank_delta; ++i) {
      reshape.d[i] = 1;
    }
    const auto dims = tensor->getDimensions();
    for (int i = 0; i < rank; ++i) {
      reshape.d[rank_delta + i] = dims.d[i];
    }

    auto* layer = network_->addShuffle(*tensor);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT elementwise rank expansion");
    }
    layer->setReshapeDimensions(reshape);
    return layer->getOutput(0);
  }

  Expected<void> MatchElementwiseRanks(nvinfer1::ITensor*& lhs,
                                       nvinfer1::ITensor*& rhs) {
    const int lhs_rank = lhs->getDimensions().nbDims;
    const int rhs_rank = rhs->getDimensions().nbDims;
    const int target_rank = std::max(lhs_rank, rhs_rank);
    LITERT_ASSIGN_OR_RETURN(lhs, ExpandRankForElementwise(lhs, target_rank));
    LITERT_ASSIGN_OR_RETURN(rhs, ExpandRankForElementwise(rhs, target_rank));
    return MatchElementwiseFloatTypes(lhs, rhs);
  }

  // TensorRT elementwise layers require identical operand types. Float
  // operands can still differ here: an FP16-typed LiteRT op may receive a
  // BF16 value produced by a BF16-only lowering (the CUDA GEMV plugin, an
  // FP32-widened reduction) next to a Float16 constant. Cast both to FP32
  // when either side is FP32, otherwise to the activation float type.
  Expected<void> MatchElementwiseFloatTypes(nvinfer1::ITensor*& lhs,
                                            nvinfer1::ITensor*& rhs) {
    const auto lhs_type = lhs->getType();
    const auto rhs_type = rhs->getType();
    auto is_float = [](nvinfer1::DataType type) {
      return type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kHALF ||
             type == nvinfer1::DataType::kBF16;
    };
    if (lhs_type == rhs_type || !is_float(lhs_type) || !is_float(rhs_type)) {
      return {};
    }
    nvinfer1::DataType target = nvinfer1::DataType::kFLOAT;
    if (lhs_type != nvinfer1::DataType::kFLOAT &&
        rhs_type != nvinfer1::DataType::kFLOAT && Fp16ActivationsEnabled()) {
      target = ModeFloatType();
    }
    LITERT_ASSIGN_OR_RETURN(lhs, AddCastTensor(lhs, target));
    LITERT_ASSIGN_OR_RETURN(rhs, AddCastTensor(rhs, target));
    return {};
  }

  Expected<nvinfer1::ITensor*> ExpandMatrixWeightsRank(
      nvinfer1::ITensor* weights, int target_rank) {
    const auto dims = weights->getDimensions();
    if (dims.nbDims == target_rank) {
      return weights;
    }
    if (dims.nbDims != 2 || target_rank < 2 ||
        target_rank > nvinfer1::Dims::MAX_DIMS) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "TensorRT matrix weight rank expansion is not supported");
    }

    nvinfer1::Dims reshape{};
    reshape.nbDims = target_rank;
    for (int i = 0; i < target_rank - 2; ++i) {
      reshape.d[i] = 1;
    }
    reshape.d[target_rank - 2] = dims.d[0];
    reshape.d[target_rank - 1] = dims.d[1];

    auto* layer = network_->addShuffle(*weights);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT matrix weight rank expansion");
    }
    layer->setReshapeDimensions(reshape);
    return layer->getOutput(0);
  }

  Expected<void> SetOutputTensor(const Tensor& tensor, nvinfer1::ITensor* out) {
    if (lowering_exact_fp32_) {
      if (out == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "TensorRT layer produced null output");
      }
      out->setName(KeepName(UniqueName("exact_fp32_tensor_" +
                                       std::to_string(tensor.TensorIndex()))));
      exact_fp32_map_[tensor.Get()] = out;
      return {};
    }
    // FP16-activation invariant: no FP32 value enters the tensor map, so
    // every float consumer sees a uniform type and Myelin fuses freely.
    if (Fp16ActivationsEnabled() &&
        out->getType() == nvinfer1::DataType::kFLOAT) {
      LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, ModeFloatType()));
    }
    if (out == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "TensorRT layer produced null output");
    }
    out->setName(
        KeepName(UniqueName("tensor_" + std::to_string(tensor.TensorIndex()))));
    tensor_map_[tensor.Get()] = out;
    return {};
  }

  Expected<void> LowerElementwise(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* lhs, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* rhs, GetTensor(op.Inputs()[1]));
    nvinfer1::ElementWiseOperation operation;
    switch (op.Code()) {
      case kLiteRtOpCodeTflAdd:
        operation = nvinfer1::ElementWiseOperation::kSUM;
        break;
      case kLiteRtOpCodeTflMul:
        operation = nvinfer1::ElementWiseOperation::kPROD;
        break;
      case kLiteRtOpCodeTflSub:
        operation = nvinfer1::ElementWiseOperation::kSUB;
        break;
      case kLiteRtOpCodeTflDiv:
        operation = nvinfer1::ElementWiseOperation::kDIV;
        break;
      case kLiteRtOpCodeTflMaximum:
        operation = nvinfer1::ElementWiseOperation::kMAX;
        break;
      case kLiteRtOpCodeTflMinimum:
        operation = nvinfer1::ElementWiseOperation::kMIN;
        break;
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT elementwise op");
    }
    const bool quantized_int8 =
        !Fp16ActivationsEnabled() && op.Code() != kLiteRtOpCodeTflMaximum &&
        op.Code() != kLiteRtOpCodeTflMinimum &&
        op.Inputs()[0].ElementType() == litert::ElementType::Int8;
    if (quantized_int8) {
      LITERT_ASSIGN_OR_RETURN(
          lhs,
          AddDequantizeTensor(op.Inputs()[0], lhs, nvinfer1::DataType::kFLOAT));
      LITERT_ASSIGN_OR_RETURN(
          rhs,
          AddDequantizeTensor(op.Inputs()[1], rhs, nvinfer1::DataType::kFLOAT));
    }
    LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(lhs, rhs));
    auto* layer = network_->addElementWise(*lhs, *rhs, operation);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT elementwise layer");
    }
    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    LITERT_ASSIGN_OR_RETURN(
        auto* out, AddFusedActivation(layer->getOutput(0), activation));
    if (quantized_int8) {
      LITERT_ASSIGN_OR_RETURN(
          out,
          AddQuantizeTensor(out, op.Outputs()[0], nvinfer1::DataType::kINT8));
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerComparison(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* lhs, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* rhs, GetTensor(op.Inputs()[1]));
    nvinfer1::ElementWiseOperation operation;
    bool invert_result = false;
    switch (op.Code()) {
      case kLiteRtOpCodeTflLess:
        operation = nvinfer1::ElementWiseOperation::kLESS;
        break;
      case kLiteRtOpCodeTflGreater:
        operation = nvinfer1::ElementWiseOperation::kGREATER;
        break;
      case kLiteRtOpCodeTflGreaterEqual:
        operation = nvinfer1::ElementWiseOperation::kLESS;
        invert_result = true;
        break;
      case kLiteRtOpCodeTflNotEqual:
        operation = nvinfer1::ElementWiseOperation::kEQUAL;
        invert_result = true;
        break;
      case kLiteRtOpCodeTflEqual:
        operation = nvinfer1::ElementWiseOperation::kEQUAL;
        break;
      case kLiteRtOpCodeTflLogicalAnd:
        operation = nvinfer1::ElementWiseOperation::kAND;
        break;
      case kLiteRtOpCodeTflLogicalOr:
        operation = nvinfer1::ElementWiseOperation::kOR;
        break;
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT comparison op");
    }
    LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(lhs, rhs));
    auto* layer = network_->addElementWise(*lhs, *rhs, operation);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT comparison layer");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    if (invert_result) {
      auto* not_layer =
          network_->addUnary(*out, nvinfer1::UnaryOperation::kNOT);
      if (not_layer == nullptr || not_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT logical-not layer");
      }
      out = not_layer->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerActivation(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    uint32_t activation = litert::kActivationFunctionTypeNone;
    switch (op.Code()) {
      case kLiteRtOpCodeTflRelu:
        activation = litert::kActivationFunctionTypeRelu;
        break;
      case kLiteRtOpCodeTflRelu6:
        activation = litert::kActivationFunctionTypeRelu6;
        break;
      case kLiteRtOpCodeTflReluN1To1:
        activation = litert::kActivationFunctionTypeReluN1To1;
        break;
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT activation op");
    }
    LITERT_ASSIGN_OR_RETURN(auto* out, AddFusedActivation(input, activation));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerUnary(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    // Rotary angles reach whole radians at long positions; take them in FP32.
    // The result is in [-1, 1] and SetOutputTensor narrows it as usual.
    if ((op.Code() == kLiteRtOpCodeTflSin ||
         op.Code() == kLiteRtOpCodeTflCos) &&
        HasExactFp32Value(op.Inputs()[0])) {
      LITERT_ASSIGN_OR_RETURN(input, GetExactFp32Tensor(op.Inputs()[0]));
    }
    if (op.Code() == kLiteRtOpCodeTflTanh ||
        op.Code() == kLiteRtOpCodeTflLogistic) {
      auto* layer = network_->addActivation(
          *input, op.Code() == kLiteRtOpCodeTflTanh
                      ? nvinfer1::ActivationType::kTANH
                      : nvinfer1::ActivationType::kSIGMOID);
      if (layer == nullptr || layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT activation layer");
      }
      return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
    }

    nvinfer1::UnaryOperation operation;
    switch (op.Code()) {
      case kLiteRtOpCodeTflSin:
        operation = nvinfer1::UnaryOperation::kSIN;
        break;
      case kLiteRtOpCodeTflCos:
        operation = nvinfer1::UnaryOperation::kCOS;
        break;
      case kLiteRtOpCodeTflRsqrt:
      case kLiteRtOpCodeTflSqrt:
        operation = nvinfer1::UnaryOperation::kSQRT;
        break;
      case kLiteRtOpCodeTflAbs:
        operation = nvinfer1::UnaryOperation::kABS;
        break;
      case kLiteRtOpCodeTflSign:
        operation = nvinfer1::UnaryOperation::kSIGN;
        break;
      case kLiteRtOpCodeTflExp:
        operation = nvinfer1::UnaryOperation::kEXP;
        break;
      case kLiteRtOpCodeTflLog:
        operation = nvinfer1::UnaryOperation::kLOG;
        break;
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT unary op");
    }
    auto* layer = network_->addUnary(*input, operation);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT unary layer");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    if (op.Code() == kLiteRtOpCodeTflRsqrt) {
      auto* recip = network_->addUnary(*out, nvinfer1::UnaryOperation::kRECIP);
      if (recip == nullptr || recip->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT reciprocal layer");
      }
      out = recip->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerGelu(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    bool approximate = false;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetGeluApproximateOption(op.Get(), &approximate));
    auto* layer = network_->addActivation(
        *input, approximate ? nvinfer1::ActivationType::kGELU_TANH
                            : nvinfer1::ActivationType::kGELU_ERF);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT GELU layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerCast(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto trt_type,
                            ConvertDataType(output_type.ElementType()));
    // Positions turned back into integers (e.g. cache update offsets) must
    // come from the exact view, not from a rounded FP16/BF16 copy.
    if (IsPositionElementType(output_type.ElementType()) &&
        HasExactFp32Value(op.Inputs()[0])) {
      LITERT_ASSIGN_OR_RETURN(input, GetExactFp32Tensor(op.Inputs()[0]));
    }
    auto* layer = network_->addCast(*input, trt_type);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT cast layer");
    }
    // Integers and booleans turned into floats keep an exact FP32 view: the
    // valid-row counts of Qwen 3.5's prefill (a sum over a boolean mask) must
    // not be rounded to BF16 (whose integers above 256 are 2 or 4 apart).
    if (Fp16ActivationsEnabled() && trt_type == nvinfer1::DataType::kFLOAT &&
        (IsPositionElementType(op.Inputs()[0].ElementType()) ||
         op.Inputs()[0].ElementType() == litert::ElementType::Bool)) {
      exact_fp32_map_[op.Outputs()[0].Get()] = layer->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerQuantize(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    if (Fp16ActivationsEnabled()) {
      // Values already flow as FP16 reals; a (re)quantize only clamps to the
      // output tensor's representable range.
      nvinfer1::ITensor* out = input;
      if (op.Outputs()[0].QTypeId() == kLiteRtQuantizationPerTensor) {
        const float scale = op.Outputs()[0].PerTensorQuantization().scale;
        auto* clip_layer =
            network_->addActivation(*input, nvinfer1::ActivationType::kCLIP);
        if (clip_layer == nullptr || clip_layer->getOutput(0) == nullptr) {
          return Error(kLiteRtStatusErrorCompilation,
                       "Failed to add TensorRT quantize clamp layer");
        }
        clip_layer->setAlpha(-128.0f * scale);
        clip_layer->setBeta(127.0f * scale);
        out = clip_layer->getOutput(0);
      }
      return SetOutputTensor(op.Outputs()[0], out);
    }
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto trt_type,
                            ConvertDataType(output_type.ElementType()));
    LITERT_ASSIGN_OR_RETURN(
        auto* out, AddQuantizeTensor(input, op.Outputs()[0], trt_type));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerDequantize(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    if (Fp16ActivationsEnabled() &&
        input->getType() != nvinfer1::DataType::kINT8 &&
        input->getType() != nvinfer1::DataType::kINT4) {
      // The value is already the dequantized FP16 real.
      return SetOutputTensor(op.Outputs()[0], input);
    }
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto trt_type,
                            ConvertDataType(output_type.ElementType()));
    if (Fp16ActivationsEnabled()) {
      trt_type = ModeFloatType();
    }
    LITERT_ASSIGN_OR_RETURN(
        auto* out, AddDequantizeTensor(op.Inputs()[0], input, trt_type));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerFill(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(
        auto* constant,
        AddRepeatedScalarConstant(op.Inputs()[1], op.Outputs()[0]));
    return SetOutputTensor(op.Outputs()[0], constant);
  }

  Expected<void> LowerSoftmax(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    auto* layer = network_->addSoftMax(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT softmax layer");
    }
    const int rank = input->getDimensions().nbDims;
    if (rank <= 0 || rank >= 32) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported softmax tensor rank");
    }
    layer->setAxes(1U << (rank - 1));
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerReshape(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT reshape layer");
    }
    layer->setReshapeDimensions(output_dims);
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerPack(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    int32_t axis = 0;
    LITERT_RETURN_IF_ERROR(LiteRtGetPackAxisOption(op.Get(), &axis));
    const int output_rank = output_dims.nbDims;
    if (axis < 0) {
      axis += output_rank;
    }

    nvinfer1::Dims expanded_input_dims = output_dims;
    expanded_input_dims.d[axis] = 1;
    std::vector<nvinfer1::ITensor*> expanded_inputs;
    expanded_inputs.reserve(op.Inputs().size());
    for (const auto& input : op.Inputs()) {
      LITERT_ASSIGN_OR_RETURN(auto* input_tensor, GetTensor(input));
      LITERT_ASSIGN_OR_RETURN(auto* expanded,
                              ReshapeTensor(input_tensor, expanded_input_dims));
      expanded_inputs.push_back(expanded);
    }

    auto* layer = network_->addConcatenation(
        expanded_inputs.data(), static_cast<int32_t>(expanded_inputs.size()));
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT pack concatenation layer");
    }
    layer->setAxis(axis);
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerSlice(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    const int rank = input_type.Layout().Rank();
    if (output_dims.nbDims != rank) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Slice rank metadata does not match input rank");
    }
    nvinfer1::Dims start{};
    nvinfer1::Dims stride{};
    start.nbDims = rank;
    stride.nbDims = rank;
    for (int i = 0; i < rank; ++i) {
      start.d[i] = 0;
      stride.d[i] = 1;
    }
    auto begin = ReadInt32Constant(op.Inputs()[1]);
    if (!begin) {
      // Runtime slice offsets lower to a gather of `begin[axis] + iota` along
      // the single shrinking axis. A slice-layer start tensor would become a
      // TensorRT shape input, which requires optimization profiles.
      const auto input_dims = input->getDimensions();
      int slice_axis = 0;
      for (int i = 0; i < rank; ++i) {
        if (output_dims.d[i] != input_dims.d[i]) {
          slice_axis = i;
          break;
        }
      }
      LITERT_ASSIGN_OR_RETURN(auto* begin_tensor, GetTensor(op.Inputs()[1]));
      LITERT_ASSIGN_OR_RETURN(
          auto* begin_scalar,
          SliceTensor1d(begin_tensor, /*start_index=*/slice_axis, /*size=*/1));
      const int32_t gather_size = output_dims.d[slice_axis];
      std::vector<int32_t> iota(gather_size);
      for (int32_t i = 0; i < gather_size; ++i) {
        iota[i] = i;
      }
      nvinfer1::Dims iota_dims{};
      iota_dims.nbDims = 1;
      iota_dims.d[0] = gather_size;
      LITERT_ASSIGN_OR_RETURN(auto* iota_tensor,
                              AddInt32Constant(iota, iota_dims, "slice_iota"));
      auto* index_layer = network_->addElementWise(
          *iota_tensor, *begin_scalar, nvinfer1::ElementWiseOperation::kSUM);
      if (index_layer == nullptr || index_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT slice index layer");
      }
      auto* gather_layer =
          network_->addGather(*input, *index_layer->getOutput(0), slice_axis);
      if (gather_layer == nullptr || gather_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT slice gather layer");
      }
      return SetOutputTensor(op.Outputs()[0], gather_layer->getOutput(0));
    }
    if (begin->size() != static_cast<size_t>(rank)) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Slice begin size does not match input rank");
    }
    for (int i = 0; i < rank; ++i) {
      start.d[i] = (*begin)[i];
    }
    auto* layer = network_->addSlice(*input, start, output_dims, stride);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT slice layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<nvinfer1::ITensor*> SliceTensor1d(nvinfer1::ITensor* input,
                                             int start_index, int size) {
    nvinfer1::Dims start{};
    nvinfer1::Dims output_dims{};
    nvinfer1::Dims stride{};
    start.nbDims = 1;
    output_dims.nbDims = 1;
    stride.nbDims = 1;
    start.d[0] = start_index;
    output_dims.d[0] = size;
    stride.d[0] = 1;
    auto* layer = network_->addSlice(*input, start, output_dims, stride);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT 1D slice layer");
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> ReshapeTensor(nvinfer1::ITensor* input,
                                             nvinfer1::Dims dims) {
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT reshape helper layer");
    }
    layer->setReshapeDimensions(dims);
    return layer->getOutput(0);
  }

  Expected<void> LowerUnpack(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto input_type, op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto input_dims, ConvertDims(input_type));
    int32_t axis = 0;
    LITERT_RETURN_IF_ERROR(LiteRtGetUnpackAxisOption(op.Get(), &axis));
    const int input_rank = input_dims.nbDims;
    if (axis < 0) {
      axis += input_rank;
    }

    nvinfer1::Dims stride{};
    stride.nbDims = input_rank;
    for (int i = 0; i < input_rank; ++i) {
      stride.d[i] = 1;
    }

    for (int i = 0; i < op.Outputs().size(); ++i) {
      nvinfer1::Dims start{};
      nvinfer1::Dims size = input_dims;
      start.nbDims = input_rank;
      for (int dim = 0; dim < input_rank; ++dim) {
        start.d[dim] = 0;
      }
      start.d[axis] = i;
      size.d[axis] = 1;
      auto* slice = network_->addSlice(*input, start, size, stride);
      if (slice == nullptr || slice->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT unpack slice layer");
      }
      LITERT_ASSIGN_OR_RETURN(auto output_type,
                              op.Outputs()[i].RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
      LITERT_ASSIGN_OR_RETURN(auto* reshaped,
                              ReshapeTensor(slice->getOutput(0), output_dims));
      LITERT_RETURN_IF_ERROR(SetOutputTensor(op.Outputs()[i], reshaped));
    }
    return {};
  }

  struct InPlaceCache {
    LiteRtOp op = nullptr;
    int partial_axis = -1;
    // A value cache held as [B, H, S, D], the layout the attention matmul
    // reads without a transpose (see TransposedValueCacheEnabled).
    bool transposed = false;
  };

  // A closed attention join can pass through the cache patch plugin, putting
  // preparation in the layer's execution chain instead of retaining all old
  // score matrices until the end of prefill. All old-reader paths must reach
  // the join, and none of its consumers may already have been lowered.
  std::optional<Tensor> FindPrefillAttentionJoin(const Op& update) const {
    const auto update_order = op_order_.find(update.Get());
    if (update_order == op_order_.end()) return std::nullopt;
    std::vector<Tensor> readers;
    for (int i : {3, 4}) {
      for (const auto& use : update.Inputs()[i].Uses()) {
        if (use.user.Get() == update.Get()) continue;
        for (const auto& output : use.user.Outputs()) readers.push_back(output);
      }
    }
    auto update_type = update.Inputs()[0].RankedTensorType();
    if (!update_type) return std::nullopt;
    auto update_elements = NumElements(*update_type);
    if (!update_elements ||
        *update_elements > std::numeric_limits<size_t>::max() / 2) {
      return std::nullopt;
    }
    for (const auto& use : update.Inputs()[4].Uses()) {
      if (use.user.Get() == update.Get()) continue;
      for (const auto& value_result : use.user.Outputs()) {
        for (const auto& consumer : value_result.Uses()) {
          const Op& join = consumer.user;
          if (join.Code() != kLiteRtOpCodeTflAdd ||
              join.Outputs().size() != 1) {
            continue;
          }
          const Tensor candidate = join.Outputs()[0];
          const auto join_order = op_order_.find(join.Get());
          auto type = candidate.RankedTensorType();
          if (join_order == op_order_.end() ||
              join_order->second >= update_order->second || !type ||
              !IsFloatLike(type->ElementType())) {
            continue;
          }
          auto elements = NumElements(*type);
          if (!elements || *elements > 2 * *update_elements) continue;
          bool closed = true;
          for (const auto& next : candidate.Uses()) {
            const auto order = op_order_.find(next.user.Get());
            if (order == op_order_.end() ||
                order->second <= update_order->second) {
              closed = false;
              break;
            }
          }
          std::vector<Tensor> pending = readers;
          std::unordered_set<LiteRtTensor> visited;
          while (closed && !pending.empty()) {
            const Tensor current = pending.back();
            pending.pop_back();
            if (current.Get() == candidate.Get() ||
                !visited.insert(current.Get()).second) {
              continue;
            }
            const auto uses = current.Uses();
            if (graph_outputs_.count(current.Get()) != 0 || uses.empty()) {
              closed = false;
              break;
            }
            for (const auto& next : uses) {
              const auto order = op_order_.find(next.user.Get());
              if (order == op_order_.end() ||
                  order->second >= update_order->second ||
                  next.user.Outputs().empty()) {
                closed = false;
                break;
              }
              for (const auto& output : next.user.Outputs()) {
                pending.push_back(output);
              }
            }
          }
          if (closed) return candidate;
        }
      }
    }
    return std::nullopt;
  }

  // True when every reader of `tensor` is lowered for a value cache held
  // transposed: the in-place update itself, or the right operand of a
  // runtime_bmm composite or transposed batch matmul (probs @ V).
  static bool OnlyTransposedMatmulReaders(const Tensor& tensor,
                                          LiteRtOp update_op) {
    for (const auto& use : tensor.Uses()) {
      const Op& user = use.user;
      if (user.Get() == update_op &&
          (use.user_arg_ind == 0 ||
           (user.Code() == kLiteRtOpCodeShloComposite &&
            CompositeOpName(user) == "odml.cache_update" &&
            use.user_arg_ind == 4))) {
        continue;
      }
      // The value of the attention composites (Qwen 4's sparse attention
      // reads a value cache held as rows the same way: its in-place rule).
      if (user.Code() == kLiteRtOpCodeShloComposite &&
          (CompositeOpName(user) == "odml.sdpa_transposed" ||
           CompositeOpName(user) == "odml.sparse_sdpa_transposed") &&
          use.user_arg_ind == 2) {
        continue;
      }
      if (use.user_arg_ind != 1) {
        return false;
      }
      if (user.Code() == kLiteRtOpCodeShloComposite &&
          CompositeOpName(user) == "odml.runtime_bmm") {
        continue;
      }
      bool adj_y = false;
      if (user.Code() == kLiteRtOpCodeTflBatchMatmul &&
          LiteRtGetBatchMatmulAdjYOption(user.Get(), &adj_y) ==
              kLiteRtStatusOk &&
          adj_y) {
        continue;
      }
      return false;
    }
    return true;
  }

  Expected<void> FindReadOnlyValueCaches(
      const Subgraph& subgraph, absl::Span<const std::string> input_names) {
    if (input_names.empty()) {
      return {};
    }
    if (!NativeKvCacheUpdateEnabled() || !TransposedValueCacheEnabled() ||
        RuntimeBmmContextLimit() != 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Read-only value cache layout conflicts with NVIDIA "
                   "cache-update, layout, or context-limit settings");
    }
    std::unordered_set<std::string> remaining(input_names.begin(),
                                              input_names.end());
    if (remaining.size() != input_names.size() || remaining.count("") != 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Read-only value cache input names must be nonempty and "
                   "unique");
    }
    for (const Tensor& input : subgraph.Inputs()) {
      const std::string name(input.Name());
      if (remaining.erase(name) == 0) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(auto type, input.RankedTensorType());
      uint32_t strides[4];
      if (!IsFloatLike(type.ElementType()) || type.Layout().HasStrides() ||
          !GetTransposedValueCacheStrides(
              static_cast<const LiteRtLayout&>(type.Layout()), strides) ||
          input.Uses().empty() ||
          !OnlyTransposedMatmulReaders(input, /*update_op=*/nullptr)) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Read-only value cache requires a static rank-4 float "
                     "input consumed only by transposed attention matmuls: " +
                         name);
      }
      transposed_caches_.insert(input.Get());
    }
    if (!remaining.empty()) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Read-only value cache is not a partition input: " +
                       *remaining.begin());
    }
    return {};
  }

  // The KV caches another engine of the process keeps in place and this graph
  // only reads (LiteRtNvidiaSharedCacheInputs, a drafter over its base
  // model's caches): rank-4 inputs the graph uses. They are declared with the
  // storage the in-place caches get and attention reads them like the
  // outputs of an in-place update.
  Expected<void> FindSharedCacheInputs(
      const Subgraph& subgraph, absl::Span<const std::string> input_names) {
    if (input_names.empty()) {
      return {};
    }
    std::unordered_set<std::string> remaining(input_names.begin(),
                                              input_names.end());
    if (remaining.size() != input_names.size() || remaining.count("") != 0) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Shared cache input names must be nonempty and unique");
    }
    for (const Tensor& input : subgraph.Inputs()) {
      const std::string name(input.Name());
      if (remaining.erase(name) == 0) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(auto type, input.RankedTensorType());
      if (type.Layout().Rank() != 4 || type.Layout().HasStrides() ||
          input.Uses().empty()) {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     "Shared cache input must be a dense rank-4 tensor the "
                     "graph reads: " +
                         name);
      }
      shared_cache_inputs_.insert(input.Get());
    }
    if (!remaining.empty()) {
      return Error(
          kLiteRtStatusErrorInvalidArgument,
          "Shared cache is not a partition input: " + *remaining.begin());
    }
    return {};
  }

  // A KV cache update is a DynamicUpdateSlice whose operand is a subgraph
  // input and whose result is a subgraph output: the caller passes the same
  // buffer for both. Lowered as a scatter, TensorRT copies the whole cache to
  // the output on every invocation before writing the update. TensorRT-RTX's
  // KVCacheUpdate layer instead writes in place into an output that aliases
  // its input, so this pass records which inputs qualify: a rank-4 float
  // cache with batch 1 updated along its sequence axis, either [B, H, S, D]
  // (axis 2) or the transposed value layout [B, H, D, S] (axis 3).
  Expected<void> FindInPlaceCacheUpdates(const Subgraph& subgraph) {
    if (!NativeKvCacheUpdateEnabled()) {
      return {};
    }
    std::unordered_set<LiteRtTensor> inputs;
    for (const auto& input : subgraph.Inputs()) {
      inputs.insert(input.Get());
    }
    std::unordered_set<LiteRtTensor> outputs;
    for (const auto& output : subgraph.Outputs()) {
      outputs.insert(output.Get());
    }
    graph_outputs_ = outputs;
    for (const auto& op : subgraph.Ops()) {
      if (op.Code() == kLiteRtOpCodeShloComposite &&
          CompositeOpName(op) == "odml.cache_update" &&
          PrefillCacheUpdateRingMode(op).has_value()) {
        if (outputs.count(op.Outputs()[0].Get()) == 0 ||
            outputs.count(op.Outputs()[1].Get()) == 0) {
          return Error(kLiteRtStatusErrorUnsupported,
                       "Float cache update must expose both cache outputs");
        }
        const bool transposed =
            TransposedValueCacheEnabled() && RuntimeBmmContextLimit() == 0 &&
            OnlyTransposedMatmulReaders(op.Inputs()[4], op.Get()) &&
            OnlyTransposedMatmulReaders(op.Outputs()[1], op.Get());
        for (int i = 0; i < 2; ++i) {
          const Tensor cache = op.Inputs()[3 + i];
          if (in_place_caches_.count(cache.Get()) != 0) {
            return Error(kLiteRtStatusErrorUnsupported,
                         "Multiple in-place updates to the same KV cache");
          }
          InPlaceCache info{op.Get(), 2 + i, i == 1 && transposed};
          in_place_caches_[cache.Get()] = info;
          in_place_outputs_[op.Outputs()[i].Get()] = info;
          if (info.transposed) {
            transposed_caches_.insert(cache.Get());
            transposed_caches_.insert(op.Outputs()[i].Get());
          }
        }
        continue;
      }
      if (op.Code() == kLiteRtOpCodeShloComposite &&
          CompositeOpName(op) == "odml.cache_update" &&
          DecodeCacheUpdateMode(op).has_value()) {
        if (outputs.count(op.Outputs()[0].Get()) == 0 ||
            outputs.count(op.Outputs()[1].Get()) == 0) {
          return Error(kLiteRtStatusErrorUnsupported,
                       "Cache update must expose both cache outputs");
        }
        // The value cache is held as [B, H, S, D] when only the fused
        // attention (or transposed matmuls) read it.
        const auto transposed_readers = [&](const Tensor& tensor) {
          for (const auto& use : tensor.Uses()) {
            if (use.user.Get() == op.Get()) continue;
            const bool sdpa =
                use.user.Code() == kLiteRtOpCodeShloComposite &&
                (CompositeOpName(use.user) == "odml.sdpa_transposed" ||
                 CompositeOpName(use.user) == "odml.sparse_sdpa_transposed") &&
                use.user_arg_ind == 2;
            if (!sdpa &&
                !(use.user_arg_ind == 1 && IsTransposedMatmul(use.user))) {
              return false;
            }
          }
          return true;
        };
        const bool transposed = TransposedValueCacheEnabled() &&
                                RuntimeBmmContextLimit() == 0 &&
                                transposed_readers(op.Inputs()[4]) &&
                                transposed_readers(op.Outputs()[1]);
        for (int i = 0; i < 2; ++i) {
          const Tensor cache = op.Inputs()[3 + i];
          if (in_place_caches_.count(cache.Get()) != 0) {
            return Error(kLiteRtStatusErrorUnsupported,
                         "Multiple in-place updates to the same KV cache");
          }
          InPlaceCache info{op.Get(), 2 + i, i == 1 && transposed};
          in_place_caches_[cache.Get()] = info;
          in_place_outputs_[op.Outputs()[i].Get()] = info;
          if (info.transposed) {
            transposed_caches_.insert(cache.Get());
            transposed_caches_.insert(op.Outputs()[i].Get());
          }
        }
        continue;
      }
      if (op.Code() != kLiteRtOpCodeTflDynamicUpdateSlice ||
          op.Inputs().size() != 3 || op.Outputs().size() != 1) {
        continue;
      }
      const Tensor cache = op.Inputs()[0];
      if (inputs.find(cache.Get()) == inputs.end() ||
          outputs.find(op.Outputs()[0].Get()) == outputs.end() ||
          in_place_caches_.find(cache.Get()) != in_place_caches_.end() ||
          !IsFloatLike(cache.ElementType()) ||
          op.Inputs()[2].ElementType() != litert::ElementType::Int32) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(auto cache_type, cache.RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto update_type,
                              op.Inputs()[1].RankedTensorType());
      const auto cache_dims = cache_type.Layout().Dimensions();
      const auto update_dims = update_type.Layout().Dimensions();
      if (cache_dims.size() != 4 || update_dims.size() != 4 ||
          cache_dims[0] != 1) {
        continue;
      }
      int partial_axis = -1;
      bool supported = true;
      for (int i = 0; i < 4; ++i) {
        if (update_dims[i] == cache_dims[i]) {
          continue;
        }
        if (update_dims[i] > cache_dims[i] || partial_axis >= 0) {
          supported = false;
          break;
        }
        partial_axis = i;
      }
      if (!supported || (partial_axis != 2 && partial_axis != 3)) {
        continue;
      }
      const bool transposed =
          partial_axis == 3 && TransposedValueCacheEnabled() &&
          RuntimeBmmContextLimit() == 0 &&
          OnlyTransposedMatmulReaders(cache, op.Get()) &&
          OnlyTransposedMatmulReaders(op.Outputs()[0], op.Get());
      in_place_caches_[cache.Get()] = {op.Get(), partial_axis, transposed};
      in_place_outputs_[op.Outputs()[0].Get()] = in_place_caches_[cache.Get()];
      if (transposed) {
        transposed_caches_.insert(cache.Get());
        transposed_caches_.insert(op.Outputs()[0].Get());
      }
    }
    if (!in_place_caches_.empty()) {
      const size_t num_transposed_updates = std::count_if(
          in_place_caches_.begin(), in_place_caches_.end(),
          [](const auto& cache) { return cache.second.transposed; });
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX lowering %zu KV cache updates in place "
                 "(%zu value caches held as [B, H, S, D])",
                 in_place_caches_.size(), num_transposed_updates);
    }
    return {};
  }

  // One attention block, all of whose intermediate results have single uses:
  // in decode and for the global layers of prefill the four ops between the
  // in-place K/V cache updates and the output projection, which read the
  // updated caches; for the ring caches of prefill the ops between the old
  // caches, the chunk's own keys and values, and the join of the two value
  // products.
  struct DecodeAttentionBlock {
    LiteRtTensor q = nullptr;
    LiteRtTensor k_out = nullptr;  // the key cache the block reads
    LiteRtTensor v_out = nullptr;
    LiteRtTensor mask = nullptr;
    // Without a mask: the params tensor the mask comes from, and is_causal
    // (MaskFromSdpaParams).
    LiteRtTensor params = nullptr;
    bool causal = false;
    LiteRtTensor k_new = nullptr;  // the chunk's keys, for a ring cache
    LiteRtTensor v_new = nullptr;
    std::optional<Tensor> output;
    float fill = 0.0f;
    // Set for INT8 per-tensor caches: the plugin reads the raw INT8 caches
    // (the value cache in the model's [1, H, D, S] layout).
    std::optional<LiteRtNvidiaInt8CacheParams> int8_cache;
    // Set when the block is a sliding-window read (windowed_reads_): the
    // plugin takes the window's slices of the caches and the mask.
    std::optional<size_t> window;
    // Query heads per KV head of an odml.sdpa_transposed block: q
    // [1, Hq, n, D] runs as [1, H, group * n, D].
    int32_t group = 1;
    bool emitted = false;
  };

  // Records every odml.sdpa_transposed composite (SdpaTransposedGroup) as a
  // fused attention block over the in-place cache updates that feed it.
  Expected<void> FindCompositeAttentionBlocks(const Subgraph& subgraph) {
    if (!Fp16ActivationsEnabled()) {
      return {};
    }
    size_t found = 0;
    for (const auto& op : subgraph.Ops()) {
      if (op.Code() != kLiteRtOpCodeShloComposite ||
          CompositeOpName(op) != "odml.sdpa_transposed") {
        continue;
      }
      const auto group = SdpaTransposedGroup(op);
      if (!group.has_value()) {
        continue;
      }
      // The caches: outputs of this graph's in-place updates, or caches
      // another engine keeps (shared_cache_inputs_), whose value cache the
      // read-only options declared transposed.
      const LiteRtTensor k = op.Inputs()[1].Get();
      const LiteRtTensor v = op.Inputs()[2].Get();
      const bool k_ok =
          in_place_outputs_.count(k) != 0 || shared_cache_inputs_.count(k) != 0;
      const auto v_in_place = in_place_outputs_.find(v);
      const bool v_ok = v_in_place != in_place_outputs_.end()
                            ? v_in_place->second.transposed
                            : shared_cache_inputs_.count(v) != 0 &&
                                  transposed_caches_.count(v) != 0;
      if (!k_ok || !v_ok) {
        continue;
      }
      DecodeAttentionBlock block;
      block.q = op.Inputs()[0].Get();
      block.k_out = op.Inputs()[1].Get();
      block.v_out = op.Inputs()[2].Get();
      if (op.Inputs().size() == 5) {
        block.mask = op.Inputs()[3].Get();
      } else {
        block.params = op.Inputs()[3].Get();
        block.causal = CompositeBoolAttribute(op, "is_causal");
      }
      block.output = op.Outputs()[0];
      block.fill = kSdpaTransposedFill;
      block.group = *group;
      const size_t index = attention_blocks_.size();
      attention_blocks_.push_back(block);
      fused_attention_ops_[op.Get()] = index;
      ++found;
    }
    if (found != 0) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX fusing %zu sdpa_transposed composites "
                 "into the CUDA attention plugin",
                 found);
    }
    return {};
  }

  // A sliding-window attention block over a full-length cache. Gemma 4's
  // E-models keep every cache at the context length and build a local
  // layer's mask in-graph from the key positions, pos >= A[j] && pos < B[j]
  // with A[j] = j and B[j] = j + W, so the window W is in the constants.
  // Such a block sees at most W + tokens - 1 keys ending at the chunk's last
  // position; reading that span instead of the whole cache makes the layer
  // cost its window, not the context, in prefill as in decode.
  // One concatenated piece of a local mask: positions [1, R, 1] compared
  // with the key-index constants lower [1, 1, S] and upper [1, 1, S].
  struct WindowMaskPiece {
    std::optional<Tensor> positions;
    std::optional<Tensor> lower;
    std::optional<Tensor> upper;
    int64_t rows = 0;
  };

  struct WindowedCacheRead {
    LiteRtTensor k_out = nullptr;
    LiteRtTensor v_out = nullptr;
    std::optional<Tensor> mask;
    std::optional<Tensor> start_indices;  // the update's K start [4]
    std::vector<WindowMaskPiece> pieces;  // the mask's construction
    LiteRtOp scores = nullptr;
    LiteRtOp select = nullptr;
    LiteRtOp values = nullptr;
    int32_t seq = 0;
    int32_t tokens = 0;  // keys the update writes (1 decode, chunk prefill)
    int32_t window = 0;
    int32_t length = 0;  // keys read: window + tokens + 62 rounded up to 64
    int32_t mask_rows = 0;
    nvinfer1::ITensor* start = nullptr;    // [1] = s0, a multiple of 64
    nvinfer1::ITensor* indices = nullptr;  // [length] = s0 + iota
    nvinfer1::ITensor* k_slice = nullptr;
    nvinfer1::ITensor* v_slice = nullptr;
    nvinfer1::ITensor* mask_slice = nullptr;
  };

  // The window of a mask built as reshape(not_equal(concatenation(pieces),
  // 0)) with every piece cast([reshape](logical_and(greater_equal(pos, A),
  // less(pos, B)))), A[j] = j and B[j] - A[j] constant; nullopt otherwise
  // (a global layer's mask is a subgraph input). `pieces` receives the
  // positions and constants of each piece.
  static std::optional<int32_t> WindowFromMask(
      const Tensor& mask, int32_t seq, std::vector<WindowMaskPiece>* pieces) {
    Tensor tensor = mask;
    auto op = tensor.GetDefiningOp();
    while (op && op->Code() == kLiteRtOpCodeTflReshape &&
           !op->Inputs().empty()) {
      tensor = op->Inputs()[0];
      op = tensor.GetDefiningOp();
    }
    if (!op || op->Code() != kLiteRtOpCodeTflNotEqual ||
        op->Inputs().size() != 2) {
      return std::nullopt;
    }
    tensor = op->Inputs()[0];
    op = tensor.GetDefiningOp();
    if (!op) {
      return std::nullopt;
    }
    std::vector<Tensor> parts;
    if (op->Code() == kLiteRtOpCodeTflConcatenation) {
      for (const auto& part : op->Inputs()) parts.push_back(part);
    } else {
      parts.push_back(tensor);
    }
    std::optional<int32_t> window;
    for (Tensor piece : parts) {
      auto producer = piece.GetDefiningOp();
      if (producer && producer->Code() == kLiteRtOpCodeTflCast &&
          !producer->Inputs().empty()) {
        piece = producer->Inputs()[0];
        producer = piece.GetDefiningOp();
      }
      while (producer && producer->Code() == kLiteRtOpCodeTflReshape &&
             !producer->Inputs().empty()) {
        piece = producer->Inputs()[0];
        producer = piece.GetDefiningOp();
      }
      if (!producer || producer->Code() != kLiteRtOpCodeTflLogicalAnd ||
          producer->Inputs().size() != 2) {
        return std::nullopt;
      }
      std::optional<std::vector<int32_t>> lower, upper;
      WindowMaskPiece recipe;
      for (const auto& operand : producer->Inputs()) {
        auto compare = operand.GetDefiningOp();
        if (!compare || compare->Inputs().size() != 2) {
          return std::nullopt;
        }
        auto bound = ReadInt32Constant(compare->Inputs()[1]);
        if (!bound || bound->size() != static_cast<size_t>(seq)) {
          return std::nullopt;
        }
        auto positions_type = compare->Inputs()[0].RankedTensorType();
        auto bound_type = compare->Inputs()[1].RankedTensorType();
        if (!positions_type || !bound_type ||
            positions_type->Layout().Rank() != 3 ||
            bound_type->Layout().Rank() != 3 ||
            positions_type->Layout().Dimensions()[0] != 1 ||
            positions_type->Layout().Dimensions()[2] != 1 ||
            bound_type->Layout().Dimensions()[0] != 1 ||
            bound_type->Layout().Dimensions()[1] != 1) {
          return std::nullopt;
        }
        if (compare->Code() == kLiteRtOpCodeTflGreaterEqual) {
          lower = std::move(*bound);
          recipe.lower = compare->Inputs()[1];
        } else if (compare->Code() == kLiteRtOpCodeTflLess) {
          upper = std::move(*bound);
          recipe.upper = compare->Inputs()[1];
        } else {
          return std::nullopt;
        }
        if (recipe.positions.has_value() &&
            recipe.positions->Get() != compare->Inputs()[0].Get()) {
          return std::nullopt;
        }
        recipe.positions = compare->Inputs()[0];
        recipe.rows = positions_type->Layout().Dimensions()[1];
      }
      if (!lower || !upper) {
        return std::nullopt;
      }
      pieces->push_back(recipe);
      const int32_t w = (*upper)[0] - (*lower)[0];
      for (int32_t j = 0; j < seq; ++j) {
        if ((*lower)[j] != j || (*upper)[j] - (*lower)[j] != w) {
          return std::nullopt;
        }
      }
      if (w <= 0 || w >= seq || (window.has_value() && *window != w)) {
        return std::nullopt;
      }
      window = w;
    }
    return window;
  }

  // Records the sliding-window attention blocks over INT8 per-tensor caches
  // (the structure MatchInt8CacheAttentionBlock accepts) whose mask carries
  // a window. The lowering of the score product, the select and the value
  // product, or the attention plugin, then read the window's slices.
  struct WindowedReadTally {
    size_t reads = 0;
    int32_t window = 0, length = 0, tokens = 0, seq = 0;
  };

  Expected<void> FindWindowedCacheReads(const Subgraph& subgraph) {
    if (!WindowedCacheReadsEnabled() || !Fp16ActivationsEnabled() ||
        RuntimeBmmContextLimit() != 0) {
      return {};
    }
    WindowedReadTally tally;
    for (const auto& update : subgraph.Ops()) {
      if (update.Code() != kLiteRtOpCodeShloComposite ||
          CompositeOpName(update) != "odml.cache_update" ||
          update.Inputs().size() != 7 || update.Outputs().size() != 2) {
        continue;
      }
      const Tensor k_out = update.Outputs()[0];
      const Tensor v_out = update.Outputs()[1];
      if (!Int8CacheQuantization(k_out) || !Int8CacheQuantization(v_out)) {
        continue;
      }
      auto k_type = k_out.RankedTensorType();
      auto update_type = update.Inputs()[0].RankedTensorType();
      if (!k_type || !update_type || k_type->Layout().Rank() != 4 ||
          update_type->Layout().Rank() != 4) {
        continue;
      }
      const auto k_dims = k_type->Layout().Dimensions();
      const int32_t cache_len = static_cast<int32_t>(k_dims[2]);
      const int32_t chunk =
          static_cast<int32_t>(update_type->Layout().Dimensions()[2]);
      for (const auto& use : k_out.Uses()) {
        const Op scores_op = use.user;
        if (use.user_arg_ind != 1 || !IsRuntimeBmm(scores_op)) {
          continue;
        }
        LITERT_RETURN_IF_ERROR(
            MatchWindowedCacheRead(scores_op, k_out, v_out, cache_len, chunk,
                                   update.Inputs()[5], &tally));
      }
    }
    // Caches that are subgraph inputs: the MTP drafter attends over the base
    // model's caches and writes none of its own. The value cache is the value
    // product's second operand; the row position comes from the mask.
    for (const auto& k : subgraph.Inputs()) {
      auto k_type = k.RankedTensorType();
      if (!Int8CacheQuantization(k) || !k_type ||
          k_type->Layout().Rank() != 4) {
        continue;
      }
      const int32_t cache_len =
          static_cast<int32_t>(k_type->Layout().Dimensions()[2]);
      for (const auto& use : k.Uses()) {
        const Op scores_op = use.user;
        if (use.user_arg_ind != 1 || !IsRuntimeBmm(scores_op)) {
          continue;
        }
        const auto v = ValueCacheOfBlock(scores_op);
        if (!v || !v->IsSubgraphInput() || !Int8CacheQuantization(*v)) {
          continue;
        }
        LITERT_RETURN_IF_ERROR(MatchWindowedCacheRead(
            scores_op, k, *v, cache_len, /*chunk=*/0, std::nullopt, &tally));
      }
    }
    if (tally.reads != 0) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX windowing %zu attention blocks to %d of "
                 "%d keys (window %d, %d new tokens)",
                 tally.reads, tally.length, tally.seq, tally.window,
                 tally.tokens);
    }
    return {};
  }

  // The value cache of an attention block, from its score product:
  // scores -> select_v2 -> softmax -> values product, whose second operand it
  // is. Nullopt when the chain is not that.
  static std::optional<Tensor> ValueCacheOfBlock(const Op& scores_op) {
    const auto select_op = SingleUser(scores_op.Outputs()[0], 1);
    if (!select_op || select_op->Code() != kLiteRtOpCodeTflSelectV2 ||
        select_op->Inputs().size() != 3) {
      return std::nullopt;
    }
    const auto softmax_op = SingleUser(select_op->Outputs()[0], 0);
    if (!softmax_op || softmax_op->Code() != kLiteRtOpCodeTflSoftmax) {
      return std::nullopt;
    }
    const auto values_op = SingleUser(softmax_op->Outputs()[0], 0);
    if (!values_op || !IsRuntimeBmm(*values_op) ||
        values_op->Inputs().size() < 2) {
      return std::nullopt;
    }
    return values_op->Inputs()[1];
  }

  // One candidate block of FindWindowedCacheReads, from its score product.
  // `chunk` is the number of rows a cache update writes; 0 for a cache that
  // is a subgraph input, whose rows are those of the mask's position tensor.
  Expected<void> MatchWindowedCacheRead(const Op& scores_op,
                                        const Tensor& k_out,
                                        const Tensor& v_out, int32_t cache_len,
                                        int32_t chunk,
                                        std::optional<Tensor> start_indices,
                                        WindowedReadTally* tally) {
    const auto select_op = SingleUser(scores_op.Outputs()[0], 1);
    if (!select_op || select_op->Code() != kLiteRtOpCodeTflSelectV2 ||
        select_op->Inputs().size() != 3) {
      return {};
    }
    const auto softmax_op = SingleUser(select_op->Outputs()[0], 0);
    if (!softmax_op || softmax_op->Code() != kLiteRtOpCodeTflSoftmax) {
      return {};
    }
    const auto values_op = SingleUser(softmax_op->Outputs()[0], 0);
    if (!values_op || !IsRuntimeBmm(*values_op) ||
        values_op->Inputs()[1].Get() != v_out.Get()) {
      return {};
    }
    const Tensor mask = select_op->Inputs()[0];
    auto mask_type = mask.RankedTensorType();
    if (!mask_type || mask_type->Layout().Rank() != 4 ||
        mask_type->Layout().Dimensions()[3] != cache_len) {
      return {};
    }
    std::vector<WindowMaskPiece> pieces;
    const auto w = WindowFromMask(mask, cache_len, &pieces);
    if (!w.has_value() || pieces.empty()) {
      return {};
    }
    int64_t piece_rows = 0;
    int64_t most_rows = 0;
    for (const auto& piece : pieces) {
      piece_rows += piece.rows;
      most_rows = std::max(most_rows, piece.rows);
    }
    const int32_t tokens = chunk > 0 ? chunk : static_cast<int32_t>(most_rows);
    // [s0, s0 + span) with s0 = pos - window + 1 rounded down to 64 holds
    // every key a row of the chunk can see.
    const int32_t span = ((*w + tokens + 62 + 63) / 64) * 64;
    if (tokens <= 0 || span >= cache_len ||
        piece_rows != mask_type->Layout().Dimensions()[2]) {
      return {};
    }
    WindowedCacheRead read;
    read.k_out = k_out.Get();
    read.v_out = v_out.Get();
    read.mask = mask;
    read.start_indices = start_indices;
    read.pieces = std::move(pieces);
    read.mask_rows = static_cast<int32_t>(piece_rows);
    read.scores = scores_op.Get();
    read.select = select_op->Get();
    read.values = values_op->Get();
    read.seq = cache_len;
    read.tokens = tokens;
    read.window = *w;
    read.length = span;
    const size_t index = windowed_reads_.size();
    windowed_reads_.push_back(read);
    windowed_by_scores_[scores_op.Get()] = index;
    windowed_by_select_[select_op->Get()] = index;
    windowed_by_values_[values_op->Get()] = index;
    ++tally->reads;
    tally->window = *w;
    tally->length = span;
    tally->tokens = tokens;
    tally->seq = cache_len;
    return {};
  }

  // s0 = clamp(floor((pos - window + 1) / 64) * 64, 0, seq - length): a
  // multiple of 64 keeps the plugin's 16-byte value loads aligned, and the
  // span [s0, s0 + length) holds every key a row of the chunk can see. The
  // Myelin path gathers with execution-tensor indices s0 + iota (a slice with
  // a start tensor would make the position a shape input).
  Expected<void> EnsureWindowStart(WindowedCacheRead& read) {
    if (read.start != nullptr) {
      return {};
    }
    nvinfer1::ITensor* pos = nullptr;
    if (read.start_indices.has_value()) {
      LITERT_ASSIGN_OR_RETURN(auto* start_indices,
                              GetTensor(*read.start_indices));
      LITERT_ASSIGN_OR_RETURN(pos, SliceTensor1d(start_indices,
                                                 /*start_index=*/2,
                                                 /*size=*/1));
    } else {
      // A cache that is a subgraph input (the MTP drafter): the smallest row
      // position of the mask's position tensor [1, R, 1].
      if (read.pieces.empty() || !read.pieces[0].positions.has_value()) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Windowed cache read has no position tensor");
      }
      LITERT_ASSIGN_OR_RETURN(auto* positions,
                              GetTensor(*read.pieces[0].positions));
      auto* reduce = network_->addReduce(
          *positions, nvinfer1::ReduceOperation::kMIN,
          /*reduceAxes=*/(1u << 1) | (1u << 2), /*keepDimensions=*/false);
      if (reduce == nullptr || reduce->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to reduce the window's positions");
      }
      pos = reduce->getOutput(0);
    }
    const nvinfer1::Dims one{1, {1}};
    const int32_t back = read.window - 1;
    const int32_t zero = 0;
    const int32_t sixty_four = 64;
    const int32_t last = read.seq - read.length;
    LITERT_ASSIGN_OR_RETURN(
        auto* back_tensor,
        AddInt32Constant(absl::MakeConstSpan(&back, 1), one, "window_back"));
    LITERT_ASSIGN_OR_RETURN(
        auto* zero_tensor,
        AddInt32Constant(absl::MakeConstSpan(&zero, 1), one, "window_zero"));
    LITERT_ASSIGN_OR_RETURN(
        auto* align_tensor,
        AddInt32Constant(absl::MakeConstSpan(&sixty_four, 1), one,
                         "window_align"));
    LITERT_ASSIGN_OR_RETURN(
        auto* last_tensor,
        AddInt32Constant(absl::MakeConstSpan(&last, 1), one, "window_last"));
    nvinfer1::ITensor* value = pos;
    const struct {
      nvinfer1::ITensor* operand;
      nvinfer1::ElementWiseOperation op;
      const char* what;
    } steps[] = {
        {back_tensor, nvinfer1::ElementWiseOperation::kSUB, "shift"},
        {zero_tensor, nvinfer1::ElementWiseOperation::kMAX, "lower bound"},
        {align_tensor, nvinfer1::ElementWiseOperation::kFLOOR_DIV, "align"},
        {align_tensor, nvinfer1::ElementWiseOperation::kPROD, "align"},
        {last_tensor, nvinfer1::ElementWiseOperation::kMIN, "upper bound"},
    };
    for (const auto& step : steps) {
      auto* layer = network_->addElementWise(*value, *step.operand, step.op);
      if (layer == nullptr || layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add window start arithmetic");
      }
      value = layer->getOutput(0);
    }
    read.start = value;
    return {};
  }

  Expected<void> EnsureWindowIndices(WindowedCacheRead& read) {
    if (read.indices != nullptr) {
      return {};
    }
    LITERT_RETURN_IF_ERROR(EnsureWindowStart(read));
    std::vector<int32_t> iota(read.length);
    for (int32_t i = 0; i < read.length; ++i) {
      iota[i] = i;
    }
    nvinfer1::Dims iota_dims{};
    iota_dims.nbDims = 1;
    iota_dims.d[0] = read.length;
    LITERT_ASSIGN_OR_RETURN(auto* iota_tensor,
                            AddInt32Constant(iota, iota_dims, "window_iota"));
    auto* indices = network_->addElementWise(
        *iota_tensor, *read.start, nvinfer1::ElementWiseOperation::kSUM);
    if (indices == nullptr || indices->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add window index layer");
    }
    read.indices = indices->getOutput(0);
    return {};
  }

  // input[..., s0:s0 + length, ...] along `axis`. Bool tensors (the mask)
  // take an int32 round trip through the gather.
  Expected<nvinfer1::ITensor*> WindowGather(nvinfer1::ITensor* input,
                                            nvinfer1::ITensor* indices,
                                            int axis, const char* name) {
    const bool boolean = input->getType() == nvinfer1::DataType::kBOOL;
    if (boolean) {
      LITERT_ASSIGN_OR_RETURN(input,
                              AddCastTensor(input, nvinfer1::DataType::kINT32));
    }
    auto* layer = network_->addGather(*input, *indices, axis);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add window gather layer");
    }
    layer->setName(KeepName(UniqueName(name)));
    nvinfer1::ITensor* out = layer->getOutput(0);
    if (boolean) {
      LITERT_ASSIGN_OR_RETURN(out,
                              AddCastTensor(out, nvinfer1::DataType::kBOOL));
    }
    return out;
  }

  // The raw INT8 form of a cache: a cache update's output (scatter or cache
  // write) or a subgraph input (the MTP drafter's caches).
  nvinfer1::ITensor* RawInt8Cache(LiteRtTensor tensor) {
    for (const auto* map : {&raw_output_map_, &raw_input_map_}) {
      const auto raw = map->find(tensor);
      if (raw != map->end() &&
          raw->second->getType() == nvinfer1::DataType::kINT8) {
        return raw->second;
      }
    }
    return nullptr;
  }

  Expected<nvinfer1::ITensor*> WindowKeySlice(WindowedCacheRead& read) {
    if (read.k_slice != nullptr) {
      return read.k_slice;
    }
    LITERT_RETURN_IF_ERROR(EnsureWindowIndices(read));
    nvinfer1::ITensor* source = RawInt8Cache(read.k_out);
    if (source == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Windowed cache read has no raw INT8 key cache");
    }
    LITERT_ASSIGN_OR_RETURN(
        read.k_slice,
        WindowGather(source, read.indices, /*axis=*/2, "window_keys"));
    return read.k_slice;
  }

  Expected<nvinfer1::ITensor*> WindowValueSlice(WindowedCacheRead& read) {
    if (read.v_slice != nullptr) {
      return read.v_slice;
    }
    LITERT_RETURN_IF_ERROR(EnsureWindowIndices(read));
    nvinfer1::ITensor* source = RawInt8Cache(read.v_out);
    if (source == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Windowed cache read has no raw INT8 value cache");
    }
    LITERT_ASSIGN_OR_RETURN(
        read.v_slice,
        WindowGather(source, read.indices, /*axis=*/3, "window_values"));
    return read.v_slice;
  }

  // The window's mask, rebuilt the way the model builds the full one but
  // over the gathered key-index constants: for each piece
  // (pos >= lower[s0 + j]) && (pos < upper[s0 + j]), then the pieces
  // concatenated along axis 1 and reshaped to [1, 1, rows, length]. (A
  // gather of the bool mask would cast the whole [rows, S] tensor first.)
  Expected<nvinfer1::ITensor*> WindowMaskSlice(WindowedCacheRead& read) {
    if (read.mask_slice != nullptr) {
      return read.mask_slice;
    }
    LITERT_RETURN_IF_ERROR(EnsureWindowIndices(read));
    // Rank 3 like the gathered bounds [1, 1, length]: TensorRT's elementwise
    // layers broadcast only between tensors of equal rank.
    const nvinfer1::Dims one{3, {1, 1, 1}};
    const int32_t one_value = 1;
    LITERT_ASSIGN_OR_RETURN(auto* one_tensor,
                            AddInt32Constant(absl::MakeConstSpan(&one_value, 1),
                                             one, "window_one"));
    std::vector<nvinfer1::ITensor*> pieces;
    for (const auto& piece : read.pieces) {
      LITERT_ASSIGN_OR_RETURN(auto* positions, GetTensor(*piece.positions));
      LITERT_ASSIGN_OR_RETURN(auto* lower, GetTensor(*piece.lower));
      LITERT_ASSIGN_OR_RETURN(auto* upper, GetTensor(*piece.upper));
      LITERT_ASSIGN_OR_RETURN(
          lower, WindowGather(lower, read.indices, /*axis=*/2, "window_lower"));
      LITERT_ASSIGN_OR_RETURN(
          upper, WindowGather(upper, read.indices, /*axis=*/2, "window_upper"));
      // pos >= lower  <=>  pos > lower - 1
      auto* lower_minus = network_->addElementWise(
          *lower, *one_tensor, nvinfer1::ElementWiseOperation::kSUB);
      if (lower_minus == nullptr || lower_minus->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add window mask lower bound");
      }
      auto* at_least =
          network_->addElementWise(*positions, *lower_minus->getOutput(0),
                                   nvinfer1::ElementWiseOperation::kGREATER);
      auto* below = network_->addElementWise(
          *positions, *upper, nvinfer1::ElementWiseOperation::kLESS);
      if (at_least == nullptr || below == nullptr ||
          at_least->getOutput(0) == nullptr || below->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add window mask comparison");
      }
      auto* both = network_->addElementWise(
          *at_least->getOutput(0), *below->getOutput(0),
          nvinfer1::ElementWiseOperation::kAND);
      if (both == nullptr || both->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add window mask conjunction");
      }
      // [1, rows_i, length]: the pieces stack along axis 1 whatever their
      // row counts.
      LITERT_ASSIGN_OR_RETURN(
          auto* piece_mask,
          ReshapeTensor(both->getOutput(0),
                        nvinfer1::Dims3{1, static_cast<int64_t>(piece.rows),
                                        read.length}));
      pieces.push_back(piece_mask);
    }
    nvinfer1::ITensor* mask = pieces[0];
    if (pieces.size() > 1) {
      auto* concat = network_->addConcatenation(
          pieces.data(), static_cast<int32_t>(pieces.size()));
      if (concat == nullptr || concat->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add window mask concatenation");
      }
      concat->setAxis(1);
      mask = concat->getOutput(0);
    }
    LITERT_ASSIGN_OR_RETURN(
        read.mask_slice,
        ReshapeTensor(mask,
                      nvinfer1::Dims4{1, 1, read.mask_rows, read.length}));
    read.mask_slice->setName(KeepName(UniqueName("window_mask")));
    return read.mask_slice;
  }

  // A mask [1, 1, n, S] tiled over the query heads, as the model builds it:
  // reshape(concatenation(mask, ..., mask; axis 1)) = [1, 1, heads * n, S],
  // whose row r is row r % n of `mask`. Gemma 4's E-models tile the mask as
  // floats: reshape(not_equal(concatenation(cast(mask), ...; axis 1), 0)).
  struct TiledMask {
    Tensor mask;
    LiteRtOp reshape_op = nullptr;
    LiteRtOp concat_op = nullptr;
    // The float tiling's comparison and cast.
    LiteRtOp not_equal_op = nullptr;
    LiteRtOp cast_op = nullptr;
  };

  static std::optional<TiledMask> FindTiledMask(const Tensor& tiled,
                                                int32_t rows, int32_t seq) {
    auto reshape = tiled.GetDefiningOp();
    if (!reshape || reshape->Code() != kLiteRtOpCodeTflReshape ||
        reshape->Inputs().empty()) {
      return std::nullopt;
    }
    Tensor stacked = reshape->Inputs()[0];
    auto not_equal = stacked.GetDefiningOp();
    const bool floats =
        not_equal && not_equal->Code() == kLiteRtOpCodeTflNotEqual;
    if (floats) {
      if (not_equal->Inputs().size() != 2 ||
          ReadFloatScalar(not_equal->Inputs()[1]) != 0.0f) {
        return std::nullopt;
      }
      stacked = not_equal->Inputs()[0];
    }
    auto concat = stacked.GetDefiningOp();
    int32_t axis = 0;
    if (!concat || concat->Code() != kLiteRtOpCodeTflConcatenation ||
        concat->Inputs().empty() || concat->Outputs().size() != 1 ||
        LiteRtGetConcatenationAxisOption(concat->Get(), &axis) !=
            kLiteRtStatusOk ||
        (axis != 1 && axis != -3)) {
      return std::nullopt;
    }
    Tensor mask = concat->Inputs()[0];
    for (const auto& input : concat->Inputs()) {
      if (input.Get() != mask.Get()) {
        return std::nullopt;
      }
    }
    LiteRtOp cast_op = nullptr;
    if (floats) {
      auto cast = mask.GetDefiningOp();
      if (!cast || cast->Code() != kLiteRtOpCodeTflCast ||
          cast->Inputs().size() != 1) {
        return std::nullopt;
      }
      cast_op = cast->Get();
      mask = cast->Inputs()[0];
    }
    auto type = mask.RankedTensorType();
    if (!type || mask.ElementType() != litert::ElementType::Bool) {
      return std::nullopt;
    }
    const auto dims = type->Layout().Dimensions();
    const int64_t copies = static_cast<int64_t>(concat->Inputs().size());
    if (dims.size() != 4 || dims[0] != 1 || dims[1] != 1 || dims[2] <= 0 ||
        dims[3] != seq || copies * dims[2] != rows ||
        !TensorHasShape(concat->Outputs()[0],
                        {1, static_cast<int32_t>(copies), dims[2], seq})) {
      return std::nullopt;
    }
    return TiledMask{mask, reshape->Get(), concat->Get(),
                     floats ? not_equal->Get() : nullptr, cast_op};
  }

  // Leaves the ops of mask tilings unlowered when only fused attention blocks
  // read the tiled masks: from the reshape back to the cast, an op goes when
  // every reader of its result is fused or went before it.
  void DropUnreadMaskTilings(const Subgraph& subgraph,
                             const std::vector<TiledMask>& tiled_masks) {
    for (const auto& tiled : tiled_masks) {
      for (LiteRtOp op : {tiled.reshape_op, tiled.not_equal_op, tiled.concat_op,
                          tiled.cast_op}) {
        if (op == nullptr) {
          continue;
        }
        const Op tiling(subgraph.Context(), op);
        if (tiling.Outputs().size() != 1) {
          break;
        }
        const Tensor result = tiling.Outputs()[0];
        bool unread = graph_outputs_.count(result.Get()) == 0;
        for (const auto& use : result.Uses()) {
          unread = unread && (fused_attention_ops_.count(use.user.Get()) != 0 ||
                              unlowered_ops_.count(use.user.Get()) != 0);
        }
        if (!unread) {
          break;
        }
        unlowered_ops_.insert(op);
      }
    }
  }

  static bool IsRuntimeBmm(const Op& op) {
    return op.Code() == kLiteRtOpCodeShloComposite && op.Inputs().size() >= 2 &&
           op.Outputs().size() == 1 &&
           CompositeOpName(op) == "odml.runtime_bmm";
  }

  // The single user of `tensor`, if it reads it at `input_index`.
  static std::optional<Op> SingleUser(const Tensor& tensor,
                                      LiteRtParamIndex input_index) {
    const auto uses = tensor.Uses();
    if (uses.size() != 1 || uses[0].user_arg_ind != input_index) {
      return std::nullopt;
    }
    return uses[0].user;
  }

  // Matches, per in-place key cache update, the attention block
  //   scores = runtime_bmm(q [1,H,n,D], k_out [1,H,S,D])
  //   probs  = softmax(select_v2(mask [1,1,1|n,S], scores, fill))
  //   out    = runtime_bmm(probs, v_out)   with v_out a value cache held as
  //                                        [B, H, S, D]
  // and records it for the fused CUDA attention plugin: a single-head cache
  // of depth 512 with n a multiple of 16 for the tensor-core kernels (see
  // GlobalAttentionPluginEnabled), whose mask may be tiled over the query
  // heads, and n <= 16 with D in {128, 256, 512} for the experimental decode
  // kernel (see DecodeAttentionPluginEnabled).
  Expected<void> FindDecodeAttentionBlocks(const Subgraph& subgraph) {
    if ((!DecodeAttentionPluginEnabled() && !GlobalAttentionPluginEnabled()) ||
        !Fp16ActivationsEnabled() || RuntimeBmmContextLimit() != 0) {
      return {};
    }
    std::vector<TiledMask> tiled_masks;
    size_t num_global = 0;
    for (const auto& update : subgraph.Ops()) {
      // The key cache's in-place update: a dynamic update slice, or the
      // odml.cache_update composite lowered natively (Gemma 4 12B and 26B;
      // its key cache is input 3, its updated key cache output 0).
      const bool composite_update =
          update.Code() == kLiteRtOpCodeShloComposite &&
          update.Inputs().size() == 7 && update.Outputs().size() == 2 &&
          CompositeOpName(update) == "odml.cache_update";
      if (update.Code() != kLiteRtOpCodeTflDynamicUpdateSlice &&
          !composite_update) {
        continue;
      }
      const Tensor k_cache = update.Inputs()[composite_update ? 3 : 0];
      const auto info = in_place_caches_.find(k_cache.Get());
      if (info == in_place_caches_.end() || info->second.op != update.Get() ||
          info->second.partial_axis != 2) {
        continue;
      }
      const Tensor k_out = update.Outputs()[0];
      const auto scores_op = SingleUser(k_out, 1);
      if (!scores_op || !IsRuntimeBmm(*scores_op)) {
        continue;
      }
      const Tensor q = scores_op->Inputs()[0];
      const Tensor scores = scores_op->Outputs()[0];
      const auto select_op = SingleUser(scores, 1);
      if (!select_op || select_op->Code() != kLiteRtOpCodeTflSelectV2 ||
          select_op->Inputs().size() != 3) {
        continue;
      }
      Tensor mask = select_op->Inputs()[0];
      const auto fill = ReadFloatScalar(select_op->Inputs()[2]);
      if (!fill || mask.ElementType() != litert::ElementType::Bool) {
        continue;
      }
      const auto softmax_op = SingleUser(select_op->Outputs()[0], 0);
      float beta = 1.0f;
      if (!softmax_op || softmax_op->Code() != kLiteRtOpCodeTflSoftmax ||
          LiteRtGetSoftmaxBetaOption(softmax_op->Get(), &beta) !=
              kLiteRtStatusOk ||
          beta != 1.0f) {
        continue;
      }
      const auto values_op = SingleUser(softmax_op->Outputs()[0], 0);
      if (!values_op || !IsRuntimeBmm(*values_op)) {
        continue;
      }
      const Tensor v_out = values_op->Inputs()[1];
      const Tensor output = values_op->Outputs()[0];
      const auto value_info = in_place_outputs_.find(v_out.Get());
      if (value_info == in_place_outputs_.end() ||
          value_info->second.partial_axis != 3 ||
          !value_info->second.transposed) {
        continue;
      }
      auto q_type = q.RankedTensorType();
      auto k_type = k_out.RankedTensorType();
      auto scores_type = scores.RankedTensorType();
      auto output_type = output.RankedTensorType();
      if (!q_type || !k_type || !scores_type || !output_type) {
        continue;
      }
      const auto q_dims = q_type->Layout().Dimensions();
      const auto k_dims = k_type->Layout().Dimensions();
      const auto scores_dims = scores_type->Layout().Dimensions();
      const auto output_dims = output_type->Layout().Dimensions();
      if (q_dims.size() != 4 || k_dims.size() != 4 || scores_dims.size() != 4 ||
          output_dims.size() != 4) {
        continue;
      }
      const auto heads = q_dims[1];
      const auto rows = q_dims[2];
      const auto depth = q_dims[3];
      const auto seq = k_dims[2];
      if (q_dims[0] != 1 || rows < 1 || k_dims[0] != 1 || k_dims[1] != heads ||
          k_dims[3] != depth || scores_dims[0] != 1 ||
          scores_dims[1] != heads || scores_dims[2] != rows ||
          scores_dims[3] != seq || output_dims[0] != 1 ||
          output_dims[1] != heads || output_dims[2] != rows ||
          output_dims[3] != depth ||
          !TensorHasShape(select_op->Outputs()[0], {1, heads, rows, seq})) {
        continue;
      }
      // Prefer the mask the model tiled over the query heads: the tiled copy
      // is rows / n times its size and only feeds the select.
      std::optional<TiledMask> tiled;
      if (heads == 1 && GlobalAttentionPluginEnabled()) {
        tiled = FindTiledMask(mask, rows, seq);
        if (tiled.has_value()) {
          mask = tiled->mask;
        }
      }
      auto mask_type = mask.RankedTensorType();
      if (!mask_type) {
        continue;
      }
      const auto mask_dims = mask_type->Layout().Dimensions();
      if (mask_dims.size() != 4 || mask_dims[0] != 1 || mask_dims[1] != 1 ||
          mask_dims[3] != seq) {
        continue;
      }
      const auto mask_rows = mask_dims[2];
      const bool global =
          GlobalAttentionPluginEnabled() && heads == 1 &&
          *fill <= kMaxGlobalAttentionFill &&
          LiteRtNvidiaGlobalAttentionSupports(rows, mask_rows, depth);
      const bool decode = DecodeAttentionPluginEnabled() && rows <= 16 &&
                          (depth == 128 || depth == 256 || depth == 512) &&
                          (mask_rows == 1 || mask_rows == rows);
      if (!global && !decode) {
        continue;
      }
      DecodeAttentionBlock block;
      block.q = q.Get();
      block.k_out = k_out.Get();
      block.v_out = v_out.Get();
      block.mask = mask.Get();
      block.output = output;
      block.fill = *fill;
      const size_t index = attention_blocks_.size();
      attention_blocks_.push_back(block);
      for (LiteRtOp member : {scores_op->Get(), select_op->Get(),
                              softmax_op->Get(), values_op->Get()}) {
        fused_attention_ops_[member] = index;
      }
      if (tiled.has_value()) {
        tiled_masks.push_back(*tiled);
      }
      num_global += global ? 1 : 0;
    }
    // The tiling of a mask is not lowered when only fused selects read it.
    for (const auto& tiled : tiled_masks) {
      const Op reshape(subgraph.Context(), tiled.reshape_op);
      bool unused = reshape.Outputs().size() == 1 &&
                    graph_outputs_.count(reshape.Outputs()[0].Get()) == 0;
      for (const auto& use : reshape.Outputs()[0].Uses()) {
        unused = unused && fused_attention_ops_.count(use.user.Get()) != 0;
      }
      if (unused) {
        unlowered_ops_.insert(tiled.reshape_op);
      }
    }
    for (const auto& tiled : tiled_masks) {
      const Op concat(subgraph.Context(), tiled.concat_op);
      bool unused = graph_outputs_.count(concat.Outputs()[0].Get()) == 0;
      for (const auto& use : concat.Outputs()[0].Uses()) {
        unused = unused && unlowered_ops_.count(use.user.Get()) != 0;
      }
      if (unused) {
        unlowered_ops_.insert(tiled.concat_op);
      }
    }
    if (!attention_blocks_.empty()) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX fusing %zu attention blocks into the "
                 "CUDA attention plugin (%zu on the tensor-core kernels, %zu "
                 "mask tilings dropped)",
                 attention_blocks_.size(), num_global, unlowered_ops_.size());
    }
    return {};
  }

  // The op that computes `tensor`, if `tensor` has no other reader than the
  // op's consumer.
  static std::optional<Op> SingleUseProducer(const Tensor& tensor) {
    auto producer = tensor.GetDefiningOp();
    if (tensor.Uses().size() != 1 || !producer) {
      return std::nullopt;
    }
    return *producer;
  }

  // a @ b^T with b the rows of a cache or of the chunk: the runtime_bmm
  // composite or a batch matmul that transposes its right operand.
  static bool IsTransposedMatmul(const Op& op) {
    if (IsRuntimeBmm(op)) {
      return true;
    }
    bool adj_x = false;
    bool adj_y = false;
    return op.Code() == kLiteRtOpCodeTflBatchMatmul &&
           op.Inputs().size() == 2 && op.Outputs().size() == 1 &&
           LiteRtGetBatchMatmulAdjXOption(op.Get(), &adj_x) ==
               kLiteRtStatusOk &&
           LiteRtGetBatchMatmulAdjYOption(op.Get(), &adj_y) ==
               kLiteRtStatusOk &&
           !adj_x && adj_y;
  }

  // slice(tensor, {0, 0, 0, begin}) of shape [1, heads, rows, size]; the
  // static output shape determines the size operand.
  static bool IsLastAxisSlice(const Op& op, int32_t heads, int32_t rows,
                              int32_t begin, int32_t size) {
    if (op.Code() != kLiteRtOpCodeTflSlice || op.Inputs().size() != 3 ||
        op.Outputs().size() != 1) {
      return false;
    }
    const auto begins = ReadInt32Constant(op.Inputs()[1]);
    return begins && *begins == std::vector<int32_t>{0, 0, 0, begin} &&
           TensorHasShape(op.Outputs()[0], {1, heads, rows, size});
  }

  // Matches, per prefill update of a ring cache, the attention block that
  // reads the cache before the update and the chunk's own keys and values,
  //   scores = concatenation(q @ k_cache^T, q @ k_new^T)
  //   probs  = softmax(select_v2(mask [1,1,rows|tokens,S + n], scores, fill))
  //   out    = probs[..., :S] @ v_cache + probs[..., S:] @ v_new
  // with v_cache a value cache held as [B, H, S, D], and records it for the
  // fused CUDA attention plugin when the tensor-core kernels take its shape.
  // `out` is the attention join the cache update orders itself after (see
  // FindPrefillAttentionJoin).
  Expected<void> FindPrefillAttentionBlocks(const Subgraph& subgraph) {
    if (!LocalAttentionPluginEnabled() || !Fp16ActivationsEnabled() ||
        RuntimeBmmContextLimit() != 0 ||
        !LiteRtNvidiaTiledAttentionAvailable()) {
      return {};
    }
    size_t num_blocks = 0;
    for (const auto& update : subgraph.Ops()) {
      if (update.Code() != kLiteRtOpCodeShloComposite ||
          CompositeOpName(update) != "odml.cache_update" ||
          !PrefillCacheUpdateRingMode(update).has_value()) {
        continue;
      }
      const Tensor k_new = update.Inputs()[0];
      const Tensor v_new = update.Inputs()[1];
      const Tensor k_cache = update.Inputs()[3];
      const Tensor v_cache = update.Inputs()[4];
      const auto value_info = in_place_caches_.find(v_cache.Get());
      if (in_place_caches_.count(k_cache.Get()) == 0 ||
          value_info == in_place_caches_.end() ||
          value_info->second.op != update.Get() ||
          !value_info->second.transposed) {
        continue;
      }
      // The readers of the caches.
      const auto reader = [&](const Tensor& cache) -> std::optional<Op> {
        std::optional<Op> found;
        for (const auto& use : cache.Uses()) {
          if (use.user.Get() == update.Get()) continue;
          if (found.has_value() || use.user_arg_ind != 1 ||
              !IsTransposedMatmul(use.user)) {
            return std::nullopt;
          }
          found = use.user;
        }
        return found;
      };
      const auto cache_scores_op = reader(k_cache);
      const auto cache_values_op = reader(v_cache);
      if (!cache_scores_op || !cache_values_op) {
        continue;
      }
      const Tensor q = cache_scores_op->Inputs()[0];
      const Tensor output_cache = cache_values_op->Outputs()[0];
      auto q_type = q.RankedTensorType();
      auto k_type = k_cache.RankedTensorType();
      auto new_type = k_new.RankedTensorType();
      if (!q_type || !k_type || !new_type) {
        continue;
      }
      const auto q_dims = q_type->Layout().Dimensions();
      const auto k_dims = k_type->Layout().Dimensions();
      const auto new_dims = new_type->Layout().Dimensions();
      if (q_dims.size() != 4 || q_dims[0] != 1 || k_dims[1] != q_dims[1] ||
          k_dims[3] != q_dims[3]) {
        continue;
      }
      const int32_t heads = q_dims[1];
      const int32_t rows = q_dims[2];
      const int32_t depth = q_dims[3];
      const int32_t cache_len = k_dims[2];
      const int32_t new_len = new_dims[2];
      if (!TensorHasShape(cache_scores_op->Outputs()[0],
                          {1, heads, rows, cache_len}) ||
          !TensorHasShape(output_cache, {1, heads, rows, depth})) {
        continue;
      }
      // Scores: the cache and the chunk side by side.
      const auto concat_op = SingleUser(cache_scores_op->Outputs()[0], 0);
      int32_t axis = 0;
      if (!concat_op || concat_op->Code() != kLiteRtOpCodeTflConcatenation ||
          concat_op->Inputs().size() != 2 ||
          LiteRtGetConcatenationAxisOption(concat_op->Get(), &axis) !=
              kLiteRtStatusOk ||
          (axis != 3 && axis != -1) ||
          !TensorHasShape(concat_op->Outputs()[0],
                          {1, heads, rows, cache_len + new_len})) {
        continue;
      }
      const auto new_scores_op = SingleUseProducer(concat_op->Inputs()[1]);
      if (!new_scores_op || IsRuntimeBmm(*new_scores_op) ||
          !IsTransposedMatmul(*new_scores_op) ||
          new_scores_op->Inputs()[0].Get() != q.Get() ||
          new_scores_op->Inputs()[1].Get() != k_new.Get()) {
        continue;
      }
      const auto select_op = SingleUser(concat_op->Outputs()[0], 1);
      if (!select_op || select_op->Code() != kLiteRtOpCodeTflSelectV2 ||
          select_op->Inputs().size() != 3) {
        continue;
      }
      const Tensor mask = select_op->Inputs()[0];
      const auto fill = ReadFloatScalar(select_op->Inputs()[2]);
      auto mask_type = mask.RankedTensorType();
      if (!fill || *fill > kMaxGlobalAttentionFill || !mask_type ||
          mask.ElementType() != litert::ElementType::Bool) {
        continue;
      }
      const auto mask_dims = mask_type->Layout().Dimensions();
      if (mask_dims.size() != 4 || mask_dims[0] != 1 || mask_dims[1] != 1 ||
          mask_dims[3] != cache_len + new_len) {
        continue;
      }
      const auto softmax_op = SingleUser(select_op->Outputs()[0], 0);
      float beta = 1.0f;
      if (!softmax_op || softmax_op->Code() != kLiteRtOpCodeTflSoftmax ||
          LiteRtGetSoftmaxBetaOption(softmax_op->Get(), &beta) !=
              kLiteRtStatusOk ||
          beta != 1.0f) {
        continue;
      }
      // Values: the two halves of the probabilities against the cache and the
      // chunk, added up.
      const auto cache_probs_op =
          SingleUseProducer(cache_values_op->Inputs()[0]);
      const auto join_op = SingleUser(output_cache, 0);
      const auto join_other = SingleUser(output_cache, 1);
      const auto& add_op = join_op ? join_op : join_other;
      if (!cache_probs_op ||
          !IsLastAxisSlice(*cache_probs_op, heads, rows, 0, cache_len) ||
          !add_op || add_op->Code() != kLiteRtOpCodeTflAdd ||
          add_op->Inputs().size() != 2 || add_op->Outputs().size() != 1) {
        continue;
      }
      const auto new_values_op =
          SingleUseProducer(add_op->Inputs()[join_op ? 1 : 0]);
      bool adj_x = false;
      bool adj_y = false;
      if (!new_values_op ||
          new_values_op->Code() != kLiteRtOpCodeTflBatchMatmul ||
          new_values_op->Inputs().size() != 2 ||
          new_values_op->Inputs()[1].Get() != v_new.Get() ||
          LiteRtGetBatchMatmulAdjXOption(new_values_op->Get(), &adj_x) !=
              kLiteRtStatusOk ||
          LiteRtGetBatchMatmulAdjYOption(new_values_op->Get(), &adj_y) !=
              kLiteRtStatusOk ||
          adj_x || adj_y) {
        continue;
      }
      const auto new_probs_op = SingleUseProducer(new_values_op->Inputs()[0]);
      const Tensor probs = softmax_op->Outputs()[0];
      const Tensor output = add_op->Outputs()[0];
      if (!new_probs_op ||
          !IsLastAxisSlice(*new_probs_op, heads, rows, cache_len, new_len) ||
          cache_probs_op->Inputs()[0].Get() != probs.Get() ||
          new_probs_op->Inputs()[0].Get() != probs.Get() ||
          probs.Uses().size() != 2 ||
          graph_outputs_.count(output_cache.Get()) != 0 ||
          !TensorHasShape(output, {1, heads, rows, depth})) {
        continue;
      }
      // The cache update forwards the join (see LowerCompositeCacheUpdate);
      // without one it would order itself after results that are not
      // computed.
      const auto join = FindPrefillAttentionJoin(update);
      const LiteRtNvidiaAttentionShape shape = {heads, rows,      mask_dims[2],
                                                depth, cache_len, new_len};
      if (!join.has_value() || join->Get() != output.Get() ||
          !LiteRtNvidiaTiledAttentionSupports(&shape)) {
        continue;
      }
      DecodeAttentionBlock block;
      block.q = q.Get();
      block.k_out = k_cache.Get();
      block.v_out = v_cache.Get();
      block.mask = mask.Get();
      block.k_new = k_new.Get();
      block.v_new = v_new.Get();
      block.output = output;
      block.fill = *fill;
      const size_t index = attention_blocks_.size();
      attention_blocks_.push_back(block);
      for (LiteRtOp member :
           {cache_scores_op->Get(), new_scores_op->Get(), concat_op->Get(),
            select_op->Get(), softmax_op->Get(), cache_probs_op->Get(),
            new_probs_op->Get(), cache_values_op->Get(), new_values_op->Get(),
            add_op->Get()}) {
        fused_attention_ops_[member] = index;
      }
      ++num_blocks;
    }
    if (num_blocks != 0) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX fusing %zu prefill attention blocks over "
                 "ring caches into the CUDA attention plugin",
                 num_blocks);
    }
    return {};
  }

  // Emits the plugin for a block once all of its inputs are lowered (the
  // mask is usually computed after the score matmul in op order); later
  // members of the same block are no-ops.
  // Per-tensor INT8 quantization of a cache tensor, if it has one.
  static std::optional<std::pair<float, int32_t>> Int8CacheQuantization(
      const Tensor& tensor) {
    if (tensor.ElementType() != litert::ElementType::Int8 ||
        tensor.QTypeId() != kLiteRtQuantizationPerTensor) {
      return std::nullopt;
    }
    const auto q = tensor.PerTensorQuantization();
    if (!std::isfinite(q.scale) || q.scale <= 0.0f || q.zero_point < -128 ||
        q.zero_point > 127) {
      return std::nullopt;
    }
    return std::make_pair(q.scale, static_cast<int32_t>(q.zero_point));
  }

  // The INT8 blocks of many query rows that go to the tensor-core kernels,
  // and the mask tilings they read past.
  struct Int8AttentionTally {
    size_t tensor_core = 0;
    std::vector<TiledMask> tiled_masks;
  };

  // Attention blocks over INT8 per-tensor KV caches (Gemma 4 E2B and E4B):
  //   k_out, v_out = odml.cache_update(k, v, positions, k_cache, v_cache, ...)
  //   scores = runtime_bmm(q [1,H,n,D], k_out [1,H,S,D])
  //   probs  = softmax(select(mask [1,1,n|1,S], scores, fill))
  //   out    = runtime_bmm(probs, v_out [1,H,D,S])
  // The caches are updated by the quantized scatter lowering (not in place);
  // the plugin reads the raw INT8 results and dequantizes them, which spares
  // Myelin's whole-cache dequantize and its fused attention kernel.
  Expected<void> FindInt8CacheAttentionBlocks(const Subgraph& subgraph) {
    if (!Int8CacheAttentionPluginEnabled() || !Fp16ActivationsEnabled() ||
        RuntimeBmmContextLimit() != 0) {
      return {};
    }
    size_t num_blocks = 0;
    Int8AttentionTally tally;
    for (const auto& update : subgraph.Ops()) {
      if (update.Code() != kLiteRtOpCodeShloComposite ||
          CompositeOpName(update) != "odml.cache_update" ||
          update.Inputs().size() != 7 || update.Outputs().size() != 2) {
        continue;
      }
      const Tensor k_out = update.Outputs()[0];
      const Tensor v_out = update.Outputs()[1];
      const auto k_quant = Int8CacheQuantization(k_out);
      const auto v_quant = Int8CacheQuantization(v_out);
      if (!k_quant || !v_quant) {
        continue;
      }
      // Every layer that attends over this cache: the layer that wrote it and
      // the layers that share its KV cache.
      for (const auto& use : k_out.Uses()) {
        const Op scores_op = use.user;
        if (use.user_arg_ind != 1 || !IsRuntimeBmm(scores_op)) {
          continue;
        }
        LITERT_ASSIGN_OR_RETURN(
            const bool matched,
            MatchInt8CacheAttentionBlock(scores_op, k_out, v_out, *k_quant,
                                         *v_quant, &tally));
        num_blocks += matched ? 1 : 0;
      }
    }
    // Caches that are subgraph inputs (the MTP drafter attends over the base
    // model's caches): the value cache is the value product's second operand.
    for (const auto& k : subgraph.Inputs()) {
      const auto k_quant = Int8CacheQuantization(k);
      auto k_type = k.RankedTensorType();
      if (!k_quant || !k_type || k_type->Layout().Rank() != 4) {
        continue;
      }
      for (const auto& use : k.Uses()) {
        const Op scores_op = use.user;
        if (use.user_arg_ind != 1 || !IsRuntimeBmm(scores_op)) {
          continue;
        }
        const auto v = ValueCacheOfBlock(scores_op);
        if (!v || !v->IsSubgraphInput()) {
          continue;
        }
        const auto v_quant = Int8CacheQuantization(*v);
        if (!v_quant) {
          continue;
        }
        LITERT_ASSIGN_OR_RETURN(
            const bool matched,
            MatchInt8CacheAttentionBlock(scores_op, k, *v, *k_quant, *v_quant,
                                         &tally));
        num_blocks += matched ? 1 : 0;
      }
    }
    DropUnreadMaskTilings(subgraph, tally.tiled_masks);
    if (num_blocks != 0) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX fusing %zu attention blocks over INT8 "
                 "caches into the CUDA attention plugin (%zu on the "
                 "tensor-core kernels; last block: heads=%d rows=%d depth=%d "
                 "seq=%d)",
                 num_blocks, tally.tensor_core, int8_attention_geometry_[0],
                 int8_attention_geometry_[1], int8_attention_geometry_[2],
                 int8_attention_geometry_[3]);
    }
    return {};
  }

  // One block of FindInt8CacheAttentionBlocks, starting at its score product.
  Expected<bool> MatchInt8CacheAttentionBlock(const Op& scores_op,
                                              const Tensor& k_out,
                                              const Tensor& v_out,
                                              std::pair<float, int32_t> k_quant,
                                              std::pair<float, int32_t> v_quant,
                                              Int8AttentionTally* tally) {
    const Tensor q = scores_op.Inputs()[0];
    const Tensor scores = scores_op.Outputs()[0];
    const auto select_op = SingleUser(scores, 1);
    if (!select_op || select_op->Code() != kLiteRtOpCodeTflSelectV2 ||
        select_op->Inputs().size() != 3) {
      return false;
    }
    const Tensor mask = select_op->Inputs()[0];
    const auto fill = ReadFloatScalar(select_op->Inputs()[2]);
    if (!fill || mask.ElementType() != litert::ElementType::Bool) {
      return false;
    }
    const auto softmax_op = SingleUser(select_op->Outputs()[0], 0);
    float beta = 1.0f;
    if (!softmax_op || softmax_op->Code() != kLiteRtOpCodeTflSoftmax ||
        LiteRtGetSoftmaxBetaOption(softmax_op->Get(), &beta) !=
            kLiteRtStatusOk ||
        beta != 1.0f) {
      return false;
    }
    const auto values_op = SingleUser(softmax_op->Outputs()[0], 0);
    if (!values_op || !IsRuntimeBmm(*values_op) ||
        values_op->Inputs()[1].Get() != v_out.Get()) {
      return false;
    }
    const Tensor output = values_op->Outputs()[0];
    auto q_type = q.RankedTensorType();
    auto k_type = k_out.RankedTensorType();
    auto v_type = v_out.RankedTensorType();
    auto scores_type = scores.RankedTensorType();
    auto output_type = output.RankedTensorType();
    auto mask_type = mask.RankedTensorType();
    if (!q_type || !k_type || !v_type || !scores_type || !output_type ||
        !mask_type) {
      return false;
    }
    const auto q_dims = q_type->Layout().Dimensions();
    const auto k_dims = k_type->Layout().Dimensions();
    const auto v_dims = v_type->Layout().Dimensions();
    const auto scores_dims = scores_type->Layout().Dimensions();
    const auto output_dims = output_type->Layout().Dimensions();
    const auto mask_dims = mask_type->Layout().Dimensions();
    if (q_dims.size() != 4 || k_dims.size() != 4 || v_dims.size() != 4 ||
        scores_dims.size() != 4 || output_dims.size() != 4 ||
        mask_dims.size() != 4) {
      return false;
    }
    const auto heads = q_dims[1];
    const auto rows = q_dims[2];
    const auto depth = q_dims[3];
    const auto seq = k_dims[2];
    // A single KV head (Gemma 4 E2B: one head, eight query rows) over a short
    // cache is an ordinary matmul chain that TensorRT lowers to its FC and
    // softmax kernels in a few microseconds; the plugin's one block per head
    // and chunk cannot match that (282-290 against 317 steps/s on E2B at a
    // 2,048-token capacity, 255 against 270 at 9,216). Several KV heads (E4B:
    // two heads, four rows) go to Myelin's fused decode MHA kernel, 180 us per
    // layer, where the plugin doubles the decode rate; and over a long cache
    // TensorRT's lowering reads every key at a fraction of the bandwidth (0.4-1
    // ms per layer at 30k) while the plugin skips the chunks a sliding window
    // hides.
    const auto windowed = windowed_by_scores_.find(scores_op.Get());
    const int64_t keys_read = windowed != windowed_by_scores_.end()
                                  ? windowed_reads_[windowed->second].length
                                  : seq;
    // The many query rows of a prefill chunk over a global layer's caches,
    // which hold the chunk already, run on the tiled tensor-core kernel
    // (trtllm/tiled_attention.h), given the mask from before the model tiled
    // it over the query heads. The kernel reads the quantized caches and only
    // the keys some row can see, where TensorRT's lowering dequantizes the
    // whole cache and keeps [rows, context] scores, a select and the tiled
    // mask for every chunk whatever the prompt holds: E2B's prefill at a
    // 31,744-token capacity spent 13.7 ms per block and chunk there and
    // 1.2 GiB of activations.
    std::optional<TiledMask> tiled;
    Tensor block_mask = mask;
    bool tensor_core = false;
    if (rows > 16 && windowed == windowed_by_scores_.end() &&
        GlobalAttentionPluginEnabled() && *fill <= kMaxGlobalAttentionFill) {
      tiled = FindTiledMask(mask, rows, seq);
      if (tiled.has_value()) {
        block_mask = tiled->mask;
      }
      auto block_mask_type = block_mask.RankedTensorType();
      if (!block_mask_type) {
        return false;
      }
      const auto block_mask_dims = block_mask_type->Layout().Dimensions();
      const LiteRtNvidiaAttentionShape shape = {
          static_cast<int32_t>(heads),
          static_cast<int32_t>(rows),
          block_mask_dims.size() == 4 ? block_mask_dims[2] : 0,
          static_cast<int32_t>(depth),
          static_cast<int32_t>(seq),
          /*new_len=*/0};
      tensor_core = block_mask_dims.size() == 4 && block_mask_dims[0] == 1 &&
                    block_mask_dims[1] == 1 && block_mask_dims[3] == seq &&
                    LiteRtNvidiaTiledAttentionInt8Supports(&shape) &&
                    LiteRtNvidiaTiledAttentionAvailable();
    }
    if (!tensor_core && heads < 2 && keys_read <= kSingleHeadNativeCacheLimit) {
      return false;
    }
    if (q_dims[0] != 1 || rows < 1 || (rows > 16 && !tensor_core) ||
        k_dims[0] != 1 || k_dims[1] != heads || k_dims[3] != depth ||
        v_dims[0] != 1 || v_dims[1] != heads || v_dims[2] != depth ||
        v_dims[3] != seq || scores_dims[0] != 1 || scores_dims[1] != heads ||
        scores_dims[2] != rows || scores_dims[3] != seq ||
        output_dims[0] != 1 || output_dims[1] != heads ||
        output_dims[2] != rows || output_dims[3] != depth ||
        mask_dims[0] != 1 || mask_dims[1] != 1 ||
        (mask_dims[2] != 1 && mask_dims[2] != rows && !tensor_core) ||
        mask_dims[3] != seq ||
        !TensorHasShape(select_op->Outputs()[0], {1, heads, rows, seq}) ||
        !LiteRtNvidiaDecodeAttentionInt8Supports(static_cast<int32_t>(depth))) {
      return false;
    }
    int8_attention_geometry_ = {static_cast<int>(heads), static_cast<int>(rows),
                                static_cast<int>(depth), static_cast<int>(seq)};
    DecodeAttentionBlock block;
    block.q = q.Get();
    block.k_out = k_out.Get();
    block.v_out = v_out.Get();
    block.mask = mask.Get();
    if (tensor_core) {
      block.mask = block_mask.Get();
      ++tally->tensor_core;
      if (tiled.has_value()) {
        tally->tiled_masks.push_back(*tiled);
      }
    }
    block.output = output;
    block.fill = *fill;
    block.int8_cache = LiteRtNvidiaInt8CacheParams{
        k_quant.first, k_quant.second, v_quant.first, v_quant.second,
        /*value_columns=*/true};
    if (windowed != windowed_by_scores_.end()) {
      block.window = windowed->second;
    }
    const size_t index = attention_blocks_.size();
    attention_blocks_.push_back(block);
    for (LiteRtOp member : {scores_op.Get(), select_op->Get(),
                            softmax_op->Get(), values_op->Get()}) {
      fused_attention_ops_[member] = index;
    }
    return true;
  }

  // The row plan of an attention block: query rows = group * tokens. Up to
  // 16 rows run the decode kernel in one launch. A block of a few tokens
  // whose heads bring the rows past 16 (Gemma 4 26B's verify step: 8 heads x
  // 5 tokens per KV head) runs the decode kernel once per chunk of tokens
  // that fits, over the same caches: the tensor-core kernel would parallelize
  // over its three row tiles only and crawl through a long cache. Many tokens
  // (prefill) run the tensor-core kernel, the tokens padded when the rows
  // are not a multiple of its tile.
  //
  // A chunk of 16 rows at depth 512 lands on the tensor-core kernel, and
  // every such launch reads the whole cache (the 26B's verify step read its
  // global caches three times, 0.6 GB each at 30k tokens). The chunks that
  // fill 16 rows therefore go as one launch that computes each of them as
  // its own launch did and reads the caches once per two of them
  // (LiteRtNvidiaLaunchGlobalAttentionTokens); a last partial chunk keeps
  // its launch of the decode kernel.
  struct AttentionRowPlan {
    int32_t chunk_tokens = 0;   // decode kernel per chunk of this many tokens
    int32_t padded_tokens = 0;  // tensor-core kernel with the tokens padded
    // The leading tokens whose chunks share one launch (0: none do).
    int32_t shared_tokens = 0;
  };

  static std::optional<AttentionRowPlan> PlanAttentionRows(
      int32_t group, int32_t tokens, int32_t depth, int32_t heads,
      int32_t mask_rows, int32_t cache_len) {
    if (group < 1 || tokens < 1) {
      return std::nullopt;
    }
    const int32_t rows = group * tokens;
    if (rows <= 16) {
      return AttentionRowPlan{tokens, tokens};
    }
    if (tokens <= 16) {
      AttentionRowPlan plan{std::max<int32_t>(1, 16 / group), tokens};
      const int32_t full = tokens / plan.chunk_tokens * plan.chunk_tokens;
      if (full > plan.chunk_tokens &&
          LiteRtNvidiaGlobalAttentionTokensSupports(
              group * full, full, mask_rows == tokens ? full : mask_rows,
              depth)) {
        plan.shared_tokens = full;
      }
      return plan;
    }
    const auto padded = SdpaTransposedPaddedTokens(group, tokens, depth);
    if (!padded.has_value()) {
      if (SdpaTransposedTiledCache(heads, rows, mask_rows, depth, cache_len)) {
        return AttentionRowPlan{0, tokens};
      }
      return std::nullopt;
    }
    return AttentionRowPlan{0, *padded};
  }

  // `input` [.., n, ..] sliced to `size` elements from `start` along `axis`.
  Expected<nvinfer1::ITensor*> SliceTensorAxis(nvinfer1::ITensor* input,
                                               int axis, int32_t start,
                                               int32_t size) {
    const nvinfer1::Dims dims = input->getDimensions();
    nvinfer1::Dims begin{};
    nvinfer1::Dims out = dims;
    nvinfer1::Dims stride{};
    begin.nbDims = stride.nbDims = dims.nbDims;
    for (int i = 0; i < dims.nbDims; ++i) {
      begin.d[i] = 0;
      stride.d[i] = 1;
    }
    begin.d[axis] = start;
    out.d[axis] = size;
    auto* layer = network_->addSlice(*input, begin, out, stride);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT slice layer");
    }
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> ConcatTensors(
      std::vector<nvinfer1::ITensor*> parts, int axis, const char* what) {
    if (parts.size() == 1) {
      return parts[0];
    }
    auto* concat = network_->addConcatenation(parts.data(), parts.size());
    if (concat == nullptr || concat->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   std::string("Failed to concatenate ") + what);
    }
    concat->setAxis(axis);
    return concat->getOutput(0);
  }

  // The bool mask [1, 1, 1 or tokens, keys] of an odml.sdpa_transposed
  // without one, from its params [1, 1, 1, 7] as ML Drift's kernels read them
  // (sdpa_transposed_kernel.cc, GenerateImplicitCausalBoundCode): the keys
  // below the active count params[2] (all keys when it is out of (0, keys]),
  // and with is_causal over several tokens, for token x, below
  // min(active, q_start + x + 1), q_start = params[0] when in (0, active),
  // else active - tokens; a token at or past the active count sees no key.
  Expected<nvinfer1::ITensor*> MaskFromSdpaParams(nvinfer1::ITensor* params,
                                                  int32_t tokens, int32_t keys,
                                                  bool causal) {
    if (params->getType() != nvinfer1::DataType::kINT32) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "odml.sdpa_transposed params must be int32");
    }
    const auto constant = [&](const std::vector<int32_t>& values,
                              nvinfer1::Dims dims,
                              const char* name) -> nvinfer1::ITensor* {
      owned_weights_.emplace_back(values.size() * sizeof(int32_t));
      std::memcpy(owned_weights_.back().data(), values.data(),
                  owned_weights_.back().size());
      nvinfer1::Weights weights{nvinfer1::DataType::kINT32,
                                owned_weights_.back().data(),
                                static_cast<int64_t>(values.size())};
      auto* layer = network_->addConstant(dims, weights);
      if (layer == nullptr) return nullptr;
      layer->getOutput(0)->setName(KeepName(UniqueName(name)));
      return layer->getOutput(0);
    };
    const nvinfer1::Dims one{4, {1, 1, 1, 1}};
    const auto binary =
        [&](nvinfer1::ITensor* a, nvinfer1::ITensor* b,
            nvinfer1::ElementWiseOperation operation) -> nvinfer1::ITensor* {
      if (a == nullptr || b == nullptr) return nullptr;
      auto* layer = network_->addElementWise(*a, *b, operation);
      return layer == nullptr ? nullptr : layer->getOutput(0);
    };
    const auto select = [&](nvinfer1::ITensor* condition, nvinfer1::ITensor* a,
                            nvinfer1::ITensor* b) -> nvinfer1::ITensor* {
      if (condition == nullptr || a == nullptr || b == nullptr) return nullptr;
      auto* layer = network_->addSelect(*condition, *a, *b);
      return layer == nullptr ? nullptr : layer->getOutput(0);
    };
    using Op = nvinfer1::ElementWiseOperation;
    LITERT_ASSIGN_OR_RETURN(auto* q_start,
                            SliceTensorStatic(params, 3, 0, 1, "sdpa_q_start"));
    LITERT_ASSIGN_OR_RETURN(auto* count,
                            SliceTensorStatic(params, 3, 2, 1, "sdpa_active"));
    nvinfer1::ITensor* zero = constant({0}, one, "sdpa_zero");
    nvinfer1::ITensor* all = constant({keys}, one, "sdpa_keys");
    nvinfer1::ITensor* active =
        select(binary(binary(count, zero, Op::kGREATER),
                      binary(all, count, Op::kLESS), Op::kXOR),
               count, all);
    // (count > 0) xor (keys < count): in range exactly when count > 0 and
    // count <= keys, since keys < count implies count > 0.
    std::vector<int32_t> iota(keys);
    for (int32_t j = 0; j < keys; ++j) iota[j] = j;
    nvinfer1::ITensor* key = constant(iota, {4, {1, 1, 1, keys}}, "sdpa_key");
    nvinfer1::ITensor* bound = active;
    if (causal && tokens > 1) {
      nvinfer1::ITensor* fallback =
          binary(active, constant({tokens}, one, "sdpa_tokens"), Op::kSUB);
      nvinfer1::ITensor* start =
          select(binary(binary(q_start, zero, Op::kGREATER),
                        binary(q_start, active, Op::kLESS), Op::kAND),
                 q_start, fallback);
      std::vector<int32_t> rows(tokens);
      for (int32_t x = 0; x < tokens; ++x) rows[x] = x;
      nvinfer1::ITensor* position = binary(
          start, constant(rows, {4, {1, 1, tokens, 1}}, "sdpa_row"), Op::kSUM);
      nvinfer1::ITensor* causal_bound = binary(
          active, binary(position, constant({1}, one, "sdpa_one"), Op::kSUM),
          Op::kMIN);
      bound = select(binary(position, active, Op::kLESS), causal_bound, zero);
    }
    nvinfer1::ITensor* mask = binary(key, bound, Op::kLESS);
    if (mask == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to build the odml.sdpa_transposed mask from its "
                   "params");
    }
    mask->setName(KeepName(UniqueName("sdpa_params_mask")));
    return mask;
  }

  Expected<void> LowerDecodeAttentionMember(size_t index) {
    auto& block = attention_blocks_[index];
    if (block.emitted) {
      return {};
    }
    const bool new_keys = block.k_new != nullptr;
    for (LiteRtTensor input : {block.q, block.k_out, block.v_out, block.mask,
                               block.params, block.k_new, block.v_new}) {
      if (input != nullptr && tensor_map_.find(input) == tensor_map_.end()) {
        return {};
      }
    }
    nvinfer1::ITensor* q = tensor_map_[block.q];
    if (q->getType() != nvinfer1::DataType::kHALF &&
        q->getType() != nvinfer1::DataType::kBF16) {
      LITERT_ASSIGN_OR_RETURN(q, AddCastTensor(q, ModeFloatType()));
    }
    nvinfer1::ITensor* k_cache = tensor_map_[block.k_out];
    nvinfer1::ITensor* v_cache = tensor_map_[block.v_out];
    if (block.int8_cache.has_value()) {
      // The quantized scatter lowering records the raw INT8 caches.
      k_cache = RawInt8Cache(block.k_out);
      v_cache = RawInt8Cache(block.v_out);
      if (k_cache == nullptr || v_cache == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "INT8 cache attention block has no raw INT8 caches");
      }
    }
    const nvinfer1::Dims q_dims_in = q->getDimensions();
    const int32_t tokens = static_cast<int32_t>(q_dims_in.d[2]);
    const int32_t depth = static_cast<int32_t>(q_dims_in.d[3]);
    nvinfer1::ITensor* mask_in = nullptr;
    if (block.mask != nullptr) {
      mask_in = tensor_map_[block.mask];
    } else {
      LITERT_ASSIGN_OR_RETURN(
          mask_in, MaskFromSdpaParams(
                       tensor_map_[block.params], tokens,
                       static_cast<int32_t>(k_cache->getDimensions().d[2]),
                       block.causal));
    }
    // A windowed block keeps the full caches and mask: the plugin reads
    // `window` keys from the start it is handed as a fifth input.
    int32_t window = 0;
    nvinfer1::ITensor* window_start = nullptr;
    if (block.window.has_value()) {
      auto& read = windowed_reads_[*block.window];
      LITERT_RETURN_IF_ERROR(EnsureWindowStart(read));
      window = read.length;
      window_start = read.start;
    }

    // One plugin launch over `t` tokens: q [1, Hq, t, D] and a mask of 1 or
    // t rows; the group's query heads become rows of their KV head
    // ([1, Hq, t, D] -> [1, H, group * t, D], head-major within the group)
    // and the result comes back as [1, Hq, t, D]. `shared`: the tokens are
    // chunks that share the launch (AttentionRowPlan).
    const auto launch = [&](nvinfer1::ITensor* q_part,
                            nvinfer1::ITensor* mask_part, int32_t t,
                            bool shared) -> Expected<nvinfer1::ITensor*> {
      nvinfer1::ITensor* grouped_q = q_part;
      if (block.group > 1) {
        nvinfer1::Dims grouped = q_dims_in;
        grouped.d[1] = q_dims_in.d[1] / block.group;
        grouped.d[2] = t * block.group;
        LITERT_ASSIGN_OR_RETURN(grouped_q, ReshapeTensor(q_part, grouped));
      }
      nvinfer1::ITensor* mask = mask_part;
      if (block.group > 1 && t > 1 && mask->getDimensions().d[2] == t &&
          t * block.group <= 16) {
        // The decode kernel wants one mask row per query row: repeat the t
        // rows for each head of the group, in the rows' head-major order.
        std::vector<nvinfer1::ITensor*> copies(block.group, mask);
        LITERT_ASSIGN_OR_RETURN(
            mask, ConcatTensors(std::move(copies), 2, "the attention mask"));
      }
      TrtPtr<nvinfer1::IPluginV3> plugin(CreateDecodeAttentionPlugin(
          block.fill, new_keys,
          block.int8_cache.has_value() ? &*block.int8_cache : nullptr, window,
          /*row_tokens=*/shared ? t : 0));
      if (!plugin) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to create CUDA decode attention plugin");
      }
      std::vector<nvinfer1::ITensor*> inputs = {grouped_q, k_cache, v_cache,
                                                mask};
      if (window_start != nullptr) {
        inputs.push_back(window_start);
      }
      if (new_keys) {
        for (LiteRtTensor fresh : {block.k_new, block.v_new}) {
          LITERT_ASSIGN_OR_RETURN(
              auto* rows,
              AddCastTensor(tensor_map_[fresh], nvinfer1::DataType::kHALF));
          inputs.push_back(rows);
        }
      }
      auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(*network_, inputs.data(),
                                                       inputs.size(), *plugin);
      if (layer == nullptr || layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add CUDA decode attention plugin layer");
      }
      const char* name =
          new_keys ? "cuda_prefill_attention" : "cuda_decode_attention";
      layer->setName(KeepName(UniqueName(name)));
      auto* out = layer->getOutput(0);
      out->setName(KeepName(UniqueName(name)));
      owned_plugins_.push_back(std::move(plugin));
      if (block.group > 1) {
        nvinfer1::Dims ungrouped = q_dims_in;
        ungrouped.d[2] = t;
        LITERT_ASSIGN_OR_RETURN(out, ReshapeTensor(out, ungrouped));
      }
      return out;
    };

    nvinfer1::ITensor* out = nullptr;
    const bool plain_block =
        !new_keys && !block.int8_cache.has_value() && !block.window.has_value();
    std::optional<AttentionRowPlan> plan;
    if (plain_block && block.group * tokens > 16) {
      plan = PlanAttentionRows(
          block.group, tokens, depth,
          static_cast<int32_t>(k_cache->getDimensions().d[1]),
          static_cast<int32_t>(mask_in->getDimensions().d[2]),
          static_cast<int32_t>(k_cache->getDimensions().d[2]));
      if (!plan.has_value()) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Attention block rows fit no CUDA attention kernel");
      }
    }
    if (plan.has_value() && plan->chunk_tokens > 0) {
      // Chunks of tokens on the decode kernel, over the same caches; the
      // chunks that share a launch first.
      std::vector<nvinfer1::ITensor*> parts;
      for (int32_t first = 0, t = 0; first < tokens; first += t) {
        const bool shared = first < plan->shared_tokens;
        t = shared ? plan->shared_tokens
                   : std::min(plan->chunk_tokens, tokens - first);
        LITERT_ASSIGN_OR_RETURN(auto* q_part, SliceTensorAxis(q, 2, first, t));
        nvinfer1::ITensor* mask_part = mask_in;
        if (mask_in->getDimensions().d[2] == tokens) {
          LITERT_ASSIGN_OR_RETURN(mask_part,
                                  SliceTensorAxis(mask_in, 2, first, t));
        }
        LITERT_ASSIGN_OR_RETURN(auto* part,
                                launch(q_part, mask_part, t, shared));
        parts.push_back(part);
        if (shared) {
          LITERT_LOG(LITERT_INFO,
                     "NVIDIA TensorRT-RTX attention block: %d of %d tokens "
                     "of %d query heads share the cache reads of one launch",
                     t, tokens, block.group);
        }
      }
      LITERT_ASSIGN_OR_RETURN(
          out, ConcatTensors(std::move(parts), 2, "the attention chunks"));
    } else if (plan.has_value() && plan->padded_tokens != tokens) {
      // The tensor-core kernel with the tokens padded: zero query rows and
      // all-false mask rows (the kernel takes mask row r % mask rows for
      // query row r), sliced off the output.
      const int32_t padded_tokens = plan->padded_tokens;
      nvinfer1::Dims pad_dims = q_dims_in;
      pad_dims.d[2] = padded_tokens - tokens;
      const std::vector<float> zeros(static_cast<size_t>(pad_dims.d[0]) *
                                         pad_dims.d[1] * pad_dims.d[2] *
                                         pad_dims.d[3],
                                     0.0f);
      LITERT_ASSIGN_OR_RETURN(
          auto* q_pad,
          AddFloatConstant(zeros, pad_dims, "attention_q_pad", q->getType()));
      LITERT_ASSIGN_OR_RETURN(
          auto* padded_q,
          ConcatTensors({q, q_pad}, 2, "the padded attention query rows"));
      nvinfer1::ITensor* padded_mask = mask_in;
      if (mask_in->getDimensions().d[2] == tokens) {
        nvinfer1::Dims mask_pad_dims = mask_in->getDimensions();
        mask_pad_dims.d[2] = padded_tokens - tokens;
        LITERT_ASSIGN_OR_RETURN(
            auto* mask_pad,
            AddFalseConstant(mask_pad_dims, "attention_mask_pad"));
        LITERT_ASSIGN_OR_RETURN(
            padded_mask, ConcatTensors({mask_in, mask_pad}, 2,
                                       "the padded attention mask rows"));
      }
      LITERT_ASSIGN_OR_RETURN(
          auto* padded_out,
          launch(padded_q, padded_mask, padded_tokens, /*shared=*/false));
      LITERT_ASSIGN_OR_RETURN(out, SliceTensorAxis(padded_out, 2, 0, tokens));
    } else {
      LITERT_ASSIGN_OR_RETURN(out,
                              launch(q, mask_in, tokens, /*shared=*/false));
    }
    block.emitted = true;
    return SetOutputTensor(*block.output, out);
  }

  Expected<nvinfer1::ITensor*> DynamicUpdateSliceOffset(const Op& op,
                                                        int axis) {
    LITERT_ASSIGN_OR_RETURN(auto* start_indices, GetTensor(op.Inputs()[2]));
    LITERT_ASSIGN_OR_RETURN(auto* offset,
                            SliceTensor1d(start_indices, axis, /*size=*/1));
    LITERT_ASSIGN_OR_RETURN(auto operand_type,
                            op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto update_type,
                            op.Inputs()[1].RankedTensorType());
    const int32_t max_start = operand_type.Layout().Dimensions()[axis] -
                              update_type.Layout().Dimensions()[axis];
    const int32_t zero = 0;
    const nvinfer1::Dims dims{1, {1}};
    LITERT_ASSIGN_OR_RETURN(
        auto* lower,
        AddInt32Constant(absl::MakeConstSpan(&zero, 1), dims, "dus_start_min"));
    LITERT_ASSIGN_OR_RETURN(
        auto* upper, AddInt32Constant(absl::MakeConstSpan(&max_start, 1), dims,
                                      "dus_start_max"));
    // TFLite clamps every axis to [0, operand_extent - update_extent]. Only
    // one axis can be partial in this lowering; full-extent axes already
    // ignore their supplied offset, which is equivalent to clamping to zero.
    // Use the original model dimensions before any native V-cache view.
    auto* nonnegative = network_->addElementWise(
        *offset, *lower, nvinfer1::ElementWiseOperation::kMAX);
    if (nonnegative == nullptr || nonnegative->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to clamp TensorRT DynamicUpdateSlice lower bound");
    }
    auto* clamped =
        network_->addElementWise(*nonnegative->getOutput(0), *upper,
                                 nvinfer1::ElementWiseOperation::kMIN);
    if (clamped == nullptr || clamped->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to clamp TensorRT DynamicUpdateSlice upper bound");
    }
    return clamped->getOutput(0);
  }

  Expected<void> LowerInPlaceCacheUpdate(const Op& op,
                                         const InPlaceCache& info) {
    auto* cache = raw_input_map_[op.Inputs()[0].Get()];
    LITERT_ASSIGN_OR_RETURN(auto* update, GetTensor(op.Inputs()[1]));
    if (cache == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "In-place cache update operand is not a network input");
    }
    LITERT_ASSIGN_OR_RETURN(update, AddCastTensor(update, cache->getType()));
    LITERT_ASSIGN_OR_RETURN(auto cache_type, op.Inputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto cache_dims, ConvertDims(cache_type));
    if (info.transposed) {
      // The model's [B, H, D, n] value update becomes [B, H, n, D] rows.
      auto* shuffle = network_->addShuffle(*update);
      if (shuffle == nullptr || shuffle->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to transpose in-place value cache update");
      }
      shuffle->setFirstTranspose(MakePermutation({0, 1, 3, 2}));
      update = shuffle->getOutput(0);
    } else if (info.partial_axis == 3) {
      // [B, H, D, n] update viewed as [B, H*D, n, 1] to match the cache view.
      const auto update_dims = update->getDimensions();
      nvinfer1::Dims view{};
      view.nbDims = 4;
      view.d[0] = update_dims.d[0];
      view.d[1] = update_dims.d[1] * update_dims.d[2];
      view.d[2] = update_dims.d[3];
      view.d[3] = 1;
      LITERT_ASSIGN_OR_RETURN(update, ReshapeTensor(update, view));
    }
    LITERT_ASSIGN_OR_RETURN(auto* write_index,
                            DynamicUpdateSliceOffset(op, info.partial_axis));
    auto* layer = network_->addKVCacheUpdate(*cache, *update, *write_index,
                                             nvinfer1::KVCacheMode::kLINEAR);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT KV cache update layer");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    // The layer output is the network output dispatch binds to the cache
    // buffer itself; consumers inside the network read it through the
    // original layout.
    raw_output_map_[op.Outputs()[0].Get()] = out;
    nvinfer1::ITensor* value = out;
    if (info.partial_axis == 3 && !info.transposed) {
      LITERT_ASSIGN_OR_RETURN(value, ReshapeTensor(out, cache_dims));
    }
    return SetOutputTensor(op.Outputs()[0], value);
  }

  Expected<void> LowerDynamicUpdateSlice(const Op& op) {
    if (auto in_place = in_place_caches_.find(op.Inputs()[0].Get());
        in_place != in_place_caches_.end() && in_place->second.op == op.Get()) {
      return LowerInPlaceCacheUpdate(op, in_place->second);
    }
    LITERT_ASSIGN_OR_RETURN(auto* operand, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* update, GetTensor(op.Inputs()[1]));
    if (Fp16ActivationsEnabled() &&
        op.Inputs()[0].ElementType() == litert::ElementType::Int8) {
      // Cache updates scatter into the int8 boundary tensor directly: use the
      // raw int8 network input as the operand and quantize the FP16 update
      // values, so the cache never round-trips through FP16 storage.
      if (auto raw = raw_input_map_.find(op.Inputs()[0].Get());
          raw != raw_input_map_.end()) {
        operand = raw->second;
      } else if (operand->getType() != nvinfer1::DataType::kINT8) {
        LITERT_ASSIGN_OR_RETURN(operand,
                                AddQuantizeTensor(operand, op.Inputs()[0],
                                                  nvinfer1::DataType::kINT8));
      }
      if (update->getType() != nvinfer1::DataType::kINT8) {
        LITERT_ASSIGN_OR_RETURN(update,
                                AddQuantizeTensor(update, op.Inputs()[1],
                                                  nvinfer1::DataType::kINT8));
      }
    }
    const auto operand_dims = operand->getDimensions();
    const auto update_dims = update->getDimensions();
    const int rank = operand_dims.nbDims;

    // The update writes every axis in full except (at most) one partial axis,
    // so it is a scatter of elements along that axis: for each update element
    // the target coordinate is its own coordinate plus the runtime offset.
    int partial_axis = -1;
    for (int i = 0; i < rank; ++i) {
      if (update_dims.d[i] < operand_dims.d[i]) {
        partial_axis = i;
        break;
      }
    }
    if (partial_axis < 0) {
      // Full overwrite: every offset clamps to zero, so the result is the
      // update. Copy through an identity layer so the update tensor keeps its
      // name.
      auto* identity = network_->addIdentity(*update);
      if (identity == nullptr || identity->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT DynamicUpdateSlice identity");
      }
      return SetOutputTensor(op.Outputs()[0], identity->getOutput(0));
    }

    // indices[c0, .., cA, .., cn] = cA + clamped_start[A], built with a
    // linspace fill so no index constant is materialized.
    LITERT_ASSIGN_OR_RETURN(auto* offset_1d,
                            DynamicUpdateSliceOffset(op, partial_axis));
    nvinfer1::Dims scalar_dims{};
    scalar_dims.nbDims = 0;
    LITERT_ASSIGN_OR_RETURN(auto* offset_scalar,
                            ReshapeTensor(offset_1d, scalar_dims));
    std::vector<int32_t> steps(rank, 0);
    steps[partial_axis] = 1;
    nvinfer1::Dims steps_dims{};
    steps_dims.nbDims = 1;
    steps_dims.d[0] = rank;
    LITERT_ASSIGN_OR_RETURN(auto* steps_tensor,
                            AddInt32Constant(steps, steps_dims, "dus_steps"));
    auto* fill_layer =
        network_->addFill(update_dims, nvinfer1::FillOperation::kLINSPACE,
                          nvinfer1::DataType::kINT32);
    if (fill_layer == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT DynamicUpdateSlice index fill");
    }
    fill_layer->setInput(1, *offset_scalar);
    fill_layer->setInput(2, *steps_tensor);
    auto* indices = fill_layer->getOutput(0);
    if (indices == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to build TensorRT DynamicUpdateSlice indices");
    }

    auto* layer = network_->addScatter(*operand, *indices, *update,
                                       nvinfer1::ScatterMode::kELEMENT);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(
          kLiteRtStatusErrorCompilation,
          "Failed to add TensorRT scatter layer for DynamicUpdateSlice");
    }
    layer->setAxis(partial_axis);
    nvinfer1::ITensor* scatter_out = layer->getOutput(0);
    if (Fp16ActivationsEnabled() &&
        scatter_out->getType() == nvinfer1::DataType::kINT8) {
      // Keep the raw int8 result for the island boundary, and publish the
      // FP16 view for in-island consumers (attention reads the updated
      // cache); the dequantize fuses into the consuming matmul.
      raw_output_map_[op.Outputs()[0].Get()] = scatter_out;
      LITERT_ASSIGN_OR_RETURN(
          auto* value,
          AddDequantizeTensor(op.Outputs()[0], scatter_out, ModeFloatType()));
      return SetOutputTensor(op.Outputs()[0], value);
    }
    return SetOutputTensor(op.Outputs()[0], scatter_out);
  }

  Expected<void> LowerTranspose(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto perm, ReadInt32Constant(op.Inputs()[1]));
    if (perm.size() > nvinfer1::Dims::MAX_DIMS) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Transpose permutation rank is too large");
    }
    nvinfer1::Permutation trt_perm{};
    for (int i = 0; i < perm.size(); ++i) {
      trt_perm.order[i] = perm[i];
    }
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT transpose layer");
    }
    layer->setFirstTranspose(trt_perm);
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerConcatenation(const Op& op) {
    std::vector<nvinfer1::ITensor*> inputs;
    inputs.reserve(op.Inputs().size());
    for (const auto& input : op.Inputs()) {
      LITERT_ASSIGN_OR_RETURN(auto* tensor, GetTensor(input));
      inputs.push_back(tensor);
    }
    // Float inputs may arrive in different precisions (e.g. a BF16
    // runtime_bmm result next to FP16 scores); TensorRT requires one type.
    bool any_fp32 = false;
    bool mixed = false;
    for (auto* tensor : inputs) {
      any_fp32 = any_fp32 || tensor->getType() == nvinfer1::DataType::kFLOAT;
      mixed = mixed || tensor->getType() != inputs[0]->getType();
    }
    if (mixed && Fp16ActivationsEnabled()) {
      const auto common =
          any_fp32 ? nvinfer1::DataType::kFLOAT : ModeFloatType();
      for (auto& tensor : inputs) {
        const auto type = tensor->getType();
        if (type == nvinfer1::DataType::kFLOAT ||
            type == nvinfer1::DataType::kHALF ||
            type == nvinfer1::DataType::kBF16) {
          LITERT_ASSIGN_OR_RETURN(tensor, AddCastTensor(tensor, common));
        }
      }
    }
    auto* layer = network_->addConcatenation(
        inputs.data(), static_cast<int32_t>(inputs.size()));
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT concatenation layer");
    }
    int32_t axis = 0;
    LITERT_RETURN_IF_ERROR(LiteRtGetConcatenationAxisOption(op.Get(), &axis));
    const int rank = inputs[0]->getDimensions().nbDims;
    if (axis < 0) {
      axis += rank;
    }
    layer->setAxis(axis);
    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    LITERT_ASSIGN_OR_RETURN(
        auto* out, AddFusedActivation(layer->getOutput(0), activation));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  // The one-hot of a OneHotTableLookup needs no layer of its own when its
  // product is lowered here: the product gathers the selected rows.
  Expected<void> LowerOneHot(const Op& op) {
    const auto product = OneHotTableLookup(op);
    if (product && op_order_.count(product->Get()) != 0) {
      return {};
    }
    LITERT_ASSIGN_OR_RETURN(auto* indices, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto depth, ReadInt32Constant(op.Inputs()[1]));
    const float off_on[2] = {0.0f, 1.0f};
    LITERT_ASSIGN_OR_RETURN(
        auto* values,
        AddFloatConstant(off_on, nvinfer1::Dims{1, {2}}, "one_hot_values"));
    LITERT_ASSIGN_OR_RETURN(
        auto* depth_tensor,
        AddInt32Constant(depth, nvinfer1::Dims{0, {}}, "one_hot_depth"));
    auto* layer = network_->addOneHot(*indices, *values, *depth_tensor, -1);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT one-hot layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  // The product of a OneHotTableLookup: the rows its one-hot selects, by a
  // gather. Each sum of the product has one term that is not a zero, the
  // row's value, so the gather is exact. INT8 tables are kept as FP16 integers
  // (exact) and scaled after the gather in FP32, as a dequantization does. An
  // index outside [0, depth) selects no row (zeros), as in the one-hot.
  Expected<void> LowerOneHotTableLookup(const Op& selector, const Op& op) {
    const Tensor table = op.Inputs()[1];
    LITERT_ASSIGN_OR_RETURN(auto table_type, table.RankedTensorType());
    const auto table_dims = table_type.Layout().Dimensions();
    const int32_t depth = table_dims[table_dims.size() - 2];
    const int32_t columns = table_dims[table_dims.size() - 1];
    const bool quantized = table.ElementType() == litert::ElementType::Int8;
    std::vector<float> values(static_cast<size_t>(depth) * columns);
    const auto bytes = table.Weights().Bytes();
    if (bytes.size() != values.size() * (quantized ? 1 : sizeof(float))) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected one-hot lookup table size");
    }
    if (quantized) {
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int8_t>(bytes[i]));
      }
    } else {
      std::memcpy(values.data(), bytes.data(), bytes.size());
    }
    // The table stays in the plan, not among the refit constants
    // (AddFloatConstant): refitting the vision export's two tables (47 MB)
    // added seconds to every load of its engines.
    const nvinfer1::DataType table_type_trt =
        quantized ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    owned_weights_.emplace_back(values.size() *
                                (quantized ? sizeof(uint16_t) : sizeof(float)));
    if (quantized) {
      auto* half_data =
          reinterpret_cast<uint16_t*>(owned_weights_.back().data());
      for (size_t i = 0; i < values.size(); ++i) {
        half_data[i] = Fp32ToFp16Bits(values[i]);
      }
    } else {
      std::memcpy(owned_weights_.back().data(), values.data(),
                  owned_weights_.back().size());
    }
    auto* table_layer = network_->addConstant(
        nvinfer1::Dims{2, {depth, columns}},
        nvinfer1::Weights{table_type_trt, owned_weights_.back().data(),
                          static_cast<int64_t>(values.size())});
    if (table_layer == nullptr || table_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the one-hot lookup's table");
    }
    nvinfer1::ITensor* table_tensor = table_layer->getOutput(0);
    LITERT_ASSIGN_OR_RETURN(auto* indices, GetTensor(selector.Inputs()[0]));
    const nvinfer1::Dims index_dims = indices->getDimensions();
    nvinfer1::Dims ones{};
    ones.nbDims = index_dims.nbDims;
    for (int i = 0; i < ones.nbDims; ++i) {
      ones.d[i] = 1;
    }
    const int32_t zero = 0;
    const int32_t last = depth - 1;
    LITERT_ASSIGN_OR_RETURN(
        auto* zero_tensor,
        AddInt32Constant(absl::MakeConstSpan(&zero, 1), ones, "lookup_zero"));
    LITERT_ASSIGN_OR_RETURN(
        auto* last_tensor,
        AddInt32Constant(absl::MakeConstSpan(&last, 1), ones, "lookup_last"));
    auto* below = network_->addElementWise(
        *indices, *last_tensor, nvinfer1::ElementWiseOperation::kMIN);
    auto* clamped =
        below == nullptr
            ? nullptr
            : network_->addElementWise(*below->getOutput(0), *zero_tensor,
                                       nvinfer1::ElementWiseOperation::kMAX);
    auto* in_range =
        clamped == nullptr
            ? nullptr
            : network_->addElementWise(*clamped->getOutput(0), *indices,
                                       nvinfer1::ElementWiseOperation::kEQUAL);
    auto* gather =
        clamped == nullptr
            ? nullptr
            : network_->addGather(*table_tensor, *clamped->getOutput(0), 0);
    if (in_range == nullptr || gather == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT one-hot lookup layers");
    }
    nvinfer1::ITensor* rows = gather->getOutput(0);
    nvinfer1::Dims row_dims = ones;
    row_dims.nbDims = index_dims.nbDims + 1;
    row_dims.d[index_dims.nbDims] = columns;
    if (quantized) {
      LITERT_ASSIGN_OR_RETURN(rows,
                              AddCastTensor(rows, nvinfer1::DataType::kFLOAT));
      std::vector<float> scales(columns);
      if (table.QTypeId() == kLiteRtQuantizationPerChannel) {
        const auto q = table.PerChannelQuantization();
        std::copy(q.scales, q.scales + columns, scales.begin());
      } else {
        std::fill(scales.begin(), scales.end(),
                  table.PerTensorQuantization().scale);
      }
      LITERT_ASSIGN_OR_RETURN(
          auto* scale_tensor,
          AddFloatConstant(
              scales, row_dims,
              "one_hot_lookup_scales_" + std::to_string(table.TensorIndex())));
      auto* scaled = network_->addElementWise(
          *rows, *scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
      if (scaled == nullptr || scaled->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to scale the one-hot lookup's rows");
      }
      rows = scaled->getOutput(0);
    }
    auto* condition = network_->addShuffle(*in_range->getOutput(0));
    if (condition == nullptr || condition->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to shape the one-hot lookup's index check");
    }
    nvinfer1::Dims condition_dims = index_dims;
    condition_dims.nbDims = index_dims.nbDims + 1;
    condition_dims.d[index_dims.nbDims] = 1;
    condition->setReshapeDimensions(condition_dims);
    const float zero_value = 0.0f;
    nvinfer1::Dims zero_dims = ones;
    zero_dims.nbDims = index_dims.nbDims + 1;
    zero_dims.d[index_dims.nbDims] = 1;
    LITERT_ASSIGN_OR_RETURN(
        auto* zero_rows,
        AddFloatConstant(absl::MakeConstSpan(&zero_value, 1), zero_dims,
                         "lookup_zero_rows", rows->getType()));
    auto* select =
        network_->addSelect(*condition->getOutput(0), *rows, *zero_rows);
    if (select == nullptr || select->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the one-hot lookup's select");
    }
    return SetOutputTensor(op.Outputs()[0], select->getOutput(0));
  }

  Expected<void> LowerBatchMatmul(const Op& op) {
    if (const auto selector = op.Inputs()[0].GetDefiningOp();
        selector && op_order_.count(selector->Get()) != 0 &&
        IsOneHotTableLookupProduct(op)) {
      return LowerOneHotTableLookup(*selector, op);
    }
    LITERT_ASSIGN_OR_RETURN(auto* lhs, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* rhs, GetTensor(op.Inputs()[1]));
    bool adj_x = false;
    bool adj_y = false;
    LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAdjXOption(op.Get(), &adj_x));
    LITERT_RETURN_IF_ERROR(LiteRtGetBatchMatmulAdjYOption(op.Get(), &adj_y));
    if (adj_y && transposed_caches_.find(op.Inputs()[1].Get()) !=
                     transposed_caches_.end()) {
      adj_y = false;  // The value cache is already [B, H, S, D].
    }
    // Operands may arrive in different float precisions (e.g. BF16
    // probabilities from a runtime_bmm chain against FP16 chunk values) and in
    // different ranks, the lower one broadcasting over the other's leading
    // dimensions (the Gemma 4 vision pooler multiplies [1, P, T] weights with
    // [P, C] patches); TensorRT's matrix multiply wants equal ranks.
    LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(lhs, rhs));
    auto* layer = network_->addMatrixMultiply(
        *lhs,
        adj_x ? nvinfer1::MatrixOperation::kTRANSPOSE
              : nvinfer1::MatrixOperation::kNONE,
        *rhs,
        adj_y ? nvinfer1::MatrixOperation::kTRANSPOSE
              : nvinfer1::MatrixOperation::kNONE);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT batch matmul layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> FinishFullyConnected(const Op& op, nvinfer1::ITensor* out,
                                      nvinfer1::DataType compute_type) {
    if (op.Inputs().size() == 3) {
      LITERT_ASSIGN_OR_RETURN(auto* bias, GetTensor(op.Inputs()[2]));
      LITERT_ASSIGN_OR_RETURN(bias, AddCastTensor(bias, compute_type));
      LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(out, bias));
      auto* bias_layer = network_->addElementWise(
          *out, *bias, nvinfer1::ElementWiseOperation::kSUM);
      if (bias_layer == nullptr || bias_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT fully connected bias layer");
      }
      out = bias_layer->getOutput(0);
    }
    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    LITERT_ASSIGN_OR_RETURN(out, AddFusedActivation(out, activation));
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto trt_output_type,
                            ConvertDataType(output_type.ElementType()));
    if (Fp16ActivationsEnabled()) {
      // Keep the FP16 real value; boundary conversion happens at MarkOutputs.
    } else if (!IsFloatLike(op.Outputs()[0].ElementType())) {
      LITERT_ASSIGN_OR_RETURN(
          out, AddQuantizeTensor(out, op.Outputs()[0], trt_output_type));
    } else if (trt_output_type != compute_type) {
      LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, trt_output_type));
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  Expected<void> LowerFullyConnected(const Op& op) {
    if (auto fused = fused_gemv_outputs_.find(op.Get());
        fused != fused_gemv_outputs_.end()) {
      return FinishFullyConnected(op, fused->second, ModeFloatType());
    }
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    // TensorRT executes sub-byte weight-only quantization (INT4) only with
    // half-precision activations, so those matmuls compute in FP16. Wider
    // weight types keep FP32 compute.
    const bool sub_byte_weights =
        op.Inputs()[1].ElementType() == litert::ElementType::Int4 ||
        op.Inputs()[1].ElementType() == litert::ElementType::Int2;
    // INT8 per-channel weights (Gemma 4 E2B/E4B's per-layer embedding gate
    // and projection, two small products per layer) take the batched CUDA
    // GEMV over decode and verify rows too: TensorRT otherwise dequantizes
    // them every step and runs a generic matmul kernel, three launches where
    // the plugin needs one. Over many rows they keep TensorRT's GEMM.
    const bool byte_weights =
        op.Inputs()[1].ElementType() == litert::ElementType::Int8 &&
        op.Inputs()[1].HasWeights() &&
        op.Inputs()[1].QTypeId() == kLiteRtQuantizationPerChannel;
    nvinfer1::DataType compute_type = sub_byte_weights
                                          ? nvinfer1::DataType::kHALF
                                          : nvinfer1::DataType::kFLOAT;
    float weight_scale_multiplier = 1.0f;
    if (Fp16ActivationsEnabled()) {
      // Activations already flow as FP16 reals; compute in FP16 and
      // dequantize only the weights (weight-only quantization).
      compute_type = ModeFloatType();
    }
    nvinfer1::ITensor* weights = nullptr;
    const PredequantMode predequant_mode = PredequantizeFcWeightsMode();
    // Predequantization pays off only for M=1 GEMVs (decode): Myelin has no
    // fused sub-byte weight-only GEMV, so it re-materializes dequantized
    // weights every step. At M>=128 (prefill) the one-off materialization
    // amortizes and the dense GEMM is faster than a fused fp8-weight GEMM,
    // so larger-M matmuls keep the quantized-constant path.
    bool static_m_is_one = true;
    {
      const auto in_dims = input->getDimensions();
      for (int i = 0; i + 1 < in_dims.nbDims; ++i) {
        if (in_dims.d[i] != 1) {
          static_m_is_one = false;
          break;
        }
      }
    }
    int64_t rows = 1;
    {
      const auto in_dims = input->getDimensions();
      for (int i = 0; i + 1 < in_dims.nbDims; ++i) {
        rows *= in_dims.d[i];
      }
    }
    const bool blockwise = IsBlockwiseInt4Weights(op.Inputs()[1]);
    const bool cuda_gemm =
        !static_m_is_one &&
        CudaGemmTakes(
            rows, input->getDimensions().d[input->getDimensions().nbDims - 1],
            op.Inputs()[1].ElementType() == litert::ElementType::Int4 ? 4 : 2,
            blockwise ? 32 : 0);
    // The few rows of a verify step run on the batched GEMV like decode.
    const bool cuda_batched_gemv =
        !static_m_is_one && rows > 1 && BatchedGemvTakes(rows);
    // INT4 projections read their RMS norm's input and apply the norm in
    // the kernel (FindNormFusion).
    std::optional<NormFusion> norm;
    if (op.Inputs()[1].ElementType() == litert::ElementType::Int4 &&
        Fp16ActivationsEnabled() &&
        predequant_mode == PredequantMode::kCudaGemv) {
      const auto in_dims = input->getDimensions();
      LITERT_ASSIGN_OR_RETURN(
          norm, FindNormFusion(op.Inputs()[0], in_dims.d[in_dims.nbDims - 1]));
    }
    const NormFusion* norm_ptr = norm.has_value() ? &*norm : nullptr;
    nvinfer1::ITensor* gemv_input = norm.has_value() ? norm->input : input;
    // A gated pair runs as one launch: the GEMM over many rows, the
    // blockwise GEMV's gate mode over a few.
    if (Fp16ActivationsEnabled() &&
        predequant_mode == PredequantMode::kCudaGemv && sub_byte_weights &&
        (cuda_gemm || (blockwise && (static_m_is_one || cuda_batched_gemv)))) {
      if (const auto pair = FindGatedFeedForward(op); pair.has_value()) {
        LITERT_ASSIGN_OR_RETURN(auto* activation,
                                AddCastTensor(gemv_input, compute_type));
        auto gated =
            AddGatedFeedForward(op, *pair, activation, compute_type, norm_ptr);
        if (gated.HasValue()) {
          return {};
        }
        LITERT_LOG(LITERT_INFO,
                   "CUDA gated feed-forward GEMM unavailable for tensor %u: %s",
                   op.Inputs()[1].TensorIndex(),
                   gated.Error().Message().c_str());
      }
      if (const auto swiglu = FindFusedSwiglu(op); swiglu.has_value()) {
        LITERT_ASSIGN_OR_RETURN(auto* activation,
                                AddCastTensor(gemv_input, compute_type));
        auto gated =
            AddFusedSwiglu(op, *swiglu, activation, compute_type, norm_ptr);
        if (gated.HasValue()) {
          return {};
        }
        LITERT_LOG(
            LITERT_INFO, "CUDA SwiGLU GEMM unavailable for tensor %u: %s",
            op.Inputs()[1].TensorIndex(), gated.Error().Message().c_str());
      }
    }
    if (Fp16ActivationsEnabled() &&
        predequant_mode == PredequantMode::kCudaGemv &&
        ((sub_byte_weights &&
          (static_m_is_one || cuda_batched_gemv ||
           (cuda_gemm &&
            (GetCudaGemmMode() == CudaGemmMode::kAll || blockwise)))) ||
         (byte_weights && (static_m_is_one || cuda_batched_gemv)))) {
      LITERT_ASSIGN_OR_RETURN(input, AddCastTensor(input, compute_type));
      // The fused norm's kernel reads the norm's input; a product the CUDA
      // path declines falls back to TensorRT's lowering over the normed one.
      nvinfer1::ITensor* gemv_activation = input;
      if (norm.has_value()) {
        LITERT_ASSIGN_OR_RETURN(gemv_activation,
                                AddCastTensor(gemv_input, compute_type));
      }
      auto fused =
          AddCudaSubbyteGemvGroup(op, gemv_activation, compute_type, norm_ptr);
      if (fused.HasValue()) {
        return FinishFullyConnected(op, *fused, compute_type);
      }
      if (blockwise) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Blockwise weights have no lowering but the CUDA GEMV: " +
                         std::string(fused.Error().Message()));
      }
      LITERT_LOG(LITERT_INFO, "CUDA subbyte GEMV unavailable for tensor %u: %s",
                 op.Inputs()[1].TensorIndex(), fused.Error().Message().c_str());
    }
    if (blockwise) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Blockwise weights need the CUDA GEMV path (BF16 "
                   "activations, cuda_gemv predequantization)");
    }
    if (Fp16ActivationsEnabled() && predequant_mode != PredequantMode::kOff &&
        predequant_mode != PredequantMode::kCudaGemv && static_m_is_one &&
        (op.Inputs()[1].ElementType() == litert::ElementType::Int8 ||
         op.Inputs()[1].ElementType() == litert::ElementType::Int4 ||
         op.Inputs()[1].ElementType() == litert::ElementType::Int2)) {
      // fp8 mode: sub-byte integer values are exact in e4m3 and the fp8
      // dequantize fuses into the GEMV. int8 values need >4 mantissa bits,
      // so those tensors (small kv projections) take float constants.
      Expected<nvinfer1::ITensor*> predequantized =
          (predequant_mode == PredequantMode::kFp8 && sub_byte_weights)
              ? AddFp8PredequantizedFcWeights(op.Inputs()[1], compute_type)
              : AddPredequantizedFcWeights(op.Inputs()[1], compute_type);
      if (predequantized.HasValue()) {
        weights = *predequantized;
      } else {
        LITERT_LOG(LITERT_INFO,
                   "Predequantized FC weights unavailable for tensor %u: %s",
                   op.Inputs()[1].TensorIndex(),
                   predequantized.Error().Message().c_str());
      }
    }
    if (weights == nullptr) {
      LITERT_ASSIGN_OR_RETURN(weights, GetTensor(op.Inputs()[1]));
    }
    if (Fp16ActivationsEnabled()) {
      if (weights->getType() == nvinfer1::DataType::kINT4) {
        int64_t rows = 1;
        const auto in_dims = input->getDimensions();
        for (int i = 0; i + 1 < in_dims.nbDims; ++i) {
          rows *= in_dims.d[i];
        }
        LITERT_ASSIGN_OR_RETURN(
            auto block_scaled,
            AddBlockScaledDequantizeTensor(op.Inputs()[1], weights,
                                           compute_type, rows));
        if (block_scaled.has_value()) {
          weights = *block_scaled;
        }
      }
      if (weights->getType() == nvinfer1::DataType::kINT8 ||
          weights->getType() == nvinfer1::DataType::kINT4) {
        LITERT_ASSIGN_OR_RETURN(
            weights,
            AddDequantizeTensor(op.Inputs()[1], weights, compute_type));
      }
    } else if (IsFloatLike(op.Inputs()[0].ElementType())) {
      LITERT_ASSIGN_OR_RETURN(auto input_type,
                              op.Inputs()[0].RankedTensorType());
      if (!sub_byte_weights) {
        LITERT_ASSIGN_OR_RETURN(compute_type,
                                ConvertDataType(input_type.ElementType()));
      }
    } else if (sub_byte_weights ||
               EnvEnabled("LITERT_NVIDIA_TENSORRT_FOLD_INPUT_SCALE",
                          /*default_value=*/false)) {
      // TensorRT has no mixed int8-activation x int4-weight matmul pattern,
      // and folding also prevents Myelin from fusing the activation Q/DQ pair
      // with the matmul into a quantized GEMM. Feed the raw integer values
      // and fold the activation's per-tensor scale into the weight dequantize
      // scales, which is mathematically identical for symmetric quantization.
      weight_scale_multiplier = op.Inputs()[0].PerTensorQuantization().scale;
    } else {
      LITERT_ASSIGN_OR_RETURN(
          input, AddDequantizeTensor(op.Inputs()[0], input, compute_type));
    }
    if (!Fp16ActivationsEnabled() &&
        !IsFloatLike(op.Inputs()[1].ElementType())) {
      LITERT_ASSIGN_OR_RETURN(
          weights, AddDequantizeTensor(op.Inputs()[1], weights, compute_type,
                                       weight_scale_multiplier));
    }
    LITERT_ASSIGN_OR_RETURN(input, AddCastTensor(input, compute_type));
    LITERT_ASSIGN_OR_RETURN(weights, AddCastTensor(weights, compute_type));
    // Flatten the activation to [M, K] instead of expanding the weights'
    // rank: a reshape between the weight dequantize and the matmul breaks
    // TensorRT's weight-only-quantization fusion, and Myelin then
    // materializes the dequantized weights to memory on every invocation
    // (the dominant decode-step cost). Activation-side reshapes fuse freely.
    const auto input_dims = input->getDimensions();
    const auto weight_dims = weights->getDimensions();
    const bool flatten_input = input_dims.nbDims != 2;
    nvinfer1::ITensor* matmul_input = input;
    if (flatten_input) {
      int64_t m = 1;
      for (int i = 0; i + 1 < input_dims.nbDims; ++i) {
        m *= input_dims.d[i];
      }
      nvinfer1::Dims flat{};
      flat.nbDims = 2;
      flat.d[0] = m;
      flat.d[1] = input_dims.d[input_dims.nbDims - 1];
      LITERT_ASSIGN_OR_RETURN(matmul_input, ReshapeTensor(input, flat));
    }
    if (weight_dims.nbDims != 2) {
      LITERT_ASSIGN_OR_RETURN(
          weights, ExpandMatrixWeightsRank(
                       weights, matmul_input->getDimensions().nbDims));
    }
    auto* layer = network_->addMatrixMultiply(
        *matmul_input, nvinfer1::MatrixOperation::kNONE, *weights,
        nvinfer1::MatrixOperation::kTRANSPOSE);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT fully connected matmul layer");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    if (flatten_input) {
      LITERT_ASSIGN_OR_RETURN(auto out_type,
                              op.Outputs()[0].RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto out_dims, ConvertDims(out_type));
      LITERT_ASSIGN_OR_RETURN(out, ReshapeTensor(out, out_dims));
    }
    return FinishFullyConnected(op, out, compute_type);
  }

  Expected<void> LowerSelectV2(const Op& op) {
    if (const auto attention = QuantAttentionBehindMaskSelect(op);
        attention.has_value()) {
      LITERT_ASSIGN_OR_RETURN(auto* value, GetTensor(*attention));
      return SetOutputTensor(op.Outputs()[0], value);
    }
    nvinfer1::ITensor* condition = nullptr;
    if (const auto select = windowed_by_select_.find(op.Get());
        select != windowed_by_select_.end()) {
      LITERT_ASSIGN_OR_RETURN(condition,
                              WindowMaskSlice(windowed_reads_[select->second]));
    } else {
      LITERT_ASSIGN_OR_RETURN(condition, GetTensor(op.Inputs()[0]));
    }
    LITERT_ASSIGN_OR_RETURN(auto* then_tensor, GetTensor(op.Inputs()[1]));
    LITERT_ASSIGN_OR_RETURN(auto* else_tensor, GetTensor(op.Inputs()[2]));
    const int target_rank = std::max({condition->getDimensions().nbDims,
                                      then_tensor->getDimensions().nbDims,
                                      else_tensor->getDimensions().nbDims});
    LITERT_ASSIGN_OR_RETURN(condition,
                            ExpandRankForElementwise(condition, target_rank));
    LITERT_ASSIGN_OR_RETURN(then_tensor,
                            ExpandRankForElementwise(then_tensor, target_rank));
    LITERT_ASSIGN_OR_RETURN(else_tensor,
                            ExpandRankForElementwise(else_tensor, target_rank));
    LITERT_RETURN_IF_ERROR(
        MatchElementwiseFloatTypes(then_tensor, else_tensor));
    auto* layer = network_->addSelect(*condition, *then_tensor, *else_tensor);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT select layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerSum(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto axes, ReadInt32Constant(op.Inputs()[1]));
    // FP16 reduction sums overflow half range (e.g. RMSNorm sums of squares
    // across the hidden dim); accumulate in FP32 and convert back.
    const bool widen = Fp16ActivationsEnabled() &&
                       input->getType() != nvinfer1::DataType::kFLOAT;
    if (widen) {
      LITERT_ASSIGN_OR_RETURN(input,
                              AddCastTensor(input, nvinfer1::DataType::kFLOAT));
    }
    const int rank = input->getDimensions().nbDims;
    uint32_t axes_mask = 0;
    for (int32_t axis : axes) {
      if (axis < 0) {
        axis += rank;
      }
      axes_mask |= 1U << axis;
    }
    bool keep_dims = false;
    LITERT_RETURN_IF_ERROR(LiteRtGetSumKeepDimsOption(op.Get(), &keep_dims));
    auto* layer = network_->addReduce(*input, nvinfer1::ReduceOperation::kSUM,
                                      axes_mask, keep_dims);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT sum-reduce layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerMean(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto axes, ReadInt32Constant(op.Inputs()[1]));
    // As for Sum: accumulate in FP32.
    const bool widen = Fp16ActivationsEnabled() &&
                       input->getType() != nvinfer1::DataType::kFLOAT;
    if (widen) {
      LITERT_ASSIGN_OR_RETURN(input,
                              AddCastTensor(input, nvinfer1::DataType::kFLOAT));
    }
    const int rank = input->getDimensions().nbDims;
    uint32_t axes_mask = 0;
    for (int32_t axis : axes) {
      if (axis < 0) {
        axis += rank;
      }
      axes_mask |= 1U << axis;
    }
    bool keep_dims = false;
    LITERT_RETURN_IF_ERROR(LiteRtGetMeanKeepDimsOption(op.Get(), &keep_dims));
    auto* layer = network_->addReduce(*input, nvinfer1::ReduceOperation::kAVG,
                                      axes_mask, keep_dims);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT mean-reduce layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  // The k largest values of the last axis, sorted, and their indices.
  Expected<void> LowerTopkV2(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto k, ReadInt32Constant(op.Inputs()[1]));
    // TensorRT's TopK takes FP32 or FP16 values; keep the selection exact.
    if (input->getType() != nvinfer1::DataType::kFLOAT) {
      LITERT_ASSIGN_OR_RETURN(input,
                              AddCastTensor(input, nvinfer1::DataType::kFLOAT));
    }
    const int rank = input->getDimensions().nbDims;
    auto* layer = network_->addTopK(*input, nvinfer1::TopKOperation::kMAX, k[0],
                                    1U << (rank - 1));
    if (layer == nullptr || layer->getOutput(0) == nullptr ||
        layer->getOutput(1) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT TopK layer");
    }
    nvinfer1::ITensor* values = layer->getOutput(0);
    if (Fp16ActivationsEnabled()) {
      LITERT_ASSIGN_OR_RETURN(values, AddCastTensor(values, ModeFloatType()));
    }
    LITERT_RETURN_IF_ERROR(SetOutputTensor(op.Outputs()[0], values));
    return SetOutputTensor(op.Outputs()[1], layer->getOutput(1));
  }

  Expected<void> LowerReduceMax(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto axes, ReadInt32Constant(op.Inputs()[1]));
    const int rank = input->getDimensions().nbDims;
    uint32_t axes_mask = 0;
    for (int32_t axis : axes) {
      if (axis < 0) {
        axis += rank;
      }
      axes_mask |= 1U << axis;
    }
    bool keep_dims = false;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetReduceMaxKeepDimsOption(op.Get(), &keep_dims));
    auto* layer = network_->addReduce(*input, nvinfer1::ReduceOperation::kMAX,
                                      axes_mask, keep_dims);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT reduce-max layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  // a ^ b of int32 tensors bit by bit, TensorRT having no integer bitwise
  // operation: bit k of x (k < 31) is floor(x / 2^k) - 2 floor(x / 2^(k+1)),
  // bit 31 is x < 0, and a bit of the result is p + q - 2 p q. TensorRT fuses
  // the elementwise layers into one kernel.
  Expected<void> LowerBitwiseXor(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* a, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* b, GetTensor(op.Inputs()[1]));
    LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(a, b));
    const int rank = a->getDimensions().nbDims;
    const auto scalar = [&](int32_t value) -> nvinfer1::ITensor* {
      owned_weights_.emplace_back(sizeof(int32_t));
      std::memcpy(owned_weights_.back().data(), &value, sizeof(value));
      nvinfer1::Dims dims{};
      dims.nbDims = rank;
      for (int i = 0; i < rank; ++i) dims.d[i] = 1;
      auto* layer = network_->addConstant(
          dims, nvinfer1::Weights{nvinfer1::DataType::kINT32,
                                  owned_weights_.back().data(), 1});
      return layer == nullptr ? nullptr : layer->getOutput(0);
    };
    using Ew = nvinfer1::ElementWiseOperation;
    const auto ew = [&](nvinfer1::ITensor* x, nvinfer1::ITensor* y,
                        Ew operation) -> nvinfer1::ITensor* {
      if (x == nullptr || y == nullptr) return nullptr;
      auto* layer = network_->addElementWise(*x, *y, operation);
      return layer == nullptr ? nullptr : layer->getOutput(0);
    };
    const auto bit = [&](nvinfer1::ITensor* x, int k) -> nvinfer1::ITensor* {
      if (k == 31) {
        auto* negative = ew(x, scalar(0), Ew::kLESS);
        if (negative == nullptr) return nullptr;
        auto* cast = network_->addCast(*negative, nvinfer1::DataType::kINT32);
        return cast == nullptr ? nullptr : cast->getOutput(0);
      }
      auto* low = ew(x, scalar(int32_t{1} << k), Ew::kFLOOR_DIV);
      nvinfer1::ITensor* high = nullptr;
      if (k < 30) {
        high = ew(x, scalar(int32_t{1} << (k + 1)), Ew::kFLOOR_DIV);
      } else {
        // floor(x / 2^31), 2^31 not being an int32: -1 for x < 0, else 0.
        auto* negative = ew(x, scalar(0), Ew::kLESS);
        auto* cast =
            negative == nullptr
                ? nullptr
                : network_->addCast(*negative, nvinfer1::DataType::kINT32);
        high = cast == nullptr ? nullptr
                               : ew(scalar(0), cast->getOutput(0), Ew::kSUB);
      }
      return ew(low, ew(high, scalar(2), Ew::kPROD), Ew::kSUB);
    };
    nvinfer1::ITensor* result = nullptr;
    for (int k = 0; k < 32; ++k) {
      auto* p = bit(a, k);
      auto* q = bit(b, k);
      auto* x = ew(ew(p, q, Ew::kSUM),
                   ew(scalar(2), ew(p, q, Ew::kPROD), Ew::kPROD), Ew::kSUB);
      auto* term =
          ew(x, scalar(static_cast<int32_t>(uint32_t{1} << k)), Ew::kPROD);
      result = result == nullptr ? term : ew(result, term, Ew::kSUM);
      if (result == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add the TensorRT layers of BITWISE_XOR");
      }
    }
    return SetOutputTensor(op.Outputs()[0], result);
  }

  // GATHER along `axis` (batch_dims 0, IsGatherSupported).
  Expected<void> LowerGather(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* data, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* indices, GetTensor(op.Inputs()[1]));
    int32_t axis = 0;
    LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(op.Get(), &axis));
    if (axis < 0) axis += data->getDimensions().nbDims;
    auto* layer = network_->addGather(*data, *indices, axis);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT gather layer");
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerFloorMod(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* lhs, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* rhs, GetTensor(op.Inputs()[1]));
    LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(lhs, rhs));
    auto* div = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
    if (div == nullptr || div->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT floor_div layer");
    }
    auto* product = network_->addElementWise(
        *div->getOutput(0), *rhs, nvinfer1::ElementWiseOperation::kPROD);
    if (product == nullptr || product->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT floor_mod product layer");
    }
    auto* result = network_->addElementWise(
        *lhs, *product->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
    if (result == nullptr || result->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT floor_mod layer");
    }
    return SetOutputTensor(op.Outputs()[0], result->getOutput(0));
  }

  Expected<nvinfer1::ITensor*> TransposeNhcwToNchw(nvinfer1::ITensor* input) {
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT NHWC->NCHW transpose");
    }
    layer->setFirstTranspose(MakePermutation({0, 3, 1, 2}));
    return layer->getOutput(0);
  }

  Expected<nvinfer1::ITensor*> TransposeNchwToNhwc(nvinfer1::ITensor* input) {
    auto* layer = network_->addShuffle(*input);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT NCHW->NHWC transpose");
    }
    layer->setFirstTranspose(MakePermutation({0, 2, 3, 1}));
    return layer->getOutput(0);
  }

  Expected<void> LowerConv2d(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* input_nhwc, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* input_nchw, TransposeNhcwToNchw(input_nhwc));
    LITERT_ASSIGN_OR_RETURN(auto filter_type,
                            op.Inputs()[1].RankedTensorType());
    const auto filter_dims = filter_type.Layout().Dimensions();
    const int out_channels = filter_dims[0];
    const int kernel_h = filter_dims[1];
    const int kernel_w = filter_dims[2];
    LITERT_ASSIGN_OR_RETURN(auto kernel_weights,
                            MakeConv2dWeightsOihw(op.Inputs()[1]));
    nvinfer1::Weights bias_weights{};
    if (op.Inputs().size() == 3) {
      LITERT_ASSIGN_OR_RETURN(bias_weights, MakeWeights(op.Inputs()[2]));
    }

    nvinfer1::Dims kernel_dims{};
    kernel_dims.nbDims = 2;
    kernel_dims.d[0] = kernel_h;
    kernel_dims.d[1] = kernel_w;
    auto* conv = network_->addConvolutionNd(
        *input_nchw, out_channels, kernel_dims, kernel_weights, bias_weights);
    if (conv == nullptr || conv->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add TensorRT Conv2D layer");
    }
    LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
        kernel_weights,
        "conv_kernel_" + std::to_string(op.Inputs()[1].TensorIndex())));
    if (bias_weights.values != nullptr && bias_weights.count > 0) {
      LITERT_RETURN_IF_ERROR(RegisterRefitWeight(
          bias_weights,
          "conv_bias_" + std::to_string(op.Inputs()[2].TensorIndex())));
    }
    int32_t stride_w = 1;
    int32_t stride_h = 1;
    int32_t dilation_w = 1;
    int32_t dilation_h = 1;
    uint32_t padding = litert::kPaddingSame;
    LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideWOption(op.Get(), &stride_w));
    LITERT_RETURN_IF_ERROR(LiteRtGetConv2dStrideHOption(op.Get(), &stride_h));
    LITERT_RETURN_IF_ERROR(
        LiteRtGetConv2dDilationWOption(op.Get(), &dilation_w));
    LITERT_RETURN_IF_ERROR(
        LiteRtGetConv2dDilationHOption(op.Get(), &dilation_h));
    LITERT_RETURN_IF_ERROR(LiteRtGetConv2dPaddingOption(op.Get(), &padding));
    nvinfer1::Dims stride_dims{};
    stride_dims.nbDims = 2;
    stride_dims.d[0] = stride_h;
    stride_dims.d[1] = stride_w;
    conv->setStrideNd(stride_dims);
    nvinfer1::Dims dilation_dims{};
    dilation_dims.nbDims = 2;
    dilation_dims.d[0] = dilation_h;
    dilation_dims.d[1] = dilation_w;
    conv->setDilationNd(dilation_dims);
    if (padding == litert::kPaddingSame) {
      conv->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
    }

    LITERT_ASSIGN_OR_RETURN(auto activation, GetFusedActivation(op));
    LITERT_ASSIGN_OR_RETURN(auto* conv_output,
                            AddFusedActivation(conv->getOutput(0), activation));
    LITERT_ASSIGN_OR_RETURN(auto* output_nhwc,
                            TransposeNchwToNhwc(conv_output));
    return SetOutputTensor(op.Outputs()[0], output_nhwc);
  }

  // Scatters `update` into `operand` at the runtime offset held in
  // `start_indices` (i32[rank]); exactly one axis may be partial. Mirrors
  // LowerDynamicUpdateSlice's linspace/scatter construction.
  Expected<nvinfer1::ITensor*> BuildScatterUpdate(
      nvinfer1::ITensor* operand, nvinfer1::ITensor* update,
      nvinfer1::ITensor* start_indices) {
    const auto operand_dims = operand->getDimensions();
    const auto update_dims = update->getDimensions();
    const int rank = operand_dims.nbDims;
    int partial_axis = -1;
    for (int i = 0; i < rank; ++i) {
      if (update_dims.d[i] < operand_dims.d[i]) {
        partial_axis = i;
        break;
      }
    }
    if (partial_axis < 0) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Composite cache update expects a partial axis");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto* offset_1d,
        SliceTensor1d(start_indices, /*start_index=*/partial_axis,
                      /*size=*/1));
    nvinfer1::Dims scalar_dims{};
    scalar_dims.nbDims = 0;
    LITERT_ASSIGN_OR_RETURN(auto* offset_scalar,
                            ReshapeTensor(offset_1d, scalar_dims));
    std::vector<int32_t> steps(rank, 0);
    steps[partial_axis] = 1;
    nvinfer1::Dims steps_dims{};
    steps_dims.nbDims = 1;
    steps_dims.d[0] = rank;
    LITERT_ASSIGN_OR_RETURN(
        auto* steps_tensor,
        AddInt32Constant(steps, steps_dims, "composite_dus_steps"));
    auto* fill_layer =
        network_->addFill(update_dims, nvinfer1::FillOperation::kLINSPACE,
                          nvinfer1::DataType::kINT32);
    if (fill_layer == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add composite cache-update index fill");
    }
    fill_layer->setInput(1, *offset_scalar);
    fill_layer->setInput(2, *steps_tensor);
    auto* indices = fill_layer->getOutput(0);
    if (indices == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to build composite cache-update indices");
    }
    auto* layer = network_->addScatter(*operand, *indices, *update,
                                       nvinfer1::ScatterMode::kELEMENT);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add composite cache-update scatter");
    }
    layer->setAxis(partial_axis);
    return layer->getOutput(0);
  }

  // Returns the int8 boundary form of a quantized tensor's value: the raw
  // network input when available, otherwise the current value (re)quantized.
  Expected<nvinfer1::ITensor*> GetRawInt8Tensor(const Tensor& tensor) {
    if (auto raw = raw_input_map_.find(tensor.Get());
        raw != raw_input_map_.end() &&
        raw->second->getType() == nvinfer1::DataType::kINT8) {
      return raw->second;
    }
    LITERT_ASSIGN_OR_RETURN(auto* value, GetTensor(tensor));
    if (value->getType() == nvinfer1::DataType::kINT8) {
      return value;
    }
    return AddQuantizeTensor(value, tensor, nvinfer1::DataType::kINT8);
  }

  // Publishes an int8 scatter result: raw form for the island boundary and,
  // in half-precision mode, a dequantized view for in-island consumers.
  Expected<void> SetQuantizedScatterOutput(const Tensor& tensor,
                                           nvinfer1::ITensor* scatter_out) {
    if (Fp16ActivationsEnabled()) {
      raw_output_map_[tensor.Get()] = scatter_out;
      LITERT_ASSIGN_OR_RETURN(
          auto* value,
          AddDequantizeTensor(tensor, scatter_out, ModeFloatType()));
      return SetOutputTensor(tensor, value);
    }
    return SetOutputTensor(tensor, scatter_out);
  }

  // odml.rms_norm(x[, w]) = x * rsqrt(mean(x^2, last_axis) + eps) [* w].
  // eps is the composite's attribute: 1e-6 for Gemma 4, while Qwen 3.5 states
  // its linear-attention q/k L2 norms as RMS norms with eps 1e-6 / 128.
  Expected<void> LowerCompositeRmsNorm(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* x, GetTensor(op.Inputs()[0]));
    const int rank = x->getDimensions().nbDims;
    auto* sq_layer =
        network_->addElementWise(*x, *x, nvinfer1::ElementWiseOperation::kPROD);
    if (sq_layer == nullptr || sq_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm square");
    }
    auto* mean_layer = network_->addReduce(
        *sq_layer->getOutput(0), nvinfer1::ReduceOperation::kAVG,
        /*reduceAxes=*/1U << (rank - 1), /*keepDims=*/true);
    if (mean_layer == nullptr || mean_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm mean");
    }
    nvinfer1::Dims one_dims{};
    one_dims.nbDims = rank;
    for (int i = 0; i < rank; ++i) {
      one_dims.d[i] = 1;
    }
    float eps_value = 1e-6f;
    CompositeOptions options;
    if (options.InitFromOp(op.Get()) == kLiteRtStatusOk &&
        options.attributes_map.has_value() &&
        !options.attributes_map.value()["epsilon"].IsNull()) {
      eps_value = options.attributes_map.value()["epsilon"].AsFloat();
    }
    LITERT_ASSIGN_OR_RETURN(
        auto* eps, AddFloatConstant(absl::MakeConstSpan(&eps_value, 1),
                                    one_dims, "rms_norm_eps", x->getType()));
    auto* add_layer = network_->addElementWise(
        *mean_layer->getOutput(0), *eps, nvinfer1::ElementWiseOperation::kSUM);
    if (add_layer == nullptr || add_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm epsilon");
    }
    auto* sqrt_layer = network_->addUnary(*add_layer->getOutput(0),
                                          nvinfer1::UnaryOperation::kSQRT);
    if (sqrt_layer == nullptr || sqrt_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm sqrt");
    }
    auto* recip_layer = network_->addUnary(*sqrt_layer->getOutput(0),
                                           nvinfer1::UnaryOperation::kRECIP);
    if (recip_layer == nullptr || recip_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm reciprocal");
    }
    auto* norm_layer = network_->addElementWise(
        *x, *recip_layer->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (norm_layer == nullptr || norm_layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add rms_norm scale");
    }
    nvinfer1::ITensor* out = norm_layer->getOutput(0);
    if (op.Inputs().size() == 2) {
      LITERT_ASSIGN_OR_RETURN(auto* w, GetTensor(op.Inputs()[1]));
      LITERT_ASSIGN_OR_RETURN(w, AddCastTensor(w, out->getType()));
      LITERT_RETURN_IF_ERROR(MatchElementwiseRanks(out, w));
      auto* w_layer = network_->addElementWise(
          *out, *w, nvinfer1::ElementWiseOperation::kPROD);
      if (w_layer == nullptr || w_layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add rms_norm weight");
      }
      out = w_layer->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  // odml.cache_update: quantize both updates and scatter them into their
  // The INT8 cache update as the cache write plugin: the step's rows
  // quantized and written into the cache's output buffer, which LiteRT-LM
  // binds to the cache input's memory, so nothing else moves (the plugin
  // copies the cache first if the buffers differ). False when the caches
  // are not a [1, H, S, D] key cache and a [1, H, D, S] (or [1, H, S, D])
  // value cache updated with [1, H, T, D] rows, which keeps the scatter.
  Expected<bool> LowerInt8CacheWrite(const Op& op, nvinfer1::ITensor* cache_k,
                                     nvinfer1::ITensor* cache_v,
                                     nvinfer1::ITensor* update_k,
                                     nvinfer1::ITensor* update_v,
                                     nvinfer1::ITensor* start_k,
                                     nvinfer1::ITensor* start_v) {
    const auto k_quant = Int8CacheQuantization(op.Outputs()[0]);
    const auto v_quant = Int8CacheQuantization(op.Outputs()[1]);
    if (!k_quant || !v_quant) {
      return false;
    }
    const auto float_type = [](nvinfer1::DataType type) {
      return type == nvinfer1::DataType::kHALF ||
             type == nvinfer1::DataType::kBF16 ||
             type == nvinfer1::DataType::kFLOAT;
    };
    const auto kd = cache_k->getDimensions();
    const auto vd = cache_v->getDimensions();
    const auto ud = update_k->getDimensions();
    const auto uvd = update_v->getDimensions();
    if (kd.nbDims != 4 || vd.nbDims != 4 || ud.nbDims != 4 || uvd.nbDims != 4 ||
        !float_type(update_k->getType()) || !float_type(update_v->getType()) ||
        cache_k->getType() != nvinfer1::DataType::kINT8 ||
        cache_v->getType() != nvinfer1::DataType::kINT8) {
      return false;
    }
    const int64_t heads = kd.d[1];
    const int64_t seq = kd.d[2];
    const int64_t depth = kd.d[3];
    const int64_t tokens = ud.d[2];
    if (kd.d[0] != 1 || ud.d[0] != 1 || ud.d[1] != heads || ud.d[3] != depth ||
        tokens <= 0 || tokens > seq || seq == depth) {
      return false;
    }
    for (int i = 0; i < 4; ++i) {
      if (uvd.d[i] != ud.d[i]) return false;
    }
    bool value_columns = false;
    if (vd.d[0] == 1 && vd.d[1] == heads && vd.d[2] == depth &&
        vd.d[3] == seq) {
      value_columns = true;
    } else if (vd.d[0] != 1 || vd.d[1] != heads || vd.d[2] != seq ||
               vd.d[3] != depth) {
      return false;
    }
    // The row position: element 2 of the key cache's start indices; the
    // value cache's partial axis is 3 when it is stored as columns.
    LITERT_ASSIGN_OR_RETURN(auto* pos_k,
                            SliceTensor1d(start_k, /*start_index=*/2,
                                          /*size=*/1));
    LITERT_ASSIGN_OR_RETURN(auto* pos_v,
                            SliceTensor1d(start_v, value_columns ? 3 : 2,
                                          /*size=*/1));
    const struct {
      nvinfer1::ITensor* cache;
      nvinfer1::ITensor* update;
      nvinfer1::ITensor* position;
      bool columns;
      std::pair<float, int32_t> quant;
    } writes[2] = {{cache_k, update_k, pos_k, false, *k_quant},
                   {cache_v, update_v, pos_v, value_columns, *v_quant}};
    for (int i = 0; i < 2; ++i) {
      TrtPtr<nvinfer1::IPluginV3> plugin(CreateInt8CacheWritePlugin(
          writes[i].columns, writes[i].quant.first, writes[i].quant.second));
      if (!plugin) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to create CUDA INT8 cache write plugin");
      }
      std::array<nvinfer1::ITensor*, 3> inputs = {
          writes[i].cache, writes[i].update, writes[i].position};
      auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(*network_, inputs.data(),
                                                       inputs.size(), *plugin);
      if (layer == nullptr || layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add CUDA INT8 cache write plugin layer");
      }
      layer->setName(KeepName(UniqueName("int8_cache_write")));
      owned_plugins_.push_back(std::move(plugin));
      LITERT_RETURN_IF_ERROR(
          SetQuantizedScatterOutput(op.Outputs()[i], layer->getOutput(0)));
    }
    return true;
  }

  // int8 caches (the second one through a [0,1,3,2] transpose), mirroring
  // the decomposition without its inlined index glue.
  // The write position of an in-place cache write along `axis`: the start
  // index of that axis, clamped to [0, cache_extent - update_extent] as
  // TFLite's DynamicUpdateSlice clamps it.
  Expected<nvinfer1::ITensor*> CacheWriteOffset(
      nvinfer1::ITensor* start_indices, int axis, int32_t max_start,
      const std::string& name) {
    LITERT_ASSIGN_OR_RETURN(auto* offset,
                            SliceTensor1d(start_indices, axis, /*size=*/1));
    const int32_t zero = 0;
    const nvinfer1::Dims dims{1, {1}};
    LITERT_ASSIGN_OR_RETURN(
        auto* lower,
        AddInt32Constant(absl::MakeConstSpan(&zero, 1), dims, name + "_min"));
    LITERT_ASSIGN_OR_RETURN(
        auto* upper, AddInt32Constant(absl::MakeConstSpan(&max_start, 1), dims,
                                      name + "_max"));
    auto* nonnegative = network_->addElementWise(
        *offset, *lower, nvinfer1::ElementWiseOperation::kMAX);
    if (nonnegative == nullptr || nonnegative->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to clamp the cache write lower bound");
    }
    auto* clamped =
        network_->addElementWise(*nonnegative->getOutput(0), *upper,
                                 nvinfer1::ElementWiseOperation::kMIN);
    if (clamped == nullptr || clamped->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to clamp the cache write upper bound");
    }
    return clamped->getOutput(0);
  }

  // A cache update whose results the graph reads (DecodeCacheUpdateMode):
  // two native in-place writes of the n rows at the start indices, the key
  // rows into k [1, H, S, D] and the value rows into the value cache held as
  // [1, H, S, D]. The update's own gate (params[3] > 0) is not applied: the
  // runtime never issues an empty update.
  Expected<void> LowerDecodeCompositeCacheUpdate(
      const Op& op, const DecodeCacheUpdateLayout& layout) {
    const auto& value_info = in_place_caches_[op.Inputs()[4].Get()];
    if (!value_info.transposed) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Cache update needs the value cache held as [B, H, S, D]");
    }
    for (int i = 0; i < 2; ++i) {
      auto* cache = raw_input_map_[op.Inputs()[3 + i].Get()];
      if (cache == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "In-place cache update operand is not a network input");
      }
      LITERT_ASSIGN_OR_RETURN(auto* update, GetTensor(op.Inputs()[i]));
      LITERT_ASSIGN_OR_RETURN(update, AddCastTensor(update, cache->getType()));
      LITERT_ASSIGN_OR_RETURN(auto* start_indices,
                              GetTensor(op.Inputs()[5 + i]));
      // The model's value start indexes its [B, H, D, S] layout: axis 3.
      LITERT_ASSIGN_OR_RETURN(
          auto* write_index,
          CacheWriteOffset(start_indices, i == 0 ? 2 : 3,
                           layout.seq - layout.rows,
                           i == 0 ? "cache_write_k" : "cache_write_v"));
      auto* layer = network_->addKVCacheUpdate(*cache, *update, *write_index,
                                               nvinfer1::KVCacheMode::kLINEAR);
      if (layer == nullptr || layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add TensorRT KV cache update layer");
      }
      nvinfer1::ITensor* out = layer->getOutput(0);
      raw_output_map_[op.Outputs()[i].Get()] = out;
      LITERT_RETURN_IF_ERROR(SetOutputTensor(op.Outputs()[i], out));
    }
    return {};
  }

  Expected<void> LowerCompositeCacheUpdate(const Op& op) {
    if (const auto ring = PrefillCacheUpdateRingMode(op); ring.has_value()) {
      LITERT_ASSIGN_OR_RETURN(auto* update_k, GetTensor(op.Inputs()[0]));
      LITERT_ASSIGN_OR_RETURN(auto* update_v, GetTensor(op.Inputs()[1]));
      LITERT_ASSIGN_OR_RETURN(auto* params, GetTensor(op.Inputs()[2]));
      LITERT_ASSIGN_OR_RETURN(
          update_k, AddCastTensor(update_k, nvinfer1::DataType::kHALF));
      LITERT_ASSIGN_OR_RETURN(
          update_v, AddCastTensor(update_v, nvinfer1::DataType::kHALF));
      const auto info = in_place_caches_.find(op.Inputs()[4].Get());
      if (info == in_place_caches_.end() || info->second.op != op.Get()) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Float cache update has no in-place cache contract");
      }
      const auto attention_join = FindPrefillAttentionJoin(op);
      TrtPtr<nvinfer1::IPluginV3> plugin(CreateCacheUpdatePlugin(
          *ring, info->second.transposed, attention_join.has_value()));
      if (!plugin) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to create CUDA cache update plugin");
      }
      std::vector<nvinfer1::ITensor*> inputs = {
          raw_input_map_.at(op.Inputs()[3].Get()),
          raw_input_map_.at(op.Inputs()[4].Get()), update_k, update_v, params};
      // Native aliasing alone does not order readers across a plugin boundary.
      // Opaque, complete reader outputs establish that order without copying
      // the cache or computing a reduction. Do not replace them with scalar
      // slices: TensorRT could specialize such a slice into a partial read.
      if (attention_join.has_value()) {
        LITERT_ASSIGN_OR_RETURN(auto* joined, GetTensor(*attention_join));
        inputs.push_back(joined);
      } else {
        std::unordered_set<LiteRtTensor> readers;
        for (int i : {3, 4}) {
          for (const auto& use : op.Inputs()[i].Uses()) {
            if (use.user.Get() == op.Get()) continue;
            for (const auto& result : use.user.Outputs()) {
              if (readers.insert(result.Get()).second) {
                LITERT_ASSIGN_OR_RETURN(auto* completed_read,
                                        GetTensor(result));
                inputs.push_back(completed_read);
              }
            }
          }
        }
      }
      auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(*network_, inputs.data(),
                                                       inputs.size(), *plugin);
      if (layer == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add CUDA cache update plugin layer");
      }
      layer->setName(KeepName(UniqueName(attention_join.has_value()
                                             ? "cache_update_forward_attention"
                                             : "cache_update_read_barriers")));
      owned_plugins_.push_back(std::move(plugin));
      if (attention_join.has_value()) {
        // This opaque output, not a second consumer of the original join,
        // feeds the rest of attention. TensorRT cannot bypass preparation by
        // recomputing the old-cache branch for a downstream consumer.
        LITERT_RETURN_IF_ERROR(
            SetOutputTensor(*attention_join, layer->getOutput(2)));
      }
      // TensorRT requires a single aliased output per network input. Prepare
      // one contiguous patch, then let its native KV layer own the in-place
      // write and old-reader ordering. Ring patches cover only the local
      // cache; linear patches cover U rows, never the full global cache.
      const int32_t zero = 0;
      LITERT_ASSIGN_OR_RETURN(
          auto* base,
          AddInt32Constant(absl::MakeConstSpan(&zero, 1),
                           nvinfer1::Dims{1, {1}}, "cache_patch_base"));
      if (!*ring) {
        LITERT_ASSIGN_OR_RETURN(auto* flat_params,
                                ReshapeTensor(params, nvinfer1::Dims{1, {7}}));
        LITERT_ASSIGN_OR_RETURN(auto* write, SliceTensor1d(flat_params, 0, 1));
        const int32_t last =
            inputs[0]->getDimensions().d[2] - update_k->getDimensions().d[2];
        LITERT_ASSIGN_OR_RETURN(
            auto* upper,
            AddInt32Constant(absl::MakeConstSpan(&last, 1),
                             nvinfer1::Dims{1, {1}}, "cache_patch_last_base"));
        auto* nonnegative = network_->addElementWise(
            *write, *base, nvinfer1::ElementWiseOperation::kMAX);
        if (nonnegative == nullptr || nonnegative->getOutput(0) == nullptr) {
          return Error(kLiteRtStatusErrorCompilation,
                       "Failed to clamp cache patch lower bound");
        }
        auto* clamped =
            network_->addElementWise(*nonnegative->getOutput(0), *upper,
                                     nvinfer1::ElementWiseOperation::kMIN);
        if (clamped == nullptr || clamped->getOutput(0) == nullptr) {
          return Error(kLiteRtStatusErrorCompilation,
                       "Failed to clamp cache patch upper bound");
        }
        base = clamped->getOutput(0);
      }
      for (int i = 0; i < 2; ++i) {
        auto* patch = layer->getOutput(i);
        if (patch == nullptr) {
          return Error(kLiteRtStatusErrorCompilation,
                       "CUDA cache update plugin has no patch output");
        }
        auto* native = network_->addKVCacheUpdate(
            *inputs[i], *patch, *base, nvinfer1::KVCacheMode::kLINEAR);
        if (native == nullptr || native->getOutput(0) == nullptr) {
          return Error(kLiteRtStatusErrorCompilation,
                       "Failed to add native cache patch update");
        }
        auto* output = native->getOutput(0);
        raw_output_map_[op.Outputs()[i].Get()] = output;
        LITERT_RETURN_IF_ERROR(SetOutputTensor(op.Outputs()[i], output));
      }
      return {};
    }
    if (const auto layout = DecodeCacheUpdateMode(op);
        layout.has_value() &&
        in_place_caches_.count(op.Inputs()[3].Get()) != 0 &&
        in_place_caches_[op.Inputs()[3].Get()].op == op.Get()) {
      return LowerDecodeCompositeCacheUpdate(op, *layout);
    }
    LITERT_ASSIGN_OR_RETURN(auto* update_a, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(auto* update_b, GetTensor(op.Inputs()[1]));
    LITERT_ASSIGN_OR_RETURN(auto* start_a, GetTensor(op.Inputs()[5]));
    LITERT_ASSIGN_OR_RETURN(auto* start_b, GetTensor(op.Inputs()[6]));
    LITERT_ASSIGN_OR_RETURN(auto* cache_a, GetRawInt8Tensor(op.Inputs()[3]));
    LITERT_ASSIGN_OR_RETURN(auto* cache_b, GetRawInt8Tensor(op.Inputs()[4]));
    if (Int8CacheWriteEnabled()) {
      LITERT_ASSIGN_OR_RETURN(
          const bool written,
          LowerInt8CacheWrite(op, cache_a, cache_b, update_a, update_b, start_a,
                              start_b));
      if (written) {
        return {};
      }
    }
    LITERT_ASSIGN_OR_RETURN(auto* q_a,
                            AddQuantizeTensor(update_a, op.Outputs()[0],
                                              nvinfer1::DataType::kINT8));
    LITERT_ASSIGN_OR_RETURN(auto* q_b,
                            AddQuantizeTensor(update_b, op.Outputs()[1],
                                              nvinfer1::DataType::kINT8));
    auto* shuffle = network_->addShuffle(*q_b);
    if (shuffle == nullptr || shuffle->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add cache_update transpose");
    }
    nvinfer1::Permutation perm{};
    perm.order[0] = 0;
    perm.order[1] = 1;
    perm.order[2] = 3;
    perm.order[3] = 2;
    shuffle->setFirstTranspose(perm);
    LITERT_ASSIGN_OR_RETURN(auto* out_a,
                            BuildScatterUpdate(cache_a, q_a, start_a));
    LITERT_ASSIGN_OR_RETURN(
        auto* out_b,
        BuildScatterUpdate(cache_b, shuffle->getOutput(0), start_b));
    LITERT_RETURN_IF_ERROR(SetQuantizedScatterOutput(op.Outputs()[0], out_a));
    return SetQuantizedScatterOutput(op.Outputs()[1], out_b);
  }

  // The INT4 caches of the composites over quantized caches. TensorRT has no
  // INT4 tensors at an engine's boundary, so such a cache crosses it as the
  // INT8 bytes LiteRT packs it into, two values per byte: the same buffer,
  // declared [1, H, S, D / 2] for a key cache and [1, H, D / 2, S] for a
  // value cache. Only the quantized attention plugin reads or writes it.
  void FindPackedInt4Caches(const Subgraph& subgraph) {
    for (const auto& op : subgraph.Ops()) {
      if (op.Code() != kLiteRtOpCodeShloComposite) {
        continue;
      }
      const std::string name = CompositeOpName(op);
      const bool update =
          name == "odml.cache_update" && ParseQuantCacheUpdate(op).has_value();
      if (!update &&
          !(name == "odml.sdpa_transposed" && ParseQuantSdpa(op).has_value())) {
        continue;
      }
      for (int i = 0; i < 2; ++i) {
        const Tensor cache = op.Inputs()[(update ? 3 : 1) + i];
        if (cache.ElementType() != litert::ElementType::Int4) {
          continue;
        }
        packed_int4_caches_[cache.Get()] = i == 1;
        if (update) {
          packed_int4_caches_[op.Outputs()[i].Get()] = i == 1;
        }
      }
    }
  }

  // The op that produces `tensor`, if it has this code.
  static std::optional<Op> ProducerWithCode(const Tensor& tensor,
                                            LiteRtOpCode code) {
    if (tensor.IsSubgraphInput() || tensor.HasWeights()) {
      return std::nullopt;
    }
    const auto producer = tensor.GetDefiningOp();
    if (!producer || producer->Code() != code) {
      return std::nullopt;
    }
    return *producer;
  }

  // The float tensor behind a quantized input of a composite over quantized
  // caches: the input of the QUANTIZE that produces `tensor`, also through
  // one RESHAPE or TRANSPOSE of the quantized tensor (*shape_op). The plugin
  // rounds that float as the QUANTIZE does. Null without such a QUANTIZE.
  Expected<nvinfer1::ITensor*> FloatBeforeQuantize(
      const Tensor& tensor, std::optional<Op>* shape_op) {
    auto quantize = ProducerWithCode(tensor, kLiteRtOpCodeTflQuantize);
    if (!quantize.has_value()) {
      *shape_op = ProducerWithCode(tensor, kLiteRtOpCodeTflReshape);
      if (!shape_op->has_value()) {
        *shape_op = ProducerWithCode(tensor, kLiteRtOpCodeTflTranspose);
      }
      if (shape_op->has_value()) {
        quantize = ProducerWithCode((*shape_op)->Inputs()[0],
                                    kLiteRtOpCodeTflQuantize);
      }
    }
    if (!quantize.has_value() ||
        !IsFloatLike(quantize->Inputs()[0].ElementType())) {
      return nullptr;
    }
    return GetTensor(quantize->Inputs()[0]);
  }

  // `value` as a tensor of the activations' float type with `dims`.
  Expected<nvinfer1::ITensor*> AsQuantPluginFloat(nvinfer1::ITensor* value,
                                                  const nvinfer1::Dims& dims) {
    if (value->getType() == nvinfer1::DataType::kFLOAT) {
      LITERT_ASSIGN_OR_RETURN(value, AddCastTensor(value, ModeFloatType()));
    }
    if (value->getType() != nvinfer1::DataType::kHALF &&
        value->getType() != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "A quantized composite input has no float value");
    }
    const nvinfer1::Dims have = value->getDimensions();
    if (have.nbDims == dims.nbDims &&
        std::equal(have.d, have.d + have.nbDims, dims.d)) {
      return value;
    }
    return ReshapeTensor(value, dims);
  }

  // The query a composite takes quantized, as floats with `dims`: the float
  // before its QUANTIZE (a RESHAPE of the quantized query keeps the order of
  // its elements), or the query's own value, which float activations hold
  // dequantized.
  Expected<nvinfer1::ITensor*> QuantizedQueryAsFloat(
      const Tensor& tensor, const nvinfer1::Dims& dims) {
    std::optional<Op> shape_op;
    LITERT_ASSIGN_OR_RETURN(auto* value,
                            FloatBeforeQuantize(tensor, &shape_op));
    if (value == nullptr ||
        (shape_op.has_value() && shape_op->Code() != kLiteRtOpCodeTflReshape)) {
      LITERT_ASSIGN_OR_RETURN(value, GetTensor(tensor));
    }
    return AsQuantPluginFloat(value, dims);
  }

  // The keys or values of n tokens a composite takes quantized ([1, H, n, D],
  // or [1, H, D, n] with `columns`), as float rows [1, H, n, D]. The graphs
  // quantize the values as [1, n, H, D] and transpose the quantized tensor:
  // the plugin reads those rows as they are (*token_major), which spares the
  // transpose.
  Expected<nvinfer1::ITensor*> QuantizedRowsAsFloat(const Tensor& tensor,
                                                    bool columns, int32_t heads,
                                                    int32_t tokens,
                                                    int32_t depth,
                                                    bool* token_major) {
    *token_major = false;
    const nvinfer1::Dims rows{4, {1, heads, tokens, depth}};
    std::optional<Op> shape_op;
    LITERT_ASSIGN_OR_RETURN(auto* value,
                            FloatBeforeQuantize(tensor, &shape_op));
    if (value != nullptr && !shape_op.has_value() && !columns) {
      return AsQuantPluginFloat(value, rows);
    }
    if (value != nullptr && shape_op.has_value() &&
        shape_op->Code() == kLiteRtOpCodeTflReshape && tokens == 1) {
      // One token: rows and columns hold the same elements in the same order.
      return AsQuantPluginFloat(value, rows);
    }
    if (value != nullptr && shape_op.has_value() &&
        shape_op->Code() == kLiteRtOpCodeTflTranspose) {
      const auto perm = ReadInt32Constant(shape_op->Inputs()[1]);
      const std::vector<int32_t> from_token_rows =
          columns ? std::vector<int32_t>{0, 2, 3, 1}
                  : std::vector<int32_t>{0, 2, 1, 3};
      if (perm && *perm == from_token_rows) {
        *token_major = true;
        return AsQuantPluginFloat(value,
                                  nvinfer1::Dims{4, {1, tokens, heads, depth}});
      }
    }
    LITERT_ASSIGN_OR_RETURN(value, GetTensor(tensor));
    if (columns) {
      auto* layer = network_->addShuffle(*value);
      if (layer == nullptr || layer->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to transpose the columns of a quantized input");
      }
      layer->setFirstTranspose(nvinfer1::Permutation{{0, 1, 3, 2}});
      value = layer->getOutput(0);
    }
    return AsQuantPluginFloat(value, rows);
  }

  // The graphs follow odml.sdpa_transposed with a SELECT over its tokens:
  //   SELECT_V2(NOT_EQUAL(REDUCE_MAX(CAST(mask), context axis), 0),
  //             attention as [1, n, Hq, D], 0)
  // gives the tokens whose mask row sees no key zero. The quantized
  // attention plugin gives such rows zero itself (ParseQuantSdpa), so over
  // its result the SELECT is the identity, and the reduction of the mask, a
  // [tokens, context] fp32 tensor for a prefill chunk, is never built.
  // Returns the SELECT's attention input when `op` is such a SELECT.
  static std::optional<Tensor> QuantAttentionBehindMaskSelect(const Op& op) {
    if (op.Inputs().size() != 3) {
      return std::nullopt;
    }
    const auto zero = ReadFloatScalar(op.Inputs()[2]);
    if (!zero || *zero != 0.0f) {
      return std::nullopt;
    }
    // The attention [1, Hq, n, D] as [1, n, Hq, D]: a reshape for one token.
    const std::vector<int32_t> tokens_first = {0, 2, 1, 3};
    const auto as_tokens_first = [&](const Tensor& tensor,
                                     int32_t tokens) -> std::optional<Tensor> {
      if (tokens == 1) {
        if (const auto reshape =
                ProducerWithCode(tensor, kLiteRtOpCodeTflReshape)) {
          return reshape->Inputs()[0];
        }
      }
      const auto transpose =
          ProducerWithCode(tensor, kLiteRtOpCodeTflTranspose);
      if (!transpose) {
        return std::nullopt;
      }
      const auto perm = ReadInt32Constant(transpose->Inputs()[1]);
      if (!perm || *perm != tokens_first) {
        return std::nullopt;
      }
      return transpose->Inputs()[0];
    };
    const Tensor then = op.Inputs()[1];
    auto then_type = then.RankedTensorType();
    if (!then_type || then_type->Layout().Rank() != 4) {
      return std::nullopt;
    }
    const int32_t tokens = then_type->Layout().Dimensions()[1];
    const auto attention = as_tokens_first(then, tokens);
    if (!attention) {
      return std::nullopt;
    }
    const auto sdpa = ProducerWithCode(*attention, kLiteRtOpCodeShloComposite);
    if (!sdpa || CompositeOpName(*sdpa) != "odml.sdpa_transposed") {
      return std::nullopt;
    }
    const auto form = ParseQuantSdpa(*sdpa);
    if (!form.has_value() || form->tokens != tokens ||
        !TensorHasShape(then, {1, tokens, form->group * form->caches.heads,
                               form->caches.depth}) ||
        !TensorHasShape(op.Inputs()[0], {1, tokens, 1, 1})) {
      return std::nullopt;
    }
    // The condition: whether the token's row of the composite's mask, as
    // [1, n, 1, S], has a key.
    const auto not_equal =
        ProducerWithCode(op.Inputs()[0], kLiteRtOpCodeTflNotEqual);
    if (!not_equal) {
      return std::nullopt;
    }
    const auto no_key = ReadFloatScalar(not_equal->Inputs()[1]);
    const auto reduce =
        ProducerWithCode(not_equal->Inputs()[0], kLiteRtOpCodeTflReduceMax);
    if (!no_key || *no_key != 0.0f || !reduce) {
      return std::nullopt;
    }
    const auto axes = ReadInt32Constant(reduce->Inputs()[1]);
    const auto cast =
        ProducerWithCode(reduce->Inputs()[0], kLiteRtOpCodeTflCast);
    if (!axes || axes->size() != 1 || ((*axes)[0] != 3 && (*axes)[0] != -1) ||
        !cast ||
        !TensorHasShape(cast->Inputs()[0], {1, tokens, 1, form->caches.seq})) {
      return std::nullopt;
    }
    const Tensor mask = sdpa->Inputs()[3];
    const Tensor rows = cast->Inputs()[0];
    if (rows.Get() != mask.Get()) {
      // A reshape of [1, 1, n, S] to [1, n, 1, S] keeps the rows' order.
      auto moved = ProducerWithCode(rows, kLiteRtOpCodeTflReshape);
      if (!moved) {
        const auto transposed = as_tokens_first(rows, /*tokens=*/0);
        if (!transposed || transposed->Get() != mask.Get()) {
          return std::nullopt;
        }
      } else if (moved->Inputs()[0].Get() != mask.Get()) {
        return std::nullopt;
      }
    }
    return then;
  }

  // A layer of the quantized attention plugin; `name` names it in the engine.
  Expected<nvinfer1::IPluginV3Layer*> AddQuantAttentionLayer(
      const QuantAttentionOptions& options,
      std::vector<nvinfer1::ITensor*> inputs, const char* name) {
    for (const auto* input : inputs) {
      if (input == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "A quantized cache of the composite is not in the "
                     "network");
      }
    }
    TrtPtr<nvinfer1::IPluginV3> plugin(CreateQuantAttentionPlugin(options));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the CUDA quantized attention plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(*network_, inputs.data(),
                                                     inputs.size(), *plugin);
    if (layer == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the CUDA quantized attention plugin layer");
    }
    layer->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    return layer;
  }

  // An updated cache of a composite over quantized caches: the raw tensor is
  // the graph's output and what the attention composites read.
  Expected<void> SetQuantCacheOutput(const Tensor& tensor,
                                     nvinfer1::ITensor* out) {
    if (out == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "The quantized attention plugin has no cache output");
    }
    raw_output_map_[tensor.Get()] = out;
    return SetOutputTensor(tensor, out);
  }

  // Whether decode attentions alone read the caches a one-token
  // odml.cache_update writes: every use of both results is an
  // odml.sdpa_transposed that the plugin's decode kernel runs over exactly
  // these caches. The first of them to be lowered then writes the token in
  // its own launches (QuantAttentionMode::kUpdateAttention), and the update
  // needs none.
  static bool QuantCacheUpdateJoinsAttention(const Op& op,
                                             const QuantCacheUpdate& form) {
    if (form.tokens != 1) {
      return false;
    }
    const auto& caches = form.caches;
    for (int i = 0; i < 2; ++i) {
      const auto uses = op.Outputs()[i].Uses();
      if (uses.empty()) {
        return false;
      }
      for (const auto& use : uses) {
        const Op& reader = use.user;
        if (reader.Code() != kLiteRtOpCodeShloComposite ||
            CompositeOpName(reader) != "odml.sdpa_transposed") {
          return false;
        }
        const auto sdpa = ParseQuantSdpa(reader);
        if (!sdpa.has_value() ||
            reader.Inputs()[1].Get() != op.Outputs()[0].Get() ||
            reader.Inputs()[2].Get() != op.Outputs()[1].Get() ||
            !LiteRtNvidiaQuantDecodeAttentionUpdateSupports(
                sdpa->group * sdpa->tokens, caches.depth, caches.bits,
                caches.seq)) {
          return false;
        }
      }
    }
    return true;
  }

  // The token of an odml.cache_update as the plugin takes it: the keys and
  // the values before their QUANTIZE and the two start indices.
  struct QuantCacheToken {
    nvinfer1::ITensor* keys = nullptr;
    nvinfer1::ITensor* values = nullptr;
    nvinfer1::ITensor* start_k = nullptr;
    nvinfer1::ITensor* start_v = nullptr;
    bool v_token_major = false;
  };

  Expected<QuantCacheToken> QuantCacheUpdateToken(
      const Op& op, const QuantCacheUpdate& form) {
    const auto& caches = form.caches;
    QuantCacheToken token;
    bool token_major = false;
    LITERT_ASSIGN_OR_RETURN(
        token.keys,
        QuantizedRowsAsFloat(op.Inputs()[0], /*columns=*/false, caches.heads,
                             form.tokens, caches.depth, &token_major));
    if (token_major) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Quantized cache update keys must be rows of heads");
    }
    LITERT_ASSIGN_OR_RETURN(
        token.values,
        QuantizedRowsAsFloat(op.Inputs()[1], /*columns=*/false, caches.heads,
                             form.tokens, caches.depth, &token.v_token_major));
    LITERT_ASSIGN_OR_RETURN(token.start_k, GetTensor(op.Inputs()[5]));
    LITERT_ASSIGN_OR_RETURN(token.start_v, GetTensor(op.Inputs()[6]));
    return token;
  }

  // odml.cache_update over quantized caches (ParseQuantCacheUpdate): one
  // plugin launch quantizes the step's keys and values and writes them into
  // both caches at the start indices. A decode block's update waits for the
  // attention that reads its caches instead (QuantCacheUpdateJoinsAttention).
  Expected<void> LowerQuantCacheUpdate(const Op& op,
                                       const QuantCacheUpdate& form) {
    if (QuantCacheUpdateJoinsAttention(op, form)) {
      joined_quant_cache_updates_.emplace(op.Outputs()[0].Get(), op);
      return {};
    }
    return AddQuantCacheUpdate(op, form);
  }

  // The cache updates no attention took (their readers were not lowered as
  // such): a layer of their own, before the graph's outputs are marked.
  Expected<void> AddUnjoinedQuantCacheUpdates() {
    // In the graph's order: the layers' names do not depend on the map's.
    std::vector<Op> left;
    for (const auto& [cache, op] : joined_quant_cache_updates_) {
      left.push_back(op);
    }
    joined_quant_cache_updates_.clear();
    std::sort(left.begin(), left.end(), [this](const Op& a, const Op& b) {
      return op_order_[a.Get()] < op_order_[b.Get()];
    });
    for (const Op& op : left) {
      const auto form = ParseQuantCacheUpdate(op);
      if (!form.has_value()) {
        return Error(kLiteRtStatusErrorCompilation,
                     "A deferred quantized cache update no longer parses");
      }
      LITERT_RETURN_IF_ERROR(AddQuantCacheUpdate(op, *form));
    }
    return {};
  }

  Expected<void> AddQuantCacheUpdate(const Op& op,
                                     const QuantCacheUpdate& form) {
    const auto& caches = form.caches;
    LITERT_ASSIGN_OR_RETURN(const QuantCacheToken token,
                            QuantCacheUpdateToken(op, form));
    QuantAttentionOptions options;
    options.mode = QuantAttentionMode::kCacheUpdate;
    options.quant = {1.0f, caches.k_scale, caches.v_scale, caches.bits};
    options.v_token_major = token.v_token_major;
    LITERT_ASSIGN_OR_RETURN(
        auto* layer, AddQuantAttentionLayer(
                         options,
                         {raw_input_map_[op.Inputs()[3].Get()],
                          raw_input_map_[op.Inputs()[4].Get()], token.keys,
                          token.values, token.start_k, token.start_v},
                         "cuda_quant_cache_update"));
    LITERT_RETURN_IF_ERROR(
        SetQuantCacheOutput(op.Outputs()[0], layer->getOutput(0)));
    return SetQuantCacheOutput(op.Outputs()[1], layer->getOutput(1));
  }

  // odml.sdpa_transposed over quantized caches (ParseQuantSdpa): the query
  // heads of a KV head become its rows, head-major, which is the order the
  // plugin's mask rows repeat in.
  Expected<void> LowerQuantSdpa(const Op& op, const QuantSdpa& form) {
    const auto& caches = form.caches;
    const nvinfer1::Dims grouped{
        4, {1, caches.heads, form.group * form.tokens, caches.depth}};
    LITERT_ASSIGN_OR_RETURN(auto* q,
                            QuantizedQueryAsFloat(op.Inputs()[0], grouped));
    LITERT_ASSIGN_OR_RETURN(auto* mask, GetTensor(op.Inputs()[3]));
    QuantAttentionOptions options;
    options.mode = QuantAttentionMode::kAttention;
    options.quant = {form.q_scale, caches.k_scale, caches.v_scale, caches.bits};
    nvinfer1::IPluginV3Layer* layer = nullptr;
    if (const auto joined =
            joined_quant_cache_updates_.find(op.Inputs()[1].Get());
        joined != joined_quant_cache_updates_.end()) {
      // The first attention over the caches of a decode block's update
      // writes its token: the layer's cache outputs are the updated caches,
      // which the graph returns and the later attentions read.
      const Op update = joined->second;
      joined_quant_cache_updates_.erase(joined);
      const auto update_form = ParseQuantCacheUpdate(update);
      if (!update_form.has_value()) {
        return Error(kLiteRtStatusErrorCompilation,
                     "A deferred quantized cache update no longer parses");
      }
      LITERT_ASSIGN_OR_RETURN(const QuantCacheToken token,
                              QuantCacheUpdateToken(update, *update_form));
      options.mode = QuantAttentionMode::kUpdateAttention;
      options.v_token_major = token.v_token_major;
      LITERT_ASSIGN_OR_RETURN(
          layer, AddQuantAttentionLayer(
                     options,
                     {q, raw_input_map_[update.Inputs()[3].Get()],
                      raw_input_map_[update.Inputs()[4].Get()], mask,
                      token.keys, token.values, token.start_k, token.start_v},
                     "cuda_quant_update_attention"));
      LITERT_RETURN_IF_ERROR(
          SetQuantCacheOutput(update.Outputs()[0], layer->getOutput(1)));
      LITERT_RETURN_IF_ERROR(
          SetQuantCacheOutput(update.Outputs()[1], layer->getOutput(2)));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          layer,
          AddQuantAttentionLayer(options,
                                 {q, RawInt8Cache(op.Inputs()[1].Get()),
                                  RawInt8Cache(op.Inputs()[2].Get()), mask},
                                 "cuda_quant_attention"));
    }
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    LITERT_ASSIGN_OR_RETURN(auto* out,
                            ReshapeTensor(layer->getOutput(0), output_dims));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  // odml.fused_sdpa_cache_update (ParseQuantFusedSdpa): one plugin layer
  // attends to the ring caches and the chunk's keys, then writes the chunk
  // into the ring.
  Expected<void> LowerQuantFusedSdpa(const Op& op, const QuantFusedSdpa& form) {
    const auto& caches = form.caches;
    LITERT_ASSIGN_OR_RETURN(
        auto* q,
        QuantizedQueryAsFloat(
            op.Inputs()[0],
            nvinfer1::Dims{4, {1, caches.heads, form.rows, caches.depth}}));
    bool token_major = false;
    LITERT_ASSIGN_OR_RETURN(
        auto* keys,
        QuantizedRowsAsFloat(op.Inputs()[3], /*columns=*/false, caches.heads,
                             form.tokens, caches.depth, &token_major));
    if (token_major) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Fused attention keys must be rows of heads");
    }
    LITERT_ASSIGN_OR_RETURN(
        auto* values,
        QuantizedRowsAsFloat(op.Inputs()[4], /*columns=*/true, caches.heads,
                             form.tokens, caches.depth, &token_major));
    LITERT_ASSIGN_OR_RETURN(auto* mask, GetTensor(op.Inputs()[5]));
    LITERT_ASSIGN_OR_RETURN(auto* params, GetTensor(op.Inputs()[6]));
    QuantAttentionOptions options;
    options.mode = QuantAttentionMode::kFusedAttentionUpdate;
    options.quant = {form.q_scale, caches.k_scale, caches.v_scale, caches.bits};
    options.v_token_major = token_major;
    options.update_cache = form.update_cache;
    LITERT_ASSIGN_OR_RETURN(
        auto* layer,
        AddQuantAttentionLayer(
            options,
            {q, raw_input_map_[op.Inputs()[1].Get()],
             raw_input_map_[op.Inputs()[2].Get()], mask, keys, values, params},
            "cuda_quant_fused_attention"));
    LITERT_RETURN_IF_ERROR(
        SetOutputTensor(op.Outputs()[0], layer->getOutput(0)));
    LITERT_RETURN_IF_ERROR(
        SetQuantCacheOutput(op.Outputs()[1], layer->getOutput(1)));
    return SetQuantCacheOutput(op.Outputs()[2], layer->getOutput(2));
  }

  // The caches of the general forms (ParseGeneral*): packed INT4 caches as
  // FindPackedInt4Caches declares them, UINT8 caches as INT8 tensors of
  // their bytes. Only the general plugin reads or writes them.
  void FindGeneralCaches(const Subgraph& subgraph) {
    if (!GeneralAttentionEnabled()) {
      return;
    }
    const auto note = [this](const Tensor& cache, bool value) {
      if (cache.ElementType() == litert::ElementType::Int4) {
        packed_int4_caches_[cache.Get()] = value;
      } else if (cache.ElementType() == litert::ElementType::UInt8) {
        byte_caches_.insert(cache.Get());
      }
    };
    for (const auto& op : subgraph.Ops()) {
      if (op.Code() != kLiteRtOpCodeShloComposite) {
        continue;
      }
      const std::string name = CompositeOpName(op);
      if (name == "odml.cache_update" &&
          ParseGeneralCacheUpdate(op).has_value()) {
        for (int i = 0; i < 2; ++i) {
          note(op.Inputs()[3 + i], i == 1);
          note(op.Outputs()[i], i == 1);
        }
      } else if (name == "odml.sdpa_transposed" &&
                 ParseGeneralSdpa(op).has_value()) {
        for (int i = 0; i < 2; ++i) {
          if (op.Inputs()[1 + i].IsSubgraphInput()) {
            note(op.Inputs()[1 + i], i == 1);
          }
        }
      } else if (name == "odml.fused_sdpa_cache_update" &&
                 ParseGeneralFusedSdpa(op).has_value()) {
        for (int i = 0; i < 2; ++i) {
          note(op.Inputs()[1 + i], i == 1);
          if (op.Outputs().size() == 3) note(op.Outputs()[1 + i], i == 1);
        }
      }
    }
  }

  // A layer of the general attention plugin.
  Expected<nvinfer1::IPluginV3Layer*> AddGeneralAttentionLayer(
      const GeneralAttentionOptions& options,
      std::vector<nvinfer1::ITensor*> inputs, const char* name) {
    for (const auto* input : inputs) {
      if (input == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "A cache of a general attention composite is not in "
                     "the network");
      }
    }
    TrtPtr<nvinfer1::IPluginV3> plugin(CreateGeneralAttentionPlugin(options));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the general attention plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(*network_, inputs.data(),
                                                     inputs.size(), *plugin);
    if (layer == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the general attention plugin layer");
    }
    layer->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    return layer;
  }

  static GeneralAttentionOptions GeneralOptions(const GeneralCaches& caches,
                                                GeneralAttentionMode mode) {
    GeneralAttentionOptions options;
    options.mode = mode;
    options.cache_type = caches.type;
    options.k_scale = caches.k_scale;
    options.v_scale = caches.v_scale;
    return options;
  }

  // odml.cache_update in a general form (ParseGeneralCacheUpdate).
  Expected<void> LowerGeneralCacheUpdate(const Op& op,
                                         const GeneralCacheUpdate& form) {
    const auto& caches = form.caches;
    bool keys_major = false;
    bool values_major = false;
    LITERT_ASSIGN_OR_RETURN(
        auto* keys,
        QuantizedRowsAsFloat(op.Inputs()[0], /*columns=*/false, caches.heads,
                             form.tokens, caches.depth, &keys_major));
    LITERT_ASSIGN_OR_RETURN(
        auto* values,
        QuantizedRowsAsFloat(op.Inputs()[1], /*columns=*/false, caches.heads,
                             form.tokens, caches.depth, &values_major));
    if (keys_major) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Cache update keys must be rows of heads");
    }
    if (values->getType() != keys->getType()) {
      LITERT_ASSIGN_OR_RETURN(values, AddCastTensor(values, keys->getType()));
    }
    LITERT_ASSIGN_OR_RETURN(auto* start_k, GetTensor(op.Inputs()[5]));
    LITERT_ASSIGN_OR_RETURN(auto* start_v, GetTensor(op.Inputs()[6]));
    GeneralAttentionOptions options =
        GeneralOptions(caches, GeneralAttentionMode::kCacheUpdate);
    options.new_int8_scale = form.src_int8_scale;
    options.v_token_major = values_major;
    LITERT_ASSIGN_OR_RETURN(
        auto* layer,
        AddGeneralAttentionLayer(options,
                                 {raw_input_map_[op.Inputs()[3].Get()],
                                  raw_input_map_[op.Inputs()[4].Get()], keys,
                                  values, start_k, start_v},
                                 "cuda_general_cache_update"));
    LITERT_RETURN_IF_ERROR(
        SetQuantCacheOutput(op.Outputs()[0], layer->getOutput(0)));
    return SetQuantCacheOutput(op.Outputs()[1], layer->getOutput(1));
  }

  // odml.sdpa_transposed in a general form (ParseGeneralSdpa): the query
  // heads of a KV head as its rows, head-major.
  Expected<void> LowerGeneralSdpa(const Op& op, const GeneralSdpa& form) {
    const auto& caches = form.caches;
    const nvinfer1::Dims grouped{
        4, {1, caches.heads, form.group * form.tokens, caches.depth}};
    LITERT_ASSIGN_OR_RETURN(auto* q,
                            QuantizedQueryAsFloat(op.Inputs()[0], grouped));
    LITERT_ASSIGN_OR_RETURN(auto* mask, GetQsaMask(op.Inputs()[3]));
    GeneralAttentionOptions options =
        GeneralOptions(caches, GeneralAttentionMode::kAttention);
    options.q_int8_scale = form.q_int8_scale;
    options.scale = form.attributes.scale;
    options.softcap = form.attributes.softcap;
    LITERT_ASSIGN_OR_RETURN(
        auto* layer,
        AddGeneralAttentionLayer(options,
                                 {q, RawInt8Cache(op.Inputs()[1].Get()),
                                  RawInt8Cache(op.Inputs()[2].Get()), mask},
                                 "cuda_general_attention"));
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    LITERT_ASSIGN_OR_RETURN(auto* out,
                            ReshapeTensor(layer->getOutput(0), output_dims));
    return SetOutputTensor(op.Outputs()[0], out);
  }

  // odml.fused_sdpa_cache_update in a general form (ParseGeneralFusedSdpa).
  Expected<void> LowerGeneralFusedSdpa(const Op& op,
                                       const GeneralFusedSdpa& form) {
    const auto& caches = form.caches;
    LITERT_ASSIGN_OR_RETURN(
        auto* q,
        QuantizedQueryAsFloat(
            op.Inputs()[0],
            nvinfer1::Dims{4, {1, caches.heads, form.rows, caches.depth}}));
    bool keys_major = false;
    bool values_major = false;
    LITERT_ASSIGN_OR_RETURN(
        auto* keys,
        QuantizedRowsAsFloat(op.Inputs()[3], /*columns=*/false, caches.heads,
                             form.tokens, caches.depth, &keys_major));
    LITERT_ASSIGN_OR_RETURN(
        auto* values,
        QuantizedRowsAsFloat(op.Inputs()[4], /*columns=*/true, caches.heads,
                             form.tokens, caches.depth, &values_major));
    if (keys_major) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Fused attention keys must be rows of heads");
    }
    if (values->getType() != keys->getType()) {
      LITERT_ASSIGN_OR_RETURN(values, AddCastTensor(values, keys->getType()));
    }
    LITERT_ASSIGN_OR_RETURN(auto* mask, GetQsaMask(op.Inputs()[5]));
    LITERT_ASSIGN_OR_RETURN(auto* params, GetTensor(op.Inputs()[6]));
    GeneralAttentionOptions options =
        GeneralOptions(caches, GeneralAttentionMode::kFusedAttention);
    options.q_int8_scale = form.q_int8_scale;
    options.new_int8_scale = form.new_int8_scale;
    options.scale = form.attributes.scale;
    options.softcap = form.attributes.softcap;
    options.v_token_major = values_major;
    options.update_cache = form.update_cache;
    LITERT_ASSIGN_OR_RETURN(
        auto* layer,
        AddGeneralAttentionLayer(
            options,
            {q, raw_input_map_[op.Inputs()[1].Get()],
             raw_input_map_[op.Inputs()[2].Get()], mask, keys, values, params},
            "cuda_general_fused_attention"));
    LITERT_RETURN_IF_ERROR(
        SetOutputTensor(op.Outputs()[0], layer->getOutput(0)));
    if (!form.update_cache) {
      return {};
    }
    LITERT_RETURN_IF_ERROR(
        SetQuantCacheOutput(op.Outputs()[1], layer->getOutput(1)));
    return SetQuantCacheOutput(op.Outputs()[2], layer->getOutput(2));
  }

  // odml.runtime_bmm(a, cache, positions) = a @ dequantize(cache)^T over the
  // full context (window-equivalent; see IsCompositeSupported).
  Expected<void> LowerCompositeRuntimeBmm(const Op& op) {
    LITERT_ASSIGN_OR_RETURN(auto* a, GetTensor(op.Inputs()[0]));
    nvinfer1::ITensor* cache = nullptr;
    if (const auto scores = windowed_by_scores_.find(op.Get());
        scores != windowed_by_scores_.end()) {
      LITERT_ASSIGN_OR_RETURN(cache,
                              WindowKeySlice(windowed_reads_[scores->second]));
    } else if (const auto values = windowed_by_values_.find(op.Get());
               values != windowed_by_values_.end()) {
      LITERT_ASSIGN_OR_RETURN(
          cache, WindowValueSlice(windowed_reads_[values->second]));
    } else {
      LITERT_ASSIGN_OR_RETURN(cache, GetTensor(op.Inputs()[1]));
    }
    LITERT_ASSIGN_OR_RETURN(auto output_type,
                            op.Outputs()[0].RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto output_dims, ConvertDims(output_type));
    const auto a_dims = a->getDimensions();
    const auto cache_dims = cache->getDimensions();
    nvinfer1::ITensor* padding = nullptr;
    const size_t context_limit = RuntimeBmmContextLimit();
    if (context_limit > 0 && a_dims.nbDims >= 2 &&
        cache_dims.nbDims == a_dims.nbDims &&
        output_dims.nbDims == a_dims.nbDims &&
        context_limit <= static_cast<size_t>(std::numeric_limits<int>::max())) {
      const int rank = a_dims.nbDims;
      const int limit = static_cast<int>(context_limit);
      const int a_context_axis = rank - 1;
      const int cache_context_axis = rank - 2;
      const int cache_transposed_context_axis = rank - 1;
      const bool score_bmm =
          cache_dims.d[cache_context_axis] > limit &&
          output_dims.d[rank - 1] == cache_dims.d[cache_context_axis] &&
          a_dims.d[rank - 1] == cache_dims.d[rank - 1];
      const bool value_bmm =
          cache_dims.d[cache_transposed_context_axis] > limit &&
          a_dims.d[a_context_axis] ==
              cache_dims.d[cache_transposed_context_axis] &&
          output_dims.d[rank - 1] == cache_dims.d[cache_context_axis];
      if (score_bmm) {
        LITERT_ASSIGN_OR_RETURN(
            cache, SliceTensorStatic(cache, cache_context_axis,
                                     /*start_index=*/0, limit,
                                     "runtime_bmm_score_cache_prefix"));
        nvinfer1::Dims padding_dims = output_dims;
        padding_dims.d[rank - 1] -= limit;
        if (padding_dims.d[rank - 1] > 0) {
          LITERT_ASSIGN_OR_RETURN(
              padding, AddZeroFloatTensor(padding_dims, a->getType(),
                                          "runtime_bmm_score_tail_padding"));
        }
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT runtime_bmm score prefix limit=%d "
                   "full_context=%d",
                   limit, output_dims.d[rank - 1]);
      } else if (value_bmm) {
        LITERT_ASSIGN_OR_RETURN(
            a, SliceTensorStatic(a, a_context_axis, /*start_index=*/0, limit,
                                 "runtime_bmm_value_probs_prefix"));
        LITERT_ASSIGN_OR_RETURN(
            cache, SliceTensorStatic(cache, cache_transposed_context_axis,
                                     /*start_index=*/0, limit,
                                     "runtime_bmm_value_cache_prefix"));
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT runtime_bmm value prefix limit=%d "
                   "full_context=%d",
                   limit, cache_dims.d[cache_transposed_context_axis]);
      }
    }
    if (cache->getType() == nvinfer1::DataType::kINT8) {
      LITERT_ASSIGN_OR_RETURN(
          cache, AddDequantizeTensor(op.Inputs()[1], cache, a->getType()));
    }
    const bool cache_precision =
        RuntimeBmmInCachePrecision() && Fp16ActivationsEnabled() &&
        (cache->getType() == nvinfer1::DataType::kHALF ||
         cache->getType() == nvinfer1::DataType::kBF16);
    const auto activation_type =
        cache_precision ? ModeFloatType() : a->getType();
    if (cache_precision) {
      LITERT_ASSIGN_OR_RETURN(a, AddCastTensor(a, cache->getType()));
    } else {
      LITERT_ASSIGN_OR_RETURN(cache, AddCastTensor(cache, activation_type));
    }
    // A value cache held as [B, H, S, D] is already the right operand of
    // probs @ V; the model's [B, H, D, S] layout needs the transpose.
    const bool transposed_cache =
        transposed_caches_.find(op.Inputs()[1].Get()) !=
        transposed_caches_.end();
    auto* layer = network_->addMatrixMultiply(
        *a, nvinfer1::MatrixOperation::kNONE, *cache,
        transposed_cache ? nvinfer1::MatrixOperation::kNONE
                         : nvinfer1::MatrixOperation::kTRANSPOSE);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add runtime_bmm matmul");
    }
    nvinfer1::ITensor* out = layer->getOutput(0);
    LITERT_ASSIGN_OR_RETURN(out, AddCastTensor(out, activation_type));
    if (padding != nullptr) {
      std::array<nvinfer1::ITensor*, 2> inputs = {out, padding};
      auto* concat = network_->addConcatenation(inputs.data(), inputs.size());
      if (concat == nullptr || concat->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add runtime_bmm score padding concatenation");
      }
      concat->setAxis(output_dims.nbDims - 1);
      out = concat->getOutput(0);
    }
    return SetOutputTensor(op.Outputs()[0], out);
  }

  // The router of a mixture-of-experts block as the graph expresses it,
  // starting from the moe op's inputs:
  //   top_indices  <- [reshape] <- topk(softmax(fc(router_in, W_router))).1
  //   top_weights  <- [reshape] <- div(topk.0, sum(topk.0)) | topk.0
  //   router_in    <- [mul c]* <- [reshape] <- rms_norm(X, w_router)
  //   x            <- [reshape] <- rms_norm(X, w_experts)
  // with one residual X; or, as the export with a fused feed-forward selects
  // before it normalizes,
  //   top_indices  <- topk(fc(router_in, W_router)).1
  //   top_weights  <- softmax(topk.0)
  // The fused plugin takes X and does all of it.
  struct MoeRouting {
    Tensor residual;
    Tensor router_norm;
    Tensor router_weights;
    Tensor expert_norm;
    float multiplier = 1.0f;
    float eps = 1e-6f;
  };

  static Tensor StripReshapes(Tensor tensor) {
    for (int hops = 0; hops < 4; ++hops) {
      auto producer = tensor.GetDefiningOp();
      if (!producer || producer->Code() != kLiteRtOpCodeTflReshape ||
          producer->Inputs().empty()) {
        break;
      }
      tensor = producer->Inputs()[0];
    }
    return tensor;
  }

  static std::optional<Op> NormOf(const Tensor& tensor) {
    auto producer = StripReshapes(tensor).GetDefiningOp();
    if (!producer || producer->Code() != kLiteRtOpCodeShloComposite ||
        CompositeOpName(*producer) != "odml.rms_norm" ||
        producer->Inputs().size() != 2 || producer->Outputs().size() != 1) {
      return std::nullopt;
    }
    return *producer;
  }

  std::optional<MoeRouting> FindMoeRouting(const Op& op,
                                           const MoeCustomOp& moe) {
    if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_FUSE_MOE_ROUTER",
                    /*default_value=*/true)) {
      return std::nullopt;
    }
    MoeRouting routing;
    // The expert norm.
    const auto expert_norm = NormOf(op.Inputs()[0]);
    if (!expert_norm.has_value()) return std::nullopt;
    routing.residual = expert_norm->Inputs()[0];
    routing.expert_norm = expert_norm->Inputs()[1];
    CompositeOptions options;
    if (options.InitFromOp(expert_norm->Get()) == kLiteRtStatusOk &&
        options.attributes_map.has_value() &&
        !options.attributes_map.value()["epsilon"].IsNull()) {
      routing.eps = options.attributes_map.value()["epsilon"].AsFloat();
    }
    // The top-k whose indices the op takes.
    const Tensor indices = StripReshapes(op.Inputs()[2]);
    auto topk = indices.GetDefiningOp();
    if (!topk || topk->Code() != kLiteRtOpCodeTflTopkV2 ||
        topk->Inputs().size() != 2 || topk->Outputs().size() != 2 ||
        topk->Outputs()[1].Get() != indices.Get()) {
      return std::nullopt;
    }
    const auto k = ReadInt32Constant(topk->Inputs()[1]);
    if (!k || k->size() != 1 || (*k)[0] != moe.top_k) return std::nullopt;
    const auto is_plain_softmax = [](const Op& candidate) {
      float beta = 1.0f;
      return candidate.Code() == kLiteRtOpCodeTflSoftmax &&
             !candidate.Inputs().empty() &&
             LiteRtGetSoftmaxBetaOption(candidate.Get(), &beta) ==
                 kLiteRtStatusOk &&
             beta == 1.0f;
    };
    const Tensor weights = StripReshapes(op.Inputs()[1]);
    const auto normalized = weights.GetDefiningOp();
    if (!normalized) return std::nullopt;
    Tensor logits = topk->Inputs()[0];
    if (is_plain_softmax(*normalized)) {
      // The softmax of the selected logits: the same weights as the selected
      // probabilities over their sum.
      if (StripReshapes(normalized->Inputs()[0]).Get() !=
          topk->Outputs()[0].Get()) {
        return std::nullopt;
      }
    } else {
      // Its values, renormalized by their sum.
      if (normalized->Code() != kLiteRtOpCodeTflDiv ||
          normalized->Inputs().size() != 2 ||
          !HasNoFusedActivation(*normalized) ||
          StripReshapes(normalized->Inputs()[0]).Get() !=
              topk->Outputs()[0].Get()) {
        return std::nullopt;
      }
      auto sum = StripReshapes(normalized->Inputs()[1]).GetDefiningOp();
      if (!sum || sum->Code() != kLiteRtOpCodeTflSum || sum->Inputs().empty() ||
          StripReshapes(sum->Inputs()[0]).Get() != topk->Outputs()[0].Get()) {
        return std::nullopt;
      }
      // softmax(fc(router_in, W)) with a plain softmax.
      auto softmax = StripReshapes(logits).GetDefiningOp();
      if (!softmax || !is_plain_softmax(*softmax)) return std::nullopt;
      logits = softmax->Inputs()[0];
    }
    auto fc = StripReshapes(logits).GetDefiningOp();
    if (!fc || fc->Code() != kLiteRtOpCodeTflFullyConnected ||
        fc->Inputs().size() != 2 || !HasNoFusedActivation(*fc) ||
        !IsBlockwiseInt4Weights(fc->Inputs()[1])) {
      return std::nullopt;
    }
    auto w_type = fc->Inputs()[1].RankedTensorType();
    if (!w_type || w_type->Layout().Dimensions()[0] != moe.experts ||
        w_type->Layout().Dimensions()[1] != moe.model_dim) {
      return std::nullopt;
    }
    routing.router_weights = fc->Inputs()[1];
    // The router norm behind the scalar multiplies.
    Tensor value = fc->Inputs()[0];
    for (int hops = 0; hops < 4; ++hops) {
      value = StripReshapes(value);
      auto producer = value.GetDefiningOp();
      if (!producer || producer->Code() != kLiteRtOpCodeTflMul ||
          producer->Inputs().size() != 2 || !HasNoFusedActivation(*producer)) {
        break;
      }
      std::optional<float> scalar;
      int other = -1;
      for (int i = 0; i < 2; ++i) {
        if (auto constant = ReadFloatScalar(producer->Inputs()[i])) {
          scalar = *constant;
          other = 1 - i;
        }
      }
      if (!scalar.has_value()) return std::nullopt;
      routing.multiplier *= *scalar;
      value = producer->Inputs()[other];
    }
    const auto router_norm = NormOf(value);
    if (!router_norm.has_value() ||
        router_norm->Inputs()[0].Get() != routing.residual.Get()) {
      return std::nullopt;
    }
    routing.router_norm = router_norm->Inputs()[1];
    for (const Tensor* w : {&routing.router_norm, &routing.expert_norm}) {
      if (w->ElementType() != litert::ElementType::Float32 ||
          !w->HasWeights() ||
          w->Weights().Bytes().size() !=
              static_cast<size_t>(moe.model_dim) * sizeof(float)) {
        return std::nullopt;
      }
    }
    return routing;
  }

  // The one op reading `tensor`, through reshapes (collected in `through`);
  // nullopt when anything else reads it or it is a graph output.
  std::optional<Op> SoleUser(Tensor tensor, std::vector<LiteRtOp>* through) {
    for (int hops = 0; hops < 4; ++hops) {
      if (graph_outputs_.count(tensor.Get()) != 0) return std::nullopt;
      const auto uses = tensor.Uses();
      if (uses.size() != 1) return std::nullopt;
      Op user = uses[0].user;
      if (user.Code() != kLiteRtOpCodeTflReshape) return user;
      if (user.Outputs().size() != 1) return std::nullopt;
      through->push_back(user.Get());
      tensor = user.Outputs()[0];
    }
    return std::nullopt;
  }

  bool ReadOnce(const Tensor& tensor) const {
    return graph_outputs_.count(tensor.Get()) == 0 && tensor.Uses().size() == 1;
  }

  // The rest of a layer's feed-forward block around a routed moe op, as
  // Gemma 4 lays it out (X the residual the router reads):
  //   N = rms_norm(X, w_d); dense = down(gelu(gate(N)) * up(N))
  //   ADD(rms_norm(dense, w_dp), rms_norm(moe, w_mp)) -> rms_norm(w_sp)
  //   ADD(X, that) [-> MUL scalar] = the next residual stream.
  // Every intermediate has that one reader; the dense chain has no reshapes.
  // The export with a fused feed-forward has one projection over the gate and
  // up rows stacked and odml.swiglu (tanh GELU) for the gate and the product:
  //   dense = down(swiglu(gate_up(N)))
  struct MoeBlockFusion {
    Tensor dense_norm;    // the dense input norm's scale
    Tensor dense_gate_w;  // [dense_hidden, model_dim], blockwise INT4
    Tensor dense_up_w;
    // dense_gate_w and dense_up_w are one tensor [2 * dense_hidden,
    // model_dim], the gate rows first.
    bool dense_stacked = false;
    Tensor dense_down_w;  // [model_dim, dense_hidden]
    Tensor dense_post;    // the dense output norm's scale
    Tensor moe_post;      // the experts' output norm's scale
    Tensor sum_post;      // the sum's norm's scale
    Tensor output;        // the next residual stream
    int32_t dense_hidden = 0;
    int32_t gate_mode = 0;
    float multiplier = 1.0f;
    std::vector<LiteRtOp> absorbed;
  };

  std::optional<MoeBlockFusion> FindMoeBlock(const Op& op,
                                             const MoeCustomOp& moe,
                                             const MoeRouting& routing) {
    if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_FUSE_MOE_BLOCK",
                    /*default_value=*/true)) {
      return std::nullopt;
    }
    MoeBlockFusion block;
    auto is_norm = [](const Op& candidate) {
      return candidate.Code() == kLiteRtOpCodeShloComposite &&
             CompositeOpName(candidate) == "odml.rms_norm" &&
             candidate.Inputs().size() == 2 && candidate.Outputs().size() == 1;
    };
    auto is_fc = [](const Op& candidate) {
      return candidate.Code() == kLiteRtOpCodeTflFullyConnected &&
             candidate.Inputs().size() == 2 &&
             HasNoFusedActivation(candidate) &&
             IsBlockwiseInt4Weights(candidate.Inputs()[1]);
    };
    auto is_add = [](const Op& candidate) {
      return candidate.Code() == kLiteRtOpCodeTflAdd &&
             candidate.Inputs().size() == 2 &&
             HasNoFusedActivation(candidate) && candidate.Outputs().size() == 1;
    };
    const LiteRtTensor residual = StripReshapes(routing.residual).Get();
    // moe -> rms_norm(w_mp) -> ADD1
    const auto moe_norm = SoleUser(op.Outputs()[0], &block.absorbed);
    if (!moe_norm || !is_norm(*moe_norm)) return std::nullopt;
    block.moe_post = moe_norm->Inputs()[1];
    block.absorbed.push_back(moe_norm->Get());
    const auto add1 = SoleUser(moe_norm->Outputs()[0], &block.absorbed);
    if (!add1 || !is_add(*add1)) return std::nullopt;
    block.absorbed.push_back(add1->Get());
    const int moe_side =
        StripReshapes(add1->Inputs()[0]).Get() == moe_norm->Outputs()[0].Get()
            ? 0
            : 1;
    if (StripReshapes(add1->Inputs()[moe_side]).Get() !=
        moe_norm->Outputs()[0].Get()) {
      return std::nullopt;
    }
    // The dense side, back from ADD1's other input.
    const Tensor dense_normed = add1->Inputs()[1 - moe_side];
    if (!ReadOnce(dense_normed)) return std::nullopt;
    const auto dense_norm = dense_normed.GetDefiningOp();
    if (!dense_norm || !is_norm(*dense_norm) ||
        !ReadOnce(dense_norm->Inputs()[0])) {
      return std::nullopt;
    }
    block.dense_post = dense_norm->Inputs()[1];
    block.absorbed.push_back(dense_norm->Get());
    const auto down = dense_norm->Inputs()[0].GetDefiningOp();
    if (!down || !is_fc(*down) || !ReadOnce(down->Inputs()[0])) {
      return std::nullopt;
    }
    block.dense_down_w = down->Inputs()[1];
    block.absorbed.push_back(down->Get());
    const auto gated = down->Inputs()[0].GetDefiningOp();
    if (!gated) return std::nullopt;
    Tensor dense_in;
    if (gated->Code() == kLiteRtOpCodeShloComposite &&
        CompositeOpName(*gated) == "odml.swiglu") {
      // The fused export: one projection over the gate and up rows stacked,
      // gated by odml.swiglu with the tanh GELU.
      const auto swiglu = ParseSwiglu(*gated);
      if (!swiglu.has_value() ||
          swiglu->activation != kLiteRtNvidiaGemmGateGeluTanh ||
          !ReadOnce(gated->Inputs()[0])) {
        return std::nullopt;
      }
      const auto gate_up = gated->Inputs()[0].GetDefiningOp();
      if (!gate_up || !is_fc(*gate_up)) return std::nullopt;
      block.gate_mode = 1;
      block.dense_stacked = true;
      block.absorbed.push_back(gated->Get());
      block.absorbed.push_back(gate_up->Get());
      block.dense_gate_w = gate_up->Inputs()[1];
      block.dense_up_w = gate_up->Inputs()[1];
      dense_in = gate_up->Inputs()[0];
    } else {
      if (gated->Code() != kLiteRtOpCodeTflMul || gated->Inputs().size() != 2 ||
          !HasNoFusedActivation(*gated)) {
        return std::nullopt;
      }
      block.absorbed.push_back(gated->Get());
      std::optional<Op> gelu;
      std::optional<Op> up;
      for (int i = 0; i < 2; ++i) {
        if (!ReadOnce(gated->Inputs()[i])) return std::nullopt;
        const auto producer = gated->Inputs()[i].GetDefiningOp();
        if (!producer) return std::nullopt;
        if (producer->Code() == kLiteRtOpCodeTflGelu) {
          gelu = *producer;
        } else if (producer->Code() == kLiteRtOpCodeTflFullyConnected) {
          up = *producer;
        }
      }
      if (!gelu || !up || gelu->Inputs().size() != 1 || !is_fc(*up) ||
          !ReadOnce(gelu->Inputs()[0])) {
        return std::nullopt;
      }
      bool approximate = false;
      if (LiteRtGetGeluApproximateOption(gelu->Get(), &approximate) !=
          kLiteRtStatusOk) {
        return std::nullopt;
      }
      block.gate_mode = approximate ? 1 : 2;
      block.absorbed.push_back(gelu->Get());
      block.absorbed.push_back(up->Get());
      const auto gate = gelu->Inputs()[0].GetDefiningOp();
      if (!gate || !is_fc(*gate) ||
          gate->Inputs()[0].Get() != up->Inputs()[0].Get()) {
        return std::nullopt;
      }
      block.absorbed.push_back(gate->Get());
      block.dense_gate_w = gate->Inputs()[1];
      block.dense_up_w = up->Inputs()[1];
      dense_in = gate->Inputs()[0];
    }
    // Its readers: the gate and the up projection, or the one stacked one.
    if (graph_outputs_.count(dense_in.Get()) != 0 ||
        dense_in.Uses().size() != (block.dense_stacked ? 1u : 2u)) {
      return std::nullopt;
    }
    const auto dense_in_norm = dense_in.GetDefiningOp();
    if (!dense_in_norm || !is_norm(*dense_in_norm) ||
        StripReshapes(dense_in_norm->Inputs()[0]).Get() != residual) {
      return std::nullopt;
    }
    block.dense_norm = dense_in_norm->Inputs()[1];
    block.absorbed.push_back(dense_in_norm->Get());
    // The dense dims.
    auto gate_type = block.dense_gate_w.RankedTensorType();
    auto up_type = block.dense_up_w.RankedTensorType();
    auto down_type = block.dense_down_w.RankedTensorType();
    if (!gate_type || !up_type || !down_type) return std::nullopt;
    const auto gd = gate_type->Layout().Dimensions();
    const auto ud = up_type->Layout().Dimensions();
    const auto dd = down_type->Layout().Dimensions();
    if (gd.size() != 2 || ud.size() != 2 || dd.size() != 2 ||
        gd[1] != moe.model_dim || ud[1] != moe.model_dim || ud[0] != gd[0] ||
        (block.dense_stacked && gd[0] % 2 != 0)) {
      return std::nullopt;
    }
    const int32_t dense_hidden = block.dense_stacked ? gd[0] / 2 : gd[0];
    if (dd[0] != moe.model_dim || dd[1] != dense_hidden || dense_hidden < 32 ||
        dense_hidden % 32 != 0) {
      return std::nullopt;
    }
    block.dense_hidden = dense_hidden;
    // ADD1 -> rms_norm(w_sp) -> ADD2(X, .) [-> MUL scalar]
    const auto sum_norm = SoleUser(add1->Outputs()[0], &block.absorbed);
    if (!sum_norm || !is_norm(*sum_norm)) return std::nullopt;
    block.sum_post = sum_norm->Inputs()[1];
    block.absorbed.push_back(sum_norm->Get());
    const auto add2 = SoleUser(sum_norm->Outputs()[0], &block.absorbed);
    if (!add2 || !is_add(*add2)) return std::nullopt;
    const int sum_side =
        StripReshapes(add2->Inputs()[0]).Get() == sum_norm->Outputs()[0].Get()
            ? 0
            : 1;
    if (StripReshapes(add2->Inputs()[sum_side]).Get() !=
            sum_norm->Outputs()[0].Get() ||
        StripReshapes(add2->Inputs()[1 - sum_side]).Get() != residual) {
      return std::nullopt;
    }
    block.absorbed.push_back(add2->Get());
    block.output = add2->Outputs()[0];
    std::vector<LiteRtOp> through;
    const auto scaled = SoleUser(add2->Outputs()[0], &through);
    if (scaled && scaled->Code() == kLiteRtOpCodeTflMul &&
        scaled->Inputs().size() == 2 && HasNoFusedActivation(*scaled) &&
        scaled->Outputs().size() == 1) {
      std::optional<float> scalar;
      for (int i = 0; i < 2; ++i) {
        if (auto constant = ReadFloatScalar(scaled->Inputs()[i])) {
          scalar = *constant;
        }
      }
      if (scalar.has_value()) {
        block.multiplier = *scalar;
        block.absorbed.insert(block.absorbed.end(), through.begin(),
                              through.end());
        block.absorbed.push_back(scaled->Get());
        block.output = scaled->Outputs()[0];
      }
    }
    for (const Tensor* w : {&block.dense_norm, &block.dense_post,
                            &block.moe_post, &block.sum_post}) {
      if (w->ElementType() != litert::ElementType::Float32 ||
          !w->HasWeights() ||
          w->Weights().Bytes().size() !=
              static_cast<size_t>(moe.model_dim) * sizeof(float)) {
        return std::nullopt;
      }
    }
    return block;
  }

  // Whether blockwise weights and their scales are views of the model file
  // (what the MoE plugin's arena loads from).
  bool WeightsInFile(const Tensor& weights) const {
    if (weight_store_ == nullptr) return false;
    const Tensor scales(weights.Context(),
                        weights.BlockWiseQuantization().scales);
    return weight_store_->SourceOffset(weights.Weights().Bytes()).HasValue() &&
           weight_store_->SourceOffset(scales.Weights().Bytes()).HasValue();
  }

  // The MoE layer blocks, found before anything lowers: the dense
  // feed-forward's ops come before the moe op in the graph, and lowered
  // first they would put their weights into the store (where they stayed,
  // 0.3 GB of the 26B's device memory, until this pass) before the block
  // absorbed them. The plan is reused by LowerCustomOp.
  struct MoeBlockPlan {
    std::optional<MoeRouting> routing;
    std::optional<MoeBlockFusion> block;
  };

  Expected<void> FindMoeBlocks(const Subgraph& subgraph) {
    if (weight_store_ == nullptr) return {};
    for (const auto& op : subgraph.Ops()) {
      if (op.Code() != kLiteRtOpCodeTflCustom) continue;
      const auto moe = ParseMoeCustomOp(op);
      if (!moe.has_value()) continue;
      MoeBlockPlan plan;
      plan.routing = FindMoeRouting(op, *moe);
      if (!plan.routing.has_value() ||
          !WeightsInFile(plan.routing->router_weights)) {
        continue;
      }
      plan.block = FindMoeBlock(op, *moe, *plan.routing);
      if (plan.block.has_value() &&
          (!WeightsInFile(plan.block->dense_gate_w) ||
           !WeightsInFile(plan.block->dense_up_w) ||
           !WeightsInFile(plan.block->dense_down_w))) {
        plan.block.reset();
      }
      if (plan.block.has_value()) {
        for (LiteRtOp absorbed : plan.block->absorbed) {
          unlowered_ops_.insert(absorbed);
        }
      }
      moe_plans_[op.Get()] = std::move(plan);
    }
    return {};
  }

  Expected<nvinfer1::ITensor*> AddNormScaleConstant(const Tensor& w,
                                                    int32_t columns,
                                                    const std::string& name) {
    std::vector<float> values(columns);
    std::memcpy(values.data(), w.Weights().Bytes().data(),
                values.size() * sizeof(float));
    nvinfer1::Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = columns;
    return AddFloatConstant(values, dims,
                            name + std::to_string(w.TensorIndex()),
                            nvinfer1::DataType::kFLOAT);
  }

  // The FP32 view of a linear-attention state input (IsInPlaceFp32State): the
  // network input itself, which no FP16/BF16 activation cast has touched.
  Expected<nvinfer1::ITensor*> GetFp32StateInput(const Tensor& state) {
    const auto raw = raw_input_map_.find(state.Get());
    if (raw == raw_input_map_.end() || raw->second == nullptr ||
        raw->second->getType() != nvinfer1::DataType::kFLOAT) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Linear-attention state is not an FP32 network input");
    }
    return raw->second;
  }

  // A plugin's (activation, FP32 state') outputs: the state leaves the
  // network in FP32 through raw_output_map_, unrounded.
  Expected<void> SetStatefulPluginOutputs(const Op& op,
                                          nvinfer1::IPluginV3Layer* layer,
                                          const std::string& name) {
    for (int o = 0; o < 2; ++o) {
      layer->getOutput(o)->setName(
          KeepName(UniqueName(name + "_" + std::to_string(o))));
    }
    raw_output_map_[op.Outputs()[1].Get()] = layer->getOutput(1);
    LITERT_RETURN_IF_ERROR(
        SetOutputTensor(op.Outputs()[0], layer->getOutput(0)));
    return SetOutputTensor(op.Outputs()[1], layer->getOutput(1));
  }

  // A member of a fused GEMV group: the group's output and the member's
  // columns [offset, offset + columns) of its `stride`.
  struct GemvGroupSlice {
    nvinfer1::ITensor* fused = nullptr;
    int32_t offset = 0;
    int32_t columns = 0;
    int32_t stride = 0;
  };

  // A whole gated DeltaNet layer of Qwen 3.5 (trtllm/gated_delta_net.h) from
  // the outputs of its in-projections to the product of its gated norm,
  // matched around its gated_delta_update op (FindGatedDeltaNetBlocks).
  struct GatedDeltaNetBlock {
    LiteRtNvidiaGatedDeltaNetParams params{};
    // The in-projections qkv, z, a (with the dt bias) and b, and their outputs.
    std::array<LiteRtOp, 4> projections{};
    std::array<LiteRtTensor, 4> projection_outputs{};
    std::optional<Tensor> conv_state, conv_state_out, state, state_out;
    std::optional<Tensor> valid_len;
    std::optional<Tensor> output;  // the gated norm's product
    std::vector<float> constants;  // gated_delta_net.h's order, biases last
  };

  // Whether `tensor` is a row mask: a cast of a BOOL tensor (the valid rows
  // of a prefill chunk).
  static bool IsRowMask(const Tensor& tensor) {
    const auto cast = tensor.GetDefiningOp();
    return cast && cast->Code() == kLiteRtOpCodeTflCast &&
           cast->Inputs().size() == 1 &&
           cast->Inputs()[0].ElementType() == litert::ElementType::Bool;
  }

  // Whether `op` is a TRANSPOSE swapping axes 1 and 2 of a rank 3 or 4 tensor
  // (between the rows and the heads of a gated DeltaNet layer).
  static bool IsRowHeadTranspose(const Op& op) {
    if (op.Code() != kLiteRtOpCodeTflTranspose || op.Inputs().size() != 2) {
      return false;
    }
    const auto perm = ReadInt32Constant(op.Inputs()[1]);
    return perm && (*perm == std::vector<int32_t>{0, 2, 1} ||
                    *perm == std::vector<int32_t>{0, 2, 1, 3});
  }

  // The producer side of `tensor` past the layout ops and row masks of a
  // gated DeltaNet layer: single-use RESHAPE and row/head TRANSPOSE ops and
  // MULs by a row mask, which join `members`.
  static Tensor SkipLayoutAndMasks(Tensor tensor,
                                   std::vector<LiteRtOp>& members) {
    while (true) {
      const auto producer = tensor.GetDefiningOp();
      if (!producer || tensor.Uses().size() != 1 ||
          producer->Outputs().size() != 1 || producer->Inputs().empty()) {
        return tensor;
      }
      const Op& op = *producer;
      if (op.Code() == kLiteRtOpCodeTflReshape || IsRowHeadTranspose(op)) {
        members.push_back(op.Get());
        tensor = op.Inputs()[0];
        continue;
      }
      if (op.Code() == kLiteRtOpCodeTflMul && op.Inputs().size() == 2 &&
          HasNoFusedActivation(op)) {
        if (IsRowMask(op.Inputs()[1])) {
          members.push_back(op.Get());
          tensor = op.Inputs()[0];
          continue;
        }
        if (IsRowMask(op.Inputs()[0])) {
          members.push_back(op.Get());
          tensor = op.Inputs()[1];
          continue;
        }
      }
      return tensor;
    }
  }

  // An odml.rms_norm composite with an FP32 constant weight of `dim` values:
  // the weight and the epsilon.
  static std::optional<std::pair<std::vector<float>, float>> RmsNormWeights(
      const Op& op, int dim) {
    if (op.Code() != kLiteRtOpCodeShloComposite ||
        CompositeOpName(op) != "odml.rms_norm" || op.Inputs().size() != 2 ||
        op.Outputs().size() != 1) {
      return std::nullopt;
    }
    const Tensor weight = op.Inputs()[1];
    if (!weight.HasWeights() || !IsUnquantizedFloat32(weight) ||
        weight.Weights().Bytes().size() != dim * sizeof(float)) {
      return std::nullopt;
    }
    std::vector<float> values(dim);
    std::memcpy(values.data(), weight.Weights().Bytes().data(),
                dim * sizeof(float));
    float eps = 1e-6f;
    CompositeOptions options;
    if (options.InitFromOp(op.Get()) == kLiteRtStatusOk &&
        options.attributes_map.has_value() &&
        !options.attributes_map.value()["epsilon"].IsNull()) {
      eps = options.attributes_map.value()["epsilon"].AsFloat();
    }
    return std::make_pair(std::move(values), eps);
  }

  // The single-use output of a fully connected op of `columns` outputs.
  static std::optional<Op> ProjectionOf(const Tensor& tensor, int columns) {
    const auto fc = tensor.GetDefiningOp();
    auto type = tensor.RankedTensorType();
    if (!fc || fc->Code() != kLiteRtOpCodeTflFullyConnected ||
        fc->Outputs().size() != 1 || tensor.Uses().size() != 1 || !type ||
        type->Layout().Rank() < 1 ||
        type->Layout().Dimensions().back() != columns) {
      return std::nullopt;
    }
    return *fc;
  }

  // Matches the layer around one gated_delta_update op, in decode
  //   y = short_conv(qkv); q, k = rms_norm(slices of y); v = slice of y
  //   beta = sigmoid(b); g = A * log(1 + exp(min(a, 50)))
  //   o = gated_delta_update(q, k, v, beta, g, state)
  //   out = rms_norm(o) * (z * sigmoid(z)), or * sigmoid(z) (Qwen 4)
  // with reshapes between the shapes, and in prefill with row/head
  // transposes, row masks on every operand and the valid length. Every
  // intermediate result has a single reader within the layer; the ops past
  // the projections join `members`, the final product is `anchor`.
  std::optional<GatedDeltaNetBlock> MatchGatedDeltaNetBlock(
      const Op& gdn, const GatedDeltaUpdateOp& parsed,
      std::vector<LiteRtOp>& members, LiteRtOp& anchor) {
    const auto& p = parsed.params;
    const int d = 128;
    if (p.key_dim != d || p.value_dim != d) return std::nullopt;
    const int hk = p.key_heads;
    const int hv = p.heads;
    const int channels = (2 * hk + hv) * d;
    GatedDeltaNetBlock block;
    members.push_back(gdn.Get());
    std::optional<Op> conv;
    // The slice of the convolution's output at `offset`, `width` wide.
    const auto conv_slice = [&](const Tensor& tensor, int offset,
                                int width) -> bool {
      const auto slice = tensor.GetDefiningOp();
      if (!slice || tensor.Uses().size() != 1 ||
          slice->Code() != kLiteRtOpCodeTflSlice ||
          slice->Inputs().size() != 3) {
        return false;
      }
      const auto begin = ReadInt32Constant(slice->Inputs()[1]);
      const auto size = ReadInt32Constant(slice->Inputs()[2]);
      if (!begin || !size || begin->size() != 3 || size->size() != 3 ||
          (*begin)[0] != 0 || (*begin)[1] != 0 || (*begin)[2] != offset ||
          (*size)[2] != width ||
          !TensorHasShape(slice->Inputs()[0], {1, p.tokens, channels})) {
        return false;
      }
      const auto producer = slice->Inputs()[0].GetDefiningOp();
      if (!producer || (conv && conv->Get() != producer->Get())) return false;
      conv = *producer;
      members.push_back(slice->Get());
      return true;
    };
    // q and k: the norms of their slices.
    const auto normed = [&](int input, int offset, float& eps) -> bool {
      const Tensor t = SkipLayoutAndMasks(gdn.Inputs()[input], members);
      const auto norm = t.GetDefiningOp();
      if (!norm || t.Uses().size() != 1) return false;
      auto weights = RmsNormWeights(*norm, d);
      if (!weights) return false;
      members.push_back(norm->Get());
      block.constants.insert(block.constants.end(), weights->first.begin(),
                             weights->first.end());
      eps = weights->second;
      return conv_slice(SkipLayoutAndMasks(norm->Inputs()[0], members), offset,
                        hk * d);
    };
    // decay [hv], then q_scale and k_scale (filled by `normed`).
    std::vector<float> decay;
    if (!normed(0, 0, block.params.q_eps) ||
        !normed(1, hk * d, block.params.k_eps) ||
        !conv_slice(SkipLayoutAndMasks(gdn.Inputs()[2], members), 2 * hk * d,
                    hv * d)) {
      return std::nullopt;
    }
    // beta = sigmoid(b).
    const Tensor beta = SkipLayoutAndMasks(gdn.Inputs()[3], members);
    const auto sigmoid = beta.GetDefiningOp();
    if (!sigmoid || beta.Uses().size() != 1 ||
        sigmoid->Code() != kLiteRtOpCodeTflLogistic) {
      return std::nullopt;
    }
    members.push_back(sigmoid->Get());
    const auto b_fc = ProjectionOf(sigmoid->Inputs()[0], hv);
    // g = A * log(1 + exp(min(a, 50))), A a constant per head.
    const auto unary_of =
        [&](const Tensor& tensor, LiteRtOpCode code,
            std::optional<float> constant,
            std::optional<Tensor>* operand) -> std::optional<Op> {
      const auto op = tensor.GetDefiningOp();
      if (!op || tensor.Uses().size() != 1 || op->Code() != code ||
          op->Outputs().size() != 1) {
        return std::nullopt;
      }
      if (!constant.has_value()) {
        if (op->Inputs().size() != 1) return std::nullopt;
        *operand = op->Inputs()[0];
      } else {
        if (op->Inputs().size() != 2 || !HasNoFusedActivation(*op)) {
          return std::nullopt;
        }
        const int c = ReadFloat32Scalar(op->Inputs()[1]) == constant ? 1 : 0;
        if (ReadFloat32Scalar(op->Inputs()[c]) != constant) return std::nullopt;
        *operand = op->Inputs()[1 - c];
      }
      members.push_back(op->Get());
      return *op;
    };
    const Tensor g = SkipLayoutAndMasks(gdn.Inputs()[4], members);
    const auto scale = g.GetDefiningOp();
    if (!scale || g.Uses().size() != 1 ||
        scale->Code() != kLiteRtOpCodeTflMul || scale->Inputs().size() != 2 ||
        !HasNoFusedActivation(*scale)) {
      return std::nullopt;
    }
    const int a_index = scale->Inputs()[1].HasWeights() ? 1 : 0;
    const Tensor a_values = scale->Inputs()[a_index];
    if (!a_values.HasWeights() || !IsUnquantizedFloat32(a_values) ||
        a_values.Weights().Bytes().size() != hv * sizeof(float)) {
      return std::nullopt;
    }
    decay.resize(hv);
    std::memcpy(decay.data(), a_values.Weights().Bytes().data(),
                hv * sizeof(float));
    members.push_back(scale->Get());
    std::optional<Tensor> chain = scale->Inputs()[1 - a_index];
    if (!unary_of(*chain, kLiteRtOpCodeTflLog, std::nullopt, &chain) ||
        !unary_of(*chain, kLiteRtOpCodeTflAdd, 1.0f, &chain) ||
        !unary_of(*chain, kLiteRtOpCodeTflExp, std::nullopt, &chain) ||
        !unary_of(*chain, kLiteRtOpCodeTflMinimum, 50.0f, &chain)) {
      return std::nullopt;
    }
    const auto a_fc = ProjectionOf(*chain, hv);
    // The convolution and its input, the qkv projection.
    if (!conv || conv->Outputs()[0].Uses().size() != 3) return std::nullopt;
    const auto conv_parsed = ParseShortConvStep(*conv);
    if (!conv_parsed || conv_parsed->params.channels != channels ||
        conv_parsed->params.tokens != p.tokens) {
      return std::nullopt;
    }
    members.push_back(conv->Get());
    const auto qkv_fc =
        ProjectionOf(SkipLayoutAndMasks(conv->Inputs()[0], members), channels);
    // The output side: o -> layout ops -> rms_norm -> * silu(z).
    Tensor o = gdn.Outputs()[0];
    std::optional<Op> next;
    while (true) {
      const auto uses = o.Uses();
      if (uses.size() != 1) return std::nullopt;
      const Op& user = uses[0].user;
      if (user.Code() == kLiteRtOpCodeTflReshape || IsRowHeadTranspose(user)) {
        if (uses[0].user_arg_ind != 0) return std::nullopt;
        members.push_back(user.Get());
        o = user.Outputs()[0];
        continue;
      }
      next = user;
      break;
    }
    auto out_norm = RmsNormWeights(*next, d);
    if (!out_norm || next->Inputs()[0].Get() != o.Get()) return std::nullopt;
    members.push_back(next->Get());
    const auto product_uses = next->Outputs()[0].Uses();
    if (product_uses.size() != 1) return std::nullopt;
    const Op& product = product_uses[0].user;
    if (product.Code() != kLiteRtOpCodeTflMul || product.Inputs().size() != 2 ||
        !HasNoFusedActivation(product) ||
        !TensorHasShape(product.Outputs()[0], {1, p.tokens, hv, d})) {
      return std::nullopt;
    }
    const Tensor gate = product.Inputs()[1 - product_uses[0].user_arg_ind];
    const auto silu = gate.GetDefiningOp();
    if (!silu || gate.Uses().size() != 1) return std::nullopt;
    std::optional<Tensor> z;
    if (silu->Code() == kLiteRtOpCodeTflLogistic &&
        silu->Inputs().size() == 1 && silu->Inputs()[0].Uses().size() == 1) {
      // Qwen 4's output gate: rms_norm(o) * sigmoid(z).
      z = silu->Inputs()[0];
      block.params.sigmoid_gate = 1;
    } else {
      if (silu->Code() != kLiteRtOpCodeTflMul || silu->Inputs().size() != 2 ||
          !HasNoFusedActivation(*silu)) {
        return std::nullopt;
      }
      for (int i = 0; i < 2 && !z; ++i) {
        const auto logistic = silu->Inputs()[i].GetDefiningOp();
        if (logistic && logistic->Code() == kLiteRtOpCodeTflLogistic &&
            silu->Inputs()[i].Uses().size() == 1 &&
            logistic->Inputs()[0].Get() == silu->Inputs()[1 - i].Get() &&
            silu->Inputs()[1 - i].Uses().size() == 2) {
          z = silu->Inputs()[1 - i];
          members.push_back(logistic->Get());
        }
      }
    }
    if (!z) return std::nullopt;
    members.push_back(silu->Get());
    Tensor z_raw = *z;
    if (const auto reshape = z_raw.GetDefiningOp();
        reshape && reshape->Code() == kLiteRtOpCodeTflReshape &&
        reshape->Inputs()[0].Uses().size() == 1) {
      members.push_back(reshape->Get());
      z_raw = reshape->Inputs()[0];
    }
    const auto z_fc = ProjectionOf(z_raw, hv * d);
    if (!qkv_fc || !z_fc || !a_fc || !b_fc) return std::nullopt;
    // The projections' biases, which FinishFullyConnected adds past the GEMV
    // (zeros without one); a fused activation would not be in the group's
    // output either.
    std::vector<float> biases;
    for (const auto& [fc, columns] :
         {std::pair<const Op*, int>{&*qkv_fc, channels},
          {&*z_fc, hv * d},
          {&*a_fc, hv},
          {&*b_fc, hv}}) {
      if (!HasNoFusedActivation(*fc)) return std::nullopt;
      const size_t at = biases.size();
      biases.resize(at + columns, 0.0f);
      if (fc->Inputs().size() < 3) continue;
      const Tensor bias = fc->Inputs()[2];
      if (!bias.HasWeights() || !IsUnquantizedFloat32(bias) ||
          bias.Weights().Bytes().size() != columns * sizeof(float)) {
        return std::nullopt;
      }
      std::memcpy(biases.data() + at, bias.Weights().Bytes().data(),
                  columns * sizeof(float));
    }

    block.projections = {qkv_fc->Get(), z_fc->Get(), a_fc->Get(), b_fc->Get()};
    block.projection_outputs = {
        qkv_fc->Outputs()[0].Get(), z_fc->Outputs()[0].Get(),
        a_fc->Outputs()[0].Get(), b_fc->Outputs()[0].Get()};
    block.conv_state = conv->Inputs()[1];
    block.conv_state_out = conv->Outputs()[1];
    block.state = gdn.Inputs()[5];
    block.state_out = gdn.Outputs()[1];
    if (parsed.has_valid_len) block.valid_len = gdn.Inputs()[6];
    if (parsed.has_valid_len != conv_parsed->has_valid_len) return std::nullopt;
    block.output = product.Outputs()[0];
    anchor = product.Get();
    // decay, q_scale, k_scale (already in), norm_weight, conv_weight.
    block.constants.insert(block.constants.begin(), decay.begin(), decay.end());
    block.constants.insert(block.constants.end(), out_norm->first.begin(),
                           out_norm->first.end());
    const auto conv_bytes = conv->Inputs()[2].Weights().Bytes();
    const size_t conv_values = conv_bytes.size() / sizeof(float);
    const size_t before = block.constants.size();
    block.constants.resize(before + conv_values);
    std::memcpy(block.constants.data() + before, conv_bytes.data(),
                conv_values * sizeof(float));
    block.constants.insert(block.constants.end(), biases.begin(), biases.end());
    auto& bp = block.params;
    bp.tokens = p.tokens;
    bp.key_heads = hk;
    bp.value_heads = hv;
    bp.head_dim = d;
    bp.taps = conv_parsed->params.taps;
    bp.silu = conv_parsed->params.silu;
    bp.out_eps = out_norm->second;
    return block;
  }

  // Records every gated DeltaNet layer matched by MatchGatedDeltaNetBlock:
  // its ops past the in-projections are not lowered, and its final product
  // emits the LiteRtNvidiaGatedDeltaNet plugin (LowerGatedDeltaNetBlock).
  // LITERT_NVIDIA_TENSORRT_FUSE_GATED_DELTA_NET=0 keeps the op-by-op lowering.
  Expected<void> FindGatedDeltaNetBlocks(const Subgraph& subgraph) {
    if (!EnvEnabled("LITERT_NVIDIA_TENSORRT_FUSE_GATED_DELTA_NET",
                    /*default_value=*/true) ||
        !Fp16ActivationsEnabled()) {
      return {};
    }
    size_t found = 0;
    size_t candidates = 0;
    for (const auto& op : subgraph.Ops()) {
      const auto parsed = ParseGatedDeltaUpdate(op);
      if (!parsed.has_value()) continue;
      ++candidates;
      std::vector<LiteRtOp> members;
      LiteRtOp anchor = nullptr;
      auto block = MatchGatedDeltaNetBlock(op, *parsed, members, anchor);
      if (!block.has_value()) continue;
      bool ordered = op_order_.count(anchor) != 0;
      for (LiteRtOp member : members) {
        ordered = ordered && op_order_.count(member) != 0 &&
                  unlowered_ops_.count(member) == 0;
      }
      for (LiteRtOp projection : block->projections) {
        ordered = ordered && op_order_.count(projection) != 0 &&
                  op_order_[projection] < op_order_[anchor];
      }
      if (!ordered) continue;
      unlowered_ops_.insert(members.begin(), members.end());
      gdn_anchor_ops_[anchor] = gdn_blocks_.size();
      gdn_blocks_.push_back(std::move(*block));
      ++found;
    }
    if (candidates != 0) {
      LITERT_LOG(LITERT_INFO,
                 "NVIDIA TensorRT-RTX fused %zu of %zu gated DeltaNet layers "
                 "into the CUDA gated DeltaNet plugin",
                 found, candidates);
    }
    return {};
  }

  // Emits one matched gated DeltaNet layer at its final product. The
  // projections come from their fused GEMV group's output when all four are
  // its members (no copies), else from their concatenation.
  Expected<void> LowerGatedDeltaNetBlock(size_t index) {
    GatedDeltaNetBlock& block = gdn_blocks_[index];
    auto& p = block.params;
    nvinfer1::ITensor* proj = nullptr;
    std::array<int32_t, 4> offsets{};
    const GemvGroupSlice* group = nullptr;
    bool grouped = true;
    for (int i = 0; i < 4 && grouped; ++i) {
      const auto it = gemv_group_slices_.find(block.projections[i]);
      grouped = it != gemv_group_slices_.end() &&
                (group == nullptr || it->second.fused == group->fused);
      if (grouped) {
        group = &it->second;
        offsets[i] = it->second.offset;
      }
    }
    if (grouped) {
      proj = group->fused;
      p.proj_stride = group->stride;
    } else {
      std::vector<nvinfer1::ITensor*> parts;
      int32_t at = 0;
      for (int i = 0; i < 4; ++i) {
        const auto it = tensor_map_.find(block.projection_outputs[i]);
        if (it == tensor_map_.end() || it->second == nullptr) {
          return Error(kLiteRtStatusErrorCompilation,
                       "Gated DeltaNet projection was not lowered");
        }
        LITERT_ASSIGN_OR_RETURN(
            auto* part, AddCastTensor(it->second, nvinfer1::DataType::kBF16));
        const auto dims = part->getDimensions();
        offsets[i] = at;
        at += static_cast<int32_t>(dims.d[dims.nbDims - 1]);
        parts.push_back(part);
      }
      auto* concat = network_->addConcatenation(
          parts.data(), static_cast<int32_t>(parts.size()));
      if (concat == nullptr || concat->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to concatenate the gated DeltaNet projections");
      }
      concat->setAxis(parts[0]->getDimensions().nbDims - 1);
      proj = concat->getOutput(0);
      p.proj_stride = at;
      // The lowered projections carry their biases (FinishFullyConnected),
      // which the plugin would add again: it takes zeros instead.
      LiteRtNvidiaGatedDeltaNetParams sized = p;
      sized.tokens = 1;
      const size_t biases =
          static_cast<size_t>(2 * p.key_heads + p.value_heads) * p.head_dim +
          static_cast<size_t>(p.value_heads) * p.head_dim + 2 * p.value_heads;
      const size_t count = LiteRtNvidiaGatedDeltaNetConstantCount(&sized);
      if (block.constants.size() == count && count >= biases) {
        std::fill(block.constants.end() - biases, block.constants.end(), 0.0f);
      }
    }
    LITERT_ASSIGN_OR_RETURN(proj,
                            AddCastTensor(proj, nvinfer1::DataType::kBF16));
    p.qkv_offset = offsets[0];
    p.z_offset = offsets[1];
    p.a_offset = offsets[2];
    p.b_offset = offsets[3];
    if (block.constants.size() != LiteRtNvidiaGatedDeltaNetConstantCount(&p)) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Gated DeltaNet constants do not match the layer");
    }
    LITERT_ASSIGN_OR_RETURN(auto* conv_state,
                            GetFp32StateInput(*block.conv_state));
    LITERT_ASSIGN_OR_RETURN(auto* state, GetFp32StateInput(*block.state));
    nvinfer1::Dims constant_dims{};
    constant_dims.nbDims = 1;
    constant_dims.d[0] = static_cast<int64_t>(block.constants.size());
    const std::string suffix = std::to_string(block.state->TensorIndex());
    LITERT_ASSIGN_OR_RETURN(
        auto* constants, AddFloatConstant(block.constants, constant_dims,
                                          "gated_delta_net_constants_" + suffix,
                                          nvinfer1::DataType::kFLOAT));
    std::vector<nvinfer1::ITensor*> inputs = {proj, conv_state, state,
                                              constants};
    if (block.valid_len.has_value()) {
      LITERT_ASSIGN_OR_RETURN(auto* valid_len, GetTensor(*block.valid_len));
      if (valid_len->getType() != nvinfer1::DataType::kINT32) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "gated DeltaNet valid length must be int32");
      }
      inputs.push_back(valid_len);
    }
    TrtPtr<nvinfer1::IPluginV3> plugin(
        CreateGatedDeltaNetPlugin(p, block.valid_len.has_value()));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the gated DeltaNet plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr ||
        layer->getOutput(1) == nullptr || layer->getOutput(2) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the gated DeltaNet plugin layer");
    }
    const std::string name = "cuda_gated_delta_net_" + suffix;
    layer->setName(KeepName(UniqueName(name)));
    for (int o = 0; o < 3; ++o) {
      layer->getOutput(o)->setName(
          KeepName(UniqueName(name + "_" + std::to_string(o))));
    }
    owned_plugins_.push_back(std::move(plugin));
    LITERT_ASSIGN_OR_RETURN(auto out_type, block.output->RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto out_dims, ConvertDims(out_type));
    LITERT_ASSIGN_OR_RETURN(auto* out,
                            ReshapeTensor(layer->getOutput(0), out_dims));
    raw_output_map_[block.conv_state_out->Get()] = layer->getOutput(1);
    raw_output_map_[block.state_out->Get()] = layer->getOutput(2);
    LITERT_RETURN_IF_ERROR(
        SetOutputTensor(*block.conv_state_out, layer->getOutput(1)));
    LITERT_RETURN_IF_ERROR(
        SetOutputTensor(*block.state_out, layer->getOutput(2)));
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX fused CUDA gated DeltaNet layer: state=%s "
               "rows=%d projections=%s",
               suffix.c_str(), p.tokens, grouped ? "group" : "concatenated");
    return SetOutputTensor(*block.output, out);
  }

  // Qwen 3.5's gated_delta_update as the LiteRtNvidiaGatedDeltaUpdate plugin
  // (trtllm/gated_delta.h): q, k, v, beta and g in FP32, the state read from
  // and written to the runtime's FP32 buffer.
  Expected<void> LowerGatedDeltaUpdate(const Op& op,
                                       const GatedDeltaUpdateOp& parsed) {
    std::vector<nvinfer1::ITensor*> inputs;
    for (int i = 0; i < 5; ++i) {
      LITERT_ASSIGN_OR_RETURN(auto* tensor, GetTensor(op.Inputs()[i]));
      LITERT_ASSIGN_OR_RETURN(
          tensor, AddCastTensor(tensor, nvinfer1::DataType::kFLOAT));
      inputs.push_back(tensor);
    }
    LITERT_ASSIGN_OR_RETURN(auto* state, GetFp32StateInput(op.Inputs()[5]));
    inputs.push_back(state);
    if (parsed.has_valid_len) {
      LITERT_ASSIGN_OR_RETURN(auto* valid_len, GetTensor(op.Inputs()[6]));
      if (valid_len->getType() != nvinfer1::DataType::kINT32) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "gated_delta_update valid length must be int32");
      }
      inputs.push_back(valid_len);
    }
    const auto& p = parsed.params;
    TrtPtr<nvinfer1::IPluginV3> plugin(CreateGatedDeltaUpdatePlugin(
        p.heads, p.key_heads, p.key_dim, p.value_dim, parsed.has_valid_len));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the gated_delta_update plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr ||
        layer->getOutput(1) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the gated_delta_update plugin layer");
    }
    const std::string name =
        "cuda_gated_delta_" + std::to_string(op.Inputs()[5].TensorIndex());
    layer->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    return SetStatefulPluginOutputs(op, layer, name);
  }

  // odml.swiglu(x) = act(x[.., :N]) * x[.., N:] where its fully connected
  // producer did not absorb it (FindFusedSwiglu): its slices and gates.
  Expected<void> LowerCompositeSwiglu(const Op& op) {
    const auto parsed = ParseSwiglu(op);
    if (!parsed.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported swiglu composite");
    }
    LITERT_ASSIGN_OR_RETURN(auto* x, GetTensor(op.Inputs()[0]));
    const int axis = x->getDimensions().nbDims - 1;
    LITERT_ASSIGN_OR_RETURN(auto* gate,
                            SliceTensorAxis(x, axis, 0, parsed->gate_size));
    LITERT_ASSIGN_OR_RETURN(
        auto* up,
        SliceTensorAxis(x, axis, parsed->gate_size, parsed->gate_size));
    nvinfer1::ITensor* gated = nullptr;
    if (parsed->activation == kLiteRtNvidiaGemmGateGeluTanh) {
      auto* gelu =
          network_->addActivation(*gate, nvinfer1::ActivationType::kGELU_TANH);
      if (gelu == nullptr || gelu->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add swiglu gate");
      }
      gated = gelu->getOutput(0);
    } else {
      auto* sigmoid =
          network_->addActivation(*gate, nvinfer1::ActivationType::kSIGMOID);
      if (sigmoid == nullptr || sigmoid->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add swiglu gate");
      }
      auto* silu = network_->addElementWise(
          *gate, *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
      if (silu == nullptr || silu->getOutput(0) == nullptr) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Failed to add swiglu silu");
      }
      gated = silu->getOutput(0);
    }
    auto* out = network_->addElementWise(*gated, *up,
                                         nvinfer1::ElementWiseOperation::kPROD);
    if (out == nullptr || out->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add swiglu product");
    }
    return SetOutputTensor(op.Outputs()[0], out->getOutput(0));
  }

  // odml.short_conv_step as the LiteRtNvidiaShortConv plugin
  // (trtllm/gated_delta.h).
  Expected<void> LowerCompositeShortConvStep(const Op& op) {
    const auto parsed = ParseShortConvStep(op);
    if (!parsed.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported short_conv_step composite");
    }
    const auto& p = parsed->params;
    LITERT_ASSIGN_OR_RETURN(auto* x, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(x, AddCastTensor(x, nvinfer1::DataType::kFLOAT));
    LITERT_ASSIGN_OR_RETURN(auto* state, GetFp32StateInput(op.Inputs()[1]));
    const auto bytes = op.Inputs()[2].Weights().Bytes();
    std::vector<float> values(static_cast<size_t>(p.channels) * p.taps);
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
    nvinfer1::Dims dims{};
    dims.nbDims = 3;
    dims.d[0] = p.channels;
    dims.d[1] = 1;
    dims.d[2] = p.taps;
    LITERT_ASSIGN_OR_RETURN(
        auto* weight,
        AddFloatConstant(
            values, dims,
            "short_conv_weight_" + std::to_string(op.Inputs()[2].TensorIndex()),
            nvinfer1::DataType::kFLOAT));
    std::vector<nvinfer1::ITensor*> inputs = {x, state, weight};
    if (parsed->has_valid_len) {
      LITERT_ASSIGN_OR_RETURN(auto* valid_len, GetTensor(op.Inputs()[3]));
      if (valid_len->getType() != nvinfer1::DataType::kINT32) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "short_conv_step valid length must be int32");
      }
      inputs.push_back(valid_len);
    }
    TrtPtr<nvinfer1::IPluginV3> plugin(CreateShortConvPlugin(
        p.channels, p.taps, p.silu != 0, parsed->has_valid_len));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the short_conv_step plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr ||
        layer->getOutput(1) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the short_conv_step plugin layer");
    }
    const std::string name =
        "cuda_short_conv_" + std::to_string(op.Inputs()[1].TensorIndex());
    layer->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    return SetStatefulPluginOutputs(op, layer, name);
  }

  // The moe custom op as the CUDA mixture-of-experts plugin. The experts'
  // bytes are located in the model file through the weight store's mapping;
  // the plugin loads them when an execution context attaches. With the
  // router chain recognized (FindMoeRouting) the plugin takes the residual
  // stream and routes itself.
  Expected<void> LowerCustomOp(const Op& op) {
    if (const auto delta = ParseGatedDeltaUpdate(op); delta.has_value()) {
      return LowerGatedDeltaUpdate(op, *delta);
    }
    const auto moe = ParseMoeCustomOp(op);
    if (!moe.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported, "Unsupported custom op");
    }
    if (weight_store_ == nullptr) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "The moe op needs the weight store's model file mapping");
    }
    if (!Fp16ActivationsEnabled() ||
        ModeFloatType() != nvinfer1::DataType::kBF16) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "The moe op needs BF16 activations");
    }
    MoeExpertSource source;
    source.model_path = weight_store_->source_path();
    source.model_size = weight_store_->source_size();
    source.scale_type = moe->scale_type;
    const int piece_inputs[6] = {3, 5, 7, 4, 6, 8};
    for (int i = 0; i < 6; ++i) {
      const auto bytes = op.Inputs()[piece_inputs[i]].Weights().Bytes();
      LITERT_ASSIGN_OR_RETURN(source.offsets[i],
                              weight_store_->SourceOffset(bytes));
      source.sizes[i] = bytes.size();
    }
    const auto scale_bytes = op.Inputs()[9].Weights().Bytes();
    std::vector<float> expert_scales(moe->experts);
    std::memcpy(expert_scales.data(), scale_bytes.data(),
                expert_scales.size() * sizeof(float));
    nvinfer1::Dims scale_dims{};
    scale_dims.nbDims = 1;
    scale_dims.d[0] = moe->experts;
    LITERT_ASSIGN_OR_RETURN(
        auto* expert_scale,
        AddFloatConstant(
            expert_scales, scale_dims,
            "moe_expert_scale_" + std::to_string(op.Inputs()[9].TensorIndex()),
            nvinfer1::DataType::kFLOAT));
    std::optional<MoeRouting> routing;
    std::optional<MoeBlockFusion> planned_block;
    if (const auto plan = moe_plans_.find(op.Get()); plan != moe_plans_.end()) {
      routing = plan->second.routing;
      planned_block = plan->second.block;
    } else {
      routing = FindMoeRouting(op, *moe);
    }
    if (routing.has_value()) {
      // The router's rows and scales join the arena.
      const Tensor& router = routing->router_weights;
      const Tensor router_scales(router.Context(),
                                 router.BlockWiseQuantization().scales);
      const auto router_bytes = router.Weights().Bytes();
      const auto router_scale_bytes = router_scales.Weights().Bytes();
      auto offset = weight_store_->SourceOffset(router_bytes);
      auto scale_offset = weight_store_->SourceOffset(router_scale_bytes);
      if (offset && scale_offset) {
        source.offsets[6] = *offset;
        source.sizes[6] = router_bytes.size();
        source.offsets[7] = *scale_offset;
        source.sizes[7] = router_scale_bytes.size();
        source.router_scale_type =
            router_scales.ElementType() == litert::ElementType::Float32 ? 0 : 1;
      } else {
        LITERT_LOG(
            LITERT_INFO,
            "NVIDIA TensorRT-RTX: the router of the moe op at tensor %u "
            "is not a view of the model file; routing stays in the graph",
            op.Inputs()[3].TensorIndex());
        routing.reset();
      }
    }
    // The rest of the layer's feed-forward block: its dense pieces join the
    // arena too.
    std::optional<MoeBlockFusion> block;
    if (routing.has_value()) {
      block = planned_block.has_value() ? planned_block
                                        : FindMoeBlock(op, *moe, *routing);
    }
    if (block.has_value()) {
      const Tensor* dense_weights[3] = {
          &block->dense_gate_w, &block->dense_up_w, &block->dense_down_w};
      const int weight_pieces[3] = {8, 9, 12};
      const int scale_pieces[3] = {10, 11, 13};
      int scale_type = -1;
      bool in_file = true;
      for (int i = 0; i < 3 && in_file; ++i) {
        const Tensor scales(dense_weights[i]->Context(),
                            dense_weights[i]->BlockWiseQuantization().scales);
        const auto bytes = dense_weights[i]->Weights().Bytes();
        const auto scale_bytes = scales.Weights().Bytes();
        const auto offset = weight_store_->SourceOffset(bytes);
        const auto scale_offset = weight_store_->SourceOffset(scale_bytes);
        const int type =
            scales.ElementType() == litert::ElementType::Float32 ? 0 : 1;
        if (!offset || !scale_offset ||
            (scale_type >= 0 && type != scale_type)) {
          in_file = false;
          break;
        }
        scale_type = type;
        // The halves of a stacked projection are the gate and the up piece.
        const bool half = block->dense_stacked && i < 2;
        const size_t weight_size = half ? bytes.size() / 2 : bytes.size();
        const size_t scale_size =
            half ? scale_bytes.size() / 2 : scale_bytes.size();
        const size_t skipped = half && i == 1 ? 1 : 0;
        source.offsets[weight_pieces[i]] = *offset + skipped * weight_size;
        source.sizes[weight_pieces[i]] = weight_size;
        source.offsets[scale_pieces[i]] = *scale_offset + skipped * scale_size;
        source.sizes[scale_pieces[i]] = scale_size;
      }
      if (in_file) {
        source.dense_scale_type = scale_type;
        source.dense_hidden = block->dense_hidden;
      } else {
        LITERT_LOG(LITERT_INFO,
                   "NVIDIA TensorRT-RTX: the dense feed-forward beside the moe "
                   "op at tensor %u is not a view of the model file; the block "
                   "stays in the graph",
                   op.Inputs()[3].TensorIndex());
        for (int piece = 8; piece < 14; ++piece) {
          source.offsets[piece] = 0;
          source.sizes[piece] = 0;
        }
        block.reset();
      }
    }
    std::vector<nvinfer1::ITensor*> inputs;
    TrtPtr<nvinfer1::IPluginV3> plugin;
    if (routing.has_value()) {
      LITERT_ASSIGN_OR_RETURN(auto* x, GetTensor(routing->residual));
      LITERT_ASSIGN_OR_RETURN(x, AddCastTensor(x, nvinfer1::DataType::kBF16));
      LITERT_ASSIGN_OR_RETURN(
          auto* router_norm,
          AddNormScaleConstant(routing->router_norm, moe->model_dim,
                               "moe_router_norm_"));
      LITERT_ASSIGN_OR_RETURN(
          auto* expert_norm,
          AddNormScaleConstant(routing->expert_norm, moe->model_dim,
                               "moe_expert_norm_"));
      inputs = {x, expert_scale, router_norm, expert_norm};
      if (block.has_value()) {
        const Tensor* scales[4] = {&block->dense_norm, &block->dense_post,
                                   &block->moe_post, &block->sum_post};
        const char* names[4] = {"moe_dense_norm_", "moe_dense_post_",
                                "moe_post_", "moe_sum_post_"};
        for (int i = 0; i < 4; ++i) {
          LITERT_ASSIGN_OR_RETURN(
              auto* constant,
              AddNormScaleConstant(*scales[i], moe->model_dim, names[i]));
          inputs.push_back(constant);
        }
      }
      plugin.reset(CreateMoePlugin(
          moe->experts, moe->top_k, moe->model_dim, moe->hidden_dim, moe->gate,
          source, /*routed=*/true, routing->eps, routing->multiplier,
          block.has_value(), block.has_value() ? block->gate_mode : 0,
          block.has_value() ? block->multiplier : 1.0f));
    } else {
      LITERT_ASSIGN_OR_RETURN(auto* x, GetTensor(op.Inputs()[0]));
      LITERT_ASSIGN_OR_RETURN(x, AddCastTensor(x, nvinfer1::DataType::kBF16));
      LITERT_ASSIGN_OR_RETURN(auto* top_weights, GetTensor(op.Inputs()[1]));
      LITERT_ASSIGN_OR_RETURN(
          top_weights, AddCastTensor(top_weights, nvinfer1::DataType::kFLOAT));
      LITERT_ASSIGN_OR_RETURN(auto* top_indices, GetTensor(op.Inputs()[2]));
      if (top_indices->getType() != nvinfer1::DataType::kINT32) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "The moe op's expert indices must be int32");
      }
      plugin.reset(CreateMoePlugin(moe->experts, moe->top_k, moe->model_dim,
                                   moe->hidden_dim, moe->gate, source));
      inputs = {x, top_weights, top_indices, expert_scale};
    }
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the CUDA mixture-of-experts plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(*network_, inputs.data(),
                                                     inputs.size(), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the CUDA mixture-of-experts plugin layer");
    }
    const std::string name =
        "cuda_moe_" + std::to_string(op.Inputs()[3].TensorIndex());
    layer->setName(KeepName(UniqueName(name)));
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX CUDA mixture of experts: experts=%d "
               "top_k=%d model_dim=%d hidden_dim=%d rows=%d weights at %llu%s",
               moe->experts, moe->top_k, moe->model_dim, moe->hidden_dim,
               moe->rows, static_cast<unsigned long long>(source.offsets[0]),
               block.has_value()     ? " (the layer's whole block)"
               : routing.has_value() ? " (router fused)"
                                     : "");
    if (block.has_value()) {
      // The ops the block replaced never lower; its output is the next
      // residual stream, in that tensor's shape (the plugin's output has the
      // residual input's, which the graph may reshape between the two).
      for (LiteRtOp absorbed : block->absorbed) {
        unlowered_ops_.insert(absorbed);
      }
      LITERT_ASSIGN_OR_RETURN(auto out_type, block->output.RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto out_dims, ConvertDims(out_type));
      LITERT_ASSIGN_OR_RETURN(auto* result,
                              ReshapeTensor(layer->getOutput(0), out_dims));
      return SetOutputTensor(block->output, result);
    }
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  // ===========================================================================
  // Statically quantized products (trtllm/quant_linear.h,
  // quant_linear_plugin.h).
  //
  // Such a model keeps INT8 activations through every product: QUANTIZE ->
  // FULLY_CONNECTED or BATCH_MATMUL (INT4 per-channel or centered INT2 weights,
  // an INT8 result with its own scale) -> DEQUANTIZE, with the RMS norms, the
  // residual update and the rotary embedding in floats between them. The
  // rounding of every INT8 result is part of the trained function, so these ops
  // do not take the FP16-activation lowering, which carries INT8 tensors as
  // reals. FindQuantRegions matches the model's op sequences and replaces each
  // with layers of the quant plugin, which round as LiteRT's CPU kernels do:
  //   projections  FULLY_CONNECTED ops that read one INT8 tensor, one launch
  //   MLP          a factorized MLP, four launches (MatchQuantFactorizedMlp,
  //                LowerQuantFactorizedMlp)
  //   glue         the float steps in front of a QUANTIZE, one launch, or
  //                part of the projections' launch that reads its result
  //   qkv rope     the head norms and odml.rope behind q, k and v, one launch
  //   head         the vocabulary product behind its dynamic quantization
  // INT8 results stay INT8 between the layers (quant_values_) and float ones
  // FLOAT16. A region's ops are not lowered one by one: the region is emitted
  // at its first op, by which its inputs exist.

  // The INT8 values of a model tensor that a quant layer produced: columns
  // [offset, offset + width) of the rows of `int8`.
  struct QuantValue {
    nvinfer1::ITensor* int8 = nullptr;
    int32_t offset = 0;
    int32_t width = 0;
    float scale = 0.0f;
  };

  // FLOAT16 -> CAST -> odml.rms_norm -> CAST -> FLOAT16.
  struct QuantNorm {
    Tensor input;
    Tensor weights;
    float eps = 1e-6f;
  };

  // The float steps that end in the FLOAT16 tensor a QUANTIZE or the head
  // reads (LiteRtNvidiaQuantGlueParams): an optional residual update
  //   x' = x * c_x + rms_norm(DEQUANTIZE(branch)) * c_y
  // and an optional RMS norm of the result.
  struct QuantGlueChain {
    std::optional<Tensor> x;
    std::optional<Tensor> updated;  // x'
    std::optional<Tensor> branch;   // INT8, or the FLOAT16 of a quant layer
    std::optional<QuantNorm> branch_norm;
    float c_x = 1.0f;
    float c_y = 1.0f;
    std::optional<QuantNorm> norm;
    std::optional<Tensor> normed;  // the norm's FLOAT16 result
    std::vector<LiteRtOp> members;
  };

  struct QuantGlue {
    QuantGlueChain chain;
    // The QUANTIZE's INT8 result, or the FLOAT16 x of a carved factorized MLP,
    // whose QUANTIZE is inside the composite; and its INT8 scale.
    std::optional<Tensor> quantized;
    float scale = 0.0f;
  };

  // A glue as a layer runs it (PrepareQuantGlue): its kernel parameters,
  // its inputs (x, then y and the branch norm's weights, then the norm's
  // weights) and its outputs (the QuantGlueOutput mask).
  struct QuantGlueLayer {
    const QuantGlueChain* chain = nullptr;
    LiteRtNvidiaQuantGlueParams params = {};
    std::vector<nvinfer1::ITensor*> inputs;
    int32_t outputs = 0;
  };

  struct QuantProjections {
    std::optional<Tensor> input;
    std::vector<Op> ops;  // in row order
    // The glue (quant_glues_) whose INT8 result is `input`, run by the
    // product's launch.
    std::optional<size_t> glue;
  };

  // A factorized MLP on INT8 activations, all weights centered INT2:
  //   x [m, 2 C] -> two FULLY_CONNECTED [R1, C] over its halves, regrouped
  //   per block [m, B, 2 R1 / B] -> BATCH_MATMUL [B, 2 R1 / B, N] -> SiLU ->
  //   BATCH_MATMUL [B, N, 2 D] -> its halves [m, B D] each -> two
  //   FULLY_CONNECTED [R4, B D] -> y [m, 2 R4].
  struct QuantFactorizedMlp {
    // x: an INT8 tensor, or a float one to quantize with scales[0].
    std::optional<Tensor> input;
    // up-1 a, up-1 b, up-2, down-1, down-2 a, down-2 b.
    std::vector<Tensor> weights;
    // The INT8 scales of x, up-1, up-2, the SiLU, down-1 and down-2 (y).
    float scales[6] = {};
    // y: the INT8 result, or the FLOAT16 tensor its DEQUANTIZE makes.
    std::optional<Tensor> output;
    // The glue (quant_glues_) whose INT8 result is x, run by the MLP's
    // launch.
    std::optional<size_t> glue;
  };

  struct QuantQkvRope {
    size_t projections = 0;
    // Per projection in row order: its role (0 q, 1 k, 2 v), the head norm
    // and the FLOAT16 result (behind the rotation and the query multiplier).
    std::vector<int> roles;
    std::vector<QuantNorm> norms;
    std::vector<Tensor> outputs;
    std::optional<Tensor> positions;
    QuantRope rope;
    int32_t depth = 0;
    float q_multiplier = 1.0f;
  };

  struct QuantHead {
    QuantGlueChain chain;
    std::optional<Op> product;
    std::optional<Tensor> logits;  // behind the soft cap when it matched
    float cap_in = 0.0f;
    float cap_out = 0.0f;
  };

  enum class QuantRegionKind {
    kProjections,
    kFactorizedMlp,
    kGlue,
    kQkvRope,
    kHead
  };
  struct QuantRegion {
    QuantRegionKind kind;
    size_t index;
  };

  static float RoundFp16(float value) {
    return HalfBitsToFloat(Fp32ToFp16Bits(value));
  }

  static int32_t QuantTypeOf(const nvinfer1::ITensor* tensor) {
    switch (tensor->getType()) {
      case nvinfer1::DataType::kINT8:
        return kLiteRtNvidiaQuantInt8;
      case nvinfer1::DataType::kHALF:
        return kLiteRtNvidiaQuantFp16;
      case nvinfer1::DataType::kBF16:
        return kLiteRtNvidiaQuantBf16;
      default:
        return kLiteRtNvidiaQuantFp32;
    }
  }

  static int64_t QuantElements(const nvinfer1::ITensor* tensor) {
    const auto dims = tensor->getDimensions();
    int64_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) count *= dims.d[i];
    return count;
  }

  // `tensor` as rows [m, width].
  Expected<nvinfer1::ITensor*> QuantRows(nvinfer1::ITensor* tensor,
                                         int64_t width) {
    const auto dims = tensor->getDimensions();
    const int64_t count = QuantElements(tensor);
    if (width <= 0 || count <= 0 || count % width != 0) {
      return Error(kLiteRtStatusErrorCompilation,
                   "A quantized product's tensor does not hold whole rows");
    }
    if (dims.nbDims == 2 && dims.d[1] == width) return tensor;
    nvinfer1::Dims rows{};
    rows.nbDims = 2;
    rows.d[0] = count / width;
    rows.d[1] = width;
    return ReshapeTensor(tensor, rows);
  }

  // The rows [m, width] of the INT8 values of one tensor.
  Expected<nvinfer1::ITensor*> QuantInt8Rows(const QuantValue& value) {
    const auto dims = value.int8->getDimensions();
    if (dims.nbDims != 2) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Quant layer results are rows");
    }
    if (value.offset == 0 && dims.d[1] == value.width) return value.int8;
    nvinfer1::Dims start{};
    start.nbDims = 2;
    start.d[1] = value.offset;
    nvinfer1::Dims size{};
    size.nbDims = 2;
    size.d[0] = dims.d[0];
    size.d[1] = value.width;
    nvinfer1::Dims stride{};
    stride.nbDims = 2;
    stride.d[0] = stride.d[1] = 1;
    auto* layer = network_->addSlice(*value.int8, start, size, stride);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to slice a quant layer's result");
    }
    return layer->getOutput(0);
  }

  // The float value of `tensor` at the best precision the network has: the
  // FLOAT16 a quant layer made, a float network input as the runtime gives
  // it (also behind the model's CAST to FLOAT16: the kernels round as the
  // cast does), else the activation-typed tensor, which the FP16-activation
  // mode may have narrowed.
  Expected<nvinfer1::ITensor*> QuantFloatSource(const Tensor& tensor) {
    if (auto it = quant_floats_.find(tensor.Get()); it != quant_floats_.end()) {
      return it->second;
    }
    const auto raw_float = [&](const Tensor& input) -> nvinfer1::ITensor* {
      const auto raw = raw_input_map_.find(input.Get());
      return raw != raw_input_map_.end() &&
                     (raw->second->getType() == nvinfer1::DataType::kFLOAT ||
                      raw->second->getType() == nvinfer1::DataType::kHALF)
                 ? raw->second
                 : nullptr;
    };
    if (auto* raw = raw_float(tensor); raw != nullptr) {
      return raw;
    }
    if (auto cast = tensor.GetDefiningOp();
        cast && cast->Code() == kLiteRtOpCodeTflCast &&
        tensor.ElementType() == litert::ElementType::Float16 &&
        cast->Inputs().size() == 1) {
      if (auto* raw = raw_float(cast->Inputs()[0]); raw != nullptr) {
        return raw;
      }
    }
    return GetTensor(tensor);
  }

  // Records the FLOAT16 result of a quant layer as the value of `tensor`.
  Expected<void> SetQuantFloat(const Tensor& tensor, nvinfer1::ITensor* value) {
    quant_floats_[tensor.Get()] = value;
    return SetOutputTensor(tensor, value);
  }

  // The activation rows a product reads through its INT8 tensor `input`:
  // the INT8 values a quant layer made, or the float input of its QUANTIZE
  // for the kernel to quantize.
  struct QuantActivation {
    nvinfer1::ITensor* rows = nullptr;
    int32_t type = kLiteRtNvidiaQuantInt8;
    float quantize_scale = 0.0f;
  };

  Expected<QuantActivation> GetQuantActivation(const Tensor& input,
                                               int64_t width) {
    Tensor tensor = input;
    // The transposes between the QUANTIZE and the product (the prefill's
    // attention output is quantized head-major), last first.
    std::vector<Op> transposes;
    for (int hops = 0; hops < 6; ++hops) {
      if (auto it = quant_values_.find(tensor.Get());
          it != quant_values_.end() && transposes.empty()) {
        LITERT_ASSIGN_OR_RETURN(auto* rows, QuantInt8Rows(it->second));
        LITERT_ASSIGN_OR_RETURN(rows, QuantRows(rows, width));
        return QuantActivation{rows, kLiteRtNvidiaQuantInt8, 0.0f};
      }
      if (auto raw = raw_input_map_.find(tensor.Get());
          raw != raw_input_map_.end() && transposes.empty() &&
          raw->second->getType() == nvinfer1::DataType::kINT8) {
        LITERT_ASSIGN_OR_RETURN(auto* rows, QuantRows(raw->second, width));
        return QuantActivation{rows, kLiteRtNvidiaQuantInt8, 0.0f};
      }
      auto producer = tensor.GetDefiningOp();
      if (!producer || producer->Inputs().empty()) break;
      if (producer->Code() == kLiteRtOpCodeTflQuantize) {
        // Rounding commutes with moving elements: the float input takes the
        // transposes and the kernel quantizes what the product reads.
        LITERT_ASSIGN_OR_RETURN(auto* source,
                                QuantFloatSource(producer->Inputs()[0]));
        for (auto op = transposes.rbegin(); op != transposes.rend(); ++op) {
          LITERT_ASSIGN_OR_RETURN(auto type,
                                  op->Inputs()[0].RankedTensorType());
          LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
          LITERT_ASSIGN_OR_RETURN(auto perm,
                                  ReadInt32Constant(op->Inputs()[1]));
          LITERT_ASSIGN_OR_RETURN(source, ReshapeTensor(source, dims));
          nvinfer1::Permutation order{};
          for (size_t i = 0; i < perm.size(); ++i) order.order[i] = perm[i];
          auto* layer = network_->addShuffle(*source);
          if (layer == nullptr || layer->getOutput(0) == nullptr ||
              perm.size() != static_cast<size_t>(dims.nbDims)) {
            return Error(kLiteRtStatusErrorCompilation,
                         "Failed to transpose a quantized product's input");
          }
          layer->setFirstTranspose(order);
          source = layer->getOutput(0);
        }
        LITERT_ASSIGN_OR_RETURN(auto* rows, QuantRows(source, width));
        return QuantActivation{
            rows, QuantTypeOf(rows),
            RoundFp16(1.0f / tensor.PerTensorQuantization().scale)};
      }
      if (producer->Code() == kLiteRtOpCodeTflTranspose &&
          producer->Inputs().size() == 2) {
        transposes.push_back(*producer);
      } else if (producer->Code() != kLiteRtOpCodeTflReshape) {
        break;
      }
      tensor = producer->Inputs()[0];
    }
    return Error(kLiteRtStatusErrorUnsupported,
                 "A quantized product reads an INT8 tensor that neither a "
                 "QUANTIZE nor a quant layer makes");
  }

  // The FP32 values of a constant FP32 tensor as a network constant.
  Expected<nvinfer1::ITensor*> QuantFloatConstant(const Tensor& tensor,
                                                  const std::string& name) {
    const auto bytes = tensor.Weights().Bytes();
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
    nvinfer1::Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = static_cast<int64_t>(values.size());
    return AddFloatConstant(values, dims, name, nvinfer1::DataType::kFLOAT);
  }

  // The weights input of a product layer: `members` are the packed weights
  // of its matrices in row order, views of the model file when they are the
  // model's. With a weight store they stay there and the layer reads the
  // holder of their segment (`location`), else they are copied into a
  // constant.
  struct QuantWeights {
    nvinfer1::ITensor* tensor = nullptr;
    std::optional<TensorRtWeightLocation> location;
  };

  Expected<QuantWeights> AddQuantWeights(
      const LiteRtNvidiaQuantLinearParams& params,
      absl::Span<const absl::Span<const uint8_t>> members) {
    size_t bytes = 0;
    for (const auto& member : members) bytes += member.size();
    if (bytes != static_cast<size_t>(params.rows) * params.columns *
                     params.bit_width / 8) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Unexpected weights of a quantized product");
    }
    QuantWeights weights;
    if (weight_store_ != nullptr) {
      auto added = weight_store_->Add(members);
      if (added.HasValue()) {
        weights.location = *added;
      } else if (added.Error().Status() != kLiteRtStatusErrorNotFound) {
        return added.Error();
      }
    }
    if (weights.location.has_value()) {
      // The holder of the segment replaces this once the segments of the
      // partition are final (AddWeightHolders).
      LITERT_ASSIGN_OR_RETURN(weights.tensor, ProvisionalWeightHolder());
      return weights;
    }
    std::vector<uint8_t> packed;
    packed.reserve(bytes);
    for (const auto& member : members) {
      packed.insert(packed.end(), member.begin(), member.end());
    }
    owned_weights_.push_back(std::move(packed));
    nvinfer1::Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = static_cast<int64_t>(bytes);
    auto* constant = network_->addConstant(
        dims, nvinfer1::Weights{nvinfer1::DataType::kINT8,
                                owned_weights_.back().data(),
                                static_cast<int64_t>(bytes)});
    if (constant == nullptr || constant->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the weights of a quantized product");
    }
    weights.tensor = constant->getOutput(0);
    return weights;
  }

  // The INT8 table of a quantized product as a constant.
  Expected<nvinfer1::ITensor*> AddQuantTable(const std::vector<int8_t>& lut) {
    owned_weights_.emplace_back(lut.begin(), lut.end());
    nvinfer1::Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = static_cast<int64_t>(lut.size());
    auto* constant = network_->addConstant(
        dims, nvinfer1::Weights{nvinfer1::DataType::kINT8,
                                owned_weights_.back().data(), dims.d[0]});
    if (constant == nullptr || constant->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the table of a quantized product");
    }
    return constant->getOutput(0);
  }

  // Adds a product layer on the weights `members` (AddQuantWeights).
  // Returns INT8 [m, rows], or the FP32 logits. With `glue` the layer runs
  // that glue in front and reads its x instead of `activation` (an INT8
  // activation, the glue's q).
  Expected<nvinfer1::ITensor*> AddQuantLinear(
      const LiteRtNvidiaQuantLinearParams& params,
      nvinfer1::ITensor* activation,
      absl::Span<const absl::Span<const uint8_t>> members,
      const std::vector<float>& multipliers, const std::vector<int8_t>* lut,
      nvinfer1::ITensor* row_scale, const std::string& name,
      const QuantGlueLayer* glue = nullptr) {
    if (multipliers.size() != static_cast<size_t>(params.rows)) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Unexpected weights of a quantized product");
    }
    LITERT_ASSIGN_OR_RETURN(const QuantWeights added,
                            AddQuantWeights(params, members));
    nvinfer1::ITensor* weights = added.tensor;
    const std::optional<TensorRtWeightLocation>& location = added.location;
    nvinfer1::Dims multiplier_dims{};
    multiplier_dims.nbDims = 1;
    multiplier_dims.d[0] = params.rows;
    LITERT_ASSIGN_OR_RETURN(
        auto* multiplier_tensor,
        AddFloatConstant(multipliers, multiplier_dims, name + "_multipliers",
                         nvinfer1::DataType::kFLOAT));
    std::vector<nvinfer1::ITensor*> inputs = {
        glue != nullptr ? glue->inputs[0] : activation, weights,
        multiplier_tensor};
    if (lut != nullptr) {
      LITERT_ASSIGN_OR_RETURN(auto* table, AddQuantTable(*lut));
      inputs.push_back(table);
    }
    if (row_scale != nullptr) {
      inputs.push_back(row_scale);
    }
    const int64_t weight_offset =
        location.has_value() ? static_cast<int64_t>(location->offset) : 0;
    const int64_t holder_granule =
        location.has_value() ? static_cast<int64_t>(weight_store_->granule())
                             : 0;
    TrtPtr<nvinfer1::IPluginV3> plugin;
    if (glue != nullptr) {
      inputs.insert(inputs.end(), glue->inputs.begin() + 1, glue->inputs.end());
      plugin.reset(CreateQuantGlueLinearPlugin(
          glue->params, params,
          glue->outputs & (kQuantGlueOutputX | kQuantGlueOutputZ),
          weight_offset, holder_granule));
    } else {
      plugin.reset(
          CreateQuantLinearPlugin(params, weight_offset, holder_granule));
    }
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the quantized product plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the quantized product plugin layer");
    }
    if (location.has_value()) {
      holder_readers_.push_back(
          {layer, location->segment, /*scales_too=*/false});
    }
    layer->setName(KeepName(UniqueName(name)));
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    uses_cuda_subbyte_gemv_ = true;
    if (glue != nullptr) {
      int32_t next = 1;
      LITERT_RETURN_IF_ERROR(SetQuantGlueFloats(*glue, layer, next));
    }
    return layer->getOutput(0);
  }

  // Records the INT8 values of a model tensor: `width` columns from `offset`
  // of the quant layer result `int8`.
  void SetQuantValue(const Tensor& tensor, nvinfer1::ITensor* int8,
                     int32_t offset, int32_t width, float scale) {
    quant_values_[tensor.Get()] = {int8, offset, width, scale};
  }

  // The activation-typed real view of an INT8 tensor that a quant layer
  // produced, for an op outside the quant regions that reads it.
  Expected<nvinfer1::ITensor*> QuantRealView(const Tensor& tensor,
                                             const QuantValue& value) {
    LITERT_ASSIGN_OR_RETURN(auto* rows, QuantInt8Rows(value));
    LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
    LITERT_ASSIGN_OR_RETURN(rows, ReshapeTensor(rows, dims));
    if (tensor.ElementType() == litert::ElementType::Int8) {
      return AddDequantizeTensor(tensor, rows, ModeFloatType());
    }
    // The FLOAT16 result of a carved composite: the scale is the layer's.
    nvinfer1::Dims scalar{};
    const float scale[] = {value.scale};
    LITERT_ASSIGN_OR_RETURN(
        auto* scale_tensor,
        AddFloatConstant(
            scale, scalar,
            "quant_view_scale_" + std::to_string(tensor.TensorIndex()),
            QuantizationScaleType(ModeFloatType())));
    auto* layer =
        network_->addDequantize(*rows, *scale_tensor, ModeFloatType());
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to dequantize a quant layer's result");
    }
    return layer->getOutput(0);
  }

  bool QuantOpsAreFree(const std::vector<LiteRtOp>& ops) const {
    for (LiteRtOp op : ops) {
      if (op_order_.count(op) == 0 || unlowered_ops_.count(op) != 0) {
        return false;
      }
    }
    return !ops.empty();
  }

  // Makes `members` a region emitted at its first op.
  void AddQuantRegion(QuantRegionKind kind, size_t index,
                      const std::vector<LiteRtOp>& members) {
    LiteRtOp first = members.front();
    for (LiteRtOp op : members) {
      if (op_order_[op] < op_order_[first]) first = op;
    }
    unlowered_ops_.insert(members.begin(), members.end());
    quant_regions_[first] = {kind, index};
  }

  // Adds the glue quant_glues_.back() of `members` to the region that reads its
  // INT8 result `read` and runs the glue in its launch: the projections
  // (QuantProjections::glue), or the factorized MLP of a decode step's one row
  // (QuantFactorizedMlp::glue). The region is emitted at its first op, now the
  // glue's. Any other glue is a region of its own (a prefill chunk's factorized
  // MLP starts with rows too short to hide a glue).
  void AddQuantGlueRegion(const Tensor& read,
                          const std::vector<LiteRtOp>& members) {
    const size_t index = quant_glues_.size() - 1;
    QuantRegion owner;
    const auto projections = quant_projections_by_input_.find(read.Get());
    const auto factorized_mlp =
        quant_factorized_mlps_by_input_.find(read.Get());
    if (projections != quant_projections_by_input_.end()) {
      quant_projections_[projections->second].glue = index;
      owner = {QuantRegionKind::kProjections, projections->second};
    } else if (factorized_mlp != quant_factorized_mlps_by_input_.end() &&
               QuantOneRow(read)) {
      quant_factorized_mlps_[factorized_mlp->second].glue = index;
      owner = {QuantRegionKind::kFactorizedMlp, factorized_mlp->second};
    } else {
      AddQuantRegion(QuantRegionKind::kGlue, index, members);
      return;
    }
    LiteRtOp first = members.front();
    for (LiteRtOp op : members) {
      if (op_order_[op] < op_order_[first]) first = op;
    }
    for (auto it = quant_regions_.begin(); it != quant_regions_.end(); ++it) {
      if (it->second.kind == owner.kind && it->second.index == owner.index) {
        if (op_order_[it->first] < op_order_[first]) first = it->first;
        quant_regions_.erase(it);
        break;
      }
    }
    unlowered_ops_.insert(members.begin(), members.end());
    quant_regions_[first] = owner;
  }

  // Whether `tensor` [..., n] holds one row.
  static bool QuantOneRow(const Tensor& tensor) {
    const auto type = tensor.RankedTensorType();
    if (!type) return false;
    const auto dims = type->Layout().Dimensions();
    int64_t rows = 1;
    for (size_t d = 0; d + 1 < dims.size(); ++d) rows *= dims[d];
    return !dims.empty() && rows == 1;
  }

  // The glue in front of the projections (QuantProjections::glue) as their
  // layer runs it, if the kernel takes it with `params` (which read the
  // glue's INT8 result); else the glue gets its own layer here, whose INT8
  // result the projections then read.
  Expected<std::optional<QuantGlueLayer>> QuantGlueInFront(
      std::optional<size_t> index,
      const LiteRtNvidiaQuantLinearParams& params) {
    if (!index.has_value()) return std::optional<QuantGlueLayer>();
    const QuantGlue& glue = quant_glues_[*index];
    LITERT_ASSIGN_OR_RETURN(
        auto layer,
        PrepareQuantGlue(glue.chain, kLiteRtNvidiaQuantGlueStatic, glue.scale,
                         "cuda_quant_glue_" +
                             std::to_string(glue.quantized->TensorIndex())));
    if (LiteRtNvidiaQuantGlueLinearSupports(&layer.params, &params)) {
      return std::optional<QuantGlueLayer>(std::move(layer));
    }
    LITERT_RETURN_IF_ERROR(LowerQuantGlue(glue));
    return std::optional<QuantGlueLayer>();
  }

  // The op of `code` that computes `tensor`, when one op reads `tensor`.
  std::optional<Op> QuantProducer(const Tensor& tensor,
                                  LiteRtOpCode code) const {
    auto producer = tensor.GetDefiningOp();
    if (!producer || producer->Code() != code || !ReadOnce(tensor)) {
      return std::nullopt;
    }
    return *producer;
  }

  // The op that alone reads `tensor`, if its code is `code`.
  std::optional<Op> QuantUser(const Tensor& tensor, LiteRtOpCode code) const {
    if (!ReadOnce(tensor)) return std::nullopt;
    Op user = tensor.Uses()[0].user;
    if (user.Code() != code || user.Outputs().size() != 1) return std::nullopt;
    return user;
  }

  // `tensor` = CAST(odml.rms_norm(CAST(input), weights)) over the last dim,
  // FLOAT16 in and out. `shared`: other ops may read `tensor` too.
  std::optional<QuantNorm> MatchQuantNorm(
      const Tensor& tensor, bool shared, std::vector<LiteRtOp>& members) const {
    auto narrow = tensor.GetDefiningOp();
    if (!narrow || narrow->Code() != kLiteRtOpCodeTflCast ||
        narrow->Inputs().size() != 1 ||
        tensor.ElementType() != litert::ElementType::Float16 ||
        narrow->Inputs()[0].ElementType() != litert::ElementType::Float32 ||
        (!shared && !ReadOnce(tensor))) {
      return std::nullopt;
    }
    auto norm = QuantProducer(narrow->Inputs()[0], kLiteRtOpCodeShloComposite);
    if (!norm || CompositeOpName(*norm) != "odml.rms_norm" ||
        norm->Inputs().size() != 2 || norm->Outputs().size() != 1) {
      return std::nullopt;
    }
    auto widen = QuantProducer(norm->Inputs()[0], kLiteRtOpCodeTflCast);
    if (!widen || widen->Inputs().size() != 1 ||
        widen->Inputs()[0].ElementType() != litert::ElementType::Float16) {
      return std::nullopt;
    }
    const Tensor weights = norm->Inputs()[1];
    const auto type = tensor.RankedTensorType();
    if (!type || type->Layout().Rank() < 1 || !weights.HasWeights() ||
        weights.ElementType() != litert::ElementType::Float32 ||
        weights.Weights().Bytes().size() !=
            static_cast<size_t>(type->Layout().Dimensions().back()) *
                sizeof(float)) {
      return std::nullopt;
    }
    QuantNorm result{widen->Inputs()[0], weights, 1e-6f};
    CompositeOptions options;
    if (options.InitFromOp(norm->Get()) == kLiteRtStatusOk &&
        options.attributes_map.has_value() &&
        !options.attributes_map.value()["epsilon"].IsNull()) {
      result.eps = options.attributes_map.value()["epsilon"].AsFloat();
    }
    members.push_back(widen->Get());
    members.push_back(norm->Get());
    members.push_back(narrow->Get());
    return result;
  }

  // The float steps that end in `z` (QuantGlueChain), matched backwards.
  // `shared`: ops outside the chain may read `z` too.
  QuantGlueChain MatchQuantGlueChain(const Tensor& z, bool shared) const {
    QuantGlueChain chain;
    Tensor value = z;
    for (int hops = 0; hops < 3 && !shared; ++hops) {
      auto reshape = QuantProducer(value, kLiteRtOpCodeTflReshape);
      if (!reshape || reshape->Inputs().empty()) break;
      chain.members.push_back(reshape->Get());
      value = reshape->Inputs()[0];
    }
    chain.x = value;
    chain.norm = MatchQuantNorm(value, shared, chain.members);
    if (!chain.norm.has_value()) {
      return chain;
    }
    chain.normed = value;
    chain.x = chain.norm->input;
    // x' = ADD(MUL(x, c_x), MUL(rms_norm(DEQUANTIZE(branch)), c_y)).
    const Tensor stream = chain.norm->input;
    auto add = stream.GetDefiningOp();
    if (!add || add->Code() != kLiteRtOpCodeTflAdd ||
        add->Inputs().size() != 2 || !HasNoFusedActivation(*add)) {
      return chain;
    }
    struct Side {
      Tensor value;
      float scalar;
      LiteRtOp multiply;
    };
    std::vector<Side> sides;
    for (int i = 0; i < 2; ++i) {
      auto multiply = QuantProducer(add->Inputs()[i], kLiteRtOpCodeTflMul);
      if (!multiply || multiply->Inputs().size() != 2 ||
          !HasNoFusedActivation(*multiply)) {
        return chain;
      }
      for (int j = 0; j < 2; ++j) {
        if (auto scalar = ReadFloatScalar(multiply->Inputs()[j])) {
          sides.push_back(
              {multiply->Inputs()[1 - j], *scalar, multiply->Get()});
          break;
        }
      }
    }
    if (sides.size() != 2) return chain;
    for (int i = 0; i < 2; ++i) {
      std::vector<LiteRtOp> members;
      auto norm = MatchQuantNorm(sides[i].value, /*shared=*/false, members);
      if (!norm.has_value()) continue;
      // The branch: a DEQUANTIZE of a quant layer's INT8 result, or the
      // FLOAT16 tensor a quant layer stands for.
      Tensor source = norm->input;
      if (quant_outputs_.count(source.Get()) == 0) {
        auto dequantize = QuantProducer(source, kLiteRtOpCodeTflDequantize);
        if (!dequantize || dequantize->Inputs().size() != 1) continue;
        members.push_back(dequantize->Get());
        source = dequantize->Inputs()[0];
        for (int hops = 0; hops < 3 && quant_outputs_.count(source.Get()) == 0;
             ++hops) {
          auto reshape = QuantProducer(source, kLiteRtOpCodeTflReshape);
          if (!reshape || reshape->Inputs().empty()) break;
          members.push_back(reshape->Get());
          source = reshape->Inputs()[0];
        }
        if (quant_outputs_.count(source.Get()) == 0 || !ReadOnce(source)) {
          continue;
        }
      }
      chain.branch = source;
      chain.branch_norm = *norm;
      chain.c_y = sides[i].scalar;
      chain.c_x = sides[1 - i].scalar;
      chain.x = sides[1 - i].value;
      chain.updated = stream;
      chain.members.insert(chain.members.end(), members.begin(), members.end());
      chain.members.push_back(sides[0].multiply);
      chain.members.push_back(sides[1].multiply);
      chain.members.push_back(add->Get());
      break;
    }
    return chain;
  }

  // The parameters and inputs of the glue of `chain` with the quantization
  // `quantize` (LiteRtNvidiaQuantGlueQuantize) to `scale`, for its own layer
  // (AddQuantGlue) or the layer of the product behind it (AddQuantLinear).
  Expected<QuantGlueLayer> PrepareQuantGlue(const QuantGlueChain& chain,
                                            int32_t quantize, float scale,
                                            const std::string& name) {
    LITERT_ASSIGN_OR_RETURN(auto x_type, chain.x->RankedTensorType());
    const int32_t width = x_type.Layout().Dimensions().back();
    LiteRtNvidiaQuantGlueParams params = {};
    params.width = width;
    LITERT_ASSIGN_OR_RETURN(auto* x, QuantFloatSource(*chain.x));
    LITERT_ASSIGN_OR_RETURN(x, QuantRows(x, width));
    params.x_type = QuantTypeOf(x);
    params.z_type = kLiteRtNvidiaQuantFp16;
    params.quantize = quantize;
    params.quantize_scale = quantize == kLiteRtNvidiaQuantGlueStatic
                                ? RoundFp16(1.0f / scale)
                                : 0.0f;
    std::vector<nvinfer1::ITensor*> inputs = {x};
    int32_t outputs = kQuantGlueOutputQ;
    if (chain.branch.has_value()) {
      const auto value = quant_values_.find(chain.branch->Get());
      if (value == quant_values_.end()) {
        return Error(kLiteRtStatusErrorCompilation,
                     "The residual branch of a quant glue has no INT8 values");
      }
      LITERT_ASSIGN_OR_RETURN(auto* y, QuantInt8Rows(value->second));
      LITERT_ASSIGN_OR_RETURN(y, QuantRows(y, width));
      LITERT_ASSIGN_OR_RETURN(auto* weights,
                              QuantFloatConstant(chain.branch_norm->weights,
                                                 name + "_branch_norm"));
      inputs.push_back(y);
      inputs.push_back(weights);
      params.branch = 1;
      params.branch_scale = RoundFp16(value->second.scale);
      params.branch_eps = chain.branch_norm->eps;
      params.c_x = chain.c_x;
      params.c_y = chain.c_y;
      outputs |= kQuantGlueOutputX;
    }
    if (chain.norm.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          auto* weights,
          QuantFloatConstant(chain.norm->weights, name + "_norm"));
      inputs.push_back(weights);
      params.norm = 1;
      params.norm_eps = chain.norm->eps;
    }
    // The norm's result is an output for the ops outside the chain that read
    // it (the head's hidden state is a model output too).
    const bool z_output = chain.normed.has_value() && !ReadOnce(*chain.normed);
    if (z_output) outputs |= kQuantGlueOutputZ;
    if (quantize == kLiteRtNvidiaQuantGlueDynamic) {
      outputs |= kQuantGlueOutputRowScale;
    }
    return QuantGlueLayer{&chain, params, std::move(inputs), outputs};
  }

  // Records the outputs x' and z of a layer that ran `glue`, from output
  // `next` on, as the FLOAT16 values of their tensors.
  Expected<void> SetQuantGlueFloats(const QuantGlueLayer& glue,
                                    nvinfer1::ILayer* layer, int32_t& next) {
    const auto float_output = [&](const Tensor& tensor) -> Expected<void> {
      LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
      LITERT_ASSIGN_OR_RETURN(auto* value,
                              ReshapeTensor(layer->getOutput(next++), dims));
      return SetQuantFloat(tensor, value);
    };
    if ((glue.outputs & kQuantGlueOutputX) != 0) {
      LITERT_RETURN_IF_ERROR(float_output(*glue.chain->updated));
    }
    if ((glue.outputs & kQuantGlueOutputZ) != 0) {
      LITERT_RETURN_IF_ERROR(float_output(*glue.chain->normed));
    }
    return {};
  }

  // Adds the glue layer of `chain` with the quantization `quantize`
  // (LiteRtNvidiaQuantGlueQuantize) to `scale`. Returns its INT8 result
  // [m, width]; `row_scale`, when not null, receives the dynamic scales.
  // The updated residual stream and the norm's result (for other readers)
  // become the FLOAT16 values of their tensors.
  Expected<nvinfer1::ITensor*> AddQuantGlue(const QuantGlueChain& chain,
                                            int32_t quantize, float scale,
                                            nvinfer1::ITensor** row_scale,
                                            const std::string& name) {
    LITERT_ASSIGN_OR_RETURN(auto glue,
                            PrepareQuantGlue(chain, quantize, scale, name));
    TrtPtr<nvinfer1::IPluginV3> plugin(
        CreateQuantGluePlugin(glue.params, glue.outputs));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the quant glue plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, glue.inputs.data(), static_cast<int32_t>(glue.inputs.size()),
        *plugin);
    if (layer == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the quant glue plugin layer");
    }
    layer->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    int32_t next = 0;
    LITERT_RETURN_IF_ERROR(SetQuantGlueFloats(glue, layer, next));
    nvinfer1::ITensor* q = layer->getOutput(next++);
    q->setName(KeepName(UniqueName(name + "_q")));
    if (row_scale != nullptr) {
      *row_scale = layer->getOutput(next++);
    }
    return q;
  }

  // The SiLU between two INT8 tensors as the model computes it, per INT8
  // value: DEQUANTIZE (`in_scale`), x * logistic(x) in FLOAT16, QUANTIZE
  // (`out_scale`).
  static std::vector<int8_t> QuantSiluTable(float in_scale, float out_scale) {
    std::vector<int8_t> table(256);
    const float dequantize = RoundFp16(in_scale);
    const float quantize = RoundFp16(1.0f / out_scale);
    for (int q = -128; q <= 127; ++q) {
      const float x = RoundFp16(static_cast<float>(q) * dequantize);
      const float logistic = RoundFp16(
          static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(x)))));
      const float y = RoundFp16(x * logistic);
      const float scaled = y * quantize;
      table[q + 128] = static_cast<int8_t>(
          std::max(-128L, std::min(127L, std::lrintf(scaled))));
    }
    return table;
  }

  // Packed INT2 BATCH_MATMUL weights [blocks, columns, outputs] as the rows
  // [blocks * outputs, columns] of a FULLY_CONNECTED per block.
  static std::vector<uint8_t> QuantTransposedBlocks(
      absl::Span<const uint8_t> packed, int32_t blocks, int32_t columns,
      int32_t outputs) {
    std::vector<uint8_t> rows(packed.size(), 0);
    for (int32_t b = 0; b < blocks; ++b) {
      for (int32_t k = 0; k < columns; ++k) {
        const size_t from = (static_cast<size_t>(b) * columns + k) * outputs;
        for (int32_t n = 0; n < outputs; ++n) {
          const uint8_t field =
              (packed[(from + n) / 4] >> (2 * ((from + n) % 4))) & 3;
          const size_t to =
              (static_cast<size_t>(b) * outputs + n) * columns + k;
          rows[to / 4] |= field << (2 * (to % 4));
        }
      }
    }
    return rows;
  }

  // The dims [.., blocks, columns, outputs] of BATCH_MATMUL weights.
  static std::array<int32_t, 3> QuantBlockDims(const Tensor& weights) {
    const auto type = weights.RankedTensorType();
    const auto dims = type->Layout().Dimensions();
    const size_t rank = dims.size();
    return {dims[rank - 3], dims[rank - 2], dims[rank - 1]};
  }

  static std::pair<int32_t, int32_t> QuantMatrixDims(const Tensor& weights) {
    const auto type = weights.RankedTensorType();
    const auto dims = type->Layout().Dimensions();
    return {dims[0], dims[1]};
  }

  // Adds the layer of a factorized MLP's four products on one INT8 row
  // (CreateQuantFactorizedMlpPlugin): the weights `members` of each stage
  // (AddQuantWeights), its multipliers, and `lut` for the stages that have a
  // table. With `glue` the layer runs that glue in front and reads its x
  // instead of `activation`. Returns INT8 [1, rows of stage 3].
  Expected<nvinfer1::ITensor*> AddQuantFactorizedMlp(
      const LiteRtNvidiaQuantFactorizedMlpParams& stages,
      nvinfer1::ITensor* activation,
      const absl::Span<const absl::Span<const uint8_t>> (&members)[4],
      const std::vector<float> (&multipliers)[4],
      const std::vector<int8_t>& lut, const std::string& name,
      const QuantGlueLayer* glue = nullptr) {
    std::vector<nvinfer1::ITensor*> inputs = {glue != nullptr ? glue->inputs[0]
                                                              : activation};
    std::array<int64_t, 4> offsets = {};
    std::optional<TensorRtWeightLocation> locations[4];
    for (int s = 0; s < 4; ++s) {
      LITERT_ASSIGN_OR_RETURN(const QuantWeights added,
                              AddQuantWeights(stages.stages[s], members[s]));
      inputs.push_back(added.tensor);
      locations[s] = added.location;
      offsets[s] = added.location.has_value()
                       ? static_cast<int64_t>(added.location->offset)
                       : -1;
    }
    for (int s = 0; s < 4; ++s) {
      if (multipliers[s].size() != static_cast<size_t>(stages.stages[s].rows)) {
        return Error(kLiteRtStatusErrorCompilation,
                     "Unexpected weights of a quantized product");
      }
      nvinfer1::Dims dims{};
      dims.nbDims = 1;
      dims.d[0] = stages.stages[s].rows;
      LITERT_ASSIGN_OR_RETURN(
          auto* tensor,
          AddFloatConstant(multipliers[s], dims,
                           name + "_multipliers" + std::to_string(s),
                           nvinfer1::DataType::kFLOAT));
      inputs.push_back(tensor);
    }
    bool table = false;
    for (const auto& stage : stages.stages) table = table || stage.lut != 0;
    if (table) {
      LITERT_ASSIGN_OR_RETURN(auto* tensor, AddQuantTable(lut));
      inputs.push_back(tensor);
    }
    if (glue != nullptr) {
      inputs.insert(inputs.end(), glue->inputs.begin() + 1, glue->inputs.end());
    }
    bool holders = false;
    for (const int64_t offset : offsets) holders = holders || offset >= 0;
    TrtPtr<nvinfer1::IPluginV3> plugin(CreateQuantFactorizedMlpPlugin(
        stages, offsets,
        holders ? static_cast<int64_t>(weight_store_->granule()) : 0,
        glue != nullptr ? &glue->params : nullptr,
        glue != nullptr
            ? glue->outputs & (kQuantGlueOutputX | kQuantGlueOutputZ)
            : 0));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the factorized MLP plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the factorized MLP plugin layer");
    }
    for (int s = 0; s < 4; ++s) {
      if (locations[s].has_value()) {
        holder_readers_.push_back({layer, locations[s]->segment,
                                   /*scales_too=*/false, /*input=*/1 + s});
      }
    }
    layer->setName(KeepName(UniqueName(name)));
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    uses_cuda_subbyte_gemv_ = true;
    if (glue != nullptr) {
      int32_t next = 1;
      LITERT_RETURN_IF_ERROR(SetQuantGlueFloats(*glue, layer, next));
    }
    return layer->getOutput(0);
  }

  // Emits a factorized MLP as four product layers: the first quantizes a float
  // x itself, the second applies the SiLU as a table, and their INT8 results
  // are regrouped on the way out so that every stage reads rows.
  Expected<void> LowerQuantFactorizedMlp(const QuantFactorizedMlp& mlp) {
    if (mlp.weights.size() != 6 || !mlp.input.has_value() ||
        !mlp.output.has_value()) {
      return Error(kLiteRtStatusErrorCompilation, "Incomplete factorized MLP");
    }
    std::vector<std::vector<float>> scales;
    for (int i = 0; i < 6; ++i) {
      auto weight_scales = CenteredInt2Scales(
          mlp.weights[i],
          i == 2 || i == 3 ? QuantBlockRank(mlp.weights[i]) - 2 : 1);
      if (!weight_scales.has_value()) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "Factorized MLP weights are not centered INT2");
      }
      scales.push_back(std::move(*weight_scales));
    }
    const auto [up1_rows, up1_columns] = QuantMatrixDims(mlp.weights[0]);
    const auto up2 = QuantBlockDims(mlp.weights[2]);
    const auto down1 = QuantBlockDims(mlp.weights[3]);
    const auto [down2_rows, down2_columns] = QuantMatrixDims(mlp.weights[4]);
    const int32_t blocks = up2[0];
    if (QuantMatrixDims(mlp.weights[1]) !=
            std::make_pair(up1_rows, up1_columns) ||
        QuantMatrixDims(mlp.weights[5]) !=
            std::make_pair(down2_rows, down2_columns) ||
        blocks <= 0 || up1_rows % blocks != 0 ||
        up2[1] != 2 * up1_rows / blocks || down1[0] != blocks ||
        down1[1] != up2[2] || down1[2] % 2 != 0 ||
        down2_columns != blocks * down1[2] / 2) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Factorized MLP weights do not fit each other");
    }
    const std::string name = "cuda_quant_factorized_mlp_" +
                             std::to_string(mlp.output->TensorIndex());
    const auto ratio = [&](int from, int to) {
      return static_cast<double>(mlp.scales[from]) /
             static_cast<double>(mlp.scales[to]);
    };
    const auto bytes = [&](int i) { return mlp.weights[i].Weights().Bytes(); };
    const auto joined = [](const std::vector<float>& a,
                           const std::vector<float>& b) {
      std::vector<float> all(a);
      all.insert(all.end(), b.begin(), b.end());
      return all;
    };

    // The glue in front quantizes x for the MLP (an INT8 activation); its
    // layer, if the MLP's launch does not take it, comes first.
    std::optional<QuantGlueLayer> glue;
    QuantActivation activation;
    if (mlp.glue.has_value()) {
      const QuantGlue& in_front = quant_glues_[*mlp.glue];
      LITERT_ASSIGN_OR_RETURN(
          glue,
          PrepareQuantGlue(
              in_front.chain, kLiteRtNvidiaQuantGlueStatic, in_front.scale,
              "cuda_quant_glue_" +
                  std::to_string(in_front.quantized->TensorIndex())));
    } else if (mlp.input->ElementType() == litert::ElementType::Int8 ||
               quant_values_.count(mlp.input->Get()) != 0) {
      LITERT_ASSIGN_OR_RETURN(activation,
                              GetQuantActivation(*mlp.input, 2 * up1_columns));
    } else {
      LITERT_ASSIGN_OR_RETURN(auto* source, QuantFloatSource(*mlp.input));
      LITERT_ASSIGN_OR_RETURN(activation.rows,
                              QuantRows(source, 2 * up1_columns));
      activation.type = QuantTypeOf(activation.rows);
      activation.quantize_scale = RoundFp16(1.0f / mlp.scales[0]);
    }
    LiteRtNvidiaQuantFactorizedMlpParams stages = {};
    LiteRtNvidiaQuantLinearParams& params = stages.stages[0];
    params.layout = kLiteRtNvidiaQuantLayoutRows;
    params.bit_width = 2;
    params.rows = 2 * up1_rows;
    params.columns = up1_columns;
    params.segments = 2;
    params.swap0 = 2;
    params.swap1 = blocks;
    params.activation = activation.type;
    params.quantize_scale = activation.quantize_scale;
    params.requant = kLiteRtNvidiaQuantRequantDouble;
    params.ratio = ratio(0, 1);
    const absl::Span<const uint8_t> up1_members[] = {bytes(0), bytes(1)};

    // The middle stages' weights are stored [blocks, inputs, outputs], one
    // output's weights spread over the input rows. Transposed copies give
    // every output its weights as a row, and the rows kernel takes them four
    // inputs per dp4a: a third of the launch time of the blocks kernel for a
    // decode step and a seventh for a prefill chunk. The copies are constants
    // of the plan (the model file's bytes would be shared through the weight
    // store, but the decode and the prefill graphs hold separate copies of
    // these weights in the file anyway).
    LiteRtNvidiaQuantLinearParams& up2_params = stages.stages[1];
    up2_params.layout = kLiteRtNvidiaQuantLayoutRows;
    up2_params.bit_width = 2;
    up2_params.rows = blocks * up2[2];
    up2_params.columns = up2[1];
    up2_params.segments = blocks;
    up2_params.swap0 = 1;
    up2_params.swap1 = 1;
    up2_params.activation = kLiteRtNvidiaQuantInt8;
    up2_params.requant = kLiteRtNvidiaQuantRequantDouble;
    up2_params.lut = 1;
    up2_params.ratio = ratio(1, 2);
    const auto silu = QuantSiluTable(mlp.scales[2], mlp.scales[3]);
    const auto up2_rows =
        QuantTransposedBlocks(bytes(2), blocks, up2[1], up2[2]);
    const absl::Span<const uint8_t> up2_members[] = {up2_rows};

    LiteRtNvidiaQuantLinearParams& down1_params = stages.stages[2];
    down1_params = up2_params;
    down1_params.rows = blocks * down1[2];
    down1_params.columns = down1[1];
    down1_params.swap0 = blocks;
    down1_params.swap1 = 2;
    down1_params.lut = 0;
    down1_params.ratio = ratio(3, 4);
    const auto down1_rows =
        QuantTransposedBlocks(bytes(3), blocks, down1[1], down1[2]);
    const absl::Span<const uint8_t> down1_members[] = {down1_rows};

    LiteRtNvidiaQuantLinearParams& down2_params = stages.stages[3];
    down2_params.layout = kLiteRtNvidiaQuantLayoutRows;
    down2_params.bit_width = 2;
    down2_params.rows = 2 * down2_rows;
    down2_params.columns = down2_columns;
    down2_params.segments = 2;
    down2_params.swap0 = 1;
    down2_params.swap1 = 1;
    down2_params.activation = kLiteRtNvidiaQuantInt8;
    down2_params.requant = kLiteRtNvidiaQuantRequantDouble;
    down2_params.ratio = ratio(4, 5);
    const absl::Span<const uint8_t> down2_members[] = {bytes(4), bytes(5)};

    const absl::Span<const absl::Span<const uint8_t>> members[] = {
        up1_members, up2_members, down1_members, down2_members};
    const std::vector<float> multipliers[] = {joined(scales[0], scales[1]),
                                              scales[2], scales[3],
                                              joined(scales[4], scales[5])};
    nvinfer1::ITensor* stage = nullptr;
    const bool glued =
        glue.has_value() && glue->inputs[0]->getDimensions().d[0] == 1 &&
        LiteRtNvidiaQuantGlueFactorizedMlpSupports(&glue->params, &stages);
    if (glue.has_value() && !glued) {
      LITERT_RETURN_IF_ERROR(LowerQuantGlue(quant_glues_[*mlp.glue]));
      LITERT_ASSIGN_OR_RETURN(activation,
                              GetQuantActivation(*mlp.input, 2 * up1_columns));
    }
    // A decode step's MLP (one INT8 row) is one launch, with the glue in
    // front when it has one; every other MLP is the four launches.
    if (glued) {
      LITERT_ASSIGN_OR_RETURN(
          stage, AddQuantFactorizedMlp(stages, nullptr, members, multipliers,
                                       silu, name, &*glue));
    } else if (activation.type == kLiteRtNvidiaQuantInt8 &&
               activation.rows->getDimensions().d[0] == 1 &&
               LiteRtNvidiaQuantFactorizedMlpSupports(&stages)) {
      LITERT_ASSIGN_OR_RETURN(
          stage, AddQuantFactorizedMlp(stages, activation.rows, members,
                                       multipliers, silu, name));
    } else {
      static const char* const kStageNames[] = {"_up1", "_up2", "_down1",
                                                "_down2"};
      stage = activation.rows;
      for (int s = 0; s < 4; ++s) {
        LITERT_ASSIGN_OR_RETURN(
            stage, AddQuantLinear(stages.stages[s], stage, members[s],
                                  multipliers[s], s == 1 ? &silu : nullptr,
                                  nullptr, name + kStageNames[s]));
      }
    }
    SetQuantValue(*mlp.output, stage, 0, 2 * down2_rows, mlp.scales[5]);
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX quant factorized MLP: output=%u blocks=%d "
               "up=%dx%d hidden=%d down=%dx%d",
               mlp.output->TensorIndex(), blocks, up1_rows, up1_columns,
               blocks * up2[2], down2_rows, down2_columns);
    return {};
  }

  static size_t QuantBlockRank(const Tensor& weights) {
    const auto type = weights.RankedTensorType();
    return type ? type->Layout().Dimensions().size() : 0;
  }

  static bool QuantScalesMatch(const Tensor& a, const Tensor& b) {
    return IsSymmetricInt8(a) && IsSymmetricInt8(b) &&
           a.PerTensorQuantization().scale == b.PerTensorQuantization().scale;
  }

  // The constant int32 values of `tensor`, or nothing.
  static std::vector<int32_t> QuantInts(const Tensor& tensor) {
    auto values = ReadInt32Constant(tensor);
    return values.HasValue() ? values.Value() : std::vector<int32_t>();
  }

  // SLICE of `rows` [m, ...] that keeps dim 0 whole and takes `size` of
  // `axis` from `begin`, every other dim whole.
  static bool QuantIsSlice(const Op& op, size_t axis, int32_t begin,
                           int32_t size) {
    if (op.Code() != kLiteRtOpCodeTflSlice || op.Inputs().size() != 3) {
      return false;
    }
    const auto type = op.Inputs()[0].RankedTensorType();
    const auto begins = QuantInts(op.Inputs()[1]);
    const auto sizes = QuantInts(op.Inputs()[2]);
    if (!type) return false;
    const auto dims = type->Layout().Dimensions();
    if (begins.size() != dims.size() || sizes.size() != dims.size() ||
        axis >= dims.size()) {
      return false;
    }
    for (size_t d = 0; d < dims.size(); ++d) {
      const int32_t want_begin = d == axis ? begin : 0;
      const int32_t want_size = d == axis ? size : dims[d];
      if (begins[d] != want_begin ||
          (sizes[d] != want_size && !(sizes[d] == -1 && d != axis))) {
        return false;
      }
    }
    return true;
  }

  // The factorized MLP around its first BATCH_MATMUL `up2`
  // (QuantFactorizedMlp), as the model's graph has it:
  //   x INT8 [m, 2 C] -> SLICE [m, C] -> FULLY_CONNECTED -> RESHAPE
  //     [m, B, 1, R1 / B], twice -> CONCATENATION on axis 2 -> RESHAPE
  //     [m, B, 1, 2 R1 / B] -> up2 -> RESHAPE [m, B N] -> DEQUANTIZE ->
  //     LOGISTIC, MUL -> QUANTIZE -> RESHAPE [m, B, 1, N] -> BATCH_MATMUL ->
  //     RESHAPE [m, B, 2, D] -> SLICE [m, B, 1, D] -> RESHAPE [m, B D] ->
  //     FULLY_CONNECTED, twice -> CONCATENATION -> y INT8 [m, 2 R4]
  // with one reader per intermediate. `members` receives its ops.
  std::optional<QuantFactorizedMlp> MatchQuantFactorizedMlp(
      const Op& up2, std::vector<LiteRtOp>& members) const {
    QuantFactorizedMlp mlp;
    mlp.weights.resize(6, up2.Inputs()[1]);
    const auto block = QuantBlockDims(up2.Inputs()[1]);
    const int32_t blocks = block[0];
    // Behind up2: the regrouped results of the two up-1 products.
    auto regroup = QuantProducer(up2.Inputs()[0], kLiteRtOpCodeTflReshape);
    if (!regroup) return std::nullopt;
    auto concat =
        QuantProducer(regroup->Inputs()[0], kLiteRtOpCodeTflConcatenation);
    int32_t axis = 0;
    if (!concat || concat->Inputs().size() != 2 ||
        LiteRtGetConcatenationAxisOption(concat->Get(), &axis) !=
            kLiteRtStatusOk ||
        (axis != 2 && axis != -2)) {
      return std::nullopt;
    }
    members.push_back(regroup->Get());
    members.push_back(concat->Get());
    std::optional<Tensor> x;
    for (int half = 0; half < 2; ++half) {
      auto reshape =
          QuantProducer(concat->Inputs()[half], kLiteRtOpCodeTflReshape);
      if (!reshape) return std::nullopt;
      auto product =
          QuantProducer(reshape->Inputs()[0], kLiteRtOpCodeTflFullyConnected);
      if (!product || !IsQuantFullyConnected(*product)) return std::nullopt;
      const auto [rows, columns] = QuantMatrixDims(product->Inputs()[1]);
      auto slice = QuantProducer(product->Inputs()[0], kLiteRtOpCodeTflSlice);
      if (!slice || !QuantIsSlice(*slice, 1, half * columns, columns) ||
          rows % blocks != 0 ||
          !TensorHasShape(
              reshape->Outputs()[0],
              {TensorRows(slice->Inputs()[0]), blocks, 1, rows / blocks}) ||
          (x.has_value() && x->Get() != slice->Inputs()[0].Get())) {
        return std::nullopt;
      }
      x = slice->Inputs()[0];
      mlp.weights[half] = product->Inputs()[1];
      mlp.scales[1] = product->Outputs()[0].PerTensorQuantization().scale;
      members.push_back(slice->Get());
      members.push_back(product->Get());
      members.push_back(reshape->Get());
    }
    const int32_t m = TensorRows(*x);
    if (!IsSymmetricInt8(*x) ||
        !TensorHasShape(*x, {m, 2 * QuantMatrixDims(mlp.weights[0]).second}) ||
        x->Uses().size() != 2 ||
        !QuantScalesMatch(concat->Inputs()[0], concat->Inputs()[1]) ||
        !QuantScalesMatch(concat->Inputs()[0], up2.Inputs()[0]) ||
        !TensorHasShape(up2.Inputs()[0], {m, blocks, 1, block[1]})) {
      return std::nullopt;
    }
    mlp.input = *x;
    mlp.scales[0] = x->PerTensorQuantization().scale;
    mlp.weights[2] = up2.Inputs()[1];
    mlp.scales[2] = up2.Outputs()[0].PerTensorQuantization().scale;
    members.push_back(up2.Get());
    // Past up2: the SiLU between a DEQUANTIZE and a QUANTIZE.
    auto flat = QuantUser(up2.Outputs()[0], kLiteRtOpCodeTflReshape);
    if (!flat) return std::nullopt;
    auto dequantize = QuantUser(flat->Outputs()[0], kLiteRtOpCodeTflDequantize);
    if (!dequantize ||
        dequantize->Outputs()[0].ElementType() !=
            litert::ElementType::Float16 ||
        dequantize->Outputs()[0].Uses().size() != 2) {
      return std::nullopt;
    }
    std::optional<Op> logistic;
    std::optional<Op> multiply;
    for (const auto& use : dequantize->Outputs()[0].Uses()) {
      if (use.user.Code() == kLiteRtOpCodeTflLogistic) logistic = use.user;
      if (use.user.Code() == kLiteRtOpCodeTflMul) multiply = use.user;
    }
    if (!logistic || !multiply || !HasNoFusedActivation(*multiply) ||
        !ReadOnce(logistic->Outputs()[0]) ||
        logistic->Outputs()[0].Uses()[0].user.Get() != multiply->Get()) {
      return std::nullopt;
    }
    auto quantize = QuantUser(multiply->Outputs()[0], kLiteRtOpCodeTflQuantize);
    if (!quantize || !IsSymmetricInt8(quantize->Outputs()[0])) {
      return std::nullopt;
    }
    mlp.scales[3] = quantize->Outputs()[0].PerTensorQuantization().scale;
    auto blocked = QuantUser(quantize->Outputs()[0], kLiteRtOpCodeTflReshape);
    if (!blocked) return std::nullopt;
    auto down1 = QuantUser(blocked->Outputs()[0], kLiteRtOpCodeTflBatchMatmul);
    if (!down1 || !IsQuantBatchMatmul(*down1)) return std::nullopt;
    const auto down = QuantBlockDims(down1->Inputs()[1]);
    const int32_t depth = down[2] / 2;
    if (down[0] != blocks || down[1] != block[2] || down[2] % 2 != 0 ||
        !TensorHasShape(blocked->Outputs()[0], {m, blocks, 1, block[2]})) {
      return std::nullopt;
    }
    mlp.weights[3] = down1->Inputs()[1];
    mlp.scales[4] = down1->Outputs()[0].PerTensorQuantization().scale;
    for (const Op& op : {*flat, *dequantize, *logistic, *multiply, *quantize,
                         *blocked, *down1}) {
      members.push_back(op.Get());
    }
    // Past down-1: its halves feed the two down-2 products.
    auto halves = QuantUser(down1->Outputs()[0], kLiteRtOpCodeTflReshape);
    if (!halves ||
        !TensorHasShape(halves->Outputs()[0], {m, blocks, 2, depth}) ||
        halves->Outputs()[0].Uses().size() != 2) {
      return std::nullopt;
    }
    members.push_back(halves->Get());
    std::optional<Op> join;
    for (const auto& use : halves->Outputs()[0].Uses()) {
      const Op slice = use.user;
      int half = -1;
      for (int h = 0; h < 2; ++h) {
        if (QuantIsSlice(slice, 2, h, 1)) half = h;
      }
      if (half < 0 || slice.Outputs().size() != 1) return std::nullopt;
      auto reshape = QuantUser(slice.Outputs()[0], kLiteRtOpCodeTflReshape);
      if (!reshape ||
          !TensorHasShape(reshape->Outputs()[0], {m, blocks * depth})) {
        return std::nullopt;
      }
      auto product =
          QuantUser(reshape->Outputs()[0], kLiteRtOpCodeTflFullyConnected);
      if (!product || !IsQuantFullyConnected(*product)) return std::nullopt;
      auto user =
          QuantUser(product->Outputs()[0], kLiteRtOpCodeTflConcatenation);
      // The halves are joined in their order.
      if (!user || user->Inputs().size() != 2 ||
          user->Inputs()[half].Get() != product->Outputs()[0].Get() ||
          (join.has_value() && join->Get() != user->Get())) {
        return std::nullopt;
      }
      join = user;
      mlp.weights[4 + half] = product->Inputs()[1];
      mlp.scales[5] = product->Outputs()[0].PerTensorQuantization().scale;
      members.push_back(slice.Get());
      members.push_back(reshape->Get());
      members.push_back(product->Get());
    }
    const int32_t down2_rows = QuantMatrixDims(mlp.weights[4]).first;
    if (!join.has_value() ||
        !QuantScalesMatch(join->Inputs()[0], join->Inputs()[1]) ||
        !QuantScalesMatch(join->Inputs()[0], join->Outputs()[0]) ||
        !TensorHasShape(join->Outputs()[0], {m, 2 * down2_rows})) {
      return std::nullopt;
    }
    members.push_back(join->Get());
    mlp.output = join->Outputs()[0];
    for (int i = 0; i < 6; ++i) {
      if (!CenteredInt2Scales(
               mlp.weights[i],
               i == 2 || i == 3 ? QuantBlockRank(mlp.weights[i]) - 2 : 1)
               .has_value()) {
        return std::nullopt;
      }
    }
    return mlp;
  }

  // Dim 0 of a tensor: the rows of the factorized MLP's tensors.
  static int32_t TensorRows(const Tensor& tensor) {
    const auto type = tensor.RankedTensorType();
    return type && type->Layout().Rank() > 0 ? type->Layout().Dimensions()[0]
                                             : -1;
  }

  // The head norms and rotations behind the projections of `group`: per
  // FULLY_CONNECTED result
  //   RESHAPE [1, n, heads, depth] -> DEQUANTIZE -> CAST -> odml.rms_norm ->
  //   CAST, then for queries and keys TRANSPOSE (RESHAPE for one token)
  //   to [1, heads, n, depth] -> odml.rope, for queries -> MUL scalar.
  std::optional<QuantQkvRope> MatchQuantQkvRope(
      size_t index, std::vector<LiteRtOp>& members) const {
    const QuantProjections& group = quant_projections_[index];
    QuantQkvRope match;
    match.projections = index;
    bool rotated = false;
    for (const Op& product : group.ops) {
      auto reshape = QuantUser(product.Outputs()[0], kLiteRtOpCodeTflReshape);
      if (!reshape) return std::nullopt;
      auto dequantize =
          QuantUser(reshape->Outputs()[0], kLiteRtOpCodeTflDequantize);
      if (!dequantize) return std::nullopt;
      auto widen = QuantUser(dequantize->Outputs()[0], kLiteRtOpCodeTflCast);
      if (!widen) return std::nullopt;
      auto norm_op = QuantUser(widen->Outputs()[0], kLiteRtOpCodeShloComposite);
      if (!norm_op) return std::nullopt;
      auto narrow = QuantUser(norm_op->Outputs()[0], kLiteRtOpCodeTflCast);
      if (!narrow) return std::nullopt;
      const Tensor normed = narrow->Outputs()[0];
      std::vector<LiteRtOp> chain;
      auto norm = MatchQuantNorm(normed, /*shared=*/true, chain);
      const auto type = normed.RankedTensorType();
      if (!norm.has_value() ||
          norm->input.Get() != dequantize->Outputs()[0].Get() || !type ||
          type->Layout().Rank() != 4 || type->Layout().Dimensions()[0] != 1) {
        return std::nullopt;
      }
      const auto dims = type->Layout().Dimensions();
      const int32_t tokens = dims[1];
      const int32_t heads = dims[2];
      if (match.depth != 0 && match.depth != dims[3]) return std::nullopt;
      match.depth = dims[3];
      members.push_back(reshape->Get());
      members.push_back(dequantize->Get());
      members.insert(members.end(), chain.begin(), chain.end());
      // A rotation behind the norm makes a query or a key; a norm other ops
      // read is the value.
      std::optional<Op> rope;
      if (ReadOnce(normed)) {
        const Op turn = normed.Uses()[0].user;
        const bool transposes =
            turn.Code() == kLiteRtOpCodeTflTranspose &&
            turn.Inputs().size() == 2 &&
            QuantInts(turn.Inputs()[1]) == std::vector<int32_t>{0, 2, 1, 3};
        const bool reshapes =
            turn.Code() == kLiteRtOpCodeTflReshape && tokens == 1;
        if ((transposes || reshapes) && turn.Outputs().size() == 1 &&
            TensorHasShape(turn.Outputs()[0], {1, heads, tokens, dims[3]})) {
          rope = QuantUser(turn.Outputs()[0], kLiteRtOpCodeShloComposite);
          if (rope.has_value()) members.push_back(turn.Get());
        }
      }
      if (!rope.has_value()) {
        // The value: its norm is what the attention's QUANTIZE reads.
        if (match.roles.empty() || match.roles.back() == 2) return std::nullopt;
        match.roles.push_back(2);
        match.norms.push_back(*norm);
        match.outputs.push_back(normed);
        continue;
      }
      const auto parsed = ParseQuantRope(*rope);
      if (!parsed.has_value() || parsed->heads != heads ||
          parsed->tokens != tokens ||
          (rotated && (parsed->min_timescale != match.rope.min_timescale ||
                       parsed->max_timescale != match.rope.max_timescale ||
                       parsed->proportion != match.rope.proportion ||
                       rope->Inputs()[1].Get() != match.positions->Get()))) {
        return std::nullopt;
      }
      match.rope = *parsed;
      match.positions = rope->Inputs()[1];
      rotated = true;
      members.push_back(rope->Get());
      Tensor output = rope->Outputs()[0];
      // The query's scalar behind its rotation; a key has none.
      std::optional<float> multiplier;
      if (auto scale = QuantUser(output, kLiteRtOpCodeTflMul);
          scale && scale->Inputs().size() == 2 &&
          HasNoFusedActivation(*scale)) {
        for (int j = 0; j < 2; ++j) {
          if (auto scalar = ReadFloatScalar(scale->Inputs()[j])) {
            multiplier = *scalar;
            output = scale->Outputs()[0];
            members.push_back(scale->Get());
            break;
          }
        }
      }
      // Rows are queries, then keys, then values; without a scalar the
      // first rotated projection of a group of three is still the query.
      const bool is_query = match.roles.empty() &&
                            (multiplier.has_value() || group.ops.size() != 2);
      if (!is_query && (multiplier.has_value() ||
                        (!match.roles.empty() && match.roles.back() != 0))) {
        return std::nullopt;
      }
      match.roles.push_back(is_query ? 0 : 1);
      if (is_query) match.q_multiplier = multiplier.value_or(1.0f);
      match.norms.push_back(*norm);
      match.outputs.push_back(output);
    }
    if (!rotated) return std::nullopt;
    return match;
  }

  // Finds the regions of the statically quantized products (see above) and
  // takes their ops out of the op-by-op lowering.
  Expected<void> FindQuantRegions(const Subgraph& subgraph) {
    if (!QuantProductsEnabled()) {
      return {};
    }
    const auto ops = subgraph.Ops();
    // Factorized MLPs, around their first BATCH_MATMUL.
    for (const auto& op : ops) {
      if (!IsQuantBatchMatmul(op) || unlowered_ops_.count(op.Get()) != 0) {
        continue;
      }
      std::vector<LiteRtOp> members;
      auto mlp = MatchQuantFactorizedMlp(op, members);
      if (!mlp.has_value() || !QuantOpsAreFree(members)) continue;
      quant_outputs_.insert(mlp->output->Get());
      quant_product_inputs_.insert(mlp->input->Get());
      quant_factorized_mlps_by_input_[mlp->input->Get()] =
          quant_factorized_mlps_.size();
      AddQuantRegion(QuantRegionKind::kFactorizedMlp,
                     quant_factorized_mlps_.size(), members);
      quant_factorized_mlps_.push_back(std::move(*mlp));
    }
    // Factorized MLPs carved into odml.factorized_mlp: the composite is the
    // region.
    for (const auto& op : ops) {
      const auto carved = ParseQuantFactorizedMlpComposite(op);
      if (!carved.has_value() || !QuantOpsAreFree({op.Get()})) continue;
      QuantFactorizedMlp mlp;
      mlp.input = op.Inputs()[0];
      for (int i = 1; i <= 6; ++i) mlp.weights.push_back(op.Inputs()[i]);
      std::copy(std::begin(carved->scales), std::end(carved->scales),
                mlp.scales);
      mlp.output = op.Outputs()[0];
      quant_outputs_.insert(mlp.output->Get());
      quant_factorized_mlps_by_input_[mlp.input->Get()] =
          quant_factorized_mlps_.size();
      AddQuantRegion(QuantRegionKind::kFactorizedMlp,
                     quant_factorized_mlps_.size(), {op.Get()});
      quant_factorized_mlps_.push_back(std::move(mlp));
    }
    // The other INT8 products with INT4 weights, grouped by the tensor they
    // read: the q, k and v projections are one launch.
    for (const auto& op : ops) {
      if (unlowered_ops_.count(op.Get()) != 0 ||
          op_order_.count(op.Get()) == 0 || !IsQuantFullyConnected(op) ||
          op.Inputs()[1].ElementType() != litert::ElementType::Int4) {
        continue;
      }
      QuantProjections group;
      group.input = op.Inputs()[0];
      std::vector<LiteRtOp> members;
      for (const auto& use : group.input->Uses()) {
        const Op& user = use.user;
        if (use.user_arg_ind == 0 && unlowered_ops_.count(user.Get()) == 0 &&
            op_order_.count(user.Get()) != 0 && IsQuantFullyConnected(user) &&
            user.Inputs()[1].ElementType() == litert::ElementType::Int4 &&
            QuantMatrixDims(user.Inputs()[1]).second ==
                QuantMatrixDims(op.Inputs()[1]).second) {
          group.ops.push_back(user);
          members.push_back(user.Get());
        }
      }
      std::sort(group.ops.begin(), group.ops.end(),
                [&](const Op& a, const Op& b) {
                  return op_order_[a.Get()] < op_order_[b.Get()];
                });
      for (const Op& member : group.ops) {
        quant_outputs_.insert(member.Outputs()[0].Get());
      }
      quant_product_inputs_.insert(group.input->Get());
      quant_projections_by_input_[group.input->Get()] =
          quant_projections_.size();
      AddQuantRegion(QuantRegionKind::kProjections, quant_projections_.size(),
                     members);
      quant_projections_.push_back(std::move(group));
    }
    // The head norms and rotations behind the projection groups.
    for (size_t i = 0; i < quant_projections_.size(); ++i) {
      std::vector<LiteRtOp> members;
      auto match = MatchQuantQkvRope(i, members);
      if (!match.has_value() || !QuantOpsAreFree(members)) continue;
      AddQuantRegion(QuantRegionKind::kQkvRope, quant_qkv_ropes_.size(),
                     members);
      quant_qkv_ropes_.push_back(std::move(*match));
    }
    // The vocabulary head with the float steps in front of it.
    for (const auto& op : ops) {
      if (unlowered_ops_.count(op.Get()) != 0 ||
          op_order_.count(op.Get()) == 0 || !IsQuantLogitsFullyConnected(op)) {
        continue;
      }
      QuantHead head;
      head.product = op;
      head.logits = op.Outputs()[0];
      head.chain = MatchQuantGlueChain(op.Inputs()[0], /*shared=*/true);
      std::vector<LiteRtOp> members = head.chain.members;
      members.push_back(op.Get());
      // The soft cap: CAST -> MUL scalar -> TANH -> MUL scalar.
      std::vector<LiteRtOp> cap;
      auto widen = QuantUser(op.Outputs()[0], kLiteRtOpCodeTflCast);
      auto scale_in = widen
                          ? QuantUser(widen->Outputs()[0], kLiteRtOpCodeTflMul)
                          : std::nullopt;
      auto tanh = scale_in
                      ? QuantUser(scale_in->Outputs()[0], kLiteRtOpCodeTflTanh)
                      : std::nullopt;
      auto scale_out = tanh ? QuantUser(tanh->Outputs()[0], kLiteRtOpCodeTflMul)
                            : std::nullopt;
      if (scale_out.has_value() && scale_in->Inputs().size() == 2 &&
          scale_out->Inputs().size() == 2 &&
          widen->Outputs()[0].ElementType() == litert::ElementType::Float32 &&
          HasNoFusedActivation(*scale_in) && HasNoFusedActivation(*scale_out)) {
        const auto cap_in = ReadFloat32Scalar(scale_in->Inputs()[1]);
        const auto cap_out = ReadFloat32Scalar(scale_out->Inputs()[1]);
        if (cap_in.has_value() && cap_out.has_value() && *cap_in != 0.0f) {
          head.cap_in = *cap_in;
          head.cap_out = *cap_out;
          head.logits = scale_out->Outputs()[0];
          for (const Op& member : {*widen, *scale_in, *tanh, *scale_out}) {
            members.push_back(member.Get());
          }
        }
      }
      if (!QuantOpsAreFree(members)) continue;
      AddQuantRegion(QuantRegionKind::kHead, quant_heads_.size(), members);
      quant_heads_.push_back(std::move(head));
    }
    // The float steps in front of a carved factorized MLP, whose QUANTIZE is
    // inside the composite: behind an RMS norm the glue quantizes for it.
    for (const auto& mlp : quant_factorized_mlps_) {
      if (mlp.input->ElementType() != litert::ElementType::Float16 ||
          !ReadOnce(*mlp.input)) {
        continue;
      }
      QuantGlue glue;
      glue.chain = MatchQuantGlueChain(*mlp.input, /*shared=*/false);
      if (!glue.chain.norm.has_value() ||
          !QuantOpsAreFree(glue.chain.members)) {
        continue;
      }
      glue.quantized = *mlp.input;
      glue.scale = mlp.scales[0];
      const std::vector<LiteRtOp> members = glue.chain.members;
      quant_glues_.push_back(std::move(glue));
      AddQuantGlueRegion(*mlp.input, members);
    }
    // The float steps in front of the QUANTIZE ops that feed the products:
    // a QUANTIZE behind an RMS norm becomes a glue layer; any other is left
    // to the product's kernel, which quantizes its float input itself.
    for (const auto& op : ops) {
      if (op.Code() != kLiteRtOpCodeTflQuantize ||
          unlowered_ops_.count(op.Get()) != 0 ||
          op_order_.count(op.Get()) == 0 || op.Inputs().size() != 1 ||
          op.Outputs().size() != 1 ||
          op.Inputs()[0].ElementType() != litert::ElementType::Float16 ||
          !IsSymmetricInt8(op.Outputs()[0]) || !ReadOnce(op.Inputs()[0])) {
        continue;
      }
      std::vector<LiteRtOp> members = {op.Get()};
      Tensor read = op.Outputs()[0];
      for (int hops = 0;
           hops < 3 && quant_product_inputs_.count(read.Get()) == 0; ++hops) {
        auto reshape = QuantUser(read, kLiteRtOpCodeTflReshape);
        if (!reshape) break;
        members.push_back(reshape->Get());
        read = reshape->Outputs()[0];
      }
      if (quant_product_inputs_.count(read.Get()) == 0) continue;
      QuantGlue glue;
      glue.chain = MatchQuantGlueChain(op.Inputs()[0], /*shared=*/false);
      if (!glue.chain.norm.has_value()) continue;
      glue.quantized = op.Outputs()[0];
      glue.scale = op.Outputs()[0].PerTensorQuantization().scale;
      members.insert(members.end(), glue.chain.members.begin(),
                     glue.chain.members.end());
      if (!QuantOpsAreFree(members)) continue;
      quant_glues_.push_back(std::move(glue));
      AddQuantGlueRegion(read, members);
    }
    if (!quant_regions_.empty()) {
      size_t glues_in_projections = 0;
      for (const auto& group : quant_projections_) {
        glues_in_projections += group.glue.has_value() ? 1 : 0;
      }
      size_t glues_in_factorized_mlps = 0;
      for (const auto& mlp : quant_factorized_mlps_) {
        glues_in_factorized_mlps += mlp.glue.has_value() ? 1 : 0;
      }
      LITERT_LOG(
          LITERT_INFO,
          "NVIDIA TensorRT-RTX quantized products: %zu projection "
          "launches (%zu with the glue in front), %zu factorized MLPs "
          "(%zu with the glue in front), %zu glue launches, %zu qkv "
          "rope launches, %zu heads",
          quant_projections_.size(), glues_in_projections,
          quant_factorized_mlps_.size(), glues_in_factorized_mlps,
          quant_glues_.size() - glues_in_projections - glues_in_factorized_mlps,
          quant_qkv_ropes_.size(), quant_heads_.size());
    }
    return {};
  }

  Expected<void> LowerQuantProjections(const QuantProjections& group) {
    const auto [first_rows, columns] =
        QuantMatrixDims(group.ops[0].Inputs()[1]);
    static_cast<void>(first_rows);
    const float input_scale = group.input->PerTensorQuantization().scale;
    std::vector<absl::Span<const uint8_t>> members;
    std::vector<float> multipliers;
    for (const Op& op : group.ops) {
      const Tensor weights = op.Inputs()[1];
      const auto q = weights.PerChannelQuantization();
      const float output_scale = op.Outputs()[0].PerTensorQuantization().scale;
      members.push_back(weights.Weights().Bytes());
      for (uint64_t r = 0; r < q.num_channels; ++r) {
        // XNNPACK's requantization scale, in FP32.
        const float product = input_scale * q.scales[r];
        multipliers.push_back(product / output_scale);
      }
    }
    LiteRtNvidiaQuantLinearParams params = {};
    params.layout = kLiteRtNvidiaQuantLayoutRows;
    params.bit_width = 4;
    params.rows = static_cast<int32_t>(multipliers.size());
    params.columns = columns;
    params.segments = 1;
    params.swap0 = 1;
    params.swap1 = 1;
    params.activation = kLiteRtNvidiaQuantInt8;
    params.requant = kLiteRtNvidiaQuantRequantFp32;
    LITERT_ASSIGN_OR_RETURN(auto glue, QuantGlueInFront(group.glue, params));
    QuantActivation activation;
    if (!glue.has_value()) {
      LITERT_ASSIGN_OR_RETURN(activation,
                              GetQuantActivation(*group.input, columns));
      params.activation = activation.type;
      params.quantize_scale = activation.quantize_scale;
    }
    const std::string name =
        "cuda_quant_projections_" + std::to_string(group.input->TensorIndex());
    LITERT_ASSIGN_OR_RETURN(
        auto* result,
        AddQuantLinear(params, activation.rows, members, multipliers, nullptr,
                       nullptr, name, glue.has_value() ? &*glue : nullptr));
    int32_t offset = 0;
    for (const Op& op : group.ops) {
      const int32_t rows = QuantMatrixDims(op.Inputs()[1]).first;
      SetQuantValue(op.Outputs()[0], result, offset, rows,
                    op.Outputs()[0].PerTensorQuantization().scale);
      offset += rows;
    }
    return {};
  }

  Expected<void> LowerQuantGlue(const QuantGlue& glue) {
    const float scale = glue.scale;
    LITERT_ASSIGN_OR_RETURN(
        auto* q,
        AddQuantGlue(glue.chain, kLiteRtNvidiaQuantGlueStatic, scale, nullptr,
                     "cuda_quant_glue_" +
                         std::to_string(glue.quantized->TensorIndex())));
    LITERT_ASSIGN_OR_RETURN(auto type, glue.quantized->RankedTensorType());
    SetQuantValue(*glue.quantized, q, 0, type.Layout().Dimensions().back(),
                  scale);
    return {};
  }

  Expected<void> LowerQuantQkvRope(const QuantQkvRope& match) {
    const QuantProjections& group = quant_projections_[match.projections];
    // The projections' results are the columns of one quant layer result.
    nvinfer1::ITensor* qkv = nullptr;
    int32_t offset = 0;
    LiteRtNvidiaQuantQkvRopeParams params = {};
    params.depth = match.depth;
    params.epsilon = match.norms[0].eps;
    params.q_multiplier = match.q_multiplier;
    std::vector<nvinfer1::ITensor*> inputs;
    const std::string name =
        "cuda_quant_qkv_rope_" + std::to_string(group.input->TensorIndex());
    for (size_t i = 0; i < group.ops.size(); ++i) {
      const auto value = quant_values_.find(group.ops[i].Outputs()[0].Get());
      if (value == quant_values_.end() || value->second.offset != offset ||
          (qkv != nullptr && value->second.int8 != qkv) ||
          value->second.width % match.depth != 0 ||
          match.norms[i].eps != params.epsilon) {
        return Error(kLiteRtStatusErrorCompilation,
                     "The projections of a quant qkv rope are not one result");
      }
      qkv = value->second.int8;
      offset += value->second.width;
      const int32_t heads = value->second.width / match.depth;
      const float scale = RoundFp16(value->second.scale);
      switch (match.roles[i]) {
        case 0:
          params.q_heads = heads;
          params.q_scale = scale;
          break;
        case 1:
          params.k_heads = heads;
          params.k_scale = scale;
          break;
        default:
          params.v_heads = heads;
          params.v_scale = scale;
      }
    }
    if (qkv->getDimensions().d[1] != offset) {
      return Error(kLiteRtStatusErrorCompilation,
                   "A quant qkv rope does not read every projection");
    }
    LITERT_ASSIGN_OR_RETURN(auto* positions, GetTensor(*match.positions));
    if (positions->getType() != nvinfer1::DataType::kINT32) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "odml.rope positions must be int32");
    }
    inputs.push_back(qkv);
    inputs.push_back(positions);
    for (size_t i = 0; i < group.ops.size(); ++i) {
      LITERT_ASSIGN_OR_RETURN(
          auto* weights,
          QuantFloatConstant(match.norms[i].weights,
                             name + "_norm_" + std::to_string(i)));
      inputs.push_back(weights);
    }
    // The inverse timescales as the model's decomposition holds them: the
    // timescale in doubles rounded to FP32, then its reciprocal; the pairs
    // past the rotated proportion keep an angle of 0.
    const int32_t pairs = match.depth / 2;
    const int32_t rotated = static_cast<int32_t>(
        std::lround(static_cast<double>(match.rope.proportion) * pairs));
    std::vector<float> inverse(pairs, 0.0f);
    for (int32_t i = 0; i < rotated && i < pairs; ++i) {
      const float timescale = static_cast<float>(
          static_cast<double>(match.rope.min_timescale) *
          std::pow(static_cast<double>(match.rope.max_timescale) /
                       match.rope.min_timescale,
                   static_cast<double>(i) / pairs));
      inverse[i] = 1.0f / timescale;
    }
    nvinfer1::Dims inverse_dims{};
    inverse_dims.nbDims = 1;
    inverse_dims.d[0] = pairs;
    LITERT_ASSIGN_OR_RETURN(
        auto* timescales,
        AddFloatConstant(inverse, inverse_dims, name + "_timescales",
                         nvinfer1::DataType::kFLOAT));
    inputs.push_back(timescales);
    TrtPtr<nvinfer1::IPluginV3> plugin(CreateQuantQkvRopePlugin(params));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the quant qkv rope plugin");
    }
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs.data(), static_cast<int32_t>(inputs.size()), *plugin);
    if (layer == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the quant qkv rope plugin layer");
    }
    layer->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    // The plugin's outputs are in the order q, k, v, as the rows are.
    for (size_t i = 0; i < match.outputs.size(); ++i) {
      LITERT_RETURN_IF_ERROR(SetQuantFloat(
          match.outputs[i], layer->getOutput(static_cast<int32_t>(i))));
    }
    return {};
  }

  Expected<void> LowerQuantHead(const QuantHead& head) {
    const Tensor weights = head.product->Inputs()[1];
    const auto [rows, columns] = QuantMatrixDims(weights);
    auto scales = CenteredInt2Scales(weights, 1);
    if (!scales.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "The head's weights are not centered INT2");
    }
    const std::string name =
        "cuda_quant_head_" + std::to_string(weights.TensorIndex());
    nvinfer1::ITensor* row_scale = nullptr;
    LITERT_ASSIGN_OR_RETURN(
        auto* q, AddQuantGlue(head.chain, kLiteRtNvidiaQuantGlueDynamic, 0.0f,
                              &row_scale, name + "_glue"));
    LiteRtNvidiaQuantLinearParams params = {};
    params.layout = kLiteRtNvidiaQuantLayoutRows;
    params.bit_width = 2;
    params.rows = rows;
    params.columns = columns;
    params.segments = 1;
    params.swap0 = 1;
    params.swap1 = 1;
    params.activation = kLiteRtNvidiaQuantInt8;
    params.requant = kLiteRtNvidiaQuantRequantLogits;
    params.cap_in = head.cap_in;
    params.cap_out = head.cap_out;
    const absl::Span<const uint8_t> members[] = {weights.Weights().Bytes()};
    LITERT_ASSIGN_OR_RETURN(
        auto* logits,
        AddQuantLinear(params, q, members, *scales, nullptr, row_scale, name));
    LITERT_ASSIGN_OR_RETURN(auto type, head.logits->RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
    LITERT_ASSIGN_OR_RETURN(logits, ReshapeTensor(logits, dims));
    // The FP32 logits are the model's output as they are; the hidden state
    // in FP32 is too, where a CAST makes it one.
    if (head.logits->ElementType() == litert::ElementType::Float32) {
      raw_output_map_[head.logits->Get()] = logits;
    }
    if (head.chain.normed.has_value()) {
      for (const auto& use : head.chain.normed->Uses()) {
        const Op& user = use.user;
        if (user.Code() == kLiteRtOpCodeTflCast && user.Outputs().size() == 1 &&
            user.Outputs()[0].ElementType() == litert::ElementType::Float32 &&
            graph_outputs_.count(user.Outputs()[0].Get()) != 0) {
          LITERT_ASSIGN_OR_RETURN(auto* hidden, GetTensor(*head.chain.normed));
          LITERT_ASSIGN_OR_RETURN(
              hidden, AddCastTensor(hidden, nvinfer1::DataType::kFLOAT));
          raw_output_map_[user.Outputs()[0].Get()] = hidden;
        }
      }
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX quant head: weights=%u [%d, %d] soft "
               "cap=%g",
               weights.TensorIndex(), rows, columns, head.cap_out);
    return SetOutputTensor(*head.logits, logits);
  }

  Expected<void> LowerQuantRegion(const QuantRegion& region) {
    switch (region.kind) {
      case QuantRegionKind::kProjections:
        return LowerQuantProjections(quant_projections_[region.index]);
      case QuantRegionKind::kFactorizedMlp:
        return LowerQuantFactorizedMlp(quant_factorized_mlps_[region.index]);
      case QuantRegionKind::kGlue:
        return LowerQuantGlue(quant_glues_[region.index]);
      case QuantRegionKind::kQkvRope:
        return LowerQuantQkvRope(quant_qkv_ropes_[region.index]);
      case QuantRegionKind::kHead:
        return LowerQuantHead(quant_heads_[region.index]);
    }
    return Error(kLiteRtStatusErrorCompilation, "Unknown quant region");
  }

  // The fused qkv norm and rotary embedding as one plugin launch.
  Expected<void> LowerCompositeQkvNormRope(const Op& op) {
    const auto parsed = ParseQkvNormRope(op);
    if (!parsed.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported qkv_norm_rope composite");
    }
    const auto& p = parsed->params;
    LITERT_ASSIGN_OR_RETURN(auto* x, GetTensor(op.Inputs()[0]));
    LITERT_ASSIGN_OR_RETURN(x, AddCastTensor(x, nvinfer1::DataType::kBF16));
    LITERT_ASSIGN_OR_RETURN(auto* positions, GetTensor(op.Inputs()[1]));
    if (positions->getType() != nvinfer1::DataType::kINT32) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "qkv_norm_rope positions must be int32");
    }
    nvinfer1::ITensor* scales[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
      const auto bytes = op.Inputs()[2 + i].Weights().Bytes();
      std::vector<float> values(p.depth);
      std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
      nvinfer1::Dims dims{};
      dims.nbDims = 1;
      dims.d[0] = p.depth;
      LITERT_ASSIGN_OR_RETURN(
          scales[i],
          AddFloatConstant(
              values, dims,
              std::string(i == 0 ? "qkv_norm_q_scale_" : "qkv_norm_k_scale_") +
                  std::to_string(op.Inputs()[2 + i].TensorIndex()),
              nvinfer1::DataType::kFLOAT));
    }
    TrtPtr<nvinfer1::IPluginV3> plugin(CreateQkvNormRopePlugin(
        p.heads, p.kv_heads, p.depth, p.rotary, p.has_v_norm != 0, p.epsilon,
        p.min_timescale, p.max_timescale));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the qkv_norm_rope plugin");
    }
    nvinfer1::ITensor* inputs[] = {x, positions, scales[0], scales[1]};
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs, std::size(inputs), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr ||
        layer->getOutput(1) == nullptr || layer->getOutput(2) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the qkv_norm_rope plugin layer");
    }
    const std::string name =
        "cuda_qkv_norm_rope_" + std::to_string(op.Inputs()[0].TensorIndex());
    layer->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    for (int o = 0; o < 3; ++o) {
      layer->getOutput(o)->setName(
          KeepName(UniqueName(name + "_" + std::to_string(o))));
      LITERT_RETURN_IF_ERROR(
          SetOutputTensor(op.Outputs()[o], layer->getOutput(o)));
    }
    return {};
  }

  // A float operand of the QSA plugins in FP32: a constant from its bytes
  // (not through GetTensor, which rounds constants to the activation type),
  // or a runtime tensor cast.
  Expected<nvinfer1::ITensor*> GetQsaFp32Operand(const Tensor& tensor,
                                                 const std::string& name) {
    if (tensor.HasWeights()) {
      LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
      LITERT_ASSIGN_OR_RETURN(auto dims, ConvertDims(type));
      const auto bytes = tensor.Weights().Bytes();
      const int64_t count = StaticElementCount(tensor);
      std::vector<float> values(count);
      if (tensor.ElementType() == litert::ElementType::Float16) {
        if (bytes.size() != static_cast<size_t>(count) * 2) {
          return Error(kLiteRtStatusErrorUnsupported,
                       "QSA constant has the wrong size");
        }
        for (int64_t i = 0; i < count; ++i) {
          uint16_t bits;
          std::memcpy(&bits, bytes.data() + 2 * i, 2);
          values[i] = HalfBitsToFloat(bits);
        }
      } else {
        if (bytes.size() != static_cast<size_t>(count) * 4) {
          return Error(kLiteRtStatusErrorUnsupported,
                       "QSA constant has the wrong size");
        }
        std::memcpy(values.data(), bytes.data(), bytes.size());
      }
      return AddFloatConstant(values, dims,
                              name + std::to_string(tensor.TensorIndex()),
                              nvinfer1::DataType::kFLOAT);
    }
    if (const auto raw = raw_input_map_.find(tensor.Get());
        raw != raw_input_map_.end() && raw->second != nullptr &&
        raw->second->getType() == nvinfer1::DataType::kFLOAT) {
      return raw->second;
    }
    LITERT_ASSIGN_OR_RETURN(auto* value, GetTensor(tensor));
    return AddCastTensor(value, nvinfer1::DataType::kFLOAT);
  }

  // The mask of a QSA plugin: bool as it is, a float mask in FP32.
  Expected<nvinfer1::ITensor*> GetQsaMask(const Tensor& mask) {
    if (mask.ElementType() == litert::ElementType::Bool) {
      return GetTensor(mask);
    }
    return GetQsaFp32Operand(mask, "qsa_mask_");
  }

  // odml.qsa_indexer_update as the QSA indexer plugin (ParseQsaIndexerUpdate):
  // the ring and the pool enter and leave the network in the model's type.
  Expected<void> LowerCompositeQsaIndexer(const Op& op) {
    const auto parsed = ParseQsaIndexerUpdate(op);
    if (!parsed.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported qsa_indexer_update composite");
    }
    const auto& p = parsed->params;
    const auto& in = op.Inputs();
    LITERT_ASSIGN_OR_RETURN(auto* q, GetTensor(in[0]));
    LITERT_ASSIGN_OR_RETURN(auto* k_new, GetTensor(in[1]));
    if (k_new->getType() != q->getType()) {
      LITERT_ASSIGN_OR_RETURN(k_new, AddCastTensor(k_new, q->getType()));
    }
    nvinfer1::ITensor* states[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
      const auto raw = raw_input_map_.find(in[2 + i].Get());
      if (raw == raw_input_map_.end() || raw->second == nullptr) {
        return Error(kLiteRtStatusErrorUnsupported,
                     "QSA indexer state is not a network input");
      }
      states[i] = raw->second;
    }
    LITERT_ASSIGN_OR_RETURN(auto* norm,
                            GetQsaFp32Operand(in[4], "qsa_k_norm_weight_"));
    LITERT_ASSIGN_OR_RETURN(auto* cos, GetQsaFp32Operand(in[5], "qsa_cos_"));
    LITERT_ASSIGN_OR_RETURN(auto* sin, GetQsaFp32Operand(in[6], "qsa_sin_"));
    LITERT_ASSIGN_OR_RETURN(auto* start_pos, GetTensor(in[7]));
    LITERT_ASSIGN_OR_RETURN(auto* mask, GetQsaMask(in[8]));
    TrtPtr<nvinfer1::IPluginV3> plugin(
        CreateQsaIndexerPlugin(p.block_topk, p.rms_norm_eps, p.scale));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the QSA indexer plugin");
    }
    nvinfer1::ITensor* inputs[] = {q,   k_new, states[0], states[1], norm,
                                   cos, sin,   start_pos, mask};
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs, std::size(inputs), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr ||
        layer->getOutput(1) == nullptr || layer->getOutput(2) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the QSA indexer plugin layer");
    }
    const std::string name =
        "cuda_qsa_indexer_" + std::to_string(in[3].TensorIndex());
    layer->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    for (int o = 0; o < 3; ++o) {
      layer->getOutput(o)->setName(
          KeepName(UniqueName(name + "_" + std::to_string(o))));
    }
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX QSA indexer: %d rows, %d heads of %d, %d "
               "blocks, top %d",
               p.rows, p.index_heads, p.index_dim, p.max_blocks, p.block_topk);
    // The states leave the network in the model's type, unrounded.
    for (int o = 1; o < 3; ++o) {
      raw_output_map_[op.Outputs()[o].Get()] = layer->getOutput(o);
    }
    for (int o = 0; o < 3; ++o) {
      LITERT_RETURN_IF_ERROR(
          SetOutputTensor(op.Outputs()[o], layer->getOutput(o)));
    }
    return {};
  }

  // odml.sparse_sdpa_transposed as the QSA attention plugin
  // (ParseSparseSdpaTransposed), over the caches as this network holds them:
  // the in-place updates' results (FP16, the value cache as rows when only
  // attention reads it) or the graph's cache inputs.
  Expected<void> LowerCompositeSparseSdpa(const Op& op) {
    const auto parsed = ParseSparseSdpaTransposed(op);
    if (!parsed.has_value()) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "Unsupported sparse_sdpa_transposed composite");
    }
    const auto& p = parsed->params;
    const auto& in = op.Inputs();
    LITERT_ASSIGN_OR_RETURN(auto* q, GetTensor(in[0]));
    nvinfer1::ITensor* caches[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
      const Tensor cache = in[1 + i];
      if (cache.IsSubgraphInput()) {
        const auto raw = raw_input_map_.find(cache.Get());
        if (raw == raw_input_map_.end() || raw->second == nullptr) {
          return Error(kLiteRtStatusErrorUnsupported,
                       "QSA attention cache is not a network input");
        }
        caches[i] = raw->second;
      } else {
        LITERT_ASSIGN_OR_RETURN(caches[i], GetTensor(cache));
      }
    }
    if (caches[0]->getType() != nvinfer1::DataType::kHALF &&
        caches[0]->getType() != nvinfer1::DataType::kFLOAT) {
      return Error(kLiteRtStatusErrorUnsupported,
                   "QSA attention caches must be FP16 or FP32");
    }
    if (caches[1]->getType() != caches[0]->getType()) {
      LITERT_ASSIGN_OR_RETURN(caches[1],
                              AddCastTensor(caches[1], caches[0]->getType()));
    }
    // The value cache: rows [1, H, L, D] when held transposed, else the
    // model's columns [1, H, D, L] (through any view of them).
    const bool rows = transposed_caches_.count(in[2].Get()) != 0;
    nvinfer1::Dims value_dims{};
    value_dims.nbDims = 4;
    value_dims.d[0] = 1;
    value_dims.d[1] = p.kv_heads;
    value_dims.d[2] = rows ? p.cache_len : p.head_dim;
    value_dims.d[3] = rows ? p.head_dim : p.cache_len;
    const nvinfer1::Dims held = caches[1]->getDimensions();
    bool same = held.nbDims == 4;
    for (int d = 0; same && d < 4; ++d) same = held.d[d] == value_dims.d[d];
    if (!same) {
      LITERT_ASSIGN_OR_RETURN(caches[1], ReshapeTensor(caches[1], value_dims));
    }
    LITERT_ASSIGN_OR_RETURN(auto* topk, GetTensor(in[3]));
    LITERT_ASSIGN_OR_RETURN(auto* mask, GetQsaMask(in[4]));
    TrtPtr<nvinfer1::IPluginV3> plugin(
        CreateQsaAttentionPlugin(p.scale, /*value_columns=*/!rows));
    if (!plugin) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to create the QSA attention plugin");
    }
    nvinfer1::ITensor* inputs[] = {q, caches[0], caches[1], topk, mask};
    auto* layer = tensorrt_rtx_1_5_0_99::AddPluginV3(
        *network_, inputs, std::size(inputs), *plugin);
    if (layer == nullptr || layer->getOutput(0) == nullptr) {
      return Error(kLiteRtStatusErrorCompilation,
                   "Failed to add the QSA attention plugin layer");
    }
    const std::string name =
        "cuda_qsa_attention_" + std::to_string(in[1].TensorIndex());
    layer->setName(KeepName(UniqueName(name)));
    layer->getOutput(0)->setName(KeepName(UniqueName(name)));
    owned_plugins_.push_back(std::move(plugin));
    LITERT_LOG(LITERT_INFO,
               "NVIDIA TensorRT-RTX QSA attention: %d rows, %d heads on %d of "
               "%d, %d keys, top %d blocks, values as %s",
               p.rows, p.q_heads, p.kv_heads, p.head_dim, p.cache_len,
               p.block_topk, rows ? "rows" : "columns");
    return SetOutputTensor(op.Outputs()[0], layer->getOutput(0));
  }

  Expected<void> LowerComposite(const Op& op) {
    const std::string name = CompositeOpName(op);
    if (name == "odml.rms_norm") {
      return LowerCompositeRmsNorm(op);
    }
    if (name == "odml.qkv_norm_rope") {
      return LowerCompositeQkvNormRope(op);
    }
    if (name == "odml.short_conv_step") {
      return LowerCompositeShortConvStep(op);
    }
    if (name == "odml.swiglu") {
      return LowerCompositeSwiglu(op);
    }
    if (name == "odml.cache_update") {
      if (const auto form = ParseQuantCacheUpdate(op); form.has_value()) {
        return LowerQuantCacheUpdate(op, *form);
      }
      if (const auto form = ParseGeneralCacheUpdate(op); form.has_value()) {
        return LowerGeneralCacheUpdate(op, *form);
      }
      return LowerCompositeCacheUpdate(op);
    }
    if (name == "odml.sdpa_transposed") {
      if (const auto form = ParseQuantSdpa(op); form.has_value()) {
        return LowerQuantSdpa(op, *form);
      }
      if (const auto form = ParseGeneralSdpa(op); form.has_value()) {
        return LowerGeneralSdpa(op, *form);
      }
    }
    if (name == "odml.fused_sdpa_cache_update") {
      if (const auto form = ParseQuantFusedSdpa(op); form.has_value()) {
        return LowerQuantFusedSdpa(op, *form);
      }
      if (const auto form = ParseGeneralFusedSdpa(op); form.has_value()) {
        return LowerGeneralFusedSdpa(op, *form);
      }
    }
    if (name == "odml.runtime_bmm") {
      return LowerCompositeRuntimeBmm(op);
    }
    if (name == "odml.qsa_indexer_update") {
      return LowerCompositeQsaIndexer(op);
    }
    if (name == "odml.sparse_sdpa_transposed") {
      return LowerCompositeSparseSdpa(op);
    }
    return Error(kLiteRtStatusErrorUnsupported,
                 "Unsupported composite op during lowering");
  }

  Expected<void> LowerOp(const Op& op) {
    LITERT_RETURN_IF_ERROR(LowerOpByCode(op));
    if (Fp16ActivationsEnabled() && IsExactFp32PathOp(op)) {
      return LowerExactFp32(op);
    }
    return {};
  }

  // Arithmetic and data movement on values derived from positions (see
  // exact_fp32_map_).
  bool IsExactFp32PathOp(const Op& op) const {
    switch (op.Code()) {
      case kLiteRtOpCodeTflAdd:
      case kLiteRtOpCodeTflMul:
      case kLiteRtOpCodeTflSub:
      case kLiteRtOpCodeTflDiv:
      case kLiteRtOpCodeTflReshape:
      case kLiteRtOpCodeTflSlice:
      case kLiteRtOpCodeTflConcatenation:
      case kLiteRtOpCodeTflTranspose:
      case kLiteRtOpCodeTflSum:
        break;
      default:
        return false;
    }
    for (const auto& input : op.Inputs()) {
      if (HasExactFp32Value(input)) {
        return true;
      }
    }
    return false;
  }

  // Lowers `op` a second time on the FP32 views of its float inputs and
  // records its result in exact_fp32_map_; tensor_map_ keeps the
  // activation-type result. Skipped unless every float input has an FP32 view
  // (an exact value or an FP32 constant).
  Expected<void> LowerExactFp32(const Op& op) {
    struct Swap {
      LiteRtTensor tensor;
      nvinfer1::ITensor* exact;
      nvinfer1::ITensor* saved;
    };
    std::vector<Swap> swaps;
    for (const auto& input : op.Inputs()) {
      if (!IsFloatLike(input.ElementType())) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(auto* exact, GetExactFp32Tensor(input));
      if (exact == nullptr) {
        return {};
      }
      auto it = tensor_map_.find(input.Get());
      swaps.push_back(
          {input.Get(), exact, it == tensor_map_.end() ? nullptr : it->second});
    }
    for (const auto& swap : swaps) {
      tensor_map_[swap.tensor] = swap.exact;
    }
    lowering_exact_fp32_ = true;
    auto lowered = LowerOpByCode(op);
    lowering_exact_fp32_ = false;
    for (const auto& swap : swaps) {
      if (swap.saved != nullptr) {
        tensor_map_[swap.tensor] = swap.saved;
      } else {
        tensor_map_.erase(swap.tensor);
      }
    }
    return lowered;
  }

  Expected<void> LowerOpByCode(const Op& op) {
    if (auto fused = fused_attention_ops_.find(op.Get());
        fused != fused_attention_ops_.end()) {
      return LowerDecodeAttentionMember(fused->second);
    }
    if (auto gdn = gdn_anchor_ops_.find(op.Get());
        gdn != gdn_anchor_ops_.end()) {
      return LowerGatedDeltaNetBlock(gdn->second);
    }
    if (auto region = quant_regions_.find(op.Get());
        region != quant_regions_.end()) {
      return LowerQuantRegion(region->second);
    }
    if (unlowered_ops_.count(op.Get()) != 0) {
      return {};
    }
    switch (op.Code()) {
      case kLiteRtOpCodeTflAdd:
      case kLiteRtOpCodeTflMul:
      case kLiteRtOpCodeTflSub:
      case kLiteRtOpCodeTflDiv:
      case kLiteRtOpCodeTflMaximum:
      case kLiteRtOpCodeTflMinimum:
        return LowerElementwise(op);
      case kLiteRtOpCodeTflLess:
      case kLiteRtOpCodeTflGreater:
      case kLiteRtOpCodeTflGreaterEqual:
      case kLiteRtOpCodeTflNotEqual:
      case kLiteRtOpCodeTflEqual:
      case kLiteRtOpCodeTflLogicalAnd:
      case kLiteRtOpCodeTflLogicalOr:
        return LowerComparison(op);
      case kLiteRtOpCodeTflBitwiseXor:
        return LowerBitwiseXor(op);
      case kLiteRtOpCodeTflGather:
        return LowerGather(op);
      case kLiteRtOpCodeTflRelu:
      case kLiteRtOpCodeTflRelu6:
      case kLiteRtOpCodeTflReluN1To1:
        return LowerActivation(op);
      case kLiteRtOpCodeTflSin:
      case kLiteRtOpCodeTflCos:
      case kLiteRtOpCodeTflTanh:
      case kLiteRtOpCodeTflRsqrt:
      case kLiteRtOpCodeTflLogistic:
      case kLiteRtOpCodeTflExp:
      case kLiteRtOpCodeTflLog:
      case kLiteRtOpCodeTflSqrt:
      case kLiteRtOpCodeTflAbs:
      case kLiteRtOpCodeTflSign:
        return LowerUnary(op);
      case kLiteRtOpCodeTflGelu:
        return LowerGelu(op);
      case kLiteRtOpCodeTflCast:
        return LowerCast(op);
      case kLiteRtOpCodeTflQuantize:
        return LowerQuantize(op);
      case kLiteRtOpCodeTflDequantize:
        return LowerDequantize(op);
      case kLiteRtOpCodeTflFill:
        return LowerFill(op);
      case kLiteRtOpCodeTflSoftmax:
        return LowerSoftmax(op);
      case kLiteRtOpCodeTflReshape:
        return LowerReshape(op);
      case kLiteRtOpCodeTflPack:
        return LowerPack(op);
      case kLiteRtOpCodeTflSlice:
        return LowerSlice(op);
      case kLiteRtOpCodeTflUnpack:
        return LowerUnpack(op);
      case kLiteRtOpCodeTflDynamicUpdateSlice:
        return LowerDynamicUpdateSlice(op);
      case kLiteRtOpCodeTflTranspose:
        return LowerTranspose(op);
      case kLiteRtOpCodeTflConcatenation:
        return LowerConcatenation(op);
      case kLiteRtOpCodeTflBatchMatmul:
        return LowerBatchMatmul(op);
      case kLiteRtOpCodeTflOneHot:
        return LowerOneHot(op);
      case kLiteRtOpCodeTflFullyConnected:
        return LowerFullyConnected(op);
      case kLiteRtOpCodeTflSelectV2:
        return LowerSelectV2(op);
      case kLiteRtOpCodeTflSum:
        return LowerSum(op);
      case kLiteRtOpCodeTflMean:
        return LowerMean(op);
      case kLiteRtOpCodeTflTopkV2:
        return LowerTopkV2(op);
      case kLiteRtOpCodeTflReduceMax:
        return LowerReduceMax(op);
      case kLiteRtOpCodeTflFloorMod:
        return LowerFloorMod(op);
      case kLiteRtOpCodeTflConv2d:
        return LowerConv2d(op);
      case kLiteRtOpCodeShloComposite:
        return LowerComposite(op);
      case kLiteRtOpCodeTflCustom:
        return LowerCustomOp(op);
      default:
        return Error(kLiteRtStatusErrorUnsupported,
                     "Unsupported TensorRT op during lowering");
    }
  }

  MemoryProfiler memory_profiler_{"compiler"};
  TensorRtLogger logger_;
  SyncCudaGpuAllocator sync_allocator_;
  TrtPtr<nvinfer1::IBuilder> builder_;
  TrtPtr<nvinfer1::INetworkDefinition> network_;
  TrtPtr<nvinfer1::IBuilderConfig> config_;
  std::deque<std::string> names_;
  std::vector<std::vector<uint8_t>> owned_weights_;
  std::vector<OwnedRefitWeight> owned_refit_weights_;
  std::vector<TrtPtr<nvinfer1::IPluginV3>> owned_plugins_;
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> tensor_map_;
  // In FP16/BF16 activation mode, FP32 views of the floats a model derives
  // from integer positions: casts, then arithmetic and data movement up to
  // sin/cos (rotary angles) or a cast back to integers (cache update offsets).
  // Narrowing them would round the positions (BF16 keeps 8 significant bits,
  // so above 256 they are off by up to 2, above 512 by up to 4), moving cache
  // writes and rotating by angles off by whole radians. Only sin/cos and
  // integer casts read these views; every other consumer sees tensor_map_.
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> exact_fp32_map_;
  // FP32 copies of float constants used on that path (inverse frequencies).
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> exact_fp32_constants_;
  // Set while LowerExactFp32 re-lowers an op; SetOutputTensor then records the
  // FP32 result in exact_fp32_map_ instead of narrowing it into tensor_map_.
  bool lowering_exact_fp32_ = false;
  // Network inputs in their boundary (pre-FP16-conversion) form; used by
  // in-place style ops (cache updates) that operate on the raw int8 data.
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> raw_input_map_;
  // The prefill's token positions input and its element count (0: none, -1:
  // more than one), for the CUDA GEMM (AddSubbyteGemvPlugin).
  nvinfer1::ITensor* row_positions_ = nullptr;
  int64_t row_positions_count_ = 0;
  // Produced tensors that already carry the boundary format (e.g. int8 cache
  // scatter results); MarkOutputs prefers these over the FP16 view.
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> raw_output_map_;
  // The INT4 caches the engine declares as their INT8 bytes
  // (FindPackedInt4Caches); true for a value cache.
  std::unordered_map<LiteRtTensor, bool> packed_int4_caches_;
  // UINT8 caches of the general forms, declared as INT8 tensors of their
  // bytes (FindGeneralCaches).
  std::unordered_set<LiteRtTensor> byte_caches_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  int next_name_id_ = 0;
  bool uses_cuda_subbyte_gemv_ = false;
  // Whether the subgraph holds a gated_delta_update op, which puts its
  // blockwise GEMMs on the int8 tensor cores by default: set by Build.
  bool graph_has_gated_delta_net_ = false;
  // The weight store of the model, if the plugin weights stay out of the plan,
  // the plugin layers that read a segment of it, and the holders added for
  // those segments.
  struct HolderReader {
    nvinfer1::ILayer* layer = nullptr;
    uint32_t segment = 0;
    bool scales_too = false;  // the blockwise scales are in the segment too
    int32_t input = 1;        // the layer's input that takes the holder
  };
  // The odml.cache_update ops whose token the attention over their caches
  // writes (LowerQuantCacheUpdate), by their updated key cache, until that
  // attention is lowered.
  std::unordered_map<LiteRtTensor, Op> joined_quant_cache_updates_;
  // Float32 caches declared to TensorRT as FP16 (Fp16CacheStorageEnabled).
  std::unordered_set<LiteRtTensor> fp16_stored_caches_;
  // Cache inputs another engine keeps in place (FindSharedCacheInputs).
  std::unordered_set<LiteRtTensor> shared_cache_inputs_;
  TensorRtWeightStoreBuilder* weight_store_ = nullptr;
  nvinfer1::ITensor* provisional_weight_holder_ = nullptr;
  std::vector<HolderReader> holder_readers_;
  std::vector<TensorRtWeightHolderBuildData> weight_holders_;
  // Lowering order of this partition's ops, and fused GEMV group outputs
  // awaiting their owning op (see AddCudaSubbyteGemvGroup).
  std::unordered_map<LiteRtOp, size_t> op_order_;
  std::unordered_map<LiteRtOp, nvinfer1::ITensor*> fused_gemv_outputs_;
  // Subgraph-input caches updated in place (see FindInPlaceCacheUpdates),
  // by cache input and by updated output.
  std::unordered_map<LiteRtTensor, InPlaceCache> in_place_caches_;
  std::unordered_map<LiteRtTensor, InPlaceCache> in_place_outputs_;
  std::unordered_set<LiteRtTensor> graph_outputs_;
  // Value cache inputs and their updated outputs held as [B, H, S, D],
  // including read-only inputs explicitly shared from another compiled model.
  std::unordered_set<LiteRtTensor> transposed_caches_;
  // Attention blocks fused into the CUDA plugin, by member op, and the mask
  // tilings that only fed them.
  std::vector<DecodeAttentionBlock> attention_blocks_;
  std::vector<WindowedCacheRead> windowed_reads_;
  std::unordered_map<LiteRtOp, size_t> windowed_by_scores_;
  std::unordered_map<LiteRtOp, size_t> windowed_by_select_;
  std::unordered_map<LiteRtOp, size_t> windowed_by_values_;
  // heads, rows, depth, seq of the last INT8-cache block matched (log).
  std::array<int, 4> int8_attention_geometry_ = {0, 0, 0, 0};
  std::unordered_map<LiteRtOp, size_t> fused_attention_ops_;
  std::unordered_set<LiteRtOp> unlowered_ops_;
  std::unordered_map<LiteRtOp, MoeBlockPlan> moe_plans_;
  // Each member of a fused GEMV group: the group's output and the member's
  // columns in it (see AddCudaSubbyteGemvGroup).
  std::unordered_map<LiteRtOp, GemvGroupSlice> gemv_group_slices_;
  // Fused gated DeltaNet layers by the op that emits them
  // (FindGatedDeltaNetBlocks).
  std::vector<GatedDeltaNetBlock> gdn_blocks_;
  std::unordered_map<LiteRtOp, size_t> gdn_anchor_ops_;
  // The statically quantized products (FindQuantRegions): the INT8 values
  // the quant layers produced, by model tensor; the regions, by the op that
  // emits them; the INT8 tensors the product regions will produce and read.
  std::unordered_map<LiteRtTensor, QuantValue> quant_values_;
  // The FLOAT16 results of the quant layers, by model tensor.
  std::unordered_map<LiteRtTensor, nvinfer1::ITensor*> quant_floats_;
  std::unordered_map<LiteRtOp, QuantRegion> quant_regions_;
  std::unordered_set<LiteRtTensor> quant_outputs_;
  std::unordered_set<LiteRtTensor> quant_product_inputs_;
  // The projections regions (quant_projections_) by the tensor they read.
  std::unordered_map<LiteRtTensor, size_t> quant_projections_by_input_;
  std::unordered_map<LiteRtTensor, size_t> quant_factorized_mlps_by_input_;
  std::vector<QuantProjections> quant_projections_;
  std::vector<QuantFactorizedMlp> quant_factorized_mlps_;
  std::vector<QuantGlue> quant_glues_;
  std::vector<QuantQkvRope> quant_qkv_ropes_;
  std::vector<QuantHead> quant_heads_;

  // Experimental, inert unless LITERT_NVIDIA_MEGAKERNEL is set: the decode
  // megakernel's lowering, as members of this builder.
#include "litert/vendors/nvidia/experimental/megakernel/megakernel_lowering.inc"
};

}  // namespace

bool TensorRtSharedWeightsEnabled() {
  // On by default when a weight store can hold the plans' packed weights: an
  // AOT cache directory, the model file the store maps (compiler_plugin.cc,
  // CreateWeightStore) and the store not turned off. Without the store, shared
  // weights build weight-stripped plans that run only on the current GPU and
  // are refitted at load, which LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS=1 still
  // asks for.
  const auto set = [](const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
  };
  const bool weight_store =
      set("LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR") &&
      (set("LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH") || set("G4MODEL")) &&
      EnvEnabled("LITERT_NVIDIA_TENSORRT_WEIGHT_STORE", /*default_value=*/true);
  return EnvEnabled("LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS", weight_store);
}

bool IsTensorRtOpSupported(const Op& op) {
  Expected<bool> result = false;
  switch (op.Code()) {
    case kLiteRtOpCodeTflAdd:
    case kLiteRtOpCodeTflMul:
    case kLiteRtOpCodeTflSub:
    case kLiteRtOpCodeTflDiv:
    case kLiteRtOpCodeTflMaximum:
    case kLiteRtOpCodeTflMinimum:
      result = IsElementwiseSupported(op);
      break;
    case kLiteRtOpCodeTflLess:
    case kLiteRtOpCodeTflGreater:
    case kLiteRtOpCodeTflGreaterEqual:
    case kLiteRtOpCodeTflNotEqual:
    case kLiteRtOpCodeTflEqual:
      result = IsComparisonSupported(op);
      break;
    case kLiteRtOpCodeTflLogicalAnd:
    case kLiteRtOpCodeTflLogicalOr:
      result = IsLogicalAndSupported(op);
      break;
    case kLiteRtOpCodeTflBitwiseXor:
      result = IsBitwiseXorSupported(op);
      break;
    case kLiteRtOpCodeTflGather:
      result = IsGatherSupported(op);
      break;
    case kLiteRtOpCodeTflRelu:
    case kLiteRtOpCodeTflRelu6:
    case kLiteRtOpCodeTflReluN1To1:
      result = IsUnaryActivationSupported(op);
      break;
    case kLiteRtOpCodeTflSin:
    case kLiteRtOpCodeTflCos:
    case kLiteRtOpCodeTflTanh:
    case kLiteRtOpCodeTflRsqrt:
    case kLiteRtOpCodeTflLogistic:
    case kLiteRtOpCodeTflExp:
    case kLiteRtOpCodeTflLog:
    case kLiteRtOpCodeTflSqrt:
    case kLiteRtOpCodeTflAbs:
    case kLiteRtOpCodeTflSign:
      result = IsUnaryActivationSupported(op);
      break;
    case kLiteRtOpCodeTflGelu:
      result = IsGeluSupported(op);
      break;
    case kLiteRtOpCodeTflCast:
      result = IsCastSupported(op);
      break;
    case kLiteRtOpCodeTflQuantize:
      result = IsQuantizeSupported(op);
      break;
    case kLiteRtOpCodeTflDequantize:
      result = IsDequantizeSupported(op);
      break;
    case kLiteRtOpCodeTflFill:
      result = IsFillSupported(op);
      break;
    case kLiteRtOpCodeTflSoftmax:
      result = IsSoftmaxSupported(op);
      break;
    case kLiteRtOpCodeTflReshape:
      result = IsReshapeSupported(op);
      break;
    case kLiteRtOpCodeTflPack:
      result = IsPackSupported(op);
      break;
    case kLiteRtOpCodeTflSlice:
      result = IsSliceSupported(op);
      break;
    case kLiteRtOpCodeTflUnpack:
      result = IsUnpackSupported(op);
      break;
    case kLiteRtOpCodeTflDynamicUpdateSlice:
      result = IsDynamicUpdateSliceSupported(op);
      break;
    case kLiteRtOpCodeTflTranspose:
      result = IsTransposeSupported(op);
      break;
    case kLiteRtOpCodeTflConcatenation:
      result = IsConcatenationSupported(op);
      break;
    case kLiteRtOpCodeTflBatchMatmul:
      result = IsBatchMatmulSupported(op);
      break;
    case kLiteRtOpCodeTflOneHot:
      // Only as the selector of a table lookup, whose rows are gathered.
      result = OneHotTableLookup(op).has_value();
      break;
    case kLiteRtOpCodeTflFullyConnected:
      result = IsFullyConnectedSupported(op);
      break;
    case kLiteRtOpCodeTflSelectV2:
      result = IsSelectV2Supported(op);
      break;
    case kLiteRtOpCodeTflSum:
      result = IsSumSupported(op);
      break;
    case kLiteRtOpCodeTflMean:
      result = IsMeanSupported(op);
      break;
    case kLiteRtOpCodeTflTopkV2:
      result = IsTopkV2Supported(op);
      break;
    case kLiteRtOpCodeTflReduceMax:
      result = IsReduceMaxSupported(op);
      break;
    case kLiteRtOpCodeTflFloorMod:
      result = IsFloorModSupported(op);
      break;
    case kLiteRtOpCodeTflConv2d:
      result = IsConv2dSupported(op);
      break;
    case kLiteRtOpCodeShloComposite:
      result = IsCompositeSupported(op);
      break;
    case kLiteRtOpCodeTflCustom:
      result = IsCustomOpSupported(op);
      break;
    default:
      return false;
  }
  return result.HasValue() && result.Value();
}

Expected<TensorRtBuildResult> BuildTensorRtEngine(
    const Subgraph& subgraph,
    absl::Span<const std::string> read_only_value_cache_inputs,
    TensorRtWeightStoreBuilder* weight_store,
    absl::Span<const std::string> shared_cache_inputs) {
  const MemoryProfiler memory_profiler("compiler");
  auto result = [&]() {
    TensorRtGraphBuilder builder;
    return builder.Build(subgraph, read_only_value_cache_inputs, weight_store,
                         shared_cache_inputs);
  }();
  memory_profiler.Log("graph_builder_destroyed");
  return result;
}

// Experimental (experimental/megakernel/megakernel_lowering.h).
Expected<TensorRtBuildResult> BuildMegakernelEngine(
    const Subgraph& subgraph,
    absl::Span<const std::string> read_only_value_cache_inputs,
    TensorRtWeightStoreBuilder* weight_store,
    absl::Span<const std::string> shared_cache_inputs) {
  TensorRtGraphBuilder builder;
  return builder.BuildMegakernel(subgraph, read_only_value_cache_inputs,
                                 weight_store, shared_cache_inputs);
}

}  // namespace litert::nvidia
