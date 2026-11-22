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

#include "litert/vendors/nvidia/compiler/general_attention_plugin.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/vendors/nvidia/compiler/plugin_fields.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/general_attention.h"

namespace litert::nvidia {
namespace {

using plugin_fields::BitsFloat;
using plugin_fields::ElementBytes;
using plugin_fields::FloatBits;
using plugin_fields::IntFields;
using plugin_fields::IsActivationType;
using plugin_fields::kPluginNamespace;
using plugin_fields::kPluginVersion;
using plugin_fields::Report;
using plugin_fields::Volume;

constexpr char kName[] = "LiteRtNvidiaGeneralAttention";
constexpr int kFieldCount = 10;
constexpr std::array<const char*, kFieldCount> kFieldNames = {
    "mode",         "cache_type",        "k_scale_bits",
    "v_scale_bits", "q_int8_scale_bits", "new_int8_scale_bits",
    "scale_bits",   "softcap_bits",      "v_token_major",
    "update_cache"};
enum Field {
  kMode = 0,
  kCacheType,
  kKScale,
  kVScale,
  kQInt8Scale,
  kNewInt8Scale,
  kScale,
  kSoftcap,
  kVTokenMajor,
  kUpdateCache,
};

// Inputs of each mode.
enum AttentionInput { kQ = 0, kKCache, kVCache, kMask, kNewK, kNewV, kParams };
enum UpdateInput { kUpKCache = 0, kUpVCache, kUpK, kUpV, kUpKStart, kUpVStart };

int32_t GeneralFloat(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kHALF:
      return kLiteRtNvidiaGeneralFloat16;
    case nvinfer1::DataType::kBF16:
      return kLiteRtNvidiaGeneralBFloat16;
    default:
      return kLiteRtNvidiaGeneralFloat32;
  }
}

bool IsQuantCache(int32_t cache_type) {
  return cache_type == kLiteRtNvidiaGeneralInt8 ||
         cache_type == kLiteRtNvidiaGeneralInt4 ||
         cache_type == kLiteRtNvidiaGeneralUint8;
}

// The TensorRT type a cache of `cache_type` crosses the engine as.
nvinfer1::DataType CacheTensorType(int32_t cache_type) {
  switch (cache_type) {
    case kLiteRtNvidiaGeneralFloat16:
      return nvinfer1::DataType::kHALF;
    case kLiteRtNvidiaGeneralBFloat16:
      return nvinfer1::DataType::kBF16;
    case kLiteRtNvidiaGeneralFloat32:
      return nvinfer1::DataType::kFLOAT;
    default:
      return nvinfer1::DataType::kINT8;
  }
}

class GeneralAttentionPlugin final
    : public plugin_fields::PluginBase<kFieldCount> {
 public:
  explicit GeneralAttentionPlugin(const IntFields<kFieldCount>& fields) noexcept
      : PluginBase<kFieldCount>(fields) {}

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) GeneralAttentionPlugin(fields_);
  }
  const char* getPluginName() const noexcept override { return kName; }
  int32_t getNbOutputs() const noexcept override {
    switch (Mode()) {
      case GeneralAttentionMode::kAttention:
        return 1;
      case GeneralAttentionMode::kFusedAttention:
        return fields_[kUpdateCache] ? 3 : 1;
      default:
        return 2;
    }
  }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != NumInputs() || num_outputs != getNbOutputs()) {
      return 1;
    }
    if (Mode() == GeneralAttentionMode::kCacheUpdate) {
      output_types[0] = input_types[kUpKCache];
      output_types[1] = input_types[kUpVCache];
      return 0;
    }
    output_types[0] = input_types[kQ];
    if (num_outputs == 3) {
      output_types[1] = input_types[kKCache];
      output_types[2] = input_types[kVCache];
    }
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != NumInputs() ||
        num_outputs != getNbOutputs() || position < 0 ||
        position >= num_inputs + num_outputs ||
        in_out[position].desc.format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    const nvinfer1::DataType type = in_out[position].desc.type;
    const nvinfer1::DataType cache = CacheTensorType(fields_[kCacheType]);
    if (Mode() == GeneralAttentionMode::kCacheUpdate) {
      switch (position) {
        case kUpKCache:
        case kUpVCache:
          return type == cache;
        case kUpK:
          return IsActivationType(type);
        case kUpV:
          return type == in_out[kUpK].desc.type;
        case kUpKStart:
        case kUpVStart:
          return type == nvinfer1::DataType::kINT32;
        default:
          return type == cache;
      }
    }
    if (position >= num_inputs) {
      return position == num_inputs ? type == in_out[kQ].desc.type
                                    : type == cache;
    }
    switch (position) {
      case kQ:
        return IsActivationType(type);
      case kKCache:
      case kVCache:
        return type == cache;
      case kMask:
        return type == nvinfer1::DataType::kBOOL ||
               type == nvinfer1::DataType::kFLOAT;
      case kNewK:
        return IsActivationType(type);
      case kNewV:
        return type == in_out[kNewK].desc.type;
      default:
        return type == nvinfer1::DataType::kINT32;
    }
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expr_builder);
    if (inputs == nullptr || outputs == nullptr || num_inputs != NumInputs() ||
        num_shape_inputs != 0 || num_outputs != getNbOutputs()) {
      return 1;
    }
    if (Mode() == GeneralAttentionMode::kCacheUpdate) {
      outputs[0] = inputs[kUpKCache];
      outputs[1] = inputs[kUpVCache];
      return 0;
    }
    outputs[0] = inputs[kQ];
    if (num_outputs == 3) {
      outputs[1] = inputs[kKCache];
      outputs[2] = inputs[kVCache];
    }
    return 0;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(outputs);
    if (inputs == nullptr || num_inputs != NumInputs() ||
        num_outputs != getNbOutputs() ||
        Mode() == GeneralAttentionMode::kCacheUpdate) {
      return 0;
    }
    LiteRtNvidiaGeneralAttentionParams params;
    if (!AttentionParams(inputs[kQ].max, inputs[kKCache].max, inputs[kMask].max,
                         num_inputs > kNewK ? &inputs[kNewK].max : nullptr,
                         inputs[kQ].desc.type, inputs[kMask].desc.type,
                         num_inputs > kNewK ? inputs[kNewK].desc.type
                                            : nvinfer1::DataType::kHALF,
                         &params)) {
      return 0;
    }
    return LiteRtNvidiaGeneralAttentionWorkspaceBytes(&params);
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    if (Mode() == GeneralAttentionMode::kCacheUpdate) {
      return EnqueueUpdate(input_desc, inputs, outputs, stream);
    }
    const bool fused = Mode() == GeneralAttentionMode::kFusedAttention;
    LiteRtNvidiaGeneralAttentionParams params;
    if (!AttentionParams(
            input_desc[kQ].dims, input_desc[kKCache].dims,
            input_desc[kMask].dims, fused ? &input_desc[kNewK].dims : nullptr,
            input_desc[kQ].type, input_desc[kMask].type,
            fused ? input_desc[kNewK].type : nvinfer1::DataType::kHALF,
            &params)) {
      return 1;
    }
    const bool write = fused && fields_[kUpdateCache] != 0;
    if (write) {
      for (int i = 0; i < 2; ++i) {
        if (const int32_t status =
                CopyCache(input_desc[kKCache + i], inputs[kKCache + i],
                          outputs[1 + i], stream)) {
          return status;
        }
      }
    }
    // The attention reads the caches before the chunk goes into them.
    const cudaError_t status = LiteRtNvidiaLaunchGeneralAttention(
        &params, inputs[kQ], inputs[kKCache], inputs[kVCache],
        fused ? inputs[kNewK] : nullptr, fused ? inputs[kNewV] : nullptr,
        inputs[kMask], outputs[0], workspace, stream);
    if (status != cudaSuccess || !write) return Report(kName, status);
    LiteRtNvidiaGeneralWriteParams write_params{};
    write_params.heads = params.kv_heads;
    write_params.cache_len = params.cache_len;
    write_params.depth = params.depth;
    write_params.tokens = params.new_len;
    write_params.mode = kLiteRtNvidiaGeneralWriteRing;
    write_params.caches = params.caches;
    write_params.src = params.new_kv;
    write_params.src_v_token_major = params.new_v_token_major;
    return Report(kName,
                  LiteRtNvidiaLaunchGeneralCacheWrite(
                      &write_params, outputs[1], outputs[2], inputs[kNewK],
                      inputs[kNewV], nullptr, nullptr,
                      static_cast<const int32_t*>(inputs[kParams]), stream));
  }

 protected:
  int32_t NumInputs() const noexcept override {
    switch (Mode()) {
      case GeneralAttentionMode::kAttention:
        return 4;
      case GeneralAttentionMode::kFusedAttention:
        return 7;
      default:
        return 6;
    }
  }

 private:
  GeneralAttentionMode Mode() const noexcept {
    return static_cast<GeneralAttentionMode>(fields_[kMode]);
  }
  float FieldFloat(int field) const noexcept {
    return BitsFloat(static_cast<uint32_t>(fields_[field]));
  }
  LiteRtNvidiaGeneralCaches Caches() const noexcept {
    return {fields_[kCacheType], FieldFloat(kKScale), FieldFloat(kVScale)};
  }

  bool AttentionParams(
      const nvinfer1::Dims& q, const nvinfer1::Dims& k,
      const nvinfer1::Dims& mask, const nvinfer1::Dims* new_k,
      nvinfer1::DataType q_type, nvinfer1::DataType mask_type,
      nvinfer1::DataType new_type,
      LiteRtNvidiaGeneralAttentionParams* params) const noexcept {
    if (q.nbDims != 4 || k.nbDims != 4 || mask.nbDims != 4 ||
        (new_k != nullptr && new_k->nbDims != 4)) {
      return false;
    }
    *params = {};
    params->kv_heads = static_cast<int32_t>(q.d[1]);
    params->rows = static_cast<int32_t>(q.d[2]);
    params->depth = static_cast<int32_t>(q.d[3]);
    params->cache_len = static_cast<int32_t>(k.d[2]);
    params->new_len = new_k != nullptr ? static_cast<int32_t>(new_k->d[2]) : 0;
    params->mask_rows = static_cast<int32_t>(mask.d[2]);
    params->mask_type = mask_type == nvinfer1::DataType::kBOOL
                            ? kLiteRtNvidiaGeneralMaskBool
                            : kLiteRtNvidiaGeneralMaskFloat;
    params->q = {GeneralFloat(q_type), FieldFloat(kQInt8Scale)};
    params->caches = Caches();
    params->new_kv = {GeneralFloat(new_type), FieldFloat(kNewInt8Scale)};
    params->new_v_token_major = fields_[kVTokenMajor];
    params->scale = FieldFloat(kScale);
    params->softcap = FieldFloat(kSoftcap);
    return k.d[1] == q.d[1] && mask.d[3] == k.d[2] + params->new_len &&
           LiteRtNvidiaGeneralAttentionSupports(params);
  }

  static int32_t CopyCache(const nvinfer1::PluginTensorDesc& desc,
                           const void* input, void* output,
                           cudaStream_t stream) noexcept {
    if (output == input) return 0;
    const size_t bytes =
        static_cast<size_t>(Volume(desc.dims)) * ElementBytes(desc.type);
    return Report(kName, cudaMemcpyAsync(output, input, bytes,
                                         cudaMemcpyDeviceToDevice, stream));
  }

  int32_t EnqueueUpdate(const nvinfer1::PluginTensorDesc* desc,
                        const void* const* inputs, void* const* outputs,
                        cudaStream_t stream) const noexcept {
    const nvinfer1::Dims& k = desc[kUpKCache].dims;
    const nvinfer1::Dims& src = desc[kUpK].dims;
    if (k.nbDims != 4 || src.nbDims != 4) return 1;
    LiteRtNvidiaGeneralWriteParams params{};
    params.heads = static_cast<int32_t>(src.d[1]);
    params.cache_len = static_cast<int32_t>(k.d[2]);
    params.depth = static_cast<int32_t>(src.d[3]);
    params.tokens = static_cast<int32_t>(src.d[2]);
    params.mode = kLiteRtNvidiaGeneralWriteSlice;
    params.caches = Caches();
    params.src = {GeneralFloat(desc[kUpK].type), FieldFloat(kNewInt8Scale)};
    params.src_v_token_major = fields_[kVTokenMajor];
    for (int i = 0; i < 2; ++i) {
      if (const int32_t status = CopyCache(
              desc[kUpKCache + i], inputs[kUpKCache + i], outputs[i], stream)) {
        return status;
      }
    }
    // Element 2 of the keys' start indices and element 3 of the values'.
    return Report(
        kName, LiteRtNvidiaLaunchGeneralCacheWrite(
                   &params, outputs[0], outputs[1], inputs[kUpK], inputs[kUpV],
                   static_cast<const int32_t*>(inputs[kUpKStart]) + 2,
                   static_cast<const int32_t*>(inputs[kUpVStart]) + 3, nullptr,
                   stream));
  }
};

class GeneralAttentionPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  GeneralAttentionPluginCreator() noexcept : fields_(kFieldNames) {}
  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    IntFields<kFieldCount> values(kFieldNames);
    if (!values.Read(fields)) return nullptr;
    return new (std::nothrow) GeneralAttentionPlugin(values);
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return fields_.Collection();
  }
  const char* getPluginName() const noexcept override { return kName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  IntFields<kFieldCount> fields_;
};

}  // namespace

nvinfer1::IPluginV3* CreateGeneralAttentionPlugin(
    const GeneralAttentionOptions& options) noexcept {
  IntFields<kFieldCount> fields(kFieldNames);
  fields[kMode] = static_cast<int32_t>(options.mode);
  fields[kCacheType] = options.cache_type;
  fields[kKScale] = static_cast<int32_t>(FloatBits(options.k_scale));
  fields[kVScale] = static_cast<int32_t>(FloatBits(options.v_scale));
  fields[kQInt8Scale] = static_cast<int32_t>(FloatBits(options.q_int8_scale));
  fields[kNewInt8Scale] =
      static_cast<int32_t>(FloatBits(options.new_int8_scale));
  fields[kScale] = static_cast<int32_t>(FloatBits(options.scale));
  fields[kSoftcap] = static_cast<int32_t>(FloatBits(options.softcap));
  fields[kVTokenMajor] = options.v_token_major ? 1 : 0;
  fields[kUpdateCache] = options.update_cache ? 1 : 0;
  return new (std::nothrow) GeneralAttentionPlugin(fields);
}

void EnsureGeneralAttentionPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(GeneralAttentionPluginCreator);

}  // namespace litert::nvidia
