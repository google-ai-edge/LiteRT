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

#include "litert/vendors/nvidia/compiler/decode_attention_plugin.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/global_attention.h"
#include "litert/vendors/nvidia/trtllm/tiled_attention.h"

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

constexpr char kPluginName[] = "LiteRtNvidiaDecodeAttention";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kFillField[] = "fill";
constexpr char kNewKeysField[] = "new_keys";
// INT8 caches (Gemma 4 E4B): 0 = FP16 caches, 1 = INT8 caches with value rows,
// 2 = INT8 caches with the value cache stored [1, H, D, S].
constexpr char kInt8CacheField[] = "int8_cache";
constexpr char kKeyScaleField[] = "k_scale";
constexpr char kKeyZeroField[] = "k_zero_point";
constexpr char kValueScaleField[] = "v_scale";
constexpr char kValueZeroField[] = "v_zero_point";
constexpr char kWindowField[] = "window";
constexpr int kNumFields = 8;
constexpr int32_t kMaxRows = 16;

// q, k, v and mask, then k_new and v_new; with a window its start instead.
int32_t NumInputs(bool new_keys, int32_t window) {
  return new_keys ? 6 : window != 0 ? 5 : 4;
}

bool SupportedDepth(int64_t depth) {
  return depth == 128 || depth == 256 || depth == 512;
}

// A single-head cache of depth 512 read by 16 query rows per token (decode)
// or per prompt token (prefill) runs on the tensor-core kernels; mask row
// r % mask_rows applies to query row r.
// The tensor-core kernel takes one KV head per launch: a block of several
// heads (Gemma 4 26B-A4B's global layers have two) launches it per head.
bool UsesGlobalKernel(const nvinfer1::Dims& q, const nvinfer1::Dims& mask) {
  return q.d[1] >= 1 &&
         LiteRtNvidiaGlobalAttentionSupports(static_cast<int32_t>(q.d[2]),
                                             static_cast<int32_t>(mask.d[2]),
                                             static_cast<int32_t>(q.d[3]));
}

bool ValidDimensions(const nvinfer1::Dims& dims) {
  if (dims.nbDims != 4) return false;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0 || dims.d[i] > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  return true;
}

// The attention over a cache followed by the keys of the chunk being
// prefilled (tiled_attention.h).
LiteRtNvidiaAttentionShape NewKeysShape(const nvinfer1::Dims& q,
                                        const nvinfer1::Dims& k,
                                        const nvinfer1::Dims& mask,
                                        const nvinfer1::Dims& k_new) {
  return {static_cast<int32_t>(q.d[1]),    static_cast<int32_t>(q.d[2]),
          static_cast<int32_t>(mask.d[2]), static_cast<int32_t>(q.d[3]),
          static_cast<int32_t>(k.d[2]),    static_cast<int32_t>(k_new.d[2])};
}

// Validate the concrete kernel contract before narrowing dimensions or reading
// buffers. Format negotiation alone does not establish matching K/V shapes.
nvinfer1::DataType CacheType(int32_t int8_cache) {
  return int8_cache != 0 ? nvinfer1::DataType::kINT8
                         : nvinfer1::DataType::kHALF;
}

bool ValidDescriptors(const nvinfer1::PluginTensorDesc* inputs,
                      const nvinfer1::PluginTensorDesc* output, bool new_keys,
                      int32_t int8_cache, int32_t window) {
  if (inputs == nullptr || output == nullptr ||
      !ValidDimensions(output->dims) ||
      output->format != nvinfer1::TensorFormat::kLINEAR ||
      (window != 0 && (new_keys || int8_cache == 0))) {
    return false;
  }
  const int num_inputs = NumInputs(new_keys, window);
  for (int i = 0; i < num_inputs; ++i) {
    if (inputs[i].format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    if (window != 0 && i == 4) {
      // The window's start: an int32 [1] device tensor.
      if (inputs[i].type != nvinfer1::DataType::kINT32 ||
          inputs[i].dims.nbDims != 1 || inputs[i].dims.d[0] != 1) {
        return false;
      }
      continue;
    }
    if (!ValidDimensions(inputs[i].dims)) {
      return false;
    }
  }
  if ((inputs[0].type != nvinfer1::DataType::kHALF &&
       inputs[0].type != nvinfer1::DataType::kBF16) ||
      inputs[1].type != CacheType(int8_cache) ||
      inputs[2].type != CacheType(int8_cache) ||
      inputs[3].type != nvinfer1::DataType::kBOOL ||
      output->type != inputs[0].type) {
    return false;
  }
  const auto& q = inputs[0].dims;
  const auto& k = inputs[1].dims;
  const auto& v = inputs[2].dims;
  const auto& mask = inputs[3].dims;
  for (int i = 0; i < 4; ++i) {
    if (output->dims.d[i] != q.d[i]) {
      return false;
    }
  }
  if (int8_cache == 2) {
    // The value cache as the model stores it: [1, H, D, S].
    if (v.d[0] != k.d[0] || v.d[1] != k.d[1] || v.d[2] != k.d[3] ||
        v.d[3] != k.d[2]) {
      return false;
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      if (v.d[i] != k.d[i]) {
        return false;
      }
    }
  }
  if (q.d[0] != 1 || k.d[0] != 1 || k.d[1] != q.d[1] || k.d[3] != q.d[3] ||
      mask.d[0] != 1 || mask.d[1] != 1) {
    return false;
  }
  if (int8_cache != 0) {
    // INT8 caches run only on the decode kernel.
    return !new_keys && mask.d[3] == k.d[2] && q.d[2] <= kMaxRows &&
           (window == 0 || (window <= k.d[2] && window % 16 == 0)) &&
           LiteRtNvidiaDecodeAttentionInt8Supports(
               static_cast<int32_t>(q.d[3])) &&
           (mask.d[2] == 1 || mask.d[2] == q.d[2]);
  }
  if (new_keys) {
    const auto& k_new = inputs[4].dims;
    if (inputs[4].type != nvinfer1::DataType::kHALF ||
        inputs[5].type != nvinfer1::DataType::kHALF || k_new.d[0] != 1 ||
        k_new.d[1] != k.d[1] || k_new.d[3] != k.d[3] ||
        mask.d[3] != k.d[2] + k_new.d[2]) {
      return false;
    }
    for (int i = 0; i < 4; ++i) {
      if (inputs[5].dims.d[i] != k_new.d[i]) {
        return false;
      }
    }
    const LiteRtNvidiaAttentionShape shape = NewKeysShape(q, k, mask, k_new);
    return LiteRtNvidiaTiledAttentionSupports(&shape);
  }
  if (mask.d[3] != k.d[2]) {
    return false;
  }
  return UsesGlobalKernel(q, mask) ||
         (q.d[2] <= kMaxRows && SupportedDepth(q.d[3]) &&
          (mask.d[2] == 1 || mask.d[2] == q.d[2]));
}

class DecodeAttentionPlugin final : public nvinfer1::IPluginV3,
                                    public nvinfer1::IPluginV3OneCore,
                                    public nvinfer1::IPluginV3OneBuild,
                                    public nvinfer1::IPluginV3OneRuntime {
 public:
  DecodeAttentionPlugin(float fill, bool new_keys,
                        const LiteRtNvidiaInt8CacheParams* int8,
                        int32_t window) noexcept
      : fill_(fill),
        new_keys_(new_keys ? 1 : 0),
        int8_cache_(int8 == nullptr       ? 0
                    : int8->value_columns ? 2
                                          : 1),
        k_scale_(int8 == nullptr ? 1.0f : int8->k_scale),
        k_zero_(int8 == nullptr ? 0 : int8->k_zero_point),
        v_scale_(int8 == nullptr ? 1.0f : int8->v_scale),
        v_zero_(int8 == nullptr ? 0 : int8->v_zero_point),
        window_(window),
        fields_{
            {{kFillField, &fill_, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kNewKeysField, &new_keys_, nvinfer1::PluginFieldType::kINT32, 1},
             {kInt8CacheField, &int8_cache_, nvinfer1::PluginFieldType::kINT32,
              1},
             {kKeyScaleField, &k_scale_, nvinfer1::PluginFieldType::kFLOAT32,
              1},
             {kKeyZeroField, &k_zero_, nvinfer1::PluginFieldType::kINT32, 1},
             {kValueScaleField, &v_scale_, nvinfer1::PluginFieldType::kFLOAT32,
              1},
             {kValueZeroField, &v_zero_, nvinfer1::PluginFieldType::kINT32, 1},
             {kWindowField, &window_, nvinfer1::PluginFieldType::kINT32, 1}}},
        field_collection_{static_cast<int32_t>(fields_.size()),
                          fields_.data()} {}

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
    const LiteRtNvidiaInt8CacheParams int8 = Int8Params();
    return new (std::nothrow) DecodeAttentionPlugin(
        fill_, new_keys_ != 0, int8_cache_ != 0 ? &int8 : nullptr, window_);
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
    output_types[0] = input_types[0];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != InputCount() || num_outputs != 1 ||
        position < 0 || position > num_inputs) {
      return false;
    }
    const auto& desc = in_out[position].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    if (position == num_inputs) {
      return desc.type == in_out[0].desc.type;
    }
    switch (position) {
      case 0:
        return desc.type == nvinfer1::DataType::kHALF ||
               desc.type == nvinfer1::DataType::kBF16;
      case 1:
      case 2:
        return desc.type == CacheType(int8_cache_);
      case 3:
        return desc.type == nvinfer1::DataType::kBOOL;
      case 4:
        return desc.type == (window_ != 0 ? nvinfer1::DataType::kINT32
                                          : nvinfer1::DataType::kHALF);
      default:
        return desc.type == nvinfer1::DataType::kHALF;
    }
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expr_builder);
    if (inputs == nullptr || outputs == nullptr || num_inputs != InputCount() ||
        num_shape_inputs != 0 || num_outputs != 1 || inputs[0].nbDims != 4) {
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
                   num_inputs == InputCount() && num_outputs == 1
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return num_inputs == InputCount() && num_outputs == 1 &&
                   ValidDescriptors(inputs, outputs, new_keys_ != 0,
                                    int8_cache_, window_)
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(outputs);
    if (inputs == nullptr || num_inputs != InputCount() || num_outputs != 1 ||
        !ValidDimensions(inputs[0].max) || !ValidDimensions(inputs[1].max) ||
        !ValidDimensions(inputs[3].max)) {
      return 0;
    }
    const auto& q = inputs[0].max;
    const auto& k = inputs[1].max;
    if (new_keys_ != 0) {
      if (!ValidDimensions(inputs[4].max)) {
        return 0;
      }
      const LiteRtNvidiaAttentionShape shape =
          NewKeysShape(q, k, inputs[3].max, inputs[4].max);
      return LiteRtNvidiaTiledAttentionWorkspaceBytes(&shape);
    }
    if (int8_cache_ == 0 && UsesGlobalKernel(q, inputs[3].max)) {
      return LiteRtNvidiaGlobalAttentionWorkspaceBytes(
          static_cast<int32_t>(q.d[2]),
          static_cast<int32_t>(inputs[3].max.d[2]),
          static_cast<int32_t>(k.d[2]), static_cast<int32_t>(q.d[3]));
    }
    return LiteRtNvidiaDecodeAttentionWorkspaceBytes(
        static_cast<int32_t>(q.d[1]), static_cast<int32_t>(q.d[2]),
        window_ != 0 ? window_ : static_cast<int32_t>(k.d[2]),
        static_cast<int32_t>(q.d[3]));
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    if (!ValidDescriptors(input_desc, output_desc, new_keys_ != 0, int8_cache_,
                          window_) ||
        inputs == nullptr || outputs == nullptr || outputs[0] == nullptr ||
        workspace == nullptr) {
      return 1;
    }
    for (int i = 0; i < InputCount(); ++i) {
      if (inputs[i] == nullptr) return 1;
    }
    if (SkipPluginFamily("attention")) return 0;
    const auto& q = input_desc[0].dims;
    const auto& k = input_desc[1].dims;
    const auto& mask = input_desc[3].dims;
    const int32_t heads = static_cast<int32_t>(q.d[1]);
    const int32_t rows = static_cast<int32_t>(q.d[2]);
    const int32_t depth = static_cast<int32_t>(q.d[3]);
    const int32_t seq = static_cast<int32_t>(k.d[2]);
    const int32_t mask_rows = static_cast<int32_t>(mask.d[2]);
    const bool q_bf16 = input_desc[0].type == nvinfer1::DataType::kBF16;
    cudaError_t status = cudaSuccess;
    if (new_keys_ != 0) {
      const LiteRtNvidiaAttentionShape shape =
          NewKeysShape(q, k, mask, input_desc[4].dims);
      status = LiteRtNvidiaLaunchTiledAttention(
          &shape, inputs[0], q_bf16, inputs[1], inputs[2], inputs[4], inputs[5],
          static_cast<const bool*>(inputs[3]), fill_, outputs[0], workspace,
          stream);
    } else if (int8_cache_ != 0) {
      const LiteRtNvidiaInt8CacheParams int8 = Int8Params();
      status = LiteRtNvidiaLaunchDecodeAttentionInt8Window(
          inputs[0], q_bf16, inputs[1], inputs[2], &int8,
          static_cast<const bool*>(inputs[3]), mask_rows, heads, rows, seq,
          window_ != 0 ? window_ : seq,
          window_ != 0 ? static_cast<const int32_t*>(inputs[4]) : nullptr,
          depth, fill_, outputs[0], workspace, stream);
    } else {
      if (UsesGlobalKernel(q, mask)) {
        // One launch per KV head over that head's rows, keys and values; the
        // mask is shared and the launches run in order on the stream, so
        // they share the workspace.
        const size_t q_head = static_cast<size_t>(rows) * depth * 2;
        const size_t kv_head = static_cast<size_t>(seq) * depth * 2;
        for (int32_t head = 0; head < heads && status == cudaSuccess; ++head) {
          status = LiteRtNvidiaLaunchGlobalAttention(
              static_cast<const char*>(inputs[0]) + head * q_head, q_bf16,
              static_cast<const char*>(inputs[1]) + head * kv_head,
              static_cast<const char*>(inputs[2]) + head * kv_head,
              static_cast<const bool*>(inputs[3]), mask_rows, rows, seq, depth,
              fill_, static_cast<char*>(outputs[0]) + head * q_head, workspace,
              stream);
        }
      } else {
        status = LiteRtNvidiaLaunchDecodeAttention(
            inputs[0], q_bf16, inputs[1], inputs[2],
            static_cast<const bool*>(inputs[3]), mask_rows, heads, rows, seq,
            depth, fill_, outputs[0], workspace, stream);
      }
    }
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaDecodeAttention] CUDA launch failed: %s (%d): "
                   "%s\n",
                   cudaGetErrorName(status), static_cast<int>(status),
                   cudaGetErrorString(status));
      return 1;
    }
    return 0;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    return clone();
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return &field_collection_;
  }

 private:
  int32_t InputCount() const noexcept {
    return NumInputs(new_keys_ != 0, window_);
  }
  LiteRtNvidiaInt8CacheParams Int8Params() const noexcept {
    return {k_scale_, k_zero_, v_scale_, v_zero_, int8_cache_ == 2};
  }

  float fill_;
  int32_t new_keys_;
  int32_t int8_cache_;
  float k_scale_;
  int32_t k_zero_;
  float v_scale_;
  int32_t v_zero_;
  int32_t window_;
  std::array<nvinfer1::PluginField, kNumFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class DecodeAttentionPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  DecodeAttentionPluginCreator() noexcept
      : fields_{
            {{kFillField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kNewKeysField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kInt8CacheField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kKeyScaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kKeyZeroField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kValueScaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32,
              1},
             {kValueZeroField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kWindowField, nullptr, nvinfer1::PluginFieldType::kINT32, 1}}},
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
    float fill = 0.0f;
    int32_t new_keys = 0;
    int32_t int8_cache = 0;
    int32_t window = 0;
    LiteRtNvidiaInt8CacheParams int8 = {1.0f, 0, 1.0f, 0, false};
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length != 1) {
        continue;
      }
      if (field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        const float value = *static_cast<const float*>(field.data);
        if (std::strcmp(field.name, kFillField) == 0) {
          fill = value;
        } else if (std::strcmp(field.name, kKeyScaleField) == 0) {
          int8.k_scale = value;
        } else if (std::strcmp(field.name, kValueScaleField) == 0) {
          int8.v_scale = value;
        }
      } else if (field.type == nvinfer1::PluginFieldType::kINT32) {
        const int32_t value = *static_cast<const int32_t*>(field.data);
        if (std::strcmp(field.name, kNewKeysField) == 0) {
          new_keys = value;
        } else if (std::strcmp(field.name, kInt8CacheField) == 0) {
          int8_cache = value;
        } else if (std::strcmp(field.name, kKeyZeroField) == 0) {
          int8.k_zero_point = value;
        } else if (std::strcmp(field.name, kValueZeroField) == 0) {
          int8.v_zero_point = value;
        } else if (std::strcmp(field.name, kWindowField) == 0) {
          window = value;
        }
      }
    }
    if (int8_cache < 0 || int8_cache > 2) {
      return nullptr;
    }
    int8.value_columns = int8_cache == 2;
    return CreateDecodeAttentionPlugin(
        fill, new_keys != 0, int8_cache != 0 ? &int8 : nullptr, window);
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
  std::array<nvinfer1::PluginField, kNumFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

}  // namespace

nvinfer1::IPluginV3* CreateDecodeAttentionPlugin(
    float fill, bool new_keys, const LiteRtNvidiaInt8CacheParams* int8_cache,
    int32_t window) noexcept {
  if (int8_cache != nullptr &&
      (new_keys || !std::isfinite(int8_cache->k_scale) ||
       !std::isfinite(int8_cache->v_scale) || int8_cache->k_scale <= 0.0f ||
       int8_cache->v_scale <= 0.0f)) {
    return nullptr;
  }
  if (window < 0 || (window != 0 && (int8_cache == nullptr || window % 16))) {
    return nullptr;
  }
  return new (std::nothrow)
      DecodeAttentionPlugin(fill, new_keys, int8_cache, window);
}

void EnsureDecodeAttentionPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(DecodeAttentionPluginCreator);

}  // namespace litert::nvidia
