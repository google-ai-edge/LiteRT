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
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

constexpr char kPluginName[] = "LiteRtNvidiaDecodeAttention";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kFillField[] = "fill";
constexpr char kNewKeysField[] = "new_keys";
constexpr int32_t kMaxRows = 16;

// q, k, v and mask, then k_new and v_new.
int32_t NumInputs(bool new_keys) { return new_keys ? 6 : 4; }

bool SupportedDepth(int64_t depth) {
  return depth == 128 || depth == 256 || depth == 512;
}

// A single-head cache of depth 512 read by 16 query rows per token (decode)
// or per prompt token (prefill) runs on the tensor-core kernels; mask row
// r % mask_rows applies to query row r.
bool UsesGlobalKernel(const nvinfer1::Dims& q, const nvinfer1::Dims& mask) {
  return q.d[1] == 1 &&
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
bool ValidDescriptors(const nvinfer1::PluginTensorDesc* inputs,
                      const nvinfer1::PluginTensorDesc* output, bool new_keys) {
  if (inputs == nullptr || output == nullptr ||
      !ValidDimensions(output->dims) ||
      output->format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  for (int i = 0; i < NumInputs(new_keys); ++i) {
    if (!ValidDimensions(inputs[i].dims) ||
        inputs[i].format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
  }
  if ((inputs[0].type != nvinfer1::DataType::kHALF &&
       inputs[0].type != nvinfer1::DataType::kBF16) ||
      inputs[1].type != nvinfer1::DataType::kHALF ||
      inputs[2].type != nvinfer1::DataType::kHALF ||
      inputs[3].type != nvinfer1::DataType::kBOOL ||
      output->type != inputs[0].type) {
    return false;
  }
  const auto& q = inputs[0].dims;
  const auto& k = inputs[1].dims;
  const auto& mask = inputs[3].dims;
  for (int i = 0; i < 4; ++i) {
    if (inputs[2].dims.d[i] != k.d[i] || output->dims.d[i] != q.d[i]) {
      return false;
    }
  }
  if (q.d[0] != 1 || k.d[0] != 1 || k.d[1] != q.d[1] || k.d[3] != q.d[3] ||
      mask.d[0] != 1 || mask.d[1] != 1) {
    return false;
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
  DecodeAttentionPlugin(float fill, bool new_keys) noexcept
      : fill_(fill),
        new_keys_(new_keys ? 1 : 0),
        fields_{{{kFillField, &fill_, nvinfer1::PluginFieldType::kFLOAT32, 1},
                 {kNewKeysField, &new_keys_, nvinfer1::PluginFieldType::kINT32,
                  1}}},
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
    return new (std::nothrow) DecodeAttentionPlugin(fill_, new_keys_ != 0);
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
      case 3:
        return desc.type == nvinfer1::DataType::kBOOL;
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
                   ValidDescriptors(inputs, outputs, new_keys_ != 0)
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
    if (UsesGlobalKernel(q, inputs[3].max)) {
      return LiteRtNvidiaGlobalAttentionWorkspaceBytes(
          static_cast<int32_t>(q.d[2]),
          static_cast<int32_t>(inputs[3].max.d[2]),
          static_cast<int32_t>(k.d[2]), static_cast<int32_t>(q.d[3]));
    }
    return LiteRtNvidiaDecodeAttentionWorkspaceBytes(
        static_cast<int32_t>(q.d[1]), static_cast<int32_t>(q.d[2]),
        static_cast<int32_t>(k.d[2]), static_cast<int32_t>(q.d[3]));
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    if (!ValidDescriptors(input_desc, output_desc, new_keys_ != 0) ||
        inputs == nullptr || outputs == nullptr || outputs[0] == nullptr ||
        workspace == nullptr) {
      return 1;
    }
    for (int i = 0; i < InputCount(); ++i) {
      if (inputs[i] == nullptr) return 1;
    }
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
    } else {
      status =
          UsesGlobalKernel(q, mask)
              ? LiteRtNvidiaLaunchGlobalAttention(
                    inputs[0], q_bf16, inputs[1], inputs[2],
                    static_cast<const bool*>(inputs[3]), mask_rows, rows, seq,
                    depth, fill_, outputs[0], workspace, stream)
              : LiteRtNvidiaLaunchDecodeAttention(
                    inputs[0], q_bf16, inputs[1], inputs[2],
                    static_cast<const bool*>(inputs[3]), mask_rows, heads, rows,
                    seq, depth, fill_, outputs[0], workspace, stream);
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
  int32_t InputCount() const noexcept { return NumInputs(new_keys_ != 0); }

  float fill_;
  int32_t new_keys_;
  std::array<nvinfer1::PluginField, 2> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class DecodeAttentionPluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  DecodeAttentionPluginCreator() noexcept
      : fields_{
            {{kFillField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kNewKeysField, nullptr, nvinfer1::PluginFieldType::kINT32, 1}}},
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
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length != 1) {
        continue;
      }
      if (field.type == nvinfer1::PluginFieldType::kFLOAT32 &&
          std::strcmp(field.name, kFillField) == 0) {
        fill = *static_cast<const float*>(field.data);
      } else if (field.type == nvinfer1::PluginFieldType::kINT32 &&
                 std::strcmp(field.name, kNewKeysField) == 0) {
        new_keys = *static_cast<const int32_t*>(field.data);
      }
    }
    return CreateDecodeAttentionPlugin(fill, new_keys != 0);
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
  std::array<nvinfer1::PluginField, 2> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

}  // namespace

nvinfer1::IPluginV3* CreateDecodeAttentionPlugin(float fill,
                                                 bool new_keys) noexcept {
  return new (std::nothrow) DecodeAttentionPlugin(fill, new_keys);
}

void EnsureDecodeAttentionPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(DecodeAttentionPluginCreator);

}  // namespace litert::nvidia
