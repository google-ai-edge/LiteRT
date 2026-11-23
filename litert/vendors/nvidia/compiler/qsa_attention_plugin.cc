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

#include "litert/vendors/nvidia/compiler/qsa_attention_plugin.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/qsa_indexer.h"
#include "litert/vendors/nvidia/trtllm/qsa_sparse_attention.h"

namespace litert::nvidia {
namespace {

constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";

int32_t QsaType(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kHALF:
      return kLiteRtNvidiaQsaFloat16;
    case nvinfer1::DataType::kBF16:
      return kLiteRtNvidiaQsaBFloat16;
    default:
      return kLiteRtNvidiaQsaFloat32;
  }
}

bool IsActivationType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kFLOAT ||
         type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kBF16;
}

bool IsStateType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kFLOAT ||
         type == nvinfer1::DataType::kHALF;
}

bool IsMaskType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kBOOL ||
         type == nvinfer1::DataType::kFLOAT;
}

int32_t MaskType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kBOOL ? kLiteRtNvidiaQsaMaskBool
                                           : kLiteRtNvidiaQsaMaskFloat;
}

size_t ElementBytes(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
      return 1;
    default:
      return 4;
  }
}

int64_t Volume(const nvinfer1::Dims& dims) {
  int64_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) volume *= dims.d[i];
  return volume;
}

uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

int32_t Report(const char* plugin, cudaError_t status) {
  if (status == cudaSuccess) return 0;
  std::fprintf(stderr, "[%s] CUDA launch failed: %s (%d): %s\n", plugin,
               cudaGetErrorName(status), static_cast<int>(status),
               cudaGetErrorString(status));
  return 1;
}

// The int32 fields of a plugin (floats travel as their bits).
template <int kCount>
class IntFields {
 public:
  explicit IntFields(const std::array<const char*, kCount>& names) noexcept
      : names_(names) {
    for (int i = 0; i < kCount; ++i) {
      fields_[i] = {names_[i], &values_[i], nvinfer1::PluginFieldType::kINT32,
                    1};
    }
    collection_ = {kCount, fields_.data()};
  }
  IntFields(const IntFields& other) noexcept : IntFields(other.names_) {
    values_ = other.values_;
  }
  int32_t& operator[](int i) noexcept { return values_[i]; }
  int32_t operator[](int i) const noexcept { return values_[i]; }
  const nvinfer1::PluginFieldCollection* Collection() noexcept {
    return &collection_;
  }
  bool Read(const nvinfer1::PluginFieldCollection* fields) noexcept {
    if (fields == nullptr || fields->fields == nullptr) return false;
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length != 1 ||
          field.type != nvinfer1::PluginFieldType::kINT32) {
        continue;
      }
      for (int j = 0; j < kCount; ++j) {
        if (std::strcmp(field.name, names_[j]) == 0) {
          values_[j] = *static_cast<const int32_t*>(field.data);
        }
      }
    }
    return true;
  }

 private:
  std::array<const char*, kCount> names_;
  std::array<int32_t, kCount> values_{};
  std::array<nvinfer1::PluginField, kCount> fields_;
  nvinfer1::PluginFieldCollection collection_;
};

// The IPluginV3 plumbing both plugins share.
template <int kFields>
class QsaPluginBase : public nvinfer1::IPluginV3,
                      public nvinfer1::IPluginV3OneCore,
                      public nvinfer1::IPluginV3OneBuild,
                      public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit QsaPluginBase(const IntFields<kFields>& fields) noexcept
      : fields_(fields) {}

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
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }
  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == NumInputs() && num_outputs == getNbOutputs()
               ? 0
               : 1;
  }
  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == NumInputs() && num_outputs == getNbOutputs()
               ? 0
               : 1;
  }
  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext*) noexcept override {
    return clone();
  }
  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return fields_.Collection();
  }

 protected:
  virtual int32_t NumInputs() const noexcept = 0;
  IntFields<kFields> fields_;
};

// ---------------------------------------------------------------------------
// odml.qsa_indexer_update
// ---------------------------------------------------------------------------

constexpr char kIndexerName[] = "LiteRtNvidiaQsaIndexer";
constexpr std::array<const char*, 3> kIndexerFields = {
    "block_topk", "rms_norm_eps_bits", "scale_bits"};

enum IndexerInput {
  kIndexQ = 0,
  kIndexKeyNew,
  kIndexRing,
  kIndexPool,
  kIndexNorm,
  kIndexCos,
  kIndexSin,
  kIndexStartPos,
  kIndexMask,
  kIndexInputs,
};

class QsaIndexerPlugin final : public QsaPluginBase<3> {
 public:
  explicit QsaIndexerPlugin(const IntFields<3>& fields) noexcept
      : QsaPluginBase<3>(fields) {}

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) QsaIndexerPlugin(fields_);
  }
  const char* getPluginName() const noexcept override { return kIndexerName; }
  int32_t getNbOutputs() const noexcept override { return 3; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != kIndexInputs || num_outputs != 3) {
      return 1;
    }
    output_types[0] = nvinfer1::DataType::kINT32;
    output_types[1] = input_types[kIndexRing];
    output_types[2] = input_types[kIndexPool];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != kIndexInputs || num_outputs != 3 ||
        position < 0 || position >= num_inputs + num_outputs ||
        in_out[position].desc.format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    const nvinfer1::DataType type = in_out[position].desc.type;
    switch (position) {
      case kIndexQ:
        return IsActivationType(type);
      case kIndexKeyNew:
        return type == in_out[kIndexQ].desc.type;
      case kIndexRing:
        return IsStateType(type);
      case kIndexPool:
        return type == in_out[kIndexRing].desc.type;
      case kIndexNorm:
      case kIndexCos:
      case kIndexSin:
        return type == nvinfer1::DataType::kFLOAT;
      case kIndexStartPos:
        return type == nvinfer1::DataType::kINT32;
      case kIndexMask:
        return IsMaskType(type);
      case kIndexInputs:
        return type == nvinfer1::DataType::kINT32;
      default:
        return type == in_out[kIndexRing].desc.type;
    }
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    if (inputs == nullptr || outputs == nullptr || num_inputs != kIndexInputs ||
        num_shape_inputs != 0 || num_outputs != 3 ||
        inputs[kIndexQ].nbDims != 4) {
      return 1;
    }
    outputs[0].nbDims = 3;
    outputs[0].d[0] = expr_builder.constant(1);
    outputs[0].d[1] = inputs[kIndexQ].d[1];
    outputs[0].d[2] = expr_builder.constant(fields_[0]);
    outputs[1] = inputs[kIndexRing];
    outputs[2] = inputs[kIndexPool];
    return 0;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(outputs);
    if (inputs == nullptr || num_inputs != kIndexInputs || num_outputs != 3) {
      return 0;
    }
    LiteRtNvidiaQsaIndexerParams params;
    if (!Params(inputs[kIndexQ].max, inputs[kIndexPool].max,
                inputs[kIndexCos].max, inputs[kIndexMask].max,
                inputs[kIndexQ].desc.type, inputs[kIndexRing].desc.type,
                inputs[kIndexMask].desc.type, &params)) {
      return 0;
    }
    return LiteRtNvidiaQsaIndexerWorkspaceBytes(&params);
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
        outputs == nullptr) {
      return 1;
    }
    LiteRtNvidiaQsaIndexerParams params;
    if (!Params(input_desc[kIndexQ].dims, input_desc[kIndexPool].dims,
                input_desc[kIndexCos].dims, input_desc[kIndexMask].dims,
                input_desc[kIndexQ].type, input_desc[kIndexRing].type,
                input_desc[kIndexMask].type, &params)) {
      return 1;
    }
    // The kernels update the state in place: bring it into the output
    // buffers unless the runtime bound them to the inputs.
    for (int i = 0; i < 2; ++i) {
      const int input = i == 0 ? kIndexRing : kIndexPool;
      if (outputs[1 + i] != inputs[input]) {
        const size_t bytes =
            static_cast<size_t>(Volume(input_desc[input].dims)) *
            ElementBytes(input_desc[input].type);
        const cudaError_t status =
            cudaMemcpyAsync(outputs[1 + i], inputs[input], bytes,
                            cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess) return Report(kIndexerName, status);
      }
    }
    return Report(
        kIndexerName,
        LiteRtNvidiaLaunchQsaIndexerUpdate(
            &params, inputs[kIndexQ], inputs[kIndexKeyNew], outputs[1],
            outputs[2], static_cast<const float*>(inputs[kIndexNorm]),
            static_cast<const float*>(inputs[kIndexCos]),
            static_cast<const float*>(inputs[kIndexSin]),
            static_cast<const int32_t*>(inputs[kIndexStartPos]),
            inputs[kIndexMask], static_cast<int32_t*>(outputs[0]), workspace,
            stream));
  }

 protected:
  int32_t NumInputs() const noexcept override { return kIndexInputs; }

 private:
  bool Params(const nvinfer1::Dims& q, const nvinfer1::Dims& pool,
              const nvinfer1::Dims& cos, const nvinfer1::Dims& mask,
              nvinfer1::DataType q_type, nvinfer1::DataType state_type,
              nvinfer1::DataType mask_type,
              LiteRtNvidiaQsaIndexerParams* params) const noexcept {
    if (q.nbDims != 4 || pool.nbDims != 3 || cos.nbDims != 3 ||
        mask.nbDims != 4) {
      return false;
    }
    *params = {};
    params->rows = static_cast<int32_t>(q.d[1]);
    params->index_heads = static_cast<int32_t>(q.d[2]);
    params->index_dim = static_cast<int32_t>(q.d[3]);
    params->rotary_dim = static_cast<int32_t>(cos.d[2]);
    params->max_blocks = static_cast<int32_t>(pool.d[1]);
    params->block_topk = fields_[0];
    params->mask_len = static_cast<int32_t>(mask.d[3]);
    params->q_type = QsaType(q_type);
    params->state_type = QsaType(state_type);
    params->mask_type = MaskType(mask_type);
    params->rms_norm_eps = BitsFloat(static_cast<uint32_t>(fields_[1]));
    params->scale = BitsFloat(static_cast<uint32_t>(fields_[2]));
    return LiteRtNvidiaQsaIndexerSupports(params);
  }
};

class QsaIndexerPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  QsaIndexerPluginCreator() noexcept : fields_(kIndexerFields) {}
  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    IntFields<3> values(kIndexerFields);
    if (!values.Read(fields)) return nullptr;
    return new (std::nothrow) QsaIndexerPlugin(values);
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return fields_.Collection();
  }
  const char* getPluginName() const noexcept override { return kIndexerName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  IntFields<3> fields_;
};

// ---------------------------------------------------------------------------
// odml.sparse_sdpa_transposed
// ---------------------------------------------------------------------------

constexpr char kAttentionName[] = "LiteRtNvidiaQsaAttention";
constexpr std::array<const char*, 2> kAttentionFields = {"scale_bits",
                                                         "value_columns"};

enum AttentionInput {
  kAttendQ = 0,
  kAttendKey,
  kAttendValue,
  kAttendTopk,
  kAttendMask,
  kAttendInputs,
};

class QsaAttentionPlugin final : public QsaPluginBase<2> {
 public:
  explicit QsaAttentionPlugin(const IntFields<2>& fields) noexcept
      : QsaPluginBase<2>(fields) {}

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) QsaAttentionPlugin(fields_);
  }
  const char* getPluginName() const noexcept override { return kAttentionName; }
  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != kAttendInputs || num_outputs != 1) {
      return 1;
    }
    output_types[0] = input_types[kAttendQ];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != kAttendInputs || num_outputs != 1 ||
        position < 0 || position > num_inputs ||
        in_out[position].desc.format != nvinfer1::TensorFormat::kLINEAR) {
      return false;
    }
    const nvinfer1::DataType type = in_out[position].desc.type;
    switch (position) {
      case kAttendQ:
        return IsActivationType(type);
      case kAttendKey:
        return IsStateType(type);
      case kAttendValue:
        return type == in_out[kAttendKey].desc.type;
      case kAttendTopk:
        return type == nvinfer1::DataType::kINT32;
      case kAttendMask:
        return IsMaskType(type);
      default:
        return type == in_out[kAttendQ].desc.type;
    }
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expr_builder);
    if (inputs == nullptr || outputs == nullptr ||
        num_inputs != kAttendInputs || num_shape_inputs != 0 ||
        num_outputs != 1 || inputs[kAttendQ].nbDims != 4) {
      return 1;
    }
    outputs[0].nbDims = 4;
    outputs[0].d[0] = inputs[kAttendQ].d[0];
    outputs[0].d[1] = inputs[kAttendQ].d[2];
    outputs[0].d[2] = inputs[kAttendQ].d[1];
    outputs[0].d[3] = inputs[kAttendQ].d[3];
    return 0;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(outputs);
    if (inputs == nullptr || num_inputs != kAttendInputs || num_outputs != 1) {
      return 0;
    }
    LiteRtNvidiaQsaAttentionParams params;
    if (!Params(inputs[kAttendQ].max, inputs[kAttendKey].max,
                inputs[kAttendTopk].max, inputs[kAttendQ].desc.type,
                inputs[kAttendKey].desc.type, inputs[kAttendMask].desc.type,
                &params)) {
      return 0;
    }
    return LiteRtNvidiaQsaAttentionWorkspaceBytes(&params);
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    LiteRtNvidiaQsaAttentionParams params;
    if (!Params(input_desc[kAttendQ].dims, input_desc[kAttendKey].dims,
                input_desc[kAttendTopk].dims, input_desc[kAttendQ].type,
                input_desc[kAttendKey].type, input_desc[kAttendMask].type,
                &params)) {
      return 1;
    }
    return Report(
        kAttentionName,
        LiteRtNvidiaLaunchQsaAttention(
            &params, inputs[kAttendQ], inputs[kAttendKey], inputs[kAttendValue],
            static_cast<const int32_t*>(inputs[kAttendTopk]),
            inputs[kAttendMask], nullptr, outputs[0], workspace, stream));
  }

 protected:
  int32_t NumInputs() const noexcept override { return kAttendInputs; }

 private:
  bool Params(const nvinfer1::Dims& q, const nvinfer1::Dims& k,
              const nvinfer1::Dims& topk, nvinfer1::DataType q_type,
              nvinfer1::DataType cache_type, nvinfer1::DataType mask_type,
              LiteRtNvidiaQsaAttentionParams* params) const noexcept {
    if (q.nbDims != 4 || k.nbDims != 4 || topk.nbDims != 3) return false;
    *params = {};
    params->rows = static_cast<int32_t>(q.d[2]);
    params->q_heads = static_cast<int32_t>(q.d[1]);
    params->kv_heads = static_cast<int32_t>(k.d[1]);
    params->head_dim = static_cast<int32_t>(q.d[3]);
    params->cache_len = static_cast<int32_t>(k.d[2]);
    params->block_topk = static_cast<int32_t>(topk.d[2]);
    params->q_type = QsaType(q_type);
    params->cache_type = QsaType(cache_type);
    params->value_columns = fields_[1];
    params->mask_type = MaskType(mask_type);
    params->scale = BitsFloat(static_cast<uint32_t>(fields_[0]));
    return LiteRtNvidiaQsaAttentionSupports(params);
  }
};

class QsaAttentionPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  QsaAttentionPluginCreator() noexcept : fields_(kAttentionFields) {}
  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    IntFields<2> values(kAttentionFields);
    if (!values.Read(fields)) return nullptr;
    return new (std::nothrow) QsaAttentionPlugin(values);
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return fields_.Collection();
  }
  const char* getPluginName() const noexcept override { return kAttentionName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  IntFields<2> fields_;
};

}  // namespace

nvinfer1::IPluginV3* CreateQsaIndexerPlugin(int32_t block_topk,
                                            float rms_norm_eps,
                                            float scale) noexcept {
  if (block_topk < 1) return nullptr;
  IntFields<3> fields(kIndexerFields);
  fields[0] = block_topk;
  fields[1] = static_cast<int32_t>(FloatBits(rms_norm_eps));
  fields[2] = static_cast<int32_t>(FloatBits(scale));
  return new (std::nothrow) QsaIndexerPlugin(fields);
}

nvinfer1::IPluginV3* CreateQsaAttentionPlugin(float scale,
                                              bool value_columns) noexcept {
  IntFields<2> fields(kAttentionFields);
  fields[0] = static_cast<int32_t>(FloatBits(scale));
  fields[1] = value_columns ? 1 : 0;
  return new (std::nothrow) QsaAttentionPlugin(fields);
}

void EnsureQsaPluginsRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(QsaIndexerPluginCreator);
REGISTER_TENSORRT_PLUGIN(QsaAttentionPluginCreator);

}  // namespace litert::nvidia
