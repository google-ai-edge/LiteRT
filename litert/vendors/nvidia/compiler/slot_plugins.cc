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

#include "litert/vendors/nvidia/compiler/slot_plugins.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/decode_attention_slots.h"
#include "litert/vendors/nvidia/trtllm/global_attention.h"

namespace litert::nvidia {
namespace {

// The single-sequence plugin and its fields (decode_attention_plugin.cc).
constexpr char kDecodeAttentionName[] = "LiteRtNvidiaDecodeAttention";
constexpr char kFillField[] = "fill";
constexpr char kNewKeysField[] = "new_keys";
constexpr char kInt8CacheField[] = "int8_cache";
constexpr char kKeyScaleField[] = "k_scale";
constexpr char kKeyZeroField[] = "k_zero_point";
constexpr char kValueScaleField[] = "v_scale";
constexpr char kValueZeroField[] = "v_zero_point";
constexpr char kWindowField[] = "window";
constexpr char kRowTokensField[] = "row_tokens";

constexpr char kTwinName[] = "LiteRtNvidiaDecodeAttentionSlots";
constexpr char kTwinVersion[] = "1";
constexpr char kTwinNamespace[] = "";
constexpr char kSlotsField[] = "slots";
constexpr int kTwinFields = 8;
constexpr int32_t kMaxRows = 16;

// What a decode attention layer's plugin was made with.
struct AttentionFields {
  float fill = 0.0f;
  int32_t new_keys = 0;
  int32_t int8_cache = 0;  // 0 FP16 caches, 1 INT8 value rows, 2 columns
  float k_scale = 1.0f;
  int32_t k_zero = 0;
  float v_scale = 1.0f;
  int32_t v_zero = 0;
  int32_t window = 0;
  int32_t row_tokens = 0;
  int32_t slots = 0;
};

bool ReadFields(const nvinfer1::PluginFieldCollection* fields,
                AttentionFields* out) {
  if (fields == nullptr || fields->fields == nullptr) return false;
  for (int32_t i = 0; i < fields->nbFields; ++i) {
    const auto& field = fields->fields[i];
    if (field.name == nullptr || field.data == nullptr || field.length != 1) {
      continue;
    }
    if (field.type == nvinfer1::PluginFieldType::kFLOAT32) {
      const float value = *static_cast<const float*>(field.data);
      if (std::strcmp(field.name, kFillField) == 0) out->fill = value;
      if (std::strcmp(field.name, kKeyScaleField) == 0) out->k_scale = value;
      if (std::strcmp(field.name, kValueScaleField) == 0) out->v_scale = value;
    } else if (field.type == nvinfer1::PluginFieldType::kINT32) {
      const int32_t value = *static_cast<const int32_t*>(field.data);
      if (std::strcmp(field.name, kNewKeysField) == 0) out->new_keys = value;
      if (std::strcmp(field.name, kInt8CacheField) == 0)
        out->int8_cache = value;
      if (std::strcmp(field.name, kKeyZeroField) == 0) out->k_zero = value;
      if (std::strcmp(field.name, kValueZeroField) == 0) out->v_zero = value;
      if (std::strcmp(field.name, kWindowField) == 0) out->window = value;
      if (std::strcmp(field.name, kRowTokensField) == 0)
        out->row_tokens = value;
      if (std::strcmp(field.name, kSlotsField) == 0) out->slots = value;
    }
  }
  return true;
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

// Whether one slot's tensors (q [1, H, rows, D], k [1, H, S, D], v as k or
// [1, H, D, S], mask [1, 1, mask_rows, S]) are a shape the slots launch
// takes: what the single-sequence plugin runs on
// LiteRtNvidiaLaunchDecodeAttention or its INT8 window form.
bool ValidSlotDims(const AttentionFields& f, const nvinfer1::Dims& q,
                   const nvinfer1::Dims& k, const nvinfer1::Dims& v,
                   const nvinfer1::Dims& mask) {
  if (!ValidDimensions(q) || !ValidDimensions(k) || !ValidDimensions(v) ||
      !ValidDimensions(mask) || q.d[0] != 1 || k.d[0] != 1 ||
      k.d[1] != q.d[1] || k.d[3] != q.d[3] || mask.d[0] != 1 ||
      mask.d[1] != 1 || mask.d[3] != k.d[2] || q.d[2] > kMaxRows ||
      (mask.d[2] != 1 && mask.d[2] != q.d[2]) ||
      (q.d[3] != 128 && q.d[3] != 256 && q.d[3] != 512)) {
    return false;
  }
  const bool columns = f.int8_cache == 2;
  if (v.d[0] != 1 || v.d[1] != k.d[1] ||
      v.d[2] != (columns ? k.d[3] : k.d[2]) ||
      v.d[3] != (columns ? k.d[2] : k.d[3])) {
    return false;
  }
  if (f.int8_cache == 0) {
    // 16 rows over a cache of depth 512 are the tensor-core kernel's.
    return f.window == 0 &&
           !LiteRtNvidiaGlobalAttentionSupports(static_cast<int32_t>(q.d[2]),
                                                static_cast<int32_t>(mask.d[2]),
                                                static_cast<int32_t>(q.d[3]));
  }
  return LiteRtNvidiaDecodeAttentionInt8Supports(
             static_cast<int32_t>(q.d[3])) &&
         (f.window == 0 || (f.window <= k.d[2] && f.window % 16 == 0));
}

class DecodeAttentionSlotsPlugin final : public nvinfer1::IPluginV3,
                                         public nvinfer1::IPluginV3OneCore,
                                         public nvinfer1::IPluginV3OneBuild,
                                         public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit DecodeAttentionSlotsPlugin(const AttentionFields& fields) noexcept
      : f_(fields),
        fields_{
            {{kSlotsField, &f_.slots, nvinfer1::PluginFieldType::kINT32, 1},
             {kFillField, &f_.fill, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kInt8CacheField, &f_.int8_cache,
              nvinfer1::PluginFieldType::kINT32, 1},
             {kKeyScaleField, &f_.k_scale, nvinfer1::PluginFieldType::kFLOAT32,
              1},
             {kKeyZeroField, &f_.k_zero, nvinfer1::PluginFieldType::kINT32, 1},
             {kValueScaleField, &f_.v_scale,
              nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kValueZeroField, &f_.v_zero, nvinfer1::PluginFieldType::kINT32,
              1},
             {kWindowField, &f_.window, nvinfer1::PluginFieldType::kINT32, 1}}},
        field_collection_{kTwinFields, fields_.data()} {}

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
    return new (std::nothrow) DecodeAttentionSlotsPlugin(f_);
  }

  const char* getPluginName() const noexcept override { return kTwinName; }
  const char* getPluginVersion() const noexcept override {
    return kTwinVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kTwinNamespace;
  }

  int32_t getNbOutputs() const noexcept override { return f_.slots; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != InputCount() || num_outputs != f_.slots) {
      return 1;
    }
    for (int32_t s = 0; s < f_.slots; ++s) {
      output_types[s] = input_types[s * SlotInputs()];
    }
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != InputCount() ||
        num_outputs != f_.slots || position < 0 ||
        position >= num_inputs + num_outputs) {
      return false;
    }
    const auto& desc = in_out[position].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;
    // Every slot's queries and outputs in the type of slot 0's queries.
    const nvinfer1::DataType q_type = in_out[0].desc.type;
    const bool q_ok = q_type == nvinfer1::DataType::kHALF ||
                      q_type == nvinfer1::DataType::kBF16;
    if (position >= num_inputs) return q_ok && desc.type == q_type;
    switch (position % SlotInputs()) {
      case 0:
        return q_ok && desc.type == q_type;
      case 1:
      case 2:
        return desc.type == (f_.int8_cache != 0 ? nvinfer1::DataType::kINT8
                                                : nvinfer1::DataType::kHALF);
      case 3:
        return desc.type == nvinfer1::DataType::kBOOL;
      default:
        return desc.type == nvinfer1::DataType::kINT32;
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
        num_shape_inputs != 0 || num_outputs != f_.slots) {
      return 1;
    }
    for (int32_t s = 0; s < f_.slots; ++s) {
      outputs[s] = inputs[s * SlotInputs()];
    }
    return 0;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == InputCount() && num_outputs == f_.slots
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return num_inputs == InputCount() && num_outputs == f_.slots &&
                   ValidDescriptors(inputs, outputs)
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(outputs);
    if (inputs == nullptr || num_inputs != InputCount() ||
        num_outputs != f_.slots || !ValidDimensions(inputs[0].max) ||
        !ValidDimensions(inputs[1].max)) {
      return 0;
    }
    const auto& q = inputs[0].max;
    const auto& k = inputs[1].max;
    return LiteRtNvidiaDecodeAttentionSlotsWorkspaceBytes(
        f_.slots, static_cast<int32_t>(q.d[1]), static_cast<int32_t>(q.d[2]),
        f_.window != 0 ? f_.window : static_cast<int32_t>(k.d[2]),
        static_cast<int32_t>(q.d[3]));
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    if (!ValidDescriptors(input_desc, output_desc) || inputs == nullptr ||
        outputs == nullptr || workspace == nullptr) {
      return 1;
    }
    const int32_t n = SlotInputs();
    LiteRtNvidiaDecodeAttentionSlot table[kLiteRtNvidiaDecodeAttentionMaxSlots];
    for (int32_t s = 0; s < f_.slots; ++s) {
      const void* const* in = inputs + s * n;
      for (int32_t i = 0; i < n; ++i) {
        if (in[i] == nullptr) return 1;
      }
      if (outputs[s] == nullptr) return 1;
      table[s] = {in[0],
                  in[1],
                  in[2],
                  static_cast<const bool*>(in[3]),
                  f_.window != 0 ? static_cast<const int32_t*>(in[4]) : nullptr,
                  outputs[s]};
    }
    const auto& q = input_desc[0].dims;
    const auto& k = input_desc[1].dims;
    const int32_t cache_len = static_cast<int32_t>(k.d[2]);
    const LiteRtNvidiaInt8CacheParams int8 = {f_.k_scale, f_.k_zero, f_.v_scale,
                                              f_.v_zero, f_.int8_cache == 2};
    const cudaError_t status = LiteRtNvidiaLaunchDecodeAttentionSlots(
        table, f_.slots, input_desc[0].type == nvinfer1::DataType::kBF16,
        f_.int8_cache != 0 ? &int8 : nullptr,
        static_cast<int32_t>(input_desc[3].dims.d[2]),
        static_cast<int32_t>(q.d[1]), static_cast<int32_t>(q.d[2]), cache_len,
        f_.window != 0 ? f_.window : cache_len, static_cast<int32_t>(q.d[3]),
        f_.fill, workspace, stream);
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaDecodeAttentionSlots] CUDA launch failed: %s "
                   "(%d): %s\n",
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
  // q, k, v and mask, and with a window its start.
  int32_t SlotInputs() const noexcept { return f_.window != 0 ? 5 : 4; }
  int32_t InputCount() const noexcept { return f_.slots * SlotInputs(); }

  // Every slot's tensors valid and of slot 0's shapes and types.
  bool ValidDescriptors(const nvinfer1::PluginTensorDesc* inputs,
                        const nvinfer1::PluginTensorDesc* outputs) const {
    if (inputs == nullptr || outputs == nullptr) return false;
    const int32_t n = SlotInputs();
    for (int32_t s = 0; s < f_.slots; ++s) {
      const nvinfer1::PluginTensorDesc* in = inputs + s * n;
      if (!ValidSlotDims(f_, in[0].dims, in[1].dims, in[2].dims, in[3].dims) ||
          (f_.window != 0 &&
           (in[4].dims.nbDims != 1 || in[4].dims.d[0] != 1))) {
        return false;
      }
      for (int32_t i = 0; i < 4; ++i) {
        if (in[i].type != inputs[i].type ||
            std::memcmp(&in[i].dims, &inputs[i].dims, sizeof(in[i].dims)) !=
                0) {
          return false;
        }
      }
      if (outputs[s].type != inputs[0].type ||
          std::memcmp(&outputs[s].dims, &inputs[0].dims,
                      sizeof(outputs[s].dims)) != 0) {
        return false;
      }
    }
    return true;
  }

  AttentionFields f_;
  std::array<nvinfer1::PluginField, kTwinFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

bool ValidTwinFields(const AttentionFields& f) {
  return f.slots >= 1 && f.slots <= kLiteRtNvidiaDecodeAttentionMaxSlots &&
         f.new_keys == 0 && f.row_tokens == 0 && f.int8_cache >= 0 &&
         f.int8_cache <= 2 && f.window >= 0 &&
         (f.window == 0 || (f.int8_cache != 0 && f.window % 16 == 0)) &&
         std::isfinite(f.k_scale) && std::isfinite(f.v_scale);
}

class DecodeAttentionSlotsCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  DecodeAttentionSlotsCreator()
      : fields_{
            {{kSlotsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kFillField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kInt8CacheField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kKeyScaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kKeyZeroField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kValueScaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32,
              1},
             {kValueZeroField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kWindowField, nullptr, nvinfer1::PluginFieldType::kINT32, 1}}},
        field_collection_{kTwinFields, fields_.data()} {}

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    AttentionFields f;
    if (!ReadFields(fields, &f) || !ValidTwinFields(f)) return nullptr;
    return new (std::nothrow) DecodeAttentionSlotsPlugin(f);
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &field_collection_;
  }
  const char* getPluginName() const noexcept override { return kTwinName; }
  const char* getPluginVersion() const noexcept override {
    return kTwinVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kTwinNamespace;
  }

 private:
  std::array<nvinfer1::PluginField, kTwinFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

// The twin of a decode attention layer, or null.
nvinfer1::IPluginV3* DecodeAttentionTwin(nvinfer1::IPluginV3Layer& layer,
                                         nvinfer1::IPluginV3& plugin,
                                         int32_t slots) {
  auto* runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
  AttentionFields f;
  if (runtime == nullptr || !ReadFields(runtime->getFieldsToSerialize(), &f)) {
    return nullptr;
  }
  f.slots = slots;
  if (!ValidTwinFields(f) || layer.getNbInputs() != (f.window != 0 ? 5 : 4) ||
      layer.getNbOutputs() != 1) {
    return nullptr;
  }
  for (int32_t i = 0; i < 4; ++i) {
    if (layer.getInput(i) == nullptr) return nullptr;
  }
  if (!ValidSlotDims(f, layer.getInput(0)->getDimensions(),
                     layer.getInput(1)->getDimensions(),
                     layer.getInput(2)->getDimensions(),
                     layer.getInput(3)->getDimensions())) {
    return nullptr;
  }
  return new (std::nothrow) DecodeAttentionSlotsPlugin(f);
}

}  // namespace

nvinfer1::IPluginV3* CreateSlotsTwin(nvinfer1::IPluginV3Layer& layer,
                                     int32_t slots) noexcept {
  if (slots < 2) return nullptr;
  nvinfer1::IPluginV3& plugin = layer.getPlugin();
  auto* core = static_cast<nvinfer1::IPluginV3OneCore*>(
      plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kCORE));
  if (core == nullptr || core->getPluginName() == nullptr) return nullptr;
  if (std::strcmp(core->getPluginName(), kDecodeAttentionName) == 0) {
    return DecodeAttentionTwin(layer, plugin, slots);
  }
  return nullptr;
}

void EnsureSlotPluginsRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(DecodeAttentionSlotsCreator);

}  // namespace litert::nvidia
