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
#include <iterator>
#include <limits>
#include <new>

#include "cuda_runtime_api.h"
#include "litert/vendors/nvidia/compiler/moe_plugin.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/cache_write_slots.h"
#include "litert/vendors/nvidia/trtllm/decode_attention.h"
#include "litert/vendors/nvidia/trtllm/decode_attention_slots.h"
#include "litert/vendors/nvidia/trtllm/global_attention.h"
#include "litert/vendors/nvidia/trtllm/qkv_norm_rope.h"
#include "litert/vendors/nvidia/trtllm/qkv_norm_rope_slots.h"

namespace litert::nvidia {
namespace {

constexpr char kVersion[] = "1";
constexpr char kNamespace[] = "";
constexpr int32_t kMaxSlots = 8;

// The single-sequence plugins that have a twin.
constexpr char kDecodeAttentionName[] = "LiteRtNvidiaDecodeAttention";
constexpr char kQkvNormRopeName[] = "LiteRtNvidiaQkvNormRope";
constexpr char kInt8CacheWriteName[] = "LiteRtNvidiaInt8CacheWrite";

// A serialized value of a plugin: an int32 or a float member of the
// plugin's values, under the name the single-sequence plugin gives it where
// it has it too.
struct FieldSpec {
  const char* name;
  bool is_float;
  size_t offset;
};

nvinfer1::PluginFieldType TypeOf(const FieldSpec& spec) {
  return spec.is_float ? nvinfer1::PluginFieldType::kFLOAT32
                       : nvinfer1::PluginFieldType::kINT32;
}

// The values of `specs` that `fields` holds, into `values`; false without
// fields.
template <typename Values, size_t N>
bool ReadValues(const nvinfer1::PluginFieldCollection* fields,
                const FieldSpec (&specs)[N], Values* values) {
  if (fields == nullptr || fields->fields == nullptr) return false;
  for (int32_t i = 0; i < fields->nbFields; ++i) {
    const auto& field = fields->fields[i];
    if (field.name == nullptr || field.data == nullptr || field.length != 1) {
      continue;
    }
    for (const FieldSpec& spec : specs) {
      if (field.type == TypeOf(spec) &&
          std::strcmp(field.name, spec.name) == 0) {
        std::memcpy(reinterpret_cast<char*>(values) + spec.offset, field.data,
                    sizeof(int32_t));
      }
    }
  }
  return true;
}

const char* NameOf(nvinfer1::IPluginV3& plugin) {
  auto* core = static_cast<nvinfer1::IPluginV3OneCore*>(
      plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kCORE));
  const char* name = core != nullptr ? core->getPluginName() : nullptr;
  return name != nullptr ? name : "";
}

const nvinfer1::PluginFieldCollection* FieldsOf(nvinfer1::IPluginV3& plugin) {
  auto* runtime = static_cast<nvinfer1::IPluginV3OneRuntime*>(
      plugin.getCapabilityInterface(nvinfer1::PluginCapabilityType::kRUNTIME));
  return runtime != nullptr ? runtime->getFieldsToSerialize() : nullptr;
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

bool SameDims(const nvinfer1::Dims& a, const nvinfer1::Dims& b) {
  if (a.nbDims != b.nbDims) return false;
  for (int i = 0; i < a.nbDims; ++i) {
    if (a.d[i] != b.d[i]) return false;
  }
  return true;
}

bool FloatType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kHALF ||
         type == nvinfer1::DataType::kBF16 ||
         type == nvinfer1::DataType::kFLOAT;
}

void ReportLaunch(const char* plugin, cudaError_t status) {
  std::fprintf(stderr, "[%s] CUDA launch failed: %s (%d): %s\n", plugin,
               cudaGetErrorName(status), static_cast<int>(status),
               cudaGetErrorString(status));
}

// What the plugins of this file share. Traits: the plugin's Values (int32
// and float members), its name kName, its serialized values kFields, and
// Make, which creates the plugin of valid values.
template <typename Traits>
class SlotPlugin : public nvinfer1::IPluginV3,
                   public nvinfer1::IPluginV3OneCore,
                   public nvinfer1::IPluginV3OneBuild,
                   public nvinfer1::IPluginV3OneRuntime {
 public:
  using Values = typename Traits::Values;

  explicit SlotPlugin(const Values& values) noexcept : v_(values) {
    for (size_t i = 0; i < fields_.size(); ++i) {
      const FieldSpec& spec = Traits::kFields[i];
      fields_[i] = {spec.name, reinterpret_cast<const char*>(&v_) + spec.offset,
                    TypeOf(spec), 1};
    }
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

  nvinfer1::IPluginV3* clone() noexcept override { return Traits::Make(v_); }

  const char* getPluginName() const noexcept override { return Traits::kName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const char* getPluginNamespace() const noexcept override {
    return kNamespace;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == InputCount() && num_outputs == getNbOutputs()
               ? 0
               : 1;
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

 protected:
  virtual int32_t InputCount() const noexcept = 0;

  Values v_;

 private:
  std::array<nvinfer1::PluginField, std::size(Traits::kFields)> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

template <typename Traits>
class SlotCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  SlotCreator() noexcept {
    for (size_t i = 0; i < fields_.size(); ++i) {
      fields_[i] = {Traits::kFields[i].name, nullptr,
                    TypeOf(Traits::kFields[i]), 1};
    }
    field_collection_ = {static_cast<int32_t>(fields_.size()), fields_.data()};
  }

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    typename Traits::Values values;
    return ReadValues(fields, Traits::kFields, &values) ? Traits::Make(values)
                                                        : nullptr;
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &field_collection_;
  }
  const char* getPluginName() const noexcept override { return Traits::kName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const char* getPluginNamespace() const noexcept override {
    return kNamespace;
  }

 private:
  std::array<nvinfer1::PluginField, std::size(Traits::kFields)> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

// --- The decode attention ---------------------------------------------------

struct AttentionTraits {
  // What a decode attention layer's plugin was made with.
  struct Values {
    int32_t slots = 0;
    float fill = 0.0f;
    int32_t int8_cache = 0;  // 0 FP16 caches, 1 INT8 value rows, 2 columns
    float k_scale = 1.0f;
    int32_t k_zero = 0;
    float v_scale = 1.0f;
    int32_t v_zero = 0;
    int32_t window = 0;
  };
  static constexpr char kName[] = "LiteRtNvidiaDecodeAttentionSlots";
  static constexpr FieldSpec kFields[] = {
      {"slots", false, offsetof(Values, slots)},
      {"fill", true, offsetof(Values, fill)},
      {"int8_cache", false, offsetof(Values, int8_cache)},
      {"k_scale", true, offsetof(Values, k_scale)},
      {"k_zero_point", false, offsetof(Values, k_zero)},
      {"v_scale", true, offsetof(Values, v_scale)},
      {"v_zero_point", false, offsetof(Values, v_zero)},
      {"window", false, offsetof(Values, window)},
  };
  static bool Valid(const Values& v) {
    return v.slots >= 1 && v.slots <= kLiteRtNvidiaDecodeAttentionMaxSlots &&
           v.int8_cache >= 0 && v.int8_cache <= 2 && v.window >= 0 &&
           (v.window == 0 || (v.int8_cache != 0 && v.window % 16 == 0)) &&
           std::isfinite(v.k_scale) && std::isfinite(v.v_scale);
  }
  static nvinfer1::IPluginV3* Make(const Values& values) noexcept;
};

// The forms of the single-sequence plugin that have no twin.
struct AttentionForms {
  int32_t new_keys = 0;
  int32_t row_tokens = 0;
};
constexpr FieldSpec kAttentionForms[] = {
    {"new_keys", false, offsetof(AttentionForms, new_keys)},
    {"row_tokens", false, offsetof(AttentionForms, row_tokens)},
};

// Whether one slot's tensors (q [1, H, rows, D], k [1, H, S, D], v as k or
// [1, H, D, S], mask [1, 1, mask_rows, S]) are a shape the slots launch
// takes: what the single-sequence plugin runs on
// LiteRtNvidiaLaunchDecodeAttention or its INT8 window form.
bool ValidAttentionDims(const AttentionTraits::Values& f,
                        const nvinfer1::Dims& q, const nvinfer1::Dims& k,
                        const nvinfer1::Dims& v, const nvinfer1::Dims& mask) {
  constexpr int32_t kMaxRows = 16;
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

class DecodeAttentionSlotsPlugin final : public SlotPlugin<AttentionTraits> {
 public:
  using SlotPlugin::SlotPlugin;

  int32_t getNbOutputs() const noexcept override { return v_.slots; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != InputCount() || num_outputs != v_.slots) {
      return 1;
    }
    for (int32_t s = 0; s < v_.slots; ++s) {
      output_types[s] = input_types[s * SlotInputs()];
    }
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != InputCount() ||
        num_outputs != v_.slots || position < 0 ||
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
        return desc.type == (v_.int8_cache != 0 ? nvinfer1::DataType::kINT8
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
        num_shape_inputs != 0 || num_outputs != v_.slots) {
      return 1;
    }
    for (int32_t s = 0; s < v_.slots; ++s) {
      outputs[s] = inputs[s * SlotInputs()];
    }
    return 0;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return num_inputs == InputCount() && num_outputs == v_.slots &&
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
        num_outputs != v_.slots || !ValidDimensions(inputs[0].max) ||
        !ValidDimensions(inputs[1].max)) {
      return 0;
    }
    const auto& q = inputs[0].max;
    const auto& k = inputs[1].max;
    return LiteRtNvidiaDecodeAttentionSlotsWorkspaceBytes(
        v_.slots, static_cast<int32_t>(q.d[1]), static_cast<int32_t>(q.d[2]),
        v_.window != 0 ? v_.window : static_cast<int32_t>(k.d[2]),
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
    for (int32_t s = 0; s < v_.slots; ++s) {
      const void* const* in = inputs + s * n;
      for (int32_t i = 0; i < n; ++i) {
        if (in[i] == nullptr) return 1;
      }
      if (outputs[s] == nullptr) return 1;
      table[s] = {in[0],
                  in[1],
                  in[2],
                  static_cast<const bool*>(in[3]),
                  v_.window != 0 ? static_cast<const int32_t*>(in[4]) : nullptr,
                  outputs[s]};
    }
    const auto& q = input_desc[0].dims;
    const auto& k = input_desc[1].dims;
    const int32_t cache_len = static_cast<int32_t>(k.d[2]);
    const LiteRtNvidiaInt8CacheParams int8 = {v_.k_scale, v_.k_zero, v_.v_scale,
                                              v_.v_zero, v_.int8_cache == 2};
    const cudaError_t status = LiteRtNvidiaLaunchDecodeAttentionSlots(
        table, v_.slots, input_desc[0].type == nvinfer1::DataType::kBF16,
        v_.int8_cache != 0 ? &int8 : nullptr,
        static_cast<int32_t>(input_desc[3].dims.d[2]),
        static_cast<int32_t>(q.d[1]), static_cast<int32_t>(q.d[2]), cache_len,
        v_.window != 0 ? v_.window : cache_len, static_cast<int32_t>(q.d[3]),
        v_.fill, workspace, stream);
    if (status != cudaSuccess) {
      ReportLaunch(AttentionTraits::kName, status);
      return 1;
    }
    return 0;
  }

 private:
  // q, k, v and mask, and with a window its start.
  int32_t SlotInputs() const noexcept { return v_.window != 0 ? 5 : 4; }
  int32_t InputCount() const noexcept override {
    return v_.slots * SlotInputs();
  }

  // Every slot's tensors valid and of slot 0's shapes and types.
  bool ValidDescriptors(const nvinfer1::PluginTensorDesc* inputs,
                        const nvinfer1::PluginTensorDesc* outputs) const {
    if (inputs == nullptr || outputs == nullptr) return false;
    const int32_t n = SlotInputs();
    for (int32_t s = 0; s < v_.slots; ++s) {
      const nvinfer1::PluginTensorDesc* in = inputs + s * n;
      if (!ValidAttentionDims(v_, in[0].dims, in[1].dims, in[2].dims,
                              in[3].dims) ||
          (v_.window != 0 &&
           (in[4].dims.nbDims != 1 || in[4].dims.d[0] != 1))) {
        return false;
      }
      for (int32_t i = 0; i < 4; ++i) {
        if (in[i].type != inputs[i].type ||
            !SameDims(in[i].dims, inputs[i].dims)) {
          return false;
        }
      }
      if (outputs[s].type != inputs[0].type ||
          !SameDims(outputs[s].dims, inputs[0].dims)) {
        return false;
      }
    }
    return true;
  }
};

nvinfer1::IPluginV3* AttentionTraits::Make(const Values& values) noexcept {
  return Valid(values) ? new (std::nothrow) DecodeAttentionSlotsPlugin(values)
                       : nullptr;
}

// The twin of a decode attention layer, or null.
nvinfer1::IPluginV3* DecodeAttentionTwin(nvinfer1::IPluginV3Layer& layer,
                                         int32_t slots) {
  AttentionTraits::Values f;
  AttentionForms forms;
  const nvinfer1::PluginFieldCollection* fields = FieldsOf(layer.getPlugin());
  if (!ReadValues(fields, AttentionTraits::kFields, &f) ||
      !ReadValues(fields, kAttentionForms, &forms) || forms.new_keys != 0 ||
      forms.row_tokens != 0) {
    return nullptr;
  }
  f.slots = slots;
  if (!AttentionTraits::Valid(f) ||
      layer.getNbInputs() != (f.window != 0 ? 5 : 4) ||
      layer.getNbOutputs() != 1) {
    return nullptr;
  }
  for (int32_t i = 0; i < 4; ++i) {
    if (layer.getInput(i) == nullptr) return nullptr;
  }
  if (!ValidAttentionDims(f, layer.getInput(0)->getDimensions(),
                          layer.getInput(1)->getDimensions(),
                          layer.getInput(2)->getDimensions(),
                          layer.getInput(3)->getDimensions())) {
    return nullptr;
  }
  return AttentionTraits::Make(f);
}

// --- QkvNormRope ------------------------------------------------------------

struct QkvTraits {
  struct Values {
    int32_t slots = 0;
    int32_t heads = 0;
    int32_t kv_heads = 0;
    int32_t depth = 0;
    int32_t rotary = 0;
    int32_t has_v_norm = 0;
    float epsilon = 0.0f;
    float min_timescale = 1.0f;
    float max_timescale = 1.0f;
  };
  static constexpr char kName[] = "LiteRtNvidiaQkvNormRopeSlots";
  static constexpr FieldSpec kFields[] = {
      {"slots", false, offsetof(Values, slots)},
      {"heads", false, offsetof(Values, heads)},
      {"kv_heads", false, offsetof(Values, kv_heads)},
      {"depth", false, offsetof(Values, depth)},
      {"rotary", false, offsetof(Values, rotary)},
      {"has_v_norm", false, offsetof(Values, has_v_norm)},
      {"epsilon", true, offsetof(Values, epsilon)},
      {"min_timescale", true, offsetof(Values, min_timescale)},
      {"max_timescale", true, offsetof(Values, max_timescale)},
  };
  static LiteRtNvidiaQkvNormRopeParams Params(const Values& v, int32_t tokens) {
    return {tokens,       v.heads,   v.kv_heads,      v.depth,        v.rotary,
            v.has_v_norm, v.epsilon, v.min_timescale, v.max_timescale};
  }
  static bool Valid(const Values& v) {
    const LiteRtNvidiaQkvNormRopeParams params = Params(v, 1);
    return v.slots >= 1 && v.slots <= kLiteRtNvidiaQkvNormRopeMaxSlots &&
           LiteRtNvidiaQkvNormRopeSupports(&params);
  }
  static nvinfer1::IPluginV3* Make(const Values& values) noexcept;
};

// The token count of a qkv input [..., tokens, width], or 0.
int64_t QkvTokens(const nvinfer1::Dims& dims, int64_t width) {
  if (dims.nbDims < 2 || dims.d[dims.nbDims - 1] != width) return 0;
  int64_t tokens = 1;
  for (int i = 0; i + 1 < dims.nbDims; ++i) {
    if (dims.d[i] <= 0 ||
        tokens > std::numeric_limits<int32_t>::max() / dims.d[i]) {
      return 0;
    }
    tokens *= dims.d[i];
  }
  return tokens;
}

// Inputs per slot: qkv BF16 [..., tokens, (heads + 2 kv_heads) * depth],
// positions INT32 (tokens values), q_scale and k_scale FP32 [depth].
// Outputs per slot: q [1, heads, tokens, depth], k and v [1, kv_heads,
// tokens, depth], BF16.
class QkvNormRopeSlotsPlugin final : public SlotPlugin<QkvTraits> {
 public:
  using SlotPlugin::SlotPlugin;

  int32_t getNbOutputs() const noexcept override { return 3 * v_.slots; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != InputCount() || num_outputs != getNbOutputs()) {
      return 1;
    }
    for (int32_t i = 0; i < num_inputs; ++i) {
      if (input_types[i] != InputType(i)) return 1;
    }
    for (int32_t o = 0; o < num_outputs; ++o) {
      output_types[o] = nvinfer1::DataType::kBF16;
    }
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != InputCount() ||
        num_outputs != getNbOutputs() || position < 0 ||
        position >= num_inputs + num_outputs) {
      return false;
    }
    const nvinfer1::DataType type =
        position < num_inputs ? InputType(position) : nvinfer1::DataType::kBF16;
    return in_out[position].desc.type == type &&
           in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    if (inputs == nullptr || outputs == nullptr || num_inputs != InputCount() ||
        num_shape_inputs != 0 || num_outputs != getNbOutputs()) {
      return 1;
    }
    for (int32_t s = 0; s < v_.slots; ++s) {
      const nvinfer1::DimsExprs& qkv = inputs[4 * s];
      if (qkv.nbDims < 2) return 1;
      // tokens = every leading dim of the qkv input.
      const nvinfer1::IDimensionExpr* tokens = expr_builder.constant(1);
      for (int i = 0; i + 1 < qkv.nbDims; ++i) {
        tokens = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                        *tokens, *qkv.d[i]);
      }
      if (tokens == nullptr) return 1;
      for (int o = 0; o < 3; ++o) {
        nvinfer1::DimsExprs& out = outputs[3 * s + o];
        out.nbDims = 4;
        out.d[0] = expr_builder.constant(1);
        out.d[1] = expr_builder.constant(o == 0 ? v_.heads : v_.kv_heads);
        out.d[2] = tokens;
        out.d[3] = expr_builder.constant(v_.depth);
      }
    }
    return 0;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return outputs != nullptr && num_inputs == InputCount() &&
                   num_outputs == getNbOutputs() && Tokens(inputs) >= 1
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
                          const nvinfer1::DynamicPluginTensorDesc*,
                          int32_t) const noexcept override {
    return 0;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    static_cast<void>(workspace);
    const int64_t tokens = Tokens(input_desc);
    if (tokens < 1 || inputs == nullptr || outputs == nullptr) return 1;
    LiteRtNvidiaQkvNormRopeSlot table[kLiteRtNvidiaQkvNormRopeMaxSlots];
    for (int32_t s = 0; s < v_.slots; ++s) {
      table[s] = {inputs[4 * s], static_cast<const int32_t*>(inputs[4 * s + 1]),
                  outputs[3 * s], outputs[3 * s + 1], outputs[3 * s + 2]};
    }
    const LiteRtNvidiaQkvNormRopeParams params =
        QkvTraits::Params(v_, static_cast<int32_t>(tokens));
    // The scales are every slot's: slot 0's inputs.
    const cudaError_t status = LiteRtNvidiaLaunchQkvNormRopeSlots(
        &params, table, v_.slots, static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[3]), stream);
    if (status != cudaSuccess) {
      ReportLaunch(QkvTraits::kName, status);
      return 1;
    }
    return 0;
  }

 private:
  int32_t InputCount() const noexcept override { return 4 * v_.slots; }

  static nvinfer1::DataType InputType(int32_t position) {
    switch (position % 4) {
      case 0:
        return nvinfer1::DataType::kBF16;
      case 1:
        return nvinfer1::DataType::kINT32;
      default:
        return nvinfer1::DataType::kFLOAT;
    }
  }

  // The tokens of a slot when every slot's qkv has slot 0's shape, or 0.
  int64_t Tokens(const nvinfer1::PluginTensorDesc* inputs) const {
    if (inputs == nullptr) return 0;
    for (int32_t s = 1; s < v_.slots; ++s) {
      if (!SameDims(inputs[4 * s].dims, inputs[0].dims)) return 0;
    }
    return QkvTokens(
        inputs[0].dims,
        (static_cast<int64_t>(v_.heads) + 2 * v_.kv_heads) * v_.depth);
  }
};

nvinfer1::IPluginV3* QkvTraits::Make(const Values& values) noexcept {
  return Valid(values) ? new (std::nothrow) QkvNormRopeSlotsPlugin(values)
                       : nullptr;
}

nvinfer1::IPluginV3* QkvNormRopeTwin(nvinfer1::IPluginV3Layer& layer,
                                     int32_t slots) {
  QkvTraits::Values f;
  if (!ReadValues(FieldsOf(layer.getPlugin()), QkvTraits::kFields, &f) ||
      layer.getNbInputs() != 4 || layer.getNbOutputs() != 3 ||
      layer.getInput(0) == nullptr) {
    return nullptr;
  }
  f.slots = slots;
  if (QkvTokens(layer.getInput(0)->getDimensions(),
                (static_cast<int64_t>(f.heads) + 2 * f.kv_heads) * f.depth) <
      1) {
    return nullptr;
  }
  return QkvTraits::Make(f);
}

// --- Cache writes -----------------------------------------------------------

struct WriteTraits {
  struct Values {
    int32_t writes = 0;
    int32_t int8_cache = 0;  // 0 FP16 caches, 1 INT8 with scale and zero_point
    int32_t columns = 0;
    float scale = 1.0f;
    int32_t zero_point = 0;
  };
  static constexpr char kName[] = "LiteRtNvidiaCacheWriteSlots";
  static constexpr FieldSpec kFields[] = {
      {"slots", false, offsetof(Values, writes)},
      {"int8_cache", false, offsetof(Values, int8_cache)},
      {"value_columns", false, offsetof(Values, columns)},
      {"scale", true, offsetof(Values, scale)},
      {"zero_point", false, offsetof(Values, zero_point)},
  };
  static bool Valid(const Values& v) {
    return v.writes >= 1 && v.writes <= kLiteRtNvidiaCacheWriteMaxWrites &&
           (v.int8_cache == 0 || v.scale > 0.0f);
  }
  static nvinfer1::IPluginV3* Make(const Values& values) noexcept;
};

// Inputs per write: the cache [1, H, S, D] (rows) or [1, H, D, S]
// (columns), INT8 or HALF; the rows [1, H, T, D] with T <= S, a float type;
// the position INT32 [1]. Outputs per write: the cache, updated.
class CacheWriteSlotsPlugin final : public SlotPlugin<WriteTraits> {
 public:
  using SlotPlugin::SlotPlugin;

  int32_t getNbOutputs() const noexcept override { return v_.writes; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != InputCount() || num_outputs != v_.writes) {
      return 1;
    }
    for (int32_t w = 0; w < v_.writes; ++w) output_types[w] = CacheType();
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != InputCount() ||
        num_outputs != v_.writes || position < 0 ||
        position >= num_inputs + num_outputs) {
      return false;
    }
    const auto& desc = in_out[position].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;
    if (position >= num_inputs) return desc.type == CacheType();
    switch (position % 3) {
      case 0:
        return desc.type == CacheType();
      case 1:
        // One launch reads every write's rows in the type of write 0's.
        return FloatType(desc.type) && desc.type == in_out[1].desc.type;
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
        num_shape_inputs != 0 || num_outputs != v_.writes) {
      return 1;
    }
    for (int32_t w = 0; w < v_.writes; ++w) {
      if (inputs[3 * w].nbDims != 4) return 1;
      outputs[w] = inputs[3 * w];
    }
    return 0;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return num_inputs == InputCount() && num_outputs == v_.writes &&
                   ValidDescriptors(inputs, outputs)
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
                          const nvinfer1::DynamicPluginTensorDesc*,
                          int32_t) const noexcept override {
    return 0;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(workspace);
    if (!ValidDescriptors(input_desc, output_desc) || inputs == nullptr ||
        outputs == nullptr) {
      return 1;
    }
    const auto& cache = input_desc[0].dims;
    const bool columns = v_.columns != 0;
    const size_t cache_bytes = static_cast<size_t>(cache.d[1]) * cache.d[2] *
                               cache.d[3] * (v_.int8_cache != 0 ? 1 : 2);
    LiteRtNvidiaCacheWrite table[kLiteRtNvidiaCacheWriteMaxWrites];
    for (int32_t w = 0; w < v_.writes; ++w) {
      const void* const* in = inputs + 3 * w;
      if (in[0] == nullptr || in[1] == nullptr || in[2] == nullptr ||
          outputs[w] == nullptr) {
        return 1;
      }
      if (outputs[w] != in[0]) {
        // Not the in-place binding: the rest of the cache has to follow.
        const cudaError_t copied = cudaMemcpyAsync(
            outputs[w], in[0], cache_bytes, cudaMemcpyDeviceToDevice, stream);
        if (copied != cudaSuccess) {
          ReportLaunch(WriteTraits::kName, copied);
          return 1;
        }
      }
      table[w] = {
          outputs[w],   in[1],         static_cast<const int32_t*>(in[2]),
          v_.columns,   v_.int8_cache, v_.scale,
          v_.zero_point};
    }
    const nvinfer1::DataType source = input_desc[1].type;
    const cudaError_t status = LiteRtNvidiaLaunchCacheWrites(
        table, v_.writes,
        source == nvinfer1::DataType::kHALF   ? kLiteRtNvidiaCacheWriteHalf
        : source == nvinfer1::DataType::kBF16 ? kLiteRtNvidiaCacheWriteBf16
                                              : kLiteRtNvidiaCacheWriteFloat,
        static_cast<int32_t>(cache.d[1]),
        static_cast<int32_t>(columns ? cache.d[3] : cache.d[2]),
        static_cast<int32_t>(columns ? cache.d[2] : cache.d[3]),
        static_cast<int32_t>(input_desc[1].dims.d[2]), stream);
    if (status != cudaSuccess) {
      ReportLaunch(WriteTraits::kName, status);
      return 1;
    }
    return 0;
  }

 private:
  int32_t InputCount() const noexcept override { return 3 * v_.writes; }

  nvinfer1::DataType CacheType() const noexcept {
    return v_.int8_cache != 0 ? nvinfer1::DataType::kINT8
                              : nvinfer1::DataType::kHALF;
  }

  // Every write's tensors valid and of write 0's shapes and types.
  bool ValidDescriptors(const nvinfer1::PluginTensorDesc* inputs,
                        const nvinfer1::PluginTensorDesc* outputs) const {
    if (inputs == nullptr || outputs == nullptr) return false;
    for (int32_t w = 0; w < v_.writes; ++w) {
      const nvinfer1::PluginTensorDesc* in = inputs + 3 * w;
      const auto& cache = in[0].dims;
      const auto& update = in[1].dims;
      if (!ValidDimensions(cache) || !ValidDimensions(update) ||
          in[0].type != CacheType() || outputs[w].type != CacheType() ||
          !FloatType(in[1].type) || in[1].type != inputs[1].type ||
          in[2].type != nvinfer1::DataType::kINT32 || in[2].dims.nbDims != 1 ||
          in[2].dims.d[0] != 1 || !SameDims(cache, inputs[0].dims) ||
          !SameDims(update, inputs[1].dims) ||
          !SameDims(outputs[w].dims, cache)) {
        return false;
      }
      const int64_t cache_len = v_.columns != 0 ? cache.d[3] : cache.d[2];
      const int64_t depth = v_.columns != 0 ? cache.d[2] : cache.d[3];
      if (cache.d[0] != 1 || update.d[0] != 1 || update.d[1] != cache.d[1] ||
          update.d[3] != depth || update.d[2] > cache_len) {
        return false;
      }
    }
    return true;
  }
};

nvinfer1::IPluginV3* WriteTraits::Make(const Values& values) noexcept {
  return Valid(values) ? new (std::nothrow) CacheWriteSlotsPlugin(values)
                       : nullptr;
}

// The twin of an INT8 cache write layer: the layer's value_columns, scale
// and zero_point for every slot's write.
nvinfer1::IPluginV3* Int8CacheWriteTwin(nvinfer1::IPluginV3Layer& layer,
                                        int32_t slots) {
  WriteTraits::Values f;
  if (!ReadValues(FieldsOf(layer.getPlugin()), WriteTraits::kFields, &f) ||
      layer.getNbInputs() != 3 || layer.getNbOutputs() != 1) {
    return nullptr;
  }
  f.writes = slots;
  f.int8_cache = 1;
  return WriteTraits::Make(f);
}

using DecodeAttentionSlotsCreator = SlotCreator<AttentionTraits>;
using QkvNormRopeSlotsCreator = SlotCreator<QkvTraits>;
using CacheWriteSlotsCreator = SlotCreator<WriteTraits>;

}  // namespace

nvinfer1::IPluginV3* CreateSlotsTwin(nvinfer1::IPluginV3Layer& layer,
                                     int32_t slots) noexcept {
  if (slots < 2 || slots > kMaxSlots) return nullptr;
  const char* name = NameOf(layer.getPlugin());
  if (std::strcmp(name, kDecodeAttentionName) == 0) {
    return DecodeAttentionTwin(layer, slots);
  }
  if (std::strcmp(name, kQkvNormRopeName) == 0) {
    return QkvNormRopeTwin(layer, slots);
  }
  if (std::strcmp(name, kInt8CacheWriteName) == 0) {
    return Int8CacheWriteTwin(layer, slots);
  }
  return nullptr;
}

nvinfer1::IPluginV3* CreateSlotRowsPlugin(
    nvinfer1::IPluginV3& plugin) noexcept {
  return MoeSlotRowsPlugin(plugin);
}

nvinfer1::IPluginV3* CreateCacheWriteSlotsPlugin(int32_t writes,
                                                 bool columns) noexcept {
  WriteTraits::Values values;
  values.writes = writes;
  values.columns = columns ? 1 : 0;
  return WriteTraits::Make(values);
}

void EnsureSlotPluginsRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(DecodeAttentionSlotsCreator);
REGISTER_TENSORRT_PLUGIN(QkvNormRopeSlotsCreator);
REGISTER_TENSORRT_PLUGIN(CacheWriteSlotsCreator);

}  // namespace litert::nvidia
