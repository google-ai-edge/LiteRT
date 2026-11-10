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

#include "litert/vendors/nvidia/compiler/qkv_norm_rope_plugin.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/qkv_norm_rope.h"

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

constexpr char kPluginName[] = "LiteRtNvidiaQkvNormRope";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kHeadsField[] = "heads";
constexpr char kKvHeadsField[] = "kv_heads";
constexpr char kDepthField[] = "depth";
constexpr char kRotaryField[] = "rotary";
constexpr char kHasVNormField[] = "has_v_norm";
constexpr char kEpsilonField[] = "epsilon";
constexpr char kMinTimescaleField[] = "min_timescale";
constexpr char kMaxTimescaleField[] = "max_timescale";
constexpr int kFields = 8;

// The token count of a qkv input [..., tokens, width], or 0.
int64_t Tokens(const nvinfer1::Dims& dims, int32_t width) {
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

class QkvNormRopePlugin final : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime {
 public:
  QkvNormRopePlugin(int32_t heads, int32_t kv_heads, int32_t depth,
                    int32_t rotary, int32_t has_v_norm, float epsilon,
                    float min_timescale, float max_timescale) noexcept
      : heads_(heads),
        kv_heads_(kv_heads),
        depth_(depth),
        rotary_(rotary),
        has_v_norm_(has_v_norm),
        epsilon_(epsilon),
        min_timescale_(min_timescale),
        max_timescale_(max_timescale),
        fields_{
            {{kHeadsField, &heads_, nvinfer1::PluginFieldType::kINT32, 1},
             {kKvHeadsField, &kv_heads_, nvinfer1::PluginFieldType::kINT32, 1},
             {kDepthField, &depth_, nvinfer1::PluginFieldType::kINT32, 1},
             {kRotaryField, &rotary_, nvinfer1::PluginFieldType::kINT32, 1},
             {kHasVNormField, &has_v_norm_, nvinfer1::PluginFieldType::kINT32,
              1},
             {kEpsilonField, &epsilon_, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kMinTimescaleField, &min_timescale_,
              nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kMaxTimescaleField, &max_timescale_,
              nvinfer1::PluginFieldType::kFLOAT32, 1}}},
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
    return new (std::nothrow)
        QkvNormRopePlugin(heads_, kv_heads_, depth_, rotary_, has_v_norm_,
                          epsilon_, min_timescale_, max_timescale_);
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }
  int32_t getNbOutputs() const noexcept override { return 3; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr || num_inputs != 4 ||
        num_outputs != 3) {
      return 1;
    }
    for (int i = 0; i < 4; ++i) {
      if (input_types[i] != InputType(i)) return 1;
    }
    for (int i = 0; i < 3; ++i) output_types[i] = nvinfer1::DataType::kBF16;
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != 4 || num_outputs != 3 ||
        position < 0 || position > 6) {
      return false;
    }
    const nvinfer1::DataType type =
        position < 4 ? InputType(position) : nvinfer1::DataType::kBF16;
    return in_out[position].desc.type == type &&
           in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    if (inputs == nullptr || outputs == nullptr || num_inputs != 4 ||
        num_shape_inputs != 0 || num_outputs != 3 || inputs[0].nbDims < 2) {
      return 1;
    }
    // tokens = every leading dim of the qkv input.
    const nvinfer1::IDimensionExpr* tokens = expr_builder.constant(1);
    for (int i = 0; i + 1 < inputs[0].nbDims; ++i) {
      tokens = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                      *tokens, *inputs[0].d[i]);
    }
    for (int o = 0; o < 3; ++o) {
      outputs[o].nbDims = 4;
      outputs[o].d[0] = expr_builder.constant(1);
      outputs[o].d[1] = expr_builder.constant(o == 0 ? heads_ : kv_heads_);
      outputs[o].d[2] = tokens;
      outputs[o].d[3] = expr_builder.constant(depth_);
    }
    return tokens != nullptr ? 0 : 1;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr && num_inputs == 4 &&
                   num_outputs == 3
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr && num_inputs == 4 &&
                   num_outputs == 3
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
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    const int64_t tokens = Tokens(input_desc[0].dims, Width());
    if (tokens < 1) return 1;
    if (SkipPluginFamily("qkv_norm_rope")) return 0;
    const LiteRtNvidiaQkvNormRopeParams params = {static_cast<int32_t>(tokens),
                                                  heads_,
                                                  kv_heads_,
                                                  depth_,
                                                  rotary_,
                                                  has_v_norm_,
                                                  epsilon_,
                                                  min_timescale_,
                                                  max_timescale_};
    const cudaError_t status = LiteRtNvidiaLaunchQkvNormRope(
        &params, inputs[0], static_cast<const int32_t*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[3]), outputs[0], outputs[1],
        outputs[2], stream);
    if (status != cudaSuccess) {
      std::fprintf(
          stderr, "[LiteRtNvidiaQkvNormRope] CUDA launch failed: %s (%d): %s\n",
          cudaGetErrorName(status), static_cast<int>(status),
          cudaGetErrorString(status));
      return 1;
    }
    return 0;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext*) noexcept override {
    return clone();
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return &field_collection_;
  }

 private:
  static nvinfer1::DataType InputType(int position) noexcept {
    switch (position) {
      case 0:
        return nvinfer1::DataType::kBF16;
      case 1:
        return nvinfer1::DataType::kINT32;
      default:
        return nvinfer1::DataType::kFLOAT;
    }
  }
  int32_t Width() const noexcept { return (heads_ + 2 * kv_heads_) * depth_; }

  int32_t heads_;
  int32_t kv_heads_;
  int32_t depth_;
  int32_t rotary_;
  int32_t has_v_norm_;
  float epsilon_;
  float min_timescale_;
  float max_timescale_;
  std::array<nvinfer1::PluginField, kFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class QkvNormRopePluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  QkvNormRopePluginCreator() noexcept
      : fields_{
            {{kHeadsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kKvHeadsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kDepthField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kRotaryField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kHasVNormField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kEpsilonField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kMinTimescaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32,
              1},
             {kMaxTimescaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32,
              1}}},
        field_collection_{static_cast<int32_t>(fields_.size()),
                          fields_.data()} {}

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    if (fields == nullptr || fields->fields == nullptr) return nullptr;
    int32_t ints[5] = {0, 0, 0, 0, 0};
    float floats[3] = {0.0f, 0.0f, 0.0f};
    const char* int_names[5] = {kHeadsField, kKvHeadsField, kDepthField,
                                kRotaryField, kHasVNormField};
    const char* float_names[3] = {kEpsilonField, kMinTimescaleField,
                                  kMaxTimescaleField};
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length != 1) {
        continue;
      }
      if (field.type == nvinfer1::PluginFieldType::kINT32) {
        for (int j = 0; j < 5; ++j) {
          if (std::strcmp(field.name, int_names[j]) == 0) {
            ints[j] = *static_cast<const int32_t*>(field.data);
          }
        }
      } else if (field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        for (int j = 0; j < 3; ++j) {
          if (std::strcmp(field.name, float_names[j]) == 0) {
            floats[j] = *static_cast<const float*>(field.data);
          }
        }
      }
    }
    return CreateQkvNormRopePlugin(ints[0], ints[1], ints[2], ints[3],
                                   ints[4] != 0, floats[0], floats[1],
                                   floats[2]);
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

nvinfer1::IPluginV3* CreateQkvNormRopePlugin(int32_t heads, int32_t kv_heads,
                                             int32_t depth, int32_t rotary,
                                             bool has_v_norm, float epsilon,
                                             float min_timescale,
                                             float max_timescale) noexcept {
  const LiteRtNvidiaQkvNormRopeParams params = {
      1,       heads,         kv_heads,     depth, rotary, has_v_norm ? 1 : 0,
      epsilon, min_timescale, max_timescale};
  if (!LiteRtNvidiaQkvNormRopeSupports(&params)) return nullptr;
  return new (std::nothrow)
      QkvNormRopePlugin(heads, kv_heads, depth, rotary, has_v_norm ? 1 : 0,
                        epsilon, min_timescale, max_timescale);
}

void EnsureQkvNormRopePluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(QkvNormRopePluginCreator);

}  // namespace litert::nvidia
