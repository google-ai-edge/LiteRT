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

#include "litert/vendors/nvidia/compiler/ple_tail_plugin.h"

#include <array>
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

constexpr char kPluginName[] = "LiteRtNvidiaPleTail";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr int kInputs = 5;
constexpr int kOutputs = 2;

// The plugin's fields: the kernel's parameters but the rows.
constexpr int kIntFields = 4;
constexpr int kFloatFields = 5;
constexpr int kFields = kIntFields + kFloatFields;
constexpr std::array<const char*, kFields> kFieldNames = {
    "streams",   "hidden",   "taps",      "silu", "eps_key",
    "eps_query", "eps_conv", "dot_scale", "floor"};

struct FieldValues {
  std::array<int32_t, kIntFields> ints{};
  std::array<float, kFloatFields> floats{};

  static FieldValues Of(const LiteRtNvidiaPleTailParams& p) noexcept {
    FieldValues v;
    v.ints = {p.streams, p.hidden, p.taps, p.silu};
    v.floats = {p.eps_key, p.eps_query, p.eps_conv, p.dot_scale, p.floor};
    return v;
  }

  LiteRtNvidiaPleTailParams Params(int32_t rows) const noexcept {
    return {rows,      ints[0],   ints[1],   ints[2],   floats[0],
            floats[1], floats[2], floats[3], floats[4], ints[3]};
  }

  void Describe(std::array<nvinfer1::PluginField, kFields>& fields) noexcept {
    for (int i = 0; i < kIntFields; ++i) {
      fields[i] = {kFieldNames[i], &ints[i], nvinfer1::PluginFieldType::kINT32,
                   1};
    }
    for (int i = 0; i < kFloatFields; ++i) {
      fields[kIntFields + i] = {kFieldNames[kIntFields + i], &floats[i],
                                nvinfer1::PluginFieldType::kFLOAT32, 1};
    }
  }

  bool Read(const nvinfer1::PluginFieldCollection* fields) noexcept {
    if (fields == nullptr || fields->fields == nullptr) return false;
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length != 1) {
        continue;
      }
      for (int j = 0; j < kFields; ++j) {
        if (std::strcmp(field.name, kFieldNames[j]) != 0) continue;
        if (j < kIntFields) {
          if (field.type == nvinfer1::PluginFieldType::kINT32) {
            ints[j] = *static_cast<const int32_t*>(field.data);
          }
        } else if (field.type == nvinfer1::PluginFieldType::kFLOAT32) {
          floats[j - kIntFields] = *static_cast<const float*>(field.data);
        }
      }
    }
    return true;
  }
};

class PleTailPlugin final : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit PleTailPlugin(const FieldValues& values) noexcept : values_(values) {
    values_.Describe(fields_);
    collection_ = {kFields, fields_.data()};
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

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) PleTailPlugin(values_);
  }
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }
  int32_t getNbOutputs() const noexcept override { return kOutputs; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != kInputs || num_outputs != kOutputs) {
      return 1;
    }
    for (int32_t i = 0; i < num_inputs; ++i) {
      if (input_types[i] != InputType(i)) return 1;
    }
    output_types[0] = nvinfer1::DataType::kBF16;
    output_types[1] = nvinfer1::DataType::kFLOAT;
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != kInputs || num_outputs != kOutputs ||
        position < 0 || position >= num_inputs + num_outputs) {
      return false;
    }
    const nvinfer1::DataType type =
        position < kInputs ? InputType(position)
                           : (position == kInputs ? nvinfer1::DataType::kBF16
                                                  : nvinfer1::DataType::kFLOAT);
    return in_out[position].desc.type == type &&
           in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    static_cast<void>(expr_builder);
    if (inputs == nullptr || outputs == nullptr || num_inputs != kInputs ||
        num_shape_inputs != 0 || num_outputs != kOutputs) {
      return 1;
    }
    outputs[0] = inputs[2];  // like x
    outputs[1] = inputs[3];  // like the state
    return 0;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr && num_inputs == kInputs &&
                   num_outputs == kOutputs
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr && num_inputs == kInputs &&
                   num_outputs == kOutputs
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
    if (SkipPluginFamily("ple_tail")) return 0;
    // Every leading dim of x [.., streams * hidden] is a row.
    const auto& dims = input_desc[2].dims;
    int64_t rows = dims.nbDims >= 1 ? 1 : 0;
    for (int i = 0; i + 1 < dims.nbDims; ++i) rows *= dims.d[i];
    cudaError_t status = cudaErrorInvalidValue;
    if (rows >= 1 && rows <= std::numeric_limits<int32_t>::max()) {
      const LiteRtNvidiaPleTailParams p =
          values_.Params(static_cast<int32_t>(rows));
      status = LiteRtNvidiaLaunchPleTail(
          &p, static_cast<const __nv_bfloat16*>(inputs[0]),
          static_cast<const __nv_bfloat16*>(inputs[1]),
          static_cast<const __nv_bfloat16*>(inputs[2]),
          static_cast<const float*>(inputs[3]),
          static_cast<const float*>(inputs[4]),
          static_cast<__nv_bfloat16*>(outputs[0]),
          static_cast<float*>(outputs[1]), stream);
    }
    if (status != cudaSuccess) {
      std::fprintf(stderr, "[%s] CUDA launch failed: %s (%d): %s\n",
                   kPluginName, cudaGetErrorName(status),
                   static_cast<int>(status), cudaGetErrorString(status));
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
    return &collection_;
  }

 private:
  static nvinfer1::DataType InputType(int32_t index) noexcept {
    return index < 3 ? nvinfer1::DataType::kBF16 : nvinfer1::DataType::kFLOAT;
  }

  FieldValues values_;
  std::array<nvinfer1::PluginField, kFields> fields_;
  nvinfer1::PluginFieldCollection collection_;
};

class PleTailPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  PleTailPluginCreator() noexcept {
    values_.Describe(fields_);
    for (auto& field : fields_) field.data = nullptr;
    collection_ = {kFields, fields_.data()};
  }

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    FieldValues values;
    if (!values.Read(fields)) return nullptr;
    return CreatePleTailPlugin(values.Params(1));
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &collection_;
  }
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  FieldValues values_;
  std::array<nvinfer1::PluginField, kFields> fields_;
  nvinfer1::PluginFieldCollection collection_;
};

}  // namespace

nvinfer1::IPluginV3* CreatePleTailPlugin(
    const LiteRtNvidiaPleTailParams& params) noexcept {
  // One row: the kernel's limits on the shapes are the same at every row
  // count it takes.
  LiteRtNvidiaPleTailParams one = params;
  one.rows = 1;
  if (!LiteRtNvidiaPleTailSupports(&one)) return nullptr;
  return new (std::nothrow) PleTailPlugin(FieldValues::Of(params));
}

void EnsurePleTailPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(PleTailPluginCreator);

}  // namespace litert::nvidia
