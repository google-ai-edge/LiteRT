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

#include "litert/vendors/nvidia/compiler/subbyte_gemv_plugin.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "litert/vendors/nvidia/tensorrt_rtx/include/NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/int2_gemv.h"

namespace litert::nvidia {
namespace {

constexpr char kPluginName[] = "LiteRtNvidiaSubbyteGemv";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kBitWidthField[] = "bit_width";
constexpr char kRowsField[] = "rows";
constexpr char kColumnsField[] = "columns";

class SubbyteGemvPlugin final : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime {
 public:
  SubbyteGemvPlugin(int32_t bit_width, int32_t rows, int32_t columns) noexcept
      : bit_width_(bit_width),
        rows_(rows),
        columns_(columns),
        fields_{
            {{kBitWidthField, &bit_width_, nvinfer1::PluginFieldType::kINT32,
              1},
             {kRowsField, &rows_, nvinfer1::PluginFieldType::kINT32, 1},
             {kColumnsField, &columns_, nvinfer1::PluginFieldType::kINT32, 1}}},
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
    return new (std::nothrow) SubbyteGemvPlugin(bit_width_, rows_, columns_);
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
    if (output_types == nullptr || input_types == nullptr || num_inputs != 3 ||
        num_outputs != 1 || input_types[0] != nvinfer1::DataType::kBF16 ||
        input_types[1] != nvinfer1::DataType::kINT8 ||
        input_types[2] != nvinfer1::DataType::kBF16) {
      return 1;
    }
    output_types[0] = nvinfer1::DataType::kBF16;
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != 3 || num_outputs != 1 ||
        position < 0 || position >= 4) {
      return false;
    }
    constexpr std::array<nvinfer1::DataType, 4> kTypes = {
        nvinfer1::DataType::kBF16, nvinfer1::DataType::kINT8,
        nvinfer1::DataType::kBF16, nvinfer1::DataType::kBF16};
    return in_out[position].desc.type == kTypes[position] &&
           in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    if (inputs == nullptr || outputs == nullptr || num_inputs != 3 ||
        num_shape_inputs != 0 || num_outputs != 1 || inputs[0].nbDims < 1) {
      return 1;
    }
    outputs[0] = inputs[0];
    outputs[0].d[outputs[0].nbDims - 1] = expr_builder.constant(rows_);
    return outputs[0].d[outputs[0].nbDims - 1] != nullptr ? 0 : 1;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr && num_inputs == 3 &&
                   num_outputs == 1
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr && num_inputs == 3 &&
                   num_outputs == 1
               ? 0
               : 1;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(input_desc);
    static_cast<void>(output_desc);
    static_cast<void>(workspace);
    if (inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    const cudaError_t status = LiteRtNvidiaLaunchBf16SubbytePerChannelGemv(
        inputs[0], static_cast<const uint8_t*>(inputs[1]), inputs[2],
        outputs[0], bit_width_, columns_, rows_, stream);
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaSubbyteGemv] CUDA launch failed: %s (%d): "
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
  int32_t bit_width_;
  int32_t rows_;
  int32_t columns_;
  std::array<nvinfer1::PluginField, 3> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class SubbyteGemvPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  SubbyteGemvPluginCreator() noexcept
      : fields_{
            {{kBitWidthField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kRowsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kColumnsField, nullptr, nvinfer1::PluginFieldType::kINT32, 1}}},
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
    int32_t bit_width = 0;
    int32_t rows = 0;
    int32_t columns = 0;
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr ||
          field.type != nvinfer1::PluginFieldType::kINT32 ||
          field.length != 1) {
        continue;
      }
      const int32_t value = *static_cast<const int32_t*>(field.data);
      if (std::strcmp(field.name, kBitWidthField) == 0) {
        bit_width = value;
      } else if (std::strcmp(field.name, kRowsField) == 0) {
        rows = value;
      } else if (std::strcmp(field.name, kColumnsField) == 0) {
        columns = value;
      }
    }
    return CreateSubbyteGemvPlugin(bit_width, rows, columns);
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
  std::array<nvinfer1::PluginField, 3> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

}  // namespace

nvinfer1::IPluginV3* CreateSubbyteGemvPlugin(int32_t bit_width, int32_t rows,
                                             int32_t columns) noexcept {
  if ((bit_width != 2 && bit_width != 4) || rows <= 0 || columns <= 0 ||
      columns % 16 != 0) {
    return nullptr;
  }
  return new (std::nothrow) SubbyteGemvPlugin(bit_width, rows, columns);
}

void EnsureSubbyteGemvPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(SubbyteGemvPluginCreator);

}  // namespace litert::nvidia
