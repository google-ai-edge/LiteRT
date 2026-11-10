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

#include "litert/vendors/nvidia/compiler/int8_cache_write_plugin.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "litert/vendors/nvidia/compiler/int8_cache_write_kernel.h"
#include "NvInferRuntime.h"

namespace litert::nvidia {
namespace {

constexpr char kPluginName[] = "LiteRtNvidiaInt8CacheWrite";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";
constexpr char kValueColumnsField[] = "value_columns";
constexpr char kScaleField[] = "scale";
constexpr char kZeroPointField[] = "zero_point";
constexpr int kNumFields = 3;
constexpr int kNumInputs = 3;

bool ValidDimensions(const nvinfer1::Dims& dims) {
  if (dims.nbDims != 4) return false;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0 || dims.d[i] > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool FloatType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kHALF ||
         type == nvinfer1::DataType::kBF16 ||
         type == nvinfer1::DataType::kFLOAT;
}

// cache [1, H, S, D] (rows) or [1, H, D, S] (columns), update [1, H, T, D]
// with T <= S, position int32 [1], output like the cache.
bool ValidDescriptors(const nvinfer1::PluginTensorDesc* inputs,
                      const nvinfer1::PluginTensorDesc* output,
                      bool value_columns) {
  if (inputs == nullptr || output == nullptr) return false;
  for (int i = 0; i < kNumInputs; ++i) {
    if (inputs[i].format != nvinfer1::TensorFormat::kLINEAR) return false;
  }
  if (output->format != nvinfer1::TensorFormat::kLINEAR) return false;
  const auto& cache = inputs[0].dims;
  const auto& update = inputs[1].dims;
  const auto& position = inputs[2].dims;
  if (!ValidDimensions(cache) || !ValidDimensions(update) ||
      !ValidDimensions(output->dims) ||
      inputs[0].type != nvinfer1::DataType::kINT8 ||
      output->type != nvinfer1::DataType::kINT8 || !FloatType(inputs[1].type) ||
      inputs[2].type != nvinfer1::DataType::kINT32 || position.nbDims != 1 ||
      position.d[0] != 1) {
    return false;
  }
  for (int i = 0; i < 4; ++i) {
    if (output->dims.d[i] != cache.d[i]) return false;
  }
  const int64_t heads = cache.d[1];
  const int64_t cache_len = value_columns ? cache.d[3] : cache.d[2];
  const int64_t depth = value_columns ? cache.d[2] : cache.d[3];
  return update.d[0] == 1 && update.d[1] == heads && update.d[3] == depth &&
         update.d[2] <= cache_len;
}

Int8CacheWriteSource SourceOf(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kHALF:
      return Int8CacheWriteSource::kHalf;
    case nvinfer1::DataType::kBF16:
      return Int8CacheWriteSource::kBf16;
    default:
      return Int8CacheWriteSource::kFloat;
  }
}

class Int8CacheWritePlugin final : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuild,
                                   public nvinfer1::IPluginV3OneRuntime {
 public:
  Int8CacheWritePlugin(bool value_columns, float scale,
                       int32_t zero_point) noexcept
      : value_columns_(value_columns ? 1 : 0),
        scale_(scale),
        zero_point_(zero_point),
        fields_{{{kValueColumnsField, &value_columns_,
                  nvinfer1::PluginFieldType::kINT32, 1},
                 {kScaleField, &scale_, nvinfer1::PluginFieldType::kFLOAT32, 1},
                 {kZeroPointField, &zero_point_,
                  nvinfer1::PluginFieldType::kINT32, 1}}},
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
        Int8CacheWritePlugin(value_columns_ != 0, scale_, zero_point_);
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
        num_inputs != kNumInputs || num_outputs != 1) {
      return 1;
    }
    output_types[0] = nvinfer1::DataType::kINT8;
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != kNumInputs || num_outputs != 1 ||
        position < 0 || position > num_inputs) {
      return false;
    }
    const auto& desc = in_out[position].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;
    switch (position) {
      case 0:
      case 3:
        return desc.type == nvinfer1::DataType::kINT8;
      case 1:
        return FloatType(desc.type);
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
    if (inputs == nullptr || outputs == nullptr || num_inputs != kNumInputs ||
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
                   num_inputs == kNumInputs && num_outputs == 1
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return num_inputs == kNumInputs && num_outputs == 1 &&
                   ValidDescriptors(inputs, outputs, value_columns_ != 0)
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(inputs);
    static_cast<void>(num_inputs);
    static_cast<void>(outputs);
    static_cast<void>(num_outputs);
    return 0;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(workspace);
    if (!ValidDescriptors(input_desc, output_desc, value_columns_ != 0) ||
        inputs == nullptr || outputs == nullptr || outputs[0] == nullptr ||
        inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
      return 1;
    }
    const auto& cache = input_desc[0].dims;
    const auto& update = input_desc[1].dims;
    const bool columns = value_columns_ != 0;
    const int32_t heads = static_cast<int32_t>(cache.d[1]);
    const int32_t cache_len =
        static_cast<int32_t>(columns ? cache.d[3] : cache.d[2]);
    const int32_t depth =
        static_cast<int32_t>(columns ? cache.d[2] : cache.d[3]);
    const int32_t tokens = static_cast<int32_t>(update.d[2]);
    const size_t cache_bytes =
        static_cast<size_t>(cache.d[0]) * cache.d[1] * cache.d[2] * cache.d[3];
    const cudaError_t status = LaunchInt8CacheWrite(
        outputs[0], inputs[0], cache_bytes, inputs[1],
        SourceOf(input_desc[1].type), static_cast<const int32_t*>(inputs[2]),
        heads, cache_len, depth, tokens, columns, scale_, zero_point_, stream);
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "[LiteRtNvidiaInt8CacheWrite] CUDA launch failed: %s (%d): "
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
  int32_t value_columns_;
  float scale_;
  int32_t zero_point_;
  std::array<nvinfer1::PluginField, kNumFields> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class Int8CacheWritePluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  Int8CacheWritePluginCreator() noexcept
      : fields_{
            {{kValueColumnsField, nullptr, nvinfer1::PluginFieldType::kINT32,
              1},
             {kScaleField, nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
             {kZeroPointField, nullptr, nvinfer1::PluginFieldType::kINT32, 1}}},
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
    int32_t value_columns = 0;
    float scale = 0.0f;
    int32_t zero_point = 0;
    for (int32_t i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length != 1) {
        continue;
      }
      if (field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        if (std::strcmp(field.name, kScaleField) == 0) {
          scale = *static_cast<const float*>(field.data);
        }
      } else if (field.type == nvinfer1::PluginFieldType::kINT32) {
        const int32_t value = *static_cast<const int32_t*>(field.data);
        if (std::strcmp(field.name, kValueColumnsField) == 0) {
          value_columns = value;
        } else if (std::strcmp(field.name, kZeroPointField) == 0) {
          zero_point = value;
        }
      }
    }
    return CreateInt8CacheWritePlugin(value_columns != 0, scale, zero_point);
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

nvinfer1::IPluginV3* CreateInt8CacheWritePlugin(bool value_columns, float scale,
                                                int32_t zero_point) noexcept {
  if (!std::isfinite(scale) || scale <= 0.0f || zero_point < -128 ||
      zero_point > 127) {
    return nullptr;
  }
  return new (std::nothrow)
      Int8CacheWritePlugin(value_columns, scale, zero_point);
}

void EnsureInt8CacheWritePluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(Int8CacheWritePluginCreator);

}  // namespace litert::nvidia
