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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_PLUGIN_FIELDS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_PLUGIN_FIELDS_H_

// The plumbing the plugins of Qwen 4's attention composites share
// (qsa_attention_plugin.cc, general_attention_plugin.cc): int32 fields that
// travel in the engine (floats as their bits) and the IPluginV3 interfaces.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "cuda_runtime_api.h"
#include "NvInferRuntime.h"

namespace litert::nvidia::plugin_fields {

inline constexpr char kPluginVersion[] = "1";
inline constexpr char kPluginNamespace[] = "";

inline bool IsActivationType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kFLOAT ||
         type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kBF16;
}

inline bool IsStateType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kFLOAT ||
         type == nvinfer1::DataType::kHALF;
}

inline bool IsMaskType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kBOOL ||
         type == nvinfer1::DataType::kFLOAT;
}

inline size_t ElementBytes(nvinfer1::DataType type) {
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

inline int64_t Volume(const nvinfer1::Dims& dims) {
  int64_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) volume *= dims.d[i];
  return volume;
}

inline uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float BitsFloat(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline int32_t Report(const char* plugin, cudaError_t status) {
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

// The IPluginV3 plumbing of a plugin whose fields are int32 values.
template <int kFields>
class PluginBase : public nvinfer1::IPluginV3,
                   public nvinfer1::IPluginV3OneCore,
                   public nvinfer1::IPluginV3OneBuild,
                   public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit PluginBase(const IntFields<kFields>& fields) noexcept
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

}  // namespace litert::nvidia::plugin_fields

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_NVIDIA_COMPILER_PLUGIN_FIELDS_H_
