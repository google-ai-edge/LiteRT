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

#include "litert/vendors/nvidia/compiler/cache_update_plugin.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include "litert/vendors/nvidia/compiler/cache_update_kernel.h"
#include "NvInferRuntime.h"

namespace litert::nvidia {
namespace {

constexpr char kPluginName[] = "LiteRtNvidiaCacheUpdatePatch";
constexpr char kPluginVersion[] = "2";
constexpr char kPluginNamespace[] = "";
constexpr char kRingField[] = "ring_buffer";
constexpr char kTransposedField[] = "transposed_value_cache";
constexpr char kForwardField[] = "forward_read";
constexpr int kNumInputs = 5;
constexpr int kNumOutputs = 2;

bool ValidDimensions(const nvinfer1::Dims& dims) {
  if (dims.nbDims != 4 || dims.d[0] != 1) return false;
  int64_t elements = 1;
  for (int i = 0; i < 4; ++i) {
    if (dims.d[i] <= 0 || dims.d[i] > std::numeric_limits<int32_t>::max() ||
        elements > std::numeric_limits<int64_t>::max() / 2 / dims.d[i]) {
      return false;
    }
    elements *= dims.d[i];
  }
  return true;
}

bool SameDimensions(const nvinfer1::Dims& a, const nvinfer1::Dims& b) {
  if (a.nbDims != b.nbDims) return false;
  for (int i = 0; i < a.nbDims; ++i) {
    if (a.d[i] != b.d[i]) return false;
  }
  return true;
}

bool IsOrderingType(nvinfer1::DataType type) {
  return type == nvinfer1::DataType::kFLOAT ||
         type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kBF16;
}

bool ValidOrderingDescriptor(const nvinfer1::PluginTensorDesc& desc) {
  // These values are opaque dependencies, not kernel operands. Scalars and
  // arbitrary-rank tensors are equally valid; no shape or size is interpreted.
  return desc.format == nvinfer1::TensorFormat::kLINEAR &&
         IsOrderingType(desc.type);
}

size_t ForwardBytes(const nvinfer1::PluginTensorDesc& desc) {
  if (!ValidOrderingDescriptor(desc) || desc.dims.nbDims < 0 ||
      desc.dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
    return 0;
  }
  size_t bytes = desc.type == nvinfer1::DataType::kFLOAT ? 4 : 2;
  for (int i = 0; i < desc.dims.nbDims; ++i) {
    if (desc.dims.d[i] <= 0 ||
        static_cast<uint64_t>(desc.dims.d[i]) >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                bytes ||
        static_cast<uint64_t>(desc.dims.d[i]) >
            std::numeric_limits<size_t>::max() / bytes) {
      return 0;
    }
    bytes *= desc.dims.d[i];
  }
  return bytes;
}

bool ValidInputCount(int32_t count, bool forward_read) {
  return forward_read ? count == kNumInputs + 1 : count >= kNumInputs;
}

bool ValidDescriptors(const nvinfer1::PluginTensorDesc* inputs,
                      int32_t num_inputs,
                      const nvinfer1::PluginTensorDesc* outputs,
                      bool ring_buffer, bool transposed_value_cache,
                      bool forward_read) {
  if (inputs == nullptr || outputs == nullptr ||
      !ValidInputCount(num_inputs, forward_read)) {
    return false;
  }
  for (int i = 0; i < kNumInputs; ++i) {
    if (!ValidDimensions(inputs[i].dims) ||
        inputs[i].format != nvinfer1::TensorFormat::kLINEAR ||
        inputs[i].type !=
            (i == 4 ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kHALF)) {
      return false;
    }
  }
  for (int i = kNumInputs; i < num_inputs; ++i) {
    if (!ValidOrderingDescriptor(inputs[i])) return false;
  }
  if (forward_read &&
      (ForwardBytes(inputs[5]) == 0 || outputs[2].type != inputs[5].type ||
       outputs[2].format != nvinfer1::TensorFormat::kLINEAR ||
       !SameDimensions(inputs[5].dims, outputs[2].dims))) {
    return false;
  }
  const auto& cache = inputs[0].dims;
  const auto& values = inputs[1].dims;
  const auto& update = inputs[2].dims;
  const auto& params = inputs[4].dims;
  for (int i = 0; i < kNumOutputs; ++i) {
    auto expected = inputs[i].dims;
    expected.d[2] = ring_buffer ? cache.d[2] : update.d[2];
    if (outputs[i].type != nvinfer1::DataType::kHALF ||
        outputs[i].format != nvinfer1::TensorFormat::kLINEAR ||
        !SameDimensions(expected, outputs[i].dims)) {
      return false;
    }
  }
  const bool valid_values = transposed_value_cache
                                ? SameDimensions(values, cache)
                                : values.d[1] == cache.d[1] * cache.d[3] &&
                                      values.d[2] == cache.d[2] &&
                                      values.d[3] == 1;
  return valid_values && update.d[1] == cache.d[1] &&
         update.d[2] <= cache.d[2] && update.d[3] == cache.d[3] &&
         SameDimensions(update, inputs[3].dims) && params.d[1] == 1 &&
         params.d[2] == 1 && params.d[3] == 7;
}

class CacheUpdatePlugin final : public nvinfer1::IPluginV3,
                                public nvinfer1::IPluginV3OneCore,
                                public nvinfer1::IPluginV3OneBuild,
                                public nvinfer1::IPluginV3OneRuntime {
 public:
  CacheUpdatePlugin(bool ring_buffer, bool transposed_value_cache,
                    bool forward_read) noexcept
      : ring_buffer_(ring_buffer),
        transposed_value_cache_(transposed_value_cache),
        forward_read_(forward_read),
        num_inputs_(forward_read ? kNumInputs + 1 : kNumInputs),
        fields_{
            {{kRingField, &ring_buffer_, nvinfer1::PluginFieldType::kINT32, 1},
             {kTransposedField, &transposed_value_cache_,
              nvinfer1::PluginFieldType::kINT32, 1},
             {kForwardField, &forward_read_, nvinfer1::PluginFieldType::kINT32,
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
    auto* copy = new (std::nothrow) CacheUpdatePlugin(
        ring_buffer_ != 0, transposed_value_cache_ != 0, forward_read_ != 0);
    if (copy != nullptr) copy->num_inputs_ = num_inputs_;
    return copy;
  }
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }
  int32_t getNbOutputs() const noexcept override {
    return kNumOutputs + forward_read_;
  }

  int32_t getOutputDataTypes(nvinfer1::DataType* outputs, int32_t num_outputs,
                             const nvinfer1::DataType* inputs,
                             int32_t num_inputs) const noexcept override {
    if (outputs == nullptr || inputs == nullptr ||
        !ValidInputCount(num_inputs, forward_read_ != 0) ||
        num_outputs != getNbOutputs()) {
      return 1;
    }
    for (int i = kNumInputs; i < num_inputs; ++i) {
      if (!IsOrderingType(inputs[i])) return 1;
    }
    outputs[0] = outputs[1] = nvinfer1::DataType::kHALF;
    if (forward_read_ != 0) outputs[2] = inputs[5];
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || !ValidInputCount(num_inputs, forward_read_ != 0) ||
        num_outputs != getNbOutputs() || position < 0 ||
        position >= static_cast<int64_t>(num_inputs) + num_outputs) {
      return false;
    }
    if (position >= kNumInputs && position < num_inputs) {
      return ValidOrderingDescriptor(in_out[position].desc);
    }
    if (forward_read_ != 0 && position == num_inputs + 2) {
      return in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR &&
             IsOrderingType(in_out[5].desc.type) &&
             in_out[position].desc.type == in_out[5].desc.type;
    }
    return in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR &&
           in_out[position].desc.type == (position == 4
                                              ? nvinfer1::DataType::kINT32
                                              : nvinfer1::DataType::kHALF);
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* /*shape_inputs*/, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& /*builder*/) noexcept override {
    if (inputs == nullptr || outputs == nullptr ||
        !ValidInputCount(num_inputs, forward_read_ != 0) ||
        num_shape_inputs != 0 || num_outputs != getNbOutputs() ||
        inputs[0].nbDims != 4 || inputs[1].nbDims != 4 ||
        inputs[2].nbDims != 4 ||
        (forward_read_ != 0 && (inputs[5].nbDims < 0 ||
                                inputs[5].nbDims > nvinfer1::Dims::MAX_DIMS))) {
      return 1;
    }
    outputs[0] = inputs[0];
    outputs[1] = inputs[1];
    outputs[0].d[2] = outputs[1].d[2] =
        ring_buffer_ != 0 ? inputs[0].d[2] : inputs[2].d[2];
    if (forward_read_ != 0) outputs[2] = inputs[5];
    return 0;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    if (inputs == nullptr || outputs == nullptr ||
        !ValidInputCount(num_inputs, forward_read_ != 0) ||
        num_outputs != getNbOutputs()) {
      return 1;
    }
    std::array<nvinfer1::PluginTensorDesc, kNumInputs + 1> in;
    std::array<nvinfer1::PluginTensorDesc, kNumOutputs + 1> out;
    const int copied_inputs = kNumInputs + forward_read_;
    for (int i = 0; i < copied_inputs; ++i) in[i] = inputs[i].desc;
    for (int i = 0; i < num_outputs; ++i) out[i] = outputs[i].desc;
    if (!ValidDescriptors(in.data(), copied_inputs, out.data(),
                          ring_buffer_ != 0, transposed_value_cache_ != 0,
                          forward_read_ != 0)) {
      return 1;
    }
    for (int i = kNumInputs; i < num_inputs; ++i) {
      if (!ValidOrderingDescriptor(inputs[i].desc)) return 1;
    }
    num_inputs_ = num_inputs;
    return 0;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    if (num_outputs != getNbOutputs() ||
        !ValidDescriptors(inputs, num_inputs, outputs, ring_buffer_ != 0,
                          transposed_value_cache_ != 0, forward_read_ != 0)) {
      return 1;
    }
    num_inputs_ = num_inputs;
    return 0;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* /*workspace*/, cudaStream_t stream) noexcept override {
    if (!ValidDescriptors(input_desc, num_inputs_, output_desc,
                          ring_buffer_ != 0, transposed_value_cache_ != 0,
                          forward_read_ != 0) ||
        inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    for (int o = 0; o < getNbOutputs(); ++o) {
      if (outputs[o] == nullptr) return 1;
      for (int p = 0; p < o; ++p) {
        if (outputs[o] == outputs[p]) return 1;
      }
      for (int i = 0; i < num_inputs_; ++i) {
        if (inputs[i] == nullptr || outputs[o] == inputs[i]) return 1;
      }
    }
    const auto& cache = input_desc[0].dims;
    return LaunchCacheUpdatePatch(
               inputs[0], inputs[1], inputs[2], inputs[3],
               static_cast<const int32_t*>(inputs[4]),
               static_cast<int32_t>(cache.d[1]),
               static_cast<int32_t>(cache.d[2]),
               static_cast<int32_t>(input_desc[2].dims.d[2]),
               static_cast<int32_t>(cache.d[3]), ring_buffer_ != 0,
               transposed_value_cache_ != 0, outputs[0], outputs[1], stream,
               forward_read_ != 0 ? inputs[5] : nullptr,
               forward_read_ != 0 ? outputs[2] : nullptr,
               forward_read_ != 0 ? ForwardBytes(input_desc[5]) : 0) ==
                   cudaSuccess
               ? 0
               : 1;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext* /*context*/) noexcept override {
    return clone();
  }
  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return &field_collection_;
  }

 private:
  int32_t ring_buffer_;
  int32_t transposed_value_cache_;
  int32_t forward_read_;
  // TensorRT supplies the actual count during configuration/shape changes;
  // enqueue itself does not receive counts. Without forward_read, extra
  // pointers are only dependencies and their contents are never read.
  int32_t num_inputs_;
  std::array<nvinfer1::PluginField, 3> fields_;
  nvinfer1::PluginFieldCollection field_collection_;
};

class CacheUpdatePluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  CacheUpdatePluginCreator() noexcept
      : fields_{
            {{kRingField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kTransposedField, nullptr, nvinfer1::PluginFieldType::kINT32, 1},
             {kForwardField, nullptr, nvinfer1::PluginFieldType::kINT32, 1}}},
        field_collection_{static_cast<int32_t>(fields_.size()),
                          fields_.data()} {}

  nvinfer1::IPluginV3* createPlugin(
      const char* /*name*/, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase /*phase*/) noexcept override {
    if (fields == nullptr || fields->fields == nullptr ||
        fields->nbFields != 3) {
      return nullptr;
    }
    int32_t ring = -1;
    int32_t transposed = -1;
    int32_t forward = -1;
    for (int i = 0; i < fields->nbFields; ++i) {
      const auto& field = fields->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length != 1 ||
          field.type != nvinfer1::PluginFieldType::kINT32) {
        return nullptr;
      }
      int32_t value;
      std::memcpy(&value, field.data, sizeof(value));
      if (value != 0 && value != 1) return nullptr;
      if (std::strcmp(field.name, kRingField) == 0 && ring == -1) {
        ring = value;
      } else if (std::strcmp(field.name, kTransposedField) == 0 &&
                 transposed == -1) {
        transposed = value;
      } else if (std::strcmp(field.name, kForwardField) == 0 && forward == -1) {
        forward = value;
      } else {
        return nullptr;
      }
    }
    return ring >= 0 && transposed >= 0 && forward >= 0
               ? CreateCacheUpdatePlugin(ring != 0, transposed != 0,
                                         forward != 0)
               : nullptr;
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

nvinfer1::IPluginV3* CreateCacheUpdatePlugin(bool ring_buffer,
                                             bool transposed_value_cache,
                                             bool forward_read) noexcept {
  return new (std::nothrow)
      CacheUpdatePlugin(ring_buffer, transposed_value_cache, forward_read);
}

void EnsureCacheUpdatePluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(CacheUpdatePluginCreator);

}  // namespace litert::nvidia
