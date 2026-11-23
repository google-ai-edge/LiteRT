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

#include "litert/vendors/nvidia/compiler/gated_delta_plugin.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "NvInferRuntime.h"
#include "litert/vendors/nvidia/trtllm/gated_delta.h"

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

// Debug: LITERT_NVIDIA_GATED_DELTA_DEBUG=1 prints the rows and valid length
// of every launch (synchronizes the stream; run without CUDA graphs).
bool DebugLaunches() {
  static const bool enabled = [] {
    const char* value = std::getenv("LITERT_NVIDIA_GATED_DELTA_DEBUG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

void DebugLaunch(const char* plugin, int64_t tokens, const void* valid_len,
                 cudaStream_t stream) {
  int32_t valid = -1;
  if (valid_len != nullptr) {
    cudaMemcpyAsync(&valid, valid_len, sizeof(valid), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
  }
  std::fprintf(stderr, "[%s] tokens=%lld valid_len=%d\n", plugin,
               static_cast<long long>(tokens), static_cast<int>(valid));
}

constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";

// The int32 attributes of a plugin, serialized as its fields.
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
  int32_t& operator[](int i) noexcept { return values_[i]; }
  int32_t operator[](int i) const noexcept { return values_[i]; }
  const nvinfer1::PluginFieldCollection* Collection() noexcept {
    return &collection_;
  }
  // Reads the named int32 fields of `fields` (missing ones stay 0).
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

// Shared plugin plumbing: FP32 linear tensors except the optional trailing
// INT32 valid length; two outputs shaped like inputs `first_shape` and
// `second_shape`.
class StatefulPluginBase : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuild,
                           public nvinfer1::IPluginV3OneRuntime {
 public:
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
  int32_t getNbOutputs() const noexcept override { return 2; }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != NumInputs() || num_outputs != 2) {
      return 1;
    }
    for (int i = 0; i < num_inputs; ++i) {
      if (input_types[i] != InputType(i)) return 1;
    }
    output_types[0] = nvinfer1::DataType::kFLOAT;
    output_types[1] = nvinfer1::DataType::kFLOAT;
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != NumInputs() || num_outputs != 2 ||
        position < 0 || position >= num_inputs + num_outputs) {
      return false;
    }
    const nvinfer1::DataType type = position < num_inputs
                                        ? InputType(position)
                                        : nvinfer1::DataType::kFLOAT;
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
    if (inputs == nullptr || outputs == nullptr || num_inputs != NumInputs() ||
        num_shape_inputs != 0 || num_outputs != 2) {
      return 1;
    }
    outputs[0] = inputs[FirstShapeInput()];
    outputs[1] = inputs[SecondShapeInput()];
    return 0;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == NumInputs() && num_outputs == 2
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == NumInputs() && num_outputs == 2
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
                          const nvinfer1::DynamicPluginTensorDesc*,
                          int32_t) const noexcept override {
    return 0;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext*) noexcept override {
    return clone();
  }

 protected:
  virtual int32_t NumInputs() const noexcept = 0;
  virtual bool HasValidLen() const noexcept = 0;
  virtual int32_t FirstShapeInput() const noexcept = 0;
  virtual int32_t SecondShapeInput() const noexcept = 0;
  nvinfer1::DataType InputType(int position) const noexcept {
    return HasValidLen() && position == NumInputs() - 1
               ? nvinfer1::DataType::kINT32
               : nvinfer1::DataType::kFLOAT;
  }
  static int32_t Report(const char* plugin, cudaError_t status) noexcept {
    if (status == cudaSuccess) return 0;
    std::fprintf(stderr, "[%s] CUDA launch failed: %s (%d): %s\n", plugin,
                 cudaGetErrorName(status), static_cast<int>(status),
                 cudaGetErrorString(status));
    return 1;
  }
};

// ---------------------------------------------------------------------------
// gated_delta_update
// ---------------------------------------------------------------------------

constexpr char kGatedDeltaName[] = "LiteRtNvidiaGatedDeltaUpdate";
constexpr std::array<const char*, 5> kGatedDeltaFields = {
    "heads", "key_heads", "key_dim", "value_dim", "has_valid_len"};

class GatedDeltaUpdatePlugin final : public StatefulPluginBase {
 public:
  GatedDeltaUpdatePlugin(int32_t heads, int32_t key_heads, int32_t key_dim,
                         int32_t value_dim, int32_t has_valid_len) noexcept
      : fields_(kGatedDeltaFields) {
    fields_[0] = heads;
    fields_[1] = key_heads;
    fields_[2] = key_dim;
    fields_[3] = value_dim;
    fields_[4] = has_valid_len;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow) GatedDeltaUpdatePlugin(
        fields_[0], fields_[1], fields_[2], fields_[3], fields_[4]);
  }
  const char* getPluginName() const noexcept override {
    return kGatedDeltaName;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    static_cast<void>(workspace);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        input_desc[2].dims.nbDims != 4) {
      return 1;
    }
    if (SkipPluginFamily("gated_delta")) return 0;
    if (DebugLaunches()) {
      DebugLaunch(kGatedDeltaName, input_desc[2].dims.d[2],
                  fields_[4] != 0 ? inputs[6] : nullptr, stream);
    }
    const LiteRtNvidiaGatedDeltaParams params = {
        fields_[0], fields_[1], static_cast<int32_t>(input_desc[2].dims.d[2]),
        fields_[2], fields_[3]};
    return Report(
        kGatedDeltaName,
        LiteRtNvidiaLaunchGatedDeltaUpdate(
            &params, static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]),
            static_cast<const float*>(inputs[3]),
            static_cast<const float*>(inputs[4]),
            static_cast<const float*>(inputs[5]),
            fields_[4] != 0 ? static_cast<const int32_t*>(inputs[6]) : nullptr,
            static_cast<float*>(outputs[0]), static_cast<float*>(outputs[1]),
            stream));
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return fields_.Collection();
  }

 protected:
  int32_t NumInputs() const noexcept override {
    return fields_[4] != 0 ? 7 : 6;
  }
  bool HasValidLen() const noexcept override { return fields_[4] != 0; }
  int32_t FirstShapeInput() const noexcept override { return 2; }   // v
  int32_t SecondShapeInput() const noexcept override { return 5; }  // state

 private:
  IntFields<5> fields_;
};

class GatedDeltaUpdatePluginCreator final
    : public nvinfer1::IPluginCreatorV3One {
 public:
  GatedDeltaUpdatePluginCreator() noexcept : fields_(kGatedDeltaFields) {}

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    IntFields<5> values(kGatedDeltaFields);
    if (!values.Read(fields)) return nullptr;
    return CreateGatedDeltaUpdatePlugin(values[0], values[1], values[2],
                                        values[3], values[4] != 0);
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return fields_.Collection();
  }
  const char* getPluginName() const noexcept override {
    return kGatedDeltaName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  IntFields<5> fields_;
};

// ---------------------------------------------------------------------------
// odml.short_conv_step
// ---------------------------------------------------------------------------

constexpr char kShortConvName[] = "LiteRtNvidiaShortConv";
constexpr std::array<const char*, 4> kShortConvFields = {
    "channels", "taps", "silu", "has_valid_len"};

class ShortConvPlugin final : public StatefulPluginBase {
 public:
  ShortConvPlugin(int32_t channels, int32_t taps, int32_t silu,
                  int32_t has_valid_len) noexcept
      : fields_(kShortConvFields) {
    fields_[0] = channels;
    fields_[1] = taps;
    fields_[2] = silu;
    fields_[3] = has_valid_len;
  }

  nvinfer1::IPluginV3* clone() noexcept override {
    return new (std::nothrow)
        ShortConvPlugin(fields_[0], fields_[1], fields_[2], fields_[3]);
  }
  const char* getPluginName() const noexcept override { return kShortConvName; }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    static_cast<void>(workspace);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
        input_desc[0].dims.nbDims < 2) {
      return 1;
    }
    if (SkipPluginFamily("short_conv")) return 0;
    // Every leading dim of x [.., n, channels] is a token.
    int64_t tokens = 1;
    for (int i = 0; i + 1 < input_desc[0].dims.nbDims; ++i) {
      tokens *= input_desc[0].dims.d[i];
    }
    if (DebugLaunches()) {
      DebugLaunch(kShortConvName, tokens, fields_[3] != 0 ? inputs[3] : nullptr,
                  stream);
    }
    const LiteRtNvidiaShortConvParams params = {
        fields_[0], static_cast<int32_t>(tokens), fields_[1], fields_[2]};
    return Report(
        kShortConvName,
        LiteRtNvidiaLaunchShortConv(
            &params, static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]),
            fields_[3] != 0 ? static_cast<const int32_t*>(inputs[3]) : nullptr,
            static_cast<float*>(outputs[0]), static_cast<float*>(outputs[1]),
            stream));
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return fields_.Collection();
  }

 protected:
  int32_t NumInputs() const noexcept override {
    return fields_[3] != 0 ? 4 : 3;
  }
  bool HasValidLen() const noexcept override { return fields_[3] != 0; }
  int32_t FirstShapeInput() const noexcept override { return 0; }   // x
  int32_t SecondShapeInput() const noexcept override { return 1; }  // state

 private:
  IntFields<4> fields_;
};

class ShortConvPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  ShortConvPluginCreator() noexcept : fields_(kShortConvFields) {}

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    IntFields<4> values(kShortConvFields);
    if (!values.Read(fields)) return nullptr;
    return CreateShortConvPlugin(values[0], values[1], values[2] != 0,
                                 values[3] != 0);
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return fields_.Collection();
  }
  const char* getPluginName() const noexcept override { return kShortConvName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  IntFields<4> fields_;
};

// ---------------------------------------------------------------------------
// The whole gated DeltaNet layer
// ---------------------------------------------------------------------------

constexpr char kGatedDeltaNetName[] = "LiteRtNvidiaGatedDeltaNet";
// The epsilons travel as the bits of their floats.
constexpr std::array<const char*, 14> kGatedDeltaNetFields = {
    "key_heads",   "value_heads", "head_dim",     "taps",         "silu",
    "proj_stride", "qkv_offset",  "z_offset",     "a_offset",     "b_offset",
    "q_eps_bits",  "k_eps_bits",  "out_eps_bits", "has_valid_len"};
using GatedDeltaNetFields = IntFields<14>;

int32_t FloatBits(float value) noexcept {
  int32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsFloat(int32_t bits) noexcept {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

LiteRtNvidiaGatedDeltaNetParams GatedDeltaNetParams(
    const GatedDeltaNetFields& f, int64_t tokens) noexcept {
  LiteRtNvidiaGatedDeltaNetParams p{};
  p.tokens = static_cast<int32_t>(tokens);
  p.key_heads = f[0];
  p.value_heads = f[1];
  p.head_dim = f[2];
  p.taps = f[3];
  p.silu = f[4];
  p.proj_stride = f[5];
  p.qkv_offset = f[6];
  p.z_offset = f[7];
  p.a_offset = f[8];
  p.b_offset = f[9];
  p.q_eps = BitsFloat(f[10]);
  p.k_eps = BitsFloat(f[11]);
  p.out_eps = BitsFloat(f[12]);
  return p;
}

// The rows of proj [.., n, proj_stride]: every dim but the last.
int64_t ProjRows(const nvinfer1::Dims& dims) noexcept {
  int64_t rows = 1;
  for (int i = 0; i + 1 < dims.nbDims; ++i) rows *= dims.d[i];
  return rows;
}

class GatedDeltaNetPlugin final : public nvinfer1::IPluginV3,
                                  public nvinfer1::IPluginV3OneCore,
                                  public nvinfer1::IPluginV3OneBuild,
                                  public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit GatedDeltaNetPlugin(const GatedDeltaNetFields& values) noexcept
      : fields_(kGatedDeltaNetFields) {
    for (int i = 0; i < 14; ++i) fields_[i] = values[i];
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
    return new (std::nothrow) GatedDeltaNetPlugin(fields_);
  }
  const char* getPluginName() const noexcept override {
    return kGatedDeltaNetName;
  }
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
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != NumInputs() || num_outputs != 3) {
      return 1;
    }
    for (int i = 0; i < num_inputs; ++i) {
      if (input_types[i] != InputType(i)) return 1;
    }
    output_types[0] = nvinfer1::DataType::kBF16;
    output_types[1] = nvinfer1::DataType::kFLOAT;
    output_types[2] = nvinfer1::DataType::kFLOAT;
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != NumInputs() || num_outputs != 3 ||
        position < 0 || position >= num_inputs + num_outputs) {
      return false;
    }
    const nvinfer1::DataType type = position < num_inputs ? InputType(position)
                                    : position == num_inputs
                                        ? nvinfer1::DataType::kBF16
                                        : nvinfer1::DataType::kFLOAT;
    return in_out[position].desc.type == type &&
           in_out[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
  }

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs, int32_t num_inputs,
      const nvinfer1::DimsExprs* shape_inputs, int32_t num_shape_inputs,
      nvinfer1::DimsExprs* outputs, int32_t num_outputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override {
    static_cast<void>(shape_inputs);
    if (inputs == nullptr || outputs == nullptr || num_inputs != NumInputs() ||
        num_shape_inputs != 0 || num_outputs != 3 || inputs[0].nbDims < 1) {
      return 1;
    }
    outputs[0] = inputs[0];
    outputs[0].d[outputs[0].nbDims - 1] =
        expr_builder.constant(fields_[1] * fields_[2]);
    outputs[1] = inputs[1];
    outputs[2] = inputs[2];
    return outputs[0].d[outputs[0].nbDims - 1] != nullptr ? 0 : 1;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == NumInputs() && num_outputs == 3
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == NumInputs() && num_outputs == 3
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(outputs);
    static_cast<void>(num_outputs);
    if (inputs == nullptr || num_inputs != NumInputs()) return 0;
    const auto params = GatedDeltaNetParams(fields_, ProjRows(inputs[0].max));
    return LiteRtNvidiaGatedDeltaNetWorkspaceBytes(&params);
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    if (SkipPluginFamily("gated_delta")) return 0;
    const int64_t tokens = ProjRows(input_desc[0].dims);
    const bool has_valid_len = fields_[13] != 0;
    if (DebugLaunches()) {
      DebugLaunch(kGatedDeltaNetName, tokens,
                  has_valid_len ? inputs[4] : nullptr, stream);
    }
    const auto params = GatedDeltaNetParams(fields_, tokens);
    const cudaError_t status = LiteRtNvidiaLaunchGatedDeltaNet(
        &params, inputs[0], static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]),
        static_cast<const float*>(inputs[3]),
        has_valid_len ? static_cast<const int32_t*>(inputs[4]) : nullptr,
        outputs[0], static_cast<float*>(outputs[1]),
        static_cast<float*>(outputs[2]), workspace, stream);
    if (status == cudaSuccess) return 0;
    std::fprintf(stderr, "[%s] CUDA launch failed: %s (%d): %s\n",
                 kGatedDeltaNetName, cudaGetErrorName(status),
                 static_cast<int>(status), cudaGetErrorString(status));
    return 1;
  }

  nvinfer1::IPluginV3* attachToContext(
      nvinfer1::IPluginResourceContext*) noexcept override {
    return clone();
  }
  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return fields_.Collection();
  }

 private:
  int32_t NumInputs() const noexcept { return fields_[13] != 0 ? 5 : 4; }
  nvinfer1::DataType InputType(int position) const noexcept {
    if (position == 0) return nvinfer1::DataType::kBF16;
    return position == 4 ? nvinfer1::DataType::kINT32
                         : nvinfer1::DataType::kFLOAT;
  }

  GatedDeltaNetFields fields_;
};

class GatedDeltaNetPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  GatedDeltaNetPluginCreator() noexcept : fields_(kGatedDeltaNetFields) {}

  nvinfer1::IPluginV3* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fields,
      nvinfer1::TensorRTPhase phase) noexcept override {
    static_cast<void>(name);
    static_cast<void>(phase);
    GatedDeltaNetFields values(kGatedDeltaNetFields);
    if (!values.Read(fields)) return nullptr;
    return CreateGatedDeltaNetPlugin(GatedDeltaNetParams(values, 1),
                                     values[13] != 0);
  }
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return fields_.Collection();
  }
  const char* getPluginName() const noexcept override {
    return kGatedDeltaNetName;
  }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

 private:
  GatedDeltaNetFields fields_;
};

}  // namespace

nvinfer1::IPluginV3* CreateGatedDeltaNetPlugin(
    const LiteRtNvidiaGatedDeltaNetParams& params,
    bool has_valid_len) noexcept {
  LiteRtNvidiaGatedDeltaNetParams check = params;
  check.tokens = 1;
  if (!LiteRtNvidiaGatedDeltaNetSupports(&check)) return nullptr;
  GatedDeltaNetFields values(kGatedDeltaNetFields);
  values[0] = params.key_heads;
  values[1] = params.value_heads;
  values[2] = params.head_dim;
  values[3] = params.taps;
  values[4] = params.silu;
  values[5] = params.proj_stride;
  values[6] = params.qkv_offset;
  values[7] = params.z_offset;
  values[8] = params.a_offset;
  values[9] = params.b_offset;
  values[10] = FloatBits(params.q_eps);
  values[11] = FloatBits(params.k_eps);
  values[12] = FloatBits(params.out_eps);
  values[13] = has_valid_len ? 1 : 0;
  return new (std::nothrow) GatedDeltaNetPlugin(values);
}

nvinfer1::IPluginV3* CreateGatedDeltaUpdatePlugin(int32_t heads,
                                                  int32_t key_heads,
                                                  int32_t key_dim,
                                                  int32_t value_dim,
                                                  bool has_valid_len) noexcept {
  const LiteRtNvidiaGatedDeltaParams params = {heads, key_heads, 1, key_dim,
                                               value_dim};
  if (!LiteRtNvidiaGatedDeltaSupports(&params)) return nullptr;
  return new (std::nothrow) GatedDeltaUpdatePlugin(
      heads, key_heads, key_dim, value_dim, has_valid_len ? 1 : 0);
}

nvinfer1::IPluginV3* CreateShortConvPlugin(int32_t channels, int32_t taps,
                                           bool silu,
                                           bool has_valid_len) noexcept {
  const LiteRtNvidiaShortConvParams params = {channels, 1, taps, silu ? 1 : 0};
  if (!LiteRtNvidiaShortConvSupports(&params)) return nullptr;
  return new (std::nothrow)
      ShortConvPlugin(channels, taps, silu ? 1 : 0, has_valid_len ? 1 : 0);
}

void EnsureGatedDeltaPluginsRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(GatedDeltaUpdatePluginCreator);
REGISTER_TENSORRT_PLUGIN(ShortConvPluginCreator);
REGISTER_TENSORRT_PLUGIN(GatedDeltaNetPluginCreator);

}  // namespace litert::nvidia
