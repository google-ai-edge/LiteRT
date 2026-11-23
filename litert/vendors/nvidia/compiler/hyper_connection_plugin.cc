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

#include "litert/vendors/nvidia/compiler/hyper_connection_plugin.h"

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
#include "litert/vendors/nvidia/trtllm/hyper_connection.h"

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

constexpr char kPluginName[] = "LiteRtNvidiaHcMix";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";

// The plugin's fields in the order of FieldValues.
constexpr int kIntFields = 5;
constexpr int kFloatFields = 2;
constexpr int kWideFields = 9;
constexpr int kFields = kIntFields + kFloatFields + kWideFields;
constexpr std::array<const char*, kFields> kFieldNames = {"streams",
                                                          "hidden",
                                                          "low_rank",
                                                          "int4",
                                                          "extra",
                                                          "down_scale",
                                                          "eps",
                                                          "down_granule",
                                                          "down_weight_offset",
                                                          "down_scale_offset",
                                                          "up_granule",
                                                          "up_weight_offset",
                                                          "up_scale_offset",
                                                          "extra_granule",
                                                          "extra_weight_offset",
                                                          "extra_scale_offset"};

struct FieldValues {
  std::array<int32_t, kIntFields> ints{};
  std::array<float, kFloatFields> floats{};
  std::array<int64_t, kWideFields> wides{};

  static FieldValues Of(const HcMixPluginParams& p) noexcept {
    FieldValues v;
    v.ints = {p.streams, p.hidden, p.low_rank, p.int4, p.extra};
    v.floats = {p.down_scale, p.eps};
    v.wides = {p.down.granule, p.down.weight_offset, p.down.scale_offset,
               p.up.granule,   p.up.weight_offset,   p.up.scale_offset,
               p.more.granule, p.more.weight_offset, p.more.scale_offset};
    return v;
  }

  HcMixPluginParams Params() const noexcept {
    HcMixPluginParams p;
    p.streams = ints[0];
    p.hidden = ints[1];
    p.low_rank = ints[2];
    p.int4 = ints[3];
    p.extra = ints[4];
    p.down_scale = floats[0];
    p.eps = floats[1];
    p.down = {wides[0], wides[1], wides[2]};
    p.up = {wides[3], wides[4], wides[5]};
    p.more = {wides[6], wides[7], wides[8]};
    return p;
  }

  void Describe(std::array<nvinfer1::PluginField, kFields>& fields) noexcept {
    int at = 0;
    for (int i = 0; i < kIntFields; ++i, ++at) {
      fields[at] = {kFieldNames[at], &ints[i],
                    nvinfer1::PluginFieldType::kINT32, 1};
    }
    for (int i = 0; i < kFloatFields; ++i, ++at) {
      fields[at] = {kFieldNames[at], &floats[i],
                    nvinfer1::PluginFieldType::kFLOAT32, 1};
    }
    for (int i = 0; i < kWideFields; ++i, ++at) {
      fields[at] = {kFieldNames[at], &wides[i],
                    nvinfer1::PluginFieldType::kINT64, 1};
    }
  }

  // Reads the named fields of `fields`; the offsets default to "not in a
  // holder".
  bool Read(const nvinfer1::PluginFieldCollection* fields) noexcept {
    if (fields == nullptr || fields->fields == nullptr) return false;
    wides[2] = -1;
    wides[5] = -1;
    wides[8] = -1;
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
        } else if (j < kIntFields + kFloatFields) {
          if (field.type == nvinfer1::PluginFieldType::kFLOAT32) {
            floats[j - kIntFields] = *static_cast<const float*>(field.data);
          }
        } else if (field.type == nvinfer1::PluginFieldType::kINT64) {
          wides[j - kIntFields - kFloatFields] =
              *static_cast<const int64_t*>(field.data);
        }
      }
    }
    return true;
  }
};

// The rows [.., width] of static `dims`, or 0.
int64_t Rows(const nvinfer1::Dims& dims, int64_t width) {
  if (dims.nbDims < 1 || dims.d[dims.nbDims - 1] != width) return 0;
  int64_t rows = 1;
  for (int i = 0; i + 1 < dims.nbDims; ++i) {
    if (dims.d[i] <= 0 ||
        rows > std::numeric_limits<int32_t>::max() / dims.d[i]) {
      return 0;
    }
    rows *= dims.d[i];
  }
  return rows;
}

class HcMixPlugin final : public nvinfer1::IPluginV3,
                          public nvinfer1::IPluginV3OneCore,
                          public nvinfer1::IPluginV3OneBuild,
                          public nvinfer1::IPluginV3OneRuntime {
 public:
  explicit HcMixPlugin(const HcMixPluginParams& params) noexcept
      : params_(params), values_(FieldValues::Of(params)) {
    values_.Describe(fields_);
    collection_ = {kFields, fields_.data()};
  }

  ~HcMixPlugin() override {
    if (state_ != nullptr) cudaFree(state_);
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
    return new (std::nothrow) HcMixPlugin(params_);
  }

  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override {
    return kPluginVersion;
  }
  const char* getPluginNamespace() const noexcept override {
    return kPluginNamespace;
  }

  int32_t getNbOutputs() const noexcept override { return OutputCount(); }

  int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                             int32_t num_outputs,
                             const nvinfer1::DataType* input_types,
                             int32_t num_inputs) const noexcept override {
    if (output_types == nullptr || input_types == nullptr ||
        num_inputs != InputCount() || num_outputs != OutputCount()) {
      return 1;
    }
    for (int32_t i = 0; i < num_inputs; ++i) {
      if (input_types[i] != InputType(i)) return 1;
    }
    for (int32_t i = 0; i < num_outputs; ++i) {
      output_types[i] = nvinfer1::DataType::kBF16;
    }
    return 0;
  }

  bool supportsFormatCombination(
      int32_t position, const nvinfer1::DynamicPluginTensorDesc* in_out,
      int32_t num_inputs, int32_t num_outputs) noexcept override {
    if (in_out == nullptr || num_inputs != InputCount() ||
        num_outputs != OutputCount() || position < 0 ||
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
        num_shape_inputs != 0 || num_outputs != OutputCount() ||
        inputs[0].nbDims < 1) {
      return 1;
    }
    const int last = inputs[0].nbDims - 1;
    outputs[0] = inputs[0];
    outputs[0].d[last] = expr_builder.constant(params_.hidden);
    outputs[1] = inputs[0];
    if (params_.extra > 0) {
      outputs[2] = inputs[0];
      outputs[2].d[last] = expr_builder.constant(params_.extra);
      if (outputs[2].d[last] == nullptr) return 1;
    }
    return outputs[0].d[last] != nullptr ? 0 : 1;
  }

  int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == InputCount() && num_outputs == OutputCount()
               ? 0
               : 1;
  }

  int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                        int32_t num_inputs,
                        const nvinfer1::PluginTensorDesc* outputs,
                        int32_t num_outputs) noexcept override {
    return inputs != nullptr && outputs != nullptr &&
                   num_inputs == InputCount() && num_outputs == OutputCount()
               ? 0
               : 1;
  }

  size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
                          int32_t num_inputs,
                          const nvinfer1::DynamicPluginTensorDesc* outputs,
                          int32_t num_outputs) const noexcept override {
    static_cast<void>(outputs);
    if (inputs == nullptr || num_inputs != InputCount() ||
        num_outputs != OutputCount()) {
      return 0;
    }
    const LiteRtNvidiaHcMixParams p =
        KernelParams(Rows(inputs[0].max, Width()));
    return LiteRtNvidiaHcMixWorkspaceBytes(&p);
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  const void* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    static_cast<void>(output_desc);
    if (input_desc == nullptr || inputs == nullptr || outputs == nullptr) {
      return 1;
    }
    if (SkipPluginFamily("hc_mix")) return 0;
    const LiteRtNvidiaHcMixParams p =
        KernelParams(Rows(input_desc[0].dims, Width()));
    const bool int4 = params_.int4 != 0;
    const int up_weights = int4 ? 4 : 3;
    const size_t values = static_cast<size_t>(params_.low_rank) * Width();
    const size_t weight_bytes = int4 ? values / 2 : values * 2;
    const size_t scale_bytes = values / 32 * sizeof(uint16_t);
    const void* down = Place(params_.down, input_desc[2].dims, inputs[2],
                             params_.down.weight_offset, weight_bytes);
    const void* up =
        Place(params_.up, input_desc[up_weights].dims, inputs[up_weights],
              params_.up.weight_offset, weight_bytes);
    // The extra projection's inputs follow the up projection's.
    const int more_weights = int4 ? 6 : 4;
    const size_t more_values = static_cast<size_t>(params_.extra) * Width();
    const void* more = nullptr;
    const void* more_scales = nullptr;
    if (params_.extra > 0) {
      more = Place(params_.more, input_desc[more_weights].dims,
                   inputs[more_weights], params_.more.weight_offset,
                   int4 ? more_values / 2 : more_values * 2);
      if (int4) {
        more_scales = params_.more.scale_offset >= 0
                          ? Place(params_.more, input_desc[7].dims, inputs[7],
                                  params_.more.scale_offset,
                                  more_values / 32 * sizeof(uint16_t))
                          : inputs[7];
      }
    }
    const void* down_scales = nullptr;
    const void* up_scales = nullptr;
    if (int4) {
      down_scales = params_.down.scale_offset >= 0
                        ? Place(params_.down, input_desc[3].dims, inputs[3],
                                params_.down.scale_offset, scale_bytes)
                        : inputs[3];
      up_scales = params_.up.scale_offset >= 0
                      ? Place(params_.up, input_desc[5].dims, inputs[5],
                              params_.up.scale_offset, scale_bytes)
                      : inputs[5];
    }
    cudaError_t status = cudaErrorInvalidValue;
    // A null weights pointer: the input does not hold this plugin's weights.
    if (down != nullptr && up != nullptr &&
        (!int4 || (down_scales != nullptr && up_scales != nullptr)) &&
        (params_.extra == 0 ||
         (more != nullptr && (!int4 || more_scales != nullptr)))) {
      status = LiteRtNvidiaLaunchHcMixExtra(
          &p, static_cast<const __nv_bfloat16*>(inputs[0]),
          static_cast<const float*>(inputs[1]), down,
          static_cast<const __half*>(down_scales), up,
          static_cast<const __half*>(up_scales), more,
          static_cast<const __half*>(more_scales),
          static_cast<__nv_bfloat16*>(outputs[0]),
          static_cast<__nv_bfloat16*>(outputs[1]),
          params_.extra > 0 ? static_cast<__nv_bfloat16*>(outputs[2]) : nullptr,
          state_, workspace, stream);
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
      nvinfer1::IPluginResourceContext* context) noexcept override {
    static_cast<void>(context);
    auto* attached = new (std::nothrow) HcMixPlugin(params_);
    if (attached != nullptr) attached->AllocateState();
    return attached;
  }

  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept
      override {
    return &collection_;
  }

 private:
  int32_t Width() const noexcept { return params_.streams * params_.hidden; }

  LiteRtNvidiaHcMixParams KernelParams(int64_t rows) const noexcept {
    return {static_cast<int32_t>(rows),
            params_.streams,
            params_.hidden,
            params_.low_rank,
            params_.down_scale,
            params_.eps,
            params_.int4,
            params_.extra};
  }

  // A projection is two inputs with INT4 weights (weights, scales), else one.
  int32_t InputCount() const noexcept {
    return 2 + (params_.int4 != 0 ? 2 : 1) * (params_.extra > 0 ? 3 : 2);
  }

  int32_t OutputCount() const noexcept { return params_.extra > 0 ? 3 : 2; }

  nvinfer1::DataType InputType(int32_t index) const noexcept {
    if (index == 0) return nvinfer1::DataType::kBF16;
    if (index == 1) return nvinfer1::DataType::kFLOAT;
    if (params_.int4 == 0) return nvinfer1::DataType::kBF16;
    const HcMixWeightsPlace& place =
        index < 4 ? params_.down : (index < 6 ? params_.up : params_.more);
    const bool scales = index % 2 == 1;
    if (place.granule != 0 && (!scales || place.scale_offset >= 0)) {
      return nvinfer1::DataType::kINT64;
    }
    return scales ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kINT8;
  }

  // `bytes` at `offset` of a projection's weights input: the input itself
  // without a holder (offset 0), else their place in the holder's segment,
  // which starts at the first granule boundary inside the holder (one
  // granule larger than the segment). Null when they do not fit.
  static const void* Place(const HcMixWeightsPlace& place,
                           const nvinfer1::Dims& dims, const void* input,
                           int64_t offset, size_t bytes) noexcept {
    if (input == nullptr) return nullptr;
    if (place.granule == 0) return input;
    const uint64_t granule = static_cast<uint64_t>(place.granule);
    if (dims.nbDims != 1 || dims.d[0] <= 0) return nullptr;
    const uint64_t holder_bytes =
        static_cast<uint64_t>(dims.d[0]) * sizeof(int64_t);
    if (holder_bytes < granule || offset < 0 ||
        static_cast<uint64_t>(offset) + bytes > holder_bytes - granule) {
      return nullptr;
    }
    const uintptr_t address = reinterpret_cast<uintptr_t>(input);
    const uintptr_t segment = (address + granule - 1) & ~(granule - 1);
    return reinterpret_cast<const uint8_t*>(segment) + offset;
  }

  // The barrier counters of the one launch for this layer of one execution
  // context, zero between its launches; zeroed on a stream of their own (the
  // default stream is not to be used while another stream is captured).
  // Without them (no CUDA context here) the mix runs as two launches.
  void AllocateState() noexcept {
    void* state = nullptr;
    cudaStream_t stream = nullptr;
    const size_t bytes = LiteRtNvidiaHcMixStateBytes();
    if (cudaMalloc(&state, bytes) == cudaSuccess &&
        cudaStreamCreate(&stream) == cudaSuccess &&
        cudaMemsetAsync(state, 0, bytes, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess) {
      state_ = state;
    } else {
      if (state != nullptr) cudaFree(state);
      static_cast<void>(cudaGetLastError());
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  HcMixPluginParams params_;
  FieldValues values_;
  std::array<nvinfer1::PluginField, kFields> fields_;
  nvinfer1::PluginFieldCollection collection_;
  void* state_ = nullptr;
};

class HcMixPluginCreator final : public nvinfer1::IPluginCreatorV3One {
 public:
  HcMixPluginCreator() noexcept {
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
    return CreateHcMixPlugin(values.Params());
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

// A holder's granule is a power of two of at least 16 bytes, and what is in
// it starts on 16 bytes; without one the offsets are unused.
bool ValidPlace(const HcMixWeightsPlace& place, bool int4) {
  if (place.granule == 0) {
    return place.weight_offset == 0 && place.scale_offset == -1;
  }
  return int4 && place.granule >= 16 &&
         (place.granule & (place.granule - 1)) == 0 &&
         place.weight_offset >= 0 && place.weight_offset % 16 == 0 &&
         (place.scale_offset == -1 ||
          (place.scale_offset >= 0 && place.scale_offset % 16 == 0));
}

}  // namespace

nvinfer1::IPluginV3* CreateHcMixPlugin(
    const HcMixPluginParams& params) noexcept {
  if ((params.int4 != 0 && params.int4 != 1) ||
      !ValidPlace(params.down, params.int4 != 0) ||
      !ValidPlace(params.up, params.int4 != 0) ||
      !ValidPlace(params.more, params.int4 != 0) ||
      (params.extra == 0 && params.more.granule != 0) ||
      !(params.down_scale == params.down_scale)) {
    return nullptr;
  }
  // One row: the kernel's limits on the shapes are the same at every row
  // count it takes.
  const LiteRtNvidiaHcMixParams one = {1,
                                       params.streams,
                                       params.hidden,
                                       params.low_rank,
                                       params.down_scale,
                                       params.eps,
                                       params.int4,
                                       params.extra};
  if (!LiteRtNvidiaHcMixSupports(&one)) return nullptr;
  return new (std::nothrow) HcMixPlugin(params);
}

void EnsureHyperConnectionPluginRegistered() noexcept {}

REGISTER_TENSORRT_PLUGIN(HcMixPluginCreator);

}  // namespace litert::nvidia
