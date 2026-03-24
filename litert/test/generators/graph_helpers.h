// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_GRAPH_HELPERS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_GRAPH_HELPERS_H_

// Helper functions for putting together small litert models.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/generators/common.h"

namespace litert::testing {

using ::litert::internal::AttachInput;
using ::litert::internal::AttachOutput;
using ::litert::internal::LoadModelFromBuffer;
using ::litert::internal::SerializeModel;
using ::litert::internal::SetTflOpCodeInd;
using ::litert::internal::SetTflOpCodes;
using ::litert::internal::SetTflOptions;
using ::litert::internal::TflOpCode;
using ::litert::internal::TflOpCodePtr;
using ::litert::internal::TflOptions;

struct TensorDetails {
  struct QuantizationDetails {
    LiteRtQuantizationTypeId type = kLiteRtQuantizationNone;
    float scale = 0.0f;
    int64_t zero_point = 0;
    int32_t quantized_dimension = 0;
    std::vector<float> scales;
    std::vector<int64_t> zero_points;

    static QuantizationDetails PerTensor(float scale, int64_t zero_point) {
      QuantizationDetails res;
      res.type = kLiteRtQuantizationPerTensor;
      res.scale = scale;
      res.zero_point = zero_point;
      return res;
    }

    static QuantizationDetails PerChannel(int32_t quantized_dimension,
                                          std::vector<float> scales,
                                          std::vector<int64_t> zero_points) {
      QuantizationDetails res;
      res.type = kLiteRtQuantizationPerChannel;
      res.quantized_dimension = quantized_dimension;
      res.scales = std::move(scales);
      res.zero_points = std::move(zero_points);
      return res;
    }
  };

  std::vector<int32_t> dims;
  LiteRtElementType element_type;
  std::string name;
  std::optional<OwningBufferRef<uint8_t>> data = std::nullopt;
  std::optional<QuantizationDetails> quantization = std::nullopt;
};

inline void SetTensorQuantization(
    LiteRtTensorT& tensor,
    const std::optional<TensorDetails::QuantizationDetails>& quantization) {
  if (!quantization.has_value() ||
      quantization->type == kLiteRtQuantizationNone) {
    return;
  }
  if (quantization->type == kLiteRtQuantizationPerTensor) {
    tensor.SetQarams(
        MakePerTensorQuantization(quantization->scale, quantization->zero_point));
    return;
  }
  tensor.SetQarams(MakePerChannelQuantization(
      quantization->scales, quantization->zero_points,
      quantization->quantized_dimension, tensor));
}

template <LiteRtOpCode OpCode>
inline int GetBuiltinOpVersion(
    const std::vector<TensorDetails>& inputs,
    const std::vector<TensorDetails>& outputs,
    const typename FbOpTypes<OpCode>::OptionsT& options) {
  (void)inputs;
  (void)outputs;
  (void)options;
  return 1;
}

template <>
inline int GetBuiltinOpVersion<kLiteRtOpCodeTflFullyConnected>(
    const std::vector<TensorDetails>& inputs,
    const std::vector<TensorDetails>& outputs,
    const tflite::FullyConnectedOptionsT& options) {
  if (inputs.size() < 2 || outputs.empty()) {
    return 1;
  }

  const auto input_type = inputs[0].element_type;
  const auto filter_type = inputs[1].element_type;
  const auto output_type = outputs[0].element_type;
  const bool is_per_channel_quantized =
      inputs[1].quantization.has_value() &&
      inputs[1].quantization->type == kLiteRtQuantizationPerChannel;

  if (filter_type == kLiteRtElementTypeInt2) {
    return 14;
  }
  if (input_type == kLiteRtElementTypeInt16 &&
      filter_type == kLiteRtElementTypeInt4 &&
      output_type == kLiteRtElementTypeInt16) {
    return 13;
  }
  if (input_type == kLiteRtElementTypeFloat32 &&
      filter_type == kLiteRtElementTypeInt8 &&
      output_type == kLiteRtElementTypeFloat32 && is_per_channel_quantized) {
    return 12;
  }
  if (input_type == kLiteRtElementTypeInt16 &&
      filter_type == kLiteRtElementTypeInt8 &&
      output_type == kLiteRtElementTypeInt16 &&
      static_cast<int>(options.quantized_bias_type) != 0) {
    return 11;
  }
  if (input_type == kLiteRtElementTypeInt16 &&
      filter_type == kLiteRtElementTypeInt16 &&
      output_type == kLiteRtElementTypeInt16) {
    return 7;
  }
  if (inputs.size() == 2) {
    return 6;
  }
  if (options.keep_num_dims) {
    return 5;
  }
  if (input_type == kLiteRtElementTypeInt8 &&
      filter_type == kLiteRtElementTypeInt8 &&
      output_type == kLiteRtElementTypeInt8) {
    return 4;
  }
  if (input_type == kLiteRtElementTypeInt8 &&
      filter_type == kLiteRtElementTypeInt4 &&
      output_type == kLiteRtElementTypeInt8) {
    return 10;
  }
  if (input_type == kLiteRtElementTypeFloat32 &&
      filter_type == kLiteRtElementTypeInt8 &&
      output_type == kLiteRtElementTypeFloat32) {
    if (options.asymmetric_quantize_inputs) {
      return 9;
    }
    return 3;
  }
  if (options.weights_format ==
      tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8) {
    return 2;
  }
  return 1;
}

template <LiteRtOpCode OpCode>
struct OpDetails {
 private:
  using FbTypes = FbOpTypes<OpCode>;

 public:
  using OptionsT = typename FbTypes::OptionsT;

  template <typename... Args>
  explicit OpDetails(Args... args)
      : options(OptionsT{{}, std::forward<Args>(args)...}) {}

  OpDetails() = default;

  TflOptions MakeTflOptions() const {
    TflOptions res;
    res.type = FbTypes::kBuiltinOptions;
    if constexpr (FbTypes::kHasOptions) {
      res.Set(OptionsT(options));
    }
    return res;
  }

  int Version(const std::vector<TensorDetails>& inputs,
              const std::vector<TensorDetails>& outputs) const {
    return GetBuiltinOpVersion<OpCode>(inputs, outputs, options);
  }

  TflOpCodePtr MakeTflCode(int version) const {
    auto code = std::make_unique<TflOpCode>();
    code->builtin_code = FbTypes::kBuiltinOperator;
    code->version = version;
    return code;
  }

 private:
  OptionsT options;
};

// Build a single op model with the given specification.
// NOTE: `inputs` are op inputs. Only non constant inputs will be added
// as subgraph inputs.
template <LiteRtOpCode OpCode, typename... Args>
Expected<LiteRtModelT::Ptr> SingleOpModelWithInternalOutputs(
    const std::vector<TensorDetails>& inputs,
    const std::vector<TensorDetails>& outputs,
    const std::vector<TensorDetails>& internal_outputs, Args&&... args);

template <LiteRtOpCode OpCode, typename... Args>
Expected<LiteRtModelT::Ptr> SingleOpModel(
    const std::vector<TensorDetails>& inputs,
    const std::vector<TensorDetails>& outputs, Args&&... args) {
  return SingleOpModelWithInternalOutputs<OpCode>(inputs, outputs, {},
                                                  std::forward<Args>(args)...);
}

template <LiteRtOpCode OpCode, typename... Args>
Expected<LiteRtModelT::Ptr> SingleOpModelWithInternalOutputs(
    const std::vector<TensorDetails>& inputs,
    const std::vector<TensorDetails>& outputs,
    const std::vector<TensorDetails>& internal_outputs, Args&&... args) {
  using Op = OpDetails<OpCode>;
  Op op_details(std::forward<Args>(args)...);

  LiteRtModelT model;
  std::vector<TflOpCodePtr> tfl_codes;

  auto& sg = model.EmplaceSubgraph();

  auto& op = sg.EmplaceOp();
  {
    const auto op_version = op_details.Version(inputs, outputs);
    op.SetOpCode(OpCode);
    auto options = op_details.MakeTflOptions();
    SetTflOptions(op, op_details.MakeTflOptions());
    SetTflOpCodeInd(op, tfl_codes.size());
    tfl_codes.push_back(op_details.MakeTflCode(op_version));
  }

  std::vector<std::string> input_names;
  std::vector<LiteRtTensor> input_tensors;
  for (const auto& input : inputs) {
    auto& tensor = sg.EmplaceTensor();
    tensor.SetType(::MakeRankedTensorType(input.element_type, input.dims));
    tensor.SetName(input.name);
    SetTensorQuantization(tensor, input.quantization);

    if (input.data) {
      ::SetWeightsFromUnownedBuffer(tensor.Weights(), *input.data);
    } else {
      sg.Inputs().push_back(&tensor);
      input_names.push_back(input.name);
      input_tensors.push_back(&tensor);
    }

    AttachInput(&tensor, op);
  }

  std::vector<std::string> output_names;
  std::vector<LiteRtTensor> output_tensors;
  for (const auto& output : outputs) {
    auto& tensor = sg.EmplaceTensor();
    tensor.SetType(::MakeRankedTensorType(output.element_type, output.dims));
    tensor.SetName(output.name);
    SetTensorQuantization(tensor, output.quantization);
    output_names.push_back(output.name);
    sg.Outputs().push_back(&tensor);
    output_tensors.push_back(&tensor);
    AttachOutput(&tensor, op);
  }

  for (const auto& output : internal_outputs) {
    auto& tensor = sg.EmplaceTensor();
    tensor.SetType(::MakeRankedTensorType(output.element_type, output.dims));
    tensor.SetName(output.name);
    SetTensorQuantization(tensor, output.quantization);
    AttachOutput(&tensor, op);
  }

  model.EmplaceSignature(&sg, std::move(input_names), std::move(input_tensors),
                         std::move(output_names), std::move(output_tensors),
                         "default");
  SetTflOpCodes(model, std::move(tfl_codes));
  LITERT_ASSIGN_OR_RETURN(auto serialized, SerializeModel(std::move(model)));
  return LoadModelFromBuffer(std::move(serialized));
}

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_GRAPH_HELPERS_H_
