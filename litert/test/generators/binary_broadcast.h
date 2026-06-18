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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BINARY_BROADCAST_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BINARY_BROADCAST_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/model/ops/simple_binary.h"
#include "litert/core/model/shape_inference_types.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace testing {

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

template <typename Rank1, typename Rank2, typename T, typename OpCode,
          typename Fa = FaC<>>
class BinaryBroadcast : public TestGraph {
  static_assert(std::is_same_v<typename Rank1::value_type, size_t>);
  static constexpr size_t kRank1 = Rank1::value;

  static_assert(std::is_same_v<typename Rank2::value_type, size_t>);
  static constexpr size_t kRank2 = Rank2::value;

  static constexpr size_t kMaxRank = std::max(kRank1, kRank2);

  static_assert(std::is_same_v<typename OpCode::value_type, LiteRtOpCode>);
  static constexpr LiteRtOpCode kOpCode = OpCode::value;

  static_assert(
      std::is_same_v<typename Fa::value_type, tflite::ActivationFunctionType>);
  static constexpr tflite::ActivationFunctionType kFa = Fa::value;

  static constexpr ElementType kElementType = ElementType::Float32;
  static constexpr TensorNames<2> kInputNames = {"lhs", "rhs"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  struct Params {
    std::array<Layout::Dim, kRank1> input1_shape;
    std::array<Layout::Dim, kRank2> input2_shape;
    std::array<Layout::Dim, kMaxRank> output_shape;
  };

 public:
  using Traits = TestLogicTraits<TypeList<T, T>, TypeList<T>, Params>;
  using Ptr = std::unique_ptr<BinaryBroadcast>;

  static constexpr absl::string_view Name() { return "BinaryBroadcast"; }

  template <typename Rng>
  static Expected<BinaryBroadcast::Ptr> Create(Rng& rng) {
    Params params;
    std::uniform_int_distribution<int> dim_dist(2, 10);
    std::bernoulli_distribution flip_dist(0.5);
    bool has_broadcast = false;

    // Fill both shapes with random sizes initially.
    for (size_t i = 0; i < kRank1; ++i) {
      params.input1_shape[i] = dim_dist(rng);
    }
    for (size_t i = 0; i < kRank2; ++i) {
      params.input2_shape[i] = dim_dist(rng);
    }

    // Enforce compatibility scanning from right to left.
    for (size_t i = 1; i <= kMaxRank; ++i) {
      int idx1 = static_cast<int>(kRank1) - static_cast<int>(i);
      int idx2 = static_cast<int>(kRank2) - static_cast<int>(i);

      if (idx1 >= 0 && idx2 >= 0) {
        // Both shapes have this dimension!
        bool force_broadcast = (i == 1) && !has_broadcast;

        if (flip_dist(rng) || force_broadcast) {
          // Make one of them 1.
          if constexpr (kOpCode == kLiteRtOpCodeTflPrelu) {
            params.input2_shape[idx2] = 1;
          } else {
            if (flip_dist(rng)) {
              params.input2_shape[idx2] = 1;
            } else {
              params.input1_shape[idx1] = 1;
            }
          }
          has_broadcast = true;
        } else {
          // Make them equal!
          params.input2_shape[idx2] = params.input1_shape[idx1];
        }
      } else if (idx2 >= 0) {
        // Input 2 has dimension, but Input 1 does not.
        if constexpr (kOpCode == kLiteRtOpCodeTflPrelu) {
          // For Prelu, Alpha (input2) must be broadcastable to Input 1.
          params.input2_shape[idx2] = 1;
        }
      }
      // If only one has the dimension, no constraint needed (other is virtually
      // 1).
    }

    // Leverage shape inference to compute output shape
    std::vector<litert::internal::Dims> input_shapes = {
        {params.input1_shape.begin(), params.input1_shape.end()},
        {params.input2_shape.begin(), params.input2_shape.end()}};
    std::vector<litert::internal::Dims> output_shapes(1);

    LITERT_RETURN_IF_ERROR(litert::internal::InferElementwiseBinary(
        absl::MakeSpan(input_shapes), output_shapes));

    std::copy(output_shapes[0].begin(), output_shapes[0].end(),
              params.output_shape.begin());

    LITERT_ASSIGN_OR_RETURN(auto model, BuildGraph(params));
    return std::make_unique<BinaryBroadcast>(std::move(params),
                                             std::move(model));
  }

  bool HasReference() const override { return true; }

  Expected<VarBuffers> MakeInputs(
      DefaultDevice& device,
      const RandomTensorDataBuilder& data_builder) const override {
    VarBuffers inputs;
    RandomTensorDataBuilder modified_data_builder = data_builder;
    if constexpr (kOpCode == kLiteRtOpCodeTflPow) {
      modified_data_builder.SetFloatRange(0.1f, 5.0f);
    } else if constexpr (kOpCode == kLiteRtOpCodeTflDiv ||
                         kOpCode == kLiteRtOpCodeTflFloorDiv) {
      modified_data_builder.SetFloatRange(0.1f, 10.0f);
    }
    LITERT_ASSIGN_OR_RETURN(auto input1,
                            SimpleBuffer::Create<T>(params_.input1_shape));
    LITERT_RETURN_IF_ERROR(
        (input1.template WriteRandom<T>(modified_data_builder, device)));
    inputs.push_back(std::move(input1));

    LITERT_ASSIGN_OR_RETURN(auto input2,
                            SimpleBuffer::Create<T>(params_.input2_shape));
    LITERT_RETURN_IF_ERROR(
        (input2.template WriteRandom<T>(modified_data_builder, device)));
    inputs.push_back(std::move(input2));

    return inputs;
  }

  Expected<void> Reference(const VarBuffers& inputs,
                           VarBuffers& outputs) const override {
    LITERT_ASSIGN_OR_RETURN(auto ref_inputs,
                            Traits::MakeReferenceInputs(inputs));
    LITERT_ASSIGN_OR_RETURN(auto ref_outputs,
                            Traits::MakeReferenceOutputs(outputs));

    auto [lhs, rhs] = ref_inputs;
    auto [output] = ref_outputs;

    auto op = [](T a, T b) {
      if constexpr (kOpCode == kLiteRtOpCodeTflAdd) {
        return a + b;
      } else if constexpr (kOpCode == kLiteRtOpCodeTflMul) {
        return a * b;
      } else if constexpr (kOpCode == kLiteRtOpCodeTflSub) {
        return a - b;
      } else if constexpr (kOpCode == kLiteRtOpCodeTflDiv) {
        return a / b;
      } else if constexpr (kOpCode == kLiteRtOpCodeTflMaximum) {
        return std::max(a, b);
      } else if constexpr (kOpCode == kLiteRtOpCodeTflMinimum) {
        return std::min(a, b);
      } else if constexpr (kOpCode == kLiteRtOpCodeTflSquaredDifference) {
        return (a - b) * (a - b);
      } else if constexpr (kOpCode == kLiteRtOpCodeTflPow) {
        return std::pow(a, b);
      } else if constexpr (kOpCode == kLiteRtOpCodeTflFloorDiv) {
        return std::floor(a / b);
      } else if constexpr (kOpCode == kLiteRtOpCodeTflPrelu) {
        return a > 0 ? a : a * b;
      } else {
        static_assert(kOpCode == kLiteRtOpCodeTflAdd ||
                          kOpCode == kLiteRtOpCodeTflMul ||
                          kOpCode == kLiteRtOpCodeTflSub ||
                          kOpCode == kLiteRtOpCodeTflDiv ||
                          kOpCode == kLiteRtOpCodeTflMaximum ||
                          kOpCode == kLiteRtOpCodeTflMinimum ||
                          kOpCode == kLiteRtOpCodeTflSquaredDifference ||
                          kOpCode == kLiteRtOpCodeTflPow ||
                          kOpCode == kLiteRtOpCodeTflFloorDiv ||
                          kOpCode == kLiteRtOpCodeTflPrelu,
                      "Unsupported opcode in BinaryBroadcast");
        return T();
      }
    };

    litert::internal::ReferenceBinaryGeneric(
        lhs.data.data(),
        reinterpret_cast<const int32_t*>(params_.input1_shape.data()), kRank1,
        rhs.data.data(),
        reinterpret_cast<const int32_t*>(params_.input2_shape.data()), kRank2,
        output.data.data(),
        reinterpret_cast<const int32_t*>(params_.output_shape.data()), kMaxRank,
        op);

    size_t num_elements = 1;
    for (size_t i = 0; i < kMaxRank; ++i)
      num_elements *= params_.output_shape[i];

    litert::internal::ApplyActivation(output.data.data(), num_elements, kFa);

    return {};
  }

  BinaryBroadcast(Params params, LiteRtModelT::Ptr model)
      : TestGraph(std::move(model)), params_(std::move(params)) {}

 private:
  static Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    LiteRtModelT model;

    constexpr bool supports_activation =
        (kOpCode == kLiteRtOpCodeTflAdd || kOpCode == kLiteRtOpCodeTflMul ||
         kOpCode == kLiteRtOpCodeTflSub || kOpCode == kLiteRtOpCodeTflDiv);

    if constexpr (!supports_activation) {
      static_assert(kFa == tflite::ActivationFunctionType_NONE,
                    "Op does not support fused activations");
    }

    std::vector<TflOpCodePtr> tfl_codes;
    auto& sg = model.EmplaceSubgraph();
    auto& op = sg.EmplaceOp();

    op.SetOpCode(kOpCode);

    tflite::BuiltinOperator builtin_op;
    TflOptions tfl_opts;

    if constexpr (kOpCode == kLiteRtOpCodeTflAdd) {
      builtin_op = tflite::BuiltinOperator_ADD;
      auto options = std::make_unique<tflite::AddOptionsT>();
      options->fused_activation_function = kFa;
      tfl_opts.type = tflite::BuiltinOptions_AddOptions;
      tfl_opts.value = options.release();
    } else if constexpr (kOpCode == kLiteRtOpCodeTflMul) {
      builtin_op = tflite::BuiltinOperator_MUL;
      auto options = std::make_unique<tflite::MulOptionsT>();
      options->fused_activation_function = kFa;
      tfl_opts.type = tflite::BuiltinOptions_MulOptions;
      tfl_opts.value = options.release();
    } else if constexpr (kOpCode == kLiteRtOpCodeTflSub) {
      builtin_op = tflite::BuiltinOperator_SUB;
      auto options = std::make_unique<tflite::SubOptionsT>();
      options->fused_activation_function = kFa;
      tfl_opts.type = tflite::BuiltinOptions_SubOptions;
      tfl_opts.value = options.release();
    } else if constexpr (kOpCode == kLiteRtOpCodeTflDiv) {
      builtin_op = tflite::BuiltinOperator_DIV;
      auto options = std::make_unique<tflite::DivOptionsT>();
      options->fused_activation_function = kFa;
      tfl_opts.type = tflite::BuiltinOptions_DivOptions;
      tfl_opts.value = options.release();
    } else if constexpr (kOpCode == kLiteRtOpCodeTflMaximum) {
      builtin_op = tflite::BuiltinOperator_MAXIMUM;
      tfl_opts.type = tflite::BuiltinOptions_NONE;
    } else if constexpr (kOpCode == kLiteRtOpCodeTflMinimum) {
      builtin_op = tflite::BuiltinOperator_MINIMUM;
      tfl_opts.type = tflite::BuiltinOptions_NONE;
    } else if constexpr (kOpCode == kLiteRtOpCodeTflSquaredDifference) {
      builtin_op = tflite::BuiltinOperator_SQUARED_DIFFERENCE;
      tfl_opts.type = tflite::BuiltinOptions_NONE;
    } else if constexpr (kOpCode == kLiteRtOpCodeTflPow) {
      builtin_op = tflite::BuiltinOperator_POW;
      tfl_opts.type = tflite::BuiltinOptions_NONE;
    } else if constexpr (kOpCode == kLiteRtOpCodeTflFloorDiv) {
      builtin_op = tflite::BuiltinOperator_FLOOR_DIV;
      tfl_opts.type = tflite::BuiltinOptions_NONE;
    } else if constexpr (kOpCode == kLiteRtOpCodeTflPrelu) {
      builtin_op = tflite::BuiltinOperator_PRELU;
      tfl_opts.type = tflite::BuiltinOptions_NONE;
    }

    litert::internal::SetTflOptions(op, std::move(tfl_opts));
    litert::internal::SetTflOpCodeInd(op, 0);

    auto op_code = std::make_unique<tflite::OperatorCodeT>();
    op_code->builtin_code = builtin_op;
    tfl_codes.push_back(std::move(op_code));

    auto& in1_tensor = sg.EmplaceTensor();
    in1_tensor.SetType(::MakeRankedTensorType(LiteRtElementType(kElementType),
                                              params.input1_shape));
    in1_tensor.SetName(std::string(kInputNames[0]));
    sg.Inputs().push_back(&in1_tensor);
    AttachInput(&in1_tensor, op);

    auto& in2_tensor = sg.EmplaceTensor();
    in2_tensor.SetType(::MakeRankedTensorType(LiteRtElementType(kElementType),
                                              params.input2_shape));
    in2_tensor.SetName(std::string(kInputNames[1]));
    sg.Inputs().push_back(&in2_tensor);
    AttachInput(&in2_tensor, op);

    auto& out_tensor = sg.EmplaceTensor();
    out_tensor.SetType(::MakeRankedTensorType(LiteRtElementType(kElementType),
                                              params.output_shape));
    out_tensor.SetName(std::string(kOutputNames[0]));
    sg.Outputs().push_back(&out_tensor);
    AttachOutput(&out_tensor, op);

    std::vector<std::string> input_names = {std::string(kInputNames[0]),
                                            std::string(kInputNames[1])};
    std::vector<LiteRtTensor> input_tensors = {&in1_tensor, &in2_tensor};
    std::vector<std::string> output_names = {std::string(kOutputNames[0])};
    std::vector<LiteRtTensor> output_tensors = {&out_tensor};

    model.EmplaceSignature(&sg, std::move(input_names),
                           std::move(input_tensors), std::move(output_names),
                           std::move(output_tensors), "default");

    SetTflOpCodes(model, std::move(tfl_codes));

    LITERT_ASSIGN_OR_RETURN(auto serialized, SerializeModel(std::move(model)));
    return LoadModelFromBuffer(std::move(serialized));
  }

  Params params_;
};

}  // namespace testing
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TEST_GENERATORS_BINARY_BROADCAST_H_
