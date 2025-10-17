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
#include "litert/c/litert_model.h"
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
  std::vector<int32_t> dims;
  LiteRtElementType element_type;
  std::string name;
  std::optional<OwningBufferRef<uint8_t>> data = std::nullopt;
};

template <LiteRtOpCode OpCode>
struct OpDetails {
 private:
  using FbTypes = FbOpTypes<OpCode>;

 public:
  using OptionsT = typename FbTypes::OptionsT;

  template <typename... Args>
  explicit OpDetails(Args... args)
      : options(OptionsT{{}, std::forward<Args>(args)...}) {}

  TflOptions MakeTflOptions() const {
    TflOptions res;
    res.type = FbTypes::kBuiltinOptions;
    res.Set(OptionsT(options));
    return res;
  }

  TflOpCodePtr MakeTflCode() const {
    auto code = std::make_unique<TflOpCode>();
    code->builtin_code = FbTypes::kBuiltinOperator;
    code->version = 1;
    return code;
  }

 private:
  OptionsT options;
};

// Build a single op model with the given specification.
// NOTE: `inputs` are op inputs. Only non constant inputs will be added
// as subgraph inputs.
template <LiteRtOpCode OpCode, typename... Args>
Expected<LiteRtModelT::Ptr> SingleOpModel(
    const std::vector<TensorDetails>& inputs,
    const std::vector<TensorDetails>& outputs, Args&&... args) {
  using Op = OpDetails<OpCode>;
  Op op_details(std::forward<Args>(args)...);

  LiteRtModelT model;
  std::vector<TflOpCodePtr> tfl_codes;

  auto& sg = model.EmplaceSubgraph();

  auto& op = sg.EmplaceOp();
  {
    op.SetOpCode(OpCode);
    auto options = op_details.MakeTflOptions();
    internal::SetTflOptions(op, op_details.MakeTflOptions());
    internal::SetTflOpCodeInd(op, tfl_codes.size());
    tfl_codes.push_back(op_details.MakeTflCode());
  }

  std::vector<std::string> input_names;
  std::vector<LiteRtTensor> input_tensors;
  for (const auto& input : inputs) {
    auto& tensor = sg.EmplaceTensor();
    tensor.SetType(::MakeRankedTensorType(input.element_type, input.dims));
    tensor.SetName(input.name);

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
    output_names.push_back(output.name);
    sg.Outputs().push_back(&tensor);
    output_tensors.push_back(&tensor);
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
