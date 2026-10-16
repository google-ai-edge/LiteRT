/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_ARITHMETIC_HELPERS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_ARITHMETIC_HELPERS_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/internal/graph.h"
#include "third_party/odml/litert/tensor/internal/shape.h"
#include "third_party/odml/litert/tensor/internal/utils.h"
#include "third_party/odml/litert/tensor/tensor.h"
#include "third_party/odml/litert/tensor/utils/macros.h"
#include "third_party/odml/litert/tensor/utils/source_location.h"

namespace litert::tensor {

template <class T>
struct IsTensor : std::false_type {};

template <class... Mixins>
struct IsTensor<Tensor<Mixins...>> : std::true_type {};

template <>
struct IsTensor<TensorHandle> : std::true_type {};

// Adds the given tensors as inputs of an op.
template <class Op, class... Tensors,
          class = std::enable_if_t<std::conjunction_v<IsTensor<Tensors>...>>>
void AddInputs(std::shared_ptr<Op>& operation, Tensors&... tensors) {
  static_assert(std::is_base_of_v<graph::Operation, Op>,
                "The operation is not derived from graph::Operation.");
  operation->inputs.reserve(size(operation->inputs) + sizeof...(tensors));
  auto SetOpInput = [&operation](TensorHandle& t) {
    GetConsumers(t.GetRaw())->push_back(operation);
    operation->inputs.push_back(t.GetRaw());
  };
  (SetOpInput(tensors), ...);
}

template <class Op, class TensorSpecialization,
          class = std::enable_if_t<IsTensor<TensorSpecialization>::value>>
void AddInputs(std::shared_ptr<Op>& operation,
               absl::Span<TensorSpecialization> tensors) {
  operation->inputs.reserve(size(operation->inputs) + tensors.size());
  for (TensorSpecialization& t : tensors) {
    GetConsumers(t.GetRaw())->push_back(operation);
    operation->inputs.push_back(t.GetRaw());
  }
}

// Creates a new output tensor for the given op.
//
// `op_loc` should be the location of the creating operation function call.
template <class Op>
TensorHandle AddOutput(std::shared_ptr<Op>& operation, source_location op_loc) {
  static_assert(std::is_base_of_v<graph::Operation, Op>,
                "The operation is not derived from graph::Operation.");
  std::shared_ptr<graph::TensorGroup> group = operation->outputs_group.lock();
  if (!group) {
    group = graph::NewTensorGroup(1, std::move(op_loc));
    operation->outputs_group = group;
    group->producer = operation;
  } else {
    group->tensor_infos.emplace_back();
  }
  return TensorHandle(graph::GetTensor(group->tensor_infos.size() - 1, group));
}

template <class... Tensors>
absl::Status CheckTensors(Tensors... tensors) {
  absl::Status status = absl::OkStatus();
  (status.Update(tensors.GetStatus()), ...);
  return status;
}

template <class Op, class... Tensors>
TensorHandle ElementwiseOp(source_location loc, TensorHandle a,
                           Tensors&... tensors) {
  LRT_TENSOR_RETURN_IF_ERROR(CheckTensors(a, tensors...));
  auto operation = std::make_shared<Op>();
  AddInputs(operation, a, tensors...);
  TensorHandle output = AddOutput(operation, loc);
  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& a_info,
                              GetInfo(a.GetRaw()));
  LRT_TENSOR_ASSIGN_OR_RETURN(graph::TensorInformation & o_info,
                              GetInfo(output.GetRaw()));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      o_info.shape, BroadcastShapes({a_info.shape, tensors.GetShape()...}));
  o_info.type = IsComparisonOp(*operation) ? Type::kBOOL : a_info.type;

  const bool all_same_types = ((a.GetType() == tensors.GetType()) && ...);
  if (!all_same_types) {
    return TensorHandle(
        absl::InvalidArgumentError(absl::StrCat(
            loc.file_name(), ":", loc.line(), operation->GetName(),
            ": all tensors in an elementwise operation must have the "
            "same type. Got (",
            absl::StrJoin({a.GetType(), tensors.GetType()...}, ", "), ")")),
        loc);
  }

  graph::OpDebugger::DebugOp(*operation);
  return output;
}

}  // namespace litert::tensor

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_INTERNAL_ARITHMETIC_HELPERS_H_
