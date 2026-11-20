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

#include "third_party/odml/litert/tensor/internal/graph.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl  // IWYU pragma: keep
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/utils/macros.h"
#include "third_party/odml/litert/tensor/utils/source_location.h"

namespace litert::tensor::graph {

Operation::~Operation() = default;

Tensor ErrorTensor(absl::Status status, source_location op_loc) {
  Tensor t{.group = NewTensorGroup(0, std::move(op_loc)), .index = 0};
  t.group->status = std::move(status);
  return t;
}

Tensor NewTensor(source_location op_loc) {
  return Tensor{.group = NewTensorGroup(1, std::move(op_loc)), .index = 0};
}

Tensor NewTensor(std::shared_ptr<TensorGroup>& group) {
  group->tensor_infos.emplace_back();
  return {.group = group, .index = group->tensor_infos.size() - 1};
}

std::shared_ptr<TensorGroup> NewTensorGroup(size_t count,
                                            source_location op_loc) {
  auto group = std::make_shared<TensorGroup>();
  group->tensor_infos.resize(count);
  group->loc = std::move(op_loc);
  return group;
}

Tensor GetTensor(size_t index, std::shared_ptr<TensorGroup> group) {
  return Tensor{.group = std::move(group), .index = index};
}

absl::StatusOr<std::vector<Tensor>> GetTensors(
    const std::shared_ptr<TensorGroup>& group) {
  if (group == nullptr) {
    return absl::FailedPreconditionError(
        "Tensor group no longer exists or is invalid.");
  }
  if (!group->status.ok()) {
    return group->status;
  }
  std::vector<Tensor> tensors;
  tensors.reserve(group->tensor_infos.size());
  for (size_t i = 0; i < group->tensor_infos.size(); ++i) {
    tensors.push_back(GetTensor(i, group));
  }
  return tensors;
}

absl::StatusOr<std::vector<Tensor>> GetOutputs(const Operation& operation) {
  return GetTensors(operation.outputs_group.lock());
}

absl::StatusOr<TensorInformation&> GetInfo(Tensor& tensor) {
  LRT_TENSOR_RETURN_IF_ERROR(GetStatus(tensor));
  return tensor.group->tensor_infos[tensor.index];
}

absl::StatusOr<const TensorInformation&> GetInfo(const Tensor& tensor) {
  LRT_TENSOR_RETURN_IF_ERROR(GetStatus(tensor));
  return tensor.group->tensor_infos[tensor.index];
}

absl::StatusOr<std::vector<std::weak_ptr<Operation>>&> GetConsumers(
    Tensor& tensor) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  return info.consumers;
}

absl::StatusOr<std::shared_ptr<Operation>&> GetProducer(const Tensor& tensor) {
  LRT_TENSOR_RETURN_IF_ERROR(GetStatus(tensor));
  return tensor.group->producer;
}

absl::StatusOr<const std::string&> GetName(const Tensor& tensor) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  if (!info.name.empty()) {
    return info.name;
  }
  return absl::NotFoundError("This tensor doesn't have a name");
}

absl::Status SetName(Tensor& tensor, std::string name) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  info.name = std::move(name);
  return absl::OkStatus();
}

absl::Status SetBuffer(Tensor& tensor, std::shared_ptr<Buffer> buffer) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  info.buffer = std::move(buffer);
  return absl::OkStatus();
}

absl::StatusOr<Buffer&> GetBuffer(Tensor& tensor) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto& info, GetInfo(tensor));
  if (info.buffer) {
    return *info.buffer;
  }
  return absl::NotFoundError(
      "This tensor doesn't have an associated data buffer");
}

absl::Status GetStatus(const Tensor& tensor) {
  if (!tensor.group) {
    return absl::InvalidArgumentError(
        "Tensor doesn't point to a tensor group.");
  }
  if (!tensor.group->status.ok()) {
    return tensor.group->status;
  }
  if (tensor.index >= tensor.group->tensor_infos.size()) {
    return absl::InvalidArgumentError(
        "Tensor index doesn't exist in its group.");
  }
  return absl::OkStatus();
}

source_location GetLocation(const Tensor& tensor) {
  if (!tensor.group) {
    return source_location();
  }
  return tensor.group->loc;
}

bool OpDebugger::Enabled() {
#ifdef LITERT_OP_DEBUGGER_ENABLED
  return true;
#else
  return false;
#endif
}

void OpDebugger::DebugOp(const Operation& op) {
#ifdef LITERT_OP_DEBUGGER_ENABLED
  absl::string_view op_name = op.GetName();
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  source_location loc;

  auto locked_group = op.outputs_group.lock();
  TensorGroup* group = locked_group.get();

  if (group) {
    loc = group->loc;
    for (size_t i = 0; i < group->tensor_infos.size(); ++i) {
      outputs.push_back(
          GetTensorInfoDebugString(i, group->tensor_infos[i], group->status));
    }
  }
  for (size_t i = 0; i < op.inputs.size(); ++i) {
    auto input_info = GetInfo(op.inputs[i]);
    inputs.push_back(GetTensorInfoDebugString(
        op.inputs[i].index, input_info.value(), input_info.status()));
  }

  ABSL_LOG(INFO) << "[" << loc.file_name() << ":" << loc.line() << "] "
                 << op_name << "(" << absl::StrJoin(inputs, ", ") << ") -> ("
                 << absl::StrJoin(outputs, ", ") << ")";
#endif
}

inline std::string OpDebugger::GetTensorInfoDebugString(
    size_t index, const TensorInformation& info, const absl::Status& status) {
  if (!status.ok()) {
    return absl::StrCat("ErrorTensor(", status.ToString(), ")");
  }
  return absl::StrCat(index, ":", info.name, ":", ToString(info.type), "[",
                      absl::StrJoin(info.shape, ","), "]");
}
}  // namespace litert::tensor::graph
