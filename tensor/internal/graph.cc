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

#include "tensor/internal/graph.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl  // IWYU pragma: keep
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::graph {
namespace {

thread_local std::shared_ptr<Scope>* g_current_scope = nullptr;

std::shared_ptr<Scope>& MutableCurrentScope() {
  if (g_current_scope == nullptr) {
    g_current_scope = new std::shared_ptr<Scope>();
  }
  return *g_current_scope;
}

}  // namespace

std::string Scope::FullPath() const {
  std::vector<absl::string_view> parts;
  for (const Scope* cur = this; cur != nullptr; cur = cur->parent.get()) {
    parts.push_back(cur->name.empty() ? absl::string_view(cur->kind)
                                      : absl::string_view(cur->name));
  }
  std::string path;
  for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
    if (!path.empty()) path.push_back('/');
    path.append(it->data(), it->size());
  }
  return path;
}

const Scope* Scope::FindAncestor(absl::string_view target_kind) const {
  for (const Scope* cur = this; cur != nullptr; cur = cur->parent.get()) {
    if (cur->kind == target_kind) return cur;
  }
  return nullptr;
}

std::shared_ptr<Scope> Scope::FindAncestorShared(
    absl::string_view target_kind) {
  for (std::shared_ptr<Scope> cur = shared_from_this(); cur != nullptr;
       cur = cur->parent) {
    if (cur->kind == target_kind) return cur;
  }
  return nullptr;
}

void Scope::SetAttr(absl::string_view key, ScopeAttrValue value) {
  attrs[key] = std::move(value);
}

std::optional<int64_t> Scope::GetIntAttr(absl::string_view key) const {
  auto it = attrs.find(key);
  if (it == attrs.end()) return std::nullopt;
  if (const auto* val = std::get_if<int64_t>(&it->second)) return *val;
  return std::nullopt;
}

std::optional<double> Scope::GetFloatAttr(absl::string_view key) const {
  auto it = attrs.find(key);
  if (it == attrs.end()) return std::nullopt;
  if (const auto* val = std::get_if<double>(&it->second)) return *val;
  if (const auto* val = std::get_if<int64_t>(&it->second)) {
    return static_cast<double>(*val);
  }
  return std::nullopt;
}

std::optional<absl::string_view> Scope::GetStringAttr(
    absl::string_view key) const {
  auto it = attrs.find(key);
  if (it == attrs.end()) return std::nullopt;
  if (const auto* val = std::get_if<std::string>(&it->second)) return *val;
  return std::nullopt;
}

void Scope::BindTensor(absl::string_view role, const Tensor& tensor) {
  if (!tensor.group || !tensor.group->status.ok()) return;
  BoundaryEntry entry;
  entry.index = tensor.index;
  if (tensor.group->producer == nullptr) {
    entry.strong_group = tensor.group;
  } else {
    entry.weak_group = tensor.group;
  }
  boundary_tensors[role] = std::move(entry);
}

std::optional<Tensor> Scope::GetBoundaryTensor(absl::string_view role) const {
  auto it = boundary_tensors.find(role);
  if (it == boundary_tensors.end()) return std::nullopt;
  return it->second.Lock();
}

std::shared_ptr<Scope> CurrentScope() { return MutableCurrentScope(); }

ScopeGuard::ScopeGuard(absl::string_view kind, absl::string_view name)
    : prev_scope_(MutableCurrentScope()), scope_(std::make_shared<Scope>()) {
  scope_->kind = std::string(kind);
  scope_->name = name.empty() ? std::string(kind) : std::string(name);
  scope_->parent = prev_scope_;
  if (prev_scope_ != nullptr) {
    prev_scope_->children.push_back(scope_);
  }
  MutableCurrentScope() = scope_;
}

ScopeGuard::~ScopeGuard() { MutableCurrentScope() = std::move(prev_scope_); }

ScopeGuard& ScopeGuard::Attr(absl::string_view key, int64_t value) {
  scope_->SetAttr(key, value);
  return *this;
}

ScopeGuard& ScopeGuard::Attr(absl::string_view key, double value) {
  scope_->SetAttr(key, value);
  return *this;
}

ScopeGuard& ScopeGuard::Attr(absl::string_view key, absl::string_view value) {
  scope_->SetAttr(key, std::string(value));
  return *this;
}

ScopeGuard& ScopeGuard::Bind(absl::string_view role, const Tensor& tensor) {
  scope_->BindTensor(role, tensor);
  return *this;
}

Operation::~Operation() = default;

Tensor ErrorTensor(absl::Status status, absl::SourceLocation op_loc) {
  Tensor t{.group = NewTensorGroup(0, std::move(op_loc)), .index = 0};
  t.group->status = std::move(status);
  return t;
}

Tensor NewTensor(absl::SourceLocation op_loc) {
  return Tensor{.group = NewTensorGroup(1, std::move(op_loc)), .index = 0};
}

Tensor NewTensor(std::shared_ptr<TensorGroup>& group) {
  group->tensor_infos.emplace_back();
  return {.group = group, .index = group->tensor_infos.size() - 1};
}

std::shared_ptr<TensorGroup> NewTensorGroup(size_t count,
                                            absl::SourceLocation op_loc) {
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

absl::SourceLocation GetLocation(const Tensor& tensor) {
  if (!tensor.group) {
    return absl::SourceLocation();
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
  absl::SourceLocation loc;

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
