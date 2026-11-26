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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/source_location.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/utils/macros.h"

namespace litert::tensor::graph {
namespace {

constexpr float kMinFp16Normal = 6.103515625e-5f;
constexpr float kMaxFp16Normal = 65504.0f;

// Heap-allocated `thread_local` pointer avoids non-trivial TLS destructor
// ordering issues across shared libraries and thread teardown.
thread_local std::shared_ptr<Scope>* g_current_scope = nullptr;

std::shared_ptr<Scope>& MutableCurrentScope() {
  if (g_current_scope == nullptr) {
    g_current_scope = new std::shared_ptr<Scope>();
  }
  return *g_current_scope;
}

}  // namespace

BlockwiseQuantization::CachedBf16Scales::CachedBf16Scales(
    const CachedBf16Scales&) {}

BlockwiseQuantization::CachedBf16Scales&
BlockwiseQuantization::CachedBf16Scales::operator=(
    const CachedBf16Scales& other) {
  if (this != &other) {
    absl::MutexLock lock(mu);
    scales.reset();
    cpu_fallback.reset();
  }
  return *this;
}

BlockwiseQuantization::CachedBf16Scales::CachedBf16Scales(
    CachedBf16Scales&&) noexcept {}

BlockwiseQuantization::CachedBf16Scales&
BlockwiseQuantization::CachedBf16Scales::operator=(
    CachedBf16Scales&& other) noexcept {
  if (this != &other) {
    absl::MutexLock lock(mu);
    scales.reset();
    cpu_fallback.reset();
  }
  return *this;
}

std::shared_ptr<const std::vector<bf16_t>>
BlockwiseQuantization::GetOrCreateBf16Scales() const {
  {
    absl::MutexLock lock(bf16_cache_.mu);
    if (bf16_cache_.scales != nullptr &&
        bf16_cache_.scales->size() == scales.size()) {
      return bf16_cache_.scales;
    }
  }
  auto built = std::make_shared<std::vector<bf16_t>>(scales.size());
  bf16_t* dst = built->data();
  const float* src = scales.data();
  const size_t n = scales.size();
  for (size_t i = 0; i < n; ++i) {
    dst[i] = bf16_t(src[i]);
  }
  absl::MutexLock lock(bf16_cache_.mu);
  if (bf16_cache_.scales == nullptr ||
      bf16_cache_.scales->size() != scales.size()) {
    bf16_cache_.scales = std::move(built);
  }
  return bf16_cache_.scales;
}

bool BlockwiseQuantization::HasCachedBf16Scales() const {
  absl::MutexLock lock(bf16_cache_.mu);
  return bf16_cache_.scales != nullptr;
}

void BlockwiseQuantization::InvalidateBf16ScalesCache() {
  absl::MutexLock lock(bf16_cache_.mu);
  bf16_cache_.scales.reset();
  bf16_cache_.cpu_fallback.reset();
}

absl::StatusOr<
    std::shared_ptr<const BlockwiseQuantization::CpuSymmetricFallback>>
BlockwiseQuantization::GetOrCreateCpuSymmetricFallback(Type type, int rows,
                                                       int cols,
                                                       const void* raw_data,
                                                       size_t raw_bytes) const {
  if (rows <= 0 || cols <= 0 || block_size <= 0 || cols % block_size != 0 ||
      quantized_dimension != 0 || raw_data == nullptr) {
    return absl::InvalidArgumentError(
        "Invalid BlockwiseQuantization geometry for CPU symmetric fallback.");
  }
  const size_t blocks_per_row = static_cast<size_t>(cols / block_size);
  const size_t total_blocks = static_cast<size_t>(rows) * blocks_per_row;
  const size_t total_elements =
      static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (scales.size() < total_blocks) {
    return absl::InvalidArgumentError(
        "BlockwiseQuantization scale count smaller than block count.");
  }
  if (type == Type::kI8) {
    if (raw_bytes < total_elements) {
      return absl::InvalidArgumentError(
          "INT8 blockwise weight buffer smaller than element count.");
    }
  } else if (type == Type::kI4) {
    if (cols % 2 != 0 || block_size % 2 != 0 ||
        raw_bytes < total_elements / 2) {
      return absl::InvalidArgumentError(
          "INT4 blockwise weight buffer has invalid size or odd block_size.");
    }
  } else {
    return absl::UnimplementedError(
        "Unsupported element type for CPU symmetric blockwise fallback.");
  }

  {
    absl::MutexLock lock(bf16_cache_.mu);
    if (bf16_cache_.cpu_fallback != nullptr &&
        bf16_cache_.cpu_fallback->type == type &&
        bf16_cache_.cpu_fallback->rows == rows &&
        bf16_cache_.cpu_fallback->cols == cols) {
      return bf16_cache_.cpu_fallback;
    }
  }

  auto built = std::make_shared<CpuSymmetricFallback>();
  built->type = type;
  built->rows = rows;
  built->cols = cols;
  if (type == Type::kI8) {
    std::shared_ptr<OwningCpuBuffer> out_buf =
        OwningCpuBuffer::Allocate<Type::kI8>(total_elements);
    if (out_buf == nullptr) {
      return absl::ResourceExhaustedError(
          "Failed to allocate CPU symmetric INT8 fallback buffer.");
    }
    const int8_t* src = static_cast<const int8_t*>(raw_data);
    int8_t* dst = out_buf->Span<int8_t>().data();
    built->scales.resize(rows);
    std::vector<float> row_vals(cols);
    bool has_non_finite = false;
    LRT_TENSOR_RETURN_IF_ERROR(
        zero_points.Visit(total_blocks, [&](auto zp_view) {
          for (int r = 0; r < rows; ++r) {
            const int8_t* src_row = src + static_cast<size_t>(r) * cols;
            float amax = 0.0f;
            for (size_t b = 0; b < blocks_per_row; ++b) {
              const size_t blk_idx =
                  static_cast<size_t>(r) * blocks_per_row + b;
              const float s = scales[blk_idx];
              const float zp = static_cast<float>(zp_view[blk_idx]);
              if (!std::isfinite(s) || !std::isfinite(zp)) {
                has_non_finite = true;
                return;
              }
              const size_t c_base = b * static_cast<size_t>(block_size);
              for (int j = 0; j < block_size; ++j) {
                const float v =
                    (static_cast<float>(src_row[c_base + j]) - zp) * s;
                if (!std::isfinite(v)) {
                  has_non_finite = true;
                  return;
                }
                row_vals[c_base + j] = v;
                amax = std::max(amax, std::abs(v));
              }
            }
            const float row_scale =
                std::max(amax / 127.0f, std::numeric_limits<float>::min());
            built->scales[r] = row_scale;
            const float inv_scale = 1.0f / row_scale;
            int8_t* dst_row = dst + static_cast<size_t>(r) * cols;
            for (int c = 0; c < cols; ++c) {
              const float scaled = row_vals[c] * inv_scale;
              dst_row[c] = static_cast<int8_t>(
                  std::isfinite(scaled)
                      ? std::clamp(std::rint(scaled), -127.0f, 127.0f)
                      : 0.0f);
            }
          }
        }));
    if (has_non_finite) {
      return absl::InvalidArgumentError(
          "Non-finite scale or zero_point in blockwise INT8 tensor.");
    }
    built->data = std::move(out_buf);
  } else {
    std::shared_ptr<OwningCpuBuffer> out_buf =
        OwningCpuBuffer::Allocate<Type::kI4>(total_elements);
    if (out_buf == nullptr) {
      return absl::ResourceExhaustedError(
          "Failed to allocate CPU symmetric INT4 fallback buffer.");
    }
    const int4_t* src_packed = static_cast<const int4_t*>(raw_data);
    int4_t* dst_packed = out_buf->Span<int4_t>().data();
    auto bf16_vec = std::make_shared<std::vector<bf16_t>>(total_blocks);
    const size_t pairs_per_block = static_cast<size_t>(block_size / 2);
    std::vector<float> blk_vals(block_size);
    bool has_non_finite = false;
    LRT_TENSOR_RETURN_IF_ERROR(
        zero_points.Visit(total_blocks, [&](auto zp_view) {
          for (size_t b = 0; b < total_blocks; ++b) {
            const float s = scales[b];
            const float zp = static_cast<float>(zp_view[b]);
            if (!std::isfinite(s) || !std::isfinite(zp)) {
              has_non_finite = true;
              return;
            }
            const int4_t* sp = src_packed + b * pairs_per_block;
            int4_t* dp = dst_packed + b * pairs_per_block;
            if (zp == 0.0f && s >= kMinFp16Normal && s <= kMaxFp16Normal) {
              std::memcpy(dp, sp, pairs_per_block * sizeof(int4_t));
              (*bf16_vec)[b] = bf16_t(s);
              continue;
            }
            float min_val = 0.0f;
            float max_val = 0.0f;
            for (int j = 0; j < block_size; j += 2) {
              const float v0 = (static_cast<float>(sp[j / 2].a) - zp) * s;
              const float v1 = (static_cast<float>(sp[j / 2].b) - zp) * s;
              if (!std::isfinite(v0) || !std::isfinite(v1)) {
                has_non_finite = true;
                return;
              }
              blk_vals[j] = v0;
              blk_vals[j + 1] = v1;
              if (j == 0) {
                min_val = std::min(v0, v1);
                max_val = std::max(v0, v1);
              } else {
                min_val = std::min({min_val, v0, v1});
                max_val = std::max({max_val, v0, v1});
              }
            }
            const float raw_d =
                std::max(std::abs(min_val) / 8.0f, std::abs(max_val) / 7.0f);
            const float d = static_cast<float>(
                fp16_t(std::clamp(raw_d, kMinFp16Normal, kMaxFp16Normal)));
            (*bf16_vec)[b] = bf16_t(d);
            const float inv_d = 1.0f / d;
            for (int j = 0; j < block_size; j += 2) {
              const float q0 = std::clamp(blk_vals[j] * inv_d, -8.0f, 7.0f);
              const float q1 = std::clamp(blk_vals[j + 1] * inv_d, -8.0f, 7.0f);
              dp[j / 2].a = static_cast<int8_t>(std::rint(q0));
              dp[j / 2].b = static_cast<int8_t>(std::rint(q1));
            }
          }
        }));
    if (has_non_finite) {
      return absl::InvalidArgumentError(
          "Non-finite scale or zero_point in blockwise INT4 tensor.");
    }
    built->data = std::move(out_buf);
    built->bf16_scales = std::move(bf16_vec);
  }

  absl::MutexLock lock(bf16_cache_.mu);
  if (bf16_cache_.cpu_fallback == nullptr ||
      bf16_cache_.cpu_fallback->type != type ||
      bf16_cache_.cpu_fallback->rows != rows ||
      bf16_cache_.cpu_fallback->cols != cols) {
    bf16_cache_.cpu_fallback = std::move(built);
  }
  return bf16_cache_.cpu_fallback;
}

bool BlockwiseQuantization::HasCachedCpuSymmetricFallback() const {
  absl::MutexLock lock(bf16_cache_.mu);
  return bf16_cache_.cpu_fallback != nullptr;
}
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

std::shared_ptr<Scope> CurrentScope() {
  return g_current_scope != nullptr ? *g_current_scope : nullptr;
}

ScopeGuard::ScopeGuard(absl::string_view kind, absl::string_view name)
    : prev_scope_(CurrentScope()), scope_(std::make_shared<Scope>()) {
  scope_->kind = std::string(kind);
  scope_->name = name.empty() ? std::string(kind) : std::string(name);
  scope_->parent = prev_scope_;
  if (prev_scope_ != nullptr) {
    prev_scope_->children.push_back(scope_);
  }
  MutableCurrentScope() = scope_;
}

ScopeGuard::~ScopeGuard() {
  if (prev_scope_ == nullptr) {
    delete g_current_scope;
    g_current_scope = nullptr;
  } else {
    MutableCurrentScope() = std::move(prev_scope_);
  }
}

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
