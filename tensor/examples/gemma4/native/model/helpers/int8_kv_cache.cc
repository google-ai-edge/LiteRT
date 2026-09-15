/* Copyright 2026 Google LLC.

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

#include "tensor/examples/gemma4/native/model/helpers/int8_kv_cache.h"
#include "absl/strings/str_cat.h"
#include "tensor/utils/macros.h"
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
namespace litert::tensor::examples::gemma4::native {
const std::vector<KvOwnerSpec> &PublishedE2BKvOwnerSpecs() {
  static const std::vector<KvOwnerSpec> specs = {
#include "tensor/examples/gemma4/native/model/helpers/published_kv_specs.inc"
  };
  return specs;
}
absl::Status ValidateKvOwnerSpecs(absl::Span<const KvOwnerSpec> specs,
                                  int capacity) {
  if (capacity < 1 || specs.empty())
    return absl::InvalidArgumentError(
        "Cache capacity and owner table must be nonempty");
  std::set<int> ids;
  for (const auto &s : specs) {
    if (s.owner < 0 || !ids.insert(s.owner).second || s.head_dim < 1 ||
        !std::isfinite(s.key_scale) || s.key_scale <= 0 ||
        !std::isfinite(s.value_scale) || s.value_scale <= 0 ||
        static_cast<size_t>(capacity) >
            std::numeric_limits<size_t>::max() / s.head_dim)
      return absl::InvalidArgumentError("Invalid cache owner metadata");
  }
  return absl::OkStatus();
}
std::shared_ptr<PerChannelAffineQuantization> KvQuantization(float scale) {
  return std::make_shared<PerChannelAffineQuantization>(
      std::vector<float>{scale}, std::vector<int64_t>{0});
}
absl::StatusOr<Int8KvCacheBank>
Int8KvCacheBank::Create(std::vector<KvOwnerSpec> specs, int capacity) {
  LRT_TENSOR_RETURN_IF_ERROR(ValidateKvOwnerSpecs(specs, capacity));
  Int8KvCacheBank bank;
  bank.specs_ = std::move(specs);
  bank.capacity_ = capacity;
  bank.buffers_.resize(bank.specs_.size());
  for (size_t i = 0; i < bank.specs_.size(); ++i) {
    size_t count = static_cast<size_t>(capacity) * bank.specs_[i].head_dim;
    for (int b = 0; b < 2; ++b) {
      bank.buffers_[i].keys[b].resize(count, 0);
      bank.buffers_[i].values[b].resize(count, 0);
    }
  }
  bank.keep_.resize(capacity, 1);
  bank.write_.resize(capacity, 0);
  return bank;
}
void Int8KvCacheBank::Reset() {
  for (auto &b : buffers_)
    for (int i = 0; i < 2; ++i) {
      std::fill(b.keys[i].begin(), b.keys[i].end(), 0);
      std::fill(b.values[i].begin(), b.values[i].end(), 0);
    }
  length_ = 0;
  current_ = 0;
  std::fill(keep_.begin(), keep_.end(), 1);
  std::fill(write_.begin(), write_.end(), 0);
}
size_t Int8KvCacheBank::Index(int owner) const {
  for (size_t i = 0; i < specs_.size(); ++i)
    if (specs_[i].owner == owner)
      return i;
  throw std::out_of_range("Unknown cache owner");
}
absl::Span<const int8_t> Int8KvCacheBank::Keys(int owner) const {
  return buffers_[Index(owner)].keys[current_];
}
absl::Span<const int8_t> Int8KvCacheBank::Values(int owner) const {
  return buffers_[Index(owner)].values[current_];
}
const void *Int8KvCacheBank::InputKeyAddress(int owner) const {
  return buffers_[Index(owner)].keys[current_].data();
}
const void *Int8KvCacheBank::OutputKeyAddress(int owner) const {
  return buffers_[Index(owner)].keys[1 - current_].data();
}
absl::Status
Int8KvCacheBank::ValidateHandles(absl::Span<const TensorHandle> keys,
                                 absl::Span<const TensorHandle> values) const {
  for (const auto &s : specs_) {
    if (static_cast<size_t>(s.owner) >= keys.size() ||
        static_cast<size_t>(s.owner) >= values.size())
      return absl::InvalidArgumentError(
          "Owner index missing in cache handle list");
    for (int kind = 0; kind < 2; ++kind) {
      const auto &t = kind ? values[s.owner] : keys[s.owner];
      LRT_TENSOR_RETURN_IF_ERROR(t.GetStatus());
      auto expected = kind ? Shape{1, 1, s.head_dim, capacity_}
                           : Shape{1, 1, capacity_, s.head_dim};
      if (t.GetType() != Type::kI8 || t.GetShape() != expected ||
          !t.GetQuantization())
        return absl::InvalidArgumentError(
            "Expected full-capacity baseline-layout INT8 cache handle");
      LRT_TENSOR_ASSIGN_OR_RETURN(
          const auto &q,
          t.GetQuantization()->As<PerChannelAffineQuantization>());
      if (q.scales != std::vector<float>{kind ? s.value_scale : s.key_scale} ||
          q.zero_points != std::vector<int64_t>{0})
        return absl::InvalidArgumentError(
            "Cache handle scale/zero differs from owner metadata");
    }
  }
  return absl::OkStatus();
}
absl::Status
Int8KvCacheBank::BindInputs(XnnpackRunner &r,
                            absl::Span<const TensorHandle> keys,
                            absl::Span<const TensorHandle> values) {
  LRT_TENSOR_RETURN_IF_ERROR(ValidateHandles(keys, values));
  for (size_t i = 0; i < specs_.size(); ++i) {
    int id = specs_[i].owner;
    LRT_TENSOR_RETURN_IF_ERROR(
        r.SetInput(keys[id], buffers_[i].keys[current_]));
    LRT_TENSOR_RETURN_IF_ERROR(
        r.SetInput(values[id], buffers_[i].values[current_]));
  }
  return absl::OkStatus();
}
absl::Status
Int8KvCacheBank::BindOutputs(XnnpackRunner &r,
                             absl::Span<const TensorHandle> keys,
                             absl::Span<const TensorHandle> values) {
  LRT_TENSOR_RETURN_IF_ERROR(ValidateHandles(keys, values));
  for (size_t i = 0; i < specs_.size(); ++i) {
    int id = specs_[i].owner;
    auto &k = buffers_[i].keys[1 - current_];
    auto &v = buffers_[i].values[1 - current_];
    LRT_TENSOR_RETURN_IF_ERROR(r.SetOutput(
        keys[id], absl::Span<std::byte>(reinterpret_cast<std::byte *>(k.data()),
                                        k.size())));
    LRT_TENSOR_RETURN_IF_ERROR(r.SetOutput(
        values[id], absl::Span<std::byte>(
                        reinterpret_cast<std::byte *>(v.data()), v.size())));
  }
  return absl::OkStatus();
}
absl::Status Int8KvCacheBank::BindDecodeMasks(XnnpackRunner &r,
                                              const TensorHandle &keep,
                                              const TensorHandle &write) {
  if (length_ >= capacity_)
    return absl::OutOfRangeError("INT8 cache is full");
  for (const auto *t : {&keep, &write}) {
    if (t->GetType() != Type::kI8 ||
        t->GetShape() != Shape{1, 1, capacity_, 1} || !t->GetQuantization())
      return absl::InvalidArgumentError("Invalid INT8 cache mask");
    LRT_TENSOR_ASSIGN_OR_RETURN(
        const auto &q,
        t->GetQuantization()->As<PerChannelAffineQuantization>());
    if (q.scales != std::vector<float>{1.0f} ||
        q.zero_points != std::vector<int64_t>{0})
      return absl::InvalidArgumentError("INT8 masks require scale1/zero0");
  }
  std::fill(keep_.begin(), keep_.end(), 1);
  std::fill(write_.begin(), write_.end(), 0);
  keep_[length_] = 0;
  write_[length_] = 1;
  LRT_TENSOR_RETURN_IF_ERROR(r.SetInput(keep, keep_));
  return r.SetInput(write, write_);
}
absl::Status Int8KvCacheBank::Commit(int active_tokens) {
  if (active_tokens <= 0 || active_tokens > capacity_ - length_)
    return absl::OutOfRangeError("Invalid INT8 cache append length");
  current_ = 1 - current_;
  length_ += active_tokens;
  return absl::OkStatus();
}
absl::Status FillFixedInt8KvAttentionMask(absl::Span<float> mask, int queries,
                                          int capacity, int start,
                                          int valid_tokens, bool local,
                                          int window) {
  if (queries < 1 || capacity < 1 || start < 0 || start > capacity ||
      queries > capacity - start || valid_tokens < 0 ||
      valid_tokens > capacity || window < 1 ||
      mask.size() != static_cast<size_t>(queries) * capacity)
    return absl::InvalidArgumentError(
        "Invalid fixed-capacity attention mask shape/history");
  const float negative = std::numeric_limits<float>::lowest();
  for (int q = 0; q < queries; ++q) {
    int position = start + q;
    for (int k = 0; k < capacity; ++k) {
      bool allowed = position < valid_tokens && k < valid_tokens &&
                     k <= position && (!local || k > position - window);
      mask[static_cast<size_t>(q) * capacity + k] = allowed ? 0.0f : negative;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Int8KvLocalWindows>
Int8KvLocalWindows::Create(std::vector<KvOwnerSpec> specs, int capacity,
                           int window) {
  LRT_TENSOR_RETURN_IF_ERROR(ValidateKvOwnerSpecs(specs, capacity));
  if (window < 1 || window > capacity)
    return absl::InvalidArgumentError("Invalid local cache window");
  Int8KvLocalWindows result;
  result.capacity_ = capacity;
  result.window_ = window;
  for (auto spec : specs)
    if (spec.owner % 5 != 4) {
      Entry entry{
          spec,
          std::vector<int8_t>(static_cast<size_t>(window) * spec.head_dim),
          std::vector<int8_t>(static_cast<size_t>(window) * spec.head_dim)};
      result.entries_.push_back(std::move(entry));
    }
  result.keep_.resize(window, 1);
  result.write_.resize(window, 0);
  return result;
}
absl::Span<const int8_t> Int8KvLocalWindows::Keys(int owner) const {
  for (const auto &e : entries_)
    if (e.spec.owner == owner)
      return e.keys;
  throw std::out_of_range("Unknown local K cache owner");
}
absl::Span<const int8_t> Int8KvLocalWindows::Values(int owner) const {
  for (const auto &e : entries_)
    if (e.spec.owner == owner)
      return e.values;
  throw std::out_of_range("Unknown local V cache owner");
}
absl::Status Int8KvLocalWindows::BindDecode(
    XnnpackRunner &r, const Int8KvCacheBank &bank,
    absl::Span<const TensorHandle> keys, absl::Span<const TensorHandle> values,
    const TensorHandle &keep, const TensorHandle &write) {
  if (bank.capacity() != capacity_ || bank.length() < 0 ||
      bank.length() >= capacity_)
    return absl::InvalidArgumentError(
        "Invalid full-bank state for local gather");
  begin_ = std::max(bank.length() + 1 - window_, 0);
  relative_position_ = bank.length() - begin_;
  for (auto &e : entries_) {
    const int id = e.spec.owner, dim = e.spec.head_dim;
    if (static_cast<size_t>(id) >= keys.size() ||
        static_cast<size_t>(id) >= values.size())
      return absl::InvalidArgumentError("Missing local owner handle");
    auto bank_spec = std::find_if(bank.specs().begin(), bank.specs().end(),
                                  [&](const auto &s) { return s.owner == id; });
    if (bank_spec == bank.specs().end() || bank_spec->head_dim != dim ||
        bank_spec->key_scale != e.spec.key_scale ||
        bank_spec->value_scale != e.spec.value_scale)
      return absl::InvalidArgumentError(
          "Local window and full bank metadata differ");
    for (int kind = 0; kind < 2; ++kind) {
      const auto &t = kind ? values[id] : keys[id];
      LRT_TENSOR_RETURN_IF_ERROR(t.GetStatus());
      auto q = CloneKvQuantization(t);
      auto expected =
          kind ? Shape{1, 1, dim, window_} : Shape{1, 1, window_, dim};
      if (t.GetType() != Type::kI8 || t.GetShape() != expected || !q ||
          q->scales != std::vector<float>{kind ? e.spec.value_scale
                                               : e.spec.key_scale} ||
          q->zero_points != std::vector<int64_t>{0})
        return absl::InvalidArgumentError("Invalid compact local cache tensor");
    }
    auto full_k = bank.Keys(id), full_v = bank.Values(id);
    std::copy_n(full_k.data() + static_cast<size_t>(begin_) * dim,
                e.keys.size(), e.keys.data());
    for (int d = 0; d < dim; ++d)
      std::copy_n(full_v.data() + static_cast<size_t>(d) * capacity_ + begin_,
                  window_, e.values.data() + static_cast<size_t>(d) * window_);
    LRT_TENSOR_RETURN_IF_ERROR(r.SetInput(keys[id], e.keys));
    LRT_TENSOR_RETURN_IF_ERROR(r.SetInput(values[id], e.values));
  }
  for (const auto *t : {&keep, &write}) {
    LRT_TENSOR_RETURN_IF_ERROR(t->GetStatus());
    auto q = CloneKvQuantization(*t);
    if (t->GetType() != Type::kI8 || t->GetShape() != Shape{1, 1, window_, 1} ||
        !q || q->scales != std::vector<float>{1.0f} ||
        q->zero_points != std::vector<int64_t>{0})
      return absl::InvalidArgumentError("Invalid compact INT8 update mask");
  }
  std::fill(keep_.begin(), keep_.end(), 1);
  std::fill(write_.begin(), write_.end(), 0);
  keep_[relative_position_] = 0;
  write_[relative_position_] = 1;
  LRT_TENSOR_RETURN_IF_ERROR(r.SetInput(keep, keep_));
  return r.SetInput(write, write_);
}
absl::Status FillLocalInt8KvAttentionMask(absl::Span<float> mask, int queries,
                                          int capacity, int start,
                                          int valid_tokens, int window) {
  if (queries < 1 || capacity < 1 || start < 0 || start >= capacity ||
      queries > capacity - start || window < 1 || window > capacity ||
      queries > capacity - window + 1 || valid_tokens < 0 ||
      valid_tokens > capacity)
    return absl::InvalidArgumentError("Invalid compact local mask dimensions");
  const int extent = window + queries - 1,
            begin = std::max(start + 1 - window, 0);
  if (begin > capacity - extent ||
      mask.size() != static_cast<size_t>(queries) * extent)
    return absl::InvalidArgumentError("Compact local mask exceeds capacity");
  for (int q = 0; q < queries; ++q)
    for (int j = 0; j < extent; ++j) {
      int pos = start + q, key = begin + j;
      bool visible = pos < valid_tokens && key < valid_tokens && key <= pos &&
                     key >= pos - window + 1;
      mask[static_cast<size_t>(q) * extent + j] =
          visible ? 0.0f : std::numeric_limits<float>::lowest();
    }
  return absl::OkStatus();
}
} // namespace litert::tensor::examples::gemma4::native
