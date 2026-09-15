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

// Artifact-only matched-bundle cache storage and graph helpers.
#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_INT8_KV_CACHE_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MODEL_HELPERS_INT8_KV_CACHE_H_
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/runners/xnnpack/runner.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace litert::tensor::examples::gemma4::native {
inline constexpr char kKvKeepMaskName[] = "matched.kv.keep_mask";
inline constexpr char kKvWriteMaskName[] = "matched.kv.write_mask";
struct KvOwnerSpec {
  int owner;
  int head_dim;
  float key_scale;
  float value_scale;
};
const std::vector<KvOwnerSpec> &PublishedE2BKvOwnerSpecs();
absl::Status ValidateKvOwnerSpecs(absl::Span<const KvOwnerSpec> specs,
                                  int capacity);
std::shared_ptr<PerChannelAffineQuantization> KvQuantization(float scale);
inline std::shared_ptr<PerChannelAffineQuantization>
CloneKvQuantization(const TensorHandle &cache) {
  if (!cache.GetQuantization())
    return nullptr;
  auto q = cache.GetQuantization()->As<PerChannelAffineQuantization>();
  if (!q.ok())
    return nullptr;
  return std::make_shared<PerChannelAffineQuantization>(*q);
}

template <class... M>
Tensor<M...> MakeInt8KeyCache(std::string name, int capacity, int dim,
                              float scale) {
  return Tensor<M...>({.name = std::move(name),
                       .type = Type::kI8,
                       .shape = {1, 1, capacity, dim},
                       .quantization = KvQuantization(scale)});
}
template <class... M>
Tensor<M...> MakeInt8ValueCache(std::string name, int capacity, int dim,
                                float scale) {
  return Tensor<M...>({.name = std::move(name),
                       .type = Type::kI8,
                       .shape = {1, 1, dim, capacity},
                       .quantization = KvQuantization(scale)});
}
template <class... M>
Tensor<M...> MakeInt8KvMask(std::string name, int capacity) {
  return Tensor<M...>({.name = std::move(name),
                       .type = Type::kI8,
                       .shape = {1, 1, capacity, 1},
                       .quantization = KvQuantization(1.0f)});
}
template <class... M>
Tensor<M...> QuantizeInt8Kv(Tensor<M...> fp32, const Tensor<M...> &cache) {
  auto result = Cast(fp32, Type::kI8);
  result.SetQuantization(CloneKvQuantization(cache));
  return result;
}
// Integer mask algebra only: each position selects either old codes or new
// codes. Masks have scale1, zero0 and contain exact codes0/1. Tests exhaust all
// INT8 values to guard against an unintended requantization boundary.
template <class... M>
Tensor<M...> UpdateInt8Kv(Tensor<M...> old_cache, Tensor<M...> current,
                          Tensor<M...> keep, Tensor<M...> write) {
  auto retained = Mul(old_cache, keep);
  retained.SetQuantization(old_cache.GetQuantization());
  auto inserted = Mul(current, write);
  inserted.SetQuantization(old_cache.GetQuantization());
  auto result = Add(retained, inserted);
  result.SetQuantization(old_cache.GetQuantization());
  return result;
}
// Initial fixed-signature prefill starts at position0. Its padded token rows
// may be written, but the causal/valid-position mask excludes unused positions.
template <class... M>
Tensor<M...> PadInitialInt8Kv(Tensor<M...> current, int axis, int capacity) {
  const auto shape = current.GetShape();
  if (axis < 0 || axis >= static_cast<int>(shape.size()) || shape[axis] <= 0 ||
      shape[axis] > capacity)
    return Tensor<M...>(graph::ErrorTensor(
        absl::InvalidArgumentError("Invalid initial INT8 cache extent")));
  if (shape[axis] == capacity)
    return current;
  auto tail_shape = shape;
  tail_shape[axis] = capacity - shape[axis];
  size_t count = 1;
  for (int d : tail_shape)
    count *= d;
  auto zero = Tensor<M...>({.type = Type::kI8,
                            .shape = tail_shape,
                            .buffer = OwningCpuBuffer::Copy<Type::kI8>(
                                std::vector<int8_t>(count, 0)),
                            .quantization = current.GetQuantization()});
  auto result = Concatenation({current, zero}, axis);
  result.SetQuantization(current.GetQuantization());
  return result;
}

// Owns two genuinely INT8 full-capacity banks. Inputs and outputs always refer
// to different banks; Commit swaps roles without copying/requantizing cache.
class Int8KvCacheBank {
public:
  static absl::StatusOr<Int8KvCacheBank> Create(std::vector<KvOwnerSpec> specs,
                                                int capacity = 2048);
  Int8KvCacheBank(Int8KvCacheBank &&) = default;
  Int8KvCacheBank &operator=(Int8KvCacheBank &&) = default;
  Int8KvCacheBank(const Int8KvCacheBank &) = delete;
  Int8KvCacheBank &operator=(const Int8KvCacheBank &) = delete;
  int capacity() const { return capacity_; }
  int length() const { return length_; }
  const std::vector<KvOwnerSpec> &specs() const { return specs_; }
  void Reset();
  absl::Status BindInputs(XnnpackRunner &runner,
                          absl::Span<const TensorHandle> keys,
                          absl::Span<const TensorHandle> values);
  absl::Status BindOutputs(XnnpackRunner &runner,
                           absl::Span<const TensorHandle> keys,
                           absl::Span<const TensorHandle> values);
  absl::Status BindDecodeMasks(XnnpackRunner &runner, const TensorHandle &keep,
                               const TensorHandle &write);
  // Call only after a successful graph invocation. For initial padded prefill,
  // active_tokens is the logical token count, not the fixed graph token count.
  absl::Status Commit(int active_tokens);
  absl::Span<const int8_t> Keys(int owner) const;
  absl::Span<const int8_t> Values(int owner) const;
  const void *InputKeyAddress(int owner) const;
  const void *OutputKeyAddress(int owner) const;

private:
  struct OwnerBuffers {
    std::vector<int8_t> keys[2], values[2];
  };
  Int8KvCacheBank() = default;
  absl::Status ValidateHandles(absl::Span<const TensorHandle> keys,
                               absl::Span<const TensorHandle> values) const;
  size_t Index(int owner) const;
  std::vector<KvOwnerSpec> specs_;
  std::vector<OwnerBuffers> buffers_;
  std::vector<int8_t> keep_, write_;
  int capacity_ = 0, length_ = 0, current_ = 0;
};
// Local compact windows are extra INT8 inputs. The authoritative persistent
// state remains the full-capacity bank; gathering does not mutate that state.
inline constexpr char kLocalKvKeepMaskName[] = "matched.kv.local_keep_mask";
inline constexpr char kLocalKvWriteMaskName[] = "matched.kv.local_write_mask";
inline std::string LocalKvKeyName(const std::string &attention_name) {
  return attention_name + ".matched.local_key";
}
inline std::string LocalKvValueName(const std::string &attention_name) {
  return attention_name + ".matched.local_value";
}
class Int8KvLocalWindows {
public:
  static absl::StatusOr<Int8KvLocalWindows>
  Create(std::vector<KvOwnerSpec> specs, int capacity = 2048, int window = 512);
  // Owner-indexed handles; global-owner entries are ignored. This gathers old
  // windows and binds dynamic0/1 masks at the current relative token position.
  // Keep this object alive until Run completes. Include gathering in timing.
  absl::Status BindDecode(XnnpackRunner &runner, const Int8KvCacheBank &bank,
                          absl::Span<const TensorHandle> keys,
                          absl::Span<const TensorHandle> values,
                          const TensorHandle &keep, const TensorHandle &write);
  int begin() const { return begin_; }
  int window() const { return window_; }
  int relative_position() const { return relative_position_; }
  absl::Span<const int8_t> Keys(int owner) const;
  absl::Span<const int8_t> Values(int owner) const;

private:
  struct Entry {
    KvOwnerSpec spec;
    std::vector<int8_t> keys, values;
  };
  std::vector<Entry> entries_;
  std::vector<int8_t> keep_, write_;
  int capacity_ = 0, window_ = 0, begin_ = 0, relative_position_ = 0;
};
// Compact local mask [1,1,queries,window+queries-1]. Absolute beginning is
// max(start+1-window,0). Current and preceding window-1 positions are visible.
absl::Status FillLocalInt8KvAttentionMask(absl::Span<float> mask, int queries,
                                          int capacity, int start,
                                          int valid_tokens, int window = 512);

// Full-capacity [1,1,queries,capacity] FP32 attention mask. Position q reads
// causal positions <=start+q and <valid_tokens; local attention keeps the most
// recent window positions. Padding queries use the finite lowest float mask to
// avoid NaN softmax rows.
absl::Status FillFixedInt8KvAttentionMask(absl::Span<float> mask, int queries,
                                          int capacity, int start,
                                          int valid_tokens, bool local,
                                          int window = 512);
} // namespace litert::tensor::examples::gemma4::native
#endif
