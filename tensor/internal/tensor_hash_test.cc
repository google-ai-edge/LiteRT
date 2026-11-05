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

#include "tensor/internal/tensor_hash.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/internal/type_id.h"

namespace litert::tensor::graph::hash_internal {
namespace {

class FakeLazyRawHashBuffer final : public Buffer {
 public:
  FakeLazyRawHashBuffer(std::vector<std::byte> raw_bytes,
                        bool deterministic_raw_hash, bool materialized)
      : raw_bytes_(std::move(raw_bytes)),
        deterministic_raw_hash_(deterministic_raw_hash),
        materialized_(materialized) {}

  static internal::TypeId TypeId() {
    return internal::TypeId::Get<FakeLazyRawHashBuffer>();
  }
  internal::TypeId GetTypeId() const override { return TypeId(); }
  bool IsA(internal::TypeId id) const override { return id == TypeId(); }

  absl::StatusOr<size_t> ByteSize() const override { return raw_bytes_.size(); }
  bool IsMaterialized() const override {
    return materialized_.load(std::memory_order_acquire);
  }
  bool HasDeterministicRawHash() const override {
    return deterministic_raw_hash_;
  }
  void HashDeterministicContent(void (*mix_fn)(uint64_t*, const std::byte*,
                                               size_t),
                                uint64_t* state) const override {
    if (mix_fn != nullptr && state != nullptr) {
      mix_fn(state, raw_bytes_.data(), raw_bytes_.size());
    }
  }
  LockedBufferSpan<const std::byte> Lock() override {
    lock_calls_.fetch_add(1, std::memory_order_relaxed);
    materialized_.store(true, std::memory_order_release);
    return {raw_bytes_.data(), [](const std::byte*) {}, raw_bytes_.size()};
  }

  int lock_calls() const { return lock_calls_.load(std::memory_order_relaxed); }

 private:
  std::vector<std::byte> raw_bytes_;
  bool deterministic_raw_hash_ = false;
  std::atomic<bool> materialized_{false};
  std::atomic<int> lock_calls_{0};
};

TEST(TensorHashTest, SampledBytesDistinguishesSmallAndLargeBufferBoundaries) {
  std::vector<std::byte> small_a(4096, std::byte{0x11});
  std::vector<std::byte> small_b = small_a;
  small_b[2047] = std::byte{0x22};

  uint64_t hash_small_a = kFnvOffsetBasis;
  uint64_t hash_small_b = kFnvOffsetBasis;
  FnvMixSampledBytes(&hash_small_a, small_a.data(), small_a.size());
  FnvMixSampledBytes(&hash_small_b, small_b.data(), small_b.size());
  EXPECT_NE(hash_small_a, hash_small_b);

  // `> 4096` bytes switches to head/middle/tail + 32 strided 8-byte samples.
  std::vector<std::byte> large_a(8192, std::byte{0x33});
  std::vector<std::byte> large_head = large_a;
  large_head[0] = std::byte{0x44};
  std::vector<std::byte> large_mid = large_a;
  large_mid[large_mid.size() / 2] = std::byte{0x55};
  std::vector<std::byte> large_tail = large_a;
  large_tail.back() = std::byte{0x66};

  auto hash_vec = [](const std::vector<std::byte>& buf) {
    uint64_t s = kFnvOffsetBasis;
    FnvMixSampledBytes(&s, buf.data(), buf.size());
    return s;
  };
  const uint64_t base_large = hash_vec(large_a);
  EXPECT_EQ(hash_vec(large_a), base_large);
  EXPECT_NE(hash_vec(large_head), base_large);
  EXPECT_NE(hash_vec(large_mid), base_large);
  EXPECT_NE(hash_vec(large_tail), base_large);
}

TEST(TensorHashTest, DeterministicRawHashSkipsLockAndDistinguishesMetadata) {
  auto buf_a = std::make_shared<FakeLazyRawHashBuffer>(
      std::vector<std::byte>{std::byte{1}, std::byte{2}},
      /*deterministic_raw_hash=*/true, /*materialized=*/false);
  auto buf_b = std::make_shared<FakeLazyRawHashBuffer>(
      std::vector<std::byte>{std::byte{1}, std::byte{3}},
      /*deterministic_raw_hash=*/true, /*materialized=*/false);

  TensorInformation info_a;
  info_a.name = "w";
  info_a.type = Type::kI4;
  info_a.shape = {4, 32};
  info_a.buffer = buf_a;
  info_a.quantization = std::make_shared<BlockwiseQuantization>(
      std::vector<float>(4, 0.5f), std::vector<int64_t>{}, /*block_size=*/32,
      /*quantized_dimension=*/0);

  TensorInformation info_b = info_a;
  info_b.buffer = buf_b;

  const uint64_t h_a = ComputeConstantTensorHash(info_a, nullptr, 0);
  const uint64_t h_b = ComputeConstantTensorHash(info_b, nullptr, 0);
  TensorInformation info_renamed = info_a;
  info_renamed.name = "w_other";
  const uint64_t h_renamed =
      ComputeConstantTensorHash(info_renamed, nullptr, 0);
  EXPECT_NE(h_a, 0u);
  EXPECT_EQ(ComputeConstantTensorHash(info_a, nullptr, 0), h_a);
  EXPECT_NE(h_a, h_b);
  EXPECT_NE(h_a, h_renamed);
  EXPECT_EQ(buf_a->lock_calls(), 0);
  EXPECT_EQ(buf_b->lock_calls(), 0);
}

TEST(TensorHashTest, HandlesMaterializationPolicyAndQuantizationSchemes) {
  auto unmat = std::make_shared<FakeLazyRawHashBuffer>(
      std::vector<std::byte>(16, std::byte{7}),
      /*deterministic_raw_hash=*/false, /*materialized=*/false);

  TensorInformation info;
  info.name = "w_pc";
  info.type = Type::kI8;
  info.shape = {2, 8};
  info.buffer = unmat;
  info.quantization = std::make_shared<PerChannelAffineQuantization>(
      std::vector<float>{0.25f, 0.5f}, ZeroPoints::Float({-0.1f, 0.2f}),
      /*quantized_dimension=*/0);

  // `ComputeConstantTensorHash` sets `allow_materialize = false`, so an
  // unmaterialized buffer without `raw_data` stays unmaterialized.
  const uint64_t warm_hash = ComputeConstantTensorHash(info, nullptr, 0);
  EXPECT_NE(warm_hash, 0u);
  EXPECT_EQ(unmat->lock_calls(), 0);

  // `HashConstantTensorInfo(..., allow_materialize=true)` locks the buffer and
  // differentiates quantization parameters.
  uint64_t mat_hash_1 = kFnvOffsetBasis;
  HashConstantTensorInfo(info, nullptr, 0, /*include_name=*/true,
                         /*allow_materialize=*/true, &mat_hash_1);
  EXPECT_EQ(unmat->lock_calls(), 1);

  TensorInformation info_diff_scale = info;
  info_diff_scale.quantization = std::make_shared<PerChannelAffineQuantization>(
      std::vector<float>{0.75f, 0.5f}, ZeroPoints::Float({-0.1f, 0.2f}),
      /*quantized_dimension=*/0);
  uint64_t mat_hash_2 = kFnvOffsetBasis;
  HashConstantTensorInfo(info_diff_scale, nullptr, 0, /*include_name=*/true,
                         /*allow_materialize=*/true, &mat_hash_2);
  EXPECT_NE(mat_hash_1, mat_hash_2);
}

}  // namespace
}  // namespace litert::tensor::graph::hash_internal
