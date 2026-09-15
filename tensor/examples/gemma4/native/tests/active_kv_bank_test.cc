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

// Focused artifact-only correctness checks; no model inference or timing.
#include "tensor/examples/gemma4/native/active_kv_bank.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {
using litert::tensor::examples::gemma4::native::ActiveKvBank;
using litert::tensor::examples::gemma4::native::KvOwnerSpec;

int checks = 0;
#define CHECK(condition)                                                     \
  do {                                                                       \
    ++checks;                                                                \
    if (!(condition)) {                                                      \
      std::cerr << __FILE__ << ':' << __LINE__ << ": " #condition "\n";       \
      std::exit(1);                                                          \
    }                                                                        \
  } while (false)

std::vector<int8_t> Codes(int count, int seed) {
  std::vector<int8_t> result(count);
  for (int i = 0; i < count; ++i) {
    result[i] = static_cast<int8_t>((i + seed) % 256 - 128);
  }
  return result;
}

bool Equals(absl::Span<const int8_t> actual,
            absl::Span<const int8_t> expected) {
  return actual.size() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

ActiveKvBank Make(std::vector<KvOwnerSpec> specs, int capacity) {
  auto bank = ActiveKvBank::Create(std::move(specs), capacity);
  CHECK(bank.ok());
  return std::move(*bank);
}

void MetadataAndBounds() {
  const std::vector<KvOwnerSpec> valid = {{2, 3, 0.5f, 0.25f}};
  CHECK(!ActiveKvBank::Create(valid, 0).ok());
  CHECK(!ActiveKvBank::Create(valid, -1).ok());
  CHECK(!ActiveKvBank::Create({}, 8).ok());
  CHECK(!ActiveKvBank::Create({valid[0], valid[0]}, 8).ok());
  for (const auto& invalid : std::vector<KvOwnerSpec>{
           {-1, 3, 0.5f, 0.25f}, {2, 0, 0.5f, 0.25f},
           {2, -1, 0.5f, 0.25f}, {2, 3, 0.0f, 0.25f},
           {2, 3, 0.5f, -1.0f},
           {2, 3, std::numeric_limits<float>::infinity(), 0.25f},
           {2, 3, 0.5f, std::numeric_limits<float>::quiet_NaN()}}) {
    CHECK(!ActiveKvBank::Create({invalid}, 8).ok());
  }
  auto bank = Make(valid, 8);
  CHECK(bank.specs().size() == 1 && bank.specs()[0].owner == 2);
  CHECK(bank.capacity() == 8 && bank.length() == 0 && bank.pending_end() == 0);
  CHECK(!bank.append_in_flight());
  CHECK(bank.Keys(2, 0, 0).ok() && bank.Keys(2, 0, 0)->empty());
  CHECK(!bank.Keys(99, 0, 0).ok());
  CHECK(!bank.Values(2, -1, 0).ok());
  CHECK(!bank.Keys(2, 1, 0).ok());
  CHECK(!bank.Values(2, 0, 1).ok());
  CHECK(!bank.Commit().ok());
  CHECK(!bank.Append(2, Codes(3, 0), Codes(3, 1)).ok());
  CHECK(!bank.BeginAppend(0).ok());
  CHECK(!bank.BeginAppend(-1).ok());
  CHECK(!bank.BeginAppend(9).ok());
  CHECK(!bank.BeginAppend(std::numeric_limits<int>::max()).ok());
  CHECK(bank.BeginAppend(1).ok());
  CHECK(!bank.BeginAppend(1).ok());
  CHECK(!bank.Append(99, Codes(3, 0), Codes(3, 1)).ok());
  CHECK(!bank.Append(2, Codes(2, 0), Codes(3, 1)).ok());
  CHECK(!bank.Append(2, Codes(3, 0), Codes(4, 1)).ok());
  CHECK(!bank.Commit().ok());
  CHECK(bank.length() == 0 && bank.pending_end() == 1);
  CHECK(bank.Append(2, Codes(3, 0), Codes(3, 1)).ok());
  CHECK(bank.Commit().ok());
  CHECK(bank.length() == 1 && bank.pending_end() == 1);
}

void PerOwnerPendingVisibility() {
  auto bank = Make({{8, 3, 0.5f, 0.25f}, {1, 5, 0.75f, 0.125f}}, 16);
  auto key8 = Codes(12, 11), value8 = Codes(12, 37);
  auto key1 = Codes(20, 59), value1 = Codes(20, 81);
  CHECK(bank.BeginAppend(4).ok());
  CHECK(!bank.Keys(8, 0, 4).ok());
  CHECK(bank.Append(8, key8, value8).ok());
  CHECK(bank.length() == 0 && bank.pending_end() == 4);
  CHECK(Equals(*bank.Keys(8, 0, 4), key8));
  CHECK(Equals(*bank.Values(8, 1, 3), absl::MakeConstSpan(value8).subspan(3, 6)));
  CHECK(!bank.Keys(1, 0, 4).ok());
  CHECK(!bank.Values(1, 1, 2).ok());
  CHECK(!bank.Commit().ok());
  CHECK(bank.append_in_flight());
  CHECK(!bank.Append(8, Codes(12, 0), Codes(12, 0)).ok());
  CHECK(Equals(*bank.Keys(8, 0, 4), key8));
  CHECK(bank.Append(1, key1, value1).ok());
  CHECK(Equals(*bank.Values(1, 0, 4), value1));
  CHECK(bank.Commit().ok());
  CHECK(bank.length() == 4 && !bank.append_in_flight());
  // Shared consumer layers receive views into the same owner allocation.
  const auto first_consumer = *bank.Keys(8, 0, 4);
  const auto shared_consumer = *bank.Keys(8, 1, 4);
  CHECK(shared_consumer.data() == first_consumer.data() + 3 &&
        shared_consumer.size() == 9);
  CHECK(bank.BeginAppend(1).ok());
  CHECK(Equals(*bank.Keys(1, 0, 4), key1));
  CHECK(!bank.Keys(1, 0, 5).ok());
  bank.Abort();
  CHECK(bank.length() == 4 && bank.pending_end() == 4);
}

void ExactCodesAndTokenMajorValues() {
  auto bank = Make({{3, 2, 0.333f, 0.017f}}, 128);
  const auto keys = Codes(256, 0), values = Codes(256, 129);
  CHECK(bank.BeginAppend(128).ok());
  CHECK(bank.Append(3, keys, values).ok());
  CHECK(bank.Commit().ok());
  CHECK(Equals(*bank.Keys(3, 0, 128), keys));
  CHECK(Equals(*bank.Values(3, 0, 128), values));
  for (int token = 0; token < 128; ++token) {
    CHECK(Equals(*bank.Values(3, token, token + 1),
                 absl::MakeConstSpan(values).subspan(token * 2, 2)));
  }
}

void ResetAbortAndUntouchedBytes() {
  constexpr int capacity = 16, dim = 3;
  auto bank = Make({{4, dim, 0.5f, 0.25f}, {9, 1, 0.5f, 0.25f}}, capacity);
  auto keys = Codes(capacity * dim, 3), values = Codes(capacity * dim, 101);
  auto key9 = Codes(capacity, 7), value9 = Codes(capacity, 107);
  CHECK(bank.BeginAppend(capacity).ok());
  CHECK(bank.Append(4, keys, values).ok());
  CHECK(bank.Append(9, key9, value9).ok());
  CHECK(bank.Commit().ok());
  const int8_t* key_storage = bank.Keys(4, 0, capacity)->data();
  const int8_t* value_storage = bank.Values(4, 0, capacity)->data();
  const int8_t* key9_storage = bank.Keys(9, 0, capacity)->data();
  // Save physical storage pointers solely to audit unchanged bytes. Reset
  // invalidates these full logical views, but does not free the allocations.
  bank.Reset();
  CHECK(bank.length() == 0 && bank.pending_end() == 0);
  CHECK(!bank.Keys(4, 0, 1).ok() && !bank.Values(9, 0, 1).ok());
  CHECK(Equals(absl::Span<const int8_t>(key_storage, keys.size()), keys));
  CHECK(Equals(absl::Span<const int8_t>(value_storage, values.size()), values));
  CHECK(bank.BeginAppend(2).ok());
  auto new_keys = Codes(2 * dim, 211), new_values = Codes(2 * dim, 223);
  CHECK(bank.Append(4, new_keys, new_values).ok());
  std::copy(new_keys.begin(), new_keys.end(), keys.begin());
  std::copy(new_values.begin(), new_values.end(), values.begin());
  CHECK(Equals(absl::Span<const int8_t>(key_storage, keys.size()), keys));
  CHECK(Equals(absl::Span<const int8_t>(value_storage, values.size()), values));
  CHECK(Equals(absl::Span<const int8_t>(key9_storage, key9.size()), key9));
  CHECK(!bank.Keys(4, 0, 3).ok());
  CHECK(bank.Append(9, Codes(2, 231), Codes(2, 233)).ok());
  CHECK(bank.Commit().ok());
  CHECK(bank.BeginAppend(3).ok());
  CHECK(bank.Append(4, Codes(3 * dim, 43), Codes(3 * dim, 67)).ok());
  CHECK(bank.Keys(4, 0, 5).ok());
  CHECK(!bank.Keys(9, 0, 5).ok());
  bank.Abort();
  CHECK(bank.length() == 2 && bank.pending_end() == 2);
  CHECK(Equals(*bank.Keys(4, 0, 2), new_keys));
  CHECK(Equals(*bank.Values(4, 0, 2), new_values));
  CHECK(!bank.Keys(4, 0, 3).ok());
  CHECK(bank.BeginAppend(1).ok());
  CHECK(bank.Append(4, Codes(dim, 239), Codes(dim, 241)).ok());
  CHECK(bank.Append(9, Codes(1, 243), Codes(1, 245)).ok());
  CHECK(bank.Commit().ok());
  CHECK(bank.length() == 3 && !bank.Keys(4, 0, 4).ok());
  CHECK(bank.Keys(4, 0, 3)->data() == key_storage);
  for (size_t i = capacity * dim; i < capacity * dim + XNN_EXTRA_BYTES; ++i) {
    CHECK(key_storage[i] == 0 && value_storage[i] == 0);
  }
  CHECK(bank.BeginAppend(1).ok());
  CHECK(bank.Append(4, Codes(dim, 21), Codes(dim, 22)).ok());
  bank.Reset();
  CHECK(bank.length() == 0 && !bank.append_in_flight());
  CHECK(!bank.Commit().ok() && !bank.Keys(4, 0, 1).ok());
  bank.Abort();
}

void SlidingBoundaryAndCapacity() {
  constexpr int capacity = 2048, dim = 2;
  auto bank = Make({{0, dim, 0.5f, 0.25f}}, capacity);
  auto all_keys = Codes(capacity * dim, 9);
  auto all_values = Codes(capacity * dim, 131);
  for (int end : {511, 512, 513, capacity}) {
    const int begin = bank.length();
    CHECK(bank.BeginAppend(end - begin).ok());
    CHECK(bank.pending_end() == end && bank.length() == begin);
    CHECK(bank.Append(0, absl::MakeConstSpan(all_keys).subspan(begin * dim,
                                        (end - begin) * dim),
                         absl::MakeConstSpan(all_values).subspan(begin * dim,
                                        (end - begin) * dim)).ok());
    const int window_begin = std::max(0, end - 512);
    CHECK(Equals(*bank.Keys(0, window_begin, end),
                 absl::MakeConstSpan(all_keys).subspan(window_begin * dim,
                                           (end - window_begin) * dim)));
    CHECK(Equals(*bank.Values(0, window_begin, end),
                 absl::MakeConstSpan(all_values).subspan(window_begin * dim,
                                           (end - window_begin) * dim)));
    CHECK(bank.Keys(0, end, end)->empty());
    CHECK(!bank.Keys(0, 0, end + 1).ok());
    CHECK(bank.Commit().ok());
  }
  CHECK(Equals(*bank.Keys(0, 0, capacity), all_keys));
  CHECK(Equals(*bank.Values(0, 0, capacity), all_values));
  CHECK(!bank.BeginAppend(1).ok() && bank.length() == capacity);
  const int8_t* old_pointer = bank.Keys(0, 0, capacity)->data();
  auto moved = std::move(bank);
  CHECK(moved.length() == capacity);
  CHECK(moved.Keys(0, 0, capacity)->data() == old_pointer);
  CHECK(Equals(*moved.Values(0, 0, capacity), all_values));
  moved.Reset();
  CHECK(moved.BeginAppend(capacity).ok());
  CHECK(moved.Append(0, all_keys, all_values).ok());
  CHECK(moved.Commit().ok() && moved.length() == capacity);
}

void PaddedViewsPreserveVisibilityAndStorage() {
  constexpr int capacity = 8, dim = 3;
  auto bank = Make({{2, dim, 0.5f, 0.25f}, {7, dim, 0.5f, 0.25f}}, capacity);
  const auto old_keys = Codes(capacity * dim, 9);
  const auto old_values = Codes(capacity * dim, 109);
  CHECK(bank.BeginAppend(capacity).ok());
  for (int owner : {2, 7}) CHECK(bank.Append(owner, old_keys, old_values).ok());
  CHECK(bank.Commit().ok());
  const int8_t* key_storage = bank.Keys(2, 0, capacity)->data();
  const int8_t* value_storage = bank.Values(2, 0, capacity)->data();
  bank.Reset();
  // Borrowed bytes do not restore the old logical history after Reset.
  CHECK(bank.PaddedKeys(2, 0, 0, capacity)->data() == key_storage);
  CHECK(Equals(*bank.PaddedValues(2, 0, 0, capacity), old_values));
  CHECK(bank.length() == 0 && bank.pending_end() == 0);
  CHECK(!bank.PaddedKeys(2, 0, 1, capacity).ok());
  CHECK(!bank.Keys(2, 0, capacity).ok());
  CHECK(!bank.PaddedKeys(99, 0, 0, capacity).ok());
  CHECK(!bank.PaddedValues(2, -1, 0, capacity).ok());
  CHECK(!bank.PaddedKeys(2, 1, 0, capacity).ok());
  CHECK(!bank.PaddedKeys(2, 0, 0, -1).ok());
  CHECK(!bank.PaddedValues(2, 0, 0, capacity + 1).ok());
  CHECK(bank.BeginAppend(2).ok());
  const auto new_keys = Codes(2 * dim, 201);
  const auto new_values = Codes(2 * dim, 221);
  CHECK(bank.Append(2, new_keys, new_values).ok());
  CHECK(!bank.PaddedKeys(7, 0, 2, 4).ok());
  CHECK(!bank.PaddedValues(7, 0, 2, 4).ok());
  CHECK(!bank.Commit().ok());
  CHECK(!bank.PaddedKeys(2, 0, 2, 1).ok());
  CHECK(!bank.PaddedValues(2, 0, 3, 4).ok());
  const auto padded_keys = *bank.PaddedKeys(2, 0, 2, 4);
  const auto padded_values = *bank.PaddedValues(2, 0, 2, 4);
  CHECK(padded_keys.size() == 4 * dim && padded_keys.data() == key_storage);
  CHECK(padded_values.size() == 4 * dim && padded_values.data() == value_storage);
  CHECK(Equals(padded_keys.first(2 * dim), new_keys));
  CHECK(Equals(padded_values.first(2 * dim), new_values));
  CHECK(Equals(padded_keys.subspan(2 * dim),
               absl::MakeConstSpan(old_keys).subspan(2 * dim, 2 * dim)));
  CHECK(Equals(padded_values.subspan(2 * dim),
               absl::MakeConstSpan(old_values).subspan(2 * dim, 2 * dim)));
  const auto suffix = *bank.PaddedValues(2, 1, 2, capacity);
  CHECK(suffix.data() == value_storage + dim && suffix.size() == (capacity - 1) * dim);
  CHECK(bank.length() == 0 && bank.pending_end() == 2);
  CHECK(!bank.Keys(2, 0, 4).ok() && !bank.Values(2, 0, 4).ok());
  // This caller-side masking example excludes the stale suffix explicitly.
  // Padded views supply storage only; they cannot apply attention masks.
  int masked_sum = 0;
  for (int token = 0; token < 4; ++token) {
    if (token >= 2) continue;
    for (int channel = 0; channel < dim; ++channel) {
      masked_sum += padded_values[token * dim + channel];
    }
  }
  int expected_sum = 0;
  for (int8_t code : new_values) expected_sum += code;
  CHECK(masked_sum == expected_sum);
  CHECK(bank.Append(7, new_keys, new_values).ok());
  CHECK(bank.Commit().ok());
  CHECK(bank.length() == 2 && bank.PaddedValues(2, 0, 2, 4)->data() == value_storage);
  bank.Abort();
  CHECK(bank.PaddedKeys(2, 0, 2, capacity)->size() == capacity * dim);
  bank.Reset();
  CHECK(!bank.PaddedKeys(2, 0, 2, capacity).ok());
  CHECK(Equals(bank.PaddedValues(2, 0, 0, capacity)->first(2 * dim), new_values));
}
}  // namespace

int main() {
  MetadataAndBounds();
  PerOwnerPendingVisibility();
  ExactCodesAndTokenMajorValues();
  ResetAbortAndUntouchedBytes();
  SlidingBoundaryAndCapacity();
  PaddedViewsPreserveVisibilityAndStorage();
  std::cout << "ActiveKvBank: " << checks << " checks passed\n";
  return 0;
}
