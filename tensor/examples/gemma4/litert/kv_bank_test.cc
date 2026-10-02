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

#include "tensor/examples/gemma4/litert/kv_bank.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
namespace litert::tensor::examples::gemma4::cpu {
namespace {
TEST(KvBankTest, TransactionPreservesCommittedRowsOnAbortAndReset) {
  auto result =
      ActiveKvBank::Create({{0, 256, 0.5f, 0.25f}, {1, 512, 0.25f, 0.5f}}, 4);
  ASSERT_TRUE(result.ok());
  if (!result.ok()) return;
  auto bank = std::move(*result);
  for (const auto& spec : bank.specs()) {
    auto empty = bank.Keys(spec.owner, 0, 0);
    ASSERT_TRUE(empty.ok());
    if (!empty.ok()) return;
    EXPECT_TRUE(empty->empty());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(empty->data()) % 64, 0);
  }
  ASSERT_TRUE(bank.BeginAppend(2).ok());
  std::vector<int8_t> local(512, 42), global(1024, -11);
  ASSERT_TRUE(bank.Append(0, local, local).ok());
  EXPECT_FALSE(bank.Commit().ok());
  EXPECT_FALSE(bank.Keys(1, 0, 2).ok());
  ASSERT_TRUE(bank.Append(1, global, global).ok());
  ASSERT_TRUE(bank.Commit().ok());
  ASSERT_TRUE(bank.BeginAppend(1).ok());
  std::vector<int8_t> extra(256, 99);
  ASSERT_TRUE(bank.Append(0, extra, extra).ok());
  bank.Abort();
  EXPECT_EQ(bank.length(), 2);
  EXPECT_FALSE(bank.Keys(0, 0, 3).ok());
  auto committed = bank.Keys(0, 0, 2);
  ASSERT_TRUE(committed.ok());
  if (!committed.ok()) return;
  EXPECT_EQ(std::vector<int8_t>(committed->begin(), committed->end()), local);
  bank.Reset();
  EXPECT_EQ(bank.length(), 0);
  EXPECT_FALSE(bank.Keys(0, 0, 1).ok());
  EXPECT_FALSE(bank.BeginAppend(5).ok());
}
TEST(KvBankTest, RejectsInvalidOwnersAndAppendSizes) {
  EXPECT_FALSE(ActiveKvBank::Create({}, 4).ok());
  EXPECT_FALSE(ActiveKvBank::Create({{0, 256, 0, 1}}, 4).ok());
  EXPECT_FALSE(ActiveKvBank::Create({{0, 256, 1, 1}, {0, 256, 1, 1}}, 4).ok());
  auto result = ActiveKvBank::Create({{0, 256, 1, 1}}, 2);
  ASSERT_TRUE(result.ok());
  if (!result.ok()) return;
  auto bank = std::move(*result);
  ASSERT_TRUE(bank.BeginAppend(1).ok());
  EXPECT_FALSE(bank.Append(0, {}, {}).ok());
  EXPECT_FALSE(bank.Values(0, -1, 0).ok());
  EXPECT_FALSE(bank.Values(1, 0, 0).ok());
}
}  // namespace
}  // namespace litert::tensor::examples::gemma4::cpu
