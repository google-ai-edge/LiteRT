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

#include "tensor/runners/common_nnpack/runner.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "tensor/datatypes.h"
#include "tensor/internal/graph.h"
#include "tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

TEST(ExternalBufferTest, DefaultIsOwnedAndEmpty) {
  ExternalBuffer buf;
  EXPECT_TRUE(buf.IsOwned());
  EXPECT_TRUE(buf.data().empty());
}

TEST(ExternalBufferTest, SetOwnedBuffer) {
  const std::vector<uint8_t> src = {1, 2, 3, 4};
  ExternalBuffer buf;
  buf.SetOwnedBuffer(absl::MakeSpan(
      reinterpret_cast<const std::byte*>(src.data()), src.size()));

  EXPECT_TRUE(buf.IsOwned());
  EXPECT_EQ(buf.data().size(), 4);
  EXPECT_EQ(static_cast<uint8_t>(buf.data()[0]), 1);
  EXPECT_EQ(static_cast<uint8_t>(buf.data()[3]), 4);
}

TEST(ExternalBufferTest, SetExternalView) {
  const std::vector<uint8_t> src = {10, 20, 30};
  ExternalBuffer buf;
  buf.SetExternalView(absl::MakeSpan(
      reinterpret_cast<const std::byte*>(src.data()), src.size()));

  EXPECT_FALSE(buf.IsOwned());
  EXPECT_EQ(buf.data().size(), 3);
  EXPECT_EQ(static_cast<uint8_t>(buf.data()[0]), 10);
  EXPECT_EQ(static_cast<uint8_t>(buf.data()[2]), 30);
}

TEST(ExternalBufferTest, ResizeOwnedBuffer) {
  ExternalBuffer buf;
  EXPECT_THAT(buf.Resize(8), IsOk());
  EXPECT_TRUE(buf.IsOwned());
  EXPECT_EQ(buf.data().size(), 8);
}

TEST(ExternalBufferTest, ResizeExternalViewSmallerOrEqualNoOp) {
  const std::vector<uint8_t> src = {1, 2, 3, 4, 5};
  ExternalBuffer buf;
  buf.SetExternalView(absl::MakeSpan(
      reinterpret_cast<const std::byte*>(src.data()), src.size()));

  EXPECT_THAT(buf.Resize(3), IsOk());
  EXPECT_FALSE(buf.IsOwned());
  EXPECT_EQ(buf.data().size(), 5);
}

TEST(ExternalBufferTest,
     ResizeExternalViewLargerTransitionsToOwnedAndPreservesData) {
  const std::vector<uint8_t> src = {10, 20, 30};
  ExternalBuffer buf;
  buf.SetExternalView(absl::MakeSpan(
      reinterpret_cast<const std::byte*>(src.data()), src.size()));

  EXPECT_THAT(buf.Resize(6), IsOk());
  EXPECT_TRUE(buf.IsOwned());
  EXPECT_EQ(buf.data().size(), 6);
  EXPECT_EQ(static_cast<uint8_t>(buf.data()[0]), 10);
  EXPECT_EQ(static_cast<uint8_t>(buf.data()[1]), 20);
  EXPECT_EQ(static_cast<uint8_t>(buf.data()[2]), 30);
}

TEST(ByteSizeTest, ComputesCorrectSize) {
  graph::TensorInformation info{
      .type = Type::kFP32,
      .shape = {2, 3, 4},
  };
  EXPECT_EQ(ByteSize(info), 2 * 3 * 4 * sizeof(float));
}

}  // namespace
}  // namespace litert::tensor
