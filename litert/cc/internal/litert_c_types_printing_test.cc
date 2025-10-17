// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/cc/internal/litert_c_types_printing.h"  // IWYU pragma: keep

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"

namespace litert {
namespace {

TEST(LitertCTypesPrintingTest, LiteRtElementType) {
  EXPECT_EQ(absl::StrFormat("%v", kLiteRtElementTypeInt32), "i32");
  EXPECT_EQ(absl::StrFormat("%v", kLiteRtElementTypeFloat32), "f32");
}

TEST(LitertCTypesPrintingTest, LiteRtLayoutScalar) {
  LiteRtLayout layout = {0, false, {}, {}};
  EXPECT_EQ(absl::StrFormat("%v", layout), "<>");
}

TEST(LitertCTypesPrintingTest, LiteRtLayoutMultiDim) {
  LiteRtLayout layout = {2, false, {1, 2}, {}};
  EXPECT_EQ(absl::StrFormat("%v", layout), "<1x2>");
}

TEST(LitertCTypesPrintingTest, LiteRtRankedTensorTypeScalar) {
  LiteRtRankedTensorType type = {
      kLiteRtElementTypeInt32,
      {0, false, {}, {}},
  };
  EXPECT_EQ(absl::StrFormat("%v", type), "0d_i32<>");
}

TEST(LitertCTypesPrintingTest, LiteRtRankedTensorTypeMultiDim) {
  LiteRtRankedTensorType type = {
      kLiteRtElementTypeFloat32,
      {2, false, {1, 2}, {}},
  };
  EXPECT_EQ(absl::StrFormat("%v", type), "2d_f32<1x2>");
}

TEST(LitertCTypesPrintingTest, LiteRtOpCode) {
  EXPECT_EQ(absl::StrFormat("%v", kLiteRtOpCodeTflAdd), "tfl.add");
  EXPECT_EQ(absl::StrFormat("%v", kLiteRtOpCodeTflMul), "tfl.mul");
  EXPECT_EQ(absl::StrFormat("%v", kLiteRtOpCodeTflCustom), "tfl.custom_op");
}

}  // namespace
}  // namespace litert
