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

// Host-only tests (no GPU) of the INT4 FullyConnected weight preparation of
// `CompiledMlDriftModelRunner`: the packed rearrangement must produce the same
// bytes and `sum_i` as unpacking to INT8 and rearranging that.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/weights_conversion.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
#include "tensor/runners/ml_drift/compiled_ml_drift_model_runner.h"

namespace litert::tensor::ml_drift_runner_internal {
namespace {

using ::testing::ElementsAreArray;

struct Int4Case {
  int o;
  int i;
  ml_drift::WeightsLayout layout;
};

class RearrangeInt4WeightsTest : public ::testing::TestWithParam<Int4Case> {};

// Dense signed INT4 values (two per byte, low nibble first) covering -8..7.
std::vector<int8_t> PackedInt4(int num_elements) {
  std::vector<int8_t> packed((num_elements + 1) / 2, 0);
  for (int k = 0; k < num_elements; ++k) {
    const int value = (k * 5 + 3) % 16 - 8;
    const uint8_t nibble = static_cast<uint8_t>(value) & 0xF;
    uint8_t& byte = reinterpret_cast<uint8_t&>(packed[k / 2]);
    byte |= (k % 2 == 0) ? nibble : static_cast<uint8_t>(nibble << 4);
  }
  return packed;
}

TEST_P(RearrangeInt4WeightsTest, MatchesUnpackingToInt8) {
  const Int4Case& c = GetParam();
  const ml_drift::OHWI ohwi(c.o, 1, 1, c.i);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint4;
  desc.layout = c.layout;
  desc.output_group_size = ml_drift::DivideRoundUp(c.o, 4);
  const int elements_count =
      ml_drift::GetTotalElementsCountForLayout(desc, ohwi);
  const int num_elements = c.o * c.i;
  const std::vector<int8_t> packed = PackedInt4(num_elements);

  ASSERT_OK_AND_ASSIGN(
      const RearrangedInt4Weights actual,
      RearrangeInt4Weights(packed, ohwi, desc, elements_count,
                           /*want_sum_i=*/true));

  // The previous path: unpack to INT8, rearrange, sum the unpacked rows.
  ml_drift::Tensor<ml_drift::OHWI, ml_drift::DataType::kInt8> unpacked;
  unpacked.shape = ohwi;
  unpacked.data.assign(num_elements + 64, 0);
  ml_drift::UnpackDenseInt4IntoInt8(packed.data(), num_elements,
                                    unpacked.data.data());
  std::vector<uint8_t> expected(
      (static_cast<size_t>(elements_count) * 4 + 7) / 8);
  ml_drift::RearrangeWeightsInt8AsUint4(unpacked, desc,
                                        absl::MakeSpan(expected), 8, 8u);
  EXPECT_THAT(actual.data, ElementsAreArray(expected));
  ASSERT_TRUE(actual.sum_i.has_value());
  const absl::Span<const int8_t> unpacked_rows =
      absl::MakeConstSpan(unpacked.data).first(num_elements);
  EXPECT_THAT(*actual.sum_i,
              ElementsAreArray(ComputeWeightsSumI(ohwi, unpacked_rows)));
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, RearrangeInt4WeightsTest,
    ::testing::Values(
        // Odd input sizes (rows straddle bytes) and output counts that are not
        // a multiple of 4, for both packed layouts.
        Int4Case{6, 7, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int4Case{5, 9, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int4Case{8, 32, ml_drift::WeightsLayout::kOSpatialIOGroupI4O4},
        Int4Case{6, 7,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
        Int4Case{5, 9,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4},
        Int4Case{8, 32,
                 ml_drift::WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4}),
    [](const ::testing::TestParamInfo<Int4Case>& info) {
      return absl::StrCat(
          info.param.layout == ml_drift::WeightsLayout::kOSpatialIOGroupI4O4
              ? "Buffer"
              : "Texture",
          "O", info.param.o, "I", info.param.i);
    });

TEST(RearrangeInt4WeightsTest, RejectsAShortBuffer) {
  const ml_drift::OHWI ohwi(4, 1, 1, 8);
  ml_drift::WeightsDescription desc;
  desc.type = ml_drift::DataType::kUint4;
  desc.layout = ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
  desc.output_group_size = 1;
  const std::vector<int8_t> packed(15, 0);  // 16 bytes needed.
  EXPECT_THAT(
      RearrangeInt4Weights(packed, ohwi, desc,
                           ml_drift::GetTotalElementsCountForLayout(desc, ohwi),
                           /*want_sum_i=*/false),
      ::absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::tensor::ml_drift_runner_internal
