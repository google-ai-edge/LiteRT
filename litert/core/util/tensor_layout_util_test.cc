// Copyright 2026 Google LLC.
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

#include "litert/core/util/tensor_layout_util.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::litert::IsError;
using ::testing::litert::IsOk;

std::string FormatShape(absl::Span<const int32_t> dims) {
  return absl::StrCat("[", absl::StrJoin(dims, ","), "]");
}

TEST(TensorLayoutUtilTest, CheckedByteCountReturnsProductForValidInputs) {
  LITERT_ASSERT_OK_AND_ASSIGN(size_t bytes,
                              CheckedByteCount(16, sizeof(float)));
  EXPECT_EQ(bytes, 64);

  LITERT_ASSERT_OK_AND_ASSIGN(size_t empty_bytes,
                              CheckedByteCount(0, sizeof(float)));
  EXPECT_EQ(empty_bytes, 0);

  constexpr size_t kMaxElemSize =
      static_cast<size_t>(std::numeric_limits<int>::max());
  LITERT_ASSERT_OK_AND_ASSIGN(size_t max_elem_bytes,
                              CheckedByteCount(1, kMaxElemSize));
  EXPECT_EQ(max_elem_bytes, kMaxElemSize);
}

TEST(TensorLayoutUtilTest, CheckedByteCountRejectsZeroElementSize) {
  EXPECT_THAT(CheckedByteCount(4, 0),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(CheckedByteCount(0, 0),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(TensorLayoutUtilTest, CheckedByteCountRejectsElementSizeAboveIntMax) {
  constexpr size_t kTooLarge =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_THAT(CheckedByteCount(1, kTooLarge),
              IsError(kLiteRtStatusErrorInvalidArgument));

  if constexpr (sizeof(size_t) > sizeof(unsigned int)) {
    // Truncating via `static_cast<int>` wraps to +1, verifying the upper-bound
    // guard catches values that `GetNumBytesFromElements` would otherwise
    // accept.
    constexpr size_t kWrapsPositive =
        static_cast<size_t>(std::numeric_limits<unsigned int>::max()) + 2;
    EXPECT_THAT(CheckedByteCount(1, kWrapsPositive),
                IsError(kLiteRtStatusErrorInvalidArgument));
  }
}

TEST(TensorLayoutUtilTest, CheckedByteCountRejectsOverflow) {
  EXPECT_THAT(CheckedByteCount(std::numeric_limits<size_t>::max() / 2 + 1, 4),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

struct ContiguousValidCase {
  std::string name;
  std::vector<int32_t> dims;
  size_t elem_size;
  std::vector<size_t> expected_strides;
  size_t expected_elements;
  size_t expected_bytes;
};

using PrepareContiguousLayoutValidTest =
    ::testing::TestWithParam<ContiguousValidCase>;

TEST_P(PrepareContiguousLayoutValidTest, ComputesStridesElementsAndBytes) {
  const ContiguousValidCase& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(ContiguousLayout layout,
                              PrepareContiguousLayout(tc.dims, tc.elem_size));
  EXPECT_THAT(layout.strides, ElementsAreArray(tc.expected_strides));
  EXPECT_EQ(layout.num_elements, tc.expected_elements);
  EXPECT_EQ(layout.num_bytes, tc.expected_bytes);
  EXPECT_EQ(layout.IsEmpty(), tc.expected_elements == 0);
}

INSTANTIATE_TEST_SUITE_P(
    PrepareContiguousLayoutValidTests, PrepareContiguousLayoutValidTest,
    ::testing::Values(
        ContiguousValidCase{
            .name = "Scalar",
            .dims = {},
            .elem_size = sizeof(int32_t),
            .expected_strides = {},
            .expected_elements = 1,
            .expected_bytes = 4,
        },
        ContiguousValidCase{
            .name = "3DPositiveShape",
            .dims = {2, 3, 4},
            .elem_size = sizeof(float),
            .expected_strides = {12, 4, 1},
            .expected_elements = 24,
            .expected_bytes = 96,
        },
        ContiguousValidCase{
            .name = "ZeroDimensionWithLargeTrailingDims",
            .dims = {0, std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max()},
            .elem_size = sizeof(float),
            .expected_strides = {0, 0, 0},
            .expected_elements = 0,
            .expected_bytes = 0,
        }),
    [](const ::testing::TestParamInfo<ContiguousValidCase>& info) {
      return info.param.name;
    });

struct ContiguousInvalidCase {
  std::string name;
  std::vector<int32_t> dims;
  size_t elem_size;
};

using PrepareContiguousLayoutInvalidTest =
    ::testing::TestWithParam<ContiguousInvalidCase>;

TEST_P(PrepareContiguousLayoutInvalidTest, RejectsInvalidInputs) {
  const auto& tc = GetParam();
  EXPECT_THAT(PrepareContiguousLayout(tc.dims, tc.elem_size),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    PrepareContiguousLayoutInvalidTests, PrepareContiguousLayoutInvalidTest,
    ::testing::Values(
        ContiguousInvalidCase{
            .name = "NegativeDimension",
            .dims = {2, -1},
            .elem_size = 1,
        },
        ContiguousInvalidCase{
            .name = "ElementCountOverflow",
            .dims = {std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max()},
            .elem_size = 1,
        },
        ContiguousInvalidCase{
            .name = "ByteCountOverflow",
            .dims = {1 << 30, 1 << 30, 4},
            .elem_size = 8,
        },
        ContiguousInvalidCase{
            .name = "ZeroElemSize",
            .dims = {2, 3},
            .elem_size = 0,
        }),
    [](const ::testing::TestParamInfo<ContiguousInvalidCase>& info) {
      return info.param.name;
    });

struct BroadcastValidCase {
  std::string name;
  std::vector<int32_t> input_dims;
  std::vector<int32_t> output_dims;
  size_t elem_size;
  std::vector<size_t> expected_strides;
  size_t expected_elements;
  size_t expected_bytes;
};

using PrepareBroadcastLayoutValidTest =
    ::testing::TestWithParam<BroadcastValidCase>;

TEST_P(PrepareBroadcastLayoutValidTest, ComputesBroadcastStridesAndCounts) {
  const BroadcastValidCase& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(
      BroadcastLayout layout,
      PrepareBroadcastLayout(tc.input_dims, tc.output_dims, tc.elem_size));
  EXPECT_THAT(layout.strides, ElementsAreArray(tc.expected_strides));
  EXPECT_EQ(layout.num_elements, tc.expected_elements);
  EXPECT_EQ(layout.num_bytes, tc.expected_bytes);
  EXPECT_EQ(layout.IsEmpty(), tc.expected_elements == 0);
}

INSTANTIATE_TEST_SUITE_P(
    PrepareBroadcastLayoutValidTests, PrepareBroadcastLayoutValidTest,
    ::testing::Values(
        BroadcastValidCase{
            .name = "ScalarToScalar",
            .input_dims = {},
            .output_dims = {},
            .elem_size = sizeof(int32_t),
            .expected_strides = {},
            .expected_elements = 1,
            .expected_bytes = 4,
        },
        BroadcastValidCase{
            .name = "ScalarTo3D",
            .input_dims = {},
            .output_dims = {2, 3, 4},
            .elem_size = sizeof(int16_t),
            .expected_strides = {0, 0, 0},
            .expected_elements = 24,
            .expected_bytes = 48,
        },
        BroadcastValidCase{
            .name = "MiddleSingletonBroadcast",
            .input_dims = {2, 1, 4},
            .output_dims = {5, 2, 3, 4},
            .elem_size = sizeof(float),
            .expected_strides = {0, 4, 0, 1},
            .expected_elements = 120,
            .expected_bytes = 480,
        },
        BroadcastValidCase{
            .name = "EmptySingletonToEmptyOutput",
            .input_dims = {1},
            .output_dims = {0},
            .elem_size = sizeof(float),
            .expected_strides = {0},
            .expected_elements = 0,
            .expected_bytes = 0,
        },
        BroadcastValidCase{
            .name = "BothEmptyOutput",
            .input_dims = {2, 0},
            .output_dims = {2, 0},
            .elem_size = sizeof(float),
            .expected_strides = {0, 0},
            .expected_elements = 0,
            .expected_bytes = 0,
        }),
    [](const ::testing::TestParamInfo<BroadcastValidCase>& info) {
      return info.param.name;
    });

struct BroadcastInvalidCase {
  std::string name;
  std::vector<int32_t> input_dims;
  std::vector<int32_t> output_dims;
  size_t elem_size = 1;
};

using PrepareBroadcastLayoutInvalidTest =
    ::testing::TestWithParam<BroadcastInvalidCase>;

TEST_P(PrepareBroadcastLayoutInvalidTest, RejectsInvalidShapes) {
  const auto& tc = GetParam();
  EXPECT_THAT(
      PrepareBroadcastLayout(tc.input_dims, tc.output_dims, tc.elem_size),
      IsError(kLiteRtStatusErrorInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    PrepareBroadcastLayoutInvalidTests, PrepareBroadcastLayoutInvalidTest,
    ::testing::Values(
        BroadcastInvalidCase{
            .name = "InputRankExceedsOutputRank",
            .input_dims = {1, 2, 3},
            .output_dims = {2, 3},
        },
        BroadcastInvalidCase{
            .name = "NegativeInputDim",
            .input_dims = {-1, 3},
            .output_dims = {2, 3},
        },
        BroadcastInvalidCase{
            .name = "NegativeOutputDim",
            .input_dims = {1, 3},
            .output_dims = {-1, 3},
        },
        BroadcastInvalidCase{
            .name = "IncompatibleDim",
            .input_dims = {2, 3},
            .output_dims = {4, 3},
        },
        BroadcastInvalidCase{
            .name = "IncompatibleNonZeroDimWithZeroDim",
            .input_dims = {2, 0},
            .output_dims = {3, 0},
        },
        BroadcastInvalidCase{
            .name = "EmptyInputIntoNonEmptyOutput",
            .input_dims = {0},
            .output_dims = {2},
        },
        BroadcastInvalidCase{
            .name = "InputElementCountOverflow",
            .input_dims = {std::numeric_limits<int32_t>::max(),
                           std::numeric_limits<int32_t>::max(),
                           std::numeric_limits<int32_t>::max()},
            .output_dims = {std::numeric_limits<int32_t>::max(),
                            std::numeric_limits<int32_t>::max(),
                            std::numeric_limits<int32_t>::max()},
            .elem_size = 1,
        },
        BroadcastInvalidCase{
            .name = "OutputElementCountOverflow",
            .input_dims = {1, 1, 1},
            .output_dims = {std::numeric_limits<int32_t>::max(),
                            std::numeric_limits<int32_t>::max(),
                            std::numeric_limits<int32_t>::max()},
            .elem_size = 1,
        },
        BroadcastInvalidCase{
            .name = "InputByteOverflow",
            .input_dims = {1 << 30, 1 << 30, 4},
            .output_dims = {1 << 30, 1 << 30, 4},
            .elem_size = 8,
        },
        BroadcastInvalidCase{
            .name = "OutputByteOverflow",
            .input_dims = {1, 1, 1},
            .output_dims = {1 << 30, 1 << 30, 4},
            .elem_size = 8,
        },
        BroadcastInvalidCase{
            .name = "ZeroElemSize",
            .input_dims = {1, 3},
            .output_dims = {2, 3},
            .elem_size = 0,
        }),
    [](const ::testing::TestParamInfo<BroadcastInvalidCase>& info) {
      return info.param.name;
    });

struct ConcreteBroadcastCase {
  std::string name;
  std::vector<int32_t> lhs;
  std::vector<int32_t> rhs;
  std::vector<int32_t> expected;
};

using ComputeConcreteBroadcastShapeValidTest =
    ::testing::TestWithParam<ConcreteBroadcastCase>;

TEST_P(ComputeConcreteBroadcastShapeValidTest, MergesConcreteBroadcastDims) {
  const ConcreteBroadcastCase& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<int32_t> out,
                              ComputeConcreteBroadcastShape(tc.lhs, tc.rhs));
  EXPECT_THAT(out, ElementsAreArray(tc.expected));
}

INSTANTIATE_TEST_SUITE_P(
    ComputeConcreteBroadcastShapeValidTests,
    ComputeConcreteBroadcastShapeValidTest,
    ::testing::Values(
        ConcreteBroadcastCase{
            .name = "BidirectionalWithRankDiff",
            .lhs = {2, 1, 4},
            .rhs = {3, 1},
            .expected = {2, 3, 4},
        },
        ConcreteBroadcastCase{
            .name = "ZeroAndOneYieldsZero",
            .lhs = {0},
            .rhs = {1},
            .expected = {0},
        },
        ConcreteBroadcastCase{
            .name = "OneAndZeroYieldsZero",
            .lhs = {1},
            .rhs = {0},
            .expected = {0},
        },
        ConcreteBroadcastCase{
            .name = "ZeroAndZeroYieldsZero",
            .lhs = {2, 0},
            .rhs = {1, 0},
            .expected = {2, 0},
        }),
    [](const ::testing::TestParamInfo<ConcreteBroadcastCase>& info) {
      return info.param.name;
    });

TEST(TensorLayoutUtilTest, ComputeConcreteBroadcastShapeRejectsNegativeDim) {
  const std::vector<int32_t> lhs = {-1, 3};
  const std::vector<int32_t> rhs = {2, 3};
  EXPECT_THAT(ComputeConcreteBroadcastShape(lhs, rhs),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(TensorLayoutUtilTest, ComputeConcreteBroadcastShapeRejectsIncompatible) {
  const std::vector<int32_t> lhs = {2, 3};
  const std::vector<int32_t> rhs = {2, 4};
  EXPECT_THAT(ComputeConcreteBroadcastShape(lhs, rhs),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(TensorLayoutUtilTest,
     ComputeConcreteBroadcastShapeRejectsZeroVsGreaterThanOne) {
  const std::vector<int32_t> zero = {0};
  const std::vector<int32_t> two = {2};
  EXPECT_THAT(ComputeConcreteBroadcastShape(zero, two),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(ComputeConcreteBroadcastShape(two, zero),
              IsError(kLiteRtStatusErrorInvalidArgument));

  const std::vector<int32_t> lhs_2_0 = {2, 0};
  const std::vector<int32_t> rhs_3_0 = {3, 0};
  EXPECT_THAT(ComputeConcreteBroadcastShape(lhs_2_0, rhs_3_0),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(TensorLayoutUtilTest, ComputeConcreteBroadcastShapeNaryValidAndInvalid) {
  const int32_t a[] = {2, 1, 4};
  const int32_t b[] = {1, 3, 1};
  const int32_t c[] = {4};
  const std::array<absl::Span<const int32_t>, 3> valid_3 = {a, b, c};
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<int32_t> out_3,
                              ComputeConcreteBroadcastShape(valid_3));
  EXPECT_THAT(out_3, ElementsAre(2, 3, 4));

  const int32_t az[] = {2, 1, 0};
  const int32_t cz[] = {0};
  const std::array<absl::Span<const int32_t>, 3> valid_zero_3 = {az, b, cz};
  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<int32_t> out_zero_3,
                              ComputeConcreteBroadcastShape(valid_zero_3));
  EXPECT_THAT(out_zero_3, ElementsAre(2, 3, 0));

  // Broadcasting `0` with `>= 2` is rejected even when another shape/dim is 0.
  const int32_t two[] = {2};
  const std::array<absl::Span<const int32_t>, 3> bad_zero_vs_two = {cz, two,
                                                                    cz};
  EXPECT_THAT(ComputeConcreteBroadcastShape(bad_zero_vs_two),
              IsError(kLiteRtStatusErrorInvalidArgument));

  const int32_t s0[] = {1, 0};
  const int32_t s1[] = {2, 1};
  const int32_t s2[] = {3, 0};
  const std::array<absl::Span<const int32_t>, 3> bad_with_zero_dim = {s0, s1,
                                                                      s2};
  EXPECT_THAT(ComputeConcreteBroadcastShape(bad_with_zero_dim),
              IsError(kLiteRtStatusErrorInvalidArgument));

  const int32_t neg[] = {1, -1};
  const std::array<absl::Span<const int32_t>, 3> bad_neg = {s1, neg, s1};
  EXPECT_THAT(ComputeConcreteBroadcastShape(bad_neg),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

struct AxisLayoutValidCase {
  std::string name;
  std::vector<int32_t> dims;
  int32_t axis;
  size_t elem_size;
  size_t expected_outer;
  size_t expected_axis;
  size_t expected_inner;
  size_t expected_elements;
  size_t expected_bytes;
};

using PrepareAxisLayoutValidTest =
    ::testing::TestWithParam<AxisLayoutValidCase>;

TEST_P(PrepareAxisLayoutValidTest, ComputesOuterAxisInnerAndBytes) {
  const AxisLayoutValidCase& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(
      AxisLayout layout, PrepareAxisLayout(tc.dims, tc.axis, tc.elem_size));
  EXPECT_EQ(layout.outer_size, tc.expected_outer);
  EXPECT_EQ(layout.axis_size, tc.expected_axis);
  EXPECT_EQ(layout.inner_size, tc.expected_inner);
  EXPECT_EQ(layout.num_elements, tc.expected_elements);
  EXPECT_EQ(layout.num_bytes, tc.expected_bytes);
  EXPECT_EQ(layout.IsEmpty(), tc.expected_elements == 0);
}

INSTANTIATE_TEST_SUITE_P(
    PrepareAxisLayoutValidTests, PrepareAxisLayoutValidTest,
    ::testing::Values(
        AxisLayoutValidCase{
            .name = "LeadingAxisZero",
            .dims = {2, 3, 4},
            .axis = 0,
            .elem_size = sizeof(int32_t),
            .expected_outer = 1,
            .expected_axis = 2,
            .expected_inner = 12,
            .expected_elements = 24,
            .expected_bytes = 96,
        },
        AxisLayoutValidCase{
            .name = "LeadingAxisNegativeRank",
            .dims = {2, 3, 4},
            .axis = -3,
            .elem_size = sizeof(int32_t),
            .expected_outer = 1,
            .expected_axis = 2,
            .expected_inner = 12,
            .expected_elements = 24,
            .expected_bytes = 96,
        },
        AxisLayoutValidCase{
            .name = "MiddleAxisPositive",
            .dims = {2, 3, 4},
            .axis = 1,
            .elem_size = sizeof(int32_t),
            .expected_outer = 2,
            .expected_axis = 3,
            .expected_inner = 4,
            .expected_elements = 24,
            .expected_bytes = 96,
        },
        AxisLayoutValidCase{
            .name = "LastAxisNegative",
            .dims = {2, 3, 4},
            .axis = -1,
            .elem_size = sizeof(int16_t),
            .expected_outer = 6,
            .expected_axis = 4,
            .expected_inner = 1,
            .expected_elements = 24,
            .expected_bytes = 48,
        },
        AxisLayoutValidCase{
            .name = "ZeroOuterDimension",
            .dims = {0, 3, 4},
            .axis = 1,
            .elem_size = sizeof(float),
            .expected_outer = 0,
            .expected_axis = 3,
            .expected_inner = 4,
            .expected_elements = 0,
            .expected_bytes = 0,
        },
        AxisLayoutValidCase{
            .name = "ZeroAxisDimensionPreservesOuterAndInner",
            .dims = {2, 0, 4},
            .axis = 1,
            .elem_size = sizeof(float),
            .expected_outer = 2,
            .expected_axis = 0,
            .expected_inner = 4,
            .expected_elements = 0,
            .expected_bytes = 0,
        }),
    [](const ::testing::TestParamInfo<AxisLayoutValidCase>& info) {
      return info.param.name;
    });

struct AxisLayoutInvalidCase {
  std::string name;
  std::vector<int32_t> dims;
  int32_t axis;
  size_t elem_size = 1;
};

using PrepareAxisLayoutInvalidTest =
    ::testing::TestWithParam<AxisLayoutInvalidCase>;

TEST_P(PrepareAxisLayoutInvalidTest, RejectsInvalidArgs) {
  const AxisLayoutInvalidCase& tc = GetParam();
  EXPECT_THAT(PrepareAxisLayout(tc.dims, tc.axis, tc.elem_size),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    PrepareAxisLayoutInvalidTests, PrepareAxisLayoutInvalidTest,
    ::testing::Values(
        AxisLayoutInvalidCase{.name = "EmptyDims", .dims = {}, .axis = 0},
        AxisLayoutInvalidCase{
            .name = "EmptyDimsInt32Min",
            .dims = {},
            .axis = std::numeric_limits<int32_t>::min(),
        },
        AxisLayoutInvalidCase{
            .name = "AxisTooLarge", .dims = {2, 3}, .axis = 2},
        AxisLayoutInvalidCase{
            .name = "AxisTooNegative", .dims = {2, 3}, .axis = -3},
        AxisLayoutInvalidCase{
            .name = "AxisInt32Min",
            .dims = {2, 3},
            .axis = std::numeric_limits<int32_t>::min(),
        },
        AxisLayoutInvalidCase{
            .name = "AxisInt32Max",
            .dims = {2, 3},
            .axis = std::numeric_limits<int32_t>::max(),
        },
        AxisLayoutInvalidCase{
            .name = "NegativeDim", .dims = {2, -1}, .axis = 0},
        AxisLayoutInvalidCase{
            .name = "ZeroElemSize",
            .dims = {2, 3},
            .axis = 0,
            .elem_size = 0,
        },
        AxisLayoutInvalidCase{
            .name = "ElementCountOverflow",
            .dims = {std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max()},
            .axis = 0,
            .elem_size = 1,
        },
        AxisLayoutInvalidCase{
            .name = "ByteOverflow",
            .dims = {1 << 30, 1 << 30, 4},
            .axis = 0,
            .elem_size = 8,
        },
        AxisLayoutInvalidCase{
            .name = "OuterOverflowWithZeroDim",
            .dims = {std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max(), 0},
            .axis = 3,
        },
        AxisLayoutInvalidCase{
            .name = "InnerOverflowWithZeroDim",
            .dims = {0, std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max(),
                     std::numeric_limits<int32_t>::max()},
            .axis = 0,
        }),
    [](const ::testing::TestParamInfo<AxisLayoutInvalidCase>& info) {
      return info.param.name;
    });

TEST(TensorLayoutUtilTest, PrepareAxisLayoutRejectsRankAboveInt32Max) {
  const absl::Span<const int32_t> huge_rank(
      /*array=*/nullptr,
      static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1);
  EXPECT_THAT(PrepareAxisLayout(huge_rank, 0),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

struct InsertAxisValidCase {
  std::string name;
  std::vector<int32_t> dims;
  int32_t axis;
  size_t elem_size;
  size_t expected_outer;
  size_t expected_inner;
  size_t expected_elements;
  size_t expected_bytes;
};

using PrepareInsertAxisLayoutValidTest =
    ::testing::TestWithParam<InsertAxisValidCase>;

TEST_P(PrepareInsertAxisLayoutValidTest, ComputesOuterInnerAndBytes) {
  const InsertAxisValidCase& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(
      InsertAxisLayout layout,
      PrepareInsertAxisLayout(tc.dims, tc.axis, tc.elem_size));
  EXPECT_EQ(layout.outer_size, tc.expected_outer);
  EXPECT_EQ(layout.inner_size, tc.expected_inner);
  EXPECT_EQ(layout.num_elements, tc.expected_elements);
  EXPECT_EQ(layout.num_bytes, tc.expected_bytes);
  EXPECT_EQ(layout.IsEmpty(), tc.expected_elements == 0);
}

INSTANTIATE_TEST_SUITE_P(
    PrepareInsertAxisLayoutValidTests, PrepareInsertAxisLayoutValidTest,
    ::testing::Values(
        InsertAxisValidCase{
            .name = "ScalarAxisZero",
            .dims = {},
            .axis = 0,
            .elem_size = sizeof(int32_t),
            .expected_outer = 1,
            .expected_inner = 1,
            .expected_elements = 1,
            .expected_bytes = 4,
        },
        InsertAxisValidCase{
            .name = "ScalarAxisNegOne",
            .dims = {},
            .axis = -1,
            .elem_size = sizeof(int32_t),
            .expected_outer = 1,
            .expected_inner = 1,
            .expected_elements = 1,
            .expected_bytes = 4,
        },
        InsertAxisValidCase{
            .name = "NegativeAxisInsertsAtEnd",
            .dims = {2, 3},
            .axis = -1,
            .elem_size = sizeof(float),
            .expected_outer = 6,
            .expected_inner = 1,
            .expected_elements = 6,
            .expected_bytes = 24,
        },
        InsertAxisValidCase{
            .name = "AxisEqualsRank",
            .dims = {2, 3},
            .axis = 2,
            .elem_size = sizeof(float),
            .expected_outer = 6,
            .expected_inner = 1,
            .expected_elements = 6,
            .expected_bytes = 24,
        },
        InsertAxisValidCase{
            .name = "AxisEqualsNegRankMinusOne",
            .dims = {2, 3},
            .axis = -3,
            .elem_size = sizeof(float),
            .expected_outer = 1,
            .expected_inner = 6,
            .expected_elements = 6,
            .expected_bytes = 24,
        },
        InsertAxisValidCase{
            .name = "MiddleAxis",
            .dims = {2, 3},
            .axis = 1,
            .elem_size = sizeof(int16_t),
            .expected_outer = 2,
            .expected_inner = 3,
            .expected_elements = 6,
            .expected_bytes = 12,
        },
        InsertAxisValidCase{
            .name = "EmptyTensor",
            .dims = {2, 0},
            .axis = 1,
            .elem_size = sizeof(float),
            .expected_outer = 2,
            .expected_inner = 0,
            .expected_elements = 0,
            .expected_bytes = 0,
        }),
    [](const ::testing::TestParamInfo<InsertAxisValidCase>& info) {
      return info.param.name;
    });

using PrepareInsertAxisLayoutInvalidTest =
    ::testing::TestWithParam<AxisLayoutInvalidCase>;

TEST_P(PrepareInsertAxisLayoutInvalidTest, RejectsInvalidArgs) {
  const AxisLayoutInvalidCase& tc = GetParam();
  EXPECT_THAT(PrepareInsertAxisLayout(tc.dims, tc.axis, tc.elem_size),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    PrepareInsertAxisLayoutInvalidTests, PrepareInsertAxisLayoutInvalidTest,
    ::testing::Values(AxisLayoutInvalidCase{.name = "AxisAboveRank",
                                            .dims = {2, 3},
                                            .axis = 3},
                      AxisLayoutInvalidCase{.name = "AxisBelowNegRankMinusOne",
                                            .dims = {2, 3},
                                            .axis = -4},
                      AxisLayoutInvalidCase{
                          .name = "AxisInt32Min",
                          .dims = {2, 3},
                          .axis = std::numeric_limits<int32_t>::min(),
                      },
                      AxisLayoutInvalidCase{
                          .name = "AxisInt32Max",
                          .dims = {2, 3},
                          .axis = std::numeric_limits<int32_t>::max(),
                      },
                      AxisLayoutInvalidCase{
                          .name = "NegativeDim", .dims = {2, -1}, .axis = 0},
                      AxisLayoutInvalidCase{
                          .name = "ZeroElemSize",
                          .dims = {2, 3},
                          .axis = 0,
                          .elem_size = 0,
                      },
                      AxisLayoutInvalidCase{
                          .name = "ElementCountOverflow",
                          .dims = {std::numeric_limits<int32_t>::max(),
                                   std::numeric_limits<int32_t>::max(),
                                   std::numeric_limits<int32_t>::max()},
                          .axis = 0,
                          .elem_size = 1,
                      },
                      AxisLayoutInvalidCase{
                          .name = "ByteOverflow",
                          .dims = {1 << 30, 1 << 30, 4},
                          .axis = 0,
                          .elem_size = 8,
                      },
                      AxisLayoutInvalidCase{
                          .name = "OuterOverflowWithZeroDim",
                          .dims = {std::numeric_limits<int32_t>::max(),
                                   std::numeric_limits<int32_t>::max(),
                                   std::numeric_limits<int32_t>::max(), 0},
                          .axis = 3,
                      },
                      AxisLayoutInvalidCase{
                          .name = "InnerOverflowWithZeroDim",
                          .dims = {0, std::numeric_limits<int32_t>::max(),
                                   std::numeric_limits<int32_t>::max(),
                                   std::numeric_limits<int32_t>::max()},
                          .axis = 1,
                      }),
    [](const ::testing::TestParamInfo<AxisLayoutInvalidCase>& info) {
      return info.param.name;
    });

TEST(TensorLayoutUtilTest, PrepareInsertAxisLayoutRejectsRankAtInt32Max) {
  const absl::Span<const int32_t> max_rank(
      /*array=*/nullptr,
      static_cast<size_t>(std::numeric_limits<int32_t>::max()));
  EXPECT_THAT(PrepareInsertAxisLayout(max_rank, 0),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(TensorLayoutUtilTest, DefaultConstructedLayoutsAreEmptyAndZeroed) {
  const ContiguousLayout contiguous;
  EXPECT_TRUE(contiguous.IsEmpty());
  EXPECT_TRUE(contiguous.strides.empty());
  EXPECT_EQ(contiguous.num_elements, 0);
  EXPECT_EQ(contiguous.num_bytes, 0);

  const BroadcastLayout broadcast;
  EXPECT_TRUE(broadcast.IsEmpty());
  EXPECT_TRUE(broadcast.strides.empty());
  EXPECT_EQ(broadcast.num_elements, 0);
  EXPECT_EQ(broadcast.num_bytes, 0);

  const AxisLayout axis;
  EXPECT_TRUE(axis.IsEmpty());
  EXPECT_EQ(axis.outer_size, 0);
  EXPECT_EQ(axis.axis_size, 0);
  EXPECT_EQ(axis.inner_size, 0);
  EXPECT_EQ(axis.num_elements, 0);
  EXPECT_EQ(axis.num_bytes, 0);

  const InsertAxisLayout insert_axis;
  EXPECT_TRUE(insert_axis.IsEmpty());
  EXPECT_EQ(insert_axis.outer_size, 0);
  EXPECT_EQ(insert_axis.inner_size, 0);
  EXPECT_EQ(insert_axis.num_elements, 0);
  EXPECT_EQ(insert_axis.num_bytes, 0);
}

TEST(TensorLayoutUtilTest, ValidateBufferCapacityChecksLengthAndNullity) {
  const int32_t vals[3] = {1, 2, 3};
  EXPECT_THAT(ValidateBufferCapacity(absl::MakeConstSpan(vals), 3), IsOk());
  EXPECT_THAT(ValidateBufferCapacity(absl::MakeConstSpan(vals), 2), IsOk());
  EXPECT_THAT(ValidateBufferCapacity(absl::Span<const int32_t>(), 0), IsOk());
  EXPECT_THAT(
      ValidateBufferCapacity(
          absl::Span<const int32_t>(/*array=*/nullptr, /*length=*/4), 0),
      IsOk());

  Expected<void> undersized =
      ValidateBufferCapacity(absl::MakeConstSpan(vals).first(2), 3);
  ASSERT_THAT(undersized, IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(undersized.Error().Message(), HasSubstr("smaller"));

  EXPECT_THAT(ValidateBufferCapacity(absl::Span<const int32_t>(), 1),
              IsError(kLiteRtStatusErrorInvalidArgument));

  Expected<void> null_non_empty = ValidateBufferCapacity(
      absl::Span<const int32_t>(/*array=*/nullptr, /*length=*/4), 4);
  ASSERT_THAT(null_non_empty, IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(null_non_empty.Error().Message(), HasSubstr("null"));
}

TEST(TensorLayoutUtilTest, MakeDimsSpanValidatesRankAndNullity) {
  const int32_t dims[] = {2, 3, 4};

  EXPECT_THAT(MakeDimsSpan(dims, -1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(MakeDimsSpan(/*dims=*/nullptr, 2),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_ASSERT_OK_AND_ASSIGN(absl::Span<const int32_t> scalar_dims,
                              MakeDimsSpan(/*dims=*/nullptr, 0));
  EXPECT_TRUE(scalar_dims.empty());

  LITERT_ASSERT_OK_AND_ASSIGN(absl::Span<const int32_t> non_null_scalar_dims,
                              MakeDimsSpan(dims, 0));
  EXPECT_TRUE(non_null_scalar_dims.empty());

  LITERT_ASSERT_OK_AND_ASSIGN(absl::Span<const int32_t> ranked_dims,
                              MakeDimsSpan(dims, 3));
  EXPECT_THAT(ranked_dims, ElementsAre(2, 3, 4));
}

TEST(TensorLayoutUtilTest, TensorViewImplicitlyConvertsToConstTensorView) {
  static_assert(
      std::is_convertible_v<TensorView<int32_t>, ConstTensorView<int32_t>>);
  static_assert(
      !std::is_convertible_v<ConstTensorView<int32_t>, TensorView<int32_t>>);
  static_assert(std::is_trivially_copyable_v<ConstTensorView<int32_t>>);

  int32_t data[] = {10, 20, 30, 40};
  const int32_t dims[] = {2, 2};
  TensorView<int32_t> mutable_view{absl::MakeSpan(data), dims};
  ConstTensorView<int32_t> const_view = mutable_view;
  ConstTensorView<int32_t> copied_const_view = const_view;

  EXPECT_THAT(copied_const_view.data, ElementsAre(10, 20, 30, 40));
  EXPECT_THAT(copied_const_view.dims, ElementsAre(2, 2));
}

TEST(TensorLayoutUtilTest,
     ComputeConcreteBroadcastShapeEmptyListReturnsScalar) {
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<int32_t> out,
      ComputeConcreteBroadcastShape(
          absl::Span<const absl::Span<const int32_t>>{}));
  EXPECT_TRUE(out.empty());
}

// Independent, deliberately naive oracle mapping a flat output index to a flat
// input index via explicit output->input coordinate conversion (shares no
// stride or coalescing code with `BroadcastPlan` / `BroadcastCursor`).
size_t NaiveBroadcastFlatIndex(size_t flat_out_idx,
                               absl::Span<const int32_t> out_dims,
                               absl::Span<const int32_t> in_dims) {
  std::vector<size_t> out_coords(out_dims.size(), 0);
  size_t rem = flat_out_idx;
  for (size_t i = out_dims.size(); i > 0; --i) {
    const size_t d = i - 1;
    const size_t dim = static_cast<size_t>(out_dims[d]);
    out_coords[d] = rem % dim;
    rem /= dim;
  }
  const size_t rank_diff = out_dims.size() - in_dims.size();
  size_t flat_in_idx = 0;
  for (size_t d = 0; d < in_dims.size(); ++d) {
    const size_t in_dim = static_cast<size_t>(in_dims[d]);
    const size_t coord = (in_dim == 1) ? 0 : out_coords[d + rank_diff];
    flat_in_idx = flat_in_idx * in_dim + coord;
  }
  return flat_in_idx;
}

std::vector<std::vector<int32_t>> EnumerateSmallShapes() {
  std::vector<std::vector<int32_t>> shapes;
  shapes.push_back({});  // Rank 0 scalar.
  for (int32_t d0 = 0; d0 <= 3; ++d0) {
    shapes.push_back({d0});
    for (int32_t d1 = 0; d1 <= 3; ++d1) {
      shapes.push_back({d0, d1});
      for (int32_t d2 = 0; d2 <= 3; ++d2) {
        shapes.push_back({d0, d1, d2});
      }
    }
  }
  return shapes;
}

template <typename T, typename = void>
struct HasSpans : std::false_type {};

template <typename T>
struct HasSpans<T, std::void_t<decltype(std::declval<T>().Spans())>>
    : std::true_type {};

template <size_t N>
void VerifyBroadcastPlanAgainstNaiveOracle(
    const BroadcastPlan<N>& plan,
    const std::array<absl::Span<const int32_t>, N>& in_shapes,
    absl::Span<const int32_t> out_dims) {
  EXPECT_THAT(plan.output_shape, ElementsAreArray(out_dims));
  size_t visited_elements = 0;
  size_t visited_spans = 0;
  for (const BroadcastSpan<N>& span : plan.Spans()) {
    ++visited_spans;
    EXPECT_GT(span.count, 0u);
    if (span.count == 0) {
      return;
    }
    EXPECT_EQ(span.output_offset, visited_elements);
    for (size_t k = 0; k < N; ++k) {
      EXPECT_LE(span.input_steps[k], 1u);
    }
    for (size_t i = 0; i < span.count; ++i) {
      const size_t out_idx = span.output_offset + i;
      for (size_t k = 0; k < N; ++k) {
        const size_t actual_in =
            span.input_offsets[k] + i * span.input_steps[k];
        const size_t expected_in =
            NaiveBroadcastFlatIndex(out_idx, out_dims, in_shapes[k]);
        EXPECT_EQ(actual_in, expected_in)
            << "Mismatch at output index " << out_idx << " operand " << k;
      }
    }
    visited_elements += span.count;
  }
  EXPECT_EQ(visited_elements, plan.output_num_elements);
  EXPECT_EQ(visited_spans, plan.NumSpans());
}

TEST(BroadcastPlanTest, MatchesExpectedSpansForAlternatingBroadcastAxes) {
  // Exact 6-span layout for [2,1,4] + [1,3,1] -> [2,3,4].
  const int32_t in0[] = {2, 1, 4};
  const int32_t in1[] = {1, 3, 1};
  const int32_t out[] = {2, 3, 4};

  LITERT_ASSERT_OK_AND_ASSIGN(
      BroadcastPlan<2> plan,
      PrepareBroadcastPlan<2>({in0, in1}, {sizeof(int16_t), sizeof(int32_t)},
                              out, sizeof(double)));

  EXPECT_THAT(plan.output_shape, ElementsAre(2, 3, 4));
  EXPECT_EQ(plan.output_num_elements, 24u);
  EXPECT_EQ(plan.output_num_bytes, 24u * sizeof(double));
  EXPECT_THAT(plan.input_num_elements, ElementsAre(8u, 3u));
  EXPECT_THAT(plan.input_num_bytes,
              ElementsAre(8u * sizeof(int16_t), 3u * sizeof(int32_t)));
  ASSERT_EQ(plan.NumSpans(), 6u);
  std::vector<BroadcastSpan<2>> spans;
  for (const BroadcastSpan<2>& span : plan.Spans()) {
    spans.push_back(span);
  }
  ASSERT_EQ(spans.size(), 6u);
  const size_t expected_out[] = {0, 4, 8, 12, 16, 20};
  const size_t expected_in0[] = {0, 0, 0, 4, 4, 4};
  const size_t expected_in1[] = {0, 1, 2, 0, 1, 2};
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(spans[i].output_offset, expected_out[i]);
    EXPECT_EQ(spans[i].input_offsets[0], expected_in0[i]);
    EXPECT_EQ(spans[i].input_offsets[1], expected_in1[i]);
    EXPECT_EQ(spans[i].count, 4u);
    EXPECT_EQ(spans[i].input_steps[0], 1u);
    EXPECT_EQ(spans[i].input_steps[1], 0u);
  }
}

TEST(BroadcastPlanTest, CoalescesCompatibleAdjacentDimensions) {
  const int32_t a[] = {2, 3, 4};
  const int32_t scalar_like[] = {1, 1, 1};
  const int32_t trailing_bcast[] = {2, 3, 1};

  LITERT_ASSERT_OK_AND_ASSIGN(
      BroadcastPlan<2> full_coalesce,
      PrepareBroadcastPlan<2>({a, scalar_like}, {4, 4}, a, 4));
  EXPECT_EQ(full_coalesce.NumSpans(), 1u);
  EXPECT_EQ(full_coalesce.span_count, 24u);
  EXPECT_THAT(full_coalesce.span_steps, ElementsAre(1u, 0u));

  LITERT_ASSERT_OK_AND_ASSIGN(
      BroadcastPlan<2> outer_coalesce,
      PrepareBroadcastPlan<2>({a, trailing_bcast}, {4, 4}, a, 4));
  EXPECT_EQ(outer_coalesce.NumSpans(), 6u);
  EXPECT_EQ(outer_coalesce.span_count, 4u);
  EXPECT_THAT(outer_coalesce.outer_extents, ElementsAre(6u));
  EXPECT_THAT(outer_coalesce.span_steps, ElementsAre(1u, 0u));
}

TEST(BroadcastPlanTest, UnaryExhaustiveSmallShapesMatchNaiveOracle) {
  const std::vector<std::vector<int32_t>> shapes = EnumerateSmallShapes();
  for (const std::vector<int32_t>& in_dims : shapes) {
    for (const std::vector<int32_t>& out_dims : shapes) {
      SCOPED_TRACE(absl::StrCat("in=", FormatShape(in_dims),
                                " out=", FormatShape(out_dims)));
      bool expected_valid = in_dims.size() <= out_dims.size();
      if (expected_valid) {
        const size_t diff = out_dims.size() - in_dims.size();
        for (size_t d = 0; d < in_dims.size(); ++d) {
          if (in_dims[d] != 1 && in_dims[d] != out_dims[d + diff]) {
            expected_valid = false;
            break;
          }
        }
      }
      Expected<BroadcastPlan<1>> plan = PrepareBroadcastPlan<1>(
          {in_dims}, {sizeof(int32_t)}, out_dims, sizeof(int32_t),
          BroadcastOutputMode::kTargetShape);
      ASSERT_EQ(plan.HasValue(), expected_valid);
      if (expected_valid) {
        VerifyBroadcastPlanAgainstNaiveOracle<1>(*plan, {in_dims}, out_dims);
      }
    }
  }
}

TEST(BroadcastPlanTest, BinaryExhaustiveSmallShapesMatchNaiveOracle) {
  const std::vector<std::vector<int32_t>> shapes = EnumerateSmallShapes();
  for (const std::vector<int32_t>& a_dims : shapes) {
    for (const std::vector<int32_t>& b_dims : shapes) {
      SCOPED_TRACE(
          absl::StrCat("a=", FormatShape(a_dims), " b=", FormatShape(b_dims)));
      Expected<std::vector<int32_t>> expected_out =
          ComputeConcreteBroadcastShape(a_dims, b_dims);
      if (!expected_out) {
        EXPECT_THAT(
            PrepareBroadcastPlan<2>({a_dims, b_dims}, {4, 4}, a_dims, 4),
            IsError(kLiteRtStatusErrorInvalidArgument));
        continue;
      }
      LITERT_ASSERT_OK_AND_ASSIGN(
          BroadcastPlan<2> plan,
          PrepareBroadcastPlan<2>({a_dims, b_dims}, {4, 4}, *expected_out, 4));
      VerifyBroadcastPlanAgainstNaiveOracle<2>(plan, {a_dims, b_dims},
                                               *expected_out);
    }
  }
}

TEST(BroadcastPlanTest, TernaryBroadcastMatrixMatchesNaiveOracle) {
  const std::vector<std::vector<int32_t>> shapes = {
      {},        {0},       {1},       {2},       {3},
      {0, 1},    {1, 0},    {1, 1},    {1, 2},    {2, 1},
      {2, 3},    {0, 2},    {2, 0},    {1, 1, 1}, {2, 1, 3},
      {1, 2, 1}, {1, 1, 3}, {2, 1, 1}, {2, 2, 3}, {0, 1, 3},
  };
  for (const std::vector<int32_t>& c_dims : shapes) {
    for (const std::vector<int32_t>& a_dims : shapes) {
      for (const std::vector<int32_t>& b_dims : shapes) {
        SCOPED_TRACE(absl::StrCat("c=", FormatShape(c_dims),
                                  " a=", FormatShape(a_dims),
                                  " b=", FormatShape(b_dims)));
        const std::array<absl::Span<const int32_t>, 3> in_shapes = {
            c_dims, a_dims, b_dims};
        Expected<std::vector<int32_t>> expected_out =
            ComputeConcreteBroadcastShape(in_shapes);
        if (!expected_out) {
          EXPECT_THAT(PrepareBroadcastPlan<3>(in_shapes, {1, 4, 4}, a_dims, 4),
                      IsError(kLiteRtStatusErrorInvalidArgument));
          continue;
        }
        LITERT_ASSERT_OK_AND_ASSIGN(
            BroadcastPlan<3> plan,
            PrepareBroadcastPlan<3>(in_shapes, {1, 4, 4}, *expected_out, 4));
        VerifyBroadcastPlanAgainstNaiveOracle<3>(plan, in_shapes,
                                                 *expected_out);
      }
    }
  }
}

TEST(BroadcastPlanTest, InferredExactRejectsOverExpandedOutputTargetAccepts) {
  const int32_t one[] = {1};
  const int32_t target[] = {2, 3};

  EXPECT_THAT(PrepareBroadcastPlan<2>({one, one}, {4, 4}, target, 4,
                                      BroadcastOutputMode::kInferredExact),
              IsError(kLiteRtStatusErrorInvalidArgument));

  // Same-rank mismatch where inferred broadcast shape is {2} but output is {3}.
  const int32_t two[] = {2};
  const int32_t three[] = {3};
  EXPECT_THAT(PrepareBroadcastPlan<2>({two, two}, {4, 4}, three, 4,
                                      BroadcastOutputMode::kInferredExact),
              IsError(kLiteRtStatusErrorInvalidArgument));

  LITERT_ASSERT_OK_AND_ASSIGN(
      BroadcastPlan<1> target_plan,
      PrepareBroadcastPlan<1>({one}, {4}, target, 4,
                              BroadcastOutputMode::kTargetShape));
  EXPECT_THAT(target_plan.output_shape, ElementsAre(2, 3));
  EXPECT_EQ(target_plan.output_num_elements, 6u);
  EXPECT_EQ(target_plan.output_num_bytes, 24u);
  EXPECT_EQ(target_plan.NumSpans(), 1u);
}

TEST(BroadcastPlanTest, RejectsIncompatibleDimsEvenWhenAnotherDimIsZero) {
  const int32_t a[] = {0, 2};
  const int32_t b[] = {0, 3};
  const int32_t out[] = {0, 3};

  EXPECT_THAT(PrepareBroadcastPlan<2>({a, b}, {4, 4}, out, 4),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(BroadcastPlanTest, HighRankAboveEightMatchesNaiveOracle) {
  const std::vector<int32_t> a = {2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
  const std::vector<int32_t> b = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
  const std::vector<int32_t> out = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

  LITERT_ASSERT_OK_AND_ASSIGN(BroadcastPlan<2> plan,
                              PrepareBroadcastPlan<2>({a, b}, {4, 4}, out, 4));
  VerifyBroadcastPlanAgainstNaiveOracle<2>(plan, {a, b}, out);
}

TEST(BroadcastPlanTest, RejectsElementAndByteCountOverflowWithoutAllocation) {
  const int32_t huge_elems[] = {std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::max()};
  const int32_t huge_bytes[] = {1 << 30, 1 << 30, 4};
  const int32_t small_shape[] = {1};

  // Output element and byte count overflows.
  EXPECT_THAT(PrepareBroadcastPlan<1>({huge_elems}, {1}, huge_elems, 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(PrepareBroadcastPlan<1>({huge_bytes}, {8}, huge_bytes, 8),
              IsError(kLiteRtStatusErrorInvalidArgument));

  // Input-only element count overflow with a valid output shape.
  EXPECT_THAT(PrepareBroadcastPlan<1>({huge_elems}, {1}, small_shape, 1,
                                      BroadcastOutputMode::kInferredExact),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(PrepareBroadcastPlan<1>({huge_elems}, {1}, small_shape, 1,
                                      BroadcastOutputMode::kTargetShape),
              IsError(kLiteRtStatusErrorInvalidArgument));

  // Input-only byte count overflow with a valid output byte count.
  EXPECT_THAT(PrepareBroadcastPlan<1>({huge_bytes}, {8}, huge_bytes, 1,
                                      BroadcastOutputMode::kInferredExact),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(PrepareBroadcastPlan<1>({huge_bytes}, {8}, huge_bytes, 1,
                                      BroadcastOutputMode::kTargetShape),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(BroadcastPlanTest, TargetShapeMultiInputAndInvalidArgs) {
  const int32_t a[] = {1, 3};
  const int32_t b[] = {2, 1};
  const int32_t target[] = {4, 2, 3};

  LITERT_ASSERT_OK_AND_ASSIGN(
      BroadcastPlan<2> plan,
      PrepareBroadcastPlan<2>({a, b}, {sizeof(int8_t), sizeof(int16_t)}, target,
                              sizeof(float),
                              BroadcastOutputMode::kTargetShape));
  EXPECT_THAT(plan.output_shape, ElementsAre(4, 2, 3));
  EXPECT_EQ(plan.output_num_elements, 24u);
  EXPECT_EQ(plan.output_num_bytes, 24u * sizeof(float));
  EXPECT_THAT(plan.input_num_elements, ElementsAre(3u, 2u));
  EXPECT_THAT(plan.input_num_bytes,
              ElementsAre(3u * sizeof(int8_t), 2u * sizeof(int16_t)));
  VerifyBroadcastPlanAgainstNaiveOracle<2>(plan, {a, b}, target);

  // Empty target shape with non-empty singleton input preserves input counts.
  const int32_t one[] = {1};
  const int32_t empty_target[] = {2, 0};
  LITERT_ASSERT_OK_AND_ASSIGN(
      BroadcastPlan<1> empty_plan,
      PrepareBroadcastPlan<1>({one}, {sizeof(int16_t)}, empty_target,
                              sizeof(float),
                              BroadcastOutputMode::kTargetShape));
  EXPECT_TRUE(empty_plan.IsEmpty());
  EXPECT_EQ(empty_plan.NumSpans(), 0u);
  EXPECT_THAT(empty_plan.output_shape, ElementsAre(2, 0));
  EXPECT_EQ(empty_plan.output_num_elements, 0u);
  EXPECT_EQ(empty_plan.output_num_bytes, 0u);
  EXPECT_THAT(empty_plan.input_num_elements, ElementsAre(1u));
  EXPECT_THAT(empty_plan.input_num_bytes, ElementsAre(sizeof(int16_t)));
  EXPECT_EQ(empty_plan.Spans().begin(), empty_plan.Spans().end());

  // Incompatible multi-input zero-dim shapes under kTargetShape.
  const int32_t z2[] = {2, 0};
  const int32_t z3[] = {3, 0};
  EXPECT_THAT(PrepareBroadcastPlan<2>({z2, z3}, {4, 4}, z2, 4,
                                      BroadcastOutputMode::kTargetShape),
              IsError(kLiteRtStatusErrorInvalidArgument));

  // Input rank exceeding target rank under kTargetShape.
  const int32_t rank3[] = {1, 2, 3};
  const int32_t rank2[] = {2, 3};
  EXPECT_THAT(PrepareBroadcastPlan<1>({rank3}, {4}, rank2, 4,
                                      BroadcastOutputMode::kTargetShape),
              IsError(kLiteRtStatusErrorInvalidArgument));

  // Negative output dimension.
  const int32_t neg_out[] = {2, -1};
  EXPECT_THAT(PrepareBroadcastPlan<1>({a}, {4}, neg_out, 4,
                                      BroadcastOutputMode::kTargetShape),
              IsError(kLiteRtStatusErrorInvalidArgument));

  // Negative input dimension with a valid output shape.
  const int32_t neg_in[] = {-1, 3};
  EXPECT_THAT(PrepareBroadcastPlan<1>({neg_in}, {4}, target, 4,
                                      BroadcastOutputMode::kTargetShape),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(PrepareBroadcastPlan<1>({neg_in}, {4}, target, 4,
                                      BroadcastOutputMode::kInferredExact),
              IsError(kLiteRtStatusErrorInvalidArgument));

  // Zero element sizes for input or output.
  EXPECT_THAT(PrepareBroadcastPlan<2>({a, b}, {0, 4}, target, 4,
                                      BroadcastOutputMode::kTargetShape),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(PrepareBroadcastPlan<2>({a, b}, {4, 4}, target, 0,
                                      BroadcastOutputMode::kTargetShape),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(BroadcastPlanTest, BroadcastCursorDirectOperationsAndIteratorTraits) {
  static_assert(std::is_same_v<
                std::iterator_traits<BroadcastCursor<2>>::iterator_category,
                std::input_iterator_tag>);
  static_assert(
      std::is_same_v<std::iterator_traits<BroadcastCursor<2>>::value_type,
                     BroadcastSpan<2>>);
  static_assert(
      std::is_same_v<std::iterator_traits<BroadcastCursor<2>>::difference_type,
                     std::ptrdiff_t>);
  static_assert(
      std::is_same_v<std::iterator_traits<BroadcastCursor<2>>::pointer, void>);
  static_assert(
      std::is_same_v<std::iterator_traits<BroadcastCursor<2>>::reference,
                     BroadcastSpan<2>>);
  static_assert(
      !std::is_constructible_v<BroadcastCursor<2>, BroadcastPlan<2>&&>);
  static_assert(
      !std::is_constructible_v<BroadcastCursor<2>, const BroadcastPlan<2>&&>);
  static_assert(HasSpans<BroadcastPlan<2>&>::value);
  static_assert(HasSpans<const BroadcastPlan<2>&>::value);
  static_assert(!HasSpans<BroadcastPlan<2>&&>::value);
  static_assert(!HasSpans<const BroadcastPlan<2>&&>::value);

  // Default-constructed cursor and SpanRange are done and safely yield empty.
  BroadcastCursor<2> default_cursor;
  EXPECT_TRUE(default_cursor.Done());
  EXPECT_EQ(default_cursor.Current().count, 0u);
  default_cursor.Advance();
  EXPECT_TRUE(default_cursor.Done());

  BroadcastPlan<2>::SpanRange empty_range;
  EXPECT_EQ(empty_range.plan, nullptr);
  EXPECT_EQ(empty_range.begin(), empty_range.end());
  EXPECT_TRUE(empty_range.begin().Done());

  const int32_t a[] = {2, 1};
  const int32_t b[] = {1, 3};
  const int32_t out[] = {2, 3};
  LITERT_ASSERT_OK_AND_ASSIGN(BroadcastPlan<2> plan,
                              PrepareBroadcastPlan<2>({a, b}, {4, 4}, out, 4));
  BroadcastCursor<2> c1(plan);
  BroadcastCursor<2> c2(plan);
  BroadcastCursor<2> end_cursor;

  EXPECT_FALSE(c1.Done());
  EXPECT_TRUE(c1 == c2);
  EXPECT_TRUE(c1 != end_cursor);

  BroadcastCursor<2> prev = c1++;
  EXPECT_EQ(prev.Current().output_offset, 0u);
  EXPECT_EQ(c1.Current().output_offset, 3u);
  EXPECT_TRUE(c1 != prev);

  c1.Advance();
  EXPECT_TRUE(c1.Done());
  EXPECT_TRUE(c1 == end_cursor);
  EXPECT_EQ(c1.Current().count, 0u);

  // Advance after completion is idempotent.
  c1.Advance();
  EXPECT_TRUE(c1.Done());
  EXPECT_EQ(c1.Current().count, 0u);

  // NumSpans and BroadcastCursor both guard against zero `span_count`.
  BroadcastPlan<1> mutated;
  mutated.output_num_elements = 4;
  mutated.span_count = 0;
  EXPECT_EQ(mutated.NumSpans(), 0u);
  EXPECT_TRUE(BroadcastCursor<1>(mutated).Done());
  EXPECT_EQ(mutated.Spans().begin(), mutated.Spans().end());
}

}  // namespace
}  // namespace litert::internal
