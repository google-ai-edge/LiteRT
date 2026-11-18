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
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/test/matchers.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::litert::IsError;
using ::testing::litert::IsOk;

TEST(TensorLayoutUtilTest, CheckedByteCountReturnsProductForValidInputs) {
  LITERT_ASSERT_OK_AND_ASSIGN(size_t bytes,
                              CheckedByteCount(16, sizeof(float)));
  EXPECT_EQ(bytes, 64);
}

TEST(TensorLayoutUtilTest, CheckedByteCountRejectsZeroElementSize) {
  EXPECT_THAT(CheckedByteCount(4, 0),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(TensorLayoutUtilTest, CheckedByteCountRejectsElementSizeAboveIntMax) {
  const size_t too_large =
      static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_THAT(CheckedByteCount(1, too_large),
              IsError(kLiteRtStatusErrorInvalidArgument));
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
  const auto& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(auto layout,
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
  const auto& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto layout,
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
            .name = "EmptyInputIntoNonEmptyOutput",
            .input_dims = {0},
            .output_dims = {2},
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
  const auto& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(auto out,
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

struct AxisLayoutValidCase {
  std::string name;
  std::vector<int32_t> dims;
  int32_t axis;
  size_t elem_size;
  size_t expected_outer;
  size_t expected_axis;
  size_t expected_inner;
  size_t expected_bytes;
};

using PrepareAxisLayoutValidTest =
    ::testing::TestWithParam<AxisLayoutValidCase>;

TEST_P(PrepareAxisLayoutValidTest, ComputesOuterAxisInnerAndBytes) {
  const auto& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto layout, PrepareAxisLayout(tc.dims, tc.axis, tc.elem_size));
  EXPECT_EQ(layout.outer_size, tc.expected_outer);
  EXPECT_EQ(layout.axis_size, tc.expected_axis);
  EXPECT_EQ(layout.inner_size, tc.expected_inner);
  EXPECT_EQ(layout.num_bytes, tc.expected_bytes);
}

INSTANTIATE_TEST_SUITE_P(
    PrepareAxisLayoutValidTests, PrepareAxisLayoutValidTest,
    ::testing::Values(
        AxisLayoutValidCase{
            .name = "MiddleAxisPositive",
            .dims = {2, 3, 4},
            .axis = 1,
            .elem_size = sizeof(int32_t),
            .expected_outer = 2,
            .expected_axis = 3,
            .expected_inner = 4,
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
            .expected_bytes = 48,
        },
        AxisLayoutValidCase{
            .name = "ZeroOuterDimension",
            .dims = {0, 3, 4},
            .axis = 1,
            .elem_size = sizeof(float),
            .expected_outer = 0,
            .expected_axis = 3,
            .expected_inner = 0,
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
  const auto& tc = GetParam();
  EXPECT_THAT(PrepareAxisLayout(tc.dims, tc.axis, tc.elem_size),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    PrepareAxisLayoutInvalidTests, PrepareAxisLayoutInvalidTest,
    ::testing::Values(
        AxisLayoutInvalidCase{.name = "EmptyDims", .dims = {}, .axis = 0},
        AxisLayoutInvalidCase{
            .name = "AxisTooLarge", .dims = {2, 3}, .axis = 2},
        AxisLayoutInvalidCase{
            .name = "AxisTooNegative", .dims = {2, 3}, .axis = -3},
        AxisLayoutInvalidCase{
            .name = "NegativeDim", .dims = {2, -1}, .axis = 0},
        AxisLayoutInvalidCase{
            .name = "ByteOverflow",
            .dims = {1 << 30, 1 << 30, 4},
            .axis = 0,
            .elem_size = 8,
        }),
    [](const ::testing::TestParamInfo<AxisLayoutInvalidCase>& info) {
      return info.param.name;
    });

struct InsertAxisValidCase {
  std::string name;
  std::vector<int32_t> dims;
  int32_t axis;
  size_t elem_size;
  size_t expected_outer;
  size_t expected_inner;
  size_t expected_bytes;
};

using PrepareInsertAxisLayoutValidTest =
    ::testing::TestWithParam<InsertAxisValidCase>;

TEST_P(PrepareInsertAxisLayoutValidTest, ComputesOuterInnerAndBytes) {
  const auto& tc = GetParam();
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto layout, PrepareInsertAxisLayout(tc.dims, tc.axis, tc.elem_size));
  EXPECT_EQ(layout.outer_size, tc.expected_outer);
  EXPECT_EQ(layout.inner_size, tc.expected_inner);
  EXPECT_EQ(layout.num_bytes, tc.expected_bytes);
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
            .expected_bytes = 4,
        },
        InsertAxisValidCase{
            .name = "NegativeAxisInsertsAtEnd",
            .dims = {2, 3},
            .axis = -1,
            .elem_size = sizeof(float),
            .expected_outer = 6,
            .expected_inner = 1,
            .expected_bytes = 24,
        },
        InsertAxisValidCase{
            .name = "MiddleAxis",
            .dims = {2, 3},
            .axis = 1,
            .elem_size = sizeof(int16_t),
            .expected_outer = 2,
            .expected_inner = 3,
            .expected_bytes = 12,
        },
        InsertAxisValidCase{
            .name = "EmptyTensor",
            .dims = {2, 0},
            .axis = 1,
            .elem_size = sizeof(float),
            .expected_outer = 0,
            .expected_inner = 0,
            .expected_bytes = 0,
        }),
    [](const ::testing::TestParamInfo<InsertAxisValidCase>& info) {
      return info.param.name;
    });

using PrepareInsertAxisLayoutInvalidTest =
    ::testing::TestWithParam<AxisLayoutInvalidCase>;

TEST_P(PrepareInsertAxisLayoutInvalidTest, RejectsInvalidArgs) {
  const auto& tc = GetParam();
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
                          .name = "NegativeDim", .dims = {2, -1}, .axis = 0},
                      AxisLayoutInvalidCase{
                          .name = "ByteOverflow",
                          .dims = {1 << 30, 1 << 30, 4},
                          .axis = 0,
                          .elem_size = 8,
                      }),
    [](const ::testing::TestParamInfo<AxisLayoutInvalidCase>& info) {
      return info.param.name;
    });

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

template <size_t N>
void VerifyBroadcastPlanAgainstNaiveOracle(
    const BroadcastPlan<N>& plan,
    const std::array<absl::Span<const int32_t>, N>& in_shapes,
    absl::Span<const int32_t> out_dims) {
  size_t visited_elements = 0;
  size_t visited_spans = 0;
  for (const auto& span : plan.Spans()) {
    ++visited_spans;
    ASSERT_GT(span.count, 0u);
    ASSERT_EQ(span.output_offset, visited_elements);
    for (size_t k = 0; k < N; ++k) {
      ASSERT_LE(span.input_steps[k], 1u);
    }
    for (size_t i = 0; i < span.count; ++i) {
      const size_t out_idx = span.output_offset + i;
      for (size_t k = 0; k < N; ++k) {
        const size_t actual_in =
            span.input_offsets[k] + i * span.input_steps[k];
        const size_t expected_in =
            NaiveBroadcastFlatIndex(out_idx, out_dims, in_shapes[k]);
        ASSERT_EQ(actual_in, expected_in)
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
      auto plan,
      PrepareBroadcastPlan<2>({in0, in1}, {sizeof(float), sizeof(float)}, out,
                              sizeof(float)));

  ASSERT_EQ(plan.NumSpans(), 6u);
  std::vector<BroadcastSpan<2>> spans;
  for (const auto& span : plan.Spans()) {
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
      auto full_coalesce,
      PrepareBroadcastPlan<2>({a, scalar_like}, {4, 4}, a, 4));
  EXPECT_EQ(full_coalesce.NumSpans(), 1u);
  EXPECT_EQ(full_coalesce.span_count, 24u);
  EXPECT_THAT(full_coalesce.span_steps, ElementsAre(1u, 0u));

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto outer_coalesce,
      PrepareBroadcastPlan<2>({a, trailing_bcast}, {4, 4}, a, 4));
  EXPECT_EQ(outer_coalesce.NumSpans(), 6u);
  EXPECT_EQ(outer_coalesce.span_count, 4u);
  EXPECT_THAT(outer_coalesce.outer_extents, ElementsAre(6u));
  EXPECT_THAT(outer_coalesce.span_steps, ElementsAre(1u, 0u));
}

TEST(BroadcastPlanTest, UnaryExhaustiveSmallShapesMatchNaiveOracle) {
  const auto shapes = EnumerateSmallShapes();
  for (const auto& in_dims : shapes) {
    for (const auto& out_dims : shapes) {
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
      auto plan = PrepareBroadcastPlan<1>({in_dims}, {sizeof(int32_t)},
                                          out_dims, sizeof(int32_t),
                                          BroadcastOutputMode::kTargetShape);
      ASSERT_EQ(plan.HasValue(), expected_valid);
      if (expected_valid) {
        VerifyBroadcastPlanAgainstNaiveOracle<1>(*plan, {in_dims}, out_dims);
      }
    }
  }
}

TEST(BroadcastPlanTest, BinaryExhaustiveSmallShapesMatchNaiveOracle) {
  const auto shapes = EnumerateSmallShapes();
  for (const auto& a_dims : shapes) {
    for (const auto& b_dims : shapes) {
      auto expected_out = ComputeConcreteBroadcastShape(a_dims, b_dims);
      if (!expected_out) {
        EXPECT_THAT(
            PrepareBroadcastPlan<2>({a_dims, b_dims}, {4, 4}, a_dims, 4),
            IsError(kLiteRtStatusErrorInvalidArgument));
        continue;
      }
      LITERT_ASSERT_OK_AND_ASSIGN(
          auto plan,
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
  for (const auto& c_dims : shapes) {
    for (const auto& a_dims : shapes) {
      for (const auto& b_dims : shapes) {
        const std::array<absl::Span<const int32_t>, 3> in_shapes = {
            c_dims, a_dims, b_dims};
        auto expected_out = ComputeConcreteBroadcastShape(in_shapes);
        if (!expected_out) {
          EXPECT_THAT(PrepareBroadcastPlan<3>(in_shapes, {1, 4, 4}, a_dims, 4),
                      IsError(kLiteRtStatusErrorInvalidArgument));
          continue;
        }
        LITERT_ASSERT_OK_AND_ASSIGN(
            auto plan,
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

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto target_plan,
      PrepareBroadcastPlan<1>({one}, {4}, target, 4,
                              BroadcastOutputMode::kTargetShape));
  EXPECT_EQ(target_plan.output_num_elements, 6u);
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

  LITERT_ASSERT_OK_AND_ASSIGN(auto plan,
                              PrepareBroadcastPlan<2>({a, b}, {4, 4}, out, 4));
  VerifyBroadcastPlanAgainstNaiveOracle<2>(plan, {a, b}, out);
}

TEST(BroadcastPlanTest, RejectsElementAndByteCountOverflowWithoutAllocation) {
  const int32_t huge_elems[] = {std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::max()};
  const int32_t huge_bytes[] = {1 << 30, 1 << 30, 4};

  EXPECT_THAT(PrepareBroadcastPlan<1>({huge_elems}, {1}, huge_elems, 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(PrepareBroadcastPlan<1>({huge_bytes}, {8}, huge_bytes, 8),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

TEST(BroadcastPlanTest, ValidateBufferCapacityChecksLengthAndNullity) {
  const int32_t vals[3] = {1, 2, 3};
  EXPECT_THAT(ValidateBufferCapacity(absl::MakeConstSpan(vals), 3), IsOk());
  EXPECT_THAT(ValidateBufferCapacity(absl::Span<const int32_t>(), 0), IsOk());
  EXPECT_THAT(ValidateBufferCapacity(absl::MakeConstSpan(vals).first(2), 3),
              IsError(kLiteRtStatusErrorInvalidArgument));
  EXPECT_THAT(ValidateBufferCapacity(absl::Span<const int32_t>(), 1),
              IsError(kLiteRtStatusErrorInvalidArgument));
}

}  // namespace
}  // namespace litert::internal
