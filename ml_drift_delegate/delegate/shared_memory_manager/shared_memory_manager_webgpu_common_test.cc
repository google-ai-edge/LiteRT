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

#include "ml_drift_delegate/delegate/shared_memory_manager/shared_memory_manager_webgpu_common.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/types.h"  // from @ml_drift

namespace ml_drift {
namespace webgpu_internal {
namespace {

using ::testing::ElementsAreArray;
using ::testing::status::StatusIs;

constexpr uint8_t kPadding = 0xEE;

std::vector<uint8_t> MakeRows(size_t row_size, size_t num_rows) {
  std::vector<uint8_t> rows(row_size * num_rows);
  for (size_t i = 0; i < rows.size(); ++i) {
    rows[i] = static_cast<uint8_t>(i * 7 + 3);
  }
  return rows;
}

// Returns the rows of `src` at a pitch of `pitch` bytes, with kPadding between
// them.
std::vector<uint8_t> PitchedRows(const std::vector<uint8_t>& src,
                                 size_t row_size, size_t num_rows, size_t pitch,
                                 size_t dst_size) {
  std::vector<uint8_t> expected(dst_size, kPadding);
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t i = 0; i < row_size; ++i) {
      expected[row * pitch + i] = src[row * row_size + i];
    }
  }
  return expected;
}

TEST(CopyRowsWithPitchTest, UnalignedRowsAreWrittenAtThePitch) {
  // 5 texels of 8 bytes per row, as in a kUint16 texture of width 5.
  constexpr size_t kRowSize = 40;
  constexpr size_t kNumRows = 3;
  constexpr size_t kPitch = 256;
  const std::vector<uint8_t> src = MakeRows(kRowSize, kNumRows);
  const size_t dst_size = (kNumRows - 1) * kPitch + kRowSize;
  std::vector<uint8_t> dst(dst_size, kPadding);
  ASSERT_OK(
      CopyRowsWithPitch(src, kRowSize, kNumRows, kPitch, absl::MakeSpan(dst)));
  EXPECT_THAT(dst, ElementsAreArray(
                       PitchedRows(src, kRowSize, kNumRows, kPitch, dst_size)));
}

TEST(CopyRowsWithPitchTest, RowsOfAllDepthSlicesAreWrittenAtThePitch) {
  // A texture of height 2 and depth 3 has 6 rows.
  constexpr size_t kRowSize = 24;
  constexpr size_t kNumRows = 2 * 3;
  constexpr size_t kPitch = 256;
  const std::vector<uint8_t> src = MakeRows(kRowSize, kNumRows);
  const size_t dst_size = kNumRows * kPitch;
  std::vector<uint8_t> dst(dst_size, kPadding);
  ASSERT_OK(
      CopyRowsWithPitch(src, kRowSize, kNumRows, kPitch, absl::MakeSpan(dst)));
  EXPECT_THAT(dst, ElementsAreArray(
                       PitchedRows(src, kRowSize, kNumRows, kPitch, dst_size)));
}

TEST(CopyRowsWithPitchTest, AlignedRowsAreCopiedAsIs) {
  constexpr size_t kRowSize = 256;
  constexpr size_t kNumRows = 4;
  const std::vector<uint8_t> src = MakeRows(kRowSize, kNumRows);
  std::vector<uint8_t> dst(src.size() + 16, kPadding);
  ASSERT_OK(CopyRowsWithPitch(src, kRowSize, kNumRows, kRowSize,
                              absl::MakeSpan(dst)));
  std::vector<uint8_t> expected = src;
  expected.resize(dst.size(), kPadding);
  EXPECT_THAT(dst, ElementsAreArray(expected));
}

TEST(CopyRowsWithPitchTest, TrailingSourceBytesAreIgnored) {
  constexpr size_t kRowSize = 40;
  constexpr size_t kNumRows = 3;
  constexpr size_t kPitch = 256;
  const std::vector<uint8_t> src = MakeRows(kRowSize, kNumRows + 1);
  const size_t dst_size = (kNumRows - 1) * kPitch + kRowSize;
  std::vector<uint8_t> dst(dst_size, kPadding);
  ASSERT_OK(
      CopyRowsWithPitch(src, kRowSize, kNumRows, kPitch, absl::MakeSpan(dst)));
  EXPECT_THAT(dst, ElementsAreArray(
                       PitchedRows(src, kRowSize, kNumRows, kPitch, dst_size)));
  // Same with a pitch equal to the row size.
  std::vector<uint8_t> tight_dst(kNumRows * kRowSize, kPadding);
  ASSERT_OK(CopyRowsWithPitch(src, kRowSize, kNumRows, kRowSize,
                              absl::MakeSpan(tight_dst)));
  EXPECT_THAT(tight_dst,
              ElementsAreArray(src.begin(), src.begin() + kNumRows * kRowSize));
}

TEST(CopyRowsWithPitchTest, RejectsTruncatedSource) {
  constexpr size_t kRowSize = 40;
  constexpr size_t kNumRows = 3;
  constexpr size_t kPitch = 256;
  std::vector<uint8_t> dst(kNumRows * kPitch);
  const std::vector<uint8_t> short_rows(kRowSize * kNumRows - 1);
  EXPECT_THAT(CopyRowsWithPitch(short_rows, kRowSize, kNumRows, kPitch,
                                absl::MakeSpan(dst)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CopyRowsWithPitchTest, RejectsTooSmallDestinationOrPitch) {
  constexpr size_t kRowSize = 40;
  constexpr size_t kNumRows = 3;
  const std::vector<uint8_t> src = MakeRows(kRowSize, kNumRows);
  std::vector<uint8_t> small_dst(2 * 256 + kRowSize - 1);
  EXPECT_THAT(CopyRowsWithPitch(src, kRowSize, kNumRows, /*dst_row_pitch=*/256,
                                absl::MakeSpan(small_dst)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  std::vector<uint8_t> dst(kNumRows * kRowSize);
  EXPECT_THAT(CopyRowsWithPitch(src, kRowSize, kNumRows, /*dst_row_pitch=*/32,
                                absl::MakeSpan(dst)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CopyRowsWithPitchTest, EmptyInputSucceeds) {
  EXPECT_OK(CopyRowsWithPitch(/*src=*/{}, /*row_size=*/0, /*num_rows=*/0,
                              /*dst_row_pitch=*/0, /*dst=*/{}));
  EXPECT_OK(CopyRowsWithPitch(/*src=*/{}, /*row_size=*/0, /*num_rows=*/3,
                              /*dst_row_pitch=*/256, /*dst=*/{}));
  EXPECT_OK(CopyRowsWithPitch(/*src=*/{}, /*row_size=*/40, /*num_rows=*/0,
                              /*dst_row_pitch=*/256, /*dst=*/{}));
}

TEST(CopyRowsWithPitchTest, RejectsSizesThatOverflow) {
  const std::vector<uint8_t> src(40);
  std::vector<uint8_t> dst(16);
  // (2 - 1) * pitch + 20 wraps around to 9 if computed naively.
  EXPECT_THAT(CopyRowsWithPitch(
                  src, /*row_size=*/20, /*num_rows=*/2,
                  /*dst_row_pitch=*/std::numeric_limits<size_t>::max() - 10,
                  absl::MakeSpan(dst)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // num_rows * row_size wraps around to 0.
  constexpr size_t kHalfOfSizeRange =
      size_t{1} << (std::numeric_limits<size_t>::digits - 1);
  EXPECT_THAT(CopyRowsWithPitch(/*src=*/{}, /*row_size=*/2,
                                /*num_rows=*/kHalfOfSizeRange,
                                /*dst_row_pitch=*/256, absl::MakeSpan(dst)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GetTextureStagingLayoutTest, UnalignedRowsAreRepacked) {
  // kUint16 x 4 channels = 8 bytes per texel; width 5 = 40 bytes per row.
  ASSERT_OK_AND_ASSIGN(TextureStagingLayout layout,
                       GetTextureStagingLayout(int3(5, 3, 1), /*texel_size=*/8,
                                               /*data_size=*/3 * 40));
  EXPECT_EQ(layout.bytes_per_row, 40);
  EXPECT_EQ(layout.bytes_per_row_aligned, 256);
  EXPECT_EQ(layout.num_rows, 3);
  EXPECT_EQ(layout.min_buffer_size, 2 * 256 + 40);
  EXPECT_TRUE(layout.repack_rows);
}

TEST(GetTextureStagingLayoutTest, RowsOfAllDepthSlicesAreCounted) {
  ASSERT_OK_AND_ASSIGN(TextureStagingLayout layout,
                       GetTextureStagingLayout(int3(3, 2, 3), /*texel_size=*/8,
                                               /*data_size=*/6 * 24));
  EXPECT_EQ(layout.num_rows, 6);
  EXPECT_EQ(layout.min_buffer_size, 5 * 256 + 24);
  EXPECT_TRUE(layout.repack_rows);
}

TEST(GetTextureStagingLayoutTest, AlignedRowsAreNotRepackedOrSizeChecked) {
  // Width 32 = 256 bytes per row. The data size is not checked, as before.
  for (size_t data_size : {size_t{4 * 256}, size_t{100}, size_t{8 * 256}}) {
    ASSERT_OK_AND_ASSIGN(
        TextureStagingLayout layout,
        GetTextureStagingLayout(int3(32, 4, 1), /*texel_size=*/8, data_size));
    EXPECT_EQ(layout.bytes_per_row, 256);
    EXPECT_EQ(layout.bytes_per_row_aligned, 256);
    EXPECT_EQ(layout.min_buffer_size, 4 * 256);
    EXPECT_FALSE(layout.repack_rows);
  }
}

TEST(GetTextureStagingLayoutTest, TightRowsWithTrailingBytesAreRepacked) {
  for (size_t data_size : {size_t{3 * 40 + 1}, size_t{3 * 40 + 100},
                           size_t{2 * 256 + 40 - 1}, size_t{3 * 256 + 1}}) {
    ASSERT_OK_AND_ASSIGN(
        TextureStagingLayout layout,
        GetTextureStagingLayout(int3(5, 3, 1), /*texel_size=*/8, data_size));
    EXPECT_TRUE(layout.repack_rows) << data_size;
  }
}

TEST(GetTextureStagingLayoutTest, DataPaddedToTheAlignedPitchIsCopiedAsIs) {
  // The old flat copy uploaded such data correctly, so it is not repacked:
  // with and without the padding of the last row.
  for (size_t data_size : {size_t{2 * 256 + 40}, size_t{3 * 256}}) {
    ASSERT_OK_AND_ASSIGN(
        TextureStagingLayout layout,
        GetTextureStagingLayout(int3(5, 3, 1), /*texel_size=*/8, data_size));
    EXPECT_EQ(layout.bytes_per_row_aligned, 256);
    EXPECT_EQ(layout.min_buffer_size, 2 * 256 + 40);
    EXPECT_FALSE(layout.repack_rows) << data_size;
  }
}

TEST(GetTextureStagingLayoutTest, SingleRow) {
  // Exactly one row, with or without padding: copied as is.
  for (size_t data_size : {size_t{40}, size_t{256}}) {
    ASSERT_OK_AND_ASSIGN(
        TextureStagingLayout layout,
        GetTextureStagingLayout(int3(5, 1, 1), /*texel_size=*/8, data_size));
    EXPECT_EQ(layout.min_buffer_size, 40);
    EXPECT_FALSE(layout.repack_rows);
  }
  // Trailing bytes: the row is copied without them, which uploads the same
  // texels.
  ASSERT_OK_AND_ASSIGN(TextureStagingLayout layout,
                       GetTextureStagingLayout(int3(5, 1, 1), /*texel_size=*/8,
                                               /*data_size=*/64));
  EXPECT_TRUE(layout.repack_rows);
  EXPECT_THAT(GetTextureStagingLayout(int3(5, 1, 1), /*texel_size=*/8,
                                      /*data_size=*/39),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GetTextureStagingLayoutTest, RejectsTruncatedUnalignedData) {
  for (size_t data_size : {size_t{0}, size_t{3 * 40 - 1}}) {
    EXPECT_THAT(
        GetTextureStagingLayout(int3(5, 3, 1), /*texel_size=*/8, data_size),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(GetTextureStagingLayoutTest, RejectsInvalidRegions) {
  for (const int3& region : {int3(0, 3, 1), int3(5, 0, 1), int3(5, 3, 0),
                             int3(-1, 3, 1), int3(5, -3, 1), int3(5, 3, -1)}) {
    EXPECT_THAT(GetTextureStagingLayout(region, /*texel_size=*/8,
                                        /*data_size=*/0),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  EXPECT_THAT(GetTextureStagingLayout(int3(5, 3, 1), /*texel_size=*/0,
                                      /*data_size=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GetTextureStagingLayoutTest, RejectsRowsThatOverflow) {
  EXPECT_THAT(
      GetTextureStagingLayout(int3(std::numeric_limits<int>::max(), 1, 1),
                              /*texel_size=*/16, /*data_size=*/0),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace webgpu_internal
}  // namespace ml_drift
