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

#include "litert/vendors/nvidia/compiler/weight_store_builder.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl  // from @com_google_absl
#include "litert/c/litert_common.h"

namespace litert::nvidia {
namespace {

constexpr uint64_t kGranule = 1 << 16;
constexpr uint64_t kFileBytes = 1 << 20;
// The file is mapped from this offset, as a model inside a container is.
constexpr uint64_t kMapOffset = 1 << 14;

// A file of known bytes with a private writable mapping of part of it.
class WeightStoreBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = ::testing::TempDir() + "/weight_store_builder_test.XXXXXX";
    const int fd = mkstemp(path_.data());
    ASSERT_GE(fd, 0);
    std::vector<uint8_t> bytes(kFileBytes);
    for (size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<uint8_t>(i * 7 + (i >> 9));
    }
    ASSERT_EQ(write(fd, bytes.data(), bytes.size()),
              static_cast<ssize_t>(bytes.size()));
    mapping_ = static_cast<uint8_t*>(mmap(nullptr, kFileBytes - kMapOffset,
                                          PROT_READ | PROT_WRITE, MAP_PRIVATE,
                                          fd, kMapOffset));
    close(fd);
    ASSERT_NE(mapping_, MAP_FAILED);
  }
  void TearDown() override {
    munmap(mapping_, kFileBytes - kMapOffset);
    unlink(path_.c_str());
  }

  // The bytes [offset, offset + size) of the file, through the mapping.
  absl::Span<const uint8_t> View(uint64_t offset, uint64_t size) const {
    return {mapping_ + (offset - kMapOffset), size};
  }

  std::string path_;
  uint8_t* mapping_ = nullptr;
};

TEST_F(WeightStoreBuilderTest, PlacesLaunchesAndSharesEqualOnes) {
  auto builder = TensorRtWeightStoreBuilder::Create(path_, kGranule, kGranule);
  ASSERT_TRUE(builder.HasValue()) << builder.Error().Message();
  auto& store = **builder;
  EXPECT_EQ(store.granule(), kGranule);
  EXPECT_EQ(store.source_size(), kFileBytes);

  // A launch of two buffers that are apart in the file, then one buffer.
  const absl::Span<const uint8_t> first[] = {View(100000, 1000),
                                             View(200000, 3000)};
  auto a = store.Add(first);
  ASSERT_TRUE(a.HasValue()) << a.Error().Message();
  EXPECT_EQ(a->segment, 0u);
  EXPECT_EQ(a->offset, 0u);
  const absl::Span<const uint8_t> second[] = {View(300000, 500)};
  auto b = store.Add(second);
  ASSERT_TRUE(b.HasValue()) << b.Error().Message();
  EXPECT_EQ(b->segment, 0u);
  // The next multiple of the entry alignment after 4000 bytes.
  EXPECT_EQ(b->offset, 4096u);
  // The same members again, in this or a later partition: the same place.
  auto again = store.Add(first);
  ASSERT_TRUE(again.HasValue());
  EXPECT_EQ(again->segment, a->segment);
  EXPECT_EQ(again->offset, a->offset);
  // The same bytes in another grouping are another launch.
  const absl::Span<const uint8_t> regrouped[] = {View(100000, 1000)};
  auto c = store.Add(regrouped);
  ASSERT_TRUE(c.HasValue());
  EXPECT_EQ(c->offset, 4096u + 512u);

  EXPECT_FALSE(store.segments()[0].closed);
  store.EndPartition();
  ASSERT_EQ(store.segments().size(), 1u);
  const auto& segment = store.segments()[0];
  EXPECT_TRUE(segment.closed);
  EXPECT_EQ(segment.size, kGranule);
  ASSERT_EQ(segment.pieces.size(), 4u);
  EXPECT_EQ(segment.pieces[0].source_offset, 100000u);
  EXPECT_EQ(segment.pieces[0].size, 1000u);
  EXPECT_EQ(segment.pieces[0].segment_offset, 0u);
  EXPECT_EQ(segment.pieces[1].source_offset, 200000u);
  EXPECT_EQ(segment.pieces[1].segment_offset, 1000u);
  EXPECT_EQ(segment.pieces[2].source_offset, 300000u);
  EXPECT_EQ(segment.pieces[2].segment_offset, 4096u);
  EXPECT_EQ(segment.pieces[3].segment_offset, 4096u + 512u);

  // A later partition finds the launch where the first one put it, and its
  // own launches start a new segment.
  auto later = store.Add(second);
  ASSERT_TRUE(later.HasValue());
  EXPECT_EQ(later->segment, 0u);
  EXPECT_EQ(later->offset, 4096u);
  const absl::Span<const uint8_t> fresh[] = {View(400000, 128)};
  auto d = store.Add(fresh);
  ASSERT_TRUE(d.HasValue());
  EXPECT_EQ(d->segment, 1u);
  EXPECT_EQ(d->offset, 0u);
}

TEST_F(WeightStoreBuilderTest,
       ClosesFullSegmentsAndGivesLargeLaunchesTheirOwn) {
  auto builder =
      TensorRtWeightStoreBuilder::Create(path_, kGranule, 2 * kGranule);
  ASSERT_TRUE(builder.HasValue()) << builder.Error().Message();
  auto& store = **builder;
  // 1.5 granules, then one granule does not fit any more.
  const absl::Span<const uint8_t> first[] = {
      View(kMapOffset, 3 * kGranule / 2)};
  const absl::Span<const uint8_t> second[] = {View(200000, kGranule)};
  // Larger than a segment.
  const absl::Span<const uint8_t> large[] = {View(300000, 5 * kGranule / 2)};
  const absl::Span<const uint8_t> small[] = {View(600000, 256)};
  auto a = store.Add(first);
  auto b = store.Add(second);
  auto c = store.Add(large);
  auto d = store.Add(small);
  ASSERT_TRUE(a.HasValue() && b.HasValue() && c.HasValue() && d.HasValue());
  EXPECT_EQ(a->segment, 0u);
  EXPECT_EQ(b->segment, 1u);
  EXPECT_EQ(b->offset, 0u);
  EXPECT_EQ(c->segment, 2u);
  EXPECT_EQ(c->offset, 0u);
  EXPECT_EQ(d->segment, 3u);
  store.EndPartition();
  ASSERT_EQ(store.segments().size(), 4u);
  EXPECT_EQ(store.segments()[0].size, 2 * kGranule);
  EXPECT_EQ(store.segments()[1].size, kGranule);
  EXPECT_EQ(store.segments()[2].size, 3 * kGranule);
  EXPECT_EQ(store.segments()[3].size, kGranule);
  // Segments of different pieces have different keys.
  EXPECT_FALSE(store.segments()[0].key == store.segments()[1].key);
}

TEST_F(WeightStoreBuilderTest, MergesPiecesThatContinueEachOther) {
  auto builder = TensorRtWeightStoreBuilder::Create(path_, kGranule, kGranule);
  ASSERT_TRUE(builder.HasValue()) << builder.Error().Message();
  auto& store = **builder;
  const absl::Span<const uint8_t> members[] = {View(100000, 1024),
                                               View(101024, 1024)};
  ASSERT_TRUE(store.Add(members).HasValue());
  store.EndPartition();
  ASSERT_EQ(store.segments()[0].pieces.size(), 1u);
  EXPECT_EQ(store.segments()[0].pieces[0].source_offset, 100000u);
  EXPECT_EQ(store.segments()[0].pieces[0].size, 2048u);
}

TEST_F(WeightStoreBuilderTest, RejectsBuffersThatAreNotTheFile) {
  auto builder = TensorRtWeightStoreBuilder::Create(path_, kGranule, kGranule);
  ASSERT_TRUE(builder.HasValue()) << builder.Error().Message();
  auto& store = **builder;
  // Heap memory.
  const std::vector<uint8_t> copy(4096, 1);
  const absl::Span<const uint8_t> heap[] = {copy};
  auto not_mapped = store.Add(heap);
  ASSERT_FALSE(not_mapped.HasValue());
  EXPECT_EQ(not_mapped.Error().Status(), kLiteRtStatusErrorNotFound);
  // A page of the mapping that was written to no longer is the file's.
  mapping_[200000 - kMapOffset] ^= 1;
  const absl::Span<const uint8_t> modified[] = {View(199000, 2000)};
  auto changed = store.Add(modified);
  ASSERT_FALSE(changed.HasValue());
  EXPECT_EQ(changed.Error().Status(), kLiteRtStatusErrorNotFound);
  // Other pages still are, read or not.
  const absl::Span<const uint8_t> untouched[] = {View(400000, 2000)};
  EXPECT_TRUE(store.Add(untouched).HasValue());
  volatile uint8_t read = mapping_[500000 - kMapOffset];
  static_cast<void>(read);
  const absl::Span<const uint8_t> read_before[] = {View(499000, 2000)};
  EXPECT_TRUE(store.Add(read_before).HasValue());
  EXPECT_FALSE(store.Add({}).HasValue());
}

TEST_F(WeightStoreBuilderTest, NeedsAMappedFileAndAValidGranule) {
  EXPECT_FALSE(
      TensorRtWeightStoreBuilder::Create(path_, 3 << 16, kGranule).HasValue());
  EXPECT_FALSE(
      TensorRtWeightStoreBuilder::Create(path_, kGranule, 0).HasValue());
  EXPECT_FALSE(
      TensorRtWeightStoreBuilder::Create(path_ + ".missing", kGranule, kGranule)
          .HasValue());
  // A file that exists but is not mapped.
  std::string other = ::testing::TempDir() + "/weight_store_other.XXXXXX";
  const int fd = mkstemp(other.data());
  ASSERT_GE(fd, 0);
  ASSERT_EQ(write(fd, "weights", 7), 7);
  close(fd);
  auto unmapped = TensorRtWeightStoreBuilder::Create(other, kGranule, kGranule);
  ASSERT_FALSE(unmapped.HasValue());
  EXPECT_EQ(unmapped.Error().Status(), kLiteRtStatusErrorNotFound);
  unlink(other.c_str());
}

}  // namespace
}  // namespace litert::nvidia
