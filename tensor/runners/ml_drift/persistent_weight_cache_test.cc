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

#include "tensor/runners/ml_drift/persistent_weight_cache.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "tensor/utils/matchers.h"  // IWYU pragma: keep (OSS ASSERT_OK*)

namespace litert::tensor {
namespace {

using ::testing::ElementsAreArray;

::ml_drift::TensorDescriptor MakeDescriptor(int size, int seed) {
  ::ml_drift::TensorDescriptor desc(::ml_drift::DataType::kUint8,
                                    ::ml_drift::TensorStorageType::kBuffer,
                                    ::ml_drift::Layout::kLinear);
  desc.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, size));
  // As much data as the descriptor's memory (whole 4-channel slices).
  std::vector<uint8_t> data(desc.GetMemorySizeInBytes());
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i * 7 + seed);
  }
  desc.SetData(std::move(data));
  return desc;
}

std::string CachePath(const std::string& name) {
  return absl::StrCat(testing::TempDir(), "/", name);
}

std::vector<char> ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
}

void WriteFile(const std::string& path, const std::vector<char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), bytes.size());
}

// The first file offset >= `start` of the data of `desc`, or `npos`.
size_t FindData(const std::vector<char>& bytes,
                const ::ml_drift::TensorDescriptor& desc, size_t start = 0) {
  const absl::string_view file(bytes.data(), bytes.size());
  return file.find(
      absl::string_view(reinterpret_cast<const char*>(desc.GetData().data()),
                        desc.GetData().size()),
      start);
}

constexpr PersistentWeightCache::Options kMapOptions = {
    .map_data = true, .verify_all_bytes = false};

void BuildCache(const std::string& path, uint64_t identifier) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                       PersistentWeightCache::Open(path, identifier));
  ASSERT_TRUE(cache->building());
  ASSERT_OK(cache->Insert("w.a", MakeDescriptor(1000, 1)));
  ASSERT_OK(cache->Insert("w.b+w.c", MakeDescriptor(37, 2)));
  // Repeated names are ignored.
  ASSERT_OK(cache->Insert("w.a", MakeDescriptor(5, 3)));
  EXPECT_EQ(cache->num_entries(), 2);
  ASSERT_OK(cache->Finish());
}

TEST(PersistentWeightCacheTest, LoadsWhatWasBuilt) {
  const std::string path = CachePath("roundtrip.bin");
  BuildCache(path, 42);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                       PersistentWeightCache::Open(path, 42));
  ASSERT_TRUE(cache->loaded());
  EXPECT_FALSE(cache->building());
  auto descriptors = cache->ReleaseDescriptors();
  ASSERT_EQ(descriptors.size(), 2);
  for (const auto& [name, size, seed] :
       {std::tuple<std::string, int, int>{"w.a", 1000, 1},
        std::tuple<std::string, int, int>{"w.b+w.c", 37, 2}}) {
    ASSERT_TRUE(descriptors.contains(name)) << name;
    const ::ml_drift::TensorDescriptor expected = MakeDescriptor(size, seed);
    EXPECT_EQ(descriptors[name].GetBHWCShape(), expected.GetBHWCShape());
    EXPECT_EQ(descriptors[name].GetDataType(), expected.GetDataType());
    EXPECT_THAT(descriptors[name].GetData(),
                ElementsAreArray(expected.GetData()));
  }
}

TEST(PersistentWeightCacheTest, RebuildsForAnotherIdentifier) {
  const std::string path = CachePath("identifier.bin");
  BuildCache(path, 1);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                       PersistentWeightCache::Open(path, 2));
  EXPECT_FALSE(cache->loaded());
  EXPECT_TRUE(cache->building());
}

TEST(PersistentWeightCacheTest, RebuildsACorruptFile) {
  const std::string path = CachePath("corrupt.bin");
  BuildCache(path, 7);
  const std::vector<char> orig_bytes = ReadFile(path);
  ASSERT_GT(orig_bytes.size(), 1000);
  const size_t data_pos = FindData(orig_bytes, MakeDescriptor(1000, 1)) + 500;
  // Flip a byte in the header, middle weights payload, and trailing index.
  for (size_t pos : {size_t{0}, data_pos, orig_bytes.size() - 1}) {
    std::vector<char> bytes = orig_bytes;
    bytes[pos] ^= 0x5A;
    WriteFile(path, bytes);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                         PersistentWeightCache::Open(path, 7));
    EXPECT_FALSE(cache->loaded()) << "pos=" << pos;
    EXPECT_TRUE(cache->building()) << "pos=" << pos;
  }
  // Rebuilding replaces the corrupt file.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                       PersistentWeightCache::Open(path, 7));
  ASSERT_TRUE(cache->building());
  ASSERT_OK(cache->Insert("w.a", MakeDescriptor(10, 4)));
  ASSERT_OK(cache->Finish());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> reloaded,
                       PersistentWeightCache::Open(path, 7));
  EXPECT_TRUE(reloaded->loaded());
  EXPECT_EQ(reloaded->num_entries(), 1);
}

TEST(PersistentWeightCacheTest, MapsWhatWasBuilt) {
  const std::string path = CachePath("mapped.bin");
  BuildCache(path, 42);
  absl::flat_hash_map<std::string, PersistentWeightCache::MappedWeight> mapped;
  {
    ASSERT_OK_AND_ASSIGN(auto cache,
                         PersistentWeightCache::Open(path, 42, kMapOptions));
    ASSERT_TRUE(cache->loaded());
    EXPECT_EQ(cache->num_entries(), 2);
    EXPECT_TRUE(cache->ReleaseDescriptors().empty());
    mapped = cache->ReleaseMappedWeights();
  }
  // The mappings outlive the cache.
  ASSERT_EQ(mapped.size(), 2);
  for (const auto& [name, size, seed] :
       {std::tuple<std::string, int, int>{"w.a", 1000, 1},
        std::tuple<std::string, int, int>{"w.b+w.c", 37, 2}}) {
    ASSERT_TRUE(mapped.contains(name)) << name;
    const PersistentWeightCache::MappedWeight& weight = mapped[name];
    const ::ml_drift::TensorDescriptor expected = MakeDescriptor(size, seed);
    EXPECT_EQ(weight.desc.GetBHWCShape(), expected.GetBHWCShape());
    EXPECT_EQ(weight.desc.GetDataType(), expected.GetDataType());
    EXPECT_TRUE(weight.desc.GetData().empty());
    EXPECT_NE(weight.mapping, nullptr);
    EXPECT_THAT(weight.data, ElementsAreArray(expected.GetData()));
    // Page-aligned, e.g. to back GPU buffers.
    EXPECT_EQ(reinterpret_cast<uintptr_t>(weight.data.data()) %
                  PersistentWeightCache::kEntryAlignment,
              0)
        << name;
  }
}

TEST(PersistentWeightCacheTest, EntriesStartAtTheEntryAlignment) {
  const std::string path = CachePath("aligned.bin");
  BuildCache(path, 5);
  const std::vector<char> bytes = ReadFile(path);
  const size_t a = FindData(bytes, MakeDescriptor(1000, 1));
  EXPECT_EQ(a % PersistentWeightCache::kEntryAlignment, 0);
  // After "w.a" (whose bytes contain those of "w.b+w.c").
  const size_t b = FindData(bytes, MakeDescriptor(37, 2), a + 1000);
  EXPECT_NE(b, std::string::npos);
  EXPECT_EQ(b % PersistentWeightCache::kEntryAlignment, 0);
}

// Mapping without `verify_all_bytes` checks both ends of every weight.
TEST(PersistentWeightCacheTest, MappingDetectsCorruptWeightEnds) {
  const ::ml_drift::TensorDescriptor large = MakeDescriptor(
      3 * static_cast<int>(PersistentWeightCache::kSampleBytes), 9);
  for (const bool corrupt_tail : {false, true}) {
    const std::string path = CachePath("mapped_corrupt.bin");
    {
      ASSERT_OK_AND_ASSIGN(auto cache, PersistentWeightCache::Open(path, 3));
      ASSERT_OK(cache->Insert("w.large", large));
      ASSERT_OK(cache->Finish());
    }
    std::vector<char> bytes = ReadFile(path);
    const size_t start = FindData(bytes, large);
    ASSERT_NE(start, std::string::npos);
    bytes[start + (corrupt_tail ? large.GetData().size() - 1 : 0)] ^= 0x5A;
    WriteFile(path, bytes);
    ASSERT_OK_AND_ASSIGN(auto cache,
                         PersistentWeightCache::Open(path, 3, kMapOptions));
    EXPECT_FALSE(cache->loaded()) << corrupt_tail;
    EXPECT_TRUE(cache->building());
  }
}

TEST(PersistentWeightCacheTest, MappingCanVerifyAllBytes) {
  const ::ml_drift::TensorDescriptor large = MakeDescriptor(
      3 * static_cast<int>(PersistentWeightCache::kSampleBytes), 9);
  const std::string path = CachePath("mapped_verify.bin");
  {
    ASSERT_OK_AND_ASSIGN(auto cache, PersistentWeightCache::Open(path, 3));
    ASSERT_OK(cache->Insert("w.large", large));
    ASSERT_OK(cache->Finish());
  }
  std::vector<char> bytes = ReadFile(path);
  bytes[FindData(bytes, large) + large.GetData().size() / 2] ^= 0x5A;
  WriteFile(path, bytes);
  {
    // Not in the sampled ends.
    ASSERT_OK_AND_ASSIGN(auto cache,
                         PersistentWeightCache::Open(path, 3, kMapOptions));
    EXPECT_TRUE(cache->loaded());
  }
  ASSERT_OK_AND_ASSIGN(
      auto cache, PersistentWeightCache::Open(
                      path, 3, {.map_data = true, .verify_all_bytes = true}));
  EXPECT_FALSE(cache->loaded());
}

TEST(PersistentWeightCacheTest, RebuildsATruncatedFile) {
  const std::string path = CachePath("truncated.bin");
  BuildCache(path, 7);
  std::vector<char> bytes = ReadFile(path);
  bytes.resize(bytes.size() / 2);
  WriteFile(path, bytes);
  for (const auto& options : {PersistentWeightCache::Options(), kMapOptions}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                         PersistentWeightCache::Open(path, 7, options));
    EXPECT_FALSE(cache->loaded());
    EXPECT_TRUE(cache->building());
  }
}

// A build that never finishes (e.g. a crash) leaves the old cache in place.
TEST(PersistentWeightCacheTest, UnfinishedBuildKeepsTheOldCache) {
  const std::string path = CachePath("unfinished.bin");
  BuildCache(path, 3);
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                         PersistentWeightCache::Open(path, 4));
    ASSERT_TRUE(cache->building());
    ASSERT_OK(cache->Insert("w.z", MakeDescriptor(10, 5)));
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                       PersistentWeightCache::Open(path, 3));
  EXPECT_TRUE(cache->loaded());
  EXPECT_EQ(cache->num_entries(), 2);
}

// Concurrent builders for the same target path use distinct temp files.
TEST(PersistentWeightCacheTest, ConcurrentBuildersUseDistinctTempFiles) {
  const std::string path = CachePath("concurrent.bin");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> builder1,
                       PersistentWeightCache::Open(path, 11));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> builder2,
                       PersistentWeightCache::Open(path, 11));
  ASSERT_TRUE(builder1->building());
  ASSERT_TRUE(builder2->building());
  ASSERT_OK(builder1->Insert("w.one", MakeDescriptor(16, 1)));
  ASSERT_OK(builder2->Insert("w.two", MakeDescriptor(24, 2)));
  ASSERT_OK(builder1->Finish());
  ASSERT_OK(builder2->Finish());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> reloaded,
                       PersistentWeightCache::Open(path, 11));
  ASSERT_TRUE(reloaded->loaded());
  EXPECT_TRUE(reloaded->ReleaseDescriptors().contains("w.two"));
}

TEST(PersistentWeightCacheTest, IdentifierDependsOnEveryKey) {
  const uint64_t id = WeightCacheIdentifier("model", "gpu", "options");
  EXPECT_EQ(id, WeightCacheIdentifier("model", "gpu", "options"));
  EXPECT_NE(id, WeightCacheIdentifier("model2", "gpu", "options"));
  EXPECT_NE(id, WeightCacheIdentifier("model", "gpu2", "options"));
  EXPECT_NE(id, WeightCacheIdentifier("model", "gpu", "options2"));
}

TEST(PersistentWeightCacheTest,
     AppendAddsNewEntriesPreservingExistingAndAlignment) {
  const std::string path = CachePath("append.bin");
  BuildCache(path, 11);

  std::vector<std::pair<std::string, ::ml_drift::TensorDescriptor>> extra;
  extra.emplace_back("w.a", MakeDescriptor(12, 9));  // Already cached: skipped.
  extra.emplace_back("w.c", MakeDescriptor(4096 + 17, 5));
  extra.emplace_back("w.d", MakeDescriptor(19, 6));
  extra.emplace_back("w.c", MakeDescriptor(7, 8));  // Duplicate extra: skipped.

  ASSERT_OK(PersistentWeightCache::Append(path, 11, extra,
                                          /*verify_all_bytes=*/true));
  // No-op when all names are already present.
  ASSERT_OK(PersistentWeightCache::Append(path, 11, extra,
                                          /*verify_all_bytes=*/false));

  for (const PersistentWeightCache::Options& options :
       {PersistentWeightCache::Options(), kMapOptions}) {
    ASSERT_OK_AND_ASSIGN(auto cache,
                         PersistentWeightCache::Open(path, 11, options));
    ASSERT_TRUE(cache->loaded());
    EXPECT_EQ(cache->num_entries(), 4);
    if (options.map_data) {
      auto mapped = cache->ReleaseMappedWeights();
      for (const auto& [name, size, seed] :
           {std::tuple<std::string, int, int>{"w.a", 1000, 1},
            std::tuple<std::string, int, int>{"w.b+w.c", 37, 2},
            std::tuple<std::string, int, int>{"w.c", 4096 + 17, 5},
            std::tuple<std::string, int, int>{"w.d", 19, 6}}) {
        ASSERT_TRUE(mapped.contains(name)) << name;
        const ::ml_drift::TensorDescriptor expected =
            MakeDescriptor(size, seed);
        EXPECT_THAT(mapped[name].data, ElementsAreArray(expected.GetData()));
        EXPECT_EQ(reinterpret_cast<uintptr_t>(mapped[name].data.data()) %
                      PersistentWeightCache::kEntryAlignment,
                  0);
      }
    } else {
      auto descs = cache->ReleaseDescriptors();
      for (const auto& [name, size, seed] :
           {std::tuple<std::string, int, int>{"w.a", 1000, 1},
            std::tuple<std::string, int, int>{"w.b+w.c", 37, 2},
            std::tuple<std::string, int, int>{"w.c", 4096 + 17, 5},
            std::tuple<std::string, int, int>{"w.d", 19, 6}}) {
        ASSERT_TRUE(descs.contains(name)) << name;
        EXPECT_THAT(descs[name].GetData(),
                    ElementsAreArray(MakeDescriptor(size, seed).GetData()));
      }
    }
  }

  // Wrong identifier fails and keeps the existing cache untouched.
  EXPECT_FALSE(PersistentWeightCache::Append(path, 99, extra).ok());
  ASSERT_OK_AND_ASSIGN(auto intact, PersistentWeightCache::Open(path, 11));
  EXPECT_TRUE(intact->loaded());
  EXPECT_EQ(intact->num_entries(), 4);
}

// A descriptor that isn't well-formed or whose data doesn't cover the memory
// it describes makes `Open` rebuild the cache instead of loading it.
TEST(PersistentWeightCacheTest, RebuildsForInvalidDescriptors) {
  ::ml_drift::TensorDescriptor unknown_type(
      ::ml_drift::DataType::kUnknown, ::ml_drift::TensorStorageType::kBuffer,
      ::ml_drift::Layout::kLinear);
  unknown_type.SetBHWCShape(::ml_drift::BHWC(1, 1, 1, 8));
  unknown_type.SetData(std::vector<uint8_t>(8, 1));
  ::ml_drift::TensorDescriptor short_data = MakeDescriptor(64, 3);
  short_data.SetData(std::vector<uint8_t>(16, 2));
  int i = 0;
  for (const ::ml_drift::TensorDescriptor* desc :
       {&unknown_type, &short_data}) {
    const std::string path = CachePath(absl::StrCat("invalid_desc_", i++));
    {
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                           PersistentWeightCache::Open(path, 7));
      ASSERT_TRUE(cache->building());
      ASSERT_OK(cache->Insert("w", *desc));
      ASSERT_OK(cache->Finish());
    }
    for (const auto& options :
         {PersistentWeightCache::Options(), kMapOptions}) {
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<PersistentWeightCache> cache,
                           PersistentWeightCache::Open(path, 7, options));
      EXPECT_FALSE(cache->loaded());
      EXPECT_TRUE(cache->building());
    }
  }
}

TEST(PersistentWeightCacheTest, IdentifierKeysCannotRunTogether) {
  EXPECT_NE(WeightCacheIdentifier("a|b", "c", "d"),
            WeightCacheIdentifier("a", "b|c", "d"));
  EXPECT_NE(WeightCacheIdentifier("a", "b", "c|d"),
            WeightCacheIdentifier("a", "b|c", "d"));
}

}  // namespace
}  // namespace litert::tensor
