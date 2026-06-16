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

#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_weight_cache.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "ml_drift/common/access_type.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/serialization_base_generated.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/build_identifier.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/cache_builder.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_schema_generated.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/testing_util.h"

namespace ml_drift {
class SerializationWeightCacheTestPeer {
 public:
  static void InjectMaliciousCacheEntry(SerializationWeightCache& cache,
                                        uint32_t global_tensor_id,
                                        bool is_quantization_param_tensor,
                                        size_t offset, size_t size) {
    CacheKey key{global_tensor_id, is_quantization_param_tensor};
    auto it = cache.global_tensor_id_to_cache_entry_.find(key);
    if (it != cache.global_tensor_id_to_cache_entry_.end()) {
      it->second.location.offset = offset;
      it->second.location.size = size;
    }
  }
  static void SetMMapBaseOffset(SerializationWeightCache& cache,
                                size_t offset) {
    cache.mmap_buffer_base_offset_ = offset;
  }
};
}  // namespace ml_drift

namespace mldrift {
namespace {

using ::ml_drift::DataType;
using ::ml_drift::Layout;
using ::ml_drift::TensorStorageType;
using ::mldrift::testing_util::TempFileDesc;
using ::testing::AllOf;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Property;
using ::testing::status::StatusIs;

MATCHER_P(TensorDescEq, expected, "") {
  return ExplainMatchResult(
      AllOf(Property(&ml_drift::TensorDescriptor::GetDataType,
                     Eq(expected.GetDataType())),
            Property(&ml_drift::TensorDescriptor::GetStorageType,
                     Eq(expected.GetStorageType())),
            Property(&ml_drift::TensorDescriptor::GetLayout,
                     Eq(expected.GetLayout())),
            Property(&ml_drift::TensorDescriptor::GetData,
                     ElementsAreArray(expected.GetData()))),
      arg, result_listener);
}

struct SerializationWeightCacheTest : public ::testing::TestWithParam<bool> {
  void SetUp() override { AddTensors(); }

  void AddTensors() {
    const int num_external_tensors = 20;
    // Create a few different types of tensors to help verify
    // serialized tensors are matched with their correct ids.
    static const DataType kDataTypes[] = {DataType::FLOAT32, DataType::INT8};
    static const TensorStorageType kStorageTypes[] = {
        TensorStorageType::TEXTURE_2D, TensorStorageType::IMAGE_BUFFER,
        TensorStorageType::BUFFER};
    static const Layout kLayouts[] = {Layout::HWC, Layout::BHWDC,
                                      Layout::LINEAR, Layout::BHWC};

    for (size_t i = 0; i < num_external_tensors; ++i) {
      DataType data_type = kDataTypes[i % 2];
      TensorStorageType storage_type = kStorageTypes[i % 3];
      Layout layout = kLayouts[i % 4];
      tensor_descs[i] =
          ml_drift::TensorDescriptor(data_type, storage_type, layout);
      // Fill the data with a unique value for each tensor.
      std::vector<uint8_t> data(i);
      for (size_t j = 0; j < i; ++j) {
        data[j] = j;
      }
      tensor_descs[i].SetAccess(ml_drift::AccessType::READ);
      tensor_descs[i].SetData(std::move(data));
    }
  }

  void InsertTensors(bool is_quantization_param_tensor) {
    for (const auto& [id, tensor_desc] : tensor_descs) {
      ASSERT_OK(
          cache.Insert(id, is_quantization_param_tensor,
                       /*packing_algorithm=*/
                       ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                       tensor_desc));
    }
  }

  bool GetQuantizationParamTensor() { return GetParam(); }

  absl::flat_hash_map<uint32_t, ml_drift::TensorDescriptor> tensor_descs;
  ml_drift::SerializationWeightCache cache;
  std::string tmp_dir = ::testing::TempDir();
};

TEST_P(SerializationWeightCacheTest, StartBuildFailsIfFilePathIsInvalid) {
  const uint64_t unique_model_identifier = 9876;
  EXPECT_THAT(
      cache.StartBuild("", "model_token", unique_model_identifier),
      StatusIs(::util::error::INTERNAL, HasSubstr("Could not open file")));

  EXPECT_THAT(
      cache.StartBuild("/seldf/sedsft", "model_token", unique_model_identifier),
      StatusIs(::util::error::INTERNAL, HasSubstr("Could not open file")));
}

TEST_P(SerializationWeightCacheTest, StartBuildSucceeds) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(tmp_dir, "start_build_succeeds",
                             unique_model_identifier));
  ASSERT_OK(cache.StopBuild());
}

TEST_P(SerializationWeightCacheTest,
       EmptyCacheFileIsWrittenAndLoadedSuccessfully) {
  // Start build but insert no tensors.
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(
      cache.StartBuild(tmp_dir, "empty_cache_file", unique_model_identifier));
  ASSERT_OK(cache.StopBuild());

  // Load the "empty" cache file.
  ASSERT_OK(cache.Load(tmp_dir, "empty_cache_file", unique_model_identifier));

  // Attempting to lookup any tensor should fail since it is empty.
  uint32_t global_tensor_id = 12345;
  ml_drift::TensorDescriptor tensor_desc;
  EXPECT_THAT(
      cache.LookUp(global_tensor_id, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   tensor_desc),
      StatusIs(::util::error::INVALID_ARGUMENT,
               HasSubstr("Failed to look up ")));
}

TEST_P(SerializationWeightCacheTest, InsertFailsWhenCacheIsNotBuilding) {
  EXPECT_THAT(
      cache.Insert(tensor_descs.begin()->first, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   tensor_descs.begin()->second),
      StatusIs(
          ::util::error::INVALID_ARGUMENT,
          HasSubstr("Cannot insert a buffer in a cache that is not building")));
}

TEST_P(SerializationWeightCacheTest,
       InsertFailsWhenTensorDescriptorIsAlreadyInserted) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(
      tmp_dir, "insert_fails_when_tensor_descriptor_is_already_inserted",
      unique_model_identifier));
  ASSERT_OK(
      cache.Insert(tensor_descs.begin()->first, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   tensor_descs.begin()->second));
  EXPECT_THAT(
      cache.Insert(tensor_descs.begin()->first, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   tensor_descs.begin()->second),
      StatusIs(::util::error::INVALID_ARGUMENT,
               HasSubstr("Tensor already exists in cache")));
}

TEST_P(SerializationWeightCacheTest, LookUpFailsIfKeyDoesntMatch) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(tmp_dir, "look_up_fails_if_key_doesnt_match",
                             unique_model_identifier));
  InsertTensors(/*is_quantization_param_tensor=*/GetParam());
  ASSERT_OK(cache.StopBuild());

  uint32_t global_tensor_id = 12345;
  ml_drift::TensorDescriptor tensor_desc;
  tensor_desc.SetAccess(ml_drift::AccessType::READ);
  EXPECT_THAT(
      cache.LookUp(global_tensor_id, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   tensor_desc),
      StatusIs(::util::error::INVALID_ARGUMENT,
               HasSubstr("Failed to look up ")));
}

TEST_P(SerializationWeightCacheTest, LookUpFailsIfCacheIsBuilding) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(tmp_dir, "look_up_fails_if_cache_is_building",
                             unique_model_identifier));
  InsertTensors(/*is_quantization_param_tensor=*/GetParam());

  for (const auto& [id, tensor_desc] : tensor_descs) {
    ml_drift::TensorDescriptor looked_up_tensor_desc;
    EXPECT_THAT(
        cache.LookUp(id, GetQuantizationParamTensor(),
                     /*packing_algorithm=*/
                     ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                     looked_up_tensor_desc),
        StatusIs(
            ::util::error::INVALID_ARGUMENT,
            HasSubstr("Cannot look up a buffer in a cache that is building")));
  }
}

TEST_P(SerializationWeightCacheTest, LookUpSucceeds) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(
      cache.StartBuild(tmp_dir, "look_up_succeeds", unique_model_identifier));
  InsertTensors(/*is_quantization_param_tensor=*/GetParam());
  ASSERT_OK(cache.StopBuild());
  ASSERT_OK(cache.Load(tmp_dir, "look_up_succeeds", unique_model_identifier));

  for (const auto& [id, tensor_desc] : tensor_descs) {
    ml_drift::TensorDescriptor looked_up_tensor_desc;
    ASSERT_OK(
        cache.LookUp(id, GetQuantizationParamTensor(),
                     /*packing_algorithm=*/
                     ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                     looked_up_tensor_desc));
    EXPECT_THAT(looked_up_tensor_desc, TensorDescEq(tensor_desc));
  }
}

TEST_P(SerializationWeightCacheTest,
       LookUpSucceedsEvenWithEmptyTensorDescriptorData) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(
      tmp_dir, "look_up_succeeds_even_with_empty_tensor_descriptor_data",
      unique_model_identifier));
  uint32_t global_tensor_id = 10;
  ml_drift::TensorDescriptor tensor_desc;
  tensor_desc.SetAccess(ml_drift::AccessType::READ);
  ASSERT_OK(cache.Insert(
      global_tensor_id, GetQuantizationParamTensor(),
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, tensor_desc));
  EXPECT_OK(cache.StopBuild());

  ASSERT_OK(cache.Load(
      tmp_dir, "look_up_succeeds_even_with_empty_tensor_descriptor_data",
      unique_model_identifier));

  ml_drift::TensorDescriptor looked_up_tensor_desc;
  ASSERT_OK(
      cache.LookUp(global_tensor_id, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   looked_up_tensor_desc));
  EXPECT_THAT(looked_up_tensor_desc, TensorDescEq(tensor_desc));
}

TEST_P(SerializationWeightCacheTest, LookUpIntegerOverflowFails) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(tmp_dir, "look_up_integer_overflow_fails",
                             unique_model_identifier));
  uint32_t global_tensor_id = 10;
  ml_drift::TensorDescriptor tensor_desc;
  tensor_desc.SetAccess(ml_drift::AccessType::READ);
  tensor_desc.SetData({1, 2, 3, 4});
  ASSERT_OK(cache.Insert(
      global_tensor_id, GetQuantizationParamTensor(),
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, tensor_desc));
  EXPECT_OK(cache.StopBuild());

  ASSERT_OK(cache.Load(tmp_dir, "look_up_integer_overflow_fails",
                       unique_model_identifier));

  // Inject a malicious entry that will cause integer overflow.
  ml_drift::SerializationWeightCacheTestPeer::SetMMapBaseOffset(cache, 100);
  size_t huge_offset = std::numeric_limits<size_t>::max() - 99;
  ml_drift::SerializationWeightCacheTestPeer::InjectMaliciousCacheEntry(
      cache, global_tensor_id, GetQuantizationParamTensor(), huge_offset, 4);

  ml_drift::TensorDescriptor looked_up_tensor_desc;
  EXPECT_THAT(
      cache.LookUp(global_tensor_id, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   looked_up_tensor_desc),
      StatusIs(::util::error::INVALID_ARGUMENT,
               HasSubstr("Cache entry offset integer overflow:")));
}

TEST_P(SerializationWeightCacheTest, LookUpOutOfBoundsFails) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(tmp_dir, "look_up_out_of_bounds_fails",
                             unique_model_identifier));
  uint32_t global_tensor_id = 10;
  ml_drift::TensorDescriptor tensor_desc;
  tensor_desc.SetAccess(ml_drift::AccessType::READ);
  tensor_desc.SetData({1, 2, 3, 4});
  ASSERT_OK(cache.Insert(
      global_tensor_id, GetQuantizationParamTensor(),
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, tensor_desc));
  EXPECT_OK(cache.StopBuild());

  ASSERT_OK(cache.Load(tmp_dir, "look_up_out_of_bounds_fails",
                       unique_model_identifier));

  // Inject a malicious entry with a massive size.
  ml_drift::SerializationWeightCacheTestPeer::InjectMaliciousCacheEntry(
      cache, global_tensor_id, GetQuantizationParamTensor(), 0,
      std::numeric_limits<size_t>::max() / 2);

  ml_drift::TensorDescriptor looked_up_tensor_desc;
  EXPECT_THAT(
      cache.LookUp(global_tensor_id, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   looked_up_tensor_desc),
      StatusIs(::util::error::INVALID_ARGUMENT,
               HasSubstr("Cache entry location out of bounds:")));
}

class OOMTensorDescriptor : public ml_drift::TensorDescriptor {
 public:
  absl::Span<const uint8_t> GetData() const override {
    // Return a massive span size to simulate OOM without actually allocating.
    // std::numeric_limits<size_t>::max() / 2 avoids overflow when adding
    // alignment.
    return absl::MakeSpan(reinterpret_cast<const uint8_t*>(1),
                          std::numeric_limits<size_t>::max() / 2);
  }
};

TEST_P(SerializationWeightCacheTest, InsertHandlesOOMGracefully) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(tmp_dir, "insert_handles_oom_gracefully",
                             unique_model_identifier));
  uint32_t global_tensor_id = 10;
  OOMTensorDescriptor tensor_desc;

  EXPECT_THAT(
      cache.Insert(global_tensor_id, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   tensor_desc),
      StatusIs(
          ::util::error::RESOURCE_EXHAUSTED,
          HasSubstr("Failed to allocate memory for cache staging buffer.")));
  EXPECT_OK(cache.StopBuild());
}

TEST_P(SerializationWeightCacheTest, ExceedsMaxSupportedSubgraphsFails) {
  const uint64_t unique_model_identifier_base = 1000;
  ml_drift::TensorDescriptor tensor_desc;
  tensor_desc.SetAccess(ml_drift::AccessType::READ);

  for (int i = 0; i <= ml_drift::kMaxSupportedSubgraphs; ++i) {
    ASSERT_OK(cache.StartBuild(tmp_dir, "exceeds_max_subgraphs",
                               unique_model_identifier_base + i));
    ASSERT_OK(cache.Insert(
        i, GetQuantizationParamTensor(),
        /*packing_algorithm=*/
        ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, tensor_desc));
    ASSERT_OK(cache.StopBuild());
  }

  EXPECT_THAT(cache.StartBuild(tmp_dir, "exceeds_max_subgraphs",
                               unique_model_identifier_base +
                                   ml_drift::kMaxSupportedSubgraphs + 1),
              StatusIs(::util::error::INTERNAL,
                       HasSubstr("Corrupted cache: Too many subgraphs.")));
}

TEST_P(SerializationWeightCacheTest, LookUpWithoutLoadFails) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(tmp_dir, "look_up_without_load_fails",
                             unique_model_identifier));
  InsertTensors(/*is_quantization_param_tensor=*/GetParam());
  ASSERT_OK(cache.StopBuild());

  ml_drift::SerializationWeightCache new_cache;
  for (const auto& [id, tensor_desc] : tensor_descs) {
    ml_drift::TensorDescriptor looked_up_tensor_desc;
    EXPECT_THAT(
        new_cache.LookUp(
            id, GetQuantizationParamTensor(),
            /*packing_algorithm=*/
            ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
            looked_up_tensor_desc),
        StatusIs(::util::error::INVALID_ARGUMENT,
                 HasSubstr("Cannot look up a buffer in a cache that is not "
                           "loaded.")));
  }
}

TEST_P(SerializationWeightCacheTest, UniqueModelIdentifierIsUsedToRejectCache) {
  const uint64_t unique_model_identifier = 9876;
  ASSERT_OK(cache.StartBuild(tmp_dir,
                             "unique_model_identifier_is_used_to_reject_cache",
                             unique_model_identifier));
  uint32_t global_tensor_id = 10;
  ml_drift::TensorDescriptor tensor_desc;
  tensor_desc.SetAccess(ml_drift::AccessType::READ);
  ASSERT_OK(cache.Insert(
      global_tensor_id, GetQuantizationParamTensor(),
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, tensor_desc));
  EXPECT_OK(cache.StopBuild());

  const uint64_t different_unique_model_identifier = 67890;
  EXPECT_THAT(
      cache.Load(tmp_dir, "unique_model_identifier_is_used_to_reject_cache",
                 different_unique_model_identifier),
      testing::status::StatusIs(::util::error::NOT_FOUND,
                                testing::HasSubstr("not found in cache")));

  ASSERT_OK(cache.Load(tmp_dir,
                       "unique_model_identifier_is_used_to_reject_cache",
                       unique_model_identifier));
}

TEST_P(SerializationWeightCacheTest, MultipleSubgraphsShareCacheWorks) {
  ASSERT_OK(cache.StartBuild(tmp_dir, "multiple_subgraphs_share_cache", 111));
  uint32_t id1 = 10;
  ml_drift::TensorDescriptor desc1;
  desc1.SetAccess(ml_drift::AccessType::READ);
  ASSERT_OK(cache.Insert(
      id1, GetQuantizationParamTensor(),
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, desc1));
  EXPECT_OK(cache.StopBuild());

  // Now build for another subgraph on the same file.
  // We need to reuse the file, so we use the same model_token.
  // But we use a different unique_model_identifier.
  ASSERT_OK(cache.StartBuild(tmp_dir, "multiple_subgraphs_share_cache", 222));
  uint32_t id2 = 20;
  ml_drift::TensorDescriptor desc2;
  desc2.SetAccess(ml_drift::AccessType::WRITE);
  ASSERT_OK(cache.Insert(
      id2, GetQuantizationParamTensor(),
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, desc2));
  EXPECT_OK(cache.StopBuild());

  // Verify we can load and find tensors for subgraph 1.
  ASSERT_OK(cache.Load(tmp_dir, "multiple_subgraphs_share_cache", 111));
  ml_drift::TensorDescriptor looked_up_desc1;
  ASSERT_OK(
      cache.LookUp(id1, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   looked_up_desc1));
  EXPECT_THAT(looked_up_desc1, TensorDescEq(desc1));

  // Verify we can load and find tensors for subgraph 2.
  ASSERT_OK(cache.Load(tmp_dir, "multiple_subgraphs_share_cache", 222));
  ml_drift::TensorDescriptor looked_up_desc2;
  ASSERT_OK(
      cache.LookUp(id2, GetQuantizationParamTensor(),
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   looked_up_desc2));
  EXPECT_THAT(looked_up_desc2, TensorDescEq(desc2));
}

TEST_P(SerializationWeightCacheTest, LoadFailureThenStartBuildSucceeds) {
  const uint64_t identifier1 = 111;
  const uint64_t identifier2 = 222;
  const std::string model_token = "load_failure_then_start_build_succeeds";

  // 1. Create cache file with identifier1.
  ASSERT_OK(cache.StartBuild(tmp_dir, model_token, identifier1));
  uint32_t id1 = 10;
  ml_drift::TensorDescriptor desc1;
  desc1.SetAccess(ml_drift::AccessType::READ);
  ASSERT_OK(cache.Insert(
      id1, GetQuantizationParamTensor(),
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, desc1));
  EXPECT_OK(cache.StopBuild());

  // Create a new cache instance to simulate a new session/process.
  ml_drift::SerializationWeightCache new_cache;

  // 2. Attempt to load identifier2 (which is NOT in the cache).
  // This should fail, but it will open the file O_RDONLY.
  EXPECT_THAT(new_cache.Load(tmp_dir, model_token, identifier2),
              StatusIs(::util::error::NOT_FOUND));

  // 3. Call StartBuild for identifier2.
  // With the bug, this will fail because it tries to reuse the O_RDONLY FD.
  // With the fix, it should succeed and append.
  ASSERT_OK(new_cache.StartBuild(tmp_dir, model_token, identifier2));
  uint32_t id2 = 20;
  ml_drift::TensorDescriptor desc2;
  desc2.SetAccess(ml_drift::AccessType::WRITE);
  ASSERT_OK(new_cache.Insert(
      id2, GetQuantizationParamTensor(),
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, desc2));
  EXPECT_OK(new_cache.StopBuild());

  // Verify we can load both now.
  ASSERT_OK(new_cache.Load(tmp_dir, model_token, identifier1));
  ml_drift::TensorDescriptor looked_up_desc1;
  ASSERT_OK(
      new_cache.LookUp(id1, GetQuantizationParamTensor(),
                       /*packing_algorithm=*/
                       ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                       looked_up_desc1));
  EXPECT_THAT(looked_up_desc1, TensorDescEq(desc1));

  ASSERT_OK(new_cache.Load(tmp_dir, model_token, identifier2));
  ml_drift::TensorDescriptor looked_up_desc2;
  ASSERT_OK(
      new_cache.LookUp(id2, GetQuantizationParamTensor(),
                       /*packing_algorithm=*/
                       ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                       looked_up_desc2));
  EXPECT_THAT(looked_up_desc2, TensorDescEq(desc2));
}

TEST(SerializationWeightCacheTest, FileDescriptorSupportWorks) {
  const std::string tmp_dir = std::getenv("TEST_TMPDIR");
  const std::string file_path = tmp_dir + "/fd_test_cache.bin";

  // Clean up if file exists.
  std::remove(file_path.c_str());

  int fd = open(file_path.c_str(), O_CREAT | O_RDWR,  // NOLINT: b/332641196
                0644);
  ASSERT_GE(fd, 0);

  ml_drift::SerializationWeightCache cache;
  uint64_t identifier = 12345;

  // Test StartBuild with FD.
  ASSERT_OK(cache.StartBuild(fd, identifier));

  uint32_t tensor_id = 10;
  ml_drift::TensorDescriptor desc;
  desc.SetAccess(ml_drift::AccessType::READ);
  std::vector<uint8_t> data = {1, 2, 3, 4};
  desc.SetData(std::move(data));

  ASSERT_OK(cache.Insert(
      tensor_id, /*is_quantization_param_tensor=*/false,
      /*packing_algorithm=*/
      ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN, desc));
  EXPECT_OK(cache.StopBuild());

  // Test Load with FD.
  // We need a new FD because the previous one was closed by cache destruction
  // or reset.
  int fd_read = open(file_path.c_str(), O_RDONLY);  // NOLINT: b/332641196
  ASSERT_GE(fd_read, 0);

  ASSERT_OK(cache.Load(fd_read, identifier));

  ml_drift::TensorDescriptor looked_up_desc;
  ASSERT_OK(
      cache.LookUp(tensor_id, /*is_quantization_param_tensor=*/false,
                   /*packing_algorithm=*/
                   ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                   looked_up_desc));

  EXPECT_THAT(looked_up_desc, TensorDescEq(desc));

  // Clean up.
  std::remove(file_path.c_str());
}

TEST(SerializationWeightCacheUniqueModelIdentifierTest,
     OptionsAffectFingerprint) {
  const std::string model_token = "test_model";
  const std::string prefix = "prefix";
  const MlDriftDelegatePrecision precision = kDefault;

  // Base fingerprint with default options.
  uint64_t base_fp =
      ml_drift::SerializationWeightCache::GenerateUniqueModelIdentifier(
          model_token, /*context=*/nullptr, /*delegate_params=*/nullptr, prefix,
          precision,
          /*prefer_texture_weights=*/false,
          /*allow_src_quantized_fc_conv_ops=*/false,
          /*prepare_weights_in_batches=*/false,
          /*serialize_external_tensors=*/false,
          /*ordered_by_size=*/true);

  // Test precision changes fingerprint.
  uint64_t fp_precision =
      ml_drift::SerializationWeightCache::GenerateUniqueModelIdentifier(
          model_token, /*context=*/nullptr, /*delegate_params=*/nullptr, prefix,
          kFp16,
          /*prefer_texture_weights=*/false,
          /*allow_src_quantized_fc_conv_ops=*/false,
          /*prepare_weights_in_batches=*/false,
          /*serialize_external_tensors=*/false,
          /*ordered_by_size=*/true);
  EXPECT_NE(base_fp, fp_precision);

  // Test prefer_texture_weights changes fingerprint.
  uint64_t fp_prefer_texture =
      ml_drift::SerializationWeightCache::GenerateUniqueModelIdentifier(
          model_token, /*context=*/nullptr, /*delegate_params=*/nullptr, prefix,
          precision,
          /*prefer_texture_weights=*/true,
          /*allow_src_quantized_fc_conv_ops=*/false,
          /*prepare_weights_in_batches=*/false,
          /*serialize_external_tensors=*/false,
          /*ordered_by_size=*/true);
  EXPECT_NE(base_fp, fp_prefer_texture);

  // Test allow_src_quantized_fc_conv_ops changes fingerprint.
  uint64_t fp_allow_src_quantized =
      ml_drift::SerializationWeightCache::GenerateUniqueModelIdentifier(
          model_token, /*context=*/nullptr, /*delegate_params=*/nullptr, prefix,
          precision,
          /*prefer_texture_weights=*/false,
          /*allow_src_quantized_fc_conv_ops=*/true,
          /*prepare_weights_in_batches=*/false,
          /*serialize_external_tensors=*/false,
          /*ordered_by_size=*/true);
  EXPECT_NE(base_fp, fp_allow_src_quantized);

  // Test prepare_weights_in_batches changes fingerprint.
  uint64_t fp_prepare_batches =
      ml_drift::SerializationWeightCache::GenerateUniqueModelIdentifier(
          model_token, /*context=*/nullptr, /*delegate_params=*/nullptr, prefix,
          precision,
          /*prefer_texture_weights=*/false,
          /*allow_src_quantized_fc_conv_ops=*/false,
          /*prepare_weights_in_batches=*/true,
          /*serialize_external_tensors=*/false,
          /*ordered_by_size=*/true);
  EXPECT_NE(base_fp, fp_prepare_batches);

  // Test serialize_external_tensors changes fingerprint.
  uint64_t fp_serialize_tensors =
      ml_drift::SerializationWeightCache::GenerateUniqueModelIdentifier(
          model_token, /*context=*/nullptr, /*delegate_params=*/nullptr, prefix,
          precision,
          /*prefer_texture_weights=*/false,
          /*allow_src_quantized_fc_conv_ops=*/false,
          /*prepare_weights_in_batches=*/false,
          /*serialize_external_tensors=*/true,
          /*ordered_by_size=*/true);
  EXPECT_NE(base_fp, fp_serialize_tensors);

  // Test ordered_by_size changes fingerprint.
  uint64_t fp_ordered_by_size =
      ml_drift::SerializationWeightCache::GenerateUniqueModelIdentifier(
          model_token, /*context=*/nullptr, /*delegate_params=*/nullptr, prefix,
          precision,
          /*prefer_texture_weights=*/false,
          /*allow_src_quantized_fc_conv_ops=*/false,
          /*prepare_weights_in_batches=*/false,
          /*serialize_external_tensors=*/false,
          /*ordered_by_size=*/false);
  EXPECT_NE(base_fp, fp_ordered_by_size);
}

TEST(SerializationWeightCacheTest, StartBuildClosesFdWhenAlreadyBuilding) {
  const std::string tmp_dir = std::getenv("TEST_TMPDIR");
  const std::string file_path = tmp_dir + "/fd_close_test.bin";
  std::remove(file_path.c_str());

  int fd1 = open(file_path.c_str(), O_CREAT | O_RDWR, 0644);
  ASSERT_GE(fd1, 0);

  ml_drift::SerializationWeightCache cache;
  uint64_t identifier = 12345;

  ASSERT_OK(cache.StartBuild(fd1, identifier));

  int fd2 = open(file_path.c_str(), O_RDONLY);
  ASSERT_GE(fd2, 0);

  EXPECT_THAT(cache.StartBuild(fd2, identifier),
              StatusIs(::util::error::INVALID_ARGUMENT));

  EXPECT_EQ(fcntl(fd2, F_GETFD), -1);  // NOLINT: b/332641196
  EXPECT_EQ(errno, EBADF);

  EXPECT_OK(cache.StopBuild());
  std::remove(file_path.c_str());
}

TEST(SerializationWeightCacheSecurityTest,
     LoadFailsIfFlatbufferHasMissingShape) {
  const std::string tmp_dir = std::getenv("TEST_TMPDIR");
  const std::string file_path =
      tmp_dir + "/corrupted_flatbuffer_test_mldrift_weight_cache.bin";
  std::remove(file_path.c_str());

  flatbuffers::FlatBufferBuilder builder;

  // Build a TensorDescriptor missing the optional 'shape' field.
  // 1. GPUObjectDescriptor
  ml_drift::data::GPUObjectDescriptorBuilder obj_builder(builder);
  auto obj_offset = obj_builder.Finish();

  // 2. TensorDescriptor (omitting shape)
  ml_drift::data::TensorDescriptorBuilder tensor_builder(builder);
  tensor_builder.add_base_obj(obj_offset);
  tensor_builder.add_data_type(ml_drift::data::DataType::FLOAT32);
  auto tensor_desc_offset = tensor_builder.Finish();

  // 3. Buffer
  ml_drift::cache::schema::BufferBuilder buffer_builder(builder);
  buffer_builder.add_tensor_descriptor(tensor_desc_offset);
  buffer_builder.add_global_tensor_id(42);
  buffer_builder.add_is_quantization_param_tensor(false);
  buffer_builder.add_offset(100);
  buffer_builder.add_size(10);
  auto buffer_offset = buffer_builder.Finish();

  // 4. Subgraph
  std::vector<flatbuffers::Offset<ml_drift::cache::schema::Buffer>> buffers = {
      buffer_offset};
  auto buffers_vec = builder.CreateVector(buffers);

  ml_drift::cache::schema::SubgraphBufferListBuilder subgraph_builder(builder);
  uint64_t unique_model_identifier = 12345;
  subgraph_builder.add_unique_model_identifier(unique_model_identifier);
  subgraph_builder.add_buffers(buffers_vec);
  subgraph_builder.add_base_offset(100);
  auto subgraph_offset = subgraph_builder.Finish();

  // 5. ModelCache
  std::vector<flatbuffers::Offset<ml_drift::cache::schema::SubgraphBufferList>>
      subgraphs = {subgraph_offset};
  auto subgraphs_vec = builder.CreateVector(subgraphs);

  ml_drift::cache::schema::ModelCacheBuilder model_cache_builder(builder);
  model_cache_builder.add_subgraphs(subgraphs_vec);
  auto model_cache_offset = model_cache_builder.Finish();

  ml_drift::cache::schema::FinishModelCacheBuffer(builder, model_cache_offset);

  // Write to file
  int fd = open(file_path.c_str(), O_CREAT | O_RDWR, 0644);
  ASSERT_GE(fd, 0);

  // Header
  ml_drift::MLDriftCacheHeader header{ml_drift::MLDriftCacheHeader::kVersion};
  absl::Span<const uint8_t> build_identifier = ml_drift::GetBuildIdentifier();
  std::memcpy(header.mldrift_build_identifier, build_identifier.data(),
              build_identifier.size());
  header.buffer_list_offset = sizeof(header);
  header.buffer_list_size = builder.GetSize();

  // Write header and flatbuffer
  ASSERT_TRUE(write(fd, &header, sizeof(header)) == sizeof(header));
  ASSERT_TRUE(write(fd, builder.GetBufferPointer(), builder.GetSize()) ==
              static_cast<ssize_t>(builder.GetSize()));
  close(fd);

  // Try to load
  ml_drift::SerializationWeightCache cache;
  EXPECT_THAT(
      cache.Load(tmp_dir, "corrupted_flatbuffer_test", unique_model_identifier),
      StatusIs(::util::error::INVALID_ARGUMENT,
               HasSubstr("Tensor shape is null.")));

  std::remove(file_path.c_str());
}

TEST(SerializationWeightCacheSecurityTest, RejectsPathTraversal) {
  const std::string tmp_dir = std::getenv("TEST_TMPDIR");
  ml_drift::SerializationWeightCache cache;
  EXPECT_THAT(
      cache.StartBuild(tmp_dir, "../malicious_model_token", 1234),
      StatusIs(
          ::util::error::INVALID_ARGUMENT,
          HasSubstr(
              "Invalid model_token: contains path traversal characters.")));

  EXPECT_THAT(
      cache.Load(tmp_dir, "/absolute/malicious_model_token", 1234),
      StatusIs(
          ::util::error::INVALID_ARGUMENT,
          HasSubstr(
              "Invalid model_token: contains path traversal characters.")));

  EXPECT_THAT(
      cache.Load(tmp_dir, "foo\\bar", 1234),
      StatusIs(
          ::util::error::INVALID_ARGUMENT,
          HasSubstr(
              "Invalid model_token: contains path traversal characters.")));
}

INSTANTIATE_TEST_SUITE_P(WithParam, SerializationWeightCacheTest,
                         testing::Bool());

}  // namespace
}  // namespace mldrift
