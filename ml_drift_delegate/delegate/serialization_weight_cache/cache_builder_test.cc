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

#include "ml_drift_delegate/delegate/serialization_weight_cache/cache_builder.h"

#include <fcntl.h>  // IWYU pragma: keep b/332641196

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/serialization_base_generated.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/serialization_weight_cache/file_util.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/mmap_handle.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/serialization_schema_generated.h"
#include "ml_drift_delegate/delegate/serialization_weight_cache/testing_util.h"

namespace mldrift {

namespace {

using ml_drift::DataType;
using ml_drift::Layout;
using ml_drift::MMapHandle;
using ml_drift::TensorStorageType;
using testing::ElementsAreArray;
using testing_util::TempFileDesc;

ml_drift::data::TensorStorageType ToFB(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::BUFFER:
      return ml_drift::data::TensorStorageType::BUFFER;
    case TensorStorageType::IMAGE_BUFFER:
      return ml_drift::data::TensorStorageType::IMAGE_BUFFER;
    case TensorStorageType::TEXTURE_2D:
      return ml_drift::data::TensorStorageType::TEXTURE_2D;
    case TensorStorageType::TEXTURE_ARRAY:
      return ml_drift::data::TensorStorageType::TEXTURE_ARRAY;
    case TensorStorageType::TEXTURE_3D:
      return ml_drift::data::TensorStorageType::TEXTURE_3D;
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return ml_drift::data::TensorStorageType::SINGLE_TEXTURE_2D;
    case TensorStorageType::UNKNOWN:
      return ml_drift::data::TensorStorageType::UNKNOWN;
  }
}

ml_drift::data::Layout ToFB(Layout type) {
  switch (type) {
    case Layout::HWC:
      return ml_drift::data::Layout::HWC;
    case Layout::BHWC:
      return ml_drift::data::Layout::BHWC;
    case Layout::HWDC:
      return ml_drift::data::Layout::HWDC;
    case Layout::BHWDC:
      return ml_drift::data::Layout::BHWDC;
    case Layout::LINEAR:
      return ml_drift::data::Layout::LINEAR;
    case Layout::HW:
      return ml_drift::data::Layout::HW;
    default:
      return ml_drift::data::Layout::UNKNOWN;
  }
}

ml_drift::data::DataType ToFB(DataType type) {
  switch (type) {
    case DataType::BOOL:
      return ml_drift::data::DataType::BOOL;
    case DataType::FLOAT16:
      return ml_drift::data::DataType::FLOAT16;
    case DataType::FLOAT32:
      return ml_drift::data::DataType::FLOAT32;
    case DataType::FLOAT64:
      return ml_drift::data::DataType::FLOAT64;
    case DataType::BFLOAT16:
      return ml_drift::data::DataType::BFLOAT16;
    case DataType::UINT8:
      return ml_drift::data::DataType::UINT8;
    case DataType::INT8:
      return ml_drift::data::DataType::INT8;
    case DataType::UINT16:
      return ml_drift::data::DataType::UINT16;
    case DataType::INT16:
      return ml_drift::data::DataType::INT16;
    case DataType::UINT32:
      return ml_drift::data::DataType::UINT32;
    case DataType::INT32:
      return ml_drift::data::DataType::INT32;
    case DataType::UINT64:
      return ml_drift::data::DataType::UINT64;
    case DataType::INT64:
      return ml_drift::data::DataType::INT64;
    case DataType::INT4:
      return ml_drift::data::DataType::INT4;
    case DataType::UINT4:
      return ml_drift::data::DataType::UINT4;
    case DataType::INT3:
      return ml_drift::data::DataType::INT3;
    case DataType::UINT3:
      return ml_drift::data::DataType::UINT3;
    case DataType::INT2:
      return ml_drift::data::DataType::INT2;
    case DataType::UINT2:
      return ml_drift::data::DataType::UINT2;
    case DataType::INT1:
      return ml_drift::data::DataType::INT1;
    case DataType::UINT1:
      return ml_drift::data::DataType::UINT1;
    case DataType::UNKNOWN:
      return ml_drift::data::DataType::UNKNOWN;
  }
}

TEST(CacheBuilderTest, ReserveAppendWriteWorks) {
  using std::size;

  const std::string payload = "This is some data in the file.";
  const uint64_t global_tensor_id = 12345;

  ml_drift::CacheBuilder builder;
  const uint64_t unique_model_identifier = 9876;
  const std::string cache_path = testing::TempDir() + "/cache";
  ASSERT_OK(builder.Start(cache_path, unique_model_identifier));
  ASSERT_OK(builder.StartBuildStep(unique_model_identifier));

  const size_t payload_size = size(payload);
  void* buffer = builder.Reserve(payload_size);
  std::memcpy(buffer, payload.c_str(), payload_size);
  ml_drift::TensorDescriptor tensor_desc(
      ml_drift::DataType::FLOAT32, ml_drift::TensorStorageType::IMAGE_BUFFER,
      ml_drift::Layout::LINEAR);
  ml_drift::BufferLocation loc;
  ASSERT_OK(
      builder.Append(global_tensor_id,
                     /*is_quantization_param_tensor=*/false,
                     /*packing_algorithm=*/
                     ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                     tensor_desc, buffer, payload_size, loc));

  EXPECT_EQ(loc.size, payload_size);
  EXPECT_GE(builder.capacity(), payload_size);

  ASSERT_OK(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_OK(handle.Map(cache_path.c_str()));

  const ml_drift::MLDriftCacheHeader& header =
      *reinterpret_cast<const ml_drift::MLDriftCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, ml_drift::MLDriftCacheHeader::kVersion);
  ASSERT_NE(header.buffer_list_offset, 0);
  ASSERT_NE(header.buffer_list_size, 0);
  ASSERT_LE(header.buffer_list_offset + header.buffer_list_size, handle.size());

  const ml_drift::cache::schema::ModelCache* const model_cache =
      ml_drift::cache::schema::GetModelCache(handle.data() +
                                             header.buffer_list_offset);

  ASSERT_NE(model_cache, nullptr);
  ASSERT_NE(model_cache->subgraphs(), nullptr);
  ASSERT_EQ(model_cache->subgraphs()->size(), 1);
  auto subgraph = model_cache->subgraphs()->Get(0);
  ASSERT_EQ(subgraph->unique_model_identifier(), unique_model_identifier);
  auto buffers = subgraph->buffers();
  ASSERT_NE(buffers, nullptr);
  ASSERT_EQ(buffers->size(), 1);
  auto buffer_fb = buffers->Get(0);
  ASSERT_NE(buffer_fb, nullptr);
  ASSERT_EQ(buffer_fb->global_tensor_id(), global_tensor_id);
  ASSERT_EQ(buffer_fb->is_quantization_param_tensor(), false);
  ASSERT_EQ(buffer_fb->tensor_descriptor()->data_type(),
            ToFB(tensor_desc.GetDataType()));
  ASSERT_EQ(buffer_fb->tensor_descriptor()->storage_type(),
            ToFB(tensor_desc.GetStorageType()));
  ASSERT_EQ(buffer_fb->tensor_descriptor()->layout(),
            ToFB(tensor_desc.GetLayout()));
  ASSERT_EQ(buffer_fb->size(), size(payload));

  flatbuffers::Verifier verifier(handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  EXPECT_TRUE(ml_drift::cache::schema::VerifyModelCacheBuffer(verifier));

  ASSERT_LE(subgraph->base_offset() + buffer_fb->offset(), size(handle));
  ASSERT_LE(subgraph->base_offset() + buffer_fb->offset() + buffer_fb->size(),
            size(handle));
  std::tuple<const char*, size_t> cache_data(
      reinterpret_cast<const char*>(handle.data() + subgraph->base_offset() +
                                    buffer_fb->offset()),
      buffer_fb->size());
  EXPECT_THAT(cache_data, ElementsAreArray(payload));
}

TEST(CacheBuilderTest, AppendWithoutReserveWriteWorks) {
  using std::size;

  const std::string payload = "This is some data in the file.";
  const uint64_t global_tensor_id = 12345;

  const uint64_t unique_model_identifier = 9876;
  const std::string cache_path = testing::TempDir() + "/cache";
  ml_drift::CacheBuilder builder;
  ASSERT_OK(builder.Start(cache_path, unique_model_identifier));
  ASSERT_OK(builder.StartBuildStep(unique_model_identifier));

  ml_drift::TensorDescriptor tensor_desc(
      ml_drift::DataType::FLOAT32, ml_drift::TensorStorageType::IMAGE_BUFFER,
      ml_drift::Layout::LINEAR);
  const size_t payload_size = size(payload);
  ml_drift::BufferLocation loc;
  ASSERT_OK(
      builder.Append(global_tensor_id,
                     /*is_quantization_param_tensor=*/false,
                     /*packing_algorithm=*/
                     ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                     tensor_desc, payload.data(), payload_size, loc));

  EXPECT_EQ(loc.size, payload_size);

  ASSERT_OK(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_OK(handle.Map(cache_path.c_str()));

  const ml_drift::MLDriftCacheHeader& header =
      *reinterpret_cast<const ml_drift::MLDriftCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, ml_drift::MLDriftCacheHeader::kVersion);
  ASSERT_NE(header.buffer_list_offset, 0);
  ASSERT_NE(header.buffer_list_size, 0);
  ASSERT_LE(header.buffer_list_offset + header.buffer_list_size, handle.size());

  const ml_drift::cache::schema::ModelCache* const model_cache =
      ml_drift::cache::schema::GetModelCache(handle.data() +
                                             header.buffer_list_offset);
  ASSERT_NE(model_cache, nullptr);
  ASSERT_NE(model_cache->subgraphs(), nullptr);
  ASSERT_EQ(model_cache->subgraphs()->size(), 1);
  auto subgraph = model_cache->subgraphs()->Get(0);
  ASSERT_EQ(subgraph->unique_model_identifier(), unique_model_identifier);
  auto buffers = subgraph->buffers();
  ASSERT_NE(buffers, nullptr);
  ASSERT_EQ(buffers->size(), 1);
  auto buffer_fb = buffers->Get(0);
  ASSERT_EQ(buffer_fb->global_tensor_id(), global_tensor_id);
  ASSERT_EQ(buffer_fb->is_quantization_param_tensor(), false);
  ASSERT_EQ(buffer_fb->tensor_descriptor()->data_type(),
            ToFB(tensor_desc.GetDataType()));
  ASSERT_EQ(buffer_fb->tensor_descriptor()->storage_type(),
            ToFB(tensor_desc.GetStorageType()));
  ASSERT_EQ(buffer_fb->tensor_descriptor()->layout(),
            ToFB(tensor_desc.GetLayout()));
  ASSERT_EQ(buffer_fb->size(), size(payload));

  flatbuffers::Verifier verifier(handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  EXPECT_TRUE(ml_drift::cache::schema::VerifyModelCacheBuffer(verifier));

  ASSERT_LE(subgraph->base_offset() + buffer_fb->offset(), size(handle));
  ASSERT_LE(subgraph->base_offset() + buffer_fb->offset() + buffer_fb->size(),
            size(handle));
  std::tuple<const char*, size_t> cache_data(
      reinterpret_cast<const char*>(handle.data() + subgraph->base_offset() +
                                    buffer_fb->offset()),
      buffer_fb->size());
  EXPECT_THAT(cache_data, ElementsAreArray(payload));
}

TEST(CacheBuilderTest, AppendWorksWithGlobalIdCollision) {
  using std::size;

  const std::string payload = "This is some data in the file.";
  const uint64_t global_tensor_id = 12345;
  const uint64_t unique_model_identifier = 9876;

  ml_drift::CacheBuilder builder;
  const std::string cache_path = testing::TempDir() + "/cache";
  ASSERT_OK(builder.Start(cache_path, unique_model_identifier));
  ASSERT_OK(builder.StartBuildStep(unique_model_identifier));

  const size_t payload_size = size(payload);
  void* buffer = builder.Reserve(payload_size);
  std::memcpy(buffer, payload.c_str(), payload_size);
  ml_drift::TensorDescriptor tensor_desc(
      ml_drift::DataType::FLOAT32, ml_drift::TensorStorageType::IMAGE_BUFFER,
      ml_drift::Layout::LINEAR);
  ml_drift::BufferLocation loc;
  // Add both the quantization and non-quantization tensor with the same global
  // tensor id.
  ASSERT_OK(
      builder.Append(global_tensor_id,
                     /*is_quantization_param_tensor=*/false,
                     /*packing_algorithm=*/
                     ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                     tensor_desc, buffer, payload_size, loc));
  ASSERT_OK(
      builder.Append(global_tensor_id,
                     /*is_quantization_param_tensor=*/true,
                     /*packing_algorithm=*/
                     ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                     tensor_desc, buffer, payload_size, loc));

  EXPECT_EQ(loc.size, payload_size);
  EXPECT_GE(builder.capacity(), payload_size);

  ASSERT_OK(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_OK(handle.Map(cache_path.c_str()));

  const ml_drift::MLDriftCacheHeader& header =
      *reinterpret_cast<const ml_drift::MLDriftCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, ml_drift::MLDriftCacheHeader::kVersion);
  ASSERT_NE(header.buffer_list_offset, 0);
  ASSERT_NE(header.buffer_list_size, 0);
  ASSERT_LE(header.buffer_list_offset + header.buffer_list_size, handle.size());

  const ml_drift::cache::schema::ModelCache* const model_cache =
      ml_drift::cache::schema::GetModelCache(handle.data() +
                                             header.buffer_list_offset);

  ASSERT_NE(model_cache, nullptr);
  ASSERT_NE(model_cache->subgraphs(), nullptr);
  ASSERT_EQ(model_cache->subgraphs()->size(), 1);
  auto subgraph = model_cache->subgraphs()->Get(0);
  ASSERT_EQ(subgraph->unique_model_identifier(), unique_model_identifier);
  auto buffers = subgraph->buffers();
  ASSERT_NE(buffers, nullptr);
  ASSERT_EQ(buffers->size(), 2);
  ASSERT_NE(buffers->Get(0), nullptr);

  for (int i = 0; i < buffers->size(); ++i) {
    auto buffer_fb = buffers->Get(i);
    ASSERT_EQ(buffer_fb->global_tensor_id(), global_tensor_id);
    ASSERT_EQ(buffer_fb->is_quantization_param_tensor(), i == 1);
    ASSERT_EQ(buffer_fb->tensor_descriptor()->data_type(),
              ToFB(tensor_desc.GetDataType()));
    ASSERT_EQ(buffer_fb->tensor_descriptor()->storage_type(),
              ToFB(tensor_desc.GetStorageType()));
    ASSERT_EQ(buffer_fb->tensor_descriptor()->layout(),
              ToFB(tensor_desc.GetLayout()));
    ASSERT_EQ(buffer_fb->size(), size(payload));
  }

  flatbuffers::Verifier verifier(handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  EXPECT_TRUE(ml_drift::cache::schema::VerifyModelCacheBuffer(verifier));
}

TEST(CacheBuilderTest, NonExistingPathFails) {
  ml_drift::CacheBuilder builder;
  EXPECT_THAT(
      builder.Start("", 12345),
      testing::status::StatusIs(::util::error::INTERNAL,
                                testing::HasSubstr("Could not open file")));
  // Try a path that shouldn't exist in the test environment.
  std::string nonexistent_path = "/fake_directory/fake_file";
  std::remove(nonexistent_path.c_str());
  EXPECT_THAT(
      builder.Start(nonexistent_path, 12345),
      testing::status::StatusIs(::util::error::INTERNAL,
                                testing::HasSubstr("Could not open file")));
}

TEST(CacheBuilderTest, InMemoryCacheTriggeredByCorrectPrefix) {
  if (!ml_drift::InMemoryFileDescriptorAvailable()) {
    GTEST_SKIP() << "In-memory weight cache isn't enabled for this build or "
                    "isn't supported by the current system, skipping test.";
  }
  {  // Exact in-memory flag used starts an in-memory build.
    ml_drift::CacheBuilder builder;
    EXPECT_OK(builder.Start(ml_drift::kInMemoryCachePath, 12345));
    ASSERT_OK(builder.StartBuildStep(12345));
    EXPECT_TRUE(builder.IsStarted());
    const ml_drift::FileDescriptor file_fd(
        open(ml_drift::kInMemoryCachePath, O_RDONLY));  // NOLINT: b/332641196
    EXPECT_FALSE(file_fd.IsValid());
    EXPECT_EQ(errno, ENOENT);
  }
  {  // Prefixed in-memory flag used starts an in-memory build.
    ml_drift::CacheBuilder builder;
    // Choose a path that is prefixed with the in-memory cache path but is
    // a random subdirectory that shouldn't already exist.
    const std::string path_with_in_memory_prefix =
        std::string(ml_drift::kInMemoryCachePath) + "/my_file";
    std::remove(path_with_in_memory_prefix.c_str());
    EXPECT_OK(builder.Start(path_with_in_memory_prefix, 12345));
    ASSERT_OK(builder.StartBuildStep(12345));
    EXPECT_TRUE(builder.IsStarted());
    const ml_drift::FileDescriptor file_fd(
        open(ml_drift::kInMemoryCachePath, O_RDONLY));
    EXPECT_FALSE(file_fd.IsValid());
    EXPECT_EQ(errno, ENOENT);
  }
}

TEST(CacheBuilderTest, MultipleStepBuild) {
  using std::size;

  const std::string payload1 = "This is some data in the file.";
  const uint32_t dummy_id1 = 12345;
  const std::string payload2 = "Other data in the file.";
  const uint32_t dummy_id2 = 23456;
  const std::string payload3 = "More data in the file.";
  const uint32_t dummy_id3 = 34567;

  std::string tmp_file_path =
      absl::StrCat(::testing::TempDir(), "/weight_cache_test_file.XXXXXX");
  TempFileDesc tmp_file{tmp_file_path, TempFileDesc::kAutoClose};

  ml_drift::FileDescriptor file_descriptor = ml_drift::FileDescriptor::Open(
      tmp_file.GetCPath(),
      O_CREAT | O_TRUNC | O_RDWR,  // NOLINT: b/332641196
      0644);
  ml_drift::CacheBuilder builder;
  ASSERT_OK(builder.Start(tmp_file.GetCPath(), 12345));
  ASSERT_OK(builder.StartBuildStep(12345));

  {
    const size_t payload_size = size(payload1);
    void* buffer = builder.Reserve(payload_size);
    std::memcpy(buffer, payload1.c_str(), payload_size);
    ml_drift::TensorDescriptor tensor_desc(
        ml_drift::DataType::FLOAT32, ml_drift::TensorStorageType::IMAGE_BUFFER,
        ml_drift::Layout::LINEAR);
    ml_drift::BufferLocation loc;
    ASSERT_OK(
        builder.Append(dummy_id1, false, /*packing_algorithm=*/
                       ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                       tensor_desc, buffer, payload_size, loc));
    EXPECT_EQ(loc.size, payload_size);
    EXPECT_GE(builder.capacity(), payload_size);
  }
  {
    const size_t payload_size = size(payload3);
    void* buffer = builder.Reserve(payload_size);
    std::memcpy(buffer, payload3.c_str(), payload_size);
    ml_drift::TensorDescriptor tensor_desc(
        ml_drift::DataType::FLOAT32, ml_drift::TensorStorageType::IMAGE_BUFFER,
        ml_drift::Layout::LINEAR);
    ml_drift::BufferLocation loc;
    ASSERT_OK(
        builder.Append(dummy_id3, false, /*packing_algorithm=*/
                       ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                       tensor_desc, buffer, payload_size, loc));
  }

  ASSERT_OK(builder.StopBuildStep());

  MMapHandle handle;
  ASSERT_OK(handle.Map(tmp_file.GetCPath()));

  ASSERT_OK(builder.StartBuildStep(12345));
  {
    const size_t payload_size = size(payload2);
    void* buffer = builder.Reserve(payload_size);
    std::memcpy(buffer, payload2.c_str(), payload_size);
    ml_drift::TensorDescriptor tensor_desc(
        ml_drift::DataType::FLOAT32, ml_drift::TensorStorageType::IMAGE_BUFFER,
        ml_drift::Layout::LINEAR);
    ml_drift::BufferLocation loc;
    ASSERT_OK(
        builder.Append(dummy_id2, false, /*packing_algorithm=*/
                       ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                       tensor_desc, buffer, payload_size, loc));
    EXPECT_EQ(loc.size, payload_size);
    EXPECT_GE(builder.capacity(), payload_size);
  }

  ASSERT_OK(builder.StopBuildStep());

  ASSERT_OK(handle.Map(tmp_file.GetCPath()));

  const ml_drift::MLDriftCacheHeader& header =
      *reinterpret_cast<const ml_drift::MLDriftCacheHeader*>(handle.data());

  ASSERT_EQ(header.version, ml_drift::MLDriftCacheHeader::kVersion);
  ASSERT_NE(header.buffer_list_offset, 0);
  ASSERT_NE(header.buffer_list_size, 0);
  ASSERT_LE(header.buffer_list_offset + header.buffer_list_size, handle.size());

  const ml_drift::cache::schema::ModelCache* const model_cache =
      ml_drift::cache::schema::GetModelCache(handle.data() +
                                             header.buffer_list_offset);

  ASSERT_NE(model_cache, nullptr);
  ASSERT_NE(model_cache->subgraphs(), nullptr);
  ASSERT_EQ(model_cache->subgraphs()->size(), 1);
  auto subgraph = model_cache->subgraphs()->Get(0);
  ASSERT_EQ(subgraph->unique_model_identifier(), 12345);
  auto buffers = subgraph->buffers();
  ASSERT_NE(buffers, nullptr);
  ASSERT_EQ(buffers->size(), 3);
  // Payload 1.
  const auto* buffer1 = buffers->Get(0);
  ASSERT_NE(buffer1, nullptr);
  ASSERT_EQ(buffer1->size(), size(payload1));
  ASSERT_EQ(buffer1->global_tensor_id(), dummy_id1);
  ASSERT_EQ(buffer1->is_quantization_param_tensor(), false);

  // Payload 3.
  const auto* buffer3 = buffers->Get(1);
  ASSERT_NE(buffer3, nullptr);
  ASSERT_EQ(buffer3->size(), size(payload3));
  ASSERT_EQ(buffer3->global_tensor_id(), dummy_id3);
  ASSERT_EQ(buffer3->is_quantization_param_tensor(), false);

  // Payload 2.
  const auto* buffer2 = buffers->Get(2);
  ASSERT_NE(buffer2, nullptr);
  ASSERT_EQ(buffer2->size(), size(payload2));
  ASSERT_EQ(buffer2->global_tensor_id(), dummy_id2);
  ASSERT_EQ(buffer2->is_quantization_param_tensor(), false);

  flatbuffers::Verifier verifier(handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  EXPECT_TRUE(ml_drift::cache::schema::VerifyModelCacheBuffer(verifier));

  // Payload 1.
  ASSERT_LE(subgraph->base_offset() + buffer1->offset(), size(handle));
  ASSERT_LE(subgraph->base_offset() + buffer1->offset() + buffer1->size(),
            size(handle));

  // Payload 2.
  ASSERT_LE(subgraph->base_offset() + buffer2->offset(), size(handle));
  ASSERT_LE(subgraph->base_offset() + buffer2->offset() + buffer2->size(),
            size(handle));

  // Payload 3.
  ASSERT_LE(subgraph->base_offset() + buffer3->offset(), size(handle));
  ASSERT_LE(subgraph->base_offset() + buffer3->offset() + buffer3->size(),
            size(handle));

  auto GetBufferData = [&handle, &subgraph](const auto* buffer) {
    return std::tuple<const char*, size_t>(
        reinterpret_cast<const char*>(handle.data() + subgraph->base_offset() +
                                      buffer->offset()),
        buffer->size());
  };

  EXPECT_THAT(GetBufferData(buffer1), ElementsAreArray(payload1));
  EXPECT_THAT(GetBufferData(buffer2), ElementsAreArray(payload2));
  EXPECT_THAT(GetBufferData(buffer3), ElementsAreArray(payload3));
}

TEST(CacheBuilderTest, FlatBufferDoesNotGrowUnnecessarilyAcrossSteps) {
  using std::size;

  std::string tmp_file_path =
      absl::StrCat(::testing::TempDir(), "/weight_cache_test_file.XXXXXX");
  TempFileDesc tmp_file{tmp_file_path, TempFileDesc::kAutoClose};

  ml_drift::FileDescriptor file_descriptor = ml_drift::FileDescriptor::Open(
      tmp_file.GetCPath(),
      O_CREAT | O_TRUNC | O_RDWR,  // NOLINT: b/332641196
      0644);
  ml_drift::CacheBuilder builder;
  ASSERT_OK(builder.Start(tmp_file.GetCPath(), 12345));

  uint64_t previous_size = 0;
  // Do 5 steps to see if the flatbuffer size grows quadratically.
  for (int i = 0; i < 5; ++i) {
    ASSERT_OK(builder.StartBuildStep(12345));
    const std::string payload = "Some data";
    const size_t payload_size = size(payload);
    void* buffer = builder.Reserve(payload_size);
    std::memcpy(buffer, payload.c_str(), payload_size);
    ml_drift::TensorDescriptor tensor_desc(
        ml_drift::DataType::FLOAT32, ml_drift::TensorStorageType::IMAGE_BUFFER,
        ml_drift::Layout::LINEAR);
    ml_drift::BufferLocation loc;
    ASSERT_OK(
        builder.Append(i, false, /*packing_algorithm=*/
                       ml_drift::cache::schema::PackingAlgorithm_LAYOUT_UNKNOWN,
                       tensor_desc, buffer, payload_size, loc));
    ASSERT_OK(builder.StopBuildStep());

    MMapHandle handle;
    ASSERT_OK(handle.Map(tmp_file.GetCPath()));
    const ml_drift::MLDriftCacheHeader& header =
        *reinterpret_cast<const ml_drift::MLDriftCacheHeader*>(handle.data());

    // The size of the flatbuffer should grow linearly with each appended
    // tensor. An individual tensor entry shouldn't be more than ~500 bytes.
    if (i > 0) {
      uint64_t growth = header.buffer_list_size - previous_size;
      EXPECT_LT(growth, 500)
          << "Flatbuffer grew too much between steps! Growth: " << growth
          << " bytes. Previous size: " << previous_size
          << " Current size: " << header.buffer_list_size;
    }
    previous_size = header.buffer_list_size;
  }
}

}  // namespace
}  // namespace mldrift
