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
#include "litert/compiler/mlir/dialects/litert/lazy_resource_blob.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "perftools/profiles/collector/heap/alloc_recorder.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/random.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/AsmState.h"
#include "third_party/tcmalloc/malloc_extension.h"

namespace litert {
namespace {

namespace heap_profile = ::perftools::profiles::collector::heap;

size_t GetCurrentAllocatedBytes() {
  return tcmalloc::MallocExtension::GetNumericProperty(
             "generic.current_allocated_bytes")
      .value();
}

// Prevents compiler from pre-allocating memory.
template <typename T>
std::unique_ptr<std::vector<T>> GetVectorOfRandomSize() {
  auto buffer(std::make_unique<std::vector<T>>());
  absl::BitGen bitgen;
  uint32_t res_size = absl::Uniform(absl::IntervalClosed, bitgen, 1000, 1200);
  buffer->reserve(res_size);
  buffer->resize(res_size);
  return buffer;
}

TEST(LazyResourceBlobTest, TestCreateAndCopyData_Char) {
  std::cout << "Test start allocated bytes: " << GetCurrentAllocatedBytes()
            << std::endl;
  {
    auto buffer = GetVectorOfRandomSize<uint8_t>();
    std::cout << "Post buffer creation allocated bytes: "
              << GetCurrentAllocatedBytes() << std::endl;
    auto lazy_blob =
        LazyResourceBlob::CreateAndCopyData(*buffer, alignof(uint8_t));
    delete buffer.release();
    std::cout << "Post LazyResourceBlob creation allocated bytes: "
              << GetCurrentAllocatedBytes() << std::endl;
    lazy_blob.Cleanup();
  }
  std::cout << "Test end allocated bytes: " << GetCurrentAllocatedBytes()
            << std::endl;
}

TEST(LazyResourceBlobTest, TestCreateAndCopyDataArrayRef) {
  auto buffer = GetVectorOfRandomSize<int>();
  auto lazy_blob =
      LazyResourceBlob::CreateAndCopyData(llvm::ArrayRef<int>(*buffer));
  lazy_blob.Cleanup();
}

TEST(LazyResourceBlobTest, TestGetData) {
  ASSERT_TRUE(heap_profile::AllocRecorderStartWithMmapTracking(
      "/tmp/resource_blob_test_test_get_data_heap"));
  auto buffer = GetVectorOfRandomSize<uint8_t>();
  for (int i = 0; i < buffer->size(); i++) {
    (*buffer)[i] = 'z';
  }
  heap_profile::AllocRecorderDump("Before blob creation.");
  auto lazy_blob =
      LazyResourceBlob::CreateAndCopyData(*buffer, alignof(uint8_t));
  delete buffer.release();
  heap_profile::AllocRecorderDump("After blob creation.");

  auto data_handle = lazy_blob.GetDataHandle();
  auto data = data_handle.GetRawData();

  EXPECT_EQ(data[0], 'z');
  heap_profile::AllocRecorderDump("After initial data read.");
  for (int i = 4096; i < data.size(); i++) {
    EXPECT_EQ(data[i], 'z');
  }
  heap_profile::AllocRecorderDump("After all data read.");

  lazy_blob.Cleanup();

  heap_profile::AllocRecorderStop();
}

TEST(LazyResourceBlobTest, TestCreateAndCopyDataAndGetDataAs_Int) {
  auto buffer = GetVectorOfRandomSize<int>();
  auto buffer_size = buffer->size();
  for (int i = 0; i < buffer->size(); i++) {
    (*buffer)[i] = 7;
  }
  auto lazy_blob = LazyResourceBlob::CreateAndCopyData<int>(*buffer);

  auto data_handle = lazy_blob.GetDataHandle();
  auto data = data_handle.GetDataAs<int>();
  EXPECT_EQ(buffer_size, data.size());

  for (int i = 0; i < data.size(); i++) {
    EXPECT_EQ(data[i], 7);
  }
  lazy_blob.Cleanup();
}

TEST(LazyResourceBlobTest, TestMove) {
  auto buffer = GetVectorOfRandomSize<int>();
  auto buffer_size = buffer->size();
  for (int i = 0; i < buffer->size(); i++) {
    (*buffer)[i] = 7;
  }
  auto lazy_blob = LazyResourceBlob::CreateAndCopyData<int>(*buffer);

  auto new_lazy_blob = std::move(lazy_blob);

  auto data_handle = new_lazy_blob.GetDataHandle();
  auto data = data_handle.GetDataAs<int>();
  EXPECT_EQ(buffer_size, data.size());

  for (int i = 0; i < data.size(); i++) {
    EXPECT_EQ(data[i], 7);
  }
  new_lazy_blob.Cleanup();
}

TEST(LazyResourceBlobTest, TestGetChunkReader) {
  auto data = GetVectorOfRandomSize<int>();
  for (int i = 0; i < data->size(); i++) {
    (*data)[i] = i;
  }
  auto lazy_blob = LazyResourceBlob::CreateAndCopyData<int>(*data);

  {
    auto reader = lazy_blob.GetChunkReader(/*chunk_size=*/13);
    int64_t offset = 0;
    for (absl::string_view chunk : reader) {
      absl::string_view buffer(reinterpret_cast<char*>(data->data()) + offset,
                               chunk.size());
      for (int i = 0; i < chunk.size(); i++) {
        EXPECT_EQ(chunk[i], buffer[i]);
      }
      offset += chunk.size();
    }
    EXPECT_EQ(offset, data->size() * sizeof(int) / sizeof(char));
  }

  // Execute the read a second time to ensure the first read didn't accidentally
  // close the FD and erase the underlying data.
  {
    auto reader = lazy_blob.GetChunkReader(/*chunk_size=*/13);
    int64_t offset = 0;
    for (absl::string_view chunk : reader) {
      absl::string_view buffer(reinterpret_cast<char*>(data->data()) + offset,
                               chunk.size());
      for (int i = 0; i < chunk.size(); i++) {
        EXPECT_EQ(chunk[i], buffer[i]);
      }
      offset += chunk.size();
    }
    EXPECT_EQ(offset, data->size() * sizeof(int) / sizeof(char));
  }

  lazy_blob.Cleanup();
}

}  // namespace
}  // namespace litert
