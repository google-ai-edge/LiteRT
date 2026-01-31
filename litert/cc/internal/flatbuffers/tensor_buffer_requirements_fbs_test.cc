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

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/internal/flatbuffers/tensor_buffer_requirements_generated.h"

namespace litert::internal::flatbuffers {
namespace {

TEST(TensorBufferRequirementsFbsTest, Basic) {
  ::flatbuffers::FlatBufferBuilder fbb;

  std::vector<int32_t> supported_types = {1, 2};
  auto supported_types_vec = fbb.CreateVector(supported_types);

  std::vector<uint32_t> strides = {4, 8, 16};
  auto strides_vec = fbb.CreateVector(strides);

  TensorBufferRequirementsFbsBuilder builder(fbb);
  builder.add_supported_buffer_types(supported_types_vec);
  builder.add_buffer_size(1024);
  builder.add_strides(strides_vec);
  builder.add_alignment(64);
  auto root = builder.Finish();
  fbb.Finish(root);

  auto* requirements = GetTensorBufferRequirementsFbs(fbb.GetBufferPointer());

  EXPECT_EQ(requirements->buffer_size(), 1024);
  EXPECT_EQ(requirements->alignment(), 64);
  ASSERT_EQ(requirements->supported_buffer_types()->size(), 2);
  EXPECT_EQ(requirements->supported_buffer_types()->Get(0), 1);
  EXPECT_EQ(requirements->supported_buffer_types()->Get(1), 2);
  ASSERT_EQ(requirements->strides()->size(), 3);
  EXPECT_EQ(requirements->strides()->Get(0), 4);
  EXPECT_EQ(requirements->strides()->Get(1), 8);
  EXPECT_EQ(requirements->strides()->Get(2), 16);
}

}  // namespace
}  // namespace litert::internal::flatbuffers
