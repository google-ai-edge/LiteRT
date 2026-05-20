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

#include "tensor/runners/litert/litert_buffer.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/litert_model_types.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"

namespace litert::tensor {
namespace {

TEST(LitertBufferTest, LockAndWrite) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::move(*env_or);

  std::vector<int32_t> shape = {1, 4};
  LiteRtRankedTensorType c_type;
  c_type.element_type = kLiteRtElementTypeFloat32;
  c_type.layout.rank = 2;
  c_type.layout.dimensions[0] = 1;
  c_type.layout.dimensions[1] = 4;
  RankedTensorType tensor_type(c_type);

  auto tb_or = TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                           tensor_type, 4 * sizeof(float));
  ASSERT_TRUE(tb_or.HasValue());
  auto tb = std::move(*tb_or);

  LitertBuffer buffer(std::move(tb));

  {
    auto locked_span = buffer.LockMutable();
    auto* data = reinterpret_cast<float*>(locked_span.data());
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
  }  // Unlocks!

  {
    auto locked_span = buffer.Lock();
    const auto* data = reinterpret_cast<const float*>(locked_span.data());
    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[1], 2.0f);
    EXPECT_EQ(data[2], 3.0f);
    EXPECT_EQ(data[3], 4.0f);
  }
}

}  // namespace
}  // namespace litert::tensor
