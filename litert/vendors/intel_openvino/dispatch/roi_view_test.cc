// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/intel_openvino/dispatch/roi_view.h"

#include <gtest/gtest.h>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"

// These tests fabricate plain host ov::Tensors (self-allocating, no device) and
// exercise MakePortView. The extent-overflow guard inside MakePortView is
// deliberately not covered here: a well-formed ov::Tensor always reports a
// byte size consistent with its shape and strides, so a zero-offset ROI can
// never exceed it -- the guard only fires for an externally under-allocated
// buffer, which the real strided device path exercises in roi_view_manual_test.

namespace litert::openvino {
namespace {

TEST(MakePortViewTest, EqualShapeReturnsSameBuffer) {
  ov::Tensor buffer(ov::element::f16, ov::Shape{1, 4, 8, 16});
  auto result = MakePortView(buffer, ov::Shape{1, 4, 8, 16});
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->get_shape(), (ov::Shape{1, 4, 8, 16}));
  // Passthrough: same underlying allocation.
  EXPECT_EQ(result->data(), buffer.data());
}

TEST(MakePortViewTest, LargerBufferReturnsZeroCopySubView) {
  ov::Tensor buffer(ov::element::f16, ov::Shape{1, 4, 8, 16});
  const ov::Shape port_shape{1, 4, 6, 16};  // trims the seq dim 8 -> 6
  auto result = MakePortView(buffer, port_shape);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->get_shape(), port_shape);
  // ROI starts at offset 0 on every dim: shares the parent's memory...
  EXPECT_EQ(result->data(), buffer.data());
  // ...and inherits the parent's (byte) strides for strided access.
  EXPECT_EQ(result->get_strides(), buffer.get_strides());
}

TEST(MakePortViewTest, RankMismatchIsError) {
  ov::Tensor buffer(ov::element::f16, ov::Shape{1, 4, 8, 16});
  auto result = MakePortView(buffer, ov::Shape{4, 8, 16});
  EXPECT_FALSE(static_cast<bool>(result));
}

TEST(MakePortViewTest, BufferSmallerThanPortIsError) {
  ov::Tensor buffer(ov::element::f16, ov::Shape{1, 4, 6, 16});
  auto result = MakePortView(buffer, ov::Shape{1, 4, 8, 16});
  EXPECT_FALSE(static_cast<bool>(result));
}

}  // namespace
}  // namespace litert::openvino
