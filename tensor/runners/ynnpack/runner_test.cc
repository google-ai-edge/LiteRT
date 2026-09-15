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

#include "tensor/runners/ynnpack/runner.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensor/backends/ynnpack/arithmetic.h"
#include "tensor/runners/common_nnpack/runner_test_suite.h"

namespace litert::tensor {

struct YnnpackTestTraits {
  using Tag = YnnpackMixinTag;
  using Runner = YnnpackRunner;

  static constexpr uint32_t kFlagExternalInput = YNN_VALUE_FLAG_EXTERNAL_INPUT;
  static constexpr uint32_t kFlagExternalOutput =
      YNN_VALUE_FLAG_EXTERNAL_OUTPUT;

  // YNNPACK has no convolution, pooling or resize primitives. These would have
  // to be composed out of stencil copies and dots, which isn't implemented.
  static constexpr bool kSupportsConv2D = false;
  static constexpr bool kSupportsDepthwiseConv2D = false;
  static constexpr bool kSupportsTransposeConv2D = false;
  static constexpr bool kSupportsResize = false;
};

INSTANTIATE_TYPED_TEST_SUITE_P(Ynnpack, NnpackRunnerTest, YnnpackTestTraits);

}  // namespace litert::tensor
