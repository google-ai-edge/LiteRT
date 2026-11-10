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

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_TEST_BACKENDS_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_TEST_BACKENDS_H_

#include <string>

#include <gtest/gtest.h>
#include "tensor/backends/xnnpack/arithmetic.h"
#include "tensor/backends/xnnpack/conversion.h"
#include "tensor/runners/xnnpack/runner.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::gemma4 {

// Pairs a lowering mixin tag with the runner that executes it, so that a test
// can be written once and instantiated for every backend.
//
// Usage:
//
//   template <class Backend>
//   class MyTest : public ::testing::Test {};
//   TYPED_TEST_SUITE(MyTest, TestBackends, TestBackendNames);
//
//   TYPED_TEST(MyTest, DoesSomething) {
//     using Tensor = typename TypeParam::Tensor;
//     using Runner = typename TypeParam::Runner;
//     ...
//   }
struct XnnpackBackend {
  using Tag = XnnpackMixinTag;
  using Runner = XnnpackRunner;
  using Tensor = ::litert::tensor::Tensor<XnnpackMixinTag>;
  using Operation = XnnpackOperation;

  static constexpr char kName[] = "Xnnpack";
};

using TestBackends = ::testing::Types<XnnpackBackend>;

// Gives the instantiated tests readable names ("…/Xnnpack" rather than "…/0").
struct TestBackendNames {
  template <class Backend>
  static std::string GetName(int /*index*/) {
    return Backend::kName;
  }
};

}  // namespace litert::tensor::examples::gemma4

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA4_TEST_BACKENDS_H_
