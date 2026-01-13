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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_TEST_FIXTURES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_TEST_FIXTURES_H_

#include <cstdint>

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/vendors/c/litert_dispatch.h"

// A collection of fixtures to aid in testing the Google Tensor Dispatch API
// implementation.

namespace litert::google_tensor::testing {

// A test fixture for Dispatch API tests that both owns and exposes a default
// LiteRT environment and fresh Dispatch device context.
class DispatchApiTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  LiteRtEnvironment env() { return env_; }
  LiteRtDispatchDeviceContext device_context() { return device_context_; }

 private:
  LiteRtEnvironment env_;
  LiteRtDispatchDeviceContext device_context_;
};

// A test fixture for Dispatch API tests that use the simple model.
class SimpleModelTest : public DispatchApiTest {
 protected:
  void SetUp() override;

  const LiteRtMemBuffer& model_bytecode() const { return model_bytecode_; }

 private:
  litert::OwningBufferRef<uint8_t> model_;
  LiteRtMemBuffer model_bytecode_;
};

}  // namespace litert::google_tensor::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_TEST_FIXTURES_H_
