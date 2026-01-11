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
