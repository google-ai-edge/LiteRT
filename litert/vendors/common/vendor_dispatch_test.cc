// Copyright 2025 Google LLC.
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

#include "litert/vendors/common/vendor_dispatch_base.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/common/vendor_traits.h"

namespace litert {
namespace vendors {
namespace {

using ::testing::Eq;
using ::testing::NotNull;
using ::testing::StartsWith;

// Mock vendor for testing
struct MockVendorTag {};

// Mock device context
class MockDeviceContext : public VendorDeviceContext {
 public:
  explicit MockDeviceContext(const LiteRtDispatchDeviceContext& device_context)
      : VendorDeviceContext(device_context) {}
  
  void* GetBackendContext() override { return this; }
};

// Mock invocation context
class MockInvocationContext : public VendorInvocationContext {
 public:
  LiteRtStatus AttachInput(int graph_input_idx,
                          LiteRtTensorBufferHandle handle) override {
    attached_inputs_[graph_input_idx] = handle;
    return kLiteRtStatusOk;
  }
  
  LiteRtStatus AttachOutput(int graph_output_idx,
                           LiteRtTensorBufferHandle handle) override {
    attached_outputs_[graph_output_idx] = handle;
    return kLiteRtStatusOk;
  }
  
  LiteRtStatus DetachInput(int graph_input_idx) override {
    attached_inputs_.erase(graph_input_idx);
    return kLiteRtStatusOk;
  }
  
  LiteRtStatus DetachOutput(int graph_output_idx) override {
    attached_outputs_.erase(graph_output_idx);
    return kLiteRtStatusOk;
  }
  
  LiteRtStatus Invoke() override {
    invoke_count_++;
    return kLiteRtStatusOk;
  }
  
  void* GetBackendContext() override { return this; }
  
  int GetInvokeCount() const { return invoke_count_; }
  bool HasInput(int idx) const { 
    return attached_inputs_.find(idx) != attached_inputs_.end();
  }
  bool HasOutput(int idx) const {
    return attached_outputs_.find(idx) != attached_outputs_.end();
  }
  
 private:
  std::unordered_map<int, LiteRtTensorBufferHandle> attached_inputs_;
  std::unordered_map<int, LiteRtTensorBufferHandle> attached_outputs_;
  int invoke_count_ = 0;
};

// Mock vendor traits
template <>
struct VendorTraits<MockVendorTag> {
  static constexpr const char* kVendorId = "MockVendor";
  static constexpr uint32_t kCapabilities = kLiteRtDispatchCapabilitiesBasic;
  static constexpr bool kSupportsAsync = false;
  static constexpr bool kSupportsGraph = false;
  static constexpr const char* kBackendLibraryName = "libmock.so";
  
  using BackendContext = void;
  using BackendBuffer = void;
  using BackendModel = void;
  
  static bool initialized;
  static std::string lib_dir_used;
  
  static LiteRtStatus Initialize(const std::string& lib_dir) {
    initialized = true;
    lib_dir_used = lib_dir;
    return kLiteRtStatusOk;
  }
  
  static std::string GetBuildId() {
    return "MockVendor v1.0.0";
  }
  
  static Expected<std::unique_ptr<VendorDeviceContext>> CreateDeviceContext(
      const LiteRtDispatchDeviceContext* device_context) {
    auto context = std::make_unique<MockDeviceContext>(*device_context);
    return std::unique_ptr<VendorDeviceContext>(std::move(context));
  }
  
  static LiteRtStatus RegisterTensorBuffer(
      VendorDeviceContext* context,
      LiteRtTensorBuffer buffer,
      LiteRtTensorBufferHandle* handle) {
    static LiteRtTensorBufferHandle next_handle = 1;
    *handle = next_handle++;
    return kLiteRtStatusOk;
  }
  
  static LiteRtStatus UnregisterTensorBuffer(
      VendorDeviceContext* context,
      LiteRtTensorBufferHandle handle) {
    return kLiteRtStatusOk;
  }
  
  static Expected<std::unique_ptr<VendorInvocationContext>> CreateInvocationContext(
      VendorDeviceContext* device_context,
      const void* exec_bytecode_ptr,
      size_t exec_bytecode_size,
      const char* function_name) {
    auto context = std::make_unique<MockInvocationContext>();
    return std::unique_ptr<VendorInvocationContext>(std::move(context));
  }
};

// Static member definitions
bool VendorTraits<MockVendorTag>::initialized = false;
std::string VendorTraits<MockVendorTag>::lib_dir_used;

// Test fixture
class VendorDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset mock state
    VendorTraits<MockVendorTag>::initialized = false;
    VendorTraits<MockVendorTag>::lib_dir_used.clear();
    
    // Create dispatch API
    dispatch_api_ = VendorDispatch<MockVendorTag>::GetApi();
  }
  
  LiteRtDispatchApi dispatch_api_;
};

TEST_F(VendorDispatchTest, GetVendorId) {
  const char* vendor_id = nullptr;
  EXPECT_EQ(dispatch_api_.interface->get_vendor_id(&vendor_id), kLiteRtStatusOk);
  EXPECT_THAT(vendor_id, NotNull());
  EXPECT_STREQ(vendor_id, "MockVendor");
}

TEST_F(VendorDispatchTest, GetBuildId) {
  const char* build_id = nullptr;
  EXPECT_EQ(dispatch_api_.interface->get_build_id(&build_id), kLiteRtStatusOk);
  EXPECT_THAT(build_id, NotNull());
  EXPECT_THAT(build_id, StartsWith("MockVendor"));
}

TEST_F(VendorDispatchTest, GetCapabilities) {
  LiteRtDispatchCapabilities capabilities;
  EXPECT_EQ(dispatch_api_.interface->get_capabilities(&capabilities), kLiteRtStatusOk);
  EXPECT_EQ(capabilities, kLiteRtDispatchCapabilitiesBasic);
}

TEST_F(VendorDispatchTest, Initialize) {
  // Create options with library directory
  LiteRtEnvironmentOptions env_options;
  ASSERT_EQ(LiteRtCreateEnvironmentOptions(&env_options), kLiteRtStatusOk);
  
  const char* lib_dir = "/path/to/libs";
  ASSERT_EQ(LiteRtSetEnvironmentOption(env_options, 
                                       kLiteRtEnvOptionTagDispatchLibraryDir,
                                       &lib_dir), kLiteRtStatusOk);
  
  EXPECT_FALSE(VendorTraits<MockVendorTag>::initialized);
  EXPECT_EQ(dispatch_api_.interface->initialize(env_options), kLiteRtStatusOk);
  EXPECT_TRUE(VendorTraits<MockVendorTag>::initialized);
  EXPECT_EQ(VendorTraits<MockVendorTag>::lib_dir_used, lib_dir);
  
  LiteRtDestroyEnvironmentOptions(env_options);
}

TEST_F(VendorDispatchTest, DeviceContextLifecycle) {
  LiteRtDispatchDeviceContext device_context{};
  LiteRtDispatchDeviceContextHandle handle = nullptr;
  
  // Create device context
  EXPECT_EQ(dispatch_api_.interface->device_context_create(device_context, &handle),
            kLiteRtStatusOk);
  EXPECT_THAT(handle, NotNull());
  
  // Destroy device context
  EXPECT_EQ(dispatch_api_.interface->device_context_destroy(handle), kLiteRtStatusOk);
}

TEST_F(VendorDispatchTest, TensorBufferRegistration) {
  // Create device context
  LiteRtDispatchDeviceContext device_context{};
  LiteRtDispatchDeviceContextHandle device_handle = nullptr;
  ASSERT_EQ(dispatch_api_.interface->device_context_create(device_context, &device_handle),
            kLiteRtStatusOk);
  
  // Register tensor buffer
  LiteRtTensorBuffer tensor_buffer = reinterpret_cast<LiteRtTensorBuffer>(0x1234);
  LiteRtTensorBufferHandle buffer_handle = 0;
  EXPECT_EQ(dispatch_api_.interface->register_tensor_buffer(
      device_handle, tensor_buffer, &buffer_handle), kLiteRtStatusOk);
  EXPECT_NE(buffer_handle, 0);
  
  // Unregister tensor buffer
  EXPECT_EQ(dispatch_api_.interface->unregister_tensor_buffer(
      device_handle, buffer_handle), kLiteRtStatusOk);
  
  // Cleanup
  dispatch_api_.interface->device_context_destroy(device_handle);
}

TEST_F(VendorDispatchTest, InvocationContextLifecycle) {
  // Create device context
  LiteRtDispatchDeviceContext device_context{};
  LiteRtDispatchDeviceContextHandle device_handle = nullptr;
  ASSERT_EQ(dispatch_api_.interface->device_context_create(device_context, &device_handle),
            kLiteRtStatusOk);
  
  // Create invocation context
  const char bytecode[] = "mock_bytecode";
  LiteRtDispatchInvocationContextHandle invocation_handle = nullptr;
  EXPECT_EQ(dispatch_api_.interface->invocation_context_create(
      device_handle, bytecode, sizeof(bytecode), "main", &invocation_handle),
      kLiteRtStatusOk);
  EXPECT_THAT(invocation_handle, NotNull());
  
  // Destroy invocation context
  EXPECT_EQ(dispatch_api_.interface->invocation_context_destroy(invocation_handle),
            kLiteRtStatusOk);
  
  // Cleanup
  dispatch_api_.interface->device_context_destroy(device_handle);
}

TEST_F(VendorDispatchTest, InvocationWithAttachedBuffers) {
  // Setup
  LiteRtDispatchDeviceContext device_context{};
  LiteRtDispatchDeviceContextHandle device_handle = nullptr;
  ASSERT_EQ(dispatch_api_.interface->device_context_create(device_context, &device_handle),
            kLiteRtStatusOk);
  
  const char bytecode[] = "mock_bytecode";
  LiteRtDispatchInvocationContextHandle invocation_handle = nullptr;
  ASSERT_EQ(dispatch_api_.interface->invocation_context_create(
      device_handle, bytecode, sizeof(bytecode), "main", &invocation_handle),
      kLiteRtStatusOk);
  
  // Get the actual mock context for verification
  auto* mock_context = reinterpret_cast<MockInvocationContext*>(invocation_handle);
  
  // Attach buffers
  LiteRtTensorBufferHandle input_handle = 100;
  LiteRtTensorBufferHandle output_handle = 200;
  
  EXPECT_EQ(dispatch_api_.interface->attach_input(invocation_handle, 0, input_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(dispatch_api_.interface->attach_output(invocation_handle, 0, output_handle),
            kLiteRtStatusOk);
  
  EXPECT_TRUE(mock_context->HasInput(0));
  EXPECT_TRUE(mock_context->HasOutput(0));
  
  // Invoke
  EXPECT_EQ(mock_context->GetInvokeCount(), 0);
  EXPECT_EQ(dispatch_api_.interface->invoke(invocation_handle), kLiteRtStatusOk);
  EXPECT_EQ(mock_context->GetInvokeCount(), 1);
  
  // Detach buffers
  EXPECT_EQ(dispatch_api_.interface->detach_input(invocation_handle, 0),
            kLiteRtStatusOk);
  EXPECT_EQ(dispatch_api_.interface->detach_output(invocation_handle, 0),
            kLiteRtStatusOk);
  
  EXPECT_FALSE(mock_context->HasInput(0));
  EXPECT_FALSE(mock_context->HasOutput(0));
  
  // Cleanup
  dispatch_api_.interface->invocation_context_destroy(invocation_handle);
  dispatch_api_.interface->device_context_destroy(device_handle);
}

TEST_F(VendorDispatchTest, GetRequirements) {
  // Create contexts
  LiteRtDispatchDeviceContext device_context{};
  LiteRtDispatchDeviceContextHandle device_handle = nullptr;
  ASSERT_EQ(dispatch_api_.interface->device_context_create(device_context, &device_handle),
            kLiteRtStatusOk);
  
  const char bytecode[] = "mock_bytecode";
  LiteRtDispatchInvocationContextHandle invocation_handle = nullptr;
  ASSERT_EQ(dispatch_api_.interface->invocation_context_create(
      device_handle, bytecode, sizeof(bytecode), "main", &invocation_handle),
      kLiteRtStatusOk);
  
  // Create tensor type
  LiteRtRankedTensorType tensor_type;
  tensor_type.element_type = kLiteRtElementTypeFloat32;
  tensor_type.layout.rank = 2;
  tensor_type.layout.dimensions[0] = 10;
  tensor_type.layout.dimensions[1] = 20;
  tensor_type.layout.has_strides = false;
  
  // Get input requirements
  LiteRtTensorBufferRequirements requirements = nullptr;
  EXPECT_EQ(dispatch_api_.interface->get_input_requirements(
      invocation_handle, 0, &tensor_type, &requirements), kLiteRtStatusOk);
  EXPECT_THAT(requirements, NotNull());
  
  // Verify requirements
  int num_types = 0;
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
      requirements, &num_types), kLiteRtStatusOk);
  EXPECT_GT(num_types, 0);
  
  size_t buffer_size = 0;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
      requirements, &buffer_size), kLiteRtStatusOk);
  EXPECT_EQ(buffer_size, 10 * 20 * sizeof(float));
  
  // Cleanup
  LiteRtDestroyTensorBufferRequirements(requirements);
  dispatch_api_.interface->invocation_context_destroy(invocation_handle);
  dispatch_api_.interface->device_context_destroy(device_handle);
}

// Test compile-time helpers
TEST_F(VendorDispatchTest, CompileTimeHelpers) {
  // Test capability helpers
  static_assert(!SupportsAsync<MockVendorTag>());
  static_assert(!SupportsGraph<MockVendorTag>());
  static_assert(GetCapabilities<MockVendorTag>() == kLiteRtDispatchCapabilitiesBasic);
  
  // Test trait constants
  static_assert(std::string_view(VendorTraits<MockVendorTag>::kVendorId) == "MockVendor");
  static_assert(!VendorTraits<MockVendorTag>::kSupportsAsync);
  static_assert(!VendorTraits<MockVendorTag>::kSupportsGraph);
}

}  // namespace
}  // namespace vendors
}  // namespace litert