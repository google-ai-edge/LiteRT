// Copyright 2024 Google LLC.
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

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/test/matchers.h"
#include "litert/test/simple_buffer.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

namespace litert::example {
namespace {

using ::litert::testing::SimpleBuffer;
using ::testing::ElementsAre;

struct DeviceDeleter {
  explicit DeviceDeleter(LiteRtDispatchInterface& api) : api(api) {}

  DeviceDeleter(const DeviceDeleter&) noexcept = default;
  DeviceDeleter(DeviceDeleter&&) noexcept = default;

  void operator()(LiteRtDispatchDeviceContext device_context) {
    api.device_context_destroy(device_context);
  }

  LiteRtDispatchInterface& api;
};

using DevicePtr = std::unique_ptr<LiteRtDispatchDeviceContextT, DeviceDeleter>;

DevicePtr CreateDevicePtr(LiteRtDispatchInterface& api,
                          LiteRtDispatchDeviceContext device_context) {
  return DevicePtr(device_context, DeviceDeleter(api));
}

struct InvocationContextDeleter {
  explicit InvocationContextDeleter(LiteRtDispatchInterface& api) : api(api) {}

  InvocationContextDeleter(const InvocationContextDeleter&) noexcept = default;
  InvocationContextDeleter(InvocationContextDeleter&&) noexcept = default;

  void operator()(LiteRtDispatchInvocationContext invocation_context) {
    api.invocation_context_destroy(invocation_context);
  }

  LiteRtDispatchInterface& api;
};

using InvocationContextPtr =
    std::unique_ptr<LiteRtDispatchInvocationContextT, InvocationContextDeleter>;

InvocationContextPtr CreateInvocationContextPtr(
    LiteRtDispatchInterface& api,
    LiteRtDispatchInvocationContext invocation_context) {
  return InvocationContextPtr(invocation_context,
                              InvocationContextDeleter(api));
}

class ExampleDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    LITERT_ASSERT_OK(LiteRtDispatchGetApi(&api_));
    ASSERT_NE(api_.interface, nullptr);
  }

  LiteRtDispatchInterface& Api() { return *api_.interface; }

 private:
  LiteRtDispatchApi api_;
};

TEST_F(ExampleDispatchTest, GetVendorId) {
  const char* vendor_id;
  LITERT_ASSERT_OK(Api().get_vendor_id(&vendor_id));
  EXPECT_STREQ(vendor_id, "Example");
}

TEST_F(ExampleDispatchTest, GetBuildId) {
  const char* build_id;
  LITERT_ASSERT_OK(Api().get_build_id(&build_id));
  EXPECT_STREQ(build_id, "ExampleBuild");
}

TEST_F(ExampleDispatchTest, GetCapabilities) {
  int capabilities;
  LITERT_ASSERT_OK(Api().get_capabilities(&capabilities));
  EXPECT_EQ(capabilities, kLiteRtDispatchCapabilitiesBasic);
}

TEST_F(ExampleDispatchTest, DeviceContextCreate) {
  LiteRtDispatchDeviceContext device_context;
  LITERT_ASSERT_OK(Api().device_context_create(&device_context));
  LITERT_ASSERT_OK(Api().device_context_destroy(device_context));
}

TEST_F(ExampleDispatchTest, InvocationContext) {
  // clang-format off
  static constexpr absl::string_view kSchema = R"(inputs:0,1
outputs:2
tensors:[2x2],[2x2],[2x2]
ops:mul(0,1)(2))";
  // clang-format on
  static constexpr absl::string_view kFunctionName = "partition_0";

  static constexpr int kNumInputs = 2;
  static constexpr int kNumOutputs = 1;

  LiteRtMemBuffer exec_bytecode_buffer;
  exec_bytecode_buffer.base_addr = kSchema.data();
  exec_bytecode_buffer.size = kSchema.size();
  exec_bytecode_buffer.fd = -1;
  exec_bytecode_buffer.offset = 0;

  static constexpr LiteRtDispatchExecutableType kExecType =
      kLiteRtDispatchExecutableTypeMlModel;

  LiteRtDispatchDeviceContext device_context;
  LITERT_ASSERT_OK(Api().device_context_create(&device_context));
  auto device_context_ptr = CreateDevicePtr(Api(), device_context);

  LiteRtDispatchInvocationContext invocation_context;
  LITERT_ASSERT_OK(Api().invocation_context_create(
      device_context, kExecType, &exec_bytecode_buffer, kFunctionName.data(),
      kNumInputs, kNumOutputs, &invocation_context));
  auto invocation_context_ptr =
      CreateInvocationContextPtr(Api(), invocation_context);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input1,
      SimpleBuffer::Create<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_tb1, input1.SpawnTensorBuffer());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input2,
      SimpleBuffer::Create<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_tb2, input2.SpawnTensorBuffer());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output, SimpleBuffer::Create<float>({2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_tb, output.SpawnTensorBuffer());
  LiteRtTensorBufferHandle handle1;
  LITERT_ASSERT_OK(
      Api().register_tensor_buffer(device_context, input_tb1.Get(), &handle1));
  LiteRtTensorBufferHandle handle2;
  LITERT_ASSERT_OK(
      Api().register_tensor_buffer(device_context, input_tb2.Get(), &handle2));
  LiteRtTensorBufferHandle handle3;
  LITERT_ASSERT_OK(
      Api().register_tensor_buffer(device_context, output_tb.Get(), &handle3));

  LITERT_ASSERT_OK(Api().attach_input(invocation_context, 0, handle1));
  LITERT_ASSERT_OK(Api().attach_input(invocation_context, 1, handle2));
  LITERT_ASSERT_OK(Api().attach_output(invocation_context, 0, handle3));

  LITERT_ASSERT_OK(Api().invoke(invocation_context));

  std::vector<float> out(4);
  LITERT_ASSERT_OK(output_tb.Read(absl::MakeSpan(out)));
  EXPECT_THAT(out, ElementsAre(1.0f, 4.0f, 9.0f, 16.0f));
}

TEST_F(ExampleDispatchTest, TensorBufferRequirementsInputs) {
  const auto t = MakeRankedTensorType<float>({2, 2});
  LiteRtTensorBufferRequirements requirements = nullptr;
  const auto litert_t = static_cast<LiteRtRankedTensorType>(t);
  LITERT_ASSERT_OK(
      Api().get_input_requirements(nullptr, 0, &litert_t, &requirements));
  auto req =
      TensorBufferRequirements::WrapCObject(requirements, OwnHandle::kYes);
  LITERT_ASSERT_OK_AND_ASSIGN(auto supported_types, req.SupportedTypes());
  EXPECT_THAT(supported_types, ElementsAre(kLiteRtTensorBufferTypeHostMemory));
}

TEST_F(ExampleDispatchTest, TensorBufferRequirementsOutputs) {
  const auto t = MakeRankedTensorType<float>({2, 2});
  LiteRtTensorBufferRequirements requirements = nullptr;
  const auto litert_t = static_cast<LiteRtRankedTensorType>(t);
  LITERT_ASSERT_OK(
      Api().get_output_requirements(nullptr, 0, &litert_t, &requirements));
  auto req =
      TensorBufferRequirements::WrapCObject(requirements, OwnHandle::kYes);
  LITERT_ASSERT_OK_AND_ASSIGN(auto supported_types, req.SupportedTypes());
  EXPECT_THAT(supported_types, ElementsAre(kLiteRtTensorBufferTypeHostMemory));
}

TEST_F(ExampleDispatchTest, RegisterBuffer) {
  LiteRtDispatchDeviceContext device_context;
  LITERT_ASSERT_OK(Api().device_context_create(&device_context));
  auto device_context_ptr = CreateDevicePtr(Api(), device_context);
  LiteRtTensorBufferHandle handle;
  LITERT_ASSERT_OK(
      Api().register_tensor_buffer(device_context, nullptr, &handle));
  LITERT_ASSERT_OK(Api().unregister_tensor_buffer(device_context, handle));
}

}  // namespace
}  // namespace litert::example
