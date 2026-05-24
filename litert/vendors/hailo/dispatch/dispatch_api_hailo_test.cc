#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/simple_buffer.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

namespace litert::hailo {
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

class HailoDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    LITERT_ASSERT_OK(LiteRtDispatchGetApi(&api_));
    ASSERT_NE(api_.interface, nullptr);
  }

  LiteRtDispatchInterface& Api() { return *api_.interface; }

  std::pair<DevicePtr, InvocationContextPtr>
  CreateDeviceContextInvocationContext() {
    static constexpr absl::string_view kSchema = "HAILO_EXEC_BYTECODE_DUMMY_DATA";
    static constexpr absl::string_view kFunctionName = "Partition_0";

    static constexpr int kNumInputs = 1;
    static constexpr int kNumOutputs = 1;

    LiteRtMemBuffer exec_bytecode_buffer;
    exec_bytecode_buffer.base_addr = kSchema.data();
    exec_bytecode_buffer.size = kSchema.size();
    exec_bytecode_buffer.fd = -1;
    exec_bytecode_buffer.offset = 0;

    static constexpr LiteRtDispatchExecutableType kExecType =
        kLiteRtDispatchExecutableTypeMlModel;

    LiteRtDispatchDeviceContext device_context;
    Api().device_context_create(LrtGetRuntimeContext(),
                                                 /*options=*/nullptr, &device_context);
    auto device_context_ptr = CreateDevicePtr(Api(), device_context);

    LiteRtDispatchInvocationContext invocation_context;
    Api().invocation_context_create(LrtGetRuntimeContext(), device_context,
                                                     kExecType, &exec_bytecode_buffer,
                                                     kFunctionName.data(), kNumInputs,
                                                     kNumOutputs, &invocation_context);
    auto invocation_context_ptr =
        CreateInvocationContextPtr(Api(), invocation_context);

    return std::make_pair(std::move(device_context_ptr),
                          std::move(invocation_context_ptr));
  }

 private:
  LiteRtDispatchApi api_;
};

TEST_F(HailoDispatchTest, CheckRuntimeCompatibility) {
  LiteRtApiVersion api_version = {.major = 1, .minor = 0, .patch = 0};
  LiteRtEnvironmentOptions env = nullptr;
  LiteRtOptions options = nullptr;
  LITERT_ASSERT_OK(
      Api().check_runtime_compatibility(api_version, env, options));
}

TEST_F(HailoDispatchTest, GetVendorId) {
  const char* vendor_id;
  LITERT_ASSERT_OK(Api().get_vendor_id(&vendor_id));
  EXPECT_STREQ(vendor_id, "Hailo");
}

TEST_F(HailoDispatchTest, GetBuildId) {
  const char* build_id;
  LITERT_ASSERT_OK(Api().get_build_id(&build_id));
  EXPECT_STREQ(build_id, "1.0");
}

TEST_F(HailoDispatchTest, GetCapabilities) {
  int capabilities;
  LITERT_ASSERT_OK(Api().get_capabilities(&capabilities));
  EXPECT_EQ(capabilities, kLiteRtDispatchCapabilitiesBasic);
}

TEST_F(HailoDispatchTest, DeviceContextCreate) {
  LiteRtDispatchDeviceContext device_context;
  LITERT_ASSERT_OK(Api().device_context_create(LrtGetRuntimeContext(),
                                               /*options=*/nullptr,
                                               &device_context));
  LITERT_ASSERT_OK(Api().device_context_destroy(device_context));
}

TEST_F(HailoDispatchTest, InvocationContextLifecycleAndInference) {
  auto [device_context_ptr, invocation_context_ptr] =
      CreateDeviceContextInvocationContext();
  LiteRtDispatchDeviceContext device_context = device_context_ptr.get();
  LiteRtDispatchInvocationContext invocation_context =
      invocation_context_ptr.get();

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto input1,
      SimpleBuffer::Create<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto input_tb1, input1.SpawnTensorBuffer());
  LITERT_ASSERT_OK_AND_ASSIGN(auto output, SimpleBuffer::Create<float>({2, 2}));
  LITERT_ASSERT_OK_AND_ASSIGN(auto output_tb, output.SpawnTensorBuffer());

  LiteRtTensorBufferHandle handle1;
  LITERT_ASSERT_OK(
      Api().register_tensor_buffer(device_context, input_tb1.Get(), &handle1));
  LiteRtTensorBufferHandle handle2;
  LITERT_ASSERT_OK(
      Api().register_tensor_buffer(device_context, output_tb.Get(), &handle2));

  LITERT_ASSERT_OK(Api().attach_input(invocation_context, 0, handle1));
  LITERT_ASSERT_OK(Api().attach_output(invocation_context, 0, handle2));

  LITERT_ASSERT_OK(Api().invoke(invocation_context));

  // Verify deallocation / detach
  LITERT_ASSERT_OK(Api().detach_input(invocation_context, 0, handle1));
  LITERT_ASSERT_OK(Api().detach_output(invocation_context, 0, handle2));

  LITERT_ASSERT_OK(Api().unregister_tensor_buffer(device_context, handle1));
  LITERT_ASSERT_OK(Api().unregister_tensor_buffer(device_context, handle2));
}

TEST_F(HailoDispatchTest, TensorBufferRequirementsInputs) {
  const auto t = MakeRankedTensorType<float>({2, 2});
  LiteRtTensorBufferRequirements requirements = nullptr;
  const auto litert_t = static_cast<LiteRtRankedTensorType>(t);

  auto [device_context_ptr, invocation_context_ptr] =
      CreateDeviceContextInvocationContext();

  LITERT_ASSERT_OK(Api().get_input_requirements(invocation_context_ptr.get(), 0,
                                                &litert_t, &requirements));
  int num_types;
  LITERT_ASSERT_OK(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
      requirements, &num_types));
  EXPECT_EQ(num_types, 1);
  LiteRtTensorBufferType type;
  LITERT_ASSERT_OK(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
      requirements, 0, &type));
  EXPECT_EQ(type, kLiteRtTensorBufferTypeHostMemory);
  LiteRtDestroyTensorBufferRequirements(requirements);
}

}  // namespace
}  // namespace litert::hailo
