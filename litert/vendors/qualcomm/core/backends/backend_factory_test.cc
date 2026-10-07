// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/backend_factory.h"

#include <atomic>
#include <optional>
#include <string>

#include "QnnCommon.h"  // from @qairt
#include "QnnContext.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnOpDef.h"  // from @qairt
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/dsp_backend.h"
#include "litert/vendors/qualcomm/core/backends/gpu_backend.h"
#include "litert/vendors/qualcomm/core/backends/htp_backend.h"
#include "litert/vendors/qualcomm/core/backends/ir_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

constexpr auto kDefaultSocInfo = FindSocInfo("SM8750");
static_assert(kDefaultSocInfo.has_value());

struct RegisterCall {
  Qnn_BackendHandle_t backend = nullptr;
  std::string package_path;
  std::string interface_provider;
  std::string target;
};

RegisterCall& LastRegisterCall() {
  static absl::NoDestructor<RegisterCall> call;
  return *call;
}

Qnn_ErrorHandle_t MockRegisterOpPackage(Qnn_BackendHandle_t backend,
                                        const char* package_path,
                                        const char* interface_provider,
                                        const char* target) {
  auto& call = LastRegisterCall();
  call.backend = backend;
  call.package_path = package_path ? package_path : "";
  call.interface_provider = interface_provider ? interface_provider : "";
  call.target = target ? target : "";
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockRegisterOpPackageFail(Qnn_BackendHandle_t /*backend*/,
                                            const char* /*package_path*/,
                                            const char* /*interface_provider*/,
                                            const char* /*target*/) {
  return QNN_COMMON_ERROR_NOT_SUPPORTED;
}

struct QuickResponseFactoryState {
  std::atomic<int> context_create_count{0};
  std::atomic<int> context_free_count{0};
  std::atomic<bool> op_package_registered{false};
  std::atomic<bool> context_created_after_op_package_registration{false};
};

QuickResponseFactoryState& GetQuickResponseFactoryState() {
  static absl::NoDestructor<QuickResponseFactoryState> state;
  return *state;
}

void ResetQuickResponseFactoryState() {
  auto& state = GetQuickResponseFactoryState();
  state.context_create_count = 0;
  state.context_free_count = 0;
  state.op_package_registered = false;
  state.context_created_after_op_package_registration = false;
}

Qnn_ErrorHandle_t MockQuickResponseBackendCreate(Qnn_LogHandle_t,
                                                 const QnnBackend_Config_t**,
                                                 Qnn_BackendHandle_t* backend) {
  static int backend_handle;
  *backend = &backend_handle;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseBackendFree(Qnn_BackendHandle_t) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseDeviceCreate(Qnn_LogHandle_t,
                                                const QnnDevice_Config_t**,
                                                Qnn_DeviceHandle_t* device) {
  static int device_handle;
  *device = &device_handle;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseDeviceFree(Qnn_DeviceHandle_t) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseRegisterOpPackage(Qnn_BackendHandle_t,
                                                     const char*, const char*,
                                                     const char*) {
  GetQuickResponseFactoryState().op_package_registered = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseContextCreate(Qnn_BackendHandle_t,
                                                 Qnn_DeviceHandle_t,
                                                 const QnnContext_Config_t**,
                                                 Qnn_ContextHandle_t* context) {
  auto& state = GetQuickResponseFactoryState();
  ++state.context_create_count;
  state.context_created_after_op_package_registration =
      state.op_package_registered.load();
  static int context_handle;
  *context = &context_handle;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseContextFree(Qnn_ContextHandle_t,
                                               Qnn_ProfileHandle_t) {
  ++GetQuickResponseFactoryState().context_free_count;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseGraphCreate(Qnn_ContextHandle_t, const char*,
                                               const QnnGraph_Config_t**,
                                               Qnn_GraphHandle_t* graph) {
  static int graph_handle;
  *graph = &graph_handle;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseTensorCreate(Qnn_GraphHandle_t,
                                                Qnn_Tensor_t*) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseValidateOp(Qnn_BackendHandle_t,
                                              Qnn_OpConfig_t) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseGraphAddNode(Qnn_GraphHandle_t,
                                                Qnn_OpConfig_t) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseGraphFinalize(Qnn_GraphHandle_t,
                                                 Qnn_ProfileHandle_t,
                                                 Qnn_SignalHandle_t) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t MockQuickResponseGraphExecute(Qnn_GraphHandle_t,
                                                const Qnn_Tensor_t*, uint32_t,
                                                Qnn_Tensor_t*, uint32_t,
                                                Qnn_ProfileHandle_t,
                                                Qnn_SignalHandle_t) {
  return QNN_SUCCESS;
}

QNN_INTERFACE_VER_TYPE CreateQuickResponseFactoryApi() {
  QNN_INTERFACE_VER_TYPE api{};
  api.backendCreate = MockQuickResponseBackendCreate;
  api.backendFree = MockQuickResponseBackendFree;
  api.deviceCreate = MockQuickResponseDeviceCreate;
  api.deviceFree = MockQuickResponseDeviceFree;
  api.backendRegisterOpPackage = MockQuickResponseRegisterOpPackage;
  api.contextCreate = MockQuickResponseContextCreate;
  api.contextFree = MockQuickResponseContextFree;
  api.graphCreate = MockQuickResponseGraphCreate;
  api.tensorCreateGraphTensor = MockQuickResponseTensorCreate;
  api.backendValidateOpConfig = MockQuickResponseValidateOp;
  api.graphAddNode = MockQuickResponseGraphAddNode;
  api.graphFinalize = MockQuickResponseGraphFinalize;
  api.graphExecute = MockQuickResponseGraphExecute;
  return api;
}

class TestQnnBackend : public QnnBackend {
 public:
  TestQnnBackend() : QnnBackend(&Api()) {}

  bool Init(const Options&, std::optional<SocInfo>) override { return true; }

  GraphConfigBuilder BuildGraphConfigs(const Options&,
                                       absl::string_view) override {
    return {};
  }

 private:
  static const QNN_INTERFACE_VER_TYPE& Api() {
    static const QNN_INTERFACE_VER_TYPE api{};
    return api;
  }
};

template <typename BackendT>
void TestCreateBackend(BackendType backend_type,
                       std::optional<SocInfo> soc_info = kDefaultSocInfo) {
  auto handle = CreateDLHandle(BackendT::GetLibraryName());
  if (!handle) GTEST_SKIP();
  const auto* real =
      ResolveQnnApi(handle.get(), BackendT::GetExpectedBackendVersion());
  ASSERT_TRUE(real);
  auto api = *real;
  api.backendRegisterOpPackage = MockRegisterOpPackage;

  const bool is_custom_op_supported = backend_type == BackendType::kHtpBackend;

  // Base create + empty custom-op name skips register.
  {
    LastRegisterCall() = {};
    Options options;
    options.SetBackendType(backend_type);
    auto backend = CreateBackend(&api, options, soc_info,
                                 /*is_compiler=*/true);
    EXPECT_NE(backend.get(), nullptr);
    EXPECT_EQ(LastRegisterCall().backend, nullptr);
  }

  // Shared setup for the custom-op scenarios below.
  Options options;
  options.SetBackendType(backend_type);
  options.SetCustomOpPackage("MyPackage", "MyProvider",
                             "/tmp/compile_package.so",
                             "/tmp/dispatch_package.so", "HTP");

  // Compile path overrides target to CPU.
  {
    LastRegisterCall() = {};
    auto backend = CreateBackend(&api, options, soc_info,
                                 /*is_compiler=*/true);
    ASSERT_NE(backend.get(), nullptr);
    const auto& call = LastRegisterCall();
    if (is_custom_op_supported) {
      EXPECT_NE(call.backend, nullptr);
      EXPECT_EQ(call.package_path, "/tmp/compile_package.so");
      EXPECT_EQ(call.interface_provider, "MyProvider");
      EXPECT_EQ(call.target, "CPU");
    } else {
      EXPECT_EQ(call.backend, nullptr);
    }
  }

  // Dispatch path uses options target.
  {
    LastRegisterCall() = {};
    auto backend = CreateBackend(&api, options, soc_info,
                                 /*is_compiler=*/false);
    ASSERT_NE(backend.get(), nullptr);
    const auto& call = LastRegisterCall();
    if (is_custom_op_supported) {
      EXPECT_NE(call.backend, nullptr);
      EXPECT_EQ(call.package_path, "/tmp/dispatch_package.so");
      EXPECT_EQ(call.interface_provider, "MyProvider");
      EXPECT_EQ(call.target, "HTP");
    } else {
      EXPECT_EQ(call.backend, nullptr);
    }
  }

  // Register failure returns null.
  if (is_custom_op_supported) {
    LastRegisterCall() = {};
    api.backendRegisterOpPackage = MockRegisterOpPackageFail;
    auto backend = CreateBackend(&api, options, soc_info,
                                 /*is_compiler=*/true);
    EXPECT_EQ(backend.get(), nullptr);
    EXPECT_EQ(LastRegisterCall().backend, nullptr);
  }
}

TEST(QnnBackendTest, StopBackgroundWorkBaseHookIsIdempotent) {
  TestQnnBackend backend;
  backend.StopBackgroundWork();
  backend.StopBackgroundWork();
}

TEST(CreateBackendTest, CreateReturnsNullForUnsupportedBackend) {
  Options options;
  options.SetBackendType(BackendType::kUndefinedBackend);

  auto backend = CreateBackend(nullptr, options, kDefaultSocInfo,
                               /*is_compiler=*/true);
  EXPECT_EQ(backend.get(), nullptr);
}

TEST(CreateBackendTest, HtpQuickResponseStartsOnlyForDispatch) {
  auto api = CreateQuickResponseFactoryApi();
  Options options;
  options.SetBackendType(BackendType::kHtpBackend);
  options.SetLogLevel(LogLevel::kOff);
  options.SetEnableHtpQuickResponse(true);
  options.SetCustomOpPackage("MyPackage", "MyProvider",
                             "/tmp/compile_package.so",
                             "/tmp/dispatch_package.so", "HTP");

  ResetQuickResponseFactoryState();
  auto compiler_backend =
      CreateBackend(&api, options, kDefaultSocInfo, /*is_compiler=*/true);
  ASSERT_NE(compiler_backend, nullptr);
  EXPECT_TRUE(GetQuickResponseFactoryState().op_package_registered.load());
  EXPECT_EQ(GetQuickResponseFactoryState().context_create_count.load(), 0);

  ResetQuickResponseFactoryState();
  options.SetEnableHtpQuickResponse(false);
  auto disabled_backend =
      CreateBackend(&api, options, kDefaultSocInfo, /*is_compiler=*/false);
  ASSERT_NE(disabled_backend, nullptr);
  EXPECT_TRUE(GetQuickResponseFactoryState().op_package_registered.load());
  EXPECT_EQ(GetQuickResponseFactoryState().context_create_count.load(), 0);

  ResetQuickResponseFactoryState();
  options.SetEnableHtpQuickResponse(true);
  {
    auto dispatch_backend =
        CreateBackend(&api, options, kDefaultSocInfo, /*is_compiler=*/false);
    ASSERT_NE(dispatch_backend, nullptr);
    EXPECT_TRUE(GetQuickResponseFactoryState().op_package_registered.load());
    EXPECT_EQ(GetQuickResponseFactoryState().context_create_count.load(), 1);
    EXPECT_TRUE(GetQuickResponseFactoryState()
                    .context_created_after_op_package_registration.load());
  }
  EXPECT_EQ(GetQuickResponseFactoryState().context_free_count.load(), 1);
}

TEST(CreateBackendTest, DISABLED_CreateGpuBackend) {
  TestCreateBackend<GpuBackend>(BackendType::kGpuBackend);
}

TEST(CreateBackendTest, DISABLED_CreateHtpBackend) {
#if defined(__x86_64__) || defined(_M_X64)
  TestCreateBackend<HtpBackend>(BackendType::kHtpBackend);
#else
  TestCreateBackend<HtpBackend>(BackendType::kHtpBackend, std::nullopt);
#endif
}

TEST(CreateBackendTest, DISABLED_CreateIrBackend) {
  TestCreateBackend<IrBackend>(BackendType::kIrBackend);
}

TEST(CreateBackendTest, DISABLED_CreateDspBackend) {
  TestCreateBackend<DspBackend>(BackendType::kDspBackend);
}

}  // namespace
}  // namespace qnn
