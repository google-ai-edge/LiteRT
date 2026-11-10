// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/htp_backend.h"

#include <cstdint>

#include "QnnBackend.h"    // from @qairt
#include "QnnCommon.h"     // from @qairt
#include "QnnDevice.h"     // from @qairt
#include "QnnGraph.h"      // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "QnnOpDef.h"      // from @qairt
#include "QnnTypes.h"      // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {
namespace {

class MockQnnApi {
 public:
  MockQnnApi() { active_ = this; }
  ~MockQnnApi() { active_ = nullptr; }

  MockQnnApi(const MockQnnApi&) = delete;
  MockQnnApi& operator=(const MockQnnApi&) = delete;
  MockQnnApi(MockQnnApi&&) = delete;
  MockQnnApi& operator=(MockQnnApi&&) = delete;

  QNN_INTERFACE_VER_TYPE CreateApi() {
    QNN_INTERFACE_VER_TYPE api{};
    api.backendCreate = BackendCreate;
    api.backendFree = BackendFree;
    api.deviceCreate = DeviceCreate;
    api.deviceFree = DeviceFree;
    api.contextCreate = ContextCreate;
    api.contextFree = ContextFree;
    api.graphCreate = GraphCreate;
    api.tensorCreateGraphTensor = TensorCreateGraphTensor;
    api.backendValidateOpConfig = BackendValidateOpConfig;
    api.graphAddNode = GraphAddNode;
    api.graphFinalize = GraphFinalize;
    api.graphExecute = GraphExecute;
    return api;
  }

  int context_create_count = 0;
  int context_free_count = 0;
  int graph_create_count = 0;
  int tensor_create_count = 0;
  int validate_count = 0;
  int add_node_count = 0;
  int finalize_count = 0;
  int execute_count = 0;
  bool execute_has_buffers = false;
  bool graph_is_low_priority = false;
  Qnn_ErrorHandle_t context_create_status = QNN_SUCCESS;
  Qnn_ErrorHandle_t graph_create_status = QNN_SUCCESS;
  Qnn_ErrorHandle_t graph_execute_status = QNN_SUCCESS;
  const char* captured_op_type = nullptr;

 private:
  static MockQnnApi& Get() { return *active_; }

  static Qnn_ErrorHandle_t BackendCreate(Qnn_LogHandle_t,
                                         const QnnBackend_Config_t**,
                                         Qnn_BackendHandle_t* backend) {
    *backend = &Get().backend_handle_storage_;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t BackendFree(Qnn_BackendHandle_t) {
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t DeviceCreate(Qnn_LogHandle_t,
                                        const QnnDevice_Config_t**,
                                        Qnn_DeviceHandle_t* device) {
    *device = &Get().device_handle_storage_;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t DeviceFree(Qnn_DeviceHandle_t) {
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t ContextCreate(Qnn_BackendHandle_t,
                                         Qnn_DeviceHandle_t,
                                         const QnnContext_Config_t**,
                                         Qnn_ContextHandle_t* context) {
    auto& mock = Get();
    mock.context_create_count++;
    if (mock.context_create_status != QNN_SUCCESS) {
      return mock.context_create_status;
    }
    *context = &mock.context_handle_storage_;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t ContextFree(Qnn_ContextHandle_t,
                                       Qnn_ProfileHandle_t) {
    Get().context_free_count++;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t GraphCreate(Qnn_ContextHandle_t, const char*,
                                       const QnnGraph_Config_t** configs,
                                       Qnn_GraphHandle_t* graph) {
    auto& mock = Get();
    mock.graph_create_count++;
    if (mock.graph_create_status != QNN_SUCCESS) {
      return mock.graph_create_status;
    }
    mock.graph_is_low_priority =
        configs != nullptr && configs[0] != nullptr &&
        configs[0]->option == QNN_GRAPH_CONFIG_OPTION_PRIORITY &&
        configs[0]->priority == QNN_PRIORITY_LOW;
    *graph = &mock.graph_handle_storage_;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t TensorCreateGraphTensor(Qnn_GraphHandle_t,
                                                   Qnn_Tensor_t*) {
    Get().tensor_create_count++;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t BackendValidateOpConfig(Qnn_BackendHandle_t,
                                                   Qnn_OpConfig_t op_config) {
    auto& mock = Get();
    mock.validate_count++;
    mock.captured_op_type = op_config.v1.typeName;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t GraphAddNode(Qnn_GraphHandle_t,
                                        Qnn_OpConfig_t op_config) {
    auto& mock = Get();
    mock.add_node_count++;
    mock.captured_op_type = op_config.v1.typeName;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t GraphFinalize(Qnn_GraphHandle_t, Qnn_ProfileHandle_t,
                                         Qnn_SignalHandle_t) {
    Get().finalize_count++;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t GraphExecute(
      Qnn_GraphHandle_t, const Qnn_Tensor_t* inputs, std::uint32_t num_inputs,
      Qnn_Tensor_t* outputs, std::uint32_t num_outputs, Qnn_ProfileHandle_t,
      Qnn_SignalHandle_t) {
    auto& mock = Get();
    mock.execute_count++;
    mock.execute_has_buffers =
        num_inputs == 2 && num_outputs == 1 &&
        inputs[0].v2.clientBuf.data != nullptr &&
        inputs[0].v2.clientBuf.dataSize == sizeof(std::int8_t) &&
        inputs[1].v2.clientBuf.data != nullptr &&
        inputs[1].v2.clientBuf.dataSize == sizeof(std::int8_t) &&
        outputs[0].v2.clientBuf.data != nullptr &&
        outputs[0].v2.clientBuf.dataSize == sizeof(std::int8_t);
    return mock.graph_execute_status;
  }

  int backend_handle_storage_;
  int device_handle_storage_;
  int context_handle_storage_;
  int graph_handle_storage_;
  inline static MockQnnApi* active_ = nullptr;
};

class HtpQuickResponseTest : public testing::Test {
 protected:
  void SetUp() override {
    api_ = mock_.CreateApi();
    options_.SetLogLevel(LogLevel::kOff);
    options_.SetEnableHtpQuickResponse(true);
  }

  bool InitBackend(HtpBackend& backend) {
    return backend.Init(options_, FindSocInfo("SM8750"));
  }

  MockQnnApi mock_;
  QNN_INTERFACE_VER_TYPE api_{};
  Options options_;
};

TEST_F(HtpQuickResponseTest, BuildsAndExecutesTinyGraph) {
  HtpBackend backend(&api_);
  ASSERT_TRUE(InitBackend(backend));

  backend.StartBackgroundWork(options_);
  backend.StopBackgroundWork();

  EXPECT_EQ(mock_.context_create_count, 1);
  EXPECT_EQ(mock_.context_free_count, 1);
  EXPECT_EQ(mock_.graph_create_count, 1);
  EXPECT_EQ(mock_.tensor_create_count, 3);
  EXPECT_EQ(mock_.validate_count, 1);
  EXPECT_EQ(mock_.add_node_count, 1);
  EXPECT_EQ(mock_.finalize_count, 1);
  EXPECT_STREQ(mock_.captured_op_type, QNN_OP_ELEMENT_WISE_ADD);
  EXPECT_TRUE(mock_.graph_is_low_priority);
  EXPECT_GT(mock_.execute_count, 0);
  EXPECT_TRUE(mock_.execute_has_buffers);
}

TEST_F(HtpQuickResponseTest, RepeatedStartDoesNotCreateAnotherContext) {
  HtpBackend backend(&api_);
  ASSERT_TRUE(InitBackend(backend));

  backend.StartBackgroundWork(options_);
  backend.StartBackgroundWork(options_);
  backend.StopBackgroundWork();

  EXPECT_EQ(mock_.context_create_count, 1);
  EXPECT_EQ(mock_.context_free_count, 1);
}

TEST_F(HtpQuickResponseTest, DisabledOptionDoesNotStartBackgroundWork) {
  options_.SetEnableHtpQuickResponse(false);
  HtpBackend backend(&api_);
  ASSERT_TRUE(InitBackend(backend));

  backend.StartBackgroundWork(options_);

  EXPECT_EQ(mock_.context_create_count, 0);
  EXPECT_EQ(mock_.execute_count, 0);
}

TEST_F(HtpQuickResponseTest, GraphSetupFailureReleasesContext) {
  mock_.graph_create_status = QNN_COMMON_ERROR_GENERAL;
  HtpBackend backend(&api_);
  ASSERT_TRUE(InitBackend(backend));

  backend.StartBackgroundWork(options_);

  EXPECT_EQ(mock_.context_create_count, 1);
  EXPECT_EQ(mock_.context_free_count, 1);
  EXPECT_EQ(mock_.graph_create_count, 1);
  EXPECT_EQ(mock_.execute_count, 0);
}

TEST_F(HtpQuickResponseTest, ContextCreateFailureDoesNotStartBackgroundWork) {
  mock_.context_create_status = QNN_COMMON_ERROR_GENERAL;
  HtpBackend backend(&api_);
  ASSERT_TRUE(InitBackend(backend));

  backend.StartBackgroundWork(options_);

  EXPECT_EQ(mock_.context_create_count, 1);
  EXPECT_EQ(mock_.context_free_count, 0);
  EXPECT_EQ(mock_.graph_create_count, 0);
  EXPECT_EQ(mock_.execute_count, 0);
}

TEST_F(HtpQuickResponseTest, GraphExecuteFailureStopsBackgroundWork) {
  mock_.graph_execute_status = QNN_COMMON_ERROR_GENERAL;
  HtpBackend backend(&api_);
  ASSERT_TRUE(InitBackend(backend));

  backend.StartBackgroundWork(options_);
  backend.StopBackgroundWork();

  EXPECT_EQ(mock_.execute_count, 1);
  EXPECT_EQ(mock_.context_free_count, 1);
}

}  // namespace
}  // namespace qnn
