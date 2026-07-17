// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/ir_backend.h"

#include <memory>
#include <optional>

#include "IR/QnnIrGraph.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"

namespace qnn {
namespace {

class IrBackendTest : public testing::Test {
 public:
  void SetUp() override {
    // empty string for lib path means use default
    handle_ = ::qnn::CreateDLHandle(::qnn::IrBackend::GetLibraryName());
    backend_ = std::make_unique<::qnn::IrBackend>(::qnn::ResolveQnnApi(
        handle_.get(), ::qnn::IrBackend::GetExpectedBackendVersion()));
  }

  DLHandle handle_;
  std::unique_ptr<IrBackend> backend_;
};

TEST_F(IrBackendTest, DISABLED_InitializeWithLogLevelOffTest) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);

  options.SetBackendType(BackendType::kIrBackend);
  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_FALSE(backend_->GetLogHandle());
}

TEST_F(IrBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  options.SetBackendType(BackendType::kIrBackend);
  ASSERT_TRUE(backend_->Init(options, std::nullopt));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_TRUE(backend_->GetLogHandle());
}

TEST(IrBackendGraphConfigTest, ConfigsCarrySerializationAndDlcPath) {
  QNN_INTERFACE_VER_TYPE api{};
  IrBackend backend(&api);

  Options options;
  options.SetDlcDir("/tmp/dlc");
  auto config_builder = backend.BuildGraphConfigs(options, "my_graph");
  auto configs = config_builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.size(), 2u);
  ASSERT_NE(configs[0], nullptr);
  EXPECT_EQ(configs[1], nullptr);
  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);

  auto* custom =
      static_cast<QnnIrGraph_CustomConfig_t*>(configs[0]->customConfig);
  ASSERT_NE(custom, nullptr);
  EXPECT_EQ(custom->option, QNN_IR_GRAPH_CONFIG_OPTION_SERIALIZATION);
  EXPECT_EQ(custom->serializationOption.serializationType,
            QNN_IR_GRAPH_SERIALIZATION_TYPE_FLAT_BUFFER);
  EXPECT_STREQ(custom->serializationOption.outputPath, "/tmp/dlc/my_graph.dlc");
}

}  // namespace
}  // namespace qnn
