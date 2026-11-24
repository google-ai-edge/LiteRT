// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/ir_backend.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "IR/QnnIrGraph.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {

constexpr auto kDefaultSocInfo = FindSocInfo("SM8750");
static_assert(kDefaultSocInfo.has_value());

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
  ASSERT_TRUE(backend_->Init(options, kDefaultSocInfo));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_FALSE(backend_->GetLogHandle());
  EXPECT_EQ(backend_->GetSocInfo().soc_model, kDefaultSocInfo->soc_model);
}

TEST_F(IrBackendTest, DISABLED_InitializeWithLogLevelVerboseTest) {
  Options options;
  options.SetLogLevel(LogLevel::kVerbose);

  options.SetBackendType(BackendType::kIrBackend);
  ASSERT_TRUE(backend_->Init(options, kDefaultSocInfo));
  ASSERT_FALSE(backend_->GetDeviceHandle());

  ASSERT_TRUE(backend_->GetBackendHandle());
  ASSERT_TRUE(backend_->GetLogHandle());
  EXPECT_EQ(backend_->GetSocInfo().soc_model, kDefaultSocInfo->soc_model);
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

TEST(IrBackendStripQuantTest, PerTensorStaticTensorIsDequantized) {
  TensorPool pool;
  const std::vector<std::int8_t> data = {-2, 0, 4, 6};
  auto& tensor = pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_8, ScaleOffsetQuantizeParamsWrapper(0.5f, 2),
      {4}, data.size(), data.data());

  ASSERT_TRUE(IrQuantStripper().Apply(pool));

  EXPECT_EQ(tensor.GetDataType(), QNN_DATATYPE_FLOAT_32);
  EXPECT_FALSE(tensor.IsQuant());
  auto values = tensor.GetTensorData<float>();
  ASSERT_TRUE(values.has_value());
  EXPECT_EQ(std::vector<float>(values->begin(), values->end()),
            (std::vector<float>{-2.0f, -1.0f, 1.0f, 2.0f}));
}

TEST(IrBackendStripQuantTest, PerChannelStaticTensorIsDequantized) {
  TensorPool pool;
  const std::vector<std::uint8_t> data = {10, 12, 10, 12};
  const std::vector<float> scales = {1.0f, 2.0f};
  const std::vector<std::int32_t> zero_points = {10, 10};
  auto& tensor = pool.CreateStaticTensor(
      QNN_DATATYPE_UFIXED_POINT_8,
      AxisScaleOffsetQuantizeParamsWrapper(0, scales, zero_points), {2, 2},
      data.size(), data.data());

  ASSERT_TRUE(IrQuantStripper().Apply(pool));

  auto values = tensor.GetTensorData<float>();
  ASSERT_TRUE(values.has_value());
  EXPECT_EQ(std::vector<float>(values->begin(), values->end()),
            (std::vector<float>{0.0f, 2.0f, 0.0f, 4.0f}));
}

TEST(IrBackendStripQuantTest, NativeTensorBecomesFloat) {
  TensorPool pool;
  auto& tensor = pool.CreateNativeTensor(
      QNN_DATATYPE_UFIXED_POINT_16, ScaleOffsetQuantizeParamsWrapper(0.1f, 0),
      {1, 4});

  ASSERT_TRUE(IrQuantStripper().Apply(pool));

  EXPECT_EQ(tensor.GetDataType(), QNN_DATATYPE_FLOAT_32);
  EXPECT_FALSE(tensor.IsQuant());
}

TEST(IrBackendStripQuantTest, UnsupportedStaticDataTypeFails) {
  TensorPool pool;
  const std::vector<std::uint32_t> data = {1, 2};
  pool.CreateStaticTensor(QNN_DATATYPE_UFIXED_POINT_32,
                          ScaleOffsetQuantizeParamsWrapper(1.0f, 0), {2},
                          data.size() * sizeof(std::uint32_t), data.data());

  EXPECT_FALSE(IrQuantStripper().Apply(pool));
}

TEST(IrBackendStripQuantTest, NonQuantizedTensorIsUntouched) {
  TensorPool pool;
  auto& tensor = pool.CreateNativeTensor(QNN_DATATYPE_FLOAT_16,
                                         UndefinedQuantizeParamsWrapper(), {2});

  ASSERT_TRUE(IrQuantStripper().Apply(pool));

  EXPECT_EQ(tensor.GetDataType(), QNN_DATATYPE_FLOAT_16);
}

TEST(IrBackendStripQuantTest, BuilderClearedBiasScaleIsRestored) {
  TensorPool pool;
  const std::vector<std::int32_t> data = {10, -20};
  auto& bias = pool.CreateStaticTensor(
      QNN_DATATYPE_SFIXED_POINT_32, ScaleOffsetQuantizeParamsWrapper(0.5f, 0),
      {2}, data.size() * sizeof(std::int32_t), data.data());
  IrQuantStripper stripper;
  stripper.Record({std::ref(bias)});
  bias.SetQuantParams(ScaleOffsetQuantizeParamsWrapper(0.0f, 0));  // Builder.

  ASSERT_TRUE(stripper.Apply(pool));

  auto values = bias.GetTensorData<float>();
  ASSERT_TRUE(values.has_value());
  EXPECT_EQ(std::vector<float>(values->begin(), values->end()),
            (std::vector<float>{5.0f, -10.0f}));
}

}  // namespace
}  // namespace qnn
