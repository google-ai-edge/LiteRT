// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"

#include <cstddef>
#include <string>
#include <vector>

#include "HTP/QnnHtpGraph.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnTypes.h"  // from @qairt
#include <gtest/gtest.h>

namespace qnn {
namespace {

// Length of a null-terminated array returned by GetNullTerminatedConfigs().
// Bounded by span.size() in case the terminator is missing.
size_t NumConfigs(const std::vector<const QnnGraph_Config_t*>& span) {
  size_t n = 0;
  while (n < span.size() && span[n] != nullptr) ++n;
  return n;
}

TEST(GraphConfigBuilderTest, EmptyBuilderReturnsOnlyNullTerminator) {
  GraphConfigBuilder config_builder;
  auto configs = config_builder.GetNullTerminatedConfigs();
  ASSERT_EQ(configs.size(), 1);
  ASSERT_EQ(NumConfigs(configs), 0);
  EXPECT_EQ(configs[0], nullptr);
}

TEST(GraphConfigBuilderTest, AddCustomConfigAndNullTerminates) {
  GraphConfigBuilder config_builder;
  QnnHtpGraph_CustomConfig_t cc = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  cc.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  cc.vtcmSizeInMB = 4;
  config_builder.AddCustomConfig(cc);

  auto configs = config_builder.GetNullTerminatedConfigs();
  ASSERT_EQ(configs.size(), 2);  // one config + null terminator.
  ASSERT_EQ(NumConfigs(configs), 1);
  ASSERT_NE(configs[0], nullptr);
  EXPECT_EQ(configs[1], nullptr);

  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  const auto* wrapped =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[0]->customConfig);
  ASSERT_NE(wrapped, nullptr);
  EXPECT_EQ(wrapped->option, QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE);
  EXPECT_EQ(wrapped->vtcmSizeInMB, 4);
}

TEST(GraphConfigBuilderTest, MultipleCustomConfigsKeepStableAddresses) {
  GraphConfigBuilder config_builder;
  QnnHtpGraph_CustomConfig_t vtcm_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  vtcm_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  vtcm_config.vtcmSizeInMB = 1;
  config_builder.AddCustomConfig(vtcm_config);
  QnnHtpGraph_CustomConfig_t hvx_threads_config =
      QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  hvx_threads_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
  hvx_threads_config.numHvxThreads = 2;
  config_builder.AddCustomConfig(hvx_threads_config);

  auto configs = config_builder.GetNullTerminatedConfigs();
  ASSERT_EQ(configs.size(), 3);  // two configs + null terminator.
  ASSERT_EQ(NumConfigs(configs), 2);
  // The two custom configs are copied into distinct, stable storage nodes.
  const auto* wrapped_vtcm =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[0]->customConfig);
  const auto* wrapped_hvx_threads =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[1]->customConfig);
  ASSERT_NE(wrapped_vtcm, nullptr);
  ASSERT_NE(wrapped_hvx_threads, nullptr);
  EXPECT_EQ(wrapped_vtcm->option, QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE);
  EXPECT_EQ(wrapped_hvx_threads->option,
            QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS);
}

TEST(GraphConfigBuilderTest, AddGraphConfigAndNullTerminates) {
  GraphConfigBuilder config_builder;
  QnnGraph_Config_t priority = QNN_GRAPH_CONFIG_INIT;
  priority.option = QNN_GRAPH_CONFIG_OPTION_PRIORITY;
  priority.priority = QNN_PRIORITY_HIGH;
  config_builder.AddGraphConfig(priority);

  auto configs = config_builder.GetNullTerminatedConfigs();
  ASSERT_EQ(configs.size(), 2);  // one config + null terminator.
  ASSERT_EQ(NumConfigs(configs), 1);
  ASSERT_NE(configs[0], nullptr);
  EXPECT_EQ(configs[1], nullptr);
  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);
  // Check priority below not customConfig since they share the same union,
  EXPECT_EQ(configs[0]->priority, QNN_PRIORITY_HIGH);
}

TEST(GraphConfigBuilderTest, StoreOwnsSideBufferWithStableAddress) {
  GraphConfigBuilder config_builder;
  const std::string& s =
      config_builder.Store<std::string>("/tmp/dlc/graph.dlc");

  // Enough insertions to force any contiguous container to reallocate.
  QnnHtpGraph_CustomConfig_t filler = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  for (int i = 0; i < 128; ++i) {
    config_builder.AddCustomConfig(filler);
  }

  // The reference captured before the insertions still names the same string.
  EXPECT_EQ(s, "/tmp/dlc/graph.dlc");
}

}  // namespace
}  // namespace qnn
