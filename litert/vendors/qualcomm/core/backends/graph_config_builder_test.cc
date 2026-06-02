// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "HTP/QnnHtpGraph.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt

namespace qnn {
namespace {

// The builder must be move-only: its QnnGraph_Config_t entries hold pointers
// into its own storage, so a copy would dangle those pointers at the source.
static_assert(std::is_move_constructible_v<GraphConfigBuilder>);
static_assert(std::is_move_assignable_v<GraphConfigBuilder>);
static_assert(!std::is_copy_constructible_v<GraphConfigBuilder>);
static_assert(!std::is_copy_assignable_v<GraphConfigBuilder>);

// Counts the non-null entries in a null-terminated span returned by Configs().
size_t NonNullCount(absl::Span<const QnnGraph_Config_t*> span) {
  size_t n = 0;
  for (const auto* cfg : span) {
    if (cfg != nullptr) ++n;
  }
  return n;
}

TEST(GraphConfigBuilderTest, EmptyBuilderReturnsOnlyNullTerminator) {
  GraphConfigBuilder mgr;
  auto configs = mgr.Configs();
  ASSERT_EQ(configs.size(), 1);
  EXPECT_EQ(configs[0], nullptr);
}

TEST(GraphConfigBuilderTest, AddCustomWrapsAndNullTerminates) {
  GraphConfigBuilder mgr;
  auto& cc = mgr.AddCustom<QnnHtpGraph_CustomConfig_t>(
      QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
  cc.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  cc.vtcmSizeInMB = 4;

  auto configs = mgr.Configs();
  ASSERT_EQ(configs.size(), 2);  // one config + null terminator.
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
  GraphConfigBuilder mgr;
  auto& a = mgr.AddCustom<QnnHtpGraph_CustomConfig_t>(
      QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
  a.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  a.vtcmSizeInMB = 1;
  auto& b = mgr.AddCustom<QnnHtpGraph_CustomConfig_t>(
      QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
  b.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
  b.numHvxThreads = 2;

  auto configs = mgr.Configs();
  ASSERT_EQ(NonNullCount(configs), 2);
  // The custom-config pointers must still address the original storage even
  // after a second AddCustom() (std::list keeps element addresses stable).
  EXPECT_EQ(configs[0]->customConfig, &a);
  EXPECT_EQ(configs[1]->customConfig, &b);
}

TEST(GraphConfigBuilderTest, AddGraphConfigAppendsBareConfig) {
  GraphConfigBuilder mgr;
  auto& priority = mgr.AddGraphConfig();
  priority.option = QNN_GRAPH_CONFIG_OPTION_PRIORITY;
  priority.priority = QNN_PRIORITY_HIGH;

  auto configs = mgr.Configs();
  ASSERT_EQ(configs.size(), 2);
  ASSERT_NE(configs[0], nullptr);
  EXPECT_EQ(configs[1], nullptr);
  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);
  EXPECT_EQ(configs[0]->priority, QNN_PRIORITY_HIGH);
  // Note: customConfig and priority share a union, so customConfig is not
  // meaningful for a non-custom (priority) config.
}

TEST(GraphConfigBuilderTest, OrderingIsPreserved) {
  GraphConfigBuilder mgr;
  mgr.AddCustom<QnnHtpGraph_CustomConfig_t>(
      QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);  // custom first
  auto& priority = mgr.AddGraphConfig();  // bare second
  priority.option = QNN_GRAPH_CONFIG_OPTION_PRIORITY;

  auto configs = mgr.Configs();
  ASSERT_EQ(NonNullCount(configs), 2);
  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(configs[1]->option, QNN_GRAPH_CONFIG_OPTION_PRIORITY);
}

// Store() owns a side buffer whose address a custom config can point at; that
// pointer must stay valid after further additions.
TEST(GraphConfigBuilderTest, StoreOwnsSideBufferWithStableAddress) {
  GraphConfigBuilder mgr;
  auto& s = mgr.Store<std::string>("/tmp/dlc/graph.dlc");
  const std::string* addr = &s;

  // Adding more entries must not relocate the stored string.
  mgr.AddCustom<QnnHtpGraph_CustomConfig_t>(QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
  mgr.Store<QnnHtpGraph_CustomConfig_t>();

  EXPECT_EQ(&s, addr);
  EXPECT_EQ(s, "/tmp/dlc/graph.dlc");
}

// Returning the builder by value (as BuildGraphConfigs does) must not
// invalidate the customConfig pointers baked into the QnnGraph_Config_t entries
// — std::list nodes are not relocated on move, so the owned custom config keeps
// its address and the wrapping config keeps pointing at it.
TEST(GraphConfigBuilderTest, MoveKeepsCustomConfigPointersValid) {
  GraphConfigBuilder src;
  auto& cc = src.AddCustom<QnnHtpGraph_CustomConfig_t>(
      QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
  cc.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
  cc.vtcmSizeInMB = 7;
  const QnnHtpGraph_CustomConfig_t* addr_before = &cc;

  GraphConfigBuilder mgr = std::move(src);
  auto configs = mgr.Configs();
  ASSERT_EQ(NonNullCount(configs), 1);
  // The wrapping config still points at the SAME storage node address.
  EXPECT_EQ(configs[0]->customConfig, addr_before);
  const auto* wrapped =
      static_cast<const QnnHtpGraph_CustomConfig_t*>(configs[0]->customConfig);
  ASSERT_NE(wrapped, nullptr);
  EXPECT_EQ(wrapped->option, QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE);
  EXPECT_EQ(wrapped->vtcmSizeInMB, 7);
}

}  // namespace
}  // namespace qnn
