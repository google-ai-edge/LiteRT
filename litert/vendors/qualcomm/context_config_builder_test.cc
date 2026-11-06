// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/context_config_builder.h"

#include <utility>

#include <gtest/gtest.h>

namespace qnn {
namespace {

TEST(ContextConfigBuilderTest, BuildsOwnedTypedCustomConfig) {
  QnnContextConfigsBuilder builder;
  builder.EmplaceCustomConfig<HtpFileReadMemoryBudgetConfig>(16);
  const auto configs = std::move(builder).Build();

  ASSERT_EQ(configs.Size(), 1);
  ASSERT_NE(configs.Data()[0], nullptr);
  EXPECT_EQ(configs.Data()[0]->option, QNN_CONTEXT_CONFIG_OPTION_CUSTOM);
  const auto* budget = static_cast<const QnnHtpContext_CustomConfig_t*>(
      configs.Data()[0]->customConfig);
  ASSERT_NE(budget, nullptr);
  EXPECT_EQ(budget->option,
            QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET);
  EXPECT_EQ(budget->fileReadMemoryBudgetInMb, 16);
  EXPECT_EQ(configs.Data()[1], nullptr);
}

TEST(ContextConfigBuilderTest, BuildsNullTerminatedContextConfigs) {
  QnnContextConfigsBuilder builder;
  QnnContext_Config_t priority = QNN_CONTEXT_CONFIG_INIT;
  priority.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
  priority.priority = QNN_PRIORITY_HIGH;
  builder.AddContextConfig(priority)
      .EmplaceCustomConfig<HtpInitAccelerationConfig>(true);
  const auto configs = std::move(builder).Build();

  ASSERT_EQ(configs.Size(), 2);
  EXPECT_EQ(configs.Data()[0]->option, QNN_CONTEXT_CONFIG_OPTION_PRIORITY);
  EXPECT_EQ(configs.Data()[0]->priority, QNN_PRIORITY_HIGH);
  EXPECT_EQ(configs.Data()[1]->option, QNN_CONTEXT_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(configs.Data()[2], nullptr);
}

}  // namespace
}  // namespace qnn
