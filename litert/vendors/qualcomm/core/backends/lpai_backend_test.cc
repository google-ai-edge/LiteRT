// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/lpai_backend.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "LPAI/QnnLpaiGraph.h"  // from @qairt
#include "LPAI/QnnLpaiGraphPrepare.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {
namespace {

int kFakeHandleStorage = 0;
void* const kFakeHandle = &kFakeHandleStorage;

// One LPAI custom config captured by value during the graphSetConfig call.
// The perf / affinity payload is copied out while the source pointers are still
// valid, so nothing dangles after the call-scoped builder is freed.
struct CapturedCustomConfig {
  uint32_t option = QNN_LPAI_GRAPH_SET_CFG_UNDEFINED;
  std::optional<QnnLpaiGraph_PerfCfg_t> perf;
  std::optional<QnnLpaiGraph_CoreAffinity_t> affinity;
};

struct CapturedGraphConfig {
  bool called = false;
  Qnn_GraphHandle_t graph = nullptr;
  std::vector<CapturedCustomConfig> configs;
};
absl::NoDestructor<CapturedGraphConfig> g_captured;
bool g_set_config_fail = false;

Qnn_ErrorHandle_t MockBackendCreate(Qnn_LogHandle_t,
                                    const QnnBackend_Config_t**,
                                    Qnn_BackendHandle_t* backend) {
  *backend = kFakeHandle;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t MockBackendFree(Qnn_BackendHandle_t) { return QNN_SUCCESS; }

Qnn_ErrorHandle_t MockGraphSetConfig(Qnn_GraphHandle_t graph,
                                     const QnnGraph_Config_t** config) {
  if (g_set_config_fail) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  g_captured->called = true;
  g_captured->graph = graph;
  g_captured->configs.clear();
  if (config) {
    for (size_t i = 0; config[i] != nullptr; ++i) {
      if (config[i]->option != QNN_GRAPH_CONFIG_OPTION_CUSTOM ||
          config[i]->customConfig == nullptr) {
        continue;
      }
      const auto* src =
          static_cast<QnnLpaiGraph_CustomConfig_t*>(config[i]->customConfig);
      CapturedCustomConfig& cc = g_captured->configs.emplace_back();
      cc.option = src->option;
      // Copy the leaf payload by value while src->config is still alive.
      if (src->option == QNN_LPAI_GRAPH_SET_CFG_PERF_CFG &&
          src->config != nullptr) {
        cc.perf = *static_cast<QnnLpaiGraph_PerfCfg_t*>(src->config);
      } else if (src->option == QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY &&
                 src->config != nullptr) {
        cc.affinity = *static_cast<QnnLpaiGraph_CoreAffinity_t*>(src->config);
      }
    }
  }
  return QNN_SUCCESS;
}

struct CapturedGraphFinalize {
  bool called = false;
  Qnn_GraphHandle_t graph = nullptr;
};
absl::NoDestructor<CapturedGraphFinalize> g_captured_finalize;

Qnn_ErrorHandle_t MockGraphFinalize(Qnn_GraphHandle_t graph,
                                    Qnn_ProfileHandle_t, Qnn_SignalHandle_t) {
  g_captured_finalize->called = true;
  g_captured_finalize->graph = graph;
  return QNN_SUCCESS;
}

QNN_INTERFACE_VER_TYPE MakeFakeApi() {
  QNN_INTERFACE_VER_TYPE api{};
  api.backendCreate = MockBackendCreate;
  api.backendFree = MockBackendFree;
  api.graphSetConfig = MockGraphSetConfig;
  api.graphFinalize = MockGraphFinalize;
  return api;
}

SocInfo SocWithLpai() {
  return SocInfo("LPAI_SOC", SnapdragonModel::SM8850, DspArch::V81, 8,
                 LpaiHardwareVersion::kV6);
}

class LpaiBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    *g_captured = CapturedGraphConfig{};
    *g_captured_finalize = CapturedGraphFinalize{};
    g_set_config_fail = false;
  }

  QNN_INTERFACE_VER_TYPE api_ = MakeFakeApi();
  LpaiBackend backend_{&api_};
};

TEST_F(LpaiBackendTest, GetExpectedBackendVersion) {
  auto version = LpaiBackend::GetExpectedBackendVersion();
  EXPECT_EQ(version.major, QNN_LPAI_API_VERSION_MAJOR);
  EXPECT_EQ(version.minor, QNN_LPAI_API_VERSION_MINOR);
  EXPECT_EQ(version.patch, QNN_LPAI_API_VERSION_PATCH);
}

TEST_F(LpaiBackendTest, InitSucceedsForSocWithLpai) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);
  EXPECT_TRUE(backend_.Init(options, SocWithLpai()));
}

TEST_F(LpaiBackendTest, InitFailsWithoutKnownLpaiHardwareVersion) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);
  EXPECT_FALSE(backend_.Init(options, std::nullopt));
}

TEST_F(LpaiBackendTest, BuildGraphConfigsReturnsPrepareConfig) {
  Options options;
  auto builder = backend_.BuildGraphConfigs(options, "graph");
  auto configs = builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.size(), 2u);
  ASSERT_NE(configs[0], nullptr);
  EXPECT_EQ(configs[1], nullptr);
  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);

  auto* custom_config =
      static_cast<QnnLpaiGraph_CustomConfig_t*>(configs[0]->customConfig);
  ASSERT_NE(custom_config, nullptr);
  EXPECT_EQ(custom_config->option, QNN_LPAI_GRAPH_SET_CFG_PREPARE);

  auto* prepare = static_cast<QnnLpaiGraph_CustomConfigPrepare_t*>(
      custom_config->config);
  ASSERT_NE(prepare, nullptr);
  EXPECT_STREQ(prepare->enableCoreSelection, "0,1");
}

TEST_F(LpaiBackendTest, ConfigureGraphAfterRetrievePushesConfigsAndFinalizes) {
  Options options;
  options.SetLpaiFps(30);
  options.SetLpaiFtrtRatio(5);
  options.SetLpaiClientPerfType(LpaiClientPerfType::kNonRealTime);
  options.SetLpaiCoreAffinityType(LpaiCoreAffinityType::kHard);
  options.SetLpaiCoreSelection(0x01);

  GraphConfigContext ctx{kFakeHandle, "graph", nullptr};
  EXPECT_TRUE(backend_.ConfigureGraphAfterRetrieve(ctx, options));

  // Both the perf config and the core-affinity config are pushed.
  ASSERT_TRUE(g_captured->called);
  EXPECT_EQ(g_captured->graph, kFakeHandle);
  ASSERT_EQ(g_captured->configs.size(), 2u);

  const auto& perf = g_captured->configs[0];
  EXPECT_EQ(perf.option, QNN_LPAI_GRAPH_SET_CFG_PERF_CFG);
  ASSERT_TRUE(perf.perf.has_value());
  EXPECT_EQ(perf.perf->fps, 30u);
  EXPECT_EQ(perf.perf->ftrtRatio, 5u);
  EXPECT_EQ(perf.perf->clientType, QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_NON_REAL_TIME);

  const auto& affinity = g_captured->configs[1];
  EXPECT_EQ(affinity.option, QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY);
  ASSERT_TRUE(affinity.affinity.has_value());
  EXPECT_EQ(affinity.affinity->affinity, QNN_LPAI_GRAPH_CORE_AFFINITY_HARD);
  EXPECT_EQ(affinity.affinity->coreSelection, 0x01u);

  // The graph is re-finalized after the configs are pushed.
  EXPECT_TRUE(g_captured_finalize->called);
  EXPECT_EQ(g_captured_finalize->graph, kFakeHandle);
}

TEST_F(LpaiBackendTest, ConfigureGraphAfterRetrieveUsesSdkDefaults) {
  // No LPAI options set: the SDK initializer defaults should be used.
  Options options;
  GraphConfigContext ctx{kFakeHandle, "graph", nullptr};
  EXPECT_TRUE(backend_.ConfigureGraphAfterRetrieve(ctx, options));

  ASSERT_EQ(g_captured->configs.size(), 2u);
  const auto& perf = g_captured->configs[0];
  ASSERT_TRUE(perf.perf.has_value());
  EXPECT_EQ(perf.perf->fps, 1u);  // QNN_LPAI_GRAPH_PERF_CFG_INIT default
  EXPECT_EQ(perf.perf->ftrtRatio, 10u);  // QNN_LPAI_GRAPH_PERF_CFG_INIT default
  EXPECT_EQ(perf.perf->clientType, QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_REAL_TIME);

  EXPECT_TRUE(g_captured_finalize->called);
}

TEST_F(LpaiBackendTest, ConfigureGraphAfterRetrieveFailsOnSetConfigError) {
  g_set_config_fail = true;
  Options options;
  GraphConfigContext ctx{kFakeHandle, "graph", nullptr};
  EXPECT_FALSE(backend_.ConfigureGraphAfterRetrieve(ctx, options));
}

}  // namespace
}  // namespace qnn
