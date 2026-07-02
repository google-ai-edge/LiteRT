// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/lpai_backend.h"

#include <cstdint>
#include <list>
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

// Opaque non-null handles used to stand in for the real QNN objects so the
// backend's RAII unique_ptrs treat creation as successful.
int kFakeHandleStorage = 0;
void* const kFakeHandle = &kFakeHandleStorage;

// Captured graphSetConfig arguments from the most recent call.
//
// graphSetConfig consumes its configs synchronously, and the backend owns the
// config storage only for the duration of the call (a call-scoped
// GraphConfigBuilder). So the mock must DEEP-copy the nested LPAI payloads
// during the call into stable storage owned here, and repoint the captured
// configs at that storage — inspecting the original pointers after the call
// returns would dangle.
struct CapturedGraphConfig {
  bool called = false;
  Qnn_GraphHandle_t graph = nullptr;
  std::vector<QnnGraph_Config_t> configs;
  // Stable backing storage for the deep-copied custom configs + their nested
  // sub-configs, kept in std::list so addresses stay stable as we append.
  std::list<QnnLpaiGraph_CustomConfig_t> custom_configs;
  std::list<QnnLpaiGraph_PerfCfg_t> perf_cfgs;
  std::list<QnnLpaiGraph_CoreAffinity_t> core_affinities;
};
absl::NoDestructor<CapturedGraphConfig> g_captured;

Qnn_ErrorHandle_t MockLogCreate(QnnLog_Callback_t, QnnLog_Level_t,
                                Qnn_LogHandle_t* log) {
  *log = kFakeHandle;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t MockLogFree(Qnn_LogHandle_t) { return QNN_SUCCESS; }

Qnn_ErrorHandle_t MockBackendCreate(Qnn_LogHandle_t,
                                    const QnnBackend_Config_t**,
                                    Qnn_BackendHandle_t* backend) {
  *backend = kFakeHandle;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t MockBackendFree(Qnn_BackendHandle_t) { return QNN_SUCCESS; }

Qnn_ErrorHandle_t MockDeviceCreate(Qnn_LogHandle_t, const QnnDevice_Config_t**,
                                   Qnn_DeviceHandle_t* device) {
  *device = kFakeHandle;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t MockDeviceFree(Qnn_DeviceHandle_t) { return QNN_SUCCESS; }

Qnn_ErrorHandle_t MockGraphSetConfig(Qnn_GraphHandle_t graph,
                                     const QnnGraph_Config_t** config) {
  g_captured->called = true;
  g_captured->graph = graph;
  g_captured->configs.clear();
  g_captured->custom_configs.clear();
  g_captured->perf_cfgs.clear();
  g_captured->core_affinities.clear();
  if (config) {
    for (size_t i = 0; config[i] != nullptr; ++i) {
      QnnGraph_Config_t cfg = *config[i];
      // Deep-copy the custom config and its nested sub-config into stable
      // storage, repointing the copy at it (the source is call-scoped).
      if (cfg.option == QNN_GRAPH_CONFIG_OPTION_CUSTOM &&
          cfg.customConfig != nullptr) {
        auto* src = static_cast<QnnLpaiGraph_CustomConfig_t*>(cfg.customConfig);
        auto& cc = g_captured->custom_configs.emplace_back(*src);
        if (src->option == QNN_LPAI_GRAPH_SET_CFG_PERF_CFG &&
            src->config != nullptr) {
          auto& perf = g_captured->perf_cfgs.emplace_back(
              *static_cast<QnnLpaiGraph_PerfCfg_t*>(src->config));
          cc.config = &perf;
        } else if (src->option == QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY &&
                   src->config != nullptr) {
          auto& affinity = g_captured->core_affinities.emplace_back(
              *static_cast<QnnLpaiGraph_CoreAffinity_t*>(src->config));
          cc.config = &affinity;
        }
        cfg.customConfig = &cc;
      }
      g_captured->configs.emplace_back(cfg);
    }
  }
  return QNN_SUCCESS;
}

// Captured graphFinalize arguments from the most recent call.
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
  api.logCreate = MockLogCreate;
  api.logFree = MockLogFree;
  api.backendCreate = MockBackendCreate;
  api.backendFree = MockBackendFree;
  api.deviceCreate = MockDeviceCreate;
  api.deviceFree = MockDeviceFree;
  api.graphSetConfig = MockGraphSetConfig;
  api.graphFinalize = MockGraphFinalize;
  return api;
}

SocInfo SocWithLpai() {
  return SocInfo("LPAI_SOC", SnapdragonModel::SM8850, DspArch::V81, 8,
                 LpaiHardwareVersion::kV6);
}

TEST(LpaiBackendTest, GetExpectedBackendVersion) {
  auto version = LpaiBackend::GetExpectedBackendVersion();
  EXPECT_EQ(version.major, QNN_LPAI_API_VERSION_MAJOR);
  EXPECT_EQ(version.minor, QNN_LPAI_API_VERSION_MINOR);
  EXPECT_EQ(version.patch, QNN_LPAI_API_VERSION_PATCH);
}

TEST(LpaiBackendTest, InitSucceedsForSocWithLpai) {
  auto api = MakeFakeApi();
  LpaiBackend backend(&api);

  Options options;
  options.SetLogLevel(LogLevel::kOff);
  EXPECT_TRUE(backend.Init(options, SocWithLpai()));
}

TEST(LpaiBackendTest, BuildGraphConfigsReturnsPrepareConfig) {
  auto api = MakeFakeApi();
  LpaiBackend backend(&api);

  Options options;
  auto mgr = backend.BuildGraphConfigs(options, "graph");
  auto configs = mgr.Configs();

  // The span is null-terminated, so it holds [prepare_config, nullptr].
  ASSERT_EQ(configs.size(), 2u);
  ASSERT_NE(configs[0], nullptr);
  EXPECT_EQ(configs[1], nullptr);
  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);

  auto* custom_config =
      static_cast<QnnLpaiGraph_CustomConfig_t*>(configs[0]->customConfig);
  ASSERT_NE(custom_config, nullptr);
  EXPECT_EQ(custom_config->option, QNN_LPAI_GRAPH_SET_CFG_PREPARE);
}

TEST(LpaiBackendTest, ConfigureGraphAfterRetrievePushesConfigsAndFinalizes) {
  auto api = MakeFakeApi();
  LpaiBackend backend(&api);
  *g_captured = CapturedGraphConfig{};
  *g_captured_finalize = CapturedGraphFinalize{};

  Options options;
  options.SetLpaiFps(30);
  options.SetLpaiFtrtRatio(5);
  options.SetLpaiClientPerfType(LpaiClientPerfType::kNonRealTime);
  options.SetLpaiCoreAffinityType(LpaiCoreAffinityType::kHard);
  options.SetLpaiCoreSelection(0x01);

  GraphConfigContext ctx{kFakeHandle, "graph", nullptr};
  EXPECT_TRUE(backend.ConfigureGraphAfterRetrieve(ctx, options));

  // Both the perf config and the core-affinity config are pushed.
  ASSERT_TRUE(g_captured->called);
  EXPECT_EQ(g_captured->graph, kFakeHandle);
  ASSERT_EQ(g_captured->configs.size(), 2u);
  EXPECT_EQ(g_captured->configs[0].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);
  EXPECT_EQ(g_captured->configs[1].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);

  auto* perf_custom = static_cast<QnnLpaiGraph_CustomConfig_t*>(
      g_captured->configs[0].customConfig);
  ASSERT_NE(perf_custom, nullptr);
  EXPECT_EQ(perf_custom->option, QNN_LPAI_GRAPH_SET_CFG_PERF_CFG);
  auto* perf_cfg = static_cast<QnnLpaiGraph_PerfCfg_t*>(perf_custom->config);
  ASSERT_NE(perf_cfg, nullptr);
  EXPECT_EQ(perf_cfg->fps, 30u);
  EXPECT_EQ(perf_cfg->ftrtRatio, 5u);
  EXPECT_EQ(perf_cfg->clientType,
            QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_NON_REAL_TIME);

  auto* affinity_custom = static_cast<QnnLpaiGraph_CustomConfig_t*>(
      g_captured->configs[1].customConfig);
  ASSERT_NE(affinity_custom, nullptr);
  EXPECT_EQ(affinity_custom->option, QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY);
  auto* core_affinity =
      static_cast<QnnLpaiGraph_CoreAffinity_t*>(affinity_custom->config);
  ASSERT_NE(core_affinity, nullptr);
  EXPECT_EQ(core_affinity->affinity, QNN_LPAI_GRAPH_CORE_AFFINITY_HARD);
  EXPECT_EQ(core_affinity->coreSelection, 0x01u);

  // The graph is re-finalized after the configs are pushed.
  EXPECT_TRUE(g_captured_finalize->called);
  EXPECT_EQ(g_captured_finalize->graph, kFakeHandle);
}

TEST(LpaiBackendTest, ConfigureGraphAfterRetrieveUsesSdkDefaults) {
  auto api = MakeFakeApi();
  LpaiBackend backend(&api);
  *g_captured = CapturedGraphConfig{};
  *g_captured_finalize = CapturedGraphFinalize{};

  // No LPAI options set: the SDK initializer defaults should be used.
  Options options;
  GraphConfigContext ctx{kFakeHandle, "graph", nullptr};
  EXPECT_TRUE(backend.ConfigureGraphAfterRetrieve(ctx, options));

  ASSERT_EQ(g_captured->configs.size(), 2u);
  auto* perf_custom = static_cast<QnnLpaiGraph_CustomConfig_t*>(
      g_captured->configs[0].customConfig);
  auto* perf_cfg = static_cast<QnnLpaiGraph_PerfCfg_t*>(perf_custom->config);
  ASSERT_NE(perf_cfg, nullptr);
  EXPECT_EQ(perf_cfg->fps, 1u);  // QNN_LPAI_GRAPH_PERF_CFG_INIT default
  EXPECT_EQ(perf_cfg->ftrtRatio, 10u);  // QNN_LPAI_GRAPH_PERF_CFG_INIT default
  EXPECT_EQ(perf_cfg->clientType, QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_REAL_TIME);

  EXPECT_TRUE(g_captured_finalize->called);
}

// Regression: repeatedly invoking the graph-config hooks must not grow storage.
// Each call now returns a fresh, independently-owned builder, so the returned
// config counts stay constant no matter how many graphs a backend configures.
TEST(LpaiBackendTest, BuildGraphConfigsBoundedAcrossRepeatedCalls) {
  auto api = MakeFakeApi();
  LpaiBackend backend(&api);

  Options options;
  for (int i = 0; i < 5; ++i) {
    auto mgr = backend.BuildGraphConfigs(options, "graph");
    auto configs = mgr.Configs();
    // Always [prepare_config, nullptr] — never accumulates.
    ASSERT_EQ(configs.size(), 2u) << "iteration " << i;
    ASSERT_NE(configs[0], nullptr) << "iteration " << i;
    EXPECT_EQ(configs[1], nullptr) << "iteration " << i;
    auto* custom_config =
        static_cast<QnnLpaiGraph_CustomConfig_t*>(configs[0]->customConfig);
    ASSERT_NE(custom_config, nullptr) << "iteration " << i;
    EXPECT_EQ(custom_config->option, QNN_LPAI_GRAPH_SET_CFG_PREPARE)
        << "iteration " << i;
  }
}

TEST(LpaiBackendTest, ConfigureGraphAfterRetrieveBoundedAcrossRepeatedCalls) {
  auto api = MakeFakeApi();
  LpaiBackend backend(&api);

  Options options;
  options.SetLpaiFps(30);
  options.SetLpaiCoreSelection(0x01);
  GraphConfigContext ctx{kFakeHandle, "graph", nullptr};

  for (int i = 0; i < 5; ++i) {
    *g_captured = CapturedGraphConfig{};
    *g_captured_finalize = CapturedGraphFinalize{};
    EXPECT_TRUE(backend.ConfigureGraphAfterRetrieve(ctx, options))
        << "iteration " << i;

    // Exactly the perf + core-affinity configs are pushed each time — the
    // storage lists never accumulate stale entries across calls.
    ASSERT_EQ(g_captured->configs.size(), 2u) << "iteration " << i;
    EXPECT_EQ(g_captured->configs[0].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM)
        << "iteration " << i;
    EXPECT_EQ(g_captured->configs[1].option, QNN_GRAPH_CONFIG_OPTION_CUSTOM)
        << "iteration " << i;

    // The values still reflect the options after the lists are reset.
    auto* perf_custom = static_cast<QnnLpaiGraph_CustomConfig_t*>(
        g_captured->configs[0].customConfig);
    ASSERT_NE(perf_custom, nullptr) << "iteration " << i;
    auto* perf_cfg = static_cast<QnnLpaiGraph_PerfCfg_t*>(perf_custom->config);
    ASSERT_NE(perf_cfg, nullptr) << "iteration " << i;
    EXPECT_EQ(perf_cfg->fps, 30u) << "iteration " << i;
  }
}

}  // namespace
}  // namespace qnn
