// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/lpai_backend.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "LPAI/QnnLpaiCommon.h"        // from @qairt
#include "LPAI/QnnLpaiGraph.h"         // from @qairt
#include "LPAI/QnnLpaiGraphPrepare.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnCommon.h"  // from @qairt
#include "QnnGraph.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {
namespace {

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

struct CapturedGraphFinalize {
  bool called = false;
  Qnn_GraphHandle_t graph = nullptr;
};

class LpaiBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // QNN API fields are C function pointers, so mocks must be captureless
    // lambdas; they reach per-test members through this back-pointer. Set
    // before api_ is wired.
    self_ = this;

    api_.backendCreate = [](Qnn_LogHandle_t, const QnnBackend_Config_t**,
                            Qnn_BackendHandle_t* backend) -> Qnn_ErrorHandle_t {
      *backend = self_->fake_handle_;
      return QNN_SUCCESS;
    };
    api_.backendFree = [](Qnn_BackendHandle_t) -> Qnn_ErrorHandle_t {
      return QNN_SUCCESS;
    };
    api_.graphSetConfig =
        [](Qnn_GraphHandle_t graph,
           const QnnGraph_Config_t** config) -> Qnn_ErrorHandle_t {
      auto& t = *self_;
      if (t.set_config_fail_) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
      t.captured_.called = true;
      t.captured_.graph = graph;
      t.captured_.configs.clear();
      if (config) {
        for (size_t i = 0; config[i] != nullptr; ++i) {
          if (config[i]->option != QNN_GRAPH_CONFIG_OPTION_CUSTOM ||
              config[i]->customConfig == nullptr) {
            continue;
          }
          const auto* src = static_cast<QnnLpaiGraph_CustomConfig_t*>(
              config[i]->customConfig);
          CapturedCustomConfig& cc = t.captured_.configs.emplace_back();
          cc.option = src->option;
          // Copy the leaf payload by value while src->config is still alive.
          if (src->option == QNN_LPAI_GRAPH_SET_CFG_PERF_CFG &&
              src->config != nullptr) {
            cc.perf = *static_cast<QnnLpaiGraph_PerfCfg_t*>(src->config);
          } else if (src->option == QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY &&
                     src->config != nullptr) {
            cc.affinity =
                *static_cast<QnnLpaiGraph_CoreAffinity_t*>(src->config);
          }
        }
      }
      return QNN_SUCCESS;
    };
    api_.graphFinalize = [](Qnn_GraphHandle_t graph, Qnn_ProfileHandle_t,
                            Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
      self_->captured_finalize_.called = true;
      self_->captured_finalize_.graph = graph;
      return QNN_SUCCESS;
    };

    backend_.emplace(&api_);
  }

  void TearDown() override { self_ = nullptr; }

  static LpaiBackendTest* self_;

  int fake_handle_storage_ = 0;
  void* const fake_handle_ = &fake_handle_storage_;
  QNN_INTERFACE_VER_TYPE api_{};
  std::optional<LpaiBackend> backend_;
  CapturedGraphConfig captured_;
  CapturedGraphFinalize captured_finalize_;
  bool set_config_fail_ = false;
};

LpaiBackendTest* LpaiBackendTest::self_ = nullptr;

TEST_F(LpaiBackendTest, GetExpectedBackendVersion) {
  auto version = LpaiBackend::GetExpectedBackendVersion();
  EXPECT_EQ(version.major, QNN_LPAI_API_VERSION_MAJOR);
  EXPECT_EQ(version.minor, QNN_LPAI_API_VERSION_MINOR);
  EXPECT_EQ(version.patch, QNN_LPAI_API_VERSION_PATCH);
}

TEST_F(LpaiBackendTest, InitSucceedsForSocWithLpai) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);
  EXPECT_TRUE(backend_->Init(options, FindOrCreateSocInfo("SM8850")));
}

#if !defined(__ANDROID__)
TEST_F(LpaiBackendTest, InitFailsForSocWithoutLpaiEntry) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);
  // A known-good SoC that has no LPAI capability entry.
  EXPECT_FALSE(backend_->Init(options, FindOrCreateSocInfo("SM8550")));
}

// Init ignores soc_info on device (see implementation), skipping this test.
TEST_F(LpaiBackendTest, InitFailsWithoutKnownLpaiHardwareVersion) {
  Options options;
  options.SetLogLevel(LogLevel::kOff);
  EXPECT_FALSE(backend_->Init(options, std::nullopt));
}
#endif  // !defined(__ANDROID__)

TEST_F(LpaiBackendTest, BuildGraphConfigsReturnsPrepareConfig) {
  Options options;
  auto builder = backend_->BuildGraphConfigs(options, "graph");
  auto configs = builder.GetNullTerminatedConfigs();

  ASSERT_EQ(configs.size(), 2u);
  ASSERT_NE(configs[0], nullptr);
  EXPECT_EQ(configs[1], nullptr);
  EXPECT_EQ(configs[0]->option, QNN_GRAPH_CONFIG_OPTION_CUSTOM);

  auto* custom_config =
      static_cast<QnnLpaiGraph_CustomConfig_t*>(configs[0]->customConfig);
  ASSERT_NE(custom_config, nullptr);
  EXPECT_EQ(custom_config->option, QNN_LPAI_GRAPH_SET_CFG_PREPARE);

  auto* prepare =
      static_cast<QnnLpaiGraph_CustomConfigPrepare_t*>(custom_config->config);
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

  GraphConfigContext ctx{fake_handle_, "graph", nullptr};
  EXPECT_TRUE(backend_->ConfigureGraphAfterRetrieve(ctx, options));

  // Both the perf config and the core-affinity config are pushed.
  ASSERT_TRUE(captured_.called);
  EXPECT_EQ(captured_.graph, fake_handle_);
  ASSERT_EQ(captured_.configs.size(), 2u);

  const auto& perf = captured_.configs[0];
  EXPECT_EQ(perf.option, QNN_LPAI_GRAPH_SET_CFG_PERF_CFG);
  ASSERT_TRUE(perf.perf.has_value());
  EXPECT_EQ(perf.perf->fps, 30u);
  EXPECT_EQ(perf.perf->ftrtRatio, 5u);
  EXPECT_EQ(perf.perf->clientType,
            QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_NON_REAL_TIME);

  const auto& affinity = captured_.configs[1];
  EXPECT_EQ(affinity.option, QNN_LPAI_GRAPH_SET_CFG_CORE_AFFINITY);
  ASSERT_TRUE(affinity.affinity.has_value());
  EXPECT_EQ(affinity.affinity->affinity, QNN_LPAI_GRAPH_CORE_AFFINITY_HARD);
  EXPECT_EQ(affinity.affinity->coreSelection, 0x01u);

  // The graph is re-finalized after the configs are pushed.
  EXPECT_TRUE(captured_finalize_.called);
  EXPECT_EQ(captured_finalize_.graph, fake_handle_);
}

TEST_F(LpaiBackendTest, ConfigureGraphAfterRetrieveUsesSdkDefaults) {
  // No LPAI options set: the SDK initializer defaults should be used.
  Options options;
  GraphConfigContext ctx{fake_handle_, "graph", nullptr};
  EXPECT_TRUE(backend_->ConfigureGraphAfterRetrieve(ctx, options));

  ASSERT_EQ(captured_.configs.size(), 2u);
  const auto& perf = captured_.configs[0];
  ASSERT_TRUE(perf.perf.has_value());
  EXPECT_EQ(perf.perf->fps, 1u);         // QNN_LPAI_GRAPH_PERF_CFG_INIT default
  EXPECT_EQ(perf.perf->ftrtRatio, 10u);  // QNN_LPAI_GRAPH_PERF_CFG_INIT default
  EXPECT_EQ(perf.perf->clientType, QNN_LPAI_GRAPH_CLIENT_PERF_TYPE_REAL_TIME);

  EXPECT_TRUE(captured_finalize_.called);
}

TEST_F(LpaiBackendTest, ConfigureGraphAfterRetrieveFailsOnSetConfigError) {
  set_config_fail_ = true;
  Options options;
  GraphConfigContext ctx{fake_handle_, "graph", nullptr};
  EXPECT_FALSE(backend_->ConfigureGraphAfterRetrieve(ctx, options));
}

}  // namespace
}  // namespace qnn
