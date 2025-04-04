// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include "litert/cc/litert_shared_library.h"
#include "litert/vendors/qualcomm/common.h"
#include "litert/vendors/qualcomm/qnn_manager.h"

namespace litert::test {
namespace {

static constexpr absl::string_view kDispatch = "libLiteRtDispatch_Qualcomm.so";
static constexpr absl::string_view kPlugin =
    "libLiteRtCompilerPlugin_Qualcomm.so";

using ::litert::qnn::QnnManager;

TEST(QnnSmokeTest, LoadLibsFromEnvPath) {
  auto lib_htp = SharedLibrary::Load(kLibQnnHtpSo, RtldFlags::Default());
  ASSERT_TRUE(lib_htp);

  auto lib_system = SharedLibrary::Load(kLibQnnSystemSo, RtldFlags::Default());
  ASSERT_TRUE(lib_system);

  auto lib_dispatch = SharedLibrary::Load(kDispatch, RtldFlags::Default());
  ASSERT_TRUE(lib_dispatch);

  auto lib_plugin = SharedLibrary::Load(kPlugin, RtldFlags::Default());
  ASSERT_TRUE(lib_plugin);
}

TEST(QnnSmokeTest, QnnManagerCreate) {
  auto configs = QnnManager::DefaultBackendConfigs();
  auto qnn = QnnManager::Create(configs);
  ASSERT_TRUE(qnn);
}

}  // namespace
}  // namespace litert::test
