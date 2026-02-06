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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMPILE_FIXTURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMPILE_FIXTURE_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/capture_common.h"
#include "litert/ats/common.h"
#include "litert/ats/compile_capture.h"
#include "litert/ats/configure.h"
#include "litert/c/internal/litert_logging.h"  // IWYU pragma: keep
#include "litert/cc/internal/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"

namespace litert::testing {

using internal::ApplyPlugin;
using internal::CompilerPlugin;
using internal::IsDir;
using ::testing::RegisterTest;

// Fixture for tests that run aot flow on a graph and emit the result as file.
class AtsCompileTest : public ::testing::Test {
 public:
  using Capture = CompileCapture;

  static constexpr absl::string_view Name() { return "compile"; }

  static void Register(TestGraph::Ptr graph, const AtsConf& conf,
                       const TestNames& names, Capture::Entry& cap) {
    RegisterTest(names.suite.c_str(), names.test.c_str(), nullptr, nullptr,
                 __FILE__, __LINE__,
                 [graph = std::move(graph), &conf = std::as_const(conf), names,
                  &cap]() mutable {
                   return new AtsCompileTest(std::move(graph), conf, names,
                                             cap);
                 });
  }

  void SetUp() override {
    cap_.model.SetFields(names_, graph_->Graph());
    cap_.accelerator.SetFields(conf_);
  }

  void TestBody() override {
    auto start = cap_.compilation_time.Start();
    auto stat = ApplyPlugin(conf_.Plugin()->get(), graph_->Graph());
    cap_.compilation_time.Stop(start);
    cap_.compilation_detail.SetFields(conf_, graph_->Graph(), !stat.HasValue());
    ASSERT_TRUE(stat);
    ASSERT_TRUE(internal::HasAnyCompiled(graph_->Graph()));
  }

  void TearDown() override {
    if (HasFailure()) {
      return;
    }
    LITERT_ASSERT_OK(
        conf_.SaveModel(names_.report_id, std::move(graph_->Graph())));
  }

 private:
  AtsCompileTest(TestGraph::Ptr graph, const AtsConf& conf,
                 const TestNames& names, Capture::Entry& cap)
      : graph_(std::move(graph)),
        conf_(conf),
        names_(std::move(names)),
        cap_(cap) {}

  TestGraph::Ptr graph_;
  const AtsConf& conf_;

  TestNames names_;
  Capture::Entry& cap_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_COMPILE_FIXTURE_H_
