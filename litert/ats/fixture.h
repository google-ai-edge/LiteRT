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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_FIXTURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_FIXTURE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/configure.h"
#include "litert/ats/executor.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"  // IWYU pragma: keep
#include "litert/cc/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "litert/test/simple_buffer.h"

namespace litert::testing {

using ::testing::RegisterTest;
using ::testing::litert::MeanSquaredErrorLt;

class AtsTest : public RngTest {
 public:
  template <typename T>
  using BufferView = typename SimpleBuffer::CView<T>;

  static void Register(size_t test_id, TestGraph::Ptr graph,
                       absl::string_view family_name, const AtsConf& conf,
                       bool always_reg = false) {
    const auto suite_name = absl::StrFormat("ats_%lu_%s", test_id, family_name);
    const auto test_name =
        absl::StrFormat("%v", graph->Graph().Subgraph(0).Ops());

    if (!always_reg &&
        !conf.ShouldRegister(absl::StrCat(suite_name, test_name))) {
      return;
    }

    RegisterTest(suite_name.c_str(), test_name.c_str(), nullptr, nullptr,
                 __FILE__, __LINE__,
                 [graph = std::move(graph), conf]() mutable {
                   return new AtsTest(std::move(graph), conf);
                 });
  }

  void TestBody() override {
    auto device = this->TracedDevice(conf_.DataSeed());
    LITERT_ASSERT_OK_AND_ASSIGN(auto inputs, MakeInputs(device));
    LITERT_ASSERT_OK_AND_ASSIGN(auto ref, Reference(inputs));
    LITERT_ASSERT_OK_AND_ASSIGN(auto actual, Actual(inputs));
    CheckOutputs(actual, ref);
  }

 private:
  Expected<CompiledModelExecutor::Ptr> MakeExecutor() const {
    if (conf_.IsNpu()) {
      LITERT_ASSIGN_OR_RETURN(
          auto exec, NpuCompiledModelExecutor::Create(
                         Graph(), conf_.DispatchDir(), conf_.PluginDir()));
      return std::make_unique<CompiledModelExecutor>(std::move(exec));

    } else if (conf_.IsCpu()) {
      LITERT_ASSIGN_OR_RETURN(auto exec,
                              CpuCompiledModelExecutor::Create(Graph()));
      return std::make_unique<CompiledModelExecutor>(std::move(exec));
    }

    return Error(kLiteRtStatusErrorInvalidArgument, "Unsupported backend");
  }

  template <typename Rng>
  Expected<VarBuffers> MakeInputs(Rng& device) const {
    return graph_->MakeInputs(device, conf_.DataBuilder());
  }

  Expected<VarBuffers> Actual(const VarBuffers& inputs) const {
    LITERT_ASSIGN_OR_RETURN(auto exec, MakeExecutor());
    LITERT_ASSIGN_OR_RETURN(auto actual, exec->Run(inputs));
    return actual;
  }

  Expected<VarBuffers> Reference(const VarBuffers& inputs) const {
    return graph_->HasReference() ? CustomReference(inputs)
                                  : CpuReference(inputs);
  }

  Expected<VarBuffers> CustomReference(const VarBuffers& inputs) const {
    LITERT_ASSIGN_OR_RETURN(auto outputs, MakeOutputs());
    LITERT_RETURN_IF_ERROR(graph_->Reference(inputs, outputs));
    return outputs;
  }

  Expected<VarBuffers> CpuReference(const VarBuffers& inputs) const {
    return Error(kLiteRtStatusErrorInvalidArgument, "TODO");
  }

  Expected<VarBuffers> MakeOutputs() const {
    return SimpleBuffer::LikeSignature(Graph().Subgraph(0).Outputs().begin(),
                                       Graph().Subgraph(0).Outputs().end());
  }

  void CheckOutputs(const VarBuffers& actual, const VarBuffers& ref) {
    ASSERT_EQ(actual.size(), ref.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_EQ(actual[i].Type(), ref[i].Type());
      if (actual[i].Type().ElementType() == ElementType::Float32) {
        CheckOutputImpl(actual[i].AsView<float>(), ref[i].AsView<float>());
      } else if (actual[i].Type().ElementType() == ElementType::Int32) {
        CheckOutputImpl(actual[i].AsView<int32_t>(), ref[i].AsView<int32_t>());
      } else {
        // TODO: Finish type support and pull specialization logic into
        // generic helper.
        FAIL() << "Unsupported element type";
      }
    }
  }

  template <typename T>
  void CheckOutputImpl(const BufferView<T>& actual, const BufferView<T>& ref) {
    EXPECT_THAT(actual.data, MeanSquaredErrorLt(ref.data));
  }

  LiteRtModelT& Graph() const { return graph_->Graph(); }

  AtsTest(TestGraph::Ptr graph, const AtsConf& conf)
      : graph_(std::move(graph)), conf_(conf) {}

  TestGraph::Ptr graph_;
  const AtsConf& conf_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_FIXTURE_H_
