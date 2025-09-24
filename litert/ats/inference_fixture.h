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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_INFERENCE_FIXTURE_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_INFERENCE_FIXTURE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/ats/capture.h"
#include "litert/ats/common.h"
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

// Fixture for tests that test execution on a given graph.
class AtsInferenceTest : public RngTest {
 public:
  template <typename T>
  using BufferView = typename SimpleBuffer::CView<T>;

  static void Register(TestGraph::Ptr graph, const AtsConf& conf,
                       const TestNames& names,
                       std::optional<AtsCaptureEntry::Ref> cap = {}) {
    RegisterTest(names.suite.c_str(), names.test.c_str(), nullptr, nullptr,
                 __FILE__, __LINE__,
                 [graph = std::move(graph), &conf = std::as_const(conf), cap,
                  names]() mutable {
                   return new AtsInferenceTest(std::move(graph), conf, names,
                                               cap);
                 });
  }

  void SetUp() override {
    ASSERT_EQ(Graph().NumSubgraphs(), 1);
    ASSERT_EQ(Graph().MainSubgraph()->NumOutputs(), 1);
    LITERT_LOG(LITERT_INFO, "Setting up test for %s",
               absl::StrFormat("%v", conf_.Backend()).c_str());
    Cap().model.name = names_.report_id;
    Cap().model.desc = names_.desc;
    const auto is_precompiled = GetBuildStamp(Graph()).has_value();
    Cap().model.precompiled = is_precompiled;
    Cap().accelerator.a_type = conf_.Backend();
    Cap().run.num_iterations = conf_.ItersPerTest();
    Cap().numerics.reference_type =
        graph_->HasReference() ? ReferenceType::kCustom : ReferenceType::kCpu;
  }

  void TestBody() override {
    auto device = this->TracedDevice(conf_.DataSeed());
    LITERT_ASSERT_OK_AND_ASSIGN(auto exec, MakeExecutor());
    for (auto _ : this->FuzzBlock(conf_.ItersPerTest(), conf_.MaxMsPerTest())) {
      LITERT_ASSERT_OK_AND_ASSIGN(auto inputs, MakeInputs(device));
      LITERT_ASSERT_OK_AND_ASSIGN(auto ref, Reference(inputs));
      LITERT_ASSERT_OK_AND_ASSIGN(auto actual, Actual(inputs, exec.get()));
      CheckOutputs(actual, ref);
    }
  }

  void TearDown() override {
    if (conf_.IsNpu()) {
      auto stamp = GetBuildStamp(Graph());
      if (stamp) {
        Cap().accelerator.soc_man = std::string(stamp->soc_manufacturer);
        Cap().accelerator.soc_model = std::string(stamp->soc_model);
      }
    }

    if (HasFailure()) {
      Cap().run.status = RunStatus::kError;
    } else if (TimedOut()) {
      Cap().run.status = RunStatus::kTimeout;
    } else {
      Cap().run.status = RunStatus::kOk;
    }
  }

 private:
  Expected<CompiledModelExecutor::Ptr> MakeExecutor() {
    CompiledModelExecutor::Ptr exec;
    if (conf_.IsNpu()) {
      LITERT_ASSIGN_OR_RETURN(
          auto exec, NpuCompiledModelExecutor::Create(
                         Graph(), conf_.DispatchDir(), conf_.PluginDir()));
      auto res = std::make_unique<CompiledModelExecutor>(std::move(exec));
      // TODO: Fully compiled needs a debug. The capture needs an n/a as well.
      Cap().accelerator.is_fully_accelerated =
          ::litert::internal::IsFullyCompiled(Graph());
      return res;

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

  Expected<VarBuffers> Actual(const VarBuffers& inputs,
                              CompiledModelExecutor* exec) {
    LITERT_ASSIGN_OR_RETURN(auto actual, exec->Run(inputs, Cap().latency));
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
    double mse = std::numeric_limits<double>::max();
    EXPECT_THAT(actual.data, MeanSquaredErrorLt(ref.data, 1e-5, &mse));
    Cap().numerics.NewMse(mse);
  }

  LiteRtModelT& Graph() const { return graph_->Graph(); }

  AtsCaptureEntry& Cap() { return cap_.has_value() ? cap_->get() : dummy_cap_; }

  AtsInferenceTest(TestGraph::Ptr graph, const AtsConf& conf,
                   const TestNames& names,
                   std::optional<AtsCaptureEntry::Ref> cap)
      : graph_(std::move(graph)), conf_(conf), names_(names), cap_(cap) {}

  TestGraph::Ptr graph_;
  const AtsConf& conf_;
  TestNames names_;
  std::optional<AtsCaptureEntry::Ref> cap_;

  AtsCaptureEntry dummy_cap_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_INFERENCE_FIXTURE_H_
