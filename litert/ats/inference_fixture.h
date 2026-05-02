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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/ats/common.h"
#include "litert/ats/configure.h"
#include "litert/ats/executor.h"
#include "litert/ats/inference_capture.h"
#include "litert/c/internal/litert_logging.h"  // IWYU pragma: keep
#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/internal/litert_numerics.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/test/generators/common.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "litert/test/simple_buffer.h"
#include "tflite/types/half.h"

namespace litert::testing {

using ::testing::RegisterTest;
using ::testing::litert::MeanSquaredErrorLt;

// Fixture for tests that test execution on a given graph.
class AtsInferenceTest : public RngTest {
  template <typename T>
  using BufferView = SimpleBuffer::CView<T>;

 public:
  using Capture = InferenceCapture;

  static constexpr absl::string_view Name() { return "inference"; }

  static void Register(TestGraph::Ptr graph, const AtsConf& conf,
                       const TestNames& names, typename Capture::Entry& cap) {
    RegisterTest(names.suite.c_str(), names.test.c_str(), nullptr, nullptr,
                 __FILE__, __LINE__,
                 [graph = std::move(graph), &conf = std::as_const(conf), &cap,
                  names]() mutable {
                   return new AtsInferenceTest(std::move(graph), conf, names,
                                               cap);
                 });
  }

  void SetUp() override {
    if (names_.should_skip) {
      GTEST_SKIP() << "Filtered by dont_register";
    }
    ASSERT_EQ(Graph().NumSubgraphs(), 1);
    LITERT_LOG(LITERT_INFO, "Setting up test for %s",
               absl::StrFormat("%v", conf_.Backend()).c_str());
    cap_.model.SetFields(names_, Graph());
    cap_.run.num_iterations = conf_.ItersPerTest();
    cap_.numerics.reference_type =
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
    cap_.accelerator.SetFields(conf_);
    if (conf_.IsNpu() && !cap_.accelerator.soc_man.empty()) {
      auto stamp = GetBuildStamp(Graph());
      if (stamp) {
        cap_.accelerator.soc_man = std::string(stamp->soc_manufacturer);
        cap_.accelerator.soc_model = std::string(stamp->soc_model);
      }
    }

    if (::testing::Test::IsSkipped()) {
      cap_.run.status = RunStatus::kSkipped;
    } else if (HasFailure()) {
      cap_.run.status = RunStatus::kError;
    } else if (TimedOut()) {
      cap_.run.status = RunStatus::kTimeout;
    } else {
      cap_.run.status = RunStatus::kOk;
    }
  }

 private:
  ConformanceSpec Conformance() const { return graph_->GetConformanceSpec(); }

  template <typename T>
  static double MeanSquaredError(absl::Span<const T> actual,
                                 absl::Span<const T> ref) {
    if (actual.empty()) {
      return 0.0;
    }
    double mse = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
      const double diff =
          static_cast<double>(actual[i]) - static_cast<double>(ref[i]);
      mse += diff * diff;
    }
    return mse / static_cast<double>(actual.size());
  }

  template <typename T>
  void CheckMseOutputImpl(const BufferView<T>& actual, const BufferView<T>& ref,
                          double tol) {
    double mse = std::numeric_limits<double>::max();
    EXPECT_THAT(actual.data, MeanSquaredErrorLt(ref.data, tol, &mse));
    cap_.numerics.NewMse(mse);
  }

  template <typename T>
  void CheckExactOutputImpl(const BufferView<T>& actual,
                            const BufferView<T>& ref) {
    ASSERT_EQ(actual.data.size(), ref.data.size());
    for (size_t i = 0; i < actual.data.size(); ++i) {
      EXPECT_EQ(actual.data[i], ref.data[i]) << "index=" << i;
    }
    cap_.numerics.NewMse(MeanSquaredError(actual.data, ref.data));
  }

  void CheckExactOutputBytesImpl(const SimpleBuffer& actual,
                                 const SimpleBuffer& ref) {
    const auto actual_bytes = actual.Span<uint8_t>();
    const auto ref_bytes = ref.Span<uint8_t>();
    ASSERT_EQ(actual_bytes.size(), ref_bytes.size());
    for (size_t i = 0; i < actual_bytes.size(); ++i) {
      EXPECT_EQ(actual_bytes[i], ref_bytes[i]) << "byte=" << i;
    }
    cap_.numerics.NewMse(0.0);
  }

  void CheckFloatAccumulationAwareOutputImpl(const BufferView<float>& actual,
                                             const BufferView<float>& ref,
                                             const ConformanceSpec& spec) {
    ASSERT_EQ(actual.data.size(), ref.data.size());
    for (size_t i = 0; i < actual.data.size(); ++i) {
      const double expected = static_cast<double>(ref.data[i]);
      const double diff =
          std::abs(static_cast<double>(actual.data[i]) - expected);
      const double tol = spec.absolute_tolerance +
                         spec.relative_tolerance * std::abs(expected);
      EXPECT_LE(diff, tol) << "index=" << i << ", expected=" << expected
                           << ", actual=" << actual.data[i];
    }
    cap_.numerics.NewMse(MeanSquaredError(actual.data, ref.data));
  }

  template <typename T>
  void CheckQuantizedBucketOutputImpl(const BufferView<T>& actual,
                                      const BufferView<T>& ref,
                                      int64_t bucket_tolerance) {
    ASSERT_EQ(actual.data.size(), ref.data.size());
    for (size_t i = 0; i < actual.data.size(); ++i) {
      const int64_t diff = std::llabs(static_cast<int64_t>(actual.data[i]) -
                                      static_cast<int64_t>(ref.data[i]));
      EXPECT_LE(diff, bucket_tolerance)
          << "index=" << i << ", expected=" << static_cast<int64_t>(ref.data[i])
          << ", actual=" << static_cast<int64_t>(actual.data[i]);
    }
    cap_.numerics.NewMse(MeanSquaredError(actual.data, ref.data));
  }

  Expected<CompiledModelExecutor::Ptr> MakeExecutor() {
    auto& env = conf_.GetEnvironment();

    CompiledModelExecutor::Ptr exec;
    if (conf_.IsNpu()) {
      auto exec =
          NpuCompiledModelExecutor::Create(Graph(), conf_.TargetOptions(), env);
      cap_.compilation.SetFields(conf_, Graph(), !exec.HasValue());
      if (!exec) {
        return exec.Error();
      }
      auto res = std::make_unique<CompiledModelExecutor>(std::move(*exec));
      return res;
    }
    if (conf_.IsCpu()) {
      LITERT_ASSIGN_OR_RETURN(auto exec,
                              CpuCompiledModelExecutor::Create(
                                  Graph(), conf_.TargetOptions(), env));
      return std::make_unique<CompiledModelExecutor>(std::move(exec));
    }
    if (conf_.IsGpu()) {
      LITERT_ASSIGN_OR_RETURN(auto exec,
                              GpuCompiledModelExecutor::Create(
                                  Graph(), conf_.TargetOptions(), env));
      return std::make_unique<CompiledModelExecutor>(std::move(exec));
    }
    if (conf_.IsGpu()) {
      LITERT_ASSIGN_OR_RETURN(auto exec, CpuCompiledModelExecutor::Create(
                                             Graph(), conf_.TargetOptions()));
      return std::make_unique<CompiledModelExecutor>(std::move(exec));
    }

    return Error(kLiteRtStatusErrorInvalidArgument, "Unsupported backend");
  }

  template <typename Rng>
  Expected<VarBuffers> MakeInputs(Rng& device) const {
    auto inputs = graph_->MakeInputs(device, conf_.DataBuilder());
    if (!inputs.HasValue()) return inputs.Error();
    return inputs;
  }

  Expected<VarBuffers> Actual(const VarBuffers& inputs,
                              CompiledModelExecutor* exec) {
    return exec->Run(inputs, cap_.latency);
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
    LITERT_ASSIGN_OR_RETURN(auto exec, CpuCompiledModelExecutor::Create(
                                           Graph(), conf_.ReferenceOptions(),
                                           conf_.GetEnvironment()));
    return exec.Run(inputs);
  }

  Expected<VarBuffers> MakeOutputs() const {
    return SimpleBuffer::LikeSignature(Graph().Subgraph(0).Outputs().begin(),
                                       Graph().Subgraph(0).Outputs().end());
  }

  void CheckOutputs(const VarBuffers& actual, const VarBuffers& ref) {
    // const ConformanceSpec spec = Conformance();
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
#if 0
    const ConformanceSpec spec = Conformance();
    double mse = std::numeric_limits<double>::max();
    EXPECT_THAT(actual.data, MeanSquaredErrorLt(ref.data, Tol(), &mse));
    cap_.numerics.NewMse(mse);
#else
    abort();  // NOT IMPLEMENTED
#endif
  }

  LiteRtModelT& Graph() const { return graph_->Graph(); }

  AtsInferenceTest(TestGraph::Ptr graph, const AtsConf& conf,
                   const TestNames& names, Capture::Entry& cap)
      : graph_(std::move(graph)), conf_(conf), names_(names), cap_(cap) {}

  TestGraph::Ptr graph_;
  const AtsConf& conf_;
  TestNames names_;
  Capture::Entry& cap_;
};

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_INFERENCE_FIXTURE_H_
