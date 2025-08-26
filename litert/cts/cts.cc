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

#include <cstddef>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_rng.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/cts/compiled_model_executor.h"
#include "litert/cts/cts_configure.h"
#include "litert/cts/register.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"

namespace litert {
namespace testing {

using ::litert::internal::TflOpCode;
using ::litert::internal::TflOpCodePtr;
using ::litert::internal::TflOptions;
using ::testing::RegisterTest;
using ::testing::litert::MeanSquaredErrorLt;

// Class that drives all cts test cases. These are specialized with
// fully specified test logic and executor (backend).
template <typename TestLogic, typename TestExecutor = CpuCompiledModelExecutor>
class CtsTest : public RngTest {
 private:
  using Logic = TestLogic;
  using Traits = typename Logic::Traits;
  using Params = typename Traits::Params;
  using InputBuffers = typename Traits::InputBuffers;
  using OutputBuffers = typename Traits::OutputBuffers;
  using ReferenceInputs = typename Traits::ReferenceInputs;
  using ReferenceOutputs = typename Traits::ReferenceOutputs;
  static constexpr size_t kNumInputs = Traits::kNumInputs;
  static constexpr size_t kNumOutputs = Traits::kNumOutputs;

 public:
  using Executor = TestExecutor;

  static std::string FmtSuiteName(size_t id) {
    return absl::StrFormat("%s_cts_%lu_%s", TestExecutor::Name(), id,
                           Logic::Name().data());
  }

  static std::string FmtTestName(const LiteRtModelT& model) {
    return absl::StrFormat("%v", model.Subgraph(0).Ops());
  }

  // The various objects needed to initialize a test case.
  struct SetupParams {
    LiteRtModelT::Ptr model;
    Params params;
    Logic logic;
  };

  // Generate the setup params for a test case given the random number
  // generator. These can be passed to the Register method to register
  // the test case.
  template <typename Rng>
  static Expected<SetupParams> BuildSetupParams(Rng& rng) {
    Logic logic;
    LITERT_ASSIGN_OR_RETURN(auto params, logic.GenerateParams(rng));
    LITERT_ASSIGN_OR_RETURN(auto model, logic.BuildGraph(params));
    return SetupParams{std::move(model), std::move(params), std::move(logic)};
  }

  // Register a fully specified case with gtest.
  static Expected<void> Register(const std::string& suite_name,
                                 const std::string& test_name,
                                 SetupParams&& setup_params,
                                 Executor::Args&& executor_args,
                                 RandomTensorDataBuilder&& data_builder) {
    RegisterTest(suite_name.c_str(), test_name.c_str(), nullptr, nullptr,
                 __FILE__, __LINE__,
                 [setup_params = std::move(setup_params),
                  executor_args = std::move(executor_args),
                  data_builder = std::move(data_builder)]() mutable {
                   return new CtsTest(std::move(setup_params.model),
                                      std::move(setup_params.params),
                                      std::move(setup_params.logic),
                                      std::move(executor_args),
                                      std::move(data_builder));
                 });
    return {};
  }

  // Run compiled model with random inputs and compare against the
  // reference implementation.
  void TestBody() override {
    auto device = this->TracedDevice();

    // TODO: for i in inter-test iterations
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto inputs, logic_.MakeInputs(data_builder_, device, params_));

    LITERT_ASSERT_OK_AND_ASSIGN(auto actual, GetActual(inputs));
    LITERT_ASSERT_OK_AND_ASSIGN(auto ref, GetReference(inputs));

    CheckOutputs(actual, ref, std::make_index_sequence<kNumOutputs>());
  }

  // Name of the dependent logic.
  static absl::string_view LogicName() { return Logic::Name(); }

 private:
  CtsTest(LiteRtModelT::Ptr model, typename Logic::Traits::Params params,
          Logic logic, typename Executor::Args&& executor_args,
          RandomTensorDataBuilder&& data_builder)
      : model_(std::move(model)),
        params_(std::move(params)),
        logic_(std::move(logic)),
        executor_args_(std::move(executor_args)),
        data_builder_(std::move(data_builder)) {}

  Expected<OutputBuffers> GetActual(const InputBuffers& inputs) {
    LITERT_ASSIGN_OR_RETURN(auto exec,
                            Executor::Create(*model_, executor_args_));
    LITERT_ASSIGN_OR_RETURN(auto actual, exec.Run(inputs));
    if (actual.size() != kNumOutputs) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   absl::StrFormat("Expected %d outputs, got %d", kNumOutputs,
                                   actual.size()));
    }
    return VecToArray<kNumOutputs>(std::move(actual));
  }

  Expected<OutputBuffers> GetReference(const InputBuffers& inputs) {
    LITERT_ASSIGN_OR_RETURN(auto outputs, logic_.MakeOutputs(params_));
    const auto ref_inputs = Traits::MakeReferenceInputs(inputs);
    auto ref_outputs = Traits::MakeReferenceOutputs(outputs);
    LITERT_RETURN_IF_ERROR(logic_.Reference(params_, ref_inputs, ref_outputs));
    if (outputs.size() != kNumOutputs) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   absl::StrFormat("Expected %d outputs, got %d", kNumOutputs,
                                   outputs.size()));
    }
    return outputs;
  }

  template <size_t I>
  void CheckOutput(const OutputBuffers& actual, const OutputBuffers& ref) {
    auto actual_span =
        actual[I].template Span<typename Traits::template OutputDataType<I>>();
    auto ref_span =
        ref[I].template Span<typename Traits::template OutputDataType<I>>();
    EXPECT_THAT(actual_span, MeanSquaredErrorLt(ref_span));
  }

  template <size_t... Is>
  void CheckOutputs(const OutputBuffers& actual, const OutputBuffers& ref,
                    std::index_sequence<Is...>) {
    (CheckOutput<Is>(actual, ref), ...);
  }

  typename LiteRtModelT::Ptr model_;
  typename Logic::Traits::Params params_;
  Logic logic_;
  typename Executor::Args executor_args_;
  RandomTensorDataBuilder data_builder_;
};

}  // namespace testing
}  // namespace litert

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  auto options = litert::testing::CtsConf::ParseFlagsAndDoSetup();
  if (!options) {
    LITERT_LOG(LITERT_ERROR, "Failed to create CTS configuration: %s",
               options.Error().Message().c_str());
    return 1;
  }
  ::litert::testing::RegisterCtsTests<::litert::testing::CtsTest>(*options);
  return RUN_ALL_TESTS();
}
