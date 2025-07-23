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
#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_rng.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/cts/compiled_model_executor.h"
#include "litert/cts/cts_configure.h"
#include "litert/test/generators/generators.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "tflite/schema/schema_generated.h"

namespace litert {
namespace testing {
namespace {

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
  using Executor = TestExecutor;
  using Traits = typename Logic::Traits;
  using Params = typename Traits::Params;
  using InputBuffers = typename Traits::InputBuffers;
  using OutputBuffers = typename Traits::OutputBuffers;
  using ReferenceInputs = typename Traits::ReferenceInputs;
  using ReferenceOutputs = typename Traits::ReferenceOutputs;
  static constexpr size_t kNumInputs = Traits::kNumInputs;
  static constexpr size_t kNumOutputs = Traits::kNumOutputs;

 public:
  // Register a fully specified case with gtest. This will generate an instance
  // of the random params needed to finish specifying the test logic and is
  // inteded to be called multiple times to generate coverage across the space
  // of possible random params.
  template <typename Rng>
  static Expected<void> Register(size_t id, Rng& rng) {
    using TestClass = CtsTest<Logic, Executor>;

    const auto suite_name = absl::StrFormat(
        "%s_cts_%lu_%s", TestExecutor::Name(), id, Logic::Name().data());
    LITERT_LOG(LITERT_VERBOSE, "Starting registration for %s",
               suite_name.c_str());

    Logic logic;

    LITERT_ASSIGN_OR_RETURN(auto params, logic.GenerateParams(rng));
    LITERT_LOG(LITERT_VERBOSE, "Generated params.");

    LITERT_ASSIGN_OR_RETURN(auto model, logic.BuildGraph(params));
    LITERT_LOG(LITERT_VERBOSE, "Built graph.");

    const auto test_name = absl::StrFormat("%v", model->Subgraph(0).Ops());

    RegisterTest(suite_name.data(), test_name.c_str(), nullptr, nullptr,
                 __FILE__, __LINE__,
                 [model = std::move(model), params = std::move(params),
                  logic = std::move(logic)]() mutable -> TestClass* {
                   return new TestClass(std::move(model), std::move(params),
                                        std::move(logic));
                 });

    return {};
  }

  // Run compiled model with random inputs and compare against the
  // reference implementation.
  void TestBody() override {
    auto device = this->TracedDevice();

    // TODO: for i in inter-test iterations
    LITERT_ASSERT_OK_AND_ASSIGN(auto inputs,
                                logic_.MakeInputs(device, params_));

    LITERT_ASSERT_OK_AND_ASSIGN(auto actual, GetActual(inputs));
    LITERT_ASSERT_OK_AND_ASSIGN(auto ref, GetReference(inputs));

    CheckOutputs(actual, ref, std::make_index_sequence<kNumOutputs>());
  }

  // Name of the dependent logic.
  static absl::string_view LogicName() { return Logic::Name(); }

 private:
  CtsTest(LiteRtModelT::Ptr model, Logic::Traits::Params params, Logic logic)
      : model_(std::move(model)), params_(std::move(params)) {}

  Expected<OutputBuffers> GetActual(const InputBuffers& inputs) {
    LITERT_ASSIGN_OR_RETURN(auto exec, Executor::Create(*model_));
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
};

// Utility to register a test logic a given number of times with a common
// random device.
class RegisterFunctor {
 public:
  template <typename Logic>
  void operator()() {
    DefaultDevice device(options_.GetSeedForParams(Logic::Name()));
    for (size_t i = 0; i < iters_; ++i) {
      if (options_.Backend() == CtsConf::ExecutionBackend::kCpu) {
        CallRegister<CtsTest<Logic, CpuCompiledModelExecutor>>(device);
      } else if (options_.Backend() == CtsConf::ExecutionBackend::kGpu) {
        ABSL_CHECK(false) << "GPU backend not supported yet.";
      } else if (options_.Backend() == CtsConf::ExecutionBackend::kNpu) {
        ABSL_CHECK(false) << "NPU backend not supported yet.";
      }
    }
  }

  RegisterFunctor(size_t iters, size_t& test_id, const CtsConf& options)
      : iters_(iters), test_id_(test_id), options_(options) {}

 private:
  template <typename TestClass, typename Device>
  void CallRegister(Device& device) {
    if (auto status = TestClass::Register(test_id_++, device); !status) {
      LITERT_LOG(LITERT_WARNING, "Failed to register CTS test %lu, %s: %s",
                 test_id_, TestClass::LogicName().data(),
                 status.Error().Message().c_str());
    }
  }

  const size_t iters_;
  size_t& test_id_;
  const CtsConf& options_;
};

// Specializes the given test logic template with the cartesian product of
// the given type lists and registers each specialization a given number
// of times. Each of these registrations will yield a single test case with a
// a different set of random parameters.
template <template <typename...> typename Logic, typename... Lists>
void RegisterCombinations(size_t iters, size_t& test_id,
                          const CtsConf& options) {
  RegisterFunctor f(iters, test_id, options);
  ExpandProduct<Logic, Lists...>(f);
}

}  // namespace

// Helper aliases to set some of the template params that don't need to vary
// for cts.
template <typename Ranks, typename Types, typename OpCodes, typename Fas>
using BinaryNoBroadcastCts =
    BinaryNoBroadcast<Ranks, Types, OpCodes, Fas, SizeC<1024>,
                      DefaultRandomTensorBufferTraits>;

// Register all the cts tests.
void RegisterCtsTests(const CtsConf& cts_options) {
  size_t test_id = 0;
  {
    // NO OP //
    // clang-format off
    RegisterCombinations<
        ExampleTestLogic,  // Test logic template
        SizeListC<1, 2, 3, 4>,  // Ranks
        TypeList<float, int32_t>  // Data types
    >(/*iters=*/10, test_id, cts_options);
    // clang-format on
  }

  {
    // BINARY NO BCAST //
    // clang-format off
    RegisterCombinations<
        BinaryNoBroadcastCts,  // Test logic template
        SizeListC<1, 2, 3, 4, 5, 6>,  // Ranks
        TypeList<float, int32_t>,  // Data types
        OpCodeListC<kLiteRtOpCodeTflAdd, kLiteRtOpCodeTflSub>,  // Op codes
        FaListC<::tflite::ActivationFunctionType_NONE>  // TODO: More support.
    >(/*iters=*/10, test_id, cts_options);
    // clang-format on
  }
}

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
  litert::testing::RegisterCtsTests(*options);
  return RUN_ALL_TESTS();
}
