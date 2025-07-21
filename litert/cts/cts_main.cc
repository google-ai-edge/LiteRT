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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_rng.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/cts/compiled_model_executor.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/graph_helpers.h"
#include "litert/test/matchers.h"
#include "litert/test/rng_fixture.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

ABSL_FLAG(std::vector<std::string>, seeds, std::vector<std::string>({}),
          "Comma-separated test-generator/seed pairings in the form "
          "<generator_name>:<seed>. This seed will be "
          "used to generator the randomized parameters for all invocations of "
          "the respective test-generator.");

ABSL_FLAG(bool, quiet, true, "Minimize logging.");

ABSL_FLAG(std::string, backend, "cpu",
          "Which backend to use as the \"actual\".");

namespace litert {
namespace testing {
namespace {

using ::litert::internal::TflOpCode;
using ::litert::internal::TflOpCodePtr;
using ::litert::internal::TflOptions;
using ::testing::RegisterTest;
using ::testing::litert::MeanSquaredErrorLt;

class CtsOptions {
 public:
  enum class ExecutionBackend { kCpu, kGpu, kNpu };

  static Expected<CtsOptions> FromFlags() {
    // Seeds.
    const auto seed_flags = absl::GetFlag(FLAGS_seeds);
    absl::flat_hash_map<std::string, int> seeds;
    for (const auto& seed : seed_flags) {
      std::pair<std::string, std::string> seed_pair = absl::StrSplit(seed, ':');
      int seed_int;
      if (absl::SimpleAtoi(seed_pair.second, &seed_int)) {
        seeds.insert({seed_pair.first, seed_int});
      } else {
        return Error(kLiteRtStatusErrorInvalidArgument,
                     absl::StrFormat("Failed to parse seed %s", seed.c_str()));
      }
    }

    // Backend.
    ExecutionBackend backend;
    const auto backend_flag = absl::GetFlag(FLAGS_backend);
    if (backend_flag == "cpu") {
      backend = ExecutionBackend::kCpu;
    } else if (backend_flag == "gpu") {
      backend = ExecutionBackend::kGpu;
    } else if (backend_flag == "npu") {
      backend = ExecutionBackend::kNpu;
    } else {
      return Error(
          kLiteRtStatusErrorInvalidArgument,
          absl::StrFormat("Unknown backend: %s", backend_flag.c_str()));
    }

    return CtsOptions(std::move(seeds), backend, absl::GetFlag(FLAGS_quiet));
  }

  int GetSeedForParams(absl::string_view name) const {
    static constexpr int kDefaultSeed = 42;
    auto it = seeds_for_params_.find(name);
    if (it == seeds_for_params_.end()) {
      return kDefaultSeed;
    }
    return it->second;
  }

  ExecutionBackend Backend() const { return backend_; }

  bool Quiet() const { return quiet_; }

 private:
  explicit CtsOptions(absl::flat_hash_map<std::string, int>&& seeds_for_params,
                      ExecutionBackend backend, bool quiet)
      : seeds_for_params_(std::move(seeds_for_params)),
        backend_(backend),
        quiet_(quiet) {}

  absl::flat_hash_map<std::string, int> seeds_for_params_;
  ExecutionBackend backend_;
  bool quiet_;
};

// EXAMPLE TEST LOGIC //////////////////////////////////////////////////////////

// Reference cts test logic implementation that does simply returns the input.
// Used as an example and to sanity check the framework.
//
// Note: Template parameters for all test logics define the axis of
// configurability on which we want to explicitly *enumerate* values for
// (rather than randomly generate them). This is appropriate for paramters
// where the space of possible values is small and known (e.g. types, ranks).
//    In order to work with the combinatorial parameter expand/specialize
// utilities, the template params cannot not be non-type, but these can
// be simulated simply with a standard std::integral_constant.
template <typename Rank, typename T>
struct NoOp {
 private:
  // Used by by random tensor shape generator to cap the size of the tensor.
  static constexpr size_t kMaxTensorSize = 1024;

  static_assert(std::is_same_v<typename Rank::value_type, size_t>);
  // Rank of the input and output tensor.
  static constexpr size_t kRank = Rank::value;

  // Names of the input and output tensor.
  static constexpr TensorNames<1> kInputNames = {"input"};
  static constexpr TensorNames<1> kOutputNames = {"output"};

  // Signature name of the model.
  static constexpr absl::string_view kSignatureName = "default";

  // Litert element type enum taken from c++ type T.
  static constexpr ElementType kElementType = GetElementType<T>();

  // The compliment of the "enumerated" template parameters. A "params"
  // typed is defined by all test logics and encapsulates all of the paramters
  // of this test that are randomly generated. This is suitable for larger
  // and practically unbounded spaces (like shapes).
  struct Params {
    std::array<Layout::Dim, kRank> shape;
  };

  // Utility for mapping litert ops to corresponding tflite schema types.
  using FbTypes = FbOpTypes<kLiteRtOpCodeTflAdd>;

  // Configs the random tensor data behavior. Can configure per datatype
  // if desired.
  template <typename TT>
  struct RandomTensorBufferTraits {
    using Gen = RandomTensorData<TT>;
  };

 public:
  // Defines consituent types needed by the driver code. All test logics
  // must have one of these as a public member type.
  using Traits = TestLogicTraits<TypeList<T>, TypeList<T>, Params>;

  // A common name for all tests generated by this logic. This should
  // describe the non-random parameters (see template).
  static absl::string_view Name() { return "NoOp"; }

  // Given an instance of random params, construct the graph under test.
  Expected<LiteRtModelT::Ptr> BuildGraph(const Params& params) {
    const std::vector<int32_t> dims(params.shape.begin(), params.shape.end());

    std::vector<TensorDetails> inputs(2);
    std::vector<TensorDetails> outputs(1);

    inputs[0] = TensorDetails{dims, LiteRtElementType(kElementType),
                              std::string(kInputNames[0])};

    const T cst_data = 0;
    inputs[1] = TensorDetails{
        {},
        LiteRtElementType(kElementType),
        "cst",
        OwningBufferRef<uint8_t>(reinterpret_cast<const uint8_t*>(&cst_data),
                                 sizeof(T))};

    outputs[0] = TensorDetails{dims, LiteRtElementType(kElementType),
                               std::string(kOutputNames[0])};

    return SingleOpModel<kLiteRtOpCodeTflAdd>(
        inputs, outputs, ::tflite::ActivationFunctionType_NONE,
        /*pot_scale_int16=*/false);
  }

  // Generate an instance of the random params for this test logic with the
  // given random device.
  template <typename Rng>
  Expected<Params> GenerateParams(Rng& rng) {
    RandomTensorType<kRank, kMaxTensorSize, LiteRtElementType(kElementType)>
        type;
    LITERT_ASSIGN_OR_RETURN(const auto tensor_type, type(rng));
    Params p;
    std::copy(std::cbegin(tensor_type.layout.dimensions),
              std::cbegin(tensor_type.layout.dimensions) + kRank,
              std::begin(p.shape));
    return p;
  }

  // Initialize input buffers with random data, these will be passed to the
  // compiled model api.
  template <typename Rng>
  Expected<typename Traits::InputBuffers> MakeInputs(Rng& rng,
                                                     const Params& params) {
    LITERT_ASSIGN_OR_RETURN(auto input, SimpleBuffer::Create<T>(params.shape));
    LITERT_RETURN_IF_ERROR(
        (input.template WriteRandom<T, RandomTensorBufferTraits>(rng)));
    return typename Traits::InputBuffers{std::move(input)};
  }

  // Initialize output buffers, these will be passed to the compiled model api.
  Expected<typename Traits::OutputBuffers> MakeOutputs(const Params& params) {
    LITERT_ASSIGN_OR_RETURN(auto output, SimpleBuffer::Create<T>(params.shape));
    return typename Traits::OutputBuffers{std::move(output)};
  }

  // Reference implementation which the driver code will compare the actual
  // results against.
  Expected<void> Reference(const Params& params,
                           const Traits::ReferenceInputs& inputs,
                           Traits::ReferenceOutputs& outputs) {
    auto [input] = inputs;
    auto [output] = outputs;
    const size_t num_elements = output.NumElements();
    for (size_t i = 0; i < num_elements; ++i) {
      output.data[i] = input.data[i];
    }
    return {};
  }
};

// DRIVER CODE /////////////////////////////////////////////////////////////////

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

    const auto suite_name =
        absl::StrFormat("cts_%lu_%s", id, Logic::Name().data());
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
    auto ref_inputs = Traits::MakeReferenceInputs(inputs);
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
    using TestClassType = CtsTest<Logic>;
    DefaultDevice device(options_.GetSeedForParams(Logic::Name()));
    for (size_t i = 0; i < iters_; ++i) {
      if (auto status = TestClassType::Register(test_id_++, device); !status) {
        LITERT_LOG(LITERT_WARNING, "Failed to register CTS test %lu, %s: %s",
                   test_id_, Logic::Name().data(),
                   status.Error().Message().c_str());
      }
    }
  }

  RegisterFunctor(size_t iters, size_t& test_id, const CtsOptions& options)
      : iters_(iters), test_id_(test_id), options_(options) {}

 private:
  const size_t iters_;
  size_t& test_id_;
  const CtsOptions& options_;
};

// Specializes the given test logic template with the cartesian product of
// the given type lists and registers each specialization a given number
// of times. Each of these registrations will yield a single test case with a
// a different set of random parameters.
template <template <typename...> typename Logic, typename... Lists>
void RegisterCombinations(size_t iters, size_t& test_id,
                          const CtsOptions& options) {
  RegisterFunctor f(iters, test_id, options);
  ExpandProduct<Logic, Lists...>(f);
}

}  // namespace

// Register all the cts tests.
void RegisterCtsTests(const CtsOptions& cts_options) {
  size_t test_id = 0;
  {
    // NO OP //
    // clang-format off
    RegisterCombinations<
        NoOp,  // Test logic template
        SizeListC<1, 2, 3, 4>,  // Ranks
        TypeList<float, int32_t>  // Data types
    >(/*iters=*/10, test_id, cts_options);
    // clang-format on
  }
}

}  // namespace testing
}  // namespace litert

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  auto options = litert::testing::CtsOptions::FromFlags();
  if (!options) {
    LITERT_LOG(LITERT_ERROR, "Failed to create CTS options: %s",
               options.Error().Message().c_str());
    return 1;
  }
  if (options->Quiet()) {
    LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), LITERT_SILENT);
  }
  litert::testing::RegisterCtsTests(*options);
  return RUN_ALL_TESTS();
}
