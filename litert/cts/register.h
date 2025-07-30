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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_CTS_REGISTER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_CTS_REGISTER_H_

#include <cstddef>
#include <cstdint>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "litert/c/litert_logging.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_rng.h"
#include "litert/cts/compiled_model_executor.h"
#include "litert/cts/cts_configure.h"
#include "litert/test/generators/generators.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {

// Utility to register a test logic a given number of times with a common
// random device.
template <template <typename TestLogic> typename Fixture>
class RegisterFunctor {
 public:
  template <typename Logic>
  void operator()() {
    DefaultDevice device(options_.GetSeedForParams(Logic::Name()));
    for (size_t i = 0; i < iters_; ++i) {
      if (options_.Backend() == CtsConf::ExecutionBackend::kCpu) {
        BuildParamsAndRegister<Fixture<Logic>, CpuCompiledModelExecutor>(
            device);
      } else if (options_.Backend() == CtsConf::ExecutionBackend::kGpu) {
        ABSL_CHECK(false) << "GPU backend not supported yet.";
      } else if (options_.Backend() == CtsConf::ExecutionBackend::kNpu) {
        BuildParamsAndRegister<Fixture<Logic>, NpuCompiledModelExecutor>(
            device, options_.DispatchDir(), options_.PluginDir());
      }
    }
  }

  RegisterFunctor(size_t iters, size_t& test_id, const CtsConf& options)
      : iters_(iters), test_id_(test_id), options_(options) {}

 private:
  template <typename TestClass, typename TestExecutor, typename Device,
            typename... ExecArgs>
  void BuildParamsAndRegister(Device& device, ExecArgs&&... exec_args) {
    auto params = TestClass::BuildSetupParams(device);
    if (!params) {
      LITERT_LOG(LITERT_WARNING,
                 "Failed to build params for CTS test %lu, %s: %s", test_id_,
                 TestClass::LogicName().data(),
                 params.Error().Message().c_str());
      return;
    }

    const auto suite_name =
        TestClass::FmtSuiteName(test_id_++, TestExecutor::Name());
    const auto test_name = TestClass::FmtTestName(*params->model);

    if (!options_.ShouldRegister(
            absl::StrFormat("%s.%s", suite_name, test_name))) {
      TestClass::SkippedTest::Register(suite_name, test_name);
      return;
    }

    auto exec = TestExecutor::Create(*params->model,
                                     std::forward<ExecArgs>(exec_args)...);
    if (!exec) {
      LITERT_LOG(LITERT_WARNING,
                 "Failed to create executor for CTS test %lu, %s: %s", test_id_,
                 TestClass::LogicName().data(), exec.Error().Message().c_str());
      return;
    }

    if (auto status = TestClass::Register(suite_name, test_name,
                                          std::move(*params), std::move(*exec));
        !status) {
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
template <template <typename TestLogic> typename Fixture,
          template <typename...> typename Logic, typename... Lists>
void RegisterCombinations(size_t iters, size_t& test_id,
                          const CtsConf& options) {
  RegisterFunctor<Fixture> f(iters, test_id, options);
  ExpandProduct<Logic, Lists...>(f);
}

// Helper aliases to set some of the template params that don't need to vary
// for cts.
template <typename Ranks, typename Types, typename OpCodes, typename Fas>
using BinaryNoBroadcastCts =
    BinaryNoBroadcast<Ranks, Types, OpCodes, Fas, SizeC<1024>,
                      DefaultRandomTensorBufferTraits>;

template <template <typename TestLogic> typename Fixture>
void RegisterCtsTests(const CtsConf& cts_options) {
  size_t test_id = 0;
  {
    // NO OP //
    // clang-format off
    RegisterCombinations<
        Fixture,
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
        Fixture,
        BinaryNoBroadcastCts,  // Test logic template
        SizeListC<1, 2, 3, 4, 5, 6>,  // Ranks
        TypeList<float, int32_t>,  // Data types
        OpCodeListC<kLiteRtOpCodeTflAdd, kLiteRtOpCodeTflSub>,  // Op codes
        FaListC<::tflite::ActivationFunctionType_NONE>  // TODO: More support.
    >(/*iters=*/10, test_id, cts_options);
    // clang-format on
  }
}

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_CTS_REGISTER_H_
