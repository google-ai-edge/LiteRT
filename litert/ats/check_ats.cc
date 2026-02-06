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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/flags/reflection.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/common.h"
#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/executor.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/test/common.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/generators.h"
#include "litert/test/simple_buffer.h"

// Simple validatino logic for the registration of ATS tests. We cannot use
// gtest constructs for this.

namespace litert::testing {
namespace {

Expected<AtsConf> NpuInferenceOptions() {
  absl::FlagSaver saver;
  absl::SetFlag(&FLAGS_dispatch_dir, GetLiteRtPath("vendors/examples/"));
  absl::SetFlag(&FLAGS_plugin_dir, GetLiteRtPath("vendors/examples/"));
  absl::SetFlag(&FLAGS_backend, "npu");
  absl::SetFlag(&FLAGS_soc_manufacturer, "ExampleSocManufacturer");
  return AtsConf::ParseFlagsAndDoSetup();
}

Expected<AtsConf> CpuInferenceOptions() {
  absl::FlagSaver saver;
  absl::SetFlag(&FLAGS_backend, "cpu");
  return AtsConf::ParseFlagsAndDoSetup();
}

Expected<AtsConf> CompileOptions() {
  absl::FlagSaver saver;
  absl::SetFlag(&FLAGS_dispatch_dir, GetLiteRtPath("vendors/examples/"));
  absl::SetFlag(&FLAGS_plugin_dir, GetLiteRtPath("vendors/examples/"));
  absl::SetFlag(&FLAGS_compile_mode, true);
  absl::SetFlag(&FLAGS_soc_manufacturer, "ExampleSocManufacturer");
  absl::SetFlag(&FLAGS_backend, "npu");
  return AtsConf::ParseFlagsAndDoSetup();
}

Expected<void> CheckAts() {
  absl::SetFlag(&FLAGS_extra_models, {GetLiteRtPath("test/testdata/")});

  LITERT_ASSIGN_OR_RETURN(auto dir, UniqueTestDirectory::Create());
  absl::SetFlag(&FLAGS_models_out, dir.Str());

  size_t test_id = 0;

  AtsInferenceTest::Capture i_cap;
  AtsCompileTest::Capture c_cap;

  LITERT_ASSIGN_OR_RETURN(auto cpu_inference_options, CpuInferenceOptions());
  LITERT_ASSIGN_OR_RETURN(auto compile_options, CompileOptions());
  LITERT_ASSIGN_OR_RETURN(auto npu_inference_options, NpuInferenceOptions());

  // CPU
  {
    RegisterExtraModels<AtsInferenceTest>(test_id, cpu_inference_options,
                                          i_cap);
    RegisterCombinations<AtsInferenceTest, NoOp, SizeListC<1>,
                         TypeList<float, int32_t>>(
        /*iters=*/1, test_id, cpu_inference_options, i_cap);
    RegisterCombinations<AtsInferenceTest, BinaryNoBroadcast, SizeListC<1>,
                         TypeList<float>,
                         OpCodeListC<kLiteRtOpCodeTflSub, kLiteRtOpCodeTflAdd>>(
        /*iters=*/1, test_id, cpu_inference_options, i_cap);
  }

  // NPU

  {
    RegisterCombinations<AtsInferenceTest, BinaryNoBroadcast, SizeListC<1>,
                         TypeList<float>, OpCodeListC<kLiteRtOpCodeTflSub>>(
        /*iters=*/1, test_id, npu_inference_options, i_cap);
  }

  // Compile

  {
    RegisterCombinations<AtsCompileTest, BinaryNoBroadcast, SizeListC<1>,
                         TypeList<float>, OpCodeListC<kLiteRtOpCodeTflSub>>(
        /*iters=*/1, test_id, compile_options, c_cap);
  }

  const auto* ut = ::testing::UnitTest::GetInstance();
  LITERT_ENSURE((ut->total_test_count() == test_id),
                Error(kLiteRtStatusErrorRuntimeFailure),
                "Unexpected number of tests.");

  LITERT_ENSURE(!RUN_ALL_TESTS(), Error(kLiteRtStatusErrorRuntimeFailure),
                "Failed to run all tests.");

  // Check inference capture.
  {
    const auto num_extra_models = std::count_if(
        i_cap.Rows().begin(), i_cap.Rows().end(), [](const auto& row) {
          return row.numerics.reference_type == ReferenceType::kCpu;
        });

    const auto i_cap_ok = std::all_of(
        i_cap.Rows().begin(), i_cap.Rows().end(),
        [](const auto& row) { return row.run.status != RunStatus::kError; });

    LITERT_ENSURE(
        i_cap_ok && i_cap.Rows().size() == test_id - 1 && num_extra_models == 1,
        Error(kLiteRtStatusErrorRuntimeFailure),
        "Status capture contains errors.");
  }

  // Check compile capture.
  {
    const auto c_cap_ok = std::all_of(
        c_cap.Rows().begin(), c_cap.Rows().end(), [](const auto& row) {
          return row.compilation_detail.status != CompilationStatus::kError;
        });

    LITERT_ENSURE(c_cap_ok && c_cap.Rows().size() == 1,
                  Error(kLiteRtStatusErrorRuntimeFailure),
                  "Status capture contains errors.");
  }

  i_cap.Print(std::cerr);
  i_cap.Csv(std::cerr);
  c_cap.Print(std::cerr);
  c_cap.Csv(std::cerr);

  // Check post-test saved models.
  {
    LITERT_ASSIGN_OR_RETURN(auto out_files, internal::ListDir(dir.Str()));
    LITERT_ENSURE(out_files.size() == 1,
                  Error(kLiteRtStatusErrorRuntimeFailure),
                  "Unexpected number of output files.");

    const auto& out_file = out_files.front();
    LITERT_ENSURE(EndsWith(out_file, ".tflite"),
                  Error(kLiteRtStatusErrorRuntimeFailure),
                  "Unexpected output file name.");

    // Check compiled file can be ran.
    LITERT_ASSIGN_OR_RETURN(auto model, internal::LoadModelFromFile(out_file));
    LITERT_ENSURE(internal::IsFullyCompiled(*model),
                  Error(kLiteRtStatusErrorRuntimeFailure),
                  "Model is not fully compiled.")

    LITERT_ASSIGN_OR_RETURN(auto exec,
                            NpuCompiledModelExecutor::Create(
                                *model, npu_inference_options.TargetOptions(),
                                npu_inference_options.DispatchDir()));
    const auto& subgraph = *model->Subgraphs()[0];
    LITERT_ASSIGN_OR_RETURN(
        auto inputs, SimpleBuffer::LikeSignature(subgraph.Inputs().begin(),
                                                 subgraph.Inputs().end()));
    LITERT_RETURN_IF_ERROR(exec.Run(inputs));
  }

  return {};
}

}  // namespace
}  // namespace litert::testing

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  auto res = litert::testing::CheckAts();
  return !res;
}
