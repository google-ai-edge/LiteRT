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
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
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
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/internal/litert_rng.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/filesystem.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/common.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/generators.h"
#include "litert/test/simple_buffer.h"
#include "tflite/schema/schema_generated.h"

// Simple validating logic for the registration of ATS tests. We cannot use
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

Expected<AtsConf> CompileOptions(absl::string_view models_out) {
  absl::FlagSaver saver;
  absl::SetFlag(&FLAGS_dispatch_dir, GetLiteRtPath("vendors/examples/"));
  absl::SetFlag(&FLAGS_plugin_dir, GetLiteRtPath("vendors/examples/"));
  absl::SetFlag(&FLAGS_compile_mode, true);
  absl::SetFlag(&FLAGS_soc_manufacturer, "ExampleSocManufacturer");
  absl::SetFlag(&FLAGS_backend, "npu");
  absl::SetFlag(&FLAGS_models_out, std::string(models_out));
  return AtsConf::ParseFlagsAndDoSetup();
}

Expected<void> CheckExtraModelBoundedInputs() {
  auto model = std::make_unique<LiteRtModelT>();
  auto& sg = model->EmplaceSubgraph();
  auto add_tensor = [&](LiteRtElementType type,
                        std::initializer_list<int32_t> dims) -> LiteRtTensorT& {
    auto& t = sg.EmplaceTensor();
    t.SetType(MakeRankedTensorType(type, dims));
    return t;
  };
  auto add_op = [&](LiteRtOpCode code,
                    std::initializer_list<LiteRtTensorT*> ins,
                    LiteRtTensorT& out) -> LiteRtOpT& {
    auto& op = sg.EmplaceOp();
    op.SetOpCode(code);
    for (auto* in : ins) internal::AttachInput(in, op);
    internal::AttachOutput(&out, op);
    return op;
  };

  // Input 0: INT32 token_ids -> EmbeddingLookup (vocab_size=17).
  auto& token_ids = add_tensor(kLiteRtElementTypeInt32, {64});
  auto& emb_out = add_tensor(kLiteRtElementTypeFloat32, {64, 16});
  add_op(kLiteRtOpCodeTflEmbeddingLookup,
         {&token_ids, &add_tensor(kLiteRtElementTypeFloat32, {17, 16})},
         emb_out);

  // Input 1: INT64 gather_ids -> Cast -> Reshape -> Gather (axis=1, dim=23).
  auto& gather_ids = add_tensor(kLiteRtElementTypeInt64, {8, 8});
  auto& cast_out = add_tensor(kLiteRtElementTypeInt32, {8, 8});
  auto& reshape_out = add_tensor(kLiteRtElementTypeInt32, {64});
  auto& gather_out = add_tensor(kLiteRtElementTypeFloat32, {5, 64, 16});
  add_op(kLiteRtOpCodeTflCast, {&gather_ids}, cast_out);
  add_op(kLiteRtOpCodeTflReshape, {&cast_out}, reshape_out);
  auto& gather_op = add_op(
      kLiteRtOpCodeTflGather,
      {&add_tensor(kLiteRtElementTypeFloat32, {5, 23, 16}), &reshape_out},
      gather_out);
  internal::TflOptions gather_opts;
  gather_opts.type = tflite::BuiltinOptions_GatherOptions;
  gather_opts.Set(tflite::GatherOptionsT{{}, /*axis=*/1});
  internal::SetTflOptions(gather_op, std::move(gather_opts));

  // Input 2: INT64 attention_mask -> Cast(FLOAT32) (fallback to [0, 2)).
  auto& attention_mask = add_tensor(kLiteRtElementTypeInt64, {64});
  auto& mask_out = add_tensor(kLiteRtElementTypeFloat32, {64});
  add_op(kLiteRtOpCodeTflCast, {&attention_mask}, mask_out);

  sg.Inputs() = {&token_ids, &gather_ids, &attention_mask};
  sg.Outputs() = {&emb_out, &gather_out, &mask_out};

  ExtraModel extra_model(std::move(model));
  DefaultDevice device(42);
  RandomTensorDataBuilder data_builder;
  LITERT_ASSIGN_OR_RETURN(auto inputs,
                          extra_model.MakeInputs(device, data_builder));
  auto in_bounds = [](auto span, int64_t upper) {
    return std::all_of(span.begin(), span.end(),
                       [upper](auto v) { return v >= 0 && v < upper; });
  };
  LITERT_ENSURE(inputs.size() == 3 &&
                    in_bounds(inputs[0].Span<int32_t>(), 17) &&
                    in_bounds(inputs[1].Span<int64_t>(), 23) &&
                    in_bounds(inputs[2].Span<int64_t>(), 2),
                Error(kLiteRtStatusErrorRuntimeFailure),
                "Out-of-bounds generated integer input.");
  return {};
}

Expected<void> CheckAts() {
  LITERT_RETURN_IF_ERROR(CheckExtraModelBoundedInputs());

  absl::SetFlag(&FLAGS_extra_models, {GetLiteRtPath("test/testdata/")});

  LITERT_ASSIGN_OR_RETURN(auto dir, UniqueTestDirectory::Create());

  size_t test_id = 0;

  AtsInferenceTest::Capture i_cap;
  AtsCompileTest::Capture c_cap;

  LITERT_ASSIGN_OR_RETURN(auto cpu_inference_options, CpuInferenceOptions());
  LITERT_ASSIGN_OR_RETURN(auto compile_options, CompileOptions(dir.Str()));
  LITERT_ASSIGN_OR_RETURN(auto npu_inference_options, NpuInferenceOptions());

  // CPU
  {
    RegisterExtraModels<AtsInferenceTest>(test_id, cpu_inference_options,
                                          i_cap);
    RegisterCombinations<AtsInferenceTest, BinaryNoBroadcast, SizeListC<2>,
                         TypeList<float, int32_t>,
                         OpCodeListC<kLiteRtOpCodeTflAdd>>(
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
                                npu_inference_options.GetEnvironment()));
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
