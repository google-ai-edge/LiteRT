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
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"
#include "litert/ats/register.h"
#include "litert/ats/register_batch_matmul.h"
#include "litert/ats/register_binary_broadcast.h"
#include "litert/ats/register_binary_no_bcast.h"
#include "litert/ats/register_concatenation.h"
#include "litert/ats/register_conv_2d.h"
#include "litert/ats/register_depthwise_conv_2d.h"
#include "litert/ats/register_fully_connected.h"
#include "litert/ats/register_no_op.h"
#include "litert/ats/register_one_hot.h"
#include "litert/ats/register_pad.h"
#include "litert/ats/register_pooling.h"
#include "litert/ats/register_reduction.h"
#include "litert/ats/register_reshape.h"
#include "litert/ats/register_softmax.h"
#include "litert/ats/register_transformer_layer.h"
#include "litert/ats/register_transpose.h"
#include "litert/ats/register_unary.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/internal/litert_detail.h"

ABSL_FLAG(std::string, gtest_filter, "", "GTest filter override for ATS");

namespace litert::testing {
namespace {

constexpr const char* kArt = R"(
   __   _ __      ___  __      ___ __________
  / /  (_) /____ / _ \/ /_____/ _ /_  __/ __/
 / /__/ / __/ -_) , _/ __/___/ __ |/ / _\ \
/____/_/\__/\__/_/|_|\__/   /_/ |_/_/ /___/
)";

template <typename Fixture>
void RegisterAll(const AtsConf& options, size_t& test_id,
                 typename Fixture::Capture& cap) {
  RegisterExtraModels<Fixture>(test_id, options, cap);
  RegisterNoOp(options, test_id, /*iters=*/10, cap);
  RegisterBinaryNoBroadcast(options, test_id, /*iters=*/10, cap);
  RegisterBinaryBroadcast(options, test_id, /*iters=*/10, cap);
  RegisterUnary(options, test_id, /*iters=*/10, cap);
  RegisterConv2d(options, test_id, /*iters=*/10, cap);
  RegisterDepthwiseConv2d(options, test_id, /*iters=*/10, cap);
  RegisterReduction(options, test_id, /*iters=*/10, cap);
  RegisterPooling(options, test_id, /*iters=*/10, cap);
  RegisterOneHot(options, test_id, /*iters=*/10, cap);
  RegisterReshape(options, test_id, /*iters=*/10, cap);
  RegisterTransformerLayer(options, test_id, /*iters=*/2, cap);
  RegisterTranspose(options, test_id, /*iters=*/10, cap);
  RegisterBatchMatmul(options, test_id, /*iters=*/10, cap);
  RegisterFullyConnected(options, test_id, /*iters=*/10, cap);
  RegisterConcatenation(options, test_id, /*iters=*/10, cap);
  RegisterSoftmax(options, test_id, /*iters=*/10, cap);
  RegisterPad(options, test_id, /*iters=*/10, cap);
}

int Ats() {
  std::cerr << kArt << std::endl;

  auto options = AtsConf::ParseFlagsAndDoSetup();
  ABSL_CHECK(options);

  size_t test_id = 0;
  AtsInferenceTest::Capture i_cap;
  AtsCompileTest::Capture c_cap;

  if (!options->CompileMode()) {
    RegisterAll<AtsInferenceTest>(*options, test_id, i_cap);
  } else {
    RegisterAll<AtsCompileTest>(*options, test_id, c_cap);
  }

  // Preliminary report.
  {
    const auto* unit_test = ::testing::UnitTest::GetInstance();
    LITERT_LOG(LITERT_INFO, "Registered %lu tests",
               unit_test->total_test_count());
  }

  const auto res = RUN_ALL_TESTS();

  // Final report.
  if (!options->Quiet()) {
    if (options->CompileMode()) {
      options->Csv(c_cap);
      options->Print(c_cap);
    } else {
      options->Csv(i_cap);
      options->Print(i_cap);
    }
  }

  return res;
}

}  // namespace
}  // namespace litert::testing

int main(int argc, char** argv) {
  // Shim to support repeatable flags which absl does not.
  std::vector<char*> absl_flags;
  static constexpr absl::string_view kDoRegisterPrefix = "--do_register=";
  static constexpr absl::string_view kDontRegisterPrefix = "--dont_register=";
  std::vector<std::string> do_register;
  std::vector<std::string> dont_register;
  for (int i = 0; i < argc; ++i) {
    if (::litert::StartsWith(argv[i], kDoRegisterPrefix)) {
      do_register.push_back(std::string(
          absl::string_view(argv[i]).substr(kDoRegisterPrefix.size())));
    } else if (::litert::StartsWith(argv[i], kDontRegisterPrefix)) {
      dont_register.push_back(std::string(
          absl::string_view(argv[i]).substr(kDontRegisterPrefix.size())));
    } else {
      absl_flags.push_back(argv[i]);
    }
  }

  absl::SetFlag(&FLAGS_do_register, do_register);
  absl::SetFlag(&FLAGS_dont_register, dont_register);

  int absl_argc = absl_flags.size();
  ::testing::InitGoogleTest(&absl_argc, absl_flags.data());
  absl::ParseCommandLine(absl_argc, absl_flags.data());

  std::string filter = absl::GetFlag(FLAGS_gtest_filter);
  if (!filter.empty()) {
    GTEST_FLAG_SET(filter, filter);
  }

  return litert::testing::Ats();
}
