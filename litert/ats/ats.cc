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
#include <iostream>

#include <gtest/gtest.h>
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "litert/ats/capture.h"
#include "litert/ats/configure.h"
#include "litert/ats/fixture.h"
#include "litert/ats/register.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_c_types_printing.h"  // IWYU pragma: keep
#include "litert/cc/litert_detail.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/generators.h"
#include "tflite/schema/schema_generated.h"

namespace litert::testing {
namespace {

static constexpr const char* kArt = R"(
   ###     ######   ######  ######## ##       ######## ########     ###    ########  #######  ########     ######## ########  ######  ########     ######  ##     ## #### ######## ######## 
  ## ##   ##    ## ##    ## ##       ##       ##       ##     ##   ## ##      ##    ##     ## ##     ##       ##    ##       ##    ##    ##       ##    ## ##     ##  ##     ##    ##
 ##   ##  ##       ##       ##       ##       ##       ##     ##  ##   ##     ##    ##     ## ##     ##       ##    ##       ##          ##       ##       ##     ##  ##     ##    ##
##     ## ##       ##       ######   ##       ######   ########  ##     ##    ##    ##     ## ########        ##    ######    ######     ##        ######  ##     ##  ##     ##    ######
######### ##       ##       ##       ##       ##       ##   ##   #########    ##    ##     ## ##   ##         ##    ##             ##    ##             ## ##     ##  ##     ##    ##
##     ## ##    ## ##    ## ##       ##       ##       ##    ##  ##     ##    ##    ##     ## ##    ##        ##    ##       ##    ##    ##       ##    ## ##     ##  ##     ##    ##
##     ##  ######   ######  ######## ######## ######## ##     ## ##     ##    ##     #######  ##     ##       ##    ########  ######     ##        ######   #######  ####    ##    ######## 
)";

void RegisterNoOp(const AtsConf& options, size_t& test_id, size_t iters,
                  AtsCapture::Ref cap) {
  // clang-format off
  RegisterCombinations<
      AtsTest,
      NoOp,
      SizeListC<1, 2, 3, 4>,
      TypeList<float, int32_t>>
    (iters, test_id, options, cap);
  // clang-format on
}

void RegisterBinaryNoBroadcast(const AtsConf& options, size_t& test_id,
                               size_t iters, AtsCapture::Ref cap) {
  // clang-format off
  RegisterCombinations<
      AtsTest,
      BinaryNoBroadcast,
      SizeListC<1, 2, 3, 4, 5, 6>,
      TypeList<float, int32_t>,
      OpCodeListC<kLiteRtOpCodeTflAdd, kLiteRtOpCodeTflSub>,
      FaListC<::tflite::ActivationFunctionType_NONE>>
    (iters, test_id, options, cap);
  // clang-format on
}

int Ats() {
  std::cerr << kArt << std::endl;

  auto options = AtsConf::ParseFlagsAndDoSetup();
  ABSL_CHECK(options);

  size_t test_id = 0;
  AtsCapture cap;

  RegisterNoOp(*options, test_id, /*iters=*/10, cap);
  RegisterBinaryNoBroadcast(*options, test_id, /*iters=*/10, cap);
  RegisterExtraModels<AtsTest>(test_id, *options, cap);

  // Preliminary report.
  {
    const auto* unit_test = ::testing::UnitTest::GetInstance();
    LITERT_LOG(LITERT_INFO, "Registered %lu tests",
               unit_test->total_test_count());
  }

  const auto res = RUN_ALL_TESTS();

  options->Csv(cap);
  options->Print(cap);

  return res;
}

}  // namespace
}  // namespace litert::testing

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return litert::testing::Ats();
}
