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
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/ats/capture.h"
#include "litert/ats/common.h"
#include "litert/ats/configure.h"
#include "litert/ats/fixture.h"
#include "litert/ats/register.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/test/common.h"
#include "litert/test/generators/common.h"
#include "litert/test/generators/generators.h"

// Simple validatino logic for the registration of ATS tests. We cannot use
// gtest constructs for this.

namespace litert::testing {
namespace {

int CheckAts() {
  const std::string extra_models_path = GetLiteRtPath("test/testdata/");
  absl::SetFlag(&FLAGS_extra_models, extra_models_path);
  size_t test_id = 0;
  AtsCapture cap;

  // Cpu.
  {
    auto options = AtsConf::ParseFlagsAndDoSetup();
    ABSL_CHECK(options);
    // TODO(lukeboyer): Re-enable once the cpu reference is supported.
    // RegisterExtraModels<AtsTest>(test_id, *options, cap);
    static constexpr auto kIters = 1;
    RegisterCombinations<AtsTest, NoOp, SizeListC<1>, TypeList<float, int32_t>>(
        kIters, test_id, *options, cap);
    const auto* unit_test = ::testing::UnitTest::GetInstance();
    ABSL_CHECK_EQ(unit_test->total_test_count(), 2);
  }

  // Npu.
  {
    absl::SetFlag(&FLAGS_backend, "npu");
    absl::SetFlag(&FLAGS_dispatch_dir, GetLiteRtPath("vendors/examples/"));
    absl::SetFlag(&FLAGS_plugin_dir, GetLiteRtPath("vendors/examples/"));
    auto options = AtsConf::ParseFlagsAndDoSetup();
    ABSL_CHECK(options);
    // TODO(lukeboyer): Re-enable once the cpu reference is supported..
    // RegisterExtraModels<AtsTest>(test_id, *options, cap);
    static constexpr auto kIters = 1;
    RegisterCombinations<AtsTest, BinaryNoBroadcast, SizeListC<1>,
                         TypeList<float>,
                         OpCodeListC<kLiteRtOpCodeTflSub, kLiteRtOpCodeTflAdd>>(
        kIters, test_id, *options, cap);
    const auto* unit_test = ::testing::UnitTest::GetInstance();
    ABSL_CHECK_EQ(unit_test->total_test_count(), 4);
  }

  const auto res = RUN_ALL_TESTS();

  ABSL_CHECK_EQ(cap.Rows().size(), test_id);
  auto it = cap.Rows().begin();
  const auto& entry1 = *it++;
  ABSL_CHECK_EQ(entry1.accelerator.a_type, ExecutionBackend::kCpu);
  ABSL_CHECK_EQ(entry1.run.status, RunStatus::kOk);

  const auto& entry2 = *it++;
  ABSL_CHECK_EQ(entry2.accelerator.a_type, ExecutionBackend::kCpu);
  ABSL_CHECK_EQ(entry2.run.status, RunStatus::kOk);

  const auto& entry3 = *it++;
  ABSL_CHECK_EQ(entry3.accelerator.a_type, ExecutionBackend::kNpu);
  ABSL_CHECK_EQ(entry3.accelerator.is_fully_accelerated, true);
  ABSL_CHECK_EQ(entry3.run.status, RunStatus::kOk);

  const auto& entry4 = *it++;
  ABSL_CHECK_EQ(entry4.accelerator.a_type, ExecutionBackend::kNpu);
  ABSL_CHECK_EQ(entry4.accelerator.is_fully_accelerated, false);
  ABSL_CHECK_EQ(entry4.run.status, RunStatus::kOk);

  std::ostringstream s;
  cap.Csv(s);
  const std::vector<std::string> split = absl::StrSplit(s.str(), '\n');
  ABSL_CHECK_EQ(split.size(), test_id + 2);
  ABSL_CHECK(split.back().empty());

  cap.Print(std::cerr);

  return res;
}

}  // namespace
}  // namespace litert::testing

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return litert::testing::CheckAts();
}
