// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/utils/test_utils.h"

namespace {

// Simple argument parser for --backend flag.
bool ParseArgs(int argc, char** argv) {
  constexpr absl::string_view kBackendFlag = "--backend=";
  constexpr absl::string_view kHelpFlag = "--help";
  constexpr absl::string_view kShortHelpFlag = "-h";

  for (int i = 1; i < argc; ++i) {
    absl::string_view arg = argv[i];
    if (absl::StartsWith(arg, kBackendFlag)) {  // Starts with --backend=
      absl::string_view value = arg.substr(kBackendFlag.size());
      std::string backend_str = absl::AsciiStrToLower(value);
      if (backend_str == "htp") {
        qnn::SetTestBackend(qnn::BackendType::kHtpBackend);
      } else if (backend_str == "dsp") {
        qnn::SetTestBackend(qnn::BackendType::kDspBackend);
      } else if (backend_str == "gpu") {
        qnn::SetTestBackend(qnn::BackendType::kGpuBackend);
      } else {
        std::cerr << "Unknown backend: " << backend_str << std::endl;
        return false;
      }
    } else if (arg == kHelpFlag || arg == kShortHelpFlag) {
      std::cout << "Test specific options:\n"
                << "  " << kBackendFlag << "[BACKEND_NAME]\n"
                << "      Specify the backend to run tests against. Supported "
                   "backends: htp, dsp.\n"
                << "      Default is htp.\n"
                << std::endl;
      exit(0);
    }
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  if (!ParseArgs(argc, argv)) {
    return 1;
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
