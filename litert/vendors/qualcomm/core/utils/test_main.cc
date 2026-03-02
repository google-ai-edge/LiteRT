// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "litert/vendors/qualcomm/core/utils/test_utils.h"

namespace {
constexpr char kBackendFlag[] = "--backend=";

// Simple argument parser for --backend flag.
void ParseArgs(int argc, char** argv) {
  const size_t flag_len = std::strlen(kBackendFlag);
  for (int i = 1; i < argc; ++i) {
    absl::string_view arg = argv[i];
    if (absl::StartsWith(arg, kBackendFlag)) {  // Starts with --backend=
      absl::string_view value = arg.substr(flag_len);
      qnn::SetTestBackend(absl::AsciiStrToLower(value));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Test specific options:\n"
                << "  " << kBackendFlag << "[BACKEND_NAME]\n"
                << "      Specify the backend to run tests against. Supported backends: htp, dsp, ir.\n"
                << "      Default is htp.\n"
                << "\n";
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  ParseArgs(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
