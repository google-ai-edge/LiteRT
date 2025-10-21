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

// A simple binary used for testing the litert device scripts.

#include <cstdlib>  // Required for getenv
#include <iostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "litert/core/filesystem.h"

ABSL_FLAG(std::vector<std::string>, expected_data, {},
          "Data files expected to be on the device.");
ABSL_FLAG(std::vector<std::string>, expected_libs_on_ld, {},
          "Libs expected to be on the device and expected to be locatable via "
          "link path.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto expected_data = absl::GetFlag(FLAGS_expected_data);
  const auto expected_libs_on_ld = absl::GetFlag(FLAGS_expected_libs_on_ld);

  for (const auto& data : expected_data) {
    std::cerr << "Checking data: " << data << "\n";
    ABSL_CHECK(litert::internal::Exists(data));
  }

  if (!expected_libs_on_ld.empty()) {
    const char* ld_library_path = getenv("LD_LIBRARY_PATH");
    ABSL_CHECK_NE(ld_library_path, nullptr);

    absl::flat_hash_set<std::string> ld_paths(
        absl::StrSplit(ld_library_path, ':'));

    for (const auto& lib : expected_libs_on_ld) {
      std::cerr << "Checking lib: " << lib << "\n";
      ABSL_CHECK(litert::internal::Exists(lib));
      auto p = litert::internal::Parent(lib);
      ABSL_CHECK(p.HasValue());
      ABSL_CHECK(ld_paths.find(*p) != ld_paths.end());
    }
  }

  std::cerr << "PASS\n";

  return 0;
}
