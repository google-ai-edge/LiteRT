/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/examples/utils/initialization.h"

#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/flags/usage_config.h"

namespace litert {
namespace tensor {

namespace {

bool ContainsHelpFlags(absl::string_view path) {
  return path.rfind("main.cc") != absl::string_view::npos;
}

std::string NormalizeFilename(absl::string_view path) {
  auto pos = path.find("tensor/");
  return pos != path.npos ? std::string(path.substr(pos)) : std::string(path);
}

}  // namespace

void Initialize(const char* usage, int& argc, char**& argv, bool remove_flags) {
  absl::SetProgramUsageMessage(usage);
  absl::SetFlagsUsageConfig({.contains_help_flags = ContainsHelpFlags,
                             .normalize_filename = NormalizeFilename});
  absl::ParseCommandLine(argc, argv);
}

}  // namespace tensor
}  // namespace litert
