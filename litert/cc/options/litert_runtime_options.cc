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

#include "litert/cc/options/litert_runtime_options.h"
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"
#include "third_party/tomlplusplus/toml.hpp"

namespace litert {

Expected<RuntimeOptions> RuntimeOptions::Create() {
  return RuntimeOptions();
}

Expected<OpaqueOptions> RuntimeOptions::GetOpaqueOptions() {
  toml::table tbl;
  tbl.insert_or_assign("enable_profiling", enable_profiling_);
  tbl.insert_or_assign("error_reporter_mode", error_reporter_mode_);
  tbl.insert_or_assign("compress_quantization_zero_points",
                      compress_quantization_zero_points_);
  std::stringstream ss;
  ss << tbl;
  char* payload = static_cast<char*>(malloc(ss.str().length() + 1));
  strncpy(payload, ss.str().c_str(), ss.str().length() + 1);
  return OpaqueOptions::Create(
      kPayloadIdentifier.data(), payload,
      [](void* ptr) { free(ptr); });
}
}  // namespace litert
