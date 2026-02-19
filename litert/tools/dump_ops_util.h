// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_TOOLS_DUMP_OPS_UTIL_H_
#define ODML_LITERT_LITERT_TOOLS_DUMP_OPS_UTIL_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/core/model/model.h"

namespace litert::tools {

struct DumpOptions {
  std::optional<std::string> filter_opcode;
  bool unique = false;
  std::string output_dir = "/tmp/dump_ops";
  std::string filename_prefix = "";
};

struct DumpStats {
  int models_processed = 0;
  int total_ops_dumped = 0;
  absl::flat_hash_set<std::string> unique_op_signatures;
  absl::flat_hash_map<std::string, int> op_code_counts;
  std::vector<std::string> invalid_files;
};

absl::Status DumpOps(LiteRtModelT& model, const DumpOptions& options,
                     DumpStats* stats = nullptr);

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_DUMP_OPS_UTIL_H_
