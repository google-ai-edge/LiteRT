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

#include "litert/vendors/nvidia/experimental/megakernel/megakernel_lowering.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/litert_op_code.h"
#include "litert/compiler/cc/litert_model.h"

namespace litert::nvidia {
namespace {

// The sorted names of a subgraph's outputs: what a partition that holds the
// whole of a subgraph keeps of it.
std::vector<std::string> SortedOutputNames(
    const litert::compiler::Subgraph& subgraph) {
  std::vector<std::string> names;
  for (const auto& output : subgraph.Outputs()) {
    names.emplace_back(output.Name());
  }
  std::sort(names.begin(), names.end());
  return names;
}

struct Bodies {
  std::mutex mutex;
  std::vector<std::vector<std::string>> outputs;
};

Bodies& RememberedBodies() {
  static Bodies* const bodies = new Bodies;
  return *bodies;
}

}  // namespace

bool MegakernelRequested() {
  const char* value = std::getenv("LITERT_NVIDIA_MEGAKERNEL");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool IsMegakernelBody(const litert::compiler::Subgraph& subgraph,
                      const std::string& name) {
  if (name != "decode") return false;
  for (const auto& op : subgraph.Ops()) {
    if (op.Code() != kLiteRtOpCodeTflCustom) continue;
    const auto code = op.CustomCode();
    if (code && *code == "moe") return true;
  }
  return false;
}

void RememberMegakernelBody(const litert::compiler::Subgraph& subgraph) {
  Bodies& bodies = RememberedBodies();
  const std::lock_guard<std::mutex> lock(bodies.mutex);
  bodies.outputs.push_back(SortedOutputNames(subgraph));
}

bool IsMegakernelPartition(const litert::compiler::Subgraph& partition) {
  Bodies& bodies = RememberedBodies();
  const std::lock_guard<std::mutex> lock(bodies.mutex);
  if (bodies.outputs.empty()) return false;
  const std::vector<std::string> outputs = SortedOutputNames(partition);
  return std::find(bodies.outputs.begin(), bodies.outputs.end(), outputs) !=
         bodies.outputs.end();
}

}  // namespace litert::nvidia
