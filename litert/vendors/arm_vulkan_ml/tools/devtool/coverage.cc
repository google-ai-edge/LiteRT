// Copyright 2026 The LiteRT Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/arm_vulkan_ml/tools/devtool/tool.h"
#include "litert/vendors/arm_vulkan_ml/tools/coverage_manifest.inc"
#include <iostream>
namespace litert::arm_vulkan_ml::tool {
Expected<void> Coverage(Json& report) {
  report = Json::parse(kCoverageManifest, nullptr, false);
  if (report.is_discarded()) return Error(kLiteRtStatusErrorRuntimeFailure, "Invalid compiled coverage manifest");
  size_t emitted=0, unreachable=0, emulator=0;
  for (const auto& op:report["operators"]) {
    emitted += op["implementation"] == "emitted";
    unreachable += op["implementation"] == "unreachable";
    emulator += op["emulator"] == "passed";
  }
  report["quantized_subset_count"]=report.contains("quantized_lowerings")?report["quantized_lowerings"].size():0;
  report["counts"]={{"total",report["operators"].size()},{"emitted",emitted},{"unreachable",unreachable},{"pending",report["operators"].size()-emitted-unreachable},{"recorded_emulator_qualified",emulator}};
  return {};
}
void PrintCoverage(const Json& report) {
  std::cout << report["counts"]["emitted"] << "/" << report["counts"]["total"] << " TOSA operators emitted; numerical/device qualification is separate.\n";
  for(const auto& op:report["operators"])std::cout << op["name"].get<std::string>() << ": " << op["implementation"].get<std::string>() << " — " << op["route"].get<std::string>() << "\n";
  if(report.contains("quantized_lowerings")){
    std::cout << "Quantized LiteRT subsets (qualification recorded separately):\n";
    for(const auto& op:report["quantized_lowerings"])std::cout << op["litert_operator"].get<std::string>() << ": " << op["constraints"].get<std::string>() << "\n";
  }
}
}
