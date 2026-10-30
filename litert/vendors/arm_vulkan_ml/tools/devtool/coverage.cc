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
#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"
#include "litert/vendors/arm_vulkan_ml/tools/coverage_manifest.inc"
#include <iostream>
namespace litert::arm_vulkan_ml::tool {
namespace {
Expected<void> TfliteCoverage(Json& report) {
  report = {{"schema_version", 1}, {"operator_set", "tflite"},
      {"scope", "Shared compiler registration: principal data input/first result types and analysis routes; shape/options/other operands require inspect; registration is not numerical or device qualification"},
      {"global_constraints", "Static dense data; physical rank<=6 and dimensions<=8192; bounded element counts; scalar I/O in registered decompositions; COMPLEX64 uses a trailing component dimension; constant controls; per-tensor affine INT8 activations"},
      {"operators", Json::array()}};
  size_t implemented = 0, missing = 0, excluded = 0, variants = 0;
  for (auto code : tflite::EnumValuesBuiltinOperator()) {
    const std::string name = tflite::EnumNameBuiltinOperator(code);
    const auto* registration = FindOperationRegistration(static_cast<LiteRtOpCode>(code));
    const bool outside = name.rfind("STABLEHLO_", 0) == 0 || name == "CUSTOM" ||
        name == "DELEGATE" || name == "PLACEHOLDER_FOR_GREATER_OP_CODES";
    Json row = {{"name", name}, {"opcode", static_cast<int>(code)},
        {"implementation", registration ? "implemented_subset" : outside ? "excluded" : "not_implemented"},
        {"device_support", "query_required"}, {"variants", Json::array()}};
    if (registration) {
      ++implemented;
      row["data_input"] = registration->data_input;
      for (const auto& variant : registration->variants) {
        row["variants"].push_back({
            {"data_input_type", RegisteredElementTypeName(variant.input_type)},
            {"output_type", RegisteredElementTypeName(variant.output_type)},
            {"route", LoweringRouteName(variant.route)},
            {"constraints", variant.constraints}});
        ++variants;
      }
    } else {
      outside ? ++excluded : ++missing;
      row["reason"] = outside ? "Outside TFLite builtin lowering scope" :
          "No registered TFLite lowering";
    }
    report["operators"].push_back(std::move(row));
  }
  report["counts"] = {{"total", implemented + missing},
      {"implemented_subset", implemented}, {"not_implemented", missing},
      {"excluded", excluded}, {"registered_type_pairs", variants},
      {"schema_entries", report["operators"].size()}};
  return {};
}
}  // namespace

Expected<void> Coverage(Json& report, const std::string& op_set) {
  if (op_set == "tflite") return TfliteCoverage(report);
  if (op_set == "all") {
    Json tosa, tflite;
    auto status = Coverage(tosa, "tosa");
    if (!status) return status;
    status = TfliteCoverage(tflite);
    if (!status) return status;
    report = {{"schema_version", 1}, {"operator_set", "all"},
        {"tosa", std::move(tosa)}, {"tflite", std::move(tflite)}};
    return {};
  }
  if (op_set != "tosa")
    return Error(kLiteRtStatusErrorInvalidArgument, "op_set must be tosa, tflite or all");
  report = Json::parse(kCoverageManifest, nullptr, false);
  if (report.is_discarded()) return Error(kLiteRtStatusErrorRuntimeFailure, "Invalid compiled coverage manifest");
  report["operator_set"] = "tosa";
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
  if (report.value("operator_set", "tosa") == "all") {
    PrintCoverage(report["tosa"]);
    std::cout << '\n';
    PrintCoverage(report["tflite"]);
    return;
  }
  if (report.value("operator_set", "tosa") == "tflite") {
    std::cout << report["counts"]["implemented_subset"] << "/"
              << report["counts"]["total"] << " TFLite builtins have registered subsets ("
              << report["counts"]["registered_type_pairs"] << " data/result type pairs); "
              << report["counts"]["excluded"] << " schema entries excluded.\n"
              << report["scope"].get<std::string>() << '\n'
              << report["global_constraints"].get<std::string>() << '\n';
    for (const auto& op : report["operators"]) {
      if (op["implementation"] == "excluded") continue;
      std::cout << op["name"].get<std::string>() << ": "
                << op["implementation"].get<std::string>();
      if (op.contains("data_input"))
        std::cout << " (data input " << op["data_input"] << ")";
      std::cout << '\n';
      for (const auto& variant : op["variants"]) {
        std::cout << "  " << variant["data_input_type"].get<std::string>()
                  << " -> " << variant["output_type"].get<std::string>()
                  << " (" << variant["route"].get<std::string>() << "): "
                  << variant["constraints"].get<std::string>() << '\n';
      }
    }
    return;
  }
  std::cout << report["counts"]["emitted"] << "/" << report["counts"]["total"] << " TOSA operators emitted; numerical/device qualification is separate.\n";
  for(const auto& op:report["operators"])std::cout << op["name"].get<std::string>() << ": " << op["implementation"].get<std::string>() << " — " << op["route"].get<std::string>() << "\n";
  if(report.contains("quantized_lowerings")){
    std::cout << "Quantized LiteRT subsets (qualification recorded separately):\n";
    for(const auto& op:report["quantized_lowerings"])std::cout << op["litert_operator"].get<std::string>() << ": " << op["constraints"].get<std::string>() << "\n";
  }
}
}
