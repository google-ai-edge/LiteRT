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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/arm_vulkan_ml/compiler/op_registry.h"
#include "litert/vendors/arm_vulkan_ml/tools/devtool/devtool.pb.h"
#include "litert/vendors/arm_vulkan_ml/tools/devtool/tool.h"
#include "tflite/schema/schema_generated.h"

namespace litert::arm_vulkan_ml::tool {
Expected<void> Coverage(CoverageReport& report) {
  report.Clear();
  report.set_schema_version(1);
  report.set_operator_set("tflite");
  report.set_scope(
      "Shared compiler registration: principal data input/first result "
      "types and analysis routes; shape/options/other operands require "
      "inspect; registration is not numerical or device qualification");
  report.set_global_constraints(
      "Static dense data; physical rank<=6 and dimensions<=8192 "
      "(device maxTensorDimensionCount limits queried via "
      "VK_ARM_tensors at dispatch); bounded element counts; scalar I/O in "
      "registered decompositions; COMPLEX64 uses a trailing component "
      "dimension; constant controls; per-tensor affine INT8 and "
      "symmetric per-tensor INT16 (EXT-INT16) activations");
  size_t implemented = 0;
  size_t missing = 0;
  size_t excluded = 0;
  size_t variants = 0;
  for (auto code : tflite::EnumValuesBuiltinOperator()) {
    const std::string name = tflite::EnumNameBuiltinOperator(code);
    const auto* registration =
        FindOperationRegistration(static_cast<LiteRtOpCode>(code));
    const bool outside = name.rfind("STABLEHLO_", 0) == 0 || name == "CUSTOM" ||
                         name == "DELEGATE" ||
                         name == "PLACEHOLDER_FOR_GREATER_OP_CODES";
    auto* row = report.add_operators();
    row->set_name(name);
    row->set_opcode(static_cast<int32_t>(code));
    row->set_implementation(registration ? "implemented_subset"
                            : outside    ? "excluded"
                                         : "not_implemented");
    row->set_device_support("query_required");
    if (registration) {
      ++implemented;
      row->set_data_input(registration->data_input);
      for (const auto& variant : registration->variants) {
        auto* out = row->add_variants();
        out->set_data_input_type(RegisteredElementTypeName(variant.input_type));
        out->set_output_type(RegisteredElementTypeName(variant.output_type));
        out->set_route(LoweringRouteName(variant.route));
        out->set_constraints(variant.constraints);
        ++variants;
      }
    } else {
      outside ? ++excluded : ++missing;
      row->set_reason(outside ? "Outside TFLite builtin lowering scope"
                              : "No registered TFLite lowering");
    }
  }
  for (const auto& registration : GetNormalizationRegistrations()) {
    auto* composite = report.add_named_composites();
    composite->set_name(registration.name);
    composite->set_version(registration.version);
    composite->set_data_input_type("FP32");
    composite->set_output_type("FP32");
    composite->set_constraints(registration.constraints);
    composite->set_device_support("query_required");
  }
  auto* counts = report.mutable_counts();
  counts->set_total(implemented + missing);
  counts->set_implemented_subset(implemented);
  counts->set_not_implemented(missing);
  counts->set_excluded(excluded);
  counts->set_registered_type_pairs(variants);
  counts->set_schema_entries(report.operators_size());
  counts->set_named_composites(report.named_composites_size());
  return {};
}

void PrintCoverage(const CoverageReport& report) {
  std::cout << report.counts().implemented_subset() << "/"
            << report.counts().total()
            << " TFLite builtins have registered subsets ("
            << report.counts().registered_type_pairs()
            << " data/result type pairs); " << report.counts().excluded()
            << " schema entries excluded.\n"
            << report.scope() << '\n'
            << report.global_constraints() << '\n';
  for (const auto& op : report.operators()) {
    if (op.implementation() == "excluded") continue;
    std::cout << op.name() << ": " << op.implementation();
    if (op.has_data_input()) {
      std::cout << " (data input " << op.data_input() << ")";
    }
    std::cout << '\n';
    for (const auto& variant : op.variants()) {
      std::cout << "  " << variant.data_input_type() << " -> "
                << variant.output_type() << " (" << variant.route()
                << "): " << variant.constraints() << '\n';
    }
  }
  std::cout << "Named normalization composites (outside builtin counts):\n";
  for (const auto& row : report.named_composites()) {
    std::cout << "  " << row.name() << " v" << row.version() << ": "
              << row.constraints() << '\n';
  }
}
}  // namespace litert::arm_vulkan_ml::tool
