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

#include <cstddef>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_load.h"
#include "litert/core/model/model_serialize.h"
#include "litert/tools/dump.h"
#include "litert/tools/outliner/outliner_util.h"

ABSL_FLAG(std::string, model_path, "", "(Required) Path to the tflite model");
ABSL_FLAG(std::string, output_path, "modified.tflite",
          "Output path for the modified model");
ABSL_FLAG(size_t, subgraph_index, 0, "Index of the subgraph to outline");
ABSL_FLAG(std::string, start_tensors, "",
          "Comma-separated list of start tensor names");
ABSL_FLAG(std::string, end_tensors, "",
          "Comma-separated list of end tensor names");
ABSL_FLAG(std::string, composite_name, "outlined_composite",
          "Name for the composite op");
ABSL_FLAG(std::string, attributes, "",
          "Comma-separated key:value pairs for composite attributes");
ABSL_FLAG(bool, interactive, false, "Run in interactive mode");

namespace litert::tools {

void PrintGraph(const LiteRtModelT& model) {
  std::cout << "\n--- Model Graph ---\n";
  for (size_t i = 0; i < model.NumSubgraphs(); ++i) {
    const auto& sg = *model.Subgraphs()[i];
    std::cout << "Subgraph " << i << ": (" << &sg << ")\n";
    for (const auto& op : sg.Ops()) {
      std::cout << "  [" << op->OpIndex() << "] ";
      litert::internal::Dump(*op, std::cout);
      std::cout << "    Inputs: ";
      for (const auto* t : op->Inputs())
        std::cout << (t ? t->Name() : "null") << "(" << t << ") ";
      std::cout << "\n    Outputs: ";
      for (const auto* t : op->Outputs())
        std::cout << (t ? t->Name() : "null") << "(" << t << ") ";
      std::cout << "\n";
    }
    std::cout << "  Subgraph Outputs: ";
    for (const auto* t : sg.Outputs())
      std::cout << (t ? t->Name() : "null") << "(" << t << ") ";
    std::cout << "\n";
  }
  std::cout << "-------------------\n";
}

void SanityCheck(const LiteRtModelT& model) {
  std::cout << "\n--- IR Sanity Check ---\n";
  bool consistent = true;
  for (size_t s_idx = 0; s_idx < model.NumSubgraphs(); ++s_idx) {
    const auto& sg = *model.Subgraphs()[s_idx];
    for (const auto& op : sg.Ops()) {
      for (size_t i = 0; i < op->Inputs().size(); ++i) {
        auto* t = op->Inputs()[i];
        if (!t) continue;
        bool found_user = false;
        for (size_t j = 0; j < t->Users().size(); ++j) {
          if (t->Users()[j] == op && t->UserArgInds()[j] == i) {
            found_user = true;
            break;
          }
        }
        if (!found_user) {
          std::cout
              << "  Inconsistent IR: Subgraph " << s_idx << " Op ["
              << op->OpIndex() << "] uses tensor " << t->Name()
              << " but tensor does not list it as user at correct index.\n";
          consistent = false;
        }
      }
    }
    for (const auto& t : sg.Tensors()) {
      for (size_t i = 0; i < t->Users().size(); ++i) {
        auto* user = t->Users()[i];
        auto arg_ind = t->UserArgInds()[i];
        if (arg_ind >= user->Inputs().size() || user->Inputs()[arg_ind] != t) {
          std::cout << "  Inconsistent IR: Subgraph " << s_idx << " Tensor "
                    << t->Name() << " lists Op [" << user->OpIndex()
                    << "] as user at index " << arg_ind
                    << " but Op does not use it there.\n";
          consistent = false;
        }
      }
    }
  }

  // Check Signatures
  for (const auto& sig : model.Signatures()) {
    for (size_t i = 0; i < sig->InputNames().size(); ++i) {
      const auto* t = sig->GetInputTensor(i);
      bool found = false;
      for (size_t s_idx = 0; s_idx < model.NumSubgraphs(); ++s_idx) {
        for (const auto& sg_t : model.Subgraphs()[s_idx]->Tensors()) {
          if (sg_t == t) {
            found = true;
            break;
          }
        }
        if (found) break;
      }
      if (!found) {
        std::cout << "  Signature Error: Input tensor '" << t->Name()
                  << "' not found in any subgraph.\n";
        consistent = false;
      }
    }
    for (size_t i = 0; i < sig->OutputNames().size(); ++i) {
      const auto* t = sig->GetOutputTensor(i);
      bool found = false;
      for (size_t s_idx = 0; s_idx < model.NumSubgraphs(); ++s_idx) {
        for (const auto& sg_t : model.Subgraphs()[s_idx]->Tensors()) {
          if (sg_t == t) {
            found = true;
            break;
          }
        }
        if (found) break;
      }
      if (!found) {
        std::cout << "  Signature Error: Output tensor '" << t->Name()
                  << "' not found in any subgraph.\n";
        consistent = false;
      }
    }
  }

  if (consistent) {
    std::cout << "  IR is consistent.\n";
  }
  std::cout << "-----------------------\n";
}

int RunInteractive(LiteRtModelT& model) {
  PrintGraph(model);
  std::cout << "\nInteractive mode placeholder.\n";
  return 0;
}

}  // namespace litert::tools

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    std::cerr << "Please provide --model_path" << std::endl;
    return 1;
  }

  auto model_res = litert::internal::LoadModelFromFile(model_path);
  if (!model_res) {
    ABSL_LOG(ERROR) << "Failed to load model: " << model_res.Error().Message();
    return 1;
  }
  auto& model = **model_res;

  if (absl::GetFlag(FLAGS_interactive)) {
    return litert::tools::RunInteractive(model);
  }

  litert::tools::OutlinerOptions options;
  options.start_tensors = absl::StrSplit(absl::GetFlag(FLAGS_start_tensors),
                                         ',', absl::SkipEmpty());
  options.end_tensors =
      absl::StrSplit(absl::GetFlag(FLAGS_end_tensors), ',', absl::SkipEmpty());
  options.composite_name = absl::GetFlag(FLAGS_composite_name);

  std::string attrs_str = absl::GetFlag(FLAGS_attributes);
  if (!attrs_str.empty()) {
    for (absl::string_view pair :
         absl::StrSplit(attrs_str, absl::ByAnyChar(", "), absl::SkipEmpty())) {
      std::vector<std::string> kv = absl::StrSplit(pair, ':');
      if (kv.size() == 2) {
        options.attributes[kv[0]] = kv[1];
      } else {
        options.attributes[pair] = "";
      }
    }
  }

  auto status = litert::tools::OutlineSubgraph(
      model, absl::GetFlag(FLAGS_subgraph_index), options);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Outlining failed: " << status.message();
    return 1;
  }

  litert::tools::SanityCheck(model);

  // Debug connections
  for (const auto& op : model.Subgraphs()[0]->Ops()) {
    if (op->OpCode() == kLiteRtOpCodeShloComposite) {
      std::cout << "\n--- SHLO_COMPOSITE Outputs and Users ---\n";
      for (size_t i = 0; i < op->Outputs().size(); ++i) {
        const auto* t = op->Outputs()[i];
        std::cout << "Output [" << i << "]: " << t->Name()
                  << " (Users: " << t->Users().size() << ")\n";
        for (const auto* user : t->Users()) {
          std::cout << "  User Op Index: [" << user->OpIndex() << "] ";
          litert::internal::Dump(*user, std::cout);
          std::cout << "\n    User Op All Inputs:\n";
          for (size_t k = 0; k < user->Inputs().size(); ++k) {
            const auto* in_t = user->Inputs()[k];
            std::cout << "      Input [" << k
                      << "]: " << (in_t ? in_t->Name() : "null") << "\n";
          }
        }
      }
      std::cout << "----------------------------------------\n";
    }
  }

  auto serialize_res = litert::internal::SerializeModel(std::move(**model_res));
  if (!serialize_res) {
    ABSL_LOG(ERROR) << "Failed to serialize model: "
                    << serialize_res.Error().Message();
    return 1;
  }

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  std::ofstream out(output_path, std::ios::binary);
  if (!out) {
    ABSL_LOG(ERROR) << "Failed to open output path: " << output_path;
    return 1;
  }
  out.write(reinterpret_cast<const char*>(serialize_res->Data()),
            serialize_res->Size());
  out.close();

  std::cout << "Successfully outlined subgraph and saved to: " << output_path
            << "\n";

  return 0;
}
