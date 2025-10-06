// Copyright 2024 Google LLC.
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

// Simple tool to print info about a model's structure and ops.

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/tools/dump.h"
#include "litert/tools/tool_display.h"

ABSL_FLAG(std::string, model_path, "", "Model to analyze");
ABSL_FLAG(bool, no_ops, false,
          "Don't include per-op info in output (summary will still contain the "
          "number of ops).");
ABSL_FLAG(bool, only_summarize, false,
          "Only include the summary in the output.");

namespace litert::tools {
namespace {

using ::litert::internal::Dump;

class ModelAnalyzer {
 public:
  struct Summary {
    size_t num_subgraphs = 0;
    size_t num_ops = 0;
    size_t num_tensors = 0;
    size_t num_weights = 0;
    size_t weights_size = 0;
    bool fully_compiled = true;
  };

  explicit ModelAnalyzer(Model&& model) : model_(std::move(model)) {}

  Summary Summarize() {
    Summary summary;
    summary.num_subgraphs = Model().NumSubgraphs();
    for (const auto* sg : Model().Subgraphs()) {
      summary.num_ops += sg->Ops().size();
      for (const auto* op : sg->Ops()) {
        if (op->OpCode() != kLiteRtOpCodeTflCustom) {
          summary.fully_compiled = false;
        }
      }
      summary.num_tensors += sg->Tensors().size();
      for (const auto* t : sg->Tensors()) {
        auto buf = t->Weights().Buffer();
        summary.weights_size += buf.Size();
        if (buf.Size() > 0) {
          summary.num_weights++;
        }
      }
    }
    return summary;
  }

  void AnalyzeOp(std::ostream& out, size_t subgraph_idx, size_t op_idx) {
    Dump(Model().Subgraph(subgraph_idx).Op(op_idx), out);
  }

  void AnalyzeSubgraph(std::ostream& out, size_t subgraph_idx) {
    Dump(Model().Subgraph(subgraph_idx), out);
  }

  const LiteRtModelT& Model() { return *model_.Get(); }

 private:
  class Model model_;
};

void FormatSummary(std::ostream& out, const ModelAnalyzer::Summary& summary) {
  // clang-format off
  static constexpr absl::string_view kSummaryFormat =
      R"txt(
    Model Summary:
      Num Subgraphs:   %lu
      Num Ops:         %lu
      Num Activations: %lu
      Num Weights:     %lu
      Weights Size:    %s
      Fully Compiled:  %v

)txt";

  // clang-format on

  out << absl::StreamFormat(
      kSummaryFormat, summary.num_subgraphs, summary.num_ops,
      summary.num_tensors - summary.num_weights, summary.num_weights,
      HumanReadableSize(summary.weights_size), summary.fully_compiled);
}

Expected<void> AnalyzeModel(const std::string& model_path, bool no_ops,
                            bool only_summarize) {
  ToolDisplay display(std::cerr, "LITERT_MODEL_ANALYZE");
  DumpPreamble(display);
  auto scope = display.StartS("Model analysis");

  display.Labeled() << absl::StreamFormat(
      "no_ops=%d, only_summarize=%d, model_path=\"%s\"\n", no_ops,
      only_summarize, model_path);

  display.Start("Loading model");
  auto model_obj = Model::CreateFromFile(model_path);
  if (!model_obj) {
    display.Labeled() << absl::StreamFormat("Failed to load model: %s\n",
                                            model_obj.Error().Message());
    return model_obj.Error();
  }
  display.Done("Loading model");

  ModelAnalyzer analyzer(std::move(*model_obj));

  display.Start("Summarizing model");
  const auto summary = analyzer.Summarize();
  FormatSummary(display.Display(), summary);
  display.Done("Summarizing model");

  if (only_summarize) {
    return {};
  }

  display.Start("Analyzing graph");
  display.Display() << "\n";
  for (auto i = 0; i < analyzer.Model().NumSubgraphs(); ++i) {
    display.Display() << "  ";
    analyzer.AnalyzeSubgraph(display.Display(), i);
    if (no_ops) {
      continue;
    }
    for (auto j = 0; j < analyzer.Model().Subgraph(i).Ops().size(); ++j) {
      display.Display() << "    ";
      analyzer.AnalyzeOp(display.Display(), i, j);
    }
  }
  display.Display() << "\n";

  display.Done("Analyzing graph");
  return {};
}

}  // namespace
}  // namespace litert::tools

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto model_path = absl::GetFlag(FLAGS_model_path);
  const auto no_ops = absl::GetFlag(FLAGS_no_ops);
  const auto only_summarize = absl::GetFlag(FLAGS_only_summarize);

  return !litert::tools::AnalyzeModel(model_path, no_ops, only_summarize)
              .HasValue();
}
