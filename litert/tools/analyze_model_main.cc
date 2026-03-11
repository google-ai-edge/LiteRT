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

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/core/model/shape_inference.h"
#include "litert/tools/dump.h"
#include "litert/tools/tool_display.h"

ABSL_FLAG(std::string, model_path, "", "Path to the model file to analyze.");
ABSL_FLAG(bool, no_ops, false,
          "Exclude per-op details from the output (summary will still include "
          "op counts).");
ABSL_FLAG(bool, only_summarize, false,
          "Only output the model summary, skipping detailed graph analysis.");
ABSL_FLAG(bool, validate_shape, false,
          "Perform tensor shape validation using the Shape Inference Engine.");
ABSL_FLAG(std::string, input_dir, "",
          "Directory containing .tflite models to analyze in batch mode.");
ABSL_FLAG(std::string, output_csv, "",
          "Path to the output CSV file for batch analysis results.");

namespace litert::tools {
namespace {

namespace fs = std::filesystem;
using ::litert::internal::Dump;

// Container for analysis results of a single model.
struct AnalysisResult {
  std::string model_name;
  bool load_success = false;
  std::string error_msg;

  // Model Statistics
  size_t num_subgraphs = 0;
  size_t num_ops = 0;
  size_t num_tensors = 0;
  size_t num_weights = 0;
  size_t weights_size = 0;
  bool fully_compiled = true;

  // Validation Results
  bool validation_performed = false;
  bool validation_passed = false;
  std::string validation_failure_info;
};

// Helper class for writing analysis results to a CSV file.
class CsvWriter {
 public:
  explicit CsvWriter(std::ostream& out, bool include_validation_cols)
      : out_(out), include_validation_cols_(include_validation_cols) {
    WriteAndEscape("Model");
    WriteAndEscape("LoadSuccess");
    WriteAndEscape("Subgraphs");
    WriteAndEscape("Ops");
    WriteAndEscape("Tensors");
    WriteAndEscape("Weights");
    WriteAndEscape("WeightsSize");
    WriteAndEscape("FullyCompiled");
    if (include_validation_cols_) {
      WriteAndEscape("ShapeValidation");
      WriteAndEscape("FailingOpInfo");
    }
    WriteAndEscape("ErrorMsg");
    out_ << "\n";
  }

  void WriteRow(const AnalysisResult& res) {
    WriteAndEscape(res.model_name);
    WriteAndEscape(res.load_success ? "true" : "false");

    if (res.load_success) {
      out_ << "," << res.num_subgraphs;
      out_ << "," << res.num_ops;
      out_ << "," << (res.num_tensors - res.num_weights);
      out_ << "," << res.num_weights;
      out_ << "," << res.weights_size;
      out_ << "," << (res.fully_compiled ? "true" : "false");

      if (include_validation_cols_) {
        if (res.validation_performed) {
          WriteAndEscape(res.validation_passed ? "PASS" : "FAIL");
          WriteAndEscape(res.validation_failure_info);
        } else {
          out_ << ",,";
        }
      }
      out_ << ",";  // Empty ErrorMsg column
    } else {
      // Fill empty columns for stats
      out_ << ",,,,,,,";
      if (include_validation_cols_) {
        out_ << ",,";
      }
      WriteAndEscape(res.error_msg);
    }
    out_ << "\n";
  }

 private:
  void WriteAndEscape(const std::string& s) {
    out_ << ",";
    bool needs_quotes = s.find_first_of(",\"\n") != std::string::npos;
    if (!needs_quotes) {
      out_ << s;
      return;
    }

    out_ << '"';
    for (char c : s) {
      if (c == '"') {
        out_ << "\"\"";
      } else {
        out_ << c;
      }
    }
    out_ << '"';
  }

  std::ostream& out_;
  bool include_validation_cols_;
};

// Core logic for analyzing a loaded model.
class ModelAnalyzer {
 public:
  explicit ModelAnalyzer(const LiteRtModelT& model) : model_(model) {}

  void CollectStats(AnalysisResult& result) const {
    result.num_subgraphs = model_.NumSubgraphs();
    result.fully_compiled = true;

    for (const auto* sg : model_.Subgraphs()) {
      result.num_ops += sg->Ops().size();
      for (const auto* op : sg->Ops()) {
        if (op->OpCode() != kLiteRtOpCodeTflCustom) {
          result.fully_compiled = false;
        }
      }
      result.num_tensors += sg->Tensors().size();
      for (const auto* t : sg->Tensors()) {
        auto buf = t->Weights().Buffer();
        result.weights_size += buf.Size();
        if (buf.Size() > 0) {
          result.num_weights++;
        }
      }
    }
  }

  void ValidateShapes(AnalysisResult& result) const {
    result.validation_performed = true;
    // Note: const_cast is required because ShapeInferenceEngine currently
    // takes a non-const pointer, even for validation.
    litert::internal::ShapeInferenceEngine engine(
        const_cast<LiteRtModelT*>(&model_));
    LiteRtOp failing_op = nullptr;
    auto status = engine.InferShapes(/*validation_only=*/true, &failing_op);

    if (status == kLiteRtStatusOk) {
      result.validation_passed = true;
    } else {
      result.validation_passed = false;
      std::stringstream ss;
      ss << "Status " << status;
      if (failing_op) {
        ss << ", Op: ";
        Dump(*failing_op, ss);
      }
      result.validation_failure_info = ss.str();
    }
  }

  void DumpGraph(std::ostream& out, bool no_ops) const {
    out << "\n";
    for (size_t i = 0; i < model_.NumSubgraphs(); ++i) {
      out << "  ";
      Dump(model_.Subgraph(i), out);
      if (no_ops) {
        continue;
      }
      const auto& ops = model_.Subgraph(i).Ops();
      for (size_t j = 0; j < ops.size(); ++j) {
        out << "    ";
        Dump(*ops[j], out);
      }
    }
    out << "\n";
  }

 private:
  const LiteRtModelT& model_;
};

// Prints the analysis result to the standard output in a human-readable format.
void PrintSummary(std::ostream& out, const AnalysisResult& summary) {
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

  if (summary.validation_performed) {
    out << "      Shape Validation: "
        << (summary.validation_passed ? "PASS" : "FAIL") << "\n";
    if (!summary.validation_passed) {
      out << "      Failing Op Info: " << summary.validation_failure_info
          << "\n";
    }
  }
  out << "\n";
}

// Analyzes a single model and prints details to stdout/stderr.
// This is used when the tool is run in single-model mode.
Expected<void> RunSingleModelAnalysis(const std::string& model_path,
                                      bool no_ops, bool only_summarize,
                                      bool validate_shape) {
  ToolDisplay display(std::cerr, "LITERT_MODEL_ANALYZE");
  DumpPreamble(display);
  auto scope = display.StartS("Model analysis");

  display.Labeled() << absl::StreamFormat(
      "no_ops=%d, only_summarize=%d, validate_shape=%d, model_path=\"%s\"\n",
      no_ops, only_summarize, validate_shape, model_path);

  display.Start("Loading model");
  auto model_obj = Model::CreateFromFile(model_path);
  if (!model_obj) {
    display.Labeled() << absl::StreamFormat("Failed to load model: %s\n",
                                            model_obj.Error().Message());
    return model_obj.Error();
  }
  display.Done("Loading model");

  AnalysisResult result;
  result.model_name = model_path;
  result.load_success = true;

  ModelAnalyzer analyzer(*model_obj->Get());
  analyzer.CollectStats(result);

  if (validate_shape) {
    display.Start("Validating shapes");
    analyzer.ValidateShapes(result);
    if (!result.validation_passed) {
      display.Labeled() << "Shape validation failed: "
                        << result.validation_failure_info << "\n";
    } else {
      display.Labeled() << "Shape validation passed.\n";
    }
    display.Done("Validating shapes");
  }

  display.Start("Summarizing model");
  PrintSummary(display.Display(), result);
  display.Done("Summarizing model");

  if (only_summarize) {
    return {};
  }

  display.Start("Analyzing graph");
  analyzer.DumpGraph(display.Display(), no_ops);
  display.Done("Analyzing graph");

  return {};
}

// Analyzes a model without verbose ABSL_LOGging, suitable for batch processing.
AnalysisResult AnalyzeModelSilent(const std::string& model_path,
                                  bool validate_shape) {
  AnalysisResult result;
  result.model_name = fs::path(model_path).filename().string();

  auto model_obj = Model::CreateFromFile(model_path);
  if (!model_obj) {
    result.load_success = false;
    result.error_msg = model_obj.Error().Message();
    return result;
  }
  result.load_success = true;

  ModelAnalyzer analyzer(*model_obj->Get());
  analyzer.CollectStats(result);

  if (validate_shape) {
    analyzer.ValidateShapes(result);
  }
  return result;
}

// Driver for batch analysis mode.
void RunBatchAnalysis(const std::string& input_dir,
                      const std::string& output_csv, bool validate_shape) {
  std::vector<std::string> tflite_files;
  std::error_code ec;
  auto iter = fs::directory_iterator(input_dir, ec);

  if (ec) {
    ABSL_LOG(ERROR) << "Error accessing directory " << input_dir << ": "
                    << ec.message();
    return;
  }

  for (const auto& entry : iter) {
    if (entry.is_regular_file() && entry.path().extension() == ".tflite") {
      tflite_files.push_back(entry.path().string());
    }
  }

  if (tflite_files.empty()) {
    ABSL_LOG(WARNING) << "No .tflite files found in " << input_dir;
    return;
  }

  ABSL_LOG(INFO) << "Found " << tflite_files.size() << " models. Analyzing...";

  std::ofstream csv_file;
  std::unique_ptr<CsvWriter> csv_writer;

  if (!output_csv.empty()) {
    csv_file.open(output_csv);
    if (!csv_file.is_open()) {
      ABSL_LOG(ERROR) << "Failed to open output CSV: " << output_csv;
      return;
    }
    csv_writer = std::make_unique<CsvWriter>(csv_file, validate_shape);
  }

  for (const auto& file : tflite_files) {
    ABSL_LOG(INFO) << "Processing " << fs::path(file).filename().string()
                   << " ... ";

    AnalysisResult res = AnalyzeModelSilent(file, validate_shape);

    if (res.load_success) {
      ABSL_LOG(INFO) << "Done";
    } else {
      ABSL_LOG(INFO) << "Failed: " << res.error_msg;
    }

    if (csv_writer) {
      csv_writer->WriteRow(res);
    }
  }

  if (!output_csv.empty()) {
    ABSL_LOG(INFO) << "Results written to " << output_csv;
  }
}

}  // namespace
}  // namespace litert::tools

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const auto model_path = absl::GetFlag(FLAGS_model_path);
  const auto input_dir = absl::GetFlag(FLAGS_input_dir);
  const auto output_csv = absl::GetFlag(FLAGS_output_csv);
  const auto no_ops = absl::GetFlag(FLAGS_no_ops);
  const auto only_summarize = absl::GetFlag(FLAGS_only_summarize);
  const auto validate_shape = absl::GetFlag(FLAGS_validate_shape);

  if (!input_dir.empty()) {
    litert::tools::RunBatchAnalysis(input_dir, output_csv, validate_shape);
    return 0;
  } else {
    if (model_path.empty()) {
      ABSL_LOG(ERROR)
          << "Either --model_path or --input_dir must be specified.";
      return 1;
    }
    return !litert::tools::RunSingleModelAnalysis(
                model_path, no_ops, only_summarize, validate_shape)
                .HasValue();
  }
}
