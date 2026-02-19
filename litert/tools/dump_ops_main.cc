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

#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "litert/core/model/model_load.h"
#include "litert/tools/dump_ops_util.h"

ABSL_FLAG(std::string, model_path, "", "Path to the tflite model");
ABSL_FLAG(std::string, input_dir, "",
          "Directory containing tflite models for batch processing");
ABSL_FLAG(std::string, output_dir, "/tmp/dump_ops", "Output directory");
ABSL_FLAG(std::string, filter_opcode, "",
          "Only dump ops with this opcode (substring match)");
ABSL_FLAG(bool, unique, false, "Only dump unique ops");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string input_dir = absl::GetFlag(FLAGS_input_dir);
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  std::string filter_opcode = absl::GetFlag(FLAGS_filter_opcode);
  bool unique = absl::GetFlag(FLAGS_unique);

  litert::tools::DumpOptions options;
  options.output_dir = output_dir;
  if (!filter_opcode.empty()) {
    options.filter_opcode = filter_opcode;
  }
  options.unique = unique;

  if (!input_dir.empty()) {
    litert::tools::DumpStats stats;
    std::error_code dir_ec;
    auto dir_it = std::filesystem::directory_iterator(input_dir, dir_ec);
    if (dir_ec) {
      ABSL_LOG(ERROR) << "Error iterating directory " << input_dir << ": "
                      << dir_ec.message();
      return 1;
    }

    for (auto end = std::filesystem::directory_iterator(); dir_it != end;
         dir_it.increment(dir_ec)) {
      if (dir_ec) {
        ABSL_LOG(ERROR) << "Error iterating directory: " << dir_ec.message();
        break;
      }

      const auto& entry = *dir_it;
      std::error_code file_ec;
      if (!entry.is_regular_file(file_ec)) {
        if (file_ec) {
          ABSL_LOG(ERROR) << "Error accessing file " << entry.path().string()
                          << ": " << file_ec.message();
        }
        continue;
      }

      if (entry.path().extension() != ".tflite") {
        continue;
      }

      std::string model_path = entry.path().string();
      auto model_res = litert::internal::LoadModelFromFile(model_path);
      if (!model_res) {
        ABSL_LOG(ERROR) << "Skipping invalid file " << model_path << ": "
                        << model_res.Error().Message();
        stats.invalid_files.push_back(model_path);
        continue;
      }
      stats.models_processed++;
      options.filename_prefix =
          entry.path().filename().replace_extension("").string();
      auto status = litert::tools::DumpOps(**model_res, options, &stats);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to dump ops for " << model_path << ": "
                        << status.message();
      }
    }

    std::cout << "Batch Dump Report:" << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << "Models processed: " << stats.models_processed << std::endl;
    std::cout << "Total single-op models dumped: " << stats.total_ops_dumped
              << std::endl;
    std::cout << "Unique op codes dumped: " << stats.op_code_counts.size()
              << std::endl;
    std::cout << "Op code counts:" << std::endl;
    for (const auto& pair : stats.op_code_counts) {
      std::cout << "  " << pair.first << ": " << pair.second << std::endl;
    }
    if (!stats.invalid_files.empty()) {
      std::cout << "Invalid files (" << stats.invalid_files.size()
                << "):" << std::endl;
      for (const auto& file : stats.invalid_files) {
        std::cout << "  " << file << std::endl;
      }
    }
    return 0;
  }

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

  auto status = litert::tools::DumpOps(**model_res, options);

  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to dump ops: " << status.message();
    return 1;
  }

  return 0;
}
