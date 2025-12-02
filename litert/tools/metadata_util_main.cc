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

#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/model/model.h"
#include "litert/tools/tool_display.h"

ABSL_FLAG(std::string, cmd, "view", "Command to run [add, view]");
ABSL_FLAG(std::string, model_path, "", "Model to extract bytecode from");
ABSL_FLAG(std::string, output_path, "",
          "Output model path (default: same as input)");
ABSL_FLAG(std::string, metadata_key, "", "Metadata key to add to the model");
ABSL_FLAG(std::string, metadata_value, "",
          "Metadata value to add to the model");
ABSL_FLAG(int, bytecode_alignment, 1,
          "Bytecode alignment to use for serialization");

namespace litert::tools {
namespace {

// Adds metadata to the model and serializes it to the output path.
// LiteRt internal user and friend library can follow API usage here to add
// metadata to a model and serialize it to a new file.
Expected<void> AddMetadata(const std::string& model_path) {
  const std::string metadata_key = absl::GetFlag(FLAGS_metadata_key);
  const std::string metadata_value = absl::GetFlag(FLAGS_metadata_value);
  std::string output_path = absl::GetFlag(FLAGS_output_path);

  ToolDisplay display(std::cerr, "LITERT_ADD_METADATA");
  display.Labeled() << "model_path: " << model_path << std::endl;
  display.Labeled() << "metadata_key: " << metadata_key << std::endl;
  display.Labeled() << "metadata_value: " << metadata_value << std::endl;
  display.Labeled() << "output_path: " << output_path << std::endl;

  LITERT_ASSIGN_OR_RETURN(auto model,
                          ExtendedModel::CreateFromFile(model_path));
  display.Labeled() << "Loading model done. Adding metadata..." << std::endl;
  model.AddMetadata(metadata_key, metadata_value);

  display.Labeled() << "Added metadata. Serializing model..." << std::endl;
  LiteRtModelSerializationOptions options;
  options.bytecode_alignment = absl::GetFlag(FLAGS_bytecode_alignment);
  LITERT_ASSIGN_OR_RETURN(auto serialized,
                          ExtendedModel::Serialize(std::move(model), options));

  display.Labeled() << "Serialized model. Writing to output..." << std::endl;
  std::ofstream output_stream(output_path, std::ios::binary | std::ios::out);
  output_stream.write(reinterpret_cast<const char*>(serialized.Data()),
                      serialized.Size());
  output_stream.close();

  display.Labeled() << "Done." << std::endl;
  return {};
}

// Views metadata in the model.
Expected<void> ViewMetadata(const std::string& model_path) {
  ToolDisplay display(std::cerr, "LITERT_VIEW_METADATA");
  display.Labeled() << "model_path: " << model_path << std::endl;
  LITERT_ASSIGN_OR_RETURN(auto model,
                          ExtendedModel::CreateFromFile(model_path));
  display.Labeled() << "Loading model done. Viewing metadata..." << std::endl;
  display.Labeled() << "------------------------------------------------"
                    << std::endl;
  for (auto it = model.Get()->MetadataBegin(); it != model.Get()->MetadataEnd();
       ++it) {
    LITERT_ASSIGN_OR_RETURN(const auto metadata_value,
                            model.Metadata(it->first));
    display.Labeled() << "Metadata key: " << it->first
                      << ", value: " << metadata_value.data() << std::endl;
  }
  return {};
}

}  // namespace
}  // namespace litert::tools

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string cmd = absl::GetFlag(FLAGS_cmd);
  if (model_path.empty()) {
    std::cerr << "Error: --model_path must be specified." << std::endl;
    return 1;
  }
  if (cmd == "add") {
    if (absl::GetFlag(FLAGS_metadata_key).empty()) {
      std::cerr << "Error: --metadata_key must be specified for 'add' command."
                << std::endl;
      return 1;
    }
    if (absl::GetFlag(FLAGS_metadata_value).empty()) {
      std::cerr
          << "Error: --metadata_value must be specified for 'add' command."
          << std::endl;
      return 1;
    }
    if (absl::GetFlag(FLAGS_output_path).empty()) {
      std::cerr << "Error: --output_path must be specified for 'add' command."
                << std::endl;
      return 1;
    }
    return !litert::tools::AddMetadata(model_path).HasValue();
  } else if (cmd == "view") {
    return !litert::tools::ViewMetadata(model_path).HasValue();
  } else {
    std::cerr << "Unknown command: " << cmd << std::endl;
    return 1;
  }
}
