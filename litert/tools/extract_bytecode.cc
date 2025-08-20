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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/tools/tool_display.h"

ABSL_FLAG(std::string, model_path, "", "Model to extract bytecode from");
ABSL_FLAG(std::string, output_dir, "/tmp", "Output directory for bytecode");

namespace litert::tools {
namespace {

Expected<void> ExtractBytecode(const std::string& model_path) {
  ToolDisplay display(std::cerr, "LITERT_EXTRACT_BYTECODE");

  std::ifstream infile(model_path, std::ios::binary | std::ios::ate);
  if (!infile) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      absl::StrCat("Cannot open source file: ", model_path));
  }

  // Get the file size and allocate a buffer.
  std::streamsize file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);
  std::vector<char> buffer(file_size);

  // Read the entire file into the buffer.
  if (!infile.read(buffer.data(), file_size)) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("Failed to read from source file: ", model_path));
  }
  infile.close();

  // Load the model from the buffer.
  BufferRef<uint8_t> buffer_ref(buffer.data(), buffer.size());
  LITERT_ASSIGN_OR_ABORT(auto model_obj, Model::CreateFromBuffer(buffer_ref));

  size_t dispatch_count = 0;
  for (auto i = 0; i < model_obj.NumSubgraphs(); ++i) {
    auto subgraph = model_obj.Subgraph(i);
    for (auto& op : subgraph->Ops()) {
      if (op.Code() != kLiteRtOpCodeTflCustom) {
        continue;
      }
      auto dispatch_opts =
          internal::GetDispatchOpOptions(op.Get()->CustomOptions());
      if (dispatch_opts.bytecode_offset == 0) {
        continue;
      }
      display.Labeled() << absl::StreamFormat(
          "bytecode_offset=%d, bytecode_size=%d, name=\"%s\"\n",
          dispatch_opts.bytecode_offset, dispatch_opts.bytecode_size,
          dispatch_opts.name);
      if (dispatch_opts.bytecode_offset + dispatch_opts.bytecode_size >
          buffer.size()) {
        return Unexpected(kLiteRtStatusErrorInvalidArgument,
                          "bytecode offset and size is out of bounds.");
      }
      std::string output_filepath =
          absl::StrCat(absl::GetFlag(FLAGS_output_dir), "/", dispatch_opts.name,
                       "_", dispatch_count, ".bin");
      std::ofstream outfile(output_filepath, std::ios::binary);
      if (!outfile) {
        return Unexpected(
            kLiteRtStatusErrorInvalidArgument,
            absl::StrCat("Failed to create output file: ", output_filepath));
      }
      outfile.write(buffer.data() + dispatch_opts.bytecode_offset,
                    dispatch_opts.bytecode_size);
      if (!outfile.good()) {
        return Unexpected(
            kLiteRtStatusErrorInvalidArgument,
            absl::StrCat("Failed to write to output file: ", output_filepath));
      }
      outfile.close();
      display.Labeled() << "  -> Wrote " << dispatch_opts.bytecode_size
                        << " bytes to '" << output_filepath << "' (from offset "
                        << dispatch_opts.bytecode_offset << ")" << std::endl;
      dispatch_count++;
    }
  }
  display.Labeled() << "Done." << std::endl;
  return {};
}
}  // namespace
}  // namespace litert::tools

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  return !litert::tools::ExtractBytecode(model_path).HasValue();
}
