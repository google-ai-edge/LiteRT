/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "base/init_google.h"
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"

// Define command-line flags
ABSL_FLAG(std::string, model_file, "",
          "The path to the TFLite embedding model file.");
ABSL_FLAG(std::string, token_ids, "", "A comma-separated list of token IDs.");

int main(int argc, char* argv[]) {
  InitGoogle(argv[0], &argc, &argv, true);
  absl::ParseCommandLine(argc, argv);

  // Validate that the required flags are provided.
  if (absl::GetFlag(FLAGS_model_file).empty()) {
    ABSL_LOG(FATAL) << "--model_file flag must be provided.";
  }
  if (absl::GetFlag(FLAGS_token_ids).empty()) {
    ABSL_LOG(FATAL) << "--token_ids flag must be provided.";
  }

  // Parse the token IDs from the command-line flag.
  std::vector<int> token_ids;
  std::vector<std::string> token_id_strs =
      absl::StrSplit(absl::GetFlag(FLAGS_token_ids), ',');
  for (const auto& id_str : token_id_strs) {
    int id;
    if (absl::SimpleAtoi(id_str, &id)) {
      token_ids.push_back(id);
    } else {
      ABSL_LOG(FATAL) << "Invalid token ID: " << id_str;
    }
  }

  // Create a LiteRT environment.
  auto env = litert::Environment::Create({});
  if (!env) {
    ABSL_LOG(FATAL) << "Failed to create LiteRT environment: " << env.Error();
  }

  // Compile the model.
  LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  auto compiled_model = litert::CompiledModel::Create(
      *env, absl::GetFlag(FLAGS_model_file), options);
  if (!compiled_model) {
    ABSL_LOG(FATAL) << "Failed to compile model: " << compiled_model.Error();
  }

  // Create input and output buffers from the model signature.
  auto input_buffers_expected = compiled_model->CreateInputBuffers();
  if (!input_buffers_expected) {
    ABSL_LOG(FATAL) << "Failed to create input buffers: "
                << input_buffers_expected.Error();
  }
  auto input_buffers = std::move(*input_buffers_expected);

  auto output_buffers_expected = compiled_model->CreateOutputBuffers();
  if (!output_buffers_expected) {
    ABSL_LOG(FATAL) << "Failed to create output buffers: "
                << output_buffers_expected.Error();
  }
  auto output_buffers = std::move(*output_buffers_expected);

  // Write token IDs to the input buffer.
  if (input_buffers.size() != 1) {
    ABSL_LOG(FATAL) << "Expected 1 input tensor, but got " << input_buffers.size();
  }
  auto write_status = input_buffers[0].Write<int>(token_ids);
  if (!write_status) {
    ABSL_LOG(FATAL) << "Failed to write to input tensor: " << write_status.Error();
  }

  // Run inference.
  auto status = compiled_model->Run(input_buffers, output_buffers);
  if (!status) {
    ABSL_LOG(FATAL) << "Failed to run inference: " << status.Error();
  }

  // Get the output tensor data.
  if (output_buffers.size() != 1) {
    ABSL_LOG(FATAL) << "Expected 1 output tensor, but got "
                << output_buffers.size();
  }
  auto& output_tensor = output_buffers[0];
  auto output_size_bytes_expected = output_tensor.PackedSize();
  if (!output_size_bytes_expected) {
    ABSL_LOG(FATAL) << "Failed to get output tensor size: "
                << output_size_bytes_expected.Error();
  }
  size_t output_size_bytes = *output_size_bytes_expected;
  std::vector<float> output_data(output_size_bytes / sizeof(float));
  auto read_status = output_tensor.Read(absl::MakeSpan(output_data));
  if (!read_status) {
    ABSL_LOG(FATAL) << "Failed to read output tensor: " << read_status.Error();
  }

  // Print the embedding vector.
  std::cout << absl::StrJoin(output_data, " ") << std::endl;

  return 0;
}
