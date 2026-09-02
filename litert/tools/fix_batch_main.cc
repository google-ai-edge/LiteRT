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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/flags/usage.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_model.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/tools/batch_utils.h"

ABSL_FLAG(std::string, input_tflite, "", "Path to the input TFLite model.");
ABSL_FLAG(std::string, output_tflite, "", "Path to the output TFLite model.");
ABSL_FLAG(int32_t, batch_size, -1, "The fixed batch dimension to apply.");

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Fix the batch dimension of a TFLite model.");
  absl::ParseCommandLine(argc, argv);

  const std::string input_path = absl::GetFlag(FLAGS_input_tflite);
  const std::string output_path = absl::GetFlag(FLAGS_output_tflite);
  const int32_t batch_size = absl::GetFlag(FLAGS_batch_size);

  if (input_path.empty() || output_path.empty() || batch_size <= 0) {
    ABSL_LOG(ERROR)
        << "Invalid arguments. Usage: fix_batch_main --input_tflite=... "
           "--output_tflite=... --batch_size=...";
    return 1;
  }

  auto model_expected = litert::Model::CreateFromFile(input_path);
  if (!model_expected) {
    ABSL_LOG(ERROR) << "Failed to load model: "
                    << model_expected.Error().Message();
    return 1;
  }

  LiteRtModelT* model_raw = model_expected->Get();

  if (litert::tools::ValidateModelForBatchFix(*model_raw) != kLiteRtStatusOk) {
    ABSL_LOG(ERROR) << "Model validation failed. Rejecting model.";
    return 1;
  }

  litert::tools::FixBatchDimension(*model_raw, batch_size);

  // Take ownership for serialization.
  auto serialized_expected =
      litert::internal::SerializeModel(std::move(*model_raw));
  if (!serialized_expected) {
    ABSL_LOG(ERROR) << "Failed to serialize model: "
                    << serialized_expected.Error().Message();
    return 1;
  }

  std::ofstream output_file(output_path, std::ios::binary);
  if (!output_file.is_open()) {
    ABSL_LOG(ERROR) << "Failed to open output file: " << output_path;
    return 1;
  }
  output_file.write(reinterpret_cast<const char*>(serialized_expected->Data()),
                    serialized_expected->Size());
  output_file.close();

  std::cout << "Successfully updated batch dimension to " << batch_size
            << " and saved to " << output_path << std::endl;

  return 0;
}
