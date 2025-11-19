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

#include <iostream>
#include <string>
#include <vector>

#include "base/init_google.h"
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "sentencepiece_processor.h"  // from @sentencepiece

// Define command-line flags for the model file and the sentence.
ABSL_FLAG(std::string, model_file, "",
          "The path to the SentencePiece model file.");
ABSL_FLAG(std::string, sentence, "", "The sentence to tokenize.");

int main(int argc, char* argv[]) {
  InitGoogle(argv[0], &argc, &argv, true);
  absl::ParseCommandLine(argc, argv);

  // Validate that the required flags are provided.
  if (absl::GetFlag(FLAGS_model_file).empty()) {
    ABSL_LOG(FATAL) << "--model_file flag must be provided.";
  }
  if (absl::GetFlag(FLAGS_sentence).empty()) {
    ABSL_LOG(FATAL) << "--sentence flag must be provided.";
  }

  // Initialize the SentencePiece processor.
  sentencepiece::SentencePieceProcessor processor;
  const absl::Status status = processor.Load(absl::GetFlag(FLAGS_model_file));

  // Check if the model loaded successfully.
  if (!status.ok()) {
    ABSL_LOG(FATAL) << "Failed to load SentencePiece model: " << status;
  }

  // Vector to store the output token IDs.
  std::vector<int> ids;

  // Encode the sentence into token IDs.
  const absl::Status encode_status =
      processor.Encode(absl::GetFlag(FLAGS_sentence), &ids);

  if (!encode_status.ok()) {
    ABSL_LOG(FATAL) << "Failed to encode sentence: " << encode_status;
  }

  // Print the token IDs to standard output, separated by spaces.
  std::cout << absl::StrJoin(ids, " ") << std::endl;

  return 0;
}