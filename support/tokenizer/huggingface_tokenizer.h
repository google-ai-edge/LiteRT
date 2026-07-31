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

#ifndef THIRD_PARTY_ODML_LITERT_SUPPORT_TOKENIZER_HUGGING_FACE_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_SUPPORT_TOKENIZER_HUGGING_FACE_TOKENIZER_H_

#if defined(ENABLE_HUGGINGFACE_TOKENIZER)

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizer.h"
#include "include/tokenizers_cpp.h"  // from @tokenizers_cpp

namespace litert::support {

// A Tokenizer implementation using HuggingFace.
class HuggingFaceTokenizer : public Tokenizer {
 public:
  // Creates a HuggingFaceTokenizer from the JSON file
  static absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>> CreateFromFile(
      absl::string_view json_path);

  // Creates a HuggingFaceTokenizer from a JSON string.
  static absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>> CreateFromJson(
      const std::string& json);

  TokenizerType GetTokenizerType() const override {
    return TokenizerType::kHuggingFace;
  }

  // Encodes the given text into a sequence of token ids.
  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override;

  absl::StatusOr<int> TokenToId(absl::string_view token) override;

  // Decodes the given sequence of token ids into a string.
  // Returns absl::DataLossError if any of the tokens are part of an incomplete
  // BPE sequence.
  absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) override;

  std::vector<std::string> GetTokens() const override;

  int GetVocabSize() const override;

 private:
  // Constructor.
  explicit HuggingFaceTokenizer(
      std::unique_ptr<tokenizers::Tokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {};

  // HuggingFace processor.
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

}  // namespace litert::support

#endif  // defined(ENABLE_HUGGINGFACE_TOKENIZER)

#endif  // THIRD_PARTY_ODML_LITERT_SUPPORT_TOKENIZER_HUGGING_FACE_TOKENIZER_H_
