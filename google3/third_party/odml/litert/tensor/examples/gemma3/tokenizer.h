/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_TOKENIZER_H_
#define THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_TOKENIZER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::tensor::examples {

// SentencePiece-based tokenizer for Gemma models.
// Loads vocabulary from SentencePiece .model file (tokenizer.model).
class GemmaTokenizerSP {
 public:
  // Special token IDs for Gemma models.
  // These are standard SentencePiece special tokens.
  static constexpr int32_t kPadToken = 0;
  static constexpr int32_t kEosToken = 1;
  static constexpr int32_t kBosToken = 2;
  static constexpr int32_t kUnkToken = 3;

  // Load tokenizer from SentencePiece model file (.model).
  static absl::StatusOr<GemmaTokenizerSP> Load(const std::string& model_path);

  // Encode text to token IDs.
  // If add_bos is true, prepends the BOS token.
  std::vector<int32_t> Encode(const std::string& text,
                              bool add_bos = true) const;

  // Decode token IDs to text.
  std::string Decode(const std::vector<int32_t>& tokens) const;

  // Decode single token to text.
  std::string DecodeToken(int32_t token) const;

  // Get vocabulary size.
  size_t VocabSize() const;

  // Get the underlying SentencePiece processor (for advanced usage).
  const sentencepiece::SentencePieceProcessor& Processor() const {
    return *processor_;
  }

 private:
  GemmaTokenizerSP() = default;

  std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
};

}  // namespace litert::tensor::examples

#endif  // THIRD_PARTY_ODML_LITERT_TENSOR_EXAMPLES_GEMMA3_TOKENIZER_H_
