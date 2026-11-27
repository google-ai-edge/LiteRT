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

#include "third_party/odml/litert/tensor/examples/gemma3/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::tensor::examples {

absl::StatusOr<GemmaTokenizerSP> GemmaTokenizerSP::Load(
    const std::string& model_path) {
  GemmaTokenizerSP tokenizer;
  tokenizer.processor_ =
      std::make_unique<sentencepiece::SentencePieceProcessor>();

  const auto status = tokenizer.processor_->Load(model_path);
  if (!status.ok()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to load SentencePiece model from '", model_path,
                     "': ", status.ToString()));
  }

  ABSL_LOG(INFO) << "Loaded SentencePiece model with "
                 << tokenizer.processor_->GetPieceSize() << " tokens";

  // Verify special tokens match expected IDs
  int bos_id = tokenizer.processor_->bos_id();
  int eos_id = tokenizer.processor_->eos_id();
  int pad_id = tokenizer.processor_->pad_id();
  int unk_id = tokenizer.processor_->unk_id();

  ABSL_LOG(INFO) << "Special tokens - BOS: " << bos_id << ", EOS: " << eos_id
                 << ", PAD: " << pad_id << ", UNK: " << unk_id;

  // Warn if special tokens don't match expected values
  if (bos_id != kBosToken) {
    ABSL_LOG(WARNING) << "BOS token ID mismatch: expected " << kBosToken
                      << ", got " << bos_id;
  }
  if (eos_id != kEosToken) {
    ABSL_LOG(WARNING) << "EOS token ID mismatch: expected " << kEosToken
                      << ", got " << eos_id;
  }

  return tokenizer;
}

std::vector<int32_t> GemmaTokenizerSP::Encode(const std::string& text,
                                              bool add_bos) const {
  std::vector<int> ids;
  const auto status = processor_->Encode(text, &ids);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to encode text: " << status.ToString();
    return {};
  }

  // Convert to int32_t and optionally add BOS
  std::vector<int32_t> result;
  result.reserve(ids.size() + (add_bos ? 1 : 0));

  if (add_bos) {
    result.push_back(kBosToken);
  }

  for (int id : ids) {
    result.push_back(static_cast<int32_t>(id));
  }

  return result;
}

std::string GemmaTokenizerSP::Decode(const std::vector<int32_t>& tokens) const {
  // Convert to vector<int> for SentencePiece
  std::vector<int> ids(tokens.begin(), tokens.end());

  std::string text;
  const auto status = processor_->Decode(ids, &text);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to decode tokens: " << status.ToString();
    return "";
  }

  return text;
}

std::string GemmaTokenizerSP::DecodeToken(int32_t token) const {
  // Use IdToPiece to get the raw token representation
  const std::string& piece = processor_->IdToPiece(token);

  std::vector<int> ids = {token};
  std::string text;
  const auto status = processor_->Decode(ids, &text);
  if (!status.ok()) {
    // Fallback to IdToPiece if Decode fails
    return piece;
  }

  return text;
}

size_t GemmaTokenizerSP::VocabSize() const {
  return static_cast<size_t>(processor_->GetPieceSize());
}

}  // namespace litert::tensor::examples
