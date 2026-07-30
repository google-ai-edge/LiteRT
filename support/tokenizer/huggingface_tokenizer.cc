// Copyright 2025 The ODML Authors.
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

#include "support/tokenizer/huggingface_tokenizer.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/debugging/leak_check.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/util/memory_mapped_file.h"
#include "support/util/status_macros.h"  // NOLINT
#include "include/tokenizers_cpp.h"  // from @tokenizers_cpp

namespace litert::support {

absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>>
HuggingFaceTokenizer::CreateFromFile(absl::string_view json_path) {
  ASSIGN_OR_RETURN(auto memory_mapped_file,  // NOLINT
                   MemoryMappedFile::Create(json_path));
  std::string json_data(memory_mapped_file->length(), '\0');
  memcpy(json_data.data(), memory_mapped_file->data(),
         memory_mapped_file->length());
  return CreateFromJson(json_data);
}

absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>>
HuggingFaceTokenizer::CreateFromJson(const std::string& json) {
  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json);
  if (!tokenizer) {
    return absl::InvalidArgumentError("Failed to create tokenizer from JSON.");
  }
  return absl::WrapUnique(new HuggingFaceTokenizer(std::move(tokenizer)));
}

// Encodes the given text into a TensorBuffer of token ids.
absl::StatusOr<std::vector<int>> HuggingFaceTokenizer::TextToTokenIds(
    absl::string_view text) {
  {
    // Disable leak check as Google's default leak checker does not properly
    // support Rust's lazy_static initialization.
    // TODO(b/379364190) - Remove this once the leak checker is fixed.
    absl::LeakCheckDisabler disabler;
    return tokenizer_->Encode(std::string{text});
  }
}

absl::StatusOr<int> HuggingFaceTokenizer::TokenToId(absl::string_view token) {
  int id = tokenizer_->TokenToId(std::string{token});
  if (id == -1) {
    return absl::NotFoundError(absl::StrCat("Unknown token: ", token));
  }
  return id;
}

// Decodes the given TensorBuffer of token ids into a vector of strings.
absl::StatusOr<std::string> HuggingFaceTokenizer::TokenIdsToText(
    const std::vector<int>& token_ids) {
  {
    absl::LeakCheckDisabler disabler;
    // Disable leak check as Google's default leak checker does not properly
    // support Rust's lazy_static initialization.
    // TODO(b/379364190) - Remove this once the leak checker is fixed.
    return tokenizer_->Decode(token_ids);
  }
}

std::vector<std::string> HuggingFaceTokenizer::GetTokens() const {
  std::vector<std::string> tokens;
  int vocab_size = tokenizer_->GetVocabSize();
  tokens.reserve(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    std::string token = tokenizer_->IdToToken(i);
    tokens.push_back(token);
  }
  return tokens;
}

int HuggingFaceTokenizer::GetVocabSize() const {
  return tokenizer_->GetVocabSize();
}

}  // namespace litert::support
