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

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "third_party/odml/litert/tensor/examples/gemma3/tokenizer.h"

using emscripten::class_;
using litert::tensor::examples::GemmaTokenizerSP;

namespace {

struct TokenizerWrapper {
  std::shared_ptr<GemmaTokenizerSP> tokenizer;

  bool isNull() const { return tokenizer == nullptr; }

  std::vector<int32_t> Encode(const std::string& text, bool add_bos) const {
    if (!tokenizer) return {};
    return tokenizer->Encode(text, add_bos);
  }

  std::string Decode(const std::vector<int32_t>& tokens) const {
    if (!tokenizer) return "";
    return tokenizer->Decode(tokens);
  }

  std::string DecodeToken(int32_t token) const {
    if (!tokenizer) return "";
    return tokenizer->DecodeToken(token);
  }

  size_t VocabSize() const {
    if (!tokenizer) return 0;
    return tokenizer->VocabSize();
  }
};

TokenizerWrapper LoadTokenizer(const std::string& model_path) {
  auto res = GemmaTokenizerSP::Load(model_path);
  if (!res.ok()) {
    return TokenizerWrapper{nullptr};
  }
  return TokenizerWrapper{std::make_shared<GemmaTokenizerSP>(std::move(*res))};
}

}  // namespace

EMSCRIPTEN_BINDINGS(gemma3_tokenizer) {
  emscripten::register_vector<int32_t>("VectorInt32");

  class_<TokenizerWrapper>("GemmaTokenizer")
      .constructor<>()
      .function("isNull", &TokenizerWrapper::isNull)
      .function("Encode", &TokenizerWrapper::Encode)
      .function("Decode", &TokenizerWrapper::Decode)
      .function("DecodeToken", &TokenizerWrapper::DecodeToken)
      .function("VocabSize", &TokenizerWrapper::VocabSize);

  emscripten::function("loadTokenizer", &LoadTokenizer);
}

#endif  // __EMSCRIPTEN__
