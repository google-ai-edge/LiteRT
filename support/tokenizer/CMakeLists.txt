# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ==============================================================================
# SentencePiece Tokenizers
# ==============================================================================
add_library(support_tokenizer_sentencepiece_tokenizers STATIC
  sentencepiece_tokenizers.cc
)
add_library(LiteRT::Support::Tokenizer::SentencePieceTokenizers 
  ALIAS support_tokenizer_sentencepiece_tokenizers)

target_include_directories(support_tokenizer_sentencepiece_tokenizers
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  PUBLIC
    ${LITERT_TOKENIZERS_SRC_DIR}
)

target_link_libraries(support_tokenizer_sentencepiece_tokenizers
  PUBLIC
    absl::memory
    absl::status
    absl::strings
    tokenizers_lib
    litert_cc_api
)

# ==============================================================================
# HuggingFace Tokenizers
# ==============================================================================
add_library(support_tokenizer_huggingface_tokenizers STATIC
  huggingface_tokenizers.cc
)
add_library(LiteRT::Support::Tokenizer::HuggingFaceTokenizers
  ALIAS support_tokenizer_huggingface_tokenizers)

# ==============================================================================
# Buffered Streaming Detokenizer
# ==============================================================================
target_include_directories(support_tokenizer_huggingface_tokenizers
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  PUBLIC
    ${LITERT_TOKENIZERS_SRC_DIR}
)

target_link_libraries(support_tokenizer_huggingface_tokenizers
  PUBLIC
    absl::memory
    absl::status
    absl::strings
    tokenizers_lib
    litert_cc_api
)


add_library(support_tokenizer_buffered_streaming_detokenizer STATIC
  buffered_streaming_detokenizer.cc
)
add_library(LiteRT::Support::Tokenizer::BufferedStreamingDetokenizer
  ALIAS support_tokenizer_buffered_streaming_detokenizer)

target_include_directories(support_tokenizer_buffered_streaming_detokenizer
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  PUBLIC
    ${LITERT_TOKENIZERS_SRC_DIR}
)

target_link_libraries(support_tokenizer_buffered_streaming_detokenizer
  PUBLIC
    absl::memory
    absl::status
    absl::strings
    LiteRT::Support::Util::MemoryMappedFile
    tokenizers_lib
    litert_cc_api
)