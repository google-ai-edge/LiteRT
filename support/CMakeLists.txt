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

include(FetchContent)

set(LITERT_STBLIB_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/stblib_src" CACHE PATH "Path to stblib source directory")
FetchContent_Declare(
  stblib_lib
  GIT_REPOSITORY https://github.com/nothings/stb
  GIT_TAG        master
  SOURCE_DIR ${LITERT_STBLIB_SRC_DIR}
)
FetchContent_MakeAvailable(stblib_lib)

set(LITERT_KISSFFT_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/kissfft_src" CACHE PATH "Path to kissfft source directory")
FetchContent_Declare(
  kissfft_lib
  GIT_REPOSITORY https://github.com/mborgerding/kissfft
  GIT_TAG        master
  SOURCE_DIR ${LITERT_KISSFFT_SRC_DIR}

)
FetchContent_MakeAvailable(kissfft_lib)

set(LITERT_MINIAUDIO_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/miniaudio_src" CACHE PATH "Path to miniaudio source directory")
FetchContent_Declare(
  miniaudio_lib
  GIT_REPOSITORY https://github.com/mackron/miniaudio
  GIT_TAG        master
  SOURCE_DIR ${LITERT_MINIAUDIO_SRC_DIR}
)
FetchContent_MakeAvailable(miniaudio_lib)

set(LITERT_TOKENIZERS_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/tokenizers_src" CACHE PATH "Path to tokenizers source directory")
FetchContent_Declare(
  tokenizers_lib
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp
  GIT_TAG        main
  SOURCE_DIR ${LITERT_TOKENIZERS_SRC_DIR}
)
FetchContent_MakeAvailable(tokenizers_lib)




add_subdirectory(preprocessor)
add_subdirectory(tokenizer)
add_subdirectory(util)