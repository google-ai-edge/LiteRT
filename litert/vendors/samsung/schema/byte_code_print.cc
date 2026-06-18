// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/vendors/samsung/schema/litert_samsung_header_generated.h"

ABSL_FLAG(std::string, model_path, {},
          "Please provide the bytecode path. If you don't have one,"
          "extract_bytecode tool may help you get from tflite model.");

namespace litert::samsung {

static constexpr absl::string_view kStandardFormat = R"(
    __    _ __       ____  __
   / /   (_/ /____  / __ \/ /_
  / /   / / __/ _ \/ /_/ / __/
 / /___/ / /_/  __/ _, _/ /_
/_____/_/\__/\___/_/ |_|\__/

File Size  :  %d
 ___________________________________
 |  HEADER                          |
 |----------------------------------|
 |    Version:  %2d                 |
 |----------------------------------|
 |    Dispatch: [0]                 |
 |      Offset:         : %8d  |
 |      Size            : %8d  |
 |      External Weights: %8s  |
 |      Used Weights    :           |%s
 |----------------------------------|%s
 |  BINARIES   ......               |
 |__________________________________|
)";

static constexpr absl::string_view kAuxFormat = R"(
 |         Signature [%d]:           |
 |           %16s        |)";

static constexpr absl::string_view kWeightFormat = R"(
 |    Weight:  [%d]                  |
 |      Offset   :  %8d        |
 |      Size     :  %8d        |
 |      Signature:  %16s |
 |----------------------------------|)";

Expected<void> PrintByteCode(const std::string& model_path) {
  std::ifstream infile(model_path, std::ios::binary | std::ios::ate);
  if (!infile) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      absl::StrCat("Cannot open source file: ", model_path));
  }

  std::streamsize file_size = infile.tellg();
  infile.seekg(0, std::ios::beg);
  std::vector<char> buffer(file_size);

  if (!infile.read(buffer.data(), file_size)) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrCat("Failed to read from source file: ", model_path));
  }
  infile.close();

  auto* header_buf = schema::GetLiteRTSamsungHeader(buffer.data());

  bool is_valid_header = false;
  if (header_buf != nullptr) {
    // Verify the structure (without mandatory file identifier)
    flatbuffers::Verifier verifier(
        reinterpret_cast<const uint8_t*>(buffer.data()), file_size);

    is_valid_header = header_buf->Verify(verifier);

    // Also check if file identifier exists
    bool has_identifier =
        schema::LiteRTSamsungHeaderBufferHasIdentifier(buffer.data());

    LITERT_LOG(LITERT_INFO,
               "Buffer: %p, Size: %d, Has valid vtable: %s, Has identifier: %s",
               buffer.data(), file_size, is_valid_header ? "true" : "false",
               has_identifier ? "true" : "false");
  }

  if (!is_valid_header) {
    return Unexpected(
        kLiteRtStatusErrorInvalidArgument,
        "Invalid LiteRT Samsung Header - failed to parse or verify");
  }
  auto dispatch_binary = header_buf->dispatch_binary();

  std::string w_signatures;
  int32_t external_weights_size =
      dispatch_binary->external_weights()
          ? dispatch_binary->external_weights()->size()
          : 0;
  for (int32_t index = 0; index < external_weights_size; index++) {
    w_signatures += absl::StrFormat(
        kAuxFormat, index,
        dispatch_binary->external_weights()->Get(index)->c_str());
  }

  std::string weights_memo;
  if (header_buf->separated_weights()) {
    for (int32_t index = 0; index < header_buf->separated_weights()->size();
         index++) {
      auto weight = header_buf->separated_weights()->Get(index);
      weights_memo += absl::StrFormat(
          kWeightFormat, index, weight->buf()->start_offset(),
          weight->buf()->end_offset() - weight->buf()->start_offset(),
          weight->signature()->c_str());
    }
  }

  auto show =
      absl::StrFormat(kStandardFormat, file_size, header_buf->version(),
                      dispatch_binary->buf()->start_offset(),
                      dispatch_binary->buf()->end_offset() -
                          dispatch_binary->buf()->start_offset(),
                      dispatch_binary->use_external_weights() ? "Yes" : "No",
                      w_signatures.c_str(), weights_memo.c_str());

  std::cerr << show << std::endl;

  return {};
}

}  // namespace litert::samsung

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  auto result = litert::samsung::PrintByteCode(model_path);

  return static_cast<int>(!result.HasValue());
}
