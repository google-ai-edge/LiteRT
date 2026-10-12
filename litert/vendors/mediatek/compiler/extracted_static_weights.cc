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

#include "litert/vendors/mediatek/compiler/extracted_static_weights.h"

#include <stdlib.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::mediatek {
namespace {

constexpr uint32_t kExtractedWeightsMagic = 0xbbced1ec;
constexpr size_t kFooterSize = sizeof(uint64_t) + sizeof(uint32_t);

struct ExtractedWeightEntry {
  uint64_t order = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

// Parses the first quoted decimal integer at or after `*pos` in `text` into
// `*value` and advances `*pos` past its closing quote.
bool ConsumeQuotedUint64(absl::string_view text, size_t* pos, uint64_t* value) {
  const size_t open_quote = text.find('"', *pos);
  if (open_quote == absl::string_view::npos) {
    return false;
  }
  const size_t close_quote = text.find('"', open_quote + 1);
  if (close_quote == absl::string_view::npos) {
    return false;
  }
  const absl::string_view digits =
      text.substr(open_quote + 1, close_quote - open_quote - 1);
  // `SimpleAtoi` also accepts signs and whitespace, so check for digits first.
  // It fails on overflow.
  if (digits.empty() || !absl::c_all_of(digits, absl::ascii_isdigit) ||
      !absl::SimpleAtoi(digits, value)) {
    return false;
  }
  *pos = close_quote + 1;
  return true;
}

// Parses the JSON object that describes the extracted weights. Every entry
// must lie within the first `payload_size` bytes of the file.
Expected<std::vector<ExtractedWeightEntry>> ParseEntries(
    absl::string_view json, uint64_t payload_size) {
  std::vector<ExtractedWeightEntry> entries;
  size_t pos = 0;
  while (json.find('"', pos) != absl::string_view::npos) {
    ExtractedWeightEntry entry;
    if (!ConsumeQuotedUint64(json, &pos, &entry.order)) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Failed to parse weight order in extracted weights JSON");
    }
    const size_t open_bracket = json.find('[', pos);
    if (open_bracket == absl::string_view::npos) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Missing opening bracket in extracted weights JSON");
    }
    pos = open_bracket + 1;
    if (!ConsumeQuotedUint64(json, &pos, &entry.offset) ||
        !ConsumeQuotedUint64(json, &pos, &entry.size)) {
      return Error(
          kLiteRtStatusErrorInvalidArgument,
          "Failed to parse weight offset/size in extracted weights JSON");
    }
    const size_t close_bracket = json.find(']', pos);
    if (close_bracket == absl::string_view::npos) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Missing closing bracket in extracted weights JSON");
    }
    pos = close_bracket + 1;
    if (entry.offset > payload_size ||
        entry.size > payload_size - entry.offset) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Extracted weight range out of bounds");
    }
    entries.push_back(entry);
  }

  // The orders must be consecutive, but they need not start at zero.
  absl::c_sort(entries,
               [](const ExtractedWeightEntry& a,
                  const ExtractedWeightEntry& b) { return a.order < b.order; });
  for (size_t i = 1; i < entries.size(); ++i) {
    if (entries[i].order != entries[0].order + i) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   absl::StrFormat("Non-consecutive weight order in extracted "
                                   "weights JSON: expected %d, got %d",
                                   entries[0].order + i, entries[i].order));
    }
  }
  return entries;
}

}  // namespace

Expected<std::string> StaticWeightExtractionOptions(absl::string_view path) {
  if (path.empty() || absl::c_any_of(path, absl::ascii_isspace)) {
    return Error(
        kLiteRtStatusErrorInvalidArgument,
        absl::StrFormat("Unsupported path for extracted weights: '%s'", path));
  }
  return absl::StrCat(" --opt-static-sharing --extract-static-data=", path);
}

Expected<std::string> CreateExtractedStaticWeightsFile(absl::string_view name) {
  namespace fs = std::filesystem;
  std::error_code error;
  const char* dla_dir = std::getenv("MTKNN_ADAPTER_DLA_DIR");
  const fs::path dir = (dla_dir != nullptr && dla_dir[0] != '\0')
                           ? fs::path(dla_dir)
                           : fs::temp_directory_path(error);
  if (error) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrFormat("Failed to get the temporary directory: %s",
                                 error.message()));
  }
  // Any failure to create `dir` makes `mkstemp` fail below.
  fs::create_directories(dir, error);
  std::string path =
      (dir / absl::StrFormat("mtk_%s_weights_XXXXXX", name)).string();
  const int fd = mkstemp(path.data());
  if (fd < 0) {
    return Error(
        kLiteRtStatusErrorFileIO,
        absl::StrFormat("Failed to create %s: %s", path, std::strerror(errno)));
  }
  close(fd);
  return path;
}

Expected<std::vector<absl::string_view>> ParseExtractedStaticWeights(
    absl::string_view contents) {
  std::vector<absl::string_view> weights;
  if (contents.empty()) {
    return weights;
  }
  if (contents.size() < kFooterSize) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Extracted weights file is too small");
  }
  const uint64_t footer_offset = contents.size() - kFooterSize;
  uint64_t json_offset = 0;
  uint32_t magic = 0;
  std::memcpy(&json_offset, contents.data() + footer_offset,
              sizeof(json_offset));
  std::memcpy(&magic, contents.data() + footer_offset + sizeof(json_offset),
              sizeof(magic));
  if (magic != kExtractedWeightsMagic || json_offset > footer_offset) {
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Invalid extracted weights footer");
  }

  auto entries =
      ParseEntries(contents.substr(json_offset, footer_offset - json_offset),
                   /*payload_size=*/json_offset);
  if (!entries) {
    return entries.Error();
  }
  weights.reserve(entries->size());
  for (const ExtractedWeightEntry& entry : *entries) {
    weights.push_back(contents.substr(entry.offset, entry.size));
  }
  return weights;
}

}  // namespace litert::mediatek
