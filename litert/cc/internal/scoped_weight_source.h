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

#ifndef ODML_LITERT_LITERT_CC_INTERNAL_SCOPED_WEIGHT_SOURCE_H_
#define ODML_LITERT_LITERT_CC_INTERNAL_SCOPED_WEIGHT_SOURCE_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "litert/cc/internal/scoped_file.h"

namespace litert {

// Describes a contiguous region inside a ScopedFile that backs a single
// external buffer group.
struct ScopedWeightSection {
  uint64_t offset = 0;
  uint64_t length = 0;
};

// Holds the ScopedFile handle plus all group sections that can be sliced out of
// it to satisfy external weight loads.
struct ScopedWeightSource {
  ScopedWeightSource() = default;
  ScopedWeightSource(
      ScopedFile scoped_file,
      absl::flat_hash_map<std::string, ScopedWeightSection> sections)
      : file(std::move(scoped_file)), sections(std::move(sections)) {}

  ScopedWeightSource(ScopedWeightSource&&) = default;
  ScopedWeightSource& operator=(ScopedWeightSource&&) = default;
  ScopedWeightSource(const ScopedWeightSource&) = delete;
  ScopedWeightSource& operator=(const ScopedWeightSource&) = delete;

  bool empty() const { return sections.empty(); }

  ScopedFile file;
  absl::flat_hash_map<std::string, ScopedWeightSection> sections;
};

}  // namespace litert

#endif  // ODML_LITERT_LITERT_CC_INTERNAL_SCOPED_WEIGHT_SOURCE_H_
