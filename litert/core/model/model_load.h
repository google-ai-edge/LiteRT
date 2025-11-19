// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_CORE_MODEL_MODEL_LOAD_H_
#define ODML_LITERT_LITERT_CORE_MODEL_MODEL_LOAD_H_

#include <cstdint>
#include <memory>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"

namespace litert::internal {

// Loads a model from a file. If allow_modifications is true, then the model
// can be modified in place, for example, model would be mmap'ed with writable
// flag and private mapping (not to update the model file on disk).
Expected<std::unique_ptr<LiteRtModelT>> LoadModelFromFile(
    absl::string_view filename, bool allow_modifications = false);

Expected<std::unique_ptr<LiteRtModelT>> LoadModelFromBuffer(
    BufferRef<uint8_t> buffer);

Expected<std::unique_ptr<LiteRtModelT>> LoadModelFromBuffer(
    OwningBufferRef<uint8_t>&& buffer);

}  // namespace litert::internal

#endif  // ODML_LITERT_LITERT_CORE_MODEL_MODEL_LOAD_H_
