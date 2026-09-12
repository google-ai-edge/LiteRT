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

#include "litert/vendors/common/vendor_traits.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert::vendors {

Expected<LiteRtTensorBufferRequirements>
BufferRequirements::ToLiteRtRequirements() const {
  LiteRtTensorBufferRequirements requirements = nullptr;

  // Create requirements
  LITERT_RETURN_IF_ERROR(LiteRtCreateTensorBufferRequirements(
      supported_types.size(), supported_types.data(), buffer_size,
      strides.size(), strides.empty() ? nullptr : strides.data(),
      &requirements));

  return requirements;
}

}  // namespace litert::vendors
