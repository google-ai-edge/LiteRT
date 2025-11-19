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

#include "litert/runtime/tensor_buffer_requirements.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/util/to_string.h"

std::string LiteRtTensorBufferRequirementsT::ToString() const {
  std::ostringstream os;
  os << "LiteRtTensorBufferRequirementsT["
     << "supported_types: "
     << litert::internal::ToString(supported_buffer_types_)
     << ", buffer_size: " << buffer_size_
     << ", strides: " << litert::internal::ToString(strides_)
     << ", alignment: " << alignment_ << "]";
  return os.str();
}

namespace litert::internal {

litert::Expected<std::unique_ptr<LiteRtTensorBufferRequirementsT>> Join(
    const LiteRtTensorBufferRequirementsT& src1,
    const LiteRtTensorBufferRequirementsT& src2) {
  LITERT_LOG(LITERT_INFO, "Join src1=%s src2=%s", src1.ToString().c_str(),
             src2.ToString().c_str());

  // Find buffer types common to both requirements.
  std::vector<LiteRtTensorBufferType> buffer_types;
  for (auto bt1 : src1.SupportedBufferTypes()) {
    for (auto bt2 : src2.SupportedBufferTypes()) {
      if (bt2 == bt1) {
        buffer_types.push_back(bt1);
        break;
      }
    }
  }

  if (buffer_types.empty()) {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Can't join requirements due to incompatible supported "
                      "tensor buffer types");
  }

  // Take the max as buffer size.
  auto buffer_size = std::max(src1.BufferSize(), src2.BufferSize());

  std::vector<uint32_t> strides;
  if (src1.Strides() == src2.Strides()) {
    strides = src1.Strides();
  } else {
    return Unexpected(kLiteRtStatusErrorInvalidArgument,
                      "Can't join requirements due to incompatible strides");
  }
  // Take the max alignment requirement.
  auto alignment = std::max(src1.Alignment(), src2.Alignment());

  return std::make_unique<LiteRtTensorBufferRequirementsT>(
      buffer_types.size(), buffer_types.data(), buffer_size, std::move(strides),
      alignment);
}

}  // namespace litert::internal
