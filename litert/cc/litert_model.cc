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

#include "litert/cc/litert_model.h"

#include <cstddef>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/cc/internal/litert_detail.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace litert {

Expected<SimpleTensor> SimpleSignature::InputTensor(
    absl::string_view name) const {
  LiteRtTensor tensor;
  auto status =
      LiteRtGetSignatureInputTensor(Get(), std::string(name).c_str(), &tensor);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to look up signature input tensor");
  }
  return SimpleTensor(tensor);
}

Expected<SimpleTensor> SimpleSignature::InputTensor(size_t index) const {
  LiteRtTensor tensor;
  auto status = LiteRtGetSignatureInputTensorByIndex(
      Get(), static_cast<LiteRtParamIndex>(index), &tensor);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to look up signature input tensor");
  }
  return SimpleTensor(tensor);
}

Expected<SimpleTensor> SimpleSignature::OutputTensor(
    absl::string_view name) const {
  LiteRtTensor tensor;
  auto status =
      LiteRtGetSignatureOutputTensor(Get(), std::string(name).c_str(), &tensor);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to look up signature output tensor");
  }
  return SimpleTensor(tensor);
}

Expected<SimpleTensor> SimpleSignature::OutputTensor(size_t index) const {
  LiteRtTensor tensor;
  auto status = LiteRtGetSignatureOutputTensorByIndex(
      Get(), static_cast<LiteRtParamIndex>(index), &tensor);
  if (status != kLiteRtStatusOk) {
    return Unexpected(status, "Failed to look up signature output tensor");
  }
  return SimpleTensor(tensor);
}

}  // namespace litert
