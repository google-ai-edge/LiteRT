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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_ATTRIBUTES_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_ATTRIBUTES_H_

#include <climits>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "litert/compiler/mlir/dialects/litert/callback_resource.h"
#include "litert/compiler/mlir/dialects/litert/lazy_blob_manager.h"

namespace litert {

class LITERTDialect;

using LazyResourceElementsHandle = LazyResourceBlobHandle<LITERTDialect>;
using CallbackResourceElementsHandle = CallbackResourceHandle<LITERTDialect>;

}  // namespace litert

#define GET_ATTRDEF_CLASSES
#include "litert/compiler/mlir/dialects/litert/attributes.h.inc"  // IWYU pragma: export

namespace litert {

namespace details {

std::string GenerateUUID();

}  // namespace details

template <typename T>
LazyDenseElementsAttr LazyDenseElementsAttr::get(mlir::ShapedType type,
                                                 llvm::ArrayRef<T> data) {
  if (auto element_type = type.getElementType();
      element_type.isIntOrIndexOrFloat()) {
    auto num_elements = data.size();
    ABSL_CHECK(type.getNumElements() == num_elements)
        << "The number of elements in the ShapedType must match the number of "
           "elements in the provided data buffer (expected "
        << type.getNumElements() << " but got " << data.size() << ").";

    auto bit_width = 0;
    auto data_buffer_bit_width = sizeof(T) * CHAR_BIT;
    if (auto int_type =
            mlir::dyn_cast_or_null<mlir::IntegerType>(element_type)) {
      ABSL_CHECK(std::is_integral_v<T>) << "Expected Integer type.";
      bit_width = int_type.getWidth();
    }
    if (auto float_type =
            mlir::dyn_cast_or_null<mlir::FloatType>(element_type)) {
      ABSL_CHECK(std::is_floating_point_v<T>) << "Expected Float type.";
      bit_width = float_type.getWidth();
    }
    ABSL_CHECK(bit_width == data_buffer_bit_width)
        << "Bit width of the element type does not match that of the provided "
           "data buffer type (expected "
        << bit_width << " but got " << data_buffer_bit_width << ").";
  }
  auto blob = LazyResourceBlob::CreateAndCopyData(data);
  auto& manager =
      LazyResourceElementsHandle::getManagerInterface(type.getContext());
  return get(type, manager.Insert(details::GenerateUUID(), std::move(blob)));
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_ATTRIBUTES_H_
