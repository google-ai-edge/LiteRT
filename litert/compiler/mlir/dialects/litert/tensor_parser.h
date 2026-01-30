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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TENSOR_PARSER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TENSOR_PARSER_H_

#include <cstdint>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

namespace litert {

template <typename T>
class TensorParser {
 public:
  TensorParser(mlir::AsmParser& p, mlir::ShapedType type)
      : parser_(p), type_(type) {}

  llvm::ParseResult ParseElement() {
    if constexpr (std::is_same_v<T, int32_t>) {
      T result;
      if (auto pr = parser_.parseInteger(result); pr.failed()) {
        return pr;
      }

      data_.emplace_back(result);
    } else if constexpr (std::is_same_v<T, float>) {
      auto element_type =
          mlir::dyn_cast<mlir::FloatType>(type_.getElementType());
      mlir::APFloat result = mlir::APFloat::Bogus();
      if (auto pr =
              parser_.parseFloat(element_type.getFloatSemantics(), result);
          pr.failed()) {
        return pr;
      }
      data_.emplace_back(result.convertToFloat());
    }

    return mlir::success();
  }

  llvm::ParseResult ParseList() {
    auto parse_one_element = [&]() -> llvm::ParseResult {
      return ParseElement();
    };

    if (parser_.parseCommaSeparatedList(parse_one_element)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  llvm::ParseResult Parse() {
    // Base case, no more nested brackets
    if (parser_.parseOptionalLSquare().failed()) {
      return ParseList();
    }

    // Parse contents and closing bracket
    if (Parse() || parser_.parseRSquare()) {
      return mlir::failure();
    }

    // Nothing else to do, return success
    if (parser_.parseOptionalComma().failed()) {
      return mlir::success();
    }

    return Parse();
  }

  std::vector<T>& GetData() { return data_; }

 private:
  mlir::AsmParser& parser_;
  mlir::ShapedType type_;
  llvm::SmallVector<int64_t, 4> shape_;
  std::vector<T> data_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TENSOR_PARSER_H_
