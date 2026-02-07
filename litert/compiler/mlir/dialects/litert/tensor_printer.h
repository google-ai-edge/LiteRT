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
#ifndef THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TENSOR_PRINTER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TENSOR_PRINTER_H_

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"

namespace litert {

template <typename T>
class TensorPrinter {
 public:
  TensorPrinter(llvm::ArrayRef<T> data, mlir::ShapedType type,
                mlir::AsmPrinter& printer)
      : data_(data),
        shape_(type.getShape()),
        rank_(type.getRank()),
        printer_(printer),
        stream_(printer.getStream()) {}

  void Print() {
    if (data_.size() == 1) {
      printer_ << data_[0];
      return;
    }

    printer_ << '[';
    int index = 0;
    Print(0, index);
    printer_ << ']';
  }

 private:
  void Print(int depth, int& index) {
    auto size = shape_[depth];
    if (size == 0) {
      return;
    }

    // Base case. Print elements and return.
    if (depth == rank_ - 1) {
      printer_ << data_[index++];
      for (auto i = 1; i < size; i++) {
        printer_ << ',' << data_[index++];
      }
      return;
    }

    auto new_depth = depth + 1;
    stream_ << '[';
    Print(new_depth, index);
    for (auto i = 1; i < size; i++) {
      stream_ << "],[";
      Print(new_depth, index);
    }
    stream_ << ']';
  }

  llvm::ArrayRef<T> data_;
  llvm::SmallVector<int64_t, 4> shape_;
  int64_t rank_;
  mlir::AsmPrinter& printer_;
  llvm::raw_ostream& stream_;
  mlir::OpPrintingFlags flags_;
};

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_COMPILER_MLIR_DIALECTS_LITERT_TENSOR_PRINTER_H_
