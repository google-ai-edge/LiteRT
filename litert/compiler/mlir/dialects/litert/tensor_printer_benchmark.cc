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
#include <cstdint>
#include <optional>
#include <string>

#include "testing/base/public/benchmark.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "litert/compiler/mlir/dialects/litert/tensor_printer.h"

namespace litert {
namespace {

template <typename T>
class OldTensorPrinter {
 public:
  OldTensorPrinter(llvm::ArrayRef<T> data, mlir::ShapedType type,
                   mlir::AsmPrinter& printer)
      : open_brackets_(0),
        data_(data),
        shape_(type.getShape()),
        rank_(type.getRank()),
        counter_(rank_, 0),
        printer_(printer) {}

  void Print() {
    std::optional<int64_t> limit =
        mlir::OpPrintingFlags().getLargeElementsAttrLimit();

    if (limit.has_value() && data_.size() > *limit) {
      printer_ << "__elided__";
      return;
    }

    if (data_.size() == 1) {
      printer_ << data_[0];
      return;
    }
    for (unsigned i = 0; i < data_.size(); i++) {
      if (i != 0) {
        printer_ << ", ";
      }

      while (open_brackets_++ < rank_) {
        printer_ << '[';
      }

      open_brackets_ = rank_;

      printer_ << data_[i];

      BumpCounter();
    }

    while (open_brackets_-- > 0) {
      printer_ << ']';
    }
  }

 private:
  void BumpCounter() {
    // Bump the least significant digit.
    ++counter_[rank_ - 1];
    // Iterate backwards bubbling back the increment.
    for (unsigned i = rank_ - 1; i > 0; --i) {
      if (counter_[i] >= shape_[i]) {
        // Index 'i' is rolled over. Bump (i-1) and close a bracket.
        counter_[i] = 0;
        ++counter_[i - 1];
        --open_brackets_;
        printer_ << ']';
      }
    }
  }

  unsigned open_brackets_;
  llvm::ArrayRef<T> data_;
  llvm::SmallVector<int64_t, 4> shape_;
  int64_t rank_;
  llvm::SmallVector<unsigned, 4> counter_;
  mlir::AsmPrinter& printer_;
  mlir::OpPrintingFlags flags_;
};

class DummyPrinter : public mlir::AsmPrinter {
 public:
  explicit DummyPrinter(llvm::raw_ostream& ostream) : ostream_(ostream) {}
  llvm::raw_ostream& getStream() const override { return ostream_; }

 private:
  llvm::raw_ostream& ostream_;
};

static constexpr int num_elements = 128 * 128;

void BM_OldPrint(benchmark::State& state) {
  mlir::MLIRContext ctx;
  llvm::SmallVector<int64_t, 2> shape = {128, 128};
  llvm::SmallVector<int64_t, num_elements> data(num_elements);
  for (auto i = 0; i < num_elements; i++) {
    data[i] = i;
  }
  auto rtt =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 32));
  std::string buffer;

  for (auto s : state) {
    llvm::raw_string_ostream ostream(buffer);
    DummyPrinter printer(ostream);
    OldTensorPrinter<int64_t> tensor_printer(data, rtt, printer);
    tensor_printer.Print();
  }
}

void BM_Print(benchmark::State& state) {
  mlir::MLIRContext ctx;
  llvm::SmallVector<int64_t, 2> shape = {128, 128};
  llvm::SmallVector<int64_t, num_elements> data(num_elements);
  for (auto i = 0; i < num_elements; i++) {
    data[i] = i;
  }
  auto rtt =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 32));
  std::string buffer;

  for (auto s : state) {
    llvm::raw_string_ostream ostream(buffer);
    DummyPrinter printer(ostream);
    TensorPrinter<int64_t> tensor_printer(data, rtt, printer);
    tensor_printer.Print();
  }
}

BENCHMARK(BM_OldPrint);
BENCHMARK(BM_Print);

}  // namespace
}  // namespace litert
