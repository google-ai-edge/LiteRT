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
#include "litert/compiler/mlir/dialects/litert/tensor_printer.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "litert/compiler/mlir/dialects/litert/attributes.h"
#include "litert/compiler/mlir/dialects/litert/dialect.h"

namespace litert {
namespace {

class TestPrinter : public mlir::AsmPrinter {
 public:
  explicit TestPrinter(llvm::raw_ostream& ostream) : ostream_(ostream) {}
  llvm::raw_ostream& getStream() const override { return ostream_; }

 private:
  llvm::raw_ostream& ostream_;
};

TEST(TensorPrinter, Print2x2x2x2x2) {
  std::string expected_result =
      "[[[[[0,1],[2,3]],[[4,5],[6,7]]],[[[8,9],[10,11]],[[12,13],[14,15]]]],[[["
      "[16,17],[18,19]],[[20,21],[22,23]]],[[[24,25],[26,27]],[[28,29],[30,31]]"
      "]]]";
  mlir::MLIRContext ctx;
  ctx.loadDialect<LITERTDialect>();
  llvm::SmallVector<int64_t, 5> shape = {2, 2, 2, 2, 2};
  llvm::SmallVector<int64_t, 32> data(32);
  for (auto i = 0; i < 32; i++) {
    data[i] = i;
  }
  auto rtt =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 32));
  std::string printed_values;
  llvm::raw_string_ostream ostream(printed_values);
  TestPrinter printer(ostream);
  TensorPrinter<int64_t> tensor_printer(data, rtt, printer);

  tensor_printer.Print();

  EXPECT_THAT(printed_values, ::testing::Eq(expected_result));

  std::string type_str;
  llvm::raw_string_ostream type_ostream(type_str);
  rtt.print(type_ostream);

  std::string full_attribute_str =
      "#litert.lazy_dense<" + printed_values + "> : " + type_ostream.str();

  mlir::Attribute parsed_attr = mlir::parseAttribute(full_attribute_str, &ctx);
  ASSERT_TRUE(parsed_attr) << "Failed to parse attribute string: "
                           << full_attribute_str;

  auto lazy_dense_attr = mlir::dyn_cast<LazyDenseElementsAttr>(parsed_attr);
  ASSERT_TRUE(lazy_dense_attr);

  auto handle = lazy_dense_attr.GetDataHandle();
  auto parsed_values = handle.GetDataAs<int32_t>();
  EXPECT_TRUE(std::equal(data.begin(), data.end(), parsed_values.begin()));
}

TEST(TensorPrinter, Print2x4x8) {
  std::string expected_result =
      "[[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23],["
      "24,25,26,27,28,29,30,31]],[[32,33,34,35,36,37,38,39],[40,41,42,43,44,45,"
      "46,47],[48,49,50,51,52,53,54,55],[56,57,58,59,60,61,62,63]]]";
  mlir::MLIRContext ctx;
  ctx.loadDialect<LITERTDialect>();
  llvm::SmallVector<int64_t, 3> shape = {2, 4, 8};
  llvm::SmallVector<int64_t, 64> data(64);
  for (auto i = 0; i < 64; i++) {
    data[i] = i;
  }
  auto rtt =
      mlir::RankedTensorType::get(shape, mlir::IntegerType::get(&ctx, 32));
  std::string printed_values;
  llvm::raw_string_ostream ostream(printed_values);
  TestPrinter printer(ostream);
  TensorPrinter<int64_t> tensor_printer(data, rtt, printer);

  tensor_printer.Print();

  EXPECT_THAT(printed_values, ::testing::Eq(expected_result));

  std::string type_str;
  llvm::raw_string_ostream type_ostream(type_str);
  rtt.print(type_ostream);

  std::string full_attribute_str =
      "#litert.lazy_dense<" + printed_values + "> : " + type_ostream.str();

  mlir::Attribute parsed_attr = mlir::parseAttribute(full_attribute_str, &ctx);
  ASSERT_TRUE(parsed_attr) << "Failed to parse attribute string: "
                           << full_attribute_str;

  auto lazy_dense_attr = mlir::dyn_cast<LazyDenseElementsAttr>(parsed_attr);
  ASSERT_TRUE(lazy_dense_attr);

  auto handle = lazy_dense_attr.GetDataHandle();
  auto parsed_values = handle.GetDataAs<int32_t>();
  EXPECT_TRUE(std::equal(data.begin(), data.end(), parsed_values.begin()));
}

}  // namespace
}  // namespace litert
